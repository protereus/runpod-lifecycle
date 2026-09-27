from __future__ import annotations

import logging
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from runpod_lifecycle.api import get_pod_ssh_details
from runpod_lifecycle.config import RunPodConfig
from runpod_lifecycle.pod import Pod
from tests.conftest import FakeResponse, problem, v2_pod


def test_get_pod_ssh_details_prefers_ssh_direct(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/pods/pod-123",
        FakeResponse(200, v2_pod("pod-123", ssh_direct={"host": "1.2.3.4", "port": 2201, "username": "root", "command": "ssh"})),
    )

    details = get_pod_ssh_details("pod-123", "api-key")

    assert details == {"ip": "1.2.3.4", "port": 2201, "password": "runpod"}


def test_get_pod_ssh_details_falls_back_to_runtime_ports(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/pods/pod-123",
        FakeResponse(200, v2_pod("pod-123", ports=[{"private": 22, "public": 2202, "type": "tcp", "ip": "5.6.7.8"}])),
    )

    details = get_pod_ssh_details("pod-123", "api-key")

    assert details == {"ip": "5.6.7.8", "port": 2202, "password": "runpod"}


def test_get_pod_ssh_details_returns_none_and_logs_warning(
    runpod_api,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="runpod_lifecycle.api")
    runpod_api.add("GET", "/pods/pod-123", FakeResponse(200, v2_pod("pod-123", ports=[])))

    details = get_pod_ssh_details("pod-123", "api-key")

    assert details is None
    assert "Could not get SSH details for pod pod-123" in caplog.text


def test_get_pod_ssh_details_returns_none_when_api_fails(runpod_api) -> None:
    runpod_api.add("GET", "/pods/pod-123", problem(401, "bad key"))

    assert get_pod_ssh_details("pod-123", "api-key") is None


def test_open_ssh_client_returns_connected_raw_client(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RunPodConfig(api_key="api-key")
    pod = Pod("pod-123", "worker", config)
    raw_client = MagicMock()
    wrapper = SimpleNamespace(client=raw_client, connect=MagicMock())

    monkeypatch.setattr(
        "runpod_lifecycle.pod.api.get_pod_ssh_details",
        lambda pod_id, api_key: {"ip": "1.2.3.4", "port": 2201, "password": "secret"},
    )
    monkeypatch.setattr(pod, "_build_ssh_client", lambda ssh_details: wrapper)

    result = pod.open_ssh_client()

    assert result is raw_client
    wrapper.connect.assert_called_once_with()
