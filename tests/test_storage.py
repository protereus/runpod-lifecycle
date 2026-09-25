from __future__ import annotations

import asyncio

import pytest

from runpod_lifecycle.pod import Pod
from runpod_lifecycle.storage import (
    _expand_network_volume,
    check_and_expand_storage,
    evaluate_storage_health,
    parse_df_output,
)
from tests.conftest import FakeResponse, problem


def test_parse_df_output_basic_workspace_line() -> None:
    raw_output = """
Filesystem      Size  Used Avail Use% Mounted on
/dev/root       100G   50G   50G  50% /workspace
"""

    assert parse_df_output(raw_output) == {
        "total_gb": 100,
        "used_gb": 50,
        "free_gb": 50,
        "percent_used": 50,
    }


def test_parse_df_output_handles_terabytes() -> None:
    raw_output = """
Filesystem      Size  Used Avail Use% Mounted on
/dev/root         1T  512G  512G  50% /workspace
"""

    assert parse_df_output(raw_output)["total_gb"] == 1024


def test_expand_network_volume_uses_v2_patch(runpod_api) -> None:
    runpod_api.add(
        "PATCH",
        "/network-volumes/volume-1",
        FakeResponse(200, {"id": "volume-1", "name": "v", "size": 150, "dataCenter": "EU-RO-1", "type": "STANDARD"}),
    )

    assert _expand_network_volume("api-key", "volume-1", 150) is True

    call = runpod_api.calls_to("PATCH", "/network-volumes/volume-1")[0]
    assert call.json == {"size": 150}
    assert call.headers["Authorization"] == "Bearer api-key"


def test_expand_network_volume_returns_false_on_error(runpod_api) -> None:
    runpod_api.add("PATCH", "/network-volumes/volume-1", problem(400, "size decrease attempted"))

    assert _expand_network_volume("api-key", "volume-1", 150) is False


def test_check_and_expand_storage_skips_when_already_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "runpod_lifecycle.storage.get_network_volumes",
        lambda api_key: [{"id": "volume-1", "name": "primary", "size": 120}],
    )

    def fail_expand(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Expansion should be skipped for >=100GB volumes")

    monkeypatch.setattr("runpod_lifecycle.storage._expand_network_volume", fail_expand)

    result = check_and_expand_storage("api-key", "volume-1", storage_name="primary")

    assert result["expanded"] is False
    assert result["current_size_gb"] == 120


def test_evaluate_storage_health_requests_expansion_at_threshold() -> None:
    result = evaluate_storage_health(
        {"total_gb": 100, "used_gb": 90, "free_gb": 10, "percent_used": 90},
        api_total_gb=100,
        min_free_gb=50,
        max_percent_used=85,
    )

    assert result == {
        "healthy": False,
        "needs_expansion": True,
        "message": "CRITICAL: 90% used, only 10GB free!",
    }


def test_pod_check_storage_health_integration(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    pod = Pod("pod-1", "worker", base_config, storage_volume="volume-1")

    async def fake_exec(cmd: str, timeout: int = 600) -> tuple[int, str, str]:
        assert "/workspace" in cmd
        return (
            0,
            "Filesystem      Size  Used Avail Use% Mounted on\n/dev/root       100G   40G   60G  40% /workspace\n",
            "",
        )

    monkeypatch.setattr(pod, "exec_ssh", fake_exec)
    monkeypatch.setattr(
        "runpod_lifecycle.pod.api.get_network_volumes",
        lambda api_key: [{"id": "volume-1", "size": 100}],
    )

    result = asyncio.run(pod.check_storage_health())

    assert result == {
        "healthy": True,
        "needs_expansion": False,
        "message": "OK: 60GB free (60% available)",
    }
