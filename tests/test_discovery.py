from __future__ import annotations

import asyncio

import pytest

from runpod_lifecycle import discovery
from runpod_lifecycle.errors import LaunchFailure, TerminateError
from runpod_lifecycle.events import EventHooks, PodState
from runpod_lifecycle.pod import Pod
from tests.conftest import FakeResponse, FakeRunPodAPI, problem, v2_pod


def _set_pods(runpod_api: FakeRunPodAPI, pods: list[dict]) -> None:
    runpod_api.add(
        "GET",
        "/pods",
        FakeResponse(200, {"pods": pods, "pagination": {"nextCursor": None, "hasNextPage": False}}),
    )


def test_list_pods_returns_summaries(runpod_api) -> None:
    _set_pods(runpod_api, [v2_pod("a", name="gpu_1"), v2_pod("b", name="other")])
    summaries = asyncio.run(discovery.list_pods("test"))
    assert [s.id for s in summaries] == ["a", "b"]
    first = summaries[0]
    assert first.cost_per_hr == 0.5
    assert first.gpu_type == "NVIDIA GeForce RTX 4090"
    assert first.desired_status == "RUNNING"
    assert first.image == "runpod/pytorch:latest"
    assert first.uptime_seconds == 100
    assert first.network_volume_id == "vol-1"
    assert first.ports == [{"ip": "1.2.3.4", "publicPort": 2201, "privatePort": 22, "type": "tcp"}]


def test_list_pods_handles_pod_without_runtime_or_volume(runpod_api) -> None:
    _set_pods(runpod_api, [v2_pod("a", status="PROVISIONING", uptime=None, network_volume_id=None)])
    [summary] = asyncio.run(discovery.list_pods("test"))
    assert summary.uptime_seconds is None
    assert summary.ports == []
    assert summary.network_volume_id is None


def test_list_pods_name_prefix_filter(runpod_api) -> None:
    _set_pods(runpod_api, [
        v2_pod("a", name="gpu_1"),
        v2_pod("b", name="other"),
        v2_pod("c", name="gpu_2"),
    ])
    summaries = asyncio.run(discovery.list_pods("test", name_prefix="gpu_"))
    assert [s.id for s in summaries] == ["a", "c"]


def test_find_pods_filters_by_predicate(runpod_api) -> None:
    _set_pods(runpod_api, [
        v2_pod("a", cost=0.5),
        v2_pod("b", cost=2.0),
        v2_pod("c", cost=1.5),
    ])
    summaries = asyncio.run(discovery.find_pods("test", lambda p: p.cost_per_hr > 1.0))
    assert [s.id for s in summaries] == ["b", "c"]


def test_find_orphans_excludes_known(runpod_api) -> None:
    _set_pods(runpod_api, [v2_pod(x) for x in ["a", "b", "c", "d"]])
    orphans = asyncio.run(discovery.find_orphans("test", known_pod_ids={"a", "b"}))
    assert [s.id for s in orphans] == ["c", "d"]


def test_find_orphans_filters_by_age(runpod_api) -> None:
    _set_pods(runpod_api, [
        v2_pod("young", uptime=60),
        v2_pod("old", uptime=7200),
    ])
    orphans = asyncio.run(discovery.find_orphans("test", known_pod_ids=[], older_than_seconds=3600))
    assert [s.id for s in orphans] == ["old"]


def test_find_orphans_skips_inactive_status(runpod_api) -> None:
    _set_pods(runpod_api, [
        v2_pod("running"),
        v2_pod("exited", status="EXITED"),
        v2_pod("error", status="ERROR"),
        v2_pod("terminated", status="TERMINATED"),
        v2_pod("provisioning", status="PROVISIONING"),
        v2_pod("starting", status="STARTING"),
    ])
    orphans = asyncio.run(discovery.find_orphans("test", known_pod_ids=[]))
    assert sorted(s.id for s in orphans) == ["provisioning", "running", "starting"]


def test_get_pod_attaches_to_existing(runpod_api, base_config) -> None:
    runpod_api.add("GET", "/pods/abc", FakeResponse(200, v2_pod("abc")))
    pod = asyncio.run(discovery.get_pod("abc", base_config))
    assert isinstance(pod, Pod)
    assert pod.id == "abc"


def test_get_pod_missing_raises(runpod_api, base_config) -> None:
    runpod_api.add("GET", "/pods/missing", problem(404, "pod not found", "Not Found"))
    with pytest.raises(LaunchFailure):
        asyncio.run(discovery.get_pod("missing", base_config))


def test_module_terminate_happy_path(runpod_api) -> None:
    runpod_api.add("DELETE", "/pods/p1", FakeResponse(204))
    events: list[tuple[str, str | None]] = []

    async def on_state(event) -> None:
        events.append((event.state.value, event.pod_id))

    asyncio.run(discovery.terminate("p1", "test", hooks=EventHooks(on_state_change=on_state)))
    assert len(runpod_api.calls_to("DELETE", "/pods/p1")) == 1
    assert events == [(PodState.TERMINATED.value, "p1")]


def test_module_terminate_failure_emits_error_and_raises(runpod_api) -> None:
    runpod_api.add("DELETE", "/pods/p1", problem(409, "boom", "Conflict"))
    errors: list[str] = []

    def on_error(err: Exception, detail: dict) -> None:
        errors.append(str(err))

    with pytest.raises(TerminateError):
        asyncio.run(discovery.terminate("p1", "test", hooks=EventHooks(on_error=on_error)))
    assert len(errors) == 1
    assert "HTTP 409" in errors[0] and "boom" in errors[0]


def test_cost_summary_math() -> None:
    pods = [discovery.PodSummary(
        id=str(i), name=None, desired_status="RUNNING", actual_status="RUNNING",
        gpu_type=None, image=None, created_at=None, cost_per_hr=0.5,
        uptime_seconds=0, ports=[], network_volume_id=None,
    ) for i in range(3)]
    cost = discovery.cost_summary(pods)
    assert cost == {"total_per_hr": 1.5, "daily": 36.0, "monthly": 1080.0}
