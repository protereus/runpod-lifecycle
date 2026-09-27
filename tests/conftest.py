from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from runpod_lifecycle import api
from runpod_lifecycle.config import RunPodConfig


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        *,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text or ("" if payload is None else str(payload))

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def problem(status: int, detail: str, title: str = "Error") -> FakeResponse:
    """An RFC 9457 error body, as RunPod v2 returns."""
    return FakeResponse(status, {"title": title, "status": status, "detail": detail})


@dataclass
class RecordedCall:
    method: str
    path: str
    json: Any
    params: Any
    headers: dict[str, str]


Handler = FakeResponse | BaseException | Callable[[RecordedCall], FakeResponse]


@dataclass
class FakeRunPodAPI:
    """Stands in for ``httpx.request`` against https://api.runpod.io/v2.

    ``add(method, path, *responses)`` queues responses for a route; the last
    one repeats. Unrouted requests get a 404 problem body.
    """

    routes: dict[tuple[str, str], list[Handler]] = field(default_factory=dict)
    calls: list[RecordedCall] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)

    def add(self, method: str, path: str, *responses: Handler) -> "FakeRunPodAPI":
        self.routes.setdefault((method, path), []).extend(responses)
        return self

    def calls_to(self, method: str, path: str) -> list[RecordedCall]:
        return [c for c in self.calls if c.method == method and c.path == path]

    def __call__(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        assert url.startswith(api.API_BASE_URL), url
        call = RecordedCall(
            method=method,
            path=url[len(api.API_BASE_URL):],
            json=kwargs.get("json"),
            params=kwargs.get("params"),
            headers=kwargs.get("headers") or {},
        )
        self.calls.append(call)
        queue = self.routes.get((method, call.path))
        if not queue:
            return problem(404, f"no fake route for {method} {call.path}", "Not Found")
        handler = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(handler, BaseException):
            raise handler
        if isinstance(handler, FakeResponse):
            return handler
        return handler(call)


@pytest.fixture
def runpod_api(monkeypatch: pytest.MonkeyPatch) -> FakeRunPodAPI:
    fake = FakeRunPodAPI()
    monkeypatch.setattr(api.httpx, "request", fake)
    monkeypatch.setattr(api, "_sleep", fake.sleeps.append)
    return fake


def v2_pod(
    pod_id: str = "p1",
    *,
    name: str | None = "worker",
    status: str = "RUNNING",
    cost: float = 0.5,
    uptime: int | None = 100,
    ports: list[dict[str, Any]] | None = None,
    ssh_direct: dict[str, Any] | None = None,
    gpu_id: str = "NVIDIA GeForce RTX 4090",
    network_volume_id: str | None = "vol-1",
) -> dict[str, Any]:
    """A pod object in the RunPod v2 response shape."""
    if ports is None:
        ports = [{"private": 22, "public": 2201, "type": "tcp", "ip": "1.2.3.4"}]
    runtime = {"uptime": uptime, "ports": ports} if uptime is not None else None
    mounts: dict[str, Any] = {}
    if network_volume_id:
        mounts["network"] = [{"volumeId": network_volume_id, "path": "/workspace"}]
    return {
        "id": pod_id,
        "name": name,
        "status": status,
        "image": "runpod/pytorch:latest",
        "gpu": {"id": gpu_id, "count": 1, "vcpuCount": 8, "memory": 32},
        "mounts": mounts,
        "cloud": "SECURE",
        "dataCenterId": "EU-RO-1",
        "ssh": {"proxy": None, "direct": ssh_direct},
        "cost": cost,
        "runtime": runtime,
        "createdAt": "2026-09-01T00:00:00Z",
        "startedAt": "2026-09-01T00:01:00Z",
    }


@pytest.fixture
def base_config() -> RunPodConfig:
    return RunPodConfig(
        api_key="test",
        storage_volumes=("vol-a", "vol-b"),
        ram_tiers=(64, 32, 16),
        ssh_public_key="ssh-ed25519 AAAA test",
    )


@pytest.fixture
def volumeless_config() -> RunPodConfig:
    return RunPodConfig(
        api_key="test",
        storage_name=None,
        storage_volumes=(),
    )
