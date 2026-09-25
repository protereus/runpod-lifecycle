from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import sys

# Resolve the probe submodule directly; ``runpod_lifecycle.__init__`` rebinds
# the name ``probe`` to the function, so ``from runpod_lifecycle import probe``
# does *not* give us the module.
probe_module = sys.modules.get("runpod_lifecycle.probe")
if probe_module is None:  # pragma: no cover - first-time import.
    import importlib

    probe_module = importlib.import_module("runpod_lifecycle.probe")
probe = probe_module.probe

from runpod_lifecycle import api as api_module  # noqa: E402


def _gpu(
    gpu_id: str,
    display: str,
    mem: int,
    price: float | None,
    *,
    availability: str | None = "HIGH",
    datacenters: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """A GPU type in the v2 ``GET /v2/catalog/gpus?include=AVAILABILITY`` shape."""
    if datacenters is None:
        datacenters = [("EU-RO-1", availability or "NONE")]
    entry: dict[str, Any] = {
        "id": gpu_id,
        "name": display,
        "memory": mem,
        "secure": True,
        "community": True,
        "price": {"secure": price, "community": (price or 0) * 0.8},
        "maxCount": {"secure": 8, "community": 8},
        "dataCenters": [{"id": dc, "name": dc, "availability": level} for dc, level in datacenters],
    }
    if availability is not None:
        entry["availability"] = availability
    return entry


def _patch_response(
    monkeypatch: pytest.MonkeyPatch, payload: list[dict[str, Any]]
) -> dict[str, Any]:
    """Stub httpx.request used by the API layer; record the sent params."""
    captured: dict[str, Any] = {}

    def fake_request(method: str, url: str, **kwargs: Any):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        captured["headers"] = kwargs.get("headers")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"gpus": payload}
        return response

    monkeypatch.setattr(api_module.httpx, "request", fake_request)
    return captured


@pytest.mark.asyncio
async def test_probe_filters_by_min_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA RTX A4000", "RTX A4000", 16, 0.30),
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.89),
        ],
    )

    results = await probe(api_key="k", min_memory_gb=24)
    gpu_names = [r["gpu_type"] for r in results]
    assert "RTX A4000" not in gpu_names
    assert "RTX 6000 Ada" in gpu_names
    assert "A100 80GB" in gpu_names


@pytest.mark.asyncio
async def test_probe_excludes_blackwell_when_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA B200 Blackwell", "B200 Blackwell", 180, 4.50),
            _gpu("NVIDIA GeForce RTX 5090", "RTX 5090 (Blackwell)", 32, 0.95),
        ],
    )

    with_blackwell = await probe(api_key="k", min_memory_gb=24, exclude_blackwell=False)
    assert len(with_blackwell) == 3
    assert any(r["is_blackwell"] for r in with_blackwell)

    without_blackwell = await probe(
        api_key="k", min_memory_gb=24, exclude_blackwell=True
    )
    assert len(without_blackwell) == 1
    assert without_blackwell[0]["gpu_type"] == "RTX 6000 Ada"
    assert without_blackwell[0]["is_blackwell"] is False


@pytest.mark.asyncio
async def test_probe_ranks_by_price_ascending(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.89),
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA H100 PCIe", "H100 PCIe", 80, 2.69),
        ],
    )

    results = await probe(api_key="k", min_memory_gb=24)
    prices = [r["price_per_hour"] for r in results]
    assert prices == sorted(prices)
    assert results[0]["gpu_type"] == "RTX 6000 Ada"


@pytest.mark.asyncio
async def test_probe_filters_by_max_price(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.89),
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA H100 PCIe", "H100 PCIe", 80, 2.69),
        ],
    )

    results = await probe(api_key="k", min_memory_gb=24, max_price_per_hour=1.00)
    assert len(results) == 1
    assert results[0]["gpu_type"] == "RTX 6000 Ada"


@pytest.mark.asyncio
async def test_probe_drops_entries_with_no_stock_or_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA H100 NVL", "H100 NVL", 94, 2.5, availability="NONE"),
            _gpu("NVIDIA H200", "H200", 141, 3.5, availability=None),
            _gpu("NVIDIA L40S", "L40S", 48, None),
        ],
    )
    results = await probe(api_key="k", min_memory_gb=24)
    assert [r["gpu_type"] for r in results] == ["RTX 6000 Ada"]


@pytest.mark.asyncio
async def test_probe_gpu_types_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.89),
        ],
    )
    results = await probe(
        api_key="k",
        min_memory_gb=24,
        gpu_types=["NVIDIA A100 80GB PCIe"],
    )
    assert [r["gpu_type"] for r in results] == ["A100 80GB"]


@pytest.mark.asyncio
async def test_probe_requests_availability_for_selected_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_response(monkeypatch, [])
    await probe(api_key="k", require_secure_cloud=True)
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.runpod.io/v2/catalog/gpus"
    assert captured["params"] == {
        "include": "AVAILABILITY",
        "product": "POD",
        "count": 1,
        "cloud": "SECURE",
    }
    assert captured["headers"] == {"Authorization": "Bearer k"}

    captured2 = _patch_response(monkeypatch, [])
    await probe(api_key="k", require_secure_cloud=False)
    assert captured2["params"]["cloud"] == "COMMUNITY"


@pytest.mark.asyncio
async def test_probe_uses_price_for_selected_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(monkeypatch, [_gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 2.0)])

    secure = await probe(api_key="k", require_secure_cloud=True)
    community = await probe(api_key="k", require_secure_cloud=False)

    assert secure[0]["price_per_hour"] == 2.0
    assert secure[0]["secure_cloud"] is True
    assert community[0]["price_per_hour"] == pytest.approx(1.6)
    assert community[0]["secure_cloud"] is False


@pytest.mark.asyncio
async def test_probe_reports_datacenters_with_stock(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu(
                "NVIDIA RTX 6000 Ada Generation",
                "RTX 6000 Ada",
                48,
                0.77,
                availability="MEDIUM",
                datacenters=[("EU-RO-1", "HIGH"), ("US-KS-2", "NONE"), ("US-TX-3", "LOW")],
            )
        ],
    )
    [result] = await probe(api_key="k", min_memory_gb=24)
    assert result["availability"] == "MEDIUM"
    assert result["datacenters_available"] == ["EU-RO-1", "US-TX-3"]
    assert result["memory_gb"] == 48


@pytest.mark.asyncio
async def test_probe_filters_by_datacenter_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("gpu-eu", "EU only", 48, 0.7, datacenters=[("EU-RO-1", "HIGH")]),
            _gpu("gpu-us", "US only", 48, 0.8, datacenters=[("US-TX-3", "HIGH")]),
            _gpu("gpu-both", "Both", 48, 0.9, datacenters=[("EU-RO-1", "LOW"), ("US-TX-3", "HIGH")]),
        ],
    )
    results = await probe(api_key="k", min_memory_gb=24, datacenter_ids=["US-TX-3"])
    assert [(r["gpu_type"], r["datacenters_available"]) for r in results] == [
        ("US only", ["US-TX-3"]),
        ("Both", ["US-TX-3"]),
    ]


@pytest.mark.asyncio
async def test_probe_raises_on_unexpected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(*args: Any, **kwargs: Any):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"unexpected": True}
        return response

    monkeypatch.setattr(api_module.httpx, "request", fake_request)

    with pytest.raises(RuntimeError, match="unexpected payload"):
        await probe(api_key="k")


@pytest.mark.asyncio
async def test_probe_raises_on_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(*args: Any, **kwargs: Any):
        response = MagicMock()
        response.status_code = 401
        response.json.return_value = {"title": "Unauthorized", "status": 401, "detail": "bad key"}
        return response

    monkeypatch.setattr(api_module.httpx, "request", fake_request)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        await probe(api_key="k")
