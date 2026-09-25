"""Probe RunPod GPU availability without provisioning a pod.

The :func:`probe` function answers the question "what configuration could
actually launch right now, given my constraints?" — by querying RunPod's
v2 GPU catalogue with live availability and returning a price-ranked list of
viable candidates. This avoids the trial-and-error provisioning loop where
every failed attempt costs 20s+ of RAM-tier iteration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import api

logger = logging.getLogger("runpod_lifecycle.probe")


def _is_blackwell(gpu_id: str | None, display_name: str | None) -> bool:
    haystack = f"{gpu_id or ''} {display_name or ''}".lower()
    return "blackwell" in haystack


def _cloud(require_secure_cloud: bool) -> str:
    return "SECURE" if require_secure_cloud else "COMMUNITY"


async def _fetch_gpu_types(
    api_key: str, require_secure_cloud: bool
) -> list[dict[str, Any]]:
    params = {
        "include": "AVAILABILITY",
        "product": "POD",
        "count": 1,
        "cloud": _cloud(require_secure_cloud),
    }
    return await asyncio.to_thread(api.list_gpu_types, api_key, params)


async def probe(
    *,
    api_key: str,
    gpu_types: list[str] | None = None,
    min_memory_gb: int = 24,
    max_price_per_hour: float | None = None,
    require_secure_cloud: bool = True,
    exclude_blackwell: bool = False,
    container_disk_gb: int = 100,
    datacenter_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a price-ranked list of viable pod configurations.

    Each returned entry has the shape::

        {
          "gpu_type": "NVIDIA RTX 6000 Ada Generation",
          "memory_gb": 48,
          "price_per_hour": 0.77,
          "secure_cloud": True,
          "is_blackwell": False,
          "availability": "HIGH",
          "datacenters_available": ["EU-RO-1", "US-KS-2"],
        }

    Parameters
    ----------
    api_key:
        RunPod API key used for the catalogue request.
    gpu_types:
        Optional allow-list of GPU type ``id`` values (case-sensitive).
        ``None`` means "consider every type RunPod returns".
    min_memory_gb:
        Minimum VRAM (``memory`` from RunPod) the GPU must report.
    max_price_per_hour:
        Optional cap on the hourly on-demand price.
    require_secure_cloud:
        When ``True`` availability and price are for Secure Cloud; otherwise
        Community Cloud. The returned ``secure_cloud`` flag mirrors it.
    exclude_blackwell:
        Filter out GPU types whose id/display name contains ``"Blackwell"``
        (case-insensitive). Banodoco hivemind reports a training-quality
        regression on Blackwell variants.
    container_disk_gb:
        Accepted for backwards compatibility; RunPod's catalogue does not
        scope availability by disk size, so it is unused.
    datacenter_ids:
        Optional restriction list. When set, only GPU types with stock in at
        least one of these data centres are returned, and
        ``datacenters_available`` is narrowed to them.

    Returns
    -------
    list[dict]
        Configurations sorted by ``price_per_hour`` ascending. GPU types with
        no current stock (availability ``NONE`` or missing) in the requested
        cloud are filtered out. Stock can change between probe and launch, so
        treat the result as an ordering, not a reservation.
    """
    del container_disk_gb

    raw = await _fetch_gpu_types(api_key, require_secure_cloud)
    price_key = "secure" if require_secure_cloud else "community"
    wanted_dcs: set[str] | None = set(datacenter_ids) if datacenter_ids else None

    gpu_type_allowlist: set[str] | None = (
        set(gpu_types) if gpu_types is not None else None
    )

    results: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue

        gpu_id = entry.get("id")
        display_name = entry.get("name")
        memory_gb_raw = entry.get("memory")
        try:
            memory_gb = int(memory_gb_raw) if memory_gb_raw is not None else 0
        except (TypeError, ValueError):
            memory_gb = 0

        if gpu_type_allowlist is not None and gpu_id not in gpu_type_allowlist:
            continue

        if memory_gb < min_memory_gb:
            continue

        blackwell = _is_blackwell(gpu_id, display_name)
        if exclude_blackwell and blackwell:
            continue

        availability = entry.get("availability")
        if availability in (None, "NONE"):
            continue

        prices = entry.get("price") if isinstance(entry.get("price"), dict) else {}
        try:
            price = float(prices.get(price_key))
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue

        if max_price_per_hour is not None and price > max_price_per_hour:
            continue

        datacenters = [
            dc.get("id")
            for dc in entry.get("dataCenters") or []
            if isinstance(dc, dict) and dc.get("id") and dc.get("availability") not in (None, "NONE")
        ]
        if wanted_dcs is not None:
            datacenters = [dc for dc in datacenters if dc in wanted_dcs]
            if not datacenters:
                continue

        results.append(
            {
                "gpu_type": display_name or gpu_id or "",
                "memory_gb": memory_gb,
                "price_per_hour": price,
                "secure_cloud": bool(require_secure_cloud),
                "is_blackwell": blackwell,
                "availability": availability,
                "datacenters_available": datacenters,
            }
        )

    results.sort(key=lambda r: r["price_per_hour"])
    return results


__all__ = ["probe"]
