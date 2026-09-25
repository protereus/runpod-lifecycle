"""HTTP primitives for the RunPod REST API v2 (https://api.runpod.io/v2).

Every call goes through :func:`_request`, which retries 429 responses using
``Retry-After`` and retries idempotent methods on transport errors and 5xx.
POST is never retried on 5xx or transport errors so a create cannot be billed
twice.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

import httpx
from httpx import TransportError

logger = logging.getLogger("runpod_lifecycle.api")

API_BASE_URL = "https://api.runpod.io/v2"

DEFAULT_PORTS = "22/tcp,8888/http"
# The runpod SDK mounted a pod's own persistent volume here whenever no
# network volume was attached; kept for parity with pods launched pre-v2.
PERSISTENT_VOLUME_MOUNT_PATH = "/runpod-volume"

MAX_RETRIES = 4
MAX_RETRY_DELAY_SECONDS = 60.0
_IDEMPOTENT_METHODS = frozenset({"GET", "PUT", "PATCH", "DELETE"})

_sleep = time.sleep


class RunPodAPIError(RuntimeError):
    """A non-2xx response from the RunPod API, parsed from its RFC 9457 body."""

    def __init__(
        self,
        method: str,
        path: str,
        status_code: int,
        title: str | None,
        detail: str | None,
    ) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.title = title
        self.detail = detail
        summary = ": ".join(part for part in (title, detail) if part) or "no error detail"
        super().__init__(f"RunPod API {method} {path} failed: HTTP {status_code}: {summary}")


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _error_from_response(method: str, path: str, response: Any) -> RunPodAPIError:
    title = detail = None
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        title = body.get("title")
        detail = body.get("detail") or body.get("message") or body.get("error")
    if detail is None:
        text = getattr(response, "text", "") or ""
        detail = text[:200] or None
    return RunPodAPIError(method, path, response.status_code, title, detail)


def _retry_delay(response: Any, attempt: int) -> float:
    headers = getattr(response, "headers", None) or {}
    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    try:
        delay = float(retry_after) if retry_after is not None else 2.0**attempt
    except (TypeError, ValueError):
        delay = 2.0**attempt
    return max(0.0, min(delay, MAX_RETRY_DELAY_SECONDS))


def _request(
    method: str,
    path: str,
    api_key: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30,
) -> Any:
    """Send one API request, retrying where safe, and return the 2xx response."""
    url = f"{API_BASE_URL}{path}"
    idempotent = method in _IDEMPOTENT_METHODS
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.request(
                method,
                url,
                json=json,
                params=params,
                headers=_auth_headers(api_key),
                timeout=timeout,
            )
        except TransportError as exc:
            if idempotent and attempt < MAX_RETRIES:
                logger.warning("RunPod API %s %s transport error (%s); retrying", method, path, exc)
                _sleep(2.0**attempt)
                continue
            raise

        status = response.status_code
        retryable = status == 429 or (idempotent and status >= 500)
        if retryable and attempt < MAX_RETRIES:
            delay = _retry_delay(response, attempt)
            logger.warning(
                "RunPod API %s %s returned HTTP %d; retrying in %.1fs", method, path, status, delay
            )
            _sleep(delay)
            continue
        if 200 <= status < 300:
            return response
        raise _error_from_response(method, path, response)

    raise AssertionError("unreachable")  # pragma: no cover


def _paginate(path: str, api_key: str, key: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Follow ``pagination.nextCursor`` and return every item under ``key``."""
    items: list[dict[str, Any]] = []
    query = dict(params or {})
    while True:
        body = _request("GET", path, api_key, params=query).json()
        page = body.get(key) if isinstance(body, dict) else None
        items.extend(item for item in page or [] if isinstance(item, dict))
        pagination = body.get("pagination") if isinstance(body, dict) else None
        cursor = pagination.get("nextCursor") if isinstance(pagination, dict) else None
        if not cursor or not pagination.get("hasNextPage", True):
            return items
        query = {**query, "cursor": cursor}


# ---------------------------------------------------------------------------
# Network volumes
# ---------------------------------------------------------------------------


def _normalize_volume(volume: dict[str, Any]) -> dict[str, Any]:
    # v2 renamed dataCenterId -> dataCenter; keep the old key for callers.
    normalized = dict(volume)
    normalized.setdefault("dataCenterId", volume.get("dataCenter"))
    return normalized


def get_network_volumes(api_key: str) -> list[dict[str, Any]]:
    """Return the account's RunPod network volumes, or ``[]`` if the lookup fails."""
    try:
        body = _request("GET", "/network-volumes", api_key).json()
    except Exception as exc:
        logger.warning("RunPod network volume lookup failed: %s", exc)
        return []
    volumes = body.get("networkVolumes") if isinstance(body, dict) else None
    return [_normalize_volume(v) for v in volumes or [] if isinstance(v, dict)]


def create_network_volume(
    api_key: str,
    name: str,
    size_gb: int,
    datacenter_id: str,
) -> dict[str, Any]:
    """Create a RunPod network volume and return it.

    POSTs ``{name, size, dataCenter}`` to ``/v2/network-volumes``.
    """
    payload: dict[str, Any] = {
        "name": name,
        "size": size_gb,
        "dataCenter": datacenter_id,
    }
    try:
        response = _request("POST", "/network-volumes", api_key, json=payload)
    except RunPodAPIError as exc:
        logger.error("create_network_volume failed: %s", exc)
        raise RuntimeError(f"Failed to create network volume '{name}': {exc}") from exc
    return _normalize_volume(response.json())


def update_network_volume_size(api_key: str, volume_id: str, size_gb: int) -> dict[str, Any]:
    """Grow a network volume to ``size_gb`` (RunPod cannot shrink volumes)."""
    response = _request("PATCH", f"/network-volumes/{volume_id}", api_key, json={"size": size_gb})
    return _normalize_volume(response.json())


# ---------------------------------------------------------------------------
# GPU catalogue
# ---------------------------------------------------------------------------


def _normalize_gpu_type(gpu: dict[str, Any]) -> dict[str, Any]:
    # Pre-v2 callers read displayName / memoryInGb; v2 calls them name / memory.
    normalized = dict(gpu)
    normalized.setdefault("displayName", gpu.get("name"))
    normalized.setdefault("memoryInGb", gpu.get("memory"))
    return normalized


def list_gpu_types(api_key: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the GPU catalogue from ``GET /v2/catalog/gpus``."""
    body = _request("GET", "/catalog/gpus", api_key, params=params).json()
    gpus = body.get("gpus") if isinstance(body, dict) else None
    if not isinstance(gpus, list):
        raise RuntimeError(f"RunPod GPU catalogue returned unexpected payload: {body!r}")
    return [_normalize_gpu_type(g) for g in gpus if isinstance(g, dict)]


def find_gpu_type(gpu_display_name: str, api_key: str) -> dict[str, Any] | None:
    """Find a GPU type by display name or ID."""
    try:
        gpus = list_gpu_types(api_key)
    except Exception as exc:
        logger.error("Error retrieving GPU list from RunPod: %s", exc)
        return None

    for gpu in gpus:
        if gpu_display_name in (gpu.get("displayName"), gpu.get("id")):
            return gpu
    return None


# ---------------------------------------------------------------------------
# Pods
# ---------------------------------------------------------------------------


def _parse_ports(ports: str) -> list[str]:
    return [port.strip() for port in ports.split(",") if port.strip()]


def _build_create_pod_body(
    *,
    gpu_type_id: str,
    image_name: str,
    name: str,
    network_volume_id: str | None,
    volume_mount_path: str,
    disk_in_gb: int,
    container_disk_in_gb: int,
    public_key_string: str | None,
    env_vars: dict[str, str] | None,
    min_vcpu_count: int,
    min_memory_in_gb: int,
    template_id: str | None,
    ports: str | None,
) -> dict[str, Any]:
    gpu: dict[str, Any] = {"id": gpu_type_id, "count": 1}
    # One GPU per pod, so the per-GPU floors equal the old per-pod minimums.
    if min_vcpu_count and min_vcpu_count > 0:
        gpu["minVcpuCountPerGpu"] = min_vcpu_count
    if min_memory_in_gb and min_memory_in_gb > 0:
        gpu["minRamPerGpu"] = min_memory_in_gb

    body: dict[str, Any] = {
        "name": name,
        "image": image_name,
        "cloud": "SECURE",
        "gpu": gpu,
        "disk": container_disk_in_gb,
        "ports": _parse_ports(ports or DEFAULT_PORTS),
        "startSsh": True,
    }

    if template_id:
        body["templateId"] = template_id

    # v2 allows either a network mount or a persistent pod volume, not both.
    if network_volume_id:
        body["mounts"] = {"network": [{"volumeId": network_volume_id, "path": volume_mount_path}]}
    elif disk_in_gb and disk_in_gb > 0:
        body["mounts"] = {
            "persistent": {"size": max(disk_in_gb, 10), "path": PERSISTENT_VOLUME_MOUNT_PATH}
        }

    pod_env: dict[str, str] = {}
    if env_vars:
        pod_env.update(env_vars)
    if public_key_string:
        pod_env["PUBLIC_KEY"] = public_key_string
    if pod_env:
        body["env"] = pod_env

    return body


def create_pod(
    api_key: str,
    gpu_type_id: str,
    image_name: str,
    name: str = "worker-pod",
    network_volume_id: str | None = None,
    volume_mount_path: str = "/workspace",
    disk_in_gb: int = 20,
    container_disk_in_gb: int = 10,
    public_key_string: str | None = None,
    env_vars: dict[str, str] | None = None,
    min_vcpu_count: int = 8,
    min_memory_in_gb: int = 32,
    template_id: str | None = None,
    ports: str | None = None,
) -> dict[str, Any]:
    """Create a RunPod pod and return provision metadata immediately."""
    body = _build_create_pod_body(
        gpu_type_id=gpu_type_id,
        image_name=image_name,
        name=name,
        network_volume_id=network_volume_id,
        volume_mount_path=volume_mount_path,
        disk_in_gb=disk_in_gb,
        container_disk_in_gb=container_disk_in_gb,
        public_key_string=public_key_string,
        env_vars=env_vars,
        min_vcpu_count=min_vcpu_count,
        min_memory_in_gb=min_memory_in_gb,
        template_id=template_id,
        ports=ports,
    )
    pod = _request("POST", "/pods", api_key, json=body).json()

    pod_id = pod.get("id") if isinstance(pod, dict) else None
    if not pod_id:
        raise RuntimeError("Pod creation failed (no pod ID returned)")

    status = pod.get("status") or "PROVISIONING"
    return {
        "id": pod_id,
        "status": status,
        "desiredStatus": status,
        "name": name,
        "gpu_type_id": gpu_type_id,
        "created": True,
    }


_CAPACITY_ERROR_MARKERS = (
    "no longer any instances available",
    "there are no machines available",
    "no longer have any instances",
    "not enough capacity",
    "out of stock",
)
# v2 create-pod contract: 400 covers "this GPU/data centre could not be
# placed" (no machine-readable capacity code yet) and 403 means the account
# cannot use that pool. Both mean "try the next candidate".
_CANDIDATE_MISS_STATUSES = frozenset({400, 403})


def _is_capacity_error(exc: BaseException) -> bool:
    if isinstance(exc, RunPodAPIError) and exc.status_code in _CANDIDATE_MISS_STATUSES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _CAPACITY_ERROR_MARKERS)


def create_pod_with_fallbacks(
    api_key: str,
    gpu_type_candidates: Sequence[str],
    image_name: str,
    *,
    name_prefix: str = "worker-pod",
    volume_candidates: Sequence[str] = (),
    volume_mount_path: str = "/workspace",
    disk_in_gb: int = 20,
    container_disk_in_gb: int = 10,
    public_key_string: str | None = None,
    env_vars: dict[str, str] | None = None,
    min_vcpu_count: int = 8,
    min_memory_in_gb: int = 32,
    template_id: str | None = None,
    ports: str | None = None,
    max_full_passes: int = 5,
    retry_sleep_seconds: float = 60.0,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Create a pod, iterating (gpu_type, volume) combinations on capacity errors.

    Resolves each candidate gpu_type display name once. Pre-loads the account's
    network volumes once and matches them by name to volume_candidates. Then
    iterates the cartesian product (gpu × volume, plus one no-volume slot if
    nothing matched) until ``create_pod`` succeeds. RunPod capacity errors fall
    back to the next combo. After a full pass exhausts every combo, sleeps
    ``retry_sleep_seconds`` and retries the pass — up to ``max_full_passes``
    times. Non-capacity errors abort immediately.

    Returns the same dict shape as ``create_pod``, enriched with
    ``selected_gpu_display_name`` and ``selected_volume_name`` so callers can
    log which combo actually succeeded.
    """
    if not gpu_type_candidates:
        raise ValueError("gpu_type_candidates must contain at least one GPU type")

    resolved_gpus: list[tuple[str, str]] = []  # (display_name, gpu_type_id)
    for candidate in gpu_type_candidates:
        candidate_str = str(candidate).strip()
        if not candidate_str:
            continue
        info = find_gpu_type(candidate_str, api_key)
        if not info or not info.get("id"):
            logger.warning("create_pod_with_fallbacks: GPU type not found in catalog: %r", candidate_str)
            continue
        resolved_gpus.append((str(info.get("displayName") or candidate_str), str(info["id"])))
    if not resolved_gpus:
        raise RuntimeError(f"None of the candidate GPU types resolved: {list(gpu_type_candidates)}")

    volume_lookup: list[tuple[str | None, str | None]] = []  # (volume_name, volume_id)
    if volume_candidates:
        try:
            available = get_network_volumes(api_key)
            by_name = {v.get("name"): v.get("id") for v in available if v.get("name")}
        except Exception as exc:
            logger.warning("create_pod_with_fallbacks: could not list network volumes (%s)", exc)
            by_name = {}
        for name in volume_candidates:
            vid = by_name.get(name) if name else None
            if vid:
                volume_lookup.append((name, vid))
    if not volume_lookup:
        volume_lookup.append((None, None))

    last_capacity_error: BaseException | None = None
    total_attempts = 0
    for full_pass in range(1, max_full_passes + 1):
        for gpu_display, gpu_id in resolved_gpus:
            for vol_name, vol_id in volume_lookup:
                total_attempts += 1
                if on_attempt is not None:
                    try:
                        on_attempt({
                            "attempt": total_attempts,
                            "pass": full_pass,
                            "gpu_display_name": gpu_display,
                            "gpu_type_id": gpu_id,
                            "volume_name": vol_name,
                            "volume_id": vol_id,
                        })
                    except Exception as exc:
                        logger.debug("create_pod_with_fallbacks on_attempt callback failed: %s", exc)
                pod_name = f"{name_prefix}-{int(time.time() * 1000)}"
                try:
                    pod = create_pod(
                        api_key=api_key,
                        gpu_type_id=gpu_id,
                        image_name=image_name,
                        name=pod_name,
                        network_volume_id=vol_id,
                        volume_mount_path=volume_mount_path,
                        disk_in_gb=disk_in_gb,
                        container_disk_in_gb=container_disk_in_gb,
                        public_key_string=public_key_string,
                        env_vars=env_vars,
                        min_vcpu_count=min_vcpu_count,
                        min_memory_in_gb=min_memory_in_gb,
                        template_id=template_id,
                        ports=ports,
                    )
                except Exception as exc:
                    if _is_capacity_error(exc):
                        last_capacity_error = exc
                        logger.info(
                            "create_pod_with_fallbacks: capacity miss attempt %d (gpu=%s vol=%s); trying next combo",
                            total_attempts,
                            gpu_display,
                            vol_name,
                        )
                        continue
                    raise
                pod["selected_gpu_display_name"] = gpu_display
                pod["selected_gpu_type_id"] = gpu_id
                pod["selected_volume_name"] = vol_name
                pod["selected_volume_id"] = vol_id
                pod["attempt"] = total_attempts
                pod["pass"] = full_pass
                return pod
        if full_pass < max_full_passes and retry_sleep_seconds > 0:
            logger.info(
                "create_pod_with_fallbacks: pass %d exhausted all %d combos; sleeping %.1fs before retry",
                full_pass,
                len(resolved_gpus) * len(volume_lookup),
                retry_sleep_seconds,
            )
            time.sleep(retry_sleep_seconds)

    if last_capacity_error is not None:
        raise RuntimeError(
            f"create_pod_with_fallbacks: all {total_attempts} attempts hit capacity errors. "
            f"Last error: {last_capacity_error}"
        )
    raise RuntimeError(
        f"create_pod_with_fallbacks: exhausted {total_attempts} attempts without success"
    )


def _normalize_ports(raw_ports: Any) -> list[dict[str, Any]]:
    """Map v2 ``runtime.ports`` entries onto the pre-v2 camelCase keys."""
    ports: list[dict[str, Any]] = []
    for port in raw_ports if isinstance(raw_ports, list) else []:
        if not isinstance(port, dict):
            continue
        ports.append(
            {
                "ip": port.get("ip"),
                "publicPort": port.get("public"),
                "privatePort": port.get("private"),
                "type": port.get("type"),
            }
        )
    return ports


def _direct_ssh(pod: dict[str, Any]) -> dict[str, Any] | None:
    ssh = pod.get("ssh")
    direct = ssh.get("direct") if isinstance(ssh, dict) else None
    if isinstance(direct, dict) and direct.get("host") and direct.get("port"):
        return direct
    return None


def _normalize_pod_status(runpod_id: str, pod: dict[str, Any]) -> dict[str, Any]:
    runtime = pod.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    ports = _normalize_ports(runtime.get("ports"))
    direct = _direct_ssh(pod)
    ip = (direct or {}).get("host") or next((p["ip"] for p in ports if p.get("ip")), None)
    status = pod.get("status")
    return {
        "runpod_id": runpod_id,
        "status": status,
        # v2 reports one lifecycle status; both legacy keys carry it.
        "desired_status": status,
        "actual_status": status,
        "ip": ip,
        "ports": ports,
        "ssh_password": None,
        "created_at": pod.get("createdAt"),
        "started_at": pod.get("startedAt"),
        "last_status_change": None,
        "uptime_seconds": runtime.get("uptime") or 0,
        "cost_per_hr": pod.get("cost"),
    }


def get_pod(runpod_id: str, api_key: str) -> dict[str, Any] | None:
    """Return the raw v2 pod object, or ``None`` if the pod does not exist."""
    try:
        pod = _request("GET", f"/pods/{runpod_id}", api_key).json()
    except RunPodAPIError as exc:
        if exc.status_code == 404:
            return None
        raise
    return pod if isinstance(pod, dict) else None


def list_pods(api_key: str) -> list[dict[str, Any]]:
    """Return every standalone pod on the account as raw v2 pod objects."""
    return _paginate("/pods", api_key, "pods")


def get_pod_status(runpod_id: str, api_key: str) -> dict[str, Any] | None:
    """Return normalized pod status details using snake_case keys."""
    try:
        pod = get_pod(runpod_id, api_key)
    except Exception as exc:
        logger.warning("RunPod pod status lookup failed for %s: %s", runpod_id, exc)
        return None
    return _normalize_pod_status(runpod_id, pod) if pod else None


def get_pod_ssh_details(pod_id: str, api_key: str) -> dict[str, Any] | None:
    """Return SSH details (ip, port, password) for a running pod."""
    try:
        pod = get_pod(pod_id, api_key)
    except Exception as exc:
        logger.warning("RunPod get pod failed for %s: %s", pod_id, exc)
        pod = None

    if pod:
        direct = _direct_ssh(pod)
        if direct:
            return {"ip": direct["host"], "port": direct["port"], "password": "runpod"}
        runtime = pod.get("runtime")
        ports = _normalize_ports(runtime.get("ports") if isinstance(runtime, dict) else None)
        for port_map in ports:
            if port_map.get("privatePort") == 22 and port_map.get("ip") and port_map.get("publicPort"):
                return {"ip": port_map["ip"], "port": port_map["publicPort"], "password": "runpod"}

    logger.warning("Could not get SSH details for pod %s via the RunPod API", pod_id)
    return None


def terminate_pod(pod_id: str, api_key: str) -> None:
    """Terminate a RunPod pod to stop billing."""
    _request("DELETE", f"/pods/{pod_id}", api_key)


__all__ = [
    "API_BASE_URL",
    "RunPodAPIError",
    "create_pod",
    "create_pod_with_fallbacks",
    "create_network_volume",
    "find_gpu_type",
    "get_network_volumes",
    "get_pod",
    "get_pod_ssh_details",
    "get_pod_status",
    "list_gpu_types",
    "list_pods",
    "terminate_pod",
    "update_network_volume_size",
]
