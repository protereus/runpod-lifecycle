from __future__ import annotations

import httpx
import pytest

from runpod_lifecycle import api
from tests.conftest import FakeResponse, problem, v2_pod


# ---------------------------------------------------------------------------
# _request: auth, retries, errors
# ---------------------------------------------------------------------------


def test_request_sends_bearer_auth_to_v2_base_url(runpod_api) -> None:
    runpod_api.add("GET", "/pods/p1", FakeResponse(200, v2_pod("p1")))

    api.get_pod("p1", "secret-key")

    call = runpod_api.calls[0]
    assert call.headers == {"Authorization": "Bearer secret-key"}
    assert api.API_BASE_URL == "https://api.runpod.io/v2"


def test_request_retries_429_using_retry_after(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/pods/p1",
        FakeResponse(429, {"title": "Too Many Requests"}, headers={"Retry-After": "7"}),
        FakeResponse(200, v2_pod("p1")),
    )

    assert api.get_pod("p1", "k")["id"] == "p1"
    assert runpod_api.sleeps == [7.0]


def test_request_caps_retry_after(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/pods/p1",
        FakeResponse(429, {}, headers={"Retry-After": "3600"}),
        FakeResponse(200, v2_pod("p1")),
    )

    api.get_pod("p1", "k")
    assert runpod_api.sleeps == [api.MAX_RETRY_DELAY_SECONDS]


def test_request_retries_post_on_429_only(runpod_api) -> None:
    runpod_api.add(
        "POST",
        "/pods",
        FakeResponse(429, {}, headers={"Retry-After": "1"}),
        FakeResponse(201, v2_pod("new")),
    )

    assert api.create_pod(api_key="k", gpu_type_id="g", image_name="i")["id"] == "new"
    assert len(runpod_api.calls_to("POST", "/pods")) == 2


def test_request_does_not_retry_post_on_5xx(runpod_api) -> None:
    runpod_api.add("POST", "/pods", problem(503, "upstream down"), FakeResponse(201, v2_pod("dup")))

    with pytest.raises(api.RunPodAPIError) as excinfo:
        api.create_pod(api_key="k", gpu_type_id="g", image_name="i")

    assert excinfo.value.status_code == 503
    assert len(runpod_api.calls_to("POST", "/pods")) == 1


def test_request_does_not_retry_post_on_transport_error(runpod_api) -> None:
    runpod_api.add("POST", "/pods", httpx.ConnectError("reset"), FakeResponse(201, v2_pod("dup")))

    with pytest.raises(httpx.ConnectError):
        api.create_pod(api_key="k", gpu_type_id="g", image_name="i")
    assert len(runpod_api.calls_to("POST", "/pods")) == 1


def test_request_retries_get_on_5xx_and_transport_error(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/pods/p1",
        problem(502, "bad gateway"),
        httpx.ReadTimeout("slow"),
        FakeResponse(200, v2_pod("p1")),
    )

    assert api.get_pod("p1", "k")["id"] == "p1"
    assert runpod_api.sleeps == [1.0, 2.0]


def test_request_gives_up_after_max_retries(runpod_api) -> None:
    runpod_api.add("GET", "/pods/p1", FakeResponse(429, {"title": "Too Many Requests", "detail": "slow down"}))

    with pytest.raises(api.RunPodAPIError, match="HTTP 429"):
        api.get_pod("p1", "k")
    assert len(runpod_api.calls) == api.MAX_RETRIES + 1


def test_api_error_parses_rfc9457_body(runpod_api) -> None:
    runpod_api.add("DELETE", "/pods/p1", problem(409, "Pod belongs to a cluster", "Conflict"))

    with pytest.raises(api.RunPodAPIError) as excinfo:
        api.terminate_pod("p1", "k")

    err = excinfo.value
    assert (err.status_code, err.title, err.detail) == (409, "Conflict", "Pod belongs to a cluster")
    assert "DELETE /pods/p1" in str(err)


# ---------------------------------------------------------------------------
# create_pod
# ---------------------------------------------------------------------------


def test_create_pod_builds_v2_body_with_network_volume(runpod_api) -> None:
    runpod_api.add("POST", "/pods", FakeResponse(201, v2_pod("pod-1", status="PROVISIONING")))

    pod = api.create_pod(
        api_key="k",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        image_name="runpod/pytorch:x",
        name="worker-1",
        network_volume_id="vol-9",
        volume_mount_path="/workspace",
        disk_in_gb=200,
        container_disk_in_gb=50,
        public_key_string="ssh-ed25519 AAAA",
        env_vars={"FOO": "bar"},
        min_vcpu_count=8,
        min_memory_in_gb=64,
        template_id="runpod-torch-v240",
    )

    body = runpod_api.calls_to("POST", "/pods")[0].json
    assert body == {
        "name": "worker-1",
        "image": "runpod/pytorch:x",
        "cloud": "SECURE",
        "gpu": {
            "id": "NVIDIA GeForce RTX 4090",
            "count": 1,
            "minVcpuCountPerGpu": 8,
            "minRamPerGpu": 64,
        },
        "disk": 50,
        "ports": ["22/tcp", "8888/http"],
        "startSsh": True,
        "templateId": "runpod-torch-v240",
        "mounts": {"network": [{"volumeId": "vol-9", "path": "/workspace"}]},
        "env": {"FOO": "bar", "PUBLIC_KEY": "ssh-ed25519 AAAA"},
    }
    assert pod == {
        "id": "pod-1",
        "status": "PROVISIONING",
        "desiredStatus": "PROVISIONING",
        "name": "worker-1",
        "gpu_type_id": "NVIDIA GeForce RTX 4090",
        "created": True,
    }


def test_create_pod_uses_persistent_mount_without_network_volume(runpod_api) -> None:
    runpod_api.add("POST", "/pods", FakeResponse(201, v2_pod("pod-2")))

    api.create_pod(api_key="k", gpu_type_id="g", image_name="i", disk_in_gb=20)

    body = runpod_api.calls_to("POST", "/pods")[0].json
    assert body["mounts"] == {"persistent": {"size": 20, "path": api.PERSISTENT_VOLUME_MOUNT_PATH}}
    assert "templateId" not in body
    assert "env" not in body


def test_create_pod_omits_mounts_when_disk_is_zero(runpod_api) -> None:
    runpod_api.add("POST", "/pods", FakeResponse(201, v2_pod("pod-3")))

    api.create_pod(api_key="k", gpu_type_id="g", image_name="i", disk_in_gb=0)

    assert "mounts" not in runpod_api.calls_to("POST", "/pods")[0].json


def test_create_pod_passes_custom_ports_as_list(runpod_api) -> None:
    runpod_api.add("POST", "/pods", FakeResponse(201, v2_pod("pod-4")))

    api.create_pod(api_key="k", gpu_type_id="g", image_name="i", ports="8675/http, 22/tcp")

    assert runpod_api.calls_to("POST", "/pods")[0].json["ports"] == ["8675/http", "22/tcp"]


def test_create_pod_raises_when_no_id_returned(runpod_api) -> None:
    runpod_api.add("POST", "/pods", FakeResponse(201, {}))

    with pytest.raises(RuntimeError, match="no pod ID"):
        api.create_pod(api_key="k", gpu_type_id="g", image_name="i")


@pytest.mark.parametrize("status", [400, 403])
def test_candidate_miss_statuses_count_as_capacity_errors(status: int) -> None:
    err = api.RunPodAPIError("POST", "/pods", status, "Bad Request", "could not be placed")
    assert api._is_capacity_error(err)


@pytest.mark.parametrize("status", [402, 422, 500])
def test_other_statuses_are_not_capacity_errors(status: int) -> None:
    err = api.RunPodAPIError("POST", "/pods", status, "Error", "nope")
    assert not api._is_capacity_error(err)


def test_capacity_text_markers_still_match() -> None:
    assert api._is_capacity_error(RuntimeError("There are no longer any instances available"))


def test_create_pod_with_fallbacks_moves_on_after_400(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/catalog/gpus",
        FakeResponse(200, {"gpus": [
            {"id": "gpu-a", "name": "A"},
            {"id": "gpu-b", "name": "B"},
        ]}),
    )
    runpod_api.add(
        "POST",
        "/pods",
        problem(400, "no instances available for this GPU", "Bad Request"),
        FakeResponse(201, v2_pod("won")),
    )

    pod = api.create_pod_with_fallbacks("k", ["A", "B"], "image", retry_sleep_seconds=0)

    assert pod["id"] == "won"
    assert pod["selected_gpu_type_id"] == "gpu-b"
    gpu_ids = [c.json["gpu"]["id"] for c in runpod_api.calls_to("POST", "/pods")]
    assert gpu_ids == ["gpu-a", "gpu-b"]


def test_create_pod_with_fallbacks_aborts_on_insufficient_balance(runpod_api) -> None:
    runpod_api.add("GET", "/catalog/gpus", FakeResponse(200, {"gpus": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]}))
    runpod_api.add("POST", "/pods", problem(402, "Insufficient balance", "Payment Required"))

    with pytest.raises(api.RunPodAPIError, match="HTTP 402"):
        api.create_pod_with_fallbacks("k", ["A", "B"], "image", retry_sleep_seconds=0)
    assert len(runpod_api.calls_to("POST", "/pods")) == 1


# ---------------------------------------------------------------------------
# Pod status and SSH
# ---------------------------------------------------------------------------


def test_get_pod_status_normalizes_v2_pod(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/pods/p1",
        FakeResponse(200, v2_pod(
            "p1",
            cost=0.69,
            uptime=60,
            ports=[
                {"private": 22, "public": 12345, "type": "tcp", "ip": "1.2.3.4"},
                {"private": 8888, "public": None, "type": "http", "ip": None},
            ],
        )),
    )

    assert api.get_pod_status("p1", "k") == {
        "runpod_id": "p1",
        "status": "RUNNING",
        "desired_status": "RUNNING",
        "actual_status": "RUNNING",
        "ip": "1.2.3.4",
        "ports": [
            {"ip": "1.2.3.4", "publicPort": 12345, "privatePort": 22, "type": "tcp"},
            {"ip": None, "publicPort": None, "privatePort": 8888, "type": "http"},
        ],
        "ssh_password": None,
        "created_at": "2026-09-01T00:00:00Z",
        "started_at": "2026-09-01T00:01:00Z",
        "last_status_change": None,
        "uptime_seconds": 60,
        "cost_per_hr": 0.69,
    }


def test_get_pod_status_prefers_ssh_direct_host_for_ip(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/pods/p1",
        FakeResponse(200, v2_pod("p1", ssh_direct={"host": "9.9.9.9", "port": 40022, "username": "root", "command": "ssh"})),
    )

    assert api.get_pod_status("p1", "k")["ip"] == "9.9.9.9"


def test_get_pod_status_handles_null_runtime(runpod_api) -> None:
    runpod_api.add("GET", "/pods/p1", FakeResponse(200, v2_pod("p1", status="PROVISIONING", uptime=None)))

    status = api.get_pod_status("p1", "k")

    assert status["desired_status"] == "PROVISIONING"
    assert status["ip"] is None
    assert status["ports"] == []
    assert status["uptime_seconds"] == 0


def test_get_pod_status_returns_none_for_missing_pod(runpod_api) -> None:
    runpod_api.add("GET", "/pods/gone", problem(404, "pod not found", "Not Found"))

    assert api.get_pod_status("gone", "k") is None


def test_get_pod_status_returns_none_and_logs_on_api_error(runpod_api, caplog) -> None:
    runpod_api.add("GET", "/pods/p1", problem(401, "bad key", "Unauthorized"))

    assert api.get_pod_status("p1", "k") is None
    assert "HTTP 401" in caplog.text


def test_list_pods_follows_pagination(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/pods",
        FakeResponse(200, {"pods": [v2_pod("a")], "pagination": {"nextCursor": "c1", "hasNextPage": True}}),
        FakeResponse(200, {"pods": [v2_pod("b")], "pagination": {"nextCursor": None, "hasNextPage": False}}),
    )

    pods = api.list_pods("k")

    assert [p["id"] for p in pods] == ["a", "b"]
    assert [c.params for c in runpod_api.calls_to("GET", "/pods")] == [{}, {"cursor": "c1"}]


def test_terminate_pod_sends_delete(runpod_api) -> None:
    runpod_api.add("DELETE", "/pods/p1", FakeResponse(204))

    api.terminate_pod("p1", "k")

    assert [c.method for c in runpod_api.calls] == ["DELETE"]


# ---------------------------------------------------------------------------
# GPU catalogue
# ---------------------------------------------------------------------------


def test_find_gpu_type_matches_name_or_id_and_adds_legacy_keys(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/catalog/gpus",
        FakeResponse(200, {"gpus": [{"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "memory": 24}]}),
    )

    by_name = api.find_gpu_type("RTX 4090", "k")
    by_id = api.find_gpu_type("NVIDIA GeForce RTX 4090", "k")

    assert by_name == by_id
    assert by_name["displayName"] == "RTX 4090"
    assert by_name["memoryInGb"] == 24
    assert api.find_gpu_type("H100", "k") is None


def test_find_gpu_type_returns_none_on_api_error(runpod_api) -> None:
    runpod_api.add("GET", "/catalog/gpus", problem(401, "bad key"))

    assert api.find_gpu_type("RTX 4090", "k") is None


# ---------------------------------------------------------------------------
# Network volumes
# ---------------------------------------------------------------------------


def test_get_network_volumes_unwraps_and_aliases_datacenter(runpod_api) -> None:
    runpod_api.add(
        "GET",
        "/network-volumes",
        FakeResponse(200, {"networkVolumes": [
            {"id": "v1", "name": "vol", "size": 100, "dataCenter": "EU-RO-1", "type": "STANDARD"},
        ]}),
    )

    volumes = api.get_network_volumes("k")

    assert volumes == [{
        "id": "v1",
        "name": "vol",
        "size": 100,
        "dataCenter": "EU-RO-1",
        "dataCenterId": "EU-RO-1",
        "type": "STANDARD",
    }]


def test_get_network_volumes_returns_empty_list_on_failure(runpod_api) -> None:
    runpod_api.add("GET", "/network-volumes", problem(401, "bad key"))

    assert api.get_network_volumes("k") == []


def test_create_network_volume_posts_v2_shape(runpod_api) -> None:
    runpod_api.add(
        "POST",
        "/network-volumes",
        FakeResponse(201, {"id": "vol-created-1", "name": "my-volume", "size": 100, "dataCenter": "dc-test", "type": "STANDARD"}),
    )

    result = api.create_network_volume("fake-key", "my-volume", 100, "dc-test")

    assert result["id"] == "vol-created-1"
    assert result["dataCenterId"] == "dc-test"
    assert runpod_api.calls_to("POST", "/network-volumes")[0].json == {
        "name": "my-volume",
        "size": 100,
        "dataCenter": "dc-test",
    }


def test_create_network_volume_raises_on_error(runpod_api) -> None:
    runpod_api.add("POST", "/network-volumes", problem(400, "bad request"))

    with pytest.raises(RuntimeError, match="Failed to create network volume 'bad-vol'"):
        api.create_network_volume("fake-key", "bad-vol", 100, "dc-test")


def test_update_network_volume_size_patches(runpod_api) -> None:
    runpod_api.add(
        "PATCH",
        "/network-volumes/v1",
        FakeResponse(200, {"id": "v1", "name": "vol", "size": 150, "dataCenter": "EU-RO-1", "type": "STANDARD"}),
    )

    result = api.update_network_volume_size("k", "v1", 150)

    assert result["size"] == 150
    assert runpod_api.calls_to("PATCH", "/network-volumes/v1")[0].json == {"size": 150}
