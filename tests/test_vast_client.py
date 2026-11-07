"""Tests for the Vast.ai REST client and offer ranking (HTTP mocked with ``responses``)."""

import json
from dataclasses import replace

import pytest
import requests
import responses

from app.settings import IMAGE, Settings, app_data_dir
from app.vast_client import (
    Instance,
    Offer,
    VastAmbiguousCreateError,
    VastClient,
    VastError,
    rank_offers,
)

BASE = "https://console.vast.ai/api/v0"
V1 = "https://console.vast.ai/api/v1"
API_KEY = "sekret-api-key-123"


def make_offer(**overrides) -> Offer:
    base = Offer(
        id=1,
        gpu_name="RTX 4090",
        gpu_ram_gb=24.0,
        dph_total=0.5,
        inet_down_mbps=800.0,
        reliability=0.99,
        datacenter=False,
        cuda_max_good=12.6,
    )
    return replace(base, **overrides)


def raw_offer(**overrides) -> dict:
    """An offer shaped like the docs' search response (only the fields we read)."""
    base = {
        "id": 12345678,
        "gpu_name": "RTX 4090",
        "gpu_ram": 24576,
        "dph_total": 0.234567,
        "inet_down": 9876.5,
        "reliability": 0.9999,
        "hosting_type": 0,
        "cuda_max_good": 12.6,
        "num_gpus": 1,
    }
    base.update(overrides)
    return base


def raw_instance(**overrides) -> dict:
    """An instance shaped like the docs' show-instance response."""
    base = {
        "id": 883,
        "actual_status": "running",
        "label": "gputx-abc",
        "ssh_host": "ssh2281.vast.ai",
        "ssh_port": 10882,
        "dph_total": 0.8021,
    }
    base.update(overrides)
    return base


@pytest.fixture
def client() -> VastClient:
    return VastClient(API_KEY)


@pytest.fixture
def settings() -> Settings:
    return Settings(image=IMAGE)


# --- settings ---------------------------------------------------------------------------------


def test_settings_defaults_match_the_spec():
    s = Settings(image="x")
    assert (s.min_vram_gb, s.min_reliability, s.min_down_mbps, s.min_cuda) == (
        16.0,
        0.97,
        200.0,
        12.4,
    )
    assert (s.max_price_per_hour, s.prefer_datacenter, s.disk_gb) == (1.0, True, 40)
    assert (s.poll_interval_s, s.max_machine_retries) == (10.0, 2)
    assert (s.label_prefix, s.remote_job_dir) == ("gputx-", "/work/job")
    with pytest.raises(AttributeError):
        s.disk_gb = 1  # type: ignore[misc]


def test_app_data_dir_is_created_under_appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    path = app_data_dir()
    assert path == tmp_path / "GpuTranscriber"
    assert path.is_dir()


# --- ranking ----------------------------------------------------------------------------------


def test_rank_offers_drops_unfit_and_prefers_datacenter_then_price(settings):
    good_dc_cheap = make_offer(id=1, datacenter=True, dph_total=0.60)
    good_dc_dear = make_offer(id=2, datacenter=True, dph_total=0.90)
    good_home_cheap = make_offer(id=3, datacenter=False, dph_total=0.20)
    low_vram = make_offer(id=10, gpu_ram_gb=12.0)
    low_reliability = make_offer(id=11, reliability=0.90)
    slow_download = make_offer(id=12, inet_down_mbps=50.0)
    old_cuda = make_offer(id=13, cuda_max_good=11.8)
    too_pricey = make_offer(id=14, dph_total=1.50)

    ranked = rank_offers(
        [
            too_pricey,
            good_home_cheap,
            low_vram,
            good_dc_dear,
            old_cuda,
            low_reliability,
            good_dc_cheap,
            slow_download,
        ],
        settings,
    )

    assert [o.id for o in ranked] == [1, 2, 3]


def test_rank_offers_keeps_offers_exactly_at_the_thresholds(settings):
    edge = make_offer(
        gpu_ram_gb=16.0,
        reliability=0.97,
        inet_down_mbps=200.0,
        cuda_max_good=12.4,
        dph_total=1.0,
    )
    assert rank_offers([edge], settings) == [edge]


def test_rank_offers_ignores_datacenter_when_not_preferred(settings):
    dc = make_offer(id=1, datacenter=True, dph_total=0.60)
    home = make_offer(id=2, datacenter=False, dph_total=0.20)
    ranked = rank_offers([dc, home], replace(settings, prefer_datacenter=False))
    assert [o.id for o in ranked] == [2, 1]


def test_rank_offers_returns_empty_when_nothing_fits(settings):
    assert rank_offers([make_offer(gpu_ram_gb=8.0), make_offer(dph_total=5.0)], settings) == []
    assert rank_offers([], settings) == []


def test_rank_offers_drops_offers_with_missing_fields_instead_of_crashing(settings, caplog):
    broken = make_offer(id=99, reliability=None)  # type: ignore[arg-type]
    fine = make_offer(id=1)
    with caplog.at_level("DEBUG", logger="app.vast_client"):
        ranked = rank_offers([broken, fine], settings)
    assert [o.id for o in ranked] == [1]
    assert "99" in caplog.text


# --- search -----------------------------------------------------------------------------------


@responses.activate
def test_search_offers_posts_filters_and_converts_units(client):
    responses.post(f"{BASE}/bundles/", json={"offers": [raw_offer(hosting_type=1)]})

    offers = client.search_offers()

    assert offers == [
        Offer(
            id=12345678,
            gpu_name="RTX 4090",
            gpu_ram_gb=24.576,
            dph_total=0.234567,
            inet_down_mbps=9876.5,
            reliability=0.9999,
            datacenter=True,
            cuda_max_good=12.6,
        )
    ]
    request = responses.calls[0].request
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    body = json.loads(request.body)
    assert body["type"] == "ondemand"
    assert body["num_gpus"] == {"eq": 1}
    assert body["verified"] == {"eq": True}
    assert body["rentable"] == {"eq": True}


@responses.activate
def test_search_offers_skips_offers_missing_a_needed_field(client):
    incomplete = raw_offer(id=2)
    del incomplete["reliability"]
    null_ram = raw_offer(id=3, gpu_ram=None)
    responses.post(f"{BASE}/bundles/", json={"offers": [incomplete, null_ram, raw_offer(id=4)]})

    assert [o.id for o in client.search_offers()] == [4]


@responses.activate
def test_search_offers_returns_empty_list_when_response_has_no_offers(client):
    responses.post(f"{BASE}/bundles/", json={"offers": []})
    assert client.search_offers() == []


# --- create / get / list / destroy -------------------------------------------------------------


@responses.activate
def test_create_instance_sends_image_label_and_onstart(client):
    responses.put(f"{BASE}/asks/777/", json={"success": True, "new_contract": 4242})

    instance_id = client.create_instance(
        777, "repo/img@sha256:abc", "gputx-job1", 40, "bash /work/start.sh", {"HF_TOKEN": "hf_x"}
    )

    assert instance_id == 4242
    request = responses.calls[0].request
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    body = json.loads(request.body)
    assert body["image"] == "repo/img@sha256:abc"
    assert body["label"] == "gputx-job1"
    assert body["disk"] == 40
    assert body["onstart"] == "bash /work/start.sh"
    assert body["env"] == {"HF_TOKEN": "hf_x"}
    assert body["cancel_unavail"] is True


@responses.activate
def test_create_instance_raises_when_offer_is_gone(client):
    responses.put(
        f"{BASE}/asks/777/",
        status=400,
        json={"success": False, "error": "invalid_args", "msg": "Offer no longer available"},
    )
    with pytest.raises(VastError) as err:
        client.create_instance(777, "img", "gputx-x", 40, "true", {})
    assert "777" in str(err.value)
    assert "no longer available" in str(err.value)
    assert err.value.status_code == 400


@responses.activate
def test_create_instance_raises_when_response_has_no_instance_id(client):
    responses.put(f"{BASE}/asks/777/", json={"success": True})
    with pytest.raises(VastError):
        client.create_instance(777, "img", "gputx-x", 40, "true", {})


def test_create_instance_rejects_oversized_onstart_without_calling_the_api(client):
    with responses.RequestsMock() as rsps:  # asserts no request is made
        with pytest.raises(VastError):
            client.create_instance(777, "img", "gputx-x", 40, "x" * 5000, {})
        assert len(rsps.calls) == 0


@responses.activate
def test_get_instance_parses_a_running_instance(client):
    responses.get(f"{BASE}/instances/883/", json={"instances": raw_instance()})

    assert client.get_instance(883) == Instance(
        id=883,
        status="running",
        ssh_host="ssh2281.vast.ai",
        ssh_port=10882,
        dph_total=0.8021,
        label="gputx-abc",
    )


@responses.activate
def test_get_instance_still_loading_has_no_ssh_details(client):
    loading = raw_instance(actual_status=None, ssh_host="", ssh_port=None, label=None)
    responses.get(f"{BASE}/instances/883/", json={"instances": loading})

    got = client.get_instance(883)

    assert got is not None
    assert (got.status, got.ssh_host, got.ssh_port, got.label) == ("pending", None, None, None)


@responses.activate
def test_get_instance_returns_none_when_gone(client):
    responses.get(f"{BASE}/instances/883/", status=404, json={"error": "not_found"})
    assert client.get_instance(883) is None


@responses.activate
def test_get_instance_returns_none_when_response_is_empty(client):
    responses.get(f"{BASE}/instances/883/", json={"instances": None})
    assert client.get_instance(883) is None


@responses.activate
def test_get_instance_raises_on_other_failures(client):
    responses.get(f"{BASE}/instances/883/", status=500, body="boom")
    with pytest.raises(VastError) as err:
        client.get_instance(883)
    assert "883" in str(err.value)


@responses.activate
def test_list_instances_filters_by_label_prefix(client):
    responses.get(
        f"{V1}/instances",
        json={
            "success": True,
            "next_token": None,
            "instances": [
                raw_instance(id=1, label="gputx-a"),
                raw_instance(id=2, label="someone-elses"),
                raw_instance(id=3, label=None),
                raw_instance(id=4, label="gputx-b"),
            ],
        },
    )

    found = client.list_instances("gputx-")

    assert [i.id for i in found] == [1, 4]


@responses.activate
def test_list_instances_follows_pagination(client):
    responses.get(
        f"{V1}/instances",
        json={"next_token": "page2", "instances": [raw_instance(id=1, label="gputx-a")]},
    )
    responses.get(
        f"{V1}/instances",
        json={"next_token": None, "instances": [raw_instance(id=2, label="gputx-b")]},
    )

    found = client.list_instances("gputx-")

    assert [i.id for i in found] == [1, 2]
    assert "after_token=page2" in responses.calls[1].request.url


@responses.activate
def test_list_instances_raises_on_failure(client):
    responses.get(f"{V1}/instances", status=500, body="down")
    with pytest.raises(VastError):
        client.list_instances("gputx-")


@responses.activate
def test_destroy_instance_calls_delete(client):
    responses.delete(f"{BASE}/instances/883/", json={"success": True})
    client.destroy_instance(883)
    assert responses.calls[0].request.method == "DELETE"


@responses.activate
def test_destroy_instance_raises_vasterror_on_server_error(client):
    responses.delete(f"{BASE}/instances/883/", status=500, body="Internal Server Error")
    with pytest.raises(VastError) as err:
        client.destroy_instance(883)
    assert "883" in str(err.value)


# --- balance / ssh key --------------------------------------------------------------------------


@responses.activate
def test_get_balance_reads_balance(client):
    responses.get(f"{BASE}/users/current/", json={"id": 1, "balance": 12.5})
    assert client.get_balance() == 12.5


@responses.activate
def test_get_balance_falls_back_to_credit_field(client):
    responses.get(f"{BASE}/users/current/", json={"id": 1, "credit": 25.0})
    assert client.get_balance() == 25.0


@responses.activate
def test_get_balance_raises_when_no_balance_in_response(client):
    responses.get(f"{BASE}/users/current/", json={"id": 1})
    with pytest.raises(VastError):
        client.get_balance()


@responses.activate
def test_register_ssh_key_posts_the_public_key(client):
    responses.post(f"{BASE}/ssh/", json={"success": True, "key": {"id": 1}})
    client.register_ssh_key("ssh-ed25519 AAAAC3 gputx")
    assert json.loads(responses.calls[0].request.body) == {"ssh_key": "ssh-ed25519 AAAAC3 gputx"}


# --- error handling ---------------------------------------------------------------------------


@responses.activate
def test_bad_api_key_gives_a_plain_message_without_the_key(client):
    responses.get(f"{BASE}/users/current/", status=401, json={"msg": f"Invalid user key {API_KEY}"})
    with pytest.raises(VastError) as err:
        client.get_balance()
    message = str(err.value)
    assert API_KEY not in message
    assert "API key" in message


@responses.activate
def test_long_response_bodies_are_truncated_in_errors(client):
    responses.delete(f"{BASE}/instances/5/", status=500, body="x" * 5000)
    with pytest.raises(VastError) as err:
        client.destroy_instance(5)
    assert len(str(err.value)) < 600


@responses.activate
def test_rate_limit_message_is_plain_words(client):
    responses.get(f"{BASE}/users/current/", status=429, body="API requests too frequent")
    with pytest.raises(VastError) as err:
        client.get_balance()
    assert "too many requests" in str(err.value).lower()


@responses.activate
def test_network_failure_becomes_vasterror_without_the_key(client):
    responses.get(f"{BASE}/users/current/", body=requests.ConnectionError(f"boom {API_KEY}"))
    with pytest.raises(VastError) as err:
        client.get_balance()
    assert API_KEY not in str(err.value)
    assert "reach" in str(err.value).lower()


@responses.activate
def test_non_json_success_body_raises_vasterror(client):
    responses.get(f"{BASE}/users/current/", body="<html>oops</html>")
    with pytest.raises(VastError):
        client.get_balance()


@responses.activate
def test_custom_session_and_base_url_are_used():
    session = requests.Session()
    custom = VastClient(API_KEY, session=session, base_url="https://example.test/api/v0")
    responses.get("https://example.test/api/v0/users/current/", json={"balance": 3})
    responses.get("https://example.test/api/v1/instances", json={"instances": []})

    assert custom.get_balance() == 3.0
    assert custom.list_instances("gputx-") == []
    assert "Authorization" not in session.headers  # key is never stored on a shared session


# --- server-side search filters ---------------------------------------------------------------


@responses.activate
def test_search_offers_with_settings_sends_server_side_filters(client, settings):
    responses.post(f"{BASE}/bundles/", json={"offers": []})

    client.search_offers(settings)

    body = json.loads(responses.calls[0].request.body)
    assert body["gpu_ram"] == {"gte": 16000.0}  # MB, matches the /1000 conversion
    assert body["dph_total"] == {"lte": 1.0}
    assert body["reliability"] == {"gte": 0.97}
    assert body["inet_down"] == {"gte": 200.0}
    assert body["cuda_max_good"] == {"gte": 12.4}
    assert body["num_gpus"] == {"eq": 1}  # base filters are kept


@responses.activate
def test_search_offers_filters_follow_the_given_settings(client, settings):
    responses.post(f"{BASE}/bundles/", json={"offers": []})

    client.search_offers(replace(settings, min_vram_gb=24.0, max_price_per_hour=0.5))

    body = json.loads(responses.calls[0].request.body)
    assert body["gpu_ram"] == {"gte": 24000.0}
    assert body["dph_total"] == {"lte": 0.5}


@responses.activate
def test_search_offers_without_settings_sends_no_minimums(client):
    responses.post(f"{BASE}/bundles/", json={"offers": []})

    client.search_offers()

    body = json.loads(responses.calls[0].request.body)
    for key in ("gpu_ram", "dph_total", "reliability", "inet_down", "cuda_max_good"):
        assert key not in body


# --- ambiguous create -------------------------------------------------------------------------


@responses.activate
def test_create_instance_read_timeout_is_flagged_as_maybe_created(client):
    responses.put(f"{BASE}/asks/777/", body=requests.ReadTimeout(f"slow {API_KEY}"))

    with pytest.raises(VastAmbiguousCreateError) as err:
        client.create_instance(777, "img", "gputx-x", 40, "true", {})

    assert err.value.maybe_created is True
    assert isinstance(err.value, VastError)
    assert str(err.value) == "Vast.ai did not answer in time; a machine may have been rented"
    assert API_KEY not in str(err.value)


@responses.activate
def test_create_instance_connect_timeout_is_a_plain_vasterror(client):
    responses.put(f"{BASE}/asks/777/", body=requests.ConnectTimeout(f"slow {API_KEY}"))

    with pytest.raises(VastError) as err:
        client.create_instance(777, "img", "gputx-x", 40, "true", {})

    assert not isinstance(err.value, VastAmbiguousCreateError)
    assert getattr(err.value, "maybe_created", False) is False
    assert API_KEY not in str(err.value)


@responses.activate
def test_create_instance_connection_error_is_a_plain_vasterror(client):
    responses.put(f"{BASE}/asks/777/", body=requests.ConnectionError("refused"))

    with pytest.raises(VastError) as err:
        client.create_instance(777, "img", "gputx-x", 40, "true", {})

    assert not isinstance(err.value, VastAmbiguousCreateError)


@responses.activate
def test_read_timeout_on_other_calls_is_a_plain_vasterror(client):
    responses.get(f"{BASE}/users/current/", body=requests.ReadTimeout("slow"))

    with pytest.raises(VastError) as err:
        client.get_balance()

    assert not isinstance(err.value, VastAmbiguousCreateError)


@responses.activate
def test_every_request_passes_a_timeout(client):
    responses.post(f"{BASE}/bundles/", json={"offers": []})
    responses.put(f"{BASE}/asks/7/", json={"new_contract": 1})
    responses.get(f"{BASE}/instances/1/", json={"instances": raw_instance(id=1)})
    responses.get(f"{V1}/instances", json={"instances": []})
    responses.delete(f"{BASE}/instances/1/", json={"success": True})
    responses.get(f"{BASE}/users/current/", json={"balance": 1})
    responses.post(f"{BASE}/ssh/", json={"success": True})

    client.search_offers()
    client.create_instance(7, "img", "gputx-x", 40, "true", {})
    client.get_instance(1)
    client.list_instances("gputx-")
    client.destroy_instance(1)
    client.get_balance()
    client.register_ssh_key("ssh-ed25519 AAAA")

    assert len(responses.calls) == 7
    for call in responses.calls:
        timeout = call.request.req_kwargs["timeout"]
        assert isinstance(timeout, (int, float)) and timeout > 0
