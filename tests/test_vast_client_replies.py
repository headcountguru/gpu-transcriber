"""Odd Vast.ai replies: never read as "gone"; one bad row must not hide the others."""

import logging

import pytest
import responses

from app.vast_client import VastClient, VastError
from tests.test_vast_client import API_KEY, BASE, V1, raw_instance, raw_offer

URL = f"{BASE}/instances/883/"


@pytest.fixture
def client() -> VastClient:
    return VastClient(API_KEY)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"success": True},
        [],
        {"instances": "gone?"},
        {"instances": 42},
        {"instances": [raw_instance(id=999)]},
    ],
    ids=["empty-object", "no-instances-key", "list-body", "string", "number", "other-id"],
)
@responses.activate
def test_get_instance_raises_on_a_reply_that_does_not_say_gone(
    client: VastClient, body: object
) -> None:
    responses.get(URL, json=body)

    with pytest.raises(VastError):
        client.get_instance(883)


@pytest.mark.parametrize("empty", [None, {}, []], ids=["null", "empty-object", "empty-list"])
@responses.activate
def test_get_instance_explicitly_empty_means_gone(client: VastClient, empty: object) -> None:
    responses.get(URL, json={"instances": empty})

    assert client.get_instance(883) is None


@responses.activate
def test_get_instance_finds_its_id_in_a_list(client: VastClient) -> None:
    responses.get(URL, json={"instances": [raw_instance(id=1), raw_instance(id=883)]})

    found = client.get_instance(883)

    assert found is not None and found.id == 883


@responses.activate
def test_list_instances_skips_an_unreadable_row_and_keeps_the_rest(
    client: VastClient, caplog: pytest.LogCaptureFixture
) -> None:
    bad = raw_instance(id="not-a-number", label="gputx-bad", ssh_host="secret-host.example")
    responses.get(
        f"{V1}/instances",
        json={
            "next_token": None,
            "instances": [
                raw_instance(id=1, label="gputx-a"),
                bad,
                "not a row",
                raw_instance(id=2, label="gputx-b"),
            ],
        },
    )

    with caplog.at_level(logging.WARNING):
        found = client.list_instances("gputx-")

    assert [i.id for i in found] == [1, 2]
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "an unreadable row carrying the app's label must be warned about"
    assert "secret-host.example" not in caplog.text
    assert "gputx-bad" not in caplog.text
    assert API_KEY not in caplog.text


@responses.activate
def test_search_offers_reads_the_host_machine_id(client: VastClient) -> None:
    responses.post(
        f"{BASE}/bundles/",
        json={"offers": [raw_offer(id=5, machine_id=77), raw_offer(id=6)]},
    )

    offers = client.search_offers()

    assert [(o.id, o.machine_id) for o in offers] == [(5, 77), (6, None)]
