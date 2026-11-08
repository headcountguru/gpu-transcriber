"""A destroy is confirmed only when Vast.ai's instance lookup AND its listing agree it is gone."""

from app.orchestrator_support import destroy_and_confirm
from app.settings import Settings
from app.vast_client import Instance, Offer, VastError

PREFIX = "gputx-"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class LaggingVast:
    """``get_instance`` says gone at once; the listing may still show the machine."""

    def __init__(self, listed_for_polls: int = 0, list_fails: bool = False) -> None:
        self.listed_for_polls = listed_for_polls
        self.list_fails = list_fails
        self.list_calls = 0
        self.destroyed: list[int] = []

    def search_offers(self, settings: Settings | None = None) -> list[Offer]:
        return []

    def create_instance(
        self, offer_id: int, image: str, label: str, disk_gb: int, onstart: str, env: dict
    ) -> int:
        raise AssertionError("not used")

    def get_instance(self, instance_id: int) -> Instance | None:
        return None

    def list_instances(self, label_prefix: str) -> list[Instance]:
        self.list_calls += 1
        if self.list_fails:
            raise VastError("could not reach Vast.ai")
        if self.list_calls <= self.listed_for_polls:
            return [Instance(5, "running", "h", 22, 0.3, f"{PREFIX}job")]
        return []

    def destroy_instance(self, instance_id: int) -> None:
        self.destroyed.append(instance_id)

    def get_balance(self) -> float:
        return 0.0

    def attach_ssh_key(self, instance_id: int, public_key: str) -> None:
        return None


def test_gone_only_after_the_listing_also_drops_the_machine() -> None:
    clock = Clock()
    vast = LaggingVast(listed_for_polls=2)

    assert destroy_and_confirm(vast, 5, clock, clock.sleep, PREFIX)
    assert vast.list_calls == 3


def test_not_confirmed_while_the_listing_keeps_showing_the_machine() -> None:
    clock = Clock()
    vast = LaggingVast(listed_for_polls=10_000)

    assert not destroy_and_confirm(vast, 5, clock, clock.sleep, PREFIX)


def test_not_confirmed_when_the_listing_cannot_be_read() -> None:
    clock = Clock()
    vast = LaggingVast(list_fails=True)

    assert not destroy_and_confirm(vast, 5, clock, clock.sleep, PREFIX)
