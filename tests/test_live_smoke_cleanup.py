"""Tests for the live smoke script's safety paths: destroy, orphan sweep and spend cap.

Everything uses a fake Vast client and a fake clock; nothing touches the network.
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

import scripts.live_smoke as ls
from app.settings import Settings
from app.vast_client import Instance, Offer, VastError
from scripts.live_smoke import (
    A_SELF_DESTROY,
    Prepared,
    SmokeRun,
    SpendCapReached,
    SpendGuard,
)

LABEL = "gputx-smoke-test"


def _offer(dph: float = 0.4, offer_id: int = 5) -> Offer:
    return Offer(offer_id, "RTX 4090", 24.0, dph, 900.0, 0.99, True, 12.6)


def _instance(instance_id: int = 42, status: str = "running") -> Instance:
    return Instance(instance_id, status, "host", 22, 0.4, LABEL)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeClient:
    """Scriptable stand-in for VastClient. The last scripted result repeats."""

    def __init__(self) -> None:
        self.get_result: Callable[[int], Instance | None] = lambda _id: None
        self.destroy_effects: list[Exception | None] = [None]
        self.list_results: list[list[Instance]] = [[]]
        self.offers: list[Offer] = [_offer()]
        self.destroy_calls: list[int] = []
        self.create_calls = 0
        self.list_calls = 0

    def destroy_instance(self, instance_id: int) -> None:
        self.destroy_calls.append(instance_id)
        effect = self.destroy_effects[min(len(self.destroy_calls), len(self.destroy_effects)) - 1]
        if effect is not None:
            raise effect

    def get_instance(self, instance_id: int) -> Instance | None:
        return self.get_result(instance_id)

    def list_instances(self, label_prefix: str) -> list[Instance]:
        index = min(self.list_calls, len(self.list_results) - 1)
        self.list_calls += 1
        return self.list_results[index]

    def create_instance(self, *args: object) -> int:
        self.create_calls += 1
        return 99

    def get_balance(self) -> float:
        return 10.0

    def search_offers(self, settings: Settings | None = None) -> list[Offer]:
        return self.offers

    def register_ssh_key(self, key: str) -> None:
        return None


def _run(client: FakeClient, clock: FakeClock) -> SmokeRun:
    run = SmokeRun(
        client,  # type: ignore[arg-type]
        Settings(image="img"),
        ["vast-secret", "hf-secret"],
        clock=clock,
        sleep=clock.sleep,
    )
    run.label = LABEL
    run.create_attempted = True
    return run


# --- destroy -----------------------------------------------------------------------------------


def test_destroy_is_called_even_when_get_instance_says_not_found() -> None:
    client, clock = FakeClient(), FakeClock()
    run = _run(client, clock)
    run.instance_id = 42  # get_instance returns None for everything, as if Vast lagged

    assert ls.cleanup(run) is True

    assert client.destroy_calls == [42]


def test_destroy_retries_after_an_error_then_succeeds() -> None:
    client, clock = FakeClient(), FakeClock()
    client.destroy_effects = [VastError("boom", status_code=500), None]
    run = _run(client, clock)
    run.instance_id = 42

    assert ls.cleanup(run) is True

    assert client.destroy_calls == [42, 42]
    assert clock.now >= ls.DESTROY_RETRY_S


def test_destroy_that_never_confirms_gone_is_critical(caplog: pytest.LogCaptureFixture) -> None:
    client, clock = FakeClient(), FakeClock()
    client.get_result = lambda instance_id: _instance(instance_id)  # never goes away
    run = _run(client, clock)
    run.instance_id = 42

    with caplog.at_level(logging.CRITICAL, logger="live_smoke"):
        assert ls.cleanup(run) is False

    assert any("42" in r.getMessage() and r.levelno == logging.CRITICAL for r in caplog.records)


def test_a_404_on_destroy_counts_as_gone_and_proves_self_destroy() -> None:
    client, clock = FakeClient(), FakeClock()
    client.destroy_effects = [VastError("no such instance", status_code=404)]
    run = _run(client, clock)
    run.instance_id = 42
    run.self_destroy_attempted = True

    assert ls.cleanup(run) is True

    assert client.destroy_calls == [42]
    assert run.log.result(A_SELF_DESTROY) == "PASS"
    assert run.tracker.summary()[0] == "NOT RUN"  # the 404 is not an endpoint failure


# --- ambiguous create re-sweep -----------------------------------------------------------------


def test_resweep_finds_an_instance_that_only_appears_on_the_third_look(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, clock = FakeClient(), FakeClock()
    client.list_results = [[], [], [_instance(77)]]
    client.get_result = lambda i: _instance(i) if not client.destroy_calls else None
    run = _run(client, clock)  # create attempted, no instance id known

    with caplog.at_level(logging.CRITICAL, logger="live_smoke"):
        assert ls.cleanup(run) is True

    assert client.list_calls == 3
    assert client.destroy_calls == [77]
    assert run.orphan_found is True
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_resweep_warns_with_the_label_when_nothing_appears(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, clock = FakeClient(), FakeClock()
    run = _run(client, clock)

    with caplog.at_level(logging.WARNING, logger="live_smoke"):
        assert ls.cleanup(run) is True

    assert client.list_calls > 3
    assert clock.now >= ls.RESWEEP_WINDOW_S
    assert run.orphan_found is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(LABEL in message and "Vast.ai console" in message for message in warnings)


def test_resweep_that_can_never_list_is_critical(caplog: pytest.LogCaptureFixture) -> None:
    client, clock = FakeClient(), FakeClock()

    def broken(_label: str) -> list[Instance]:
        raise VastError("down")

    client.list_instances = broken  # type: ignore[method-assign]
    run = _run(client, clock)

    with caplog.at_level(logging.CRITICAL, logger="live_smoke"):
        assert ls.cleanup(run) is False

    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


# --- spend guard -------------------------------------------------------------------------------


def test_spend_guard_raises_at_the_cap() -> None:
    clock = FakeClock()
    guard = SpendGuard(clock, created_at=0.0, price_per_hour=3600.0, max_spend=5.0)  # $1 per second

    clock.now = 4.0
    guard.check()
    clock.now = 5.0
    with pytest.raises(SpendCapReached):
        guard.check()


def test_spend_guard_stops_the_endpoint_wait() -> None:
    client, clock = FakeClient(), FakeClock()
    client.get_result = lambda i: _instance(i, status="loading")
    run = _run(client, clock)
    run.guard = SpendGuard(clock, 0.0, 3600.0, 25.0)

    with pytest.raises(SpendCapReached):
        ls.wait_for_endpoint(run, 42)

    assert clock.now < ls.STARTUP_TIMEOUT_S


def test_spend_guard_stops_an_upload_in_progress(tmp_path: Path) -> None:
    client, clock = FakeClient(), FakeClock()
    run = _run(client, clock)
    run.guard = SpendGuard(clock, 0.0, 3600.0, 50.0)
    audio = tmp_path / "audio.opus"
    audio.write_bytes(b"x")
    prepared = Prepared(audio, 100.0, _offer(), tmp_path / "key")

    class SlowSession:
        def upload(self, local: Path, remote: str, on_progress: Callable[[int, int], None]) -> None:
            for sent in range(1, 100):
                clock.now += 10.0  # each chunk takes 10 s
                on_progress(sent, 100)

    with pytest.raises(SpendCapReached):
        ls.upload_inputs(run, SlowSession(), prepared, tmp_path)  # type: ignore[arg-type]

    assert clock.now < 100.0


def test_spend_guard_stops_the_ssh_connect_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    client, clock = FakeClient(), FakeClock()
    run = _run(client, clock)
    run.guard = SpendGuard(clock, 0.0, 3600.0, 12.0)

    def fake_connect(*_args: object, sleep: Callable[[float], None], **_kw: object) -> None:
        while True:  # sshd never answers; the real connect sleeps between attempts
            sleep(5.0)

    monkeypatch.setattr(ls.SshRemoteSession, "connect", staticmethod(fake_connect))

    with pytest.raises(SpendCapReached):
        ls.connect_ssh(run, _instance(), Path("key"))


# --- pinned offers, and the whole run_smoke paths ----------------------------------------------


def _smoke_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")
    args = ls.build_parser().parse_args(["--audio", str(audio), *extra])
    args.out_dir = tmp_path / "out"
    args.report_path = tmp_path / "report.md"
    return args


def _patch_preflight(monkeypatch: pytest.MonkeyPatch, client: FakeClient, tmp_path: Path) -> None:
    monkeypatch.setattr(ls, "VastClient", lambda _key: client)
    monkeypatch.setattr(ls, "compress_audio", lambda _src, _dst: 100.0)
    monkeypatch.setattr(ls, "app_data_dir", lambda: tmp_path)
    monkeypatch.setattr(ls, "ensure_ssh_key", lambda _d: (tmp_path / "key", "ssh-ed25519 AAA"))


@pytest.mark.parametrize(
    ("dph", "extra"),
    [
        (2.0, ["--max-spend", "5"]),  # over the $1.00/h ceiling
        (0.9, ["--max-spend", "0.2"]),  # 0.9 x 0.25 h = $0.225 > --max-spend
    ],
)
def test_pinned_offer_that_is_too_expensive_is_refused_without_renting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dph: float, extra: list[str]
) -> None:
    client = FakeClient()
    client.offers = [_offer(dph=dph, offer_id=5)]
    _patch_preflight(monkeypatch, client, tmp_path)
    args = _smoke_args(tmp_path, "--offer-id", "5", *extra)

    code = ls.run_smoke(args, "vast-secret", "hf-secret")

    assert code == ls.EXIT_REFUSED
    assert client.create_calls == 0


def _fake_prepared(tmp_path: Path) -> Prepared:
    return Prepared(tmp_path / "a.opus", 100.0, _offer(), tmp_path / "key")


def test_run_exits_3_when_the_machine_cannot_be_confirmed_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, clock = FakeClient(), FakeClock()
    client.get_result = lambda i: _instance(i)  # never goes away
    _patch_preflight(monkeypatch, client, tmp_path)
    monkeypatch.setattr(ls, "prepare", lambda _run, _args, _tmp: _fake_prepared(tmp_path))

    def fake_execute(run: SmokeRun, *_a: object) -> str:
        run.instance_id = 42
        run.label = LABEL
        return "done"

    monkeypatch.setattr(ls, "execute", fake_execute)

    code = ls.run_smoke(
        _smoke_args(tmp_path), "vast-secret", "hf-secret", clock=clock, sleep=clock.sleep
    )

    assert code == ls.EXIT_MACHINE_NOT_CONFIRMED_GONE


def test_spend_cap_during_startup_is_cleaned_up_and_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, clock = FakeClient(), FakeClock()
    _patch_preflight(monkeypatch, client, tmp_path)
    monkeypatch.setattr(ls, "prepare", lambda _run, _args, _tmp: _fake_prepared(tmp_path))

    def fake_execute(run: SmokeRun, *_a: object) -> str:
        run.instance_id = 42
        run.label = LABEL
        raise SpendCapReached("cap")

    monkeypatch.setattr(ls, "execute", fake_execute)
    args = _smoke_args(tmp_path)

    code = ls.run_smoke(args, "vast-secret", "hf-secret", clock=clock, sleep=clock.sleep)

    assert code == ls.EXIT_JOB_PROBLEM
    assert client.destroy_calls == [42]
    report = args.report_path.read_text(encoding="utf-8")
    assert "stopped at spend cap" in report
