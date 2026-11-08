"""Billing-safety cases of the orchestrator found by the final review.

Reuses the in-memory world of ``tests/test_orchestrator.py``: a fake clock that moves only when
the orchestrator sleeps, a fake Vast.ai and a fake machine that advances one step per status read.
"""

import dataclasses
import json

import app.orchestrator_support as support
from app.jobs import JobState
from common.protocol import MAX_REMOTE_TEXT_CHARS
from tests.test_orchestrator import (
    CKPT_DIR,
    HF_TOKEN,
    OFFER,
    POLL_S,
    PRICE,
    STATUS_PATH,
    FakeMachine,
    World,
    assert_no_secrets,
)
from tests.test_orchestrator import world as world  # noqa: F401  (the fixture)

BIG_CAP = 100.0
HEARTBEAT_S = 60.0


class SilentMachine(FakeMachine):
    """A runner that was launched but never writes status.json (crashed on import, killed)."""

    def tick(self) -> None:
        if self.launched:
            self.ticks += 1


class FrozenMachine(FakeMachine):
    """A runner that wrote one running status (with a heartbeat) and then died."""

    def tick(self) -> None:
        if not self.launched:
            return
        self.ticks += 1
        if self.ticks == 1:
            self.fs[f"{CKPT_DIR}/words.json"] = b'{"words": []}'
            self._status("running", "labeling", 70, updated_at=1000.0, heartbeat_s=HEARTBEAT_S)


class SlowHealthyMachine(FakeMachine):
    """One long step with no progress, but the heartbeat keeps ``updated_at`` fresh."""

    def __init__(self, silent_ticks: int, heartbeat_s: float = HEARTBEAT_S) -> None:
        super().__init__()
        self.silent_ticks = silent_ticks
        self.heartbeat_s = heartbeat_s

    def tick(self) -> None:
        if not self.launched:
            return
        self.ticks += 1
        if self.ticks <= self.silent_ticks:
            beat = float(self.ticks) if self.heartbeat_s else 5.0  # an old runner never beats
            self._status("running", "aligning", 90, updated_at=beat, heartbeat_s=self.heartbeat_s)
            return
        self._step()


def rich_world(world: World) -> World:
    world.vast.balance = 10_000.0
    return world


# --- B1: a crashed or stalled runner is detected ------------------------------------------------


def test_runner_that_never_reports_fails_the_job_and_destroys_the_machine(world: World) -> None:
    rich_world(world).plans.append(SilentMachine())

    job = world.submit_and_run(cap=BIG_CAP)

    assert job.state is JobState.FAILED
    assert "stopped reporting progress" in (job.error or "")
    assert world.vast.instances == {}
    assert job.instance_id is None
    assert world.clock.now <= support.NO_FIRST_STATUS_S + 10 * POLL_S + 600  # long before the cap


def test_stale_running_status_fails_the_job_and_keeps_the_finished_parts(world: World) -> None:
    rich_world(world).plans.append(FrozenMachine())

    job = world.submit_and_run(cap=BIG_CAP)

    assert job.state is JobState.FAILED
    assert "stopped reporting progress" in (job.error or "")
    assert world.vast.instances == {}
    assert "words.json" in world.ckpt_names(job.id)
    assert job.cost_so_far < BIG_CAP / 10


def test_healthy_slow_step_with_heartbeat_never_trips_the_stall_check(world: World) -> None:
    silent_polls = int(3 * support.STALE_STATUS_S / POLL_S)  # three stale limits of one silent step
    rich_world(world).plans.append(SlowHealthyMachine(silent_ticks=silent_polls))

    job = world.submit_and_run(cap=BIG_CAP)

    assert job.state is JobState.DONE


def test_old_runner_without_heartbeat_is_never_judged_stale(world: World) -> None:
    silent_polls = int(3 * support.STALE_STATUS_S / POLL_S)
    machine = SlowHealthyMachine(silent_ticks=silent_polls, heartbeat_s=0.0)
    rich_world(world).plans.append(machine)

    job = world.submit_and_run(cap=BIG_CAP)

    assert job.state is JobState.DONE


# --- B8: remote text is bounded -----------------------------------------------------------------


def test_long_remote_error_is_shortened_and_scrubbed(world: World) -> None:
    error = "ffmpeg said " + "x" * 3000 + f" token {HF_TOKEN} " + "y" * 3000 + " the real reason"
    world.plans.append(FakeMachine(fail_error=error))

    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert_no_secrets(job)
    assert len(job.error or "") < MAX_REMOTE_TEXT_CHARS + 120
    assert (job.error or "").endswith("the real reason")


def test_long_remote_progress_message_is_shortened(world: World) -> None:
    class Chatty(FakeMachine):
        def _status(self, state: str, stage: str, percent: int, **extra: object) -> None:
            super()._status(state, stage, percent, **extra)
            body = json.loads(self.fs[STATUS_PATH])
            body["message"] = "m" * 5000
            self.fs[STATUS_PATH] = json.dumps(body).encode()

    world.plans.append(Chatty())
    world.submit_and_run()

    shown = [u.message for u in world.updates if u.message.startswith("mmm")]
    assert shown and all(len(m) <= MAX_REMOTE_TEXT_CHARS for m in shown)


# --- B2: a job marked lost at start-up still has its saved machine destroyed ---------------------


def _unfinished_job_with_unlisted_machine(world: World) -> tuple[str, int]:
    job = world.orch.submit("C:/audio/talk.mp3", is_youtube=False)
    instance_id = world.vast.add_instance(label=None)  # not yet labelled: the listing misses it
    world.store.save(
        job.with_(
            state=JobState.LABELING,
            instance_id=instance_id,
            machine_started_at=0.0,
            price_per_hour=PRICE,
        )
    )
    return job.id, instance_id


def test_resume_destroys_the_saved_machine_before_marking_the_job_lost(world: World) -> None:
    job_id, instance_id = _unfinished_job_with_unlisted_machine(world)

    assert world.orch.resume_pending() == []

    job = world.store.get(job_id)
    assert job.state is JobState.FAILED
    assert instance_id in world.vast.destroyed
    assert job.instance_id is None


def test_resume_keeps_the_machine_id_when_its_destroy_is_not_confirmed(world: World) -> None:
    job_id, instance_id = _unfinished_job_with_unlisted_machine(world)
    world.vast.destroy_always_fails = True

    world.orch.resume_pending()

    job = world.store.get(job_id)
    assert job.state is JobState.FAILED
    assert job.instance_id == instance_id
    assert str(instance_id) in job.message


# --- B5: start-up work never touches a job that is running --------------------------------------


def test_resume_and_reap_skip_a_job_that_is_running(world: World) -> None:
    job = world.orch.submit("C:/audio/talk.mp3", is_youtube=False)
    machine = world.vast.add_instance(label=f"gputx-{job.id}")
    world.store.save(job.with_(state=JobState.FAILED))  # stale on disk; the run is live
    world.orch._running.add(job.id)  # as if run() had claimed it on another thread
    try:
        assert world.orch.resume_pending() == []
        assert world.orch.reap_orphans() == []
    finally:
        world.orch._running.discard(job.id)

    assert machine in world.vast.instances
    assert world.vast.destroyed == []


def test_resume_does_not_mark_a_running_job_lost(world: World) -> None:
    job = world.orch.submit("C:/audio/talk.mp3", is_youtube=False)
    world.orch._running.add(job.id)
    try:
        world.orch.resume_pending()
    finally:
        world.orch._running.discard(job.id)

    assert world.store.get(job.id).state is JobState.PREPARING


# --- C7: retrying a destroy saves the newest job, not a stale copy -------------------------------


def test_retry_unconfirmed_destroy_keeps_changes_made_meanwhile(world: World) -> None:
    job = world.orch.submit("C:/audio/talk.mp3", is_youtube=False)
    instance_id = world.vast.add_instance(label=f"gputx-{job.id}")
    stuck = job.with_(
        state=JobState.STOPPED_AT_CAP,
        instance_id=instance_id,
        machine_started_at=0.0,
        price_per_hour=PRICE,
        cost_cap=1.0,
    )
    world.store.save(stuck)
    real_destroy = world.vast.destroy_instance

    def destroy_while_cap_is_raised(target: int) -> None:
        world.store.save(world.store.get(job.id).with_(state=JobState.PREPARING, cost_cap=9.0))
        real_destroy(target)

    world.vast.destroy_instance = destroy_while_cap_is_raised  # type: ignore[method-assign]

    world.orch.retry_unconfirmed_destroys()

    saved = world.store.get(job.id)
    assert saved.state is JobState.PREPARING
    assert saved.cost_cap == 9.0
    assert saved.instance_id is None


# --- B7: a replacement machine is never the offer or host that just failed ----------------------


def test_replacement_skips_the_failed_offer_and_its_host(world: World) -> None:
    first = dataclasses.replace(OFFER, id=7, machine_id=1)
    same_host = dataclasses.replace(OFFER, id=8, machine_id=1, dph_total=PRICE + 0.01)
    other_host = dataclasses.replace(OFFER, id=9, machine_id=2, dph_total=PRICE + 0.02)
    world.vast.offers = [first, same_host, other_host]
    world.plans.append(FakeMachine(die_after=2))

    job = world.submit_and_run()

    assert job.state is JobState.DONE
    assert [c["offer_id"] for c in world.vast.created] == [7, 9]


def test_no_untried_offer_left_fails_in_plain_words(world: World) -> None:
    world.vast.offers = [OFFER]
    world.plans.append(FakeMachine(die_after=2))

    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert "already failed" in (job.error or "")
    assert len(world.vast.created) == 1
    assert world.vast.instances == {}
