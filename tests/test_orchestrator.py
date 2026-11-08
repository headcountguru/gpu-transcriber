"""Tests for the orchestrator, with in-memory fakes for Vast.ai, the machine and media prep.

``FakeMachine`` simulates the remote runner: after ``launch``, every read of ``status.json``
advances it one step (words.json, then one ``diar_NNN.json`` per step, then the three outputs),
skipping any checkpoint that was uploaded to it. A fake clock advances only when the
orchestrator sleeps, so nothing really waits.
"""

import dataclasses
import json
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from app.jobs import Job, JobState, JobStore
from app.media_prep import MediaPrepError, PreparedAudio
from app.orchestrator import Orchestrator, build_onstart
from app.orchestrator_support import CREATE_SWEEP_WINDOW_S
from app.settings import Settings
from app.vast_client import Instance, Offer, VastAmbiguousCreateError, VastError
from common.protocol import OUTPUT_FILES

VAST_KEY = "vast-SECRET-key-123"
HF_TOKEN = "hf_SECRETtoken456"
PRICE = 3.6  # dollars per hour = $0.001 per second, so costs are easy to read
POLL_S = 100.0
REMOTE_DIR = "/work/job"
STATUS_PATH = f"{REMOTE_DIR}/status.json"
CKPT_DIR = f"{REMOTE_DIR}/checkpoints"
OUT_DIR = f"{REMOTE_DIR}/output"

OFFER = Offer(
    id=7,
    gpu_name="RTX A5000",
    gpu_ram_gb=24.0,
    dph_total=PRICE,
    inet_down_mbps=900.0,
    reliability=0.99,
    datacenter=True,
    cuda_max_good=12.8,
)
SETTINGS = Settings(image="img@sha256:abc", max_price_per_hour=5.0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeCreds:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def get(self, name: str) -> str | None:
        return self._values.get(name)


class FakeMachine:
    """The remote file system plus a step-by-step simulation of ``run_job.py``."""

    def __init__(
        self,
        chunks: int = 3,
        die_after: int | None = None,
        fail_error: str | None = None,
        garbage_ticks: int = 0,
        extra_listing: dict[str, bytes] | None = None,
        vanish_on_die: bool = True,
    ) -> None:
        self.fs: dict[str, bytes] = {}
        self.vanish_on_die = vanish_on_die
        self.chunks = chunks
        self.die_after = die_after
        self.fail_error = fail_error
        self.garbage_ticks = garbage_ticks
        self.extra_listing = extra_listing or {}
        self.launched: list[tuple[str, dict[str, str]]] = []
        self.checkpoints_at_launch: list[str] = []
        self.computed: list[str] = []
        self.downloads: list[str] = []
        self.ticks = 0
        self.reachable = True
        self.on_die: Callable[[], None] = lambda: None

    def _status(self, state: str, stage: str, percent: int, **extra: object) -> None:
        body = {"state": state, "stage": stage, "percent": percent, "message": stage}
        body.update(extra)
        self.fs[STATUS_PATH] = json.dumps(body).encode()

    def tick(self) -> None:
        if not self.launched:
            return
        self.ticks += 1
        if self.die_after is not None and self.ticks > self.die_after:
            self.reachable = False
            self.on_die()
            raise OSError("connection reset")
        if self.ticks <= self.garbage_ticks:
            garbage = [b"{not json", b'{"state": "running"}']
            self.fs[STATUS_PATH] = garbage[(self.ticks - 1) % 2]
            return
        if self.fail_error is not None:
            self._status("failed", "labeling", 50, error=self.fail_error)
            return
        self._step()

    def _step(self) -> None:
        names = ["words.json"] + [f"diar_{i:03d}.json" for i in range(self.chunks)]
        for index, name in enumerate(names):
            path = f"{CKPT_DIR}/{name}"
            if path not in self.fs:
                self.fs[path] = json.dumps({"name": name}).encode()
                self.computed.append(name)
                stage = "transcribing" if index == 0 else "labeling"
                self._status("running", stage, 10 + index, chunk=index, chunks=self.chunks)
                return
        for name in OUTPUT_FILES:
            self.fs[f"{OUT_DIR}/{name}"] = f"text of {name}".encode()
        self._status("done", "done", 100)


class FakeSession:
    def __init__(self, machine: FakeMachine) -> None:
        self.m = machine

    def _check(self) -> None:
        if not self.m.reachable:
            raise OSError("machine unreachable")

    def upload(
        self, local: Path, remote: str, on_progress: Callable[[int, int], None] | None = None
    ) -> None:
        self._check()
        data = local.read_bytes()
        if on_progress is not None:
            on_progress(len(data), len(data))
        self.m.fs[remote] = data

    def download(self, remote: str, local: Path) -> None:
        self._check()
        everything = {**self.m.fs, **self.m.extra_listing}
        if remote not in everything:
            raise FileNotFoundError(remote)
        self.m.downloads.append(remote)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(everything[remote])

    def list_files(self, remote_dir: str) -> list[str]:
        self._check()
        prefix = remote_dir.rstrip("/") + "/"
        names = [p[len(prefix) :] for p in self.m.fs if p.startswith(prefix)]
        names = [name for name in names if "/" not in name]
        # Hostile names are listed exactly as given, slashes and all.
        names += [p[len(prefix) :] for p in self.m.extra_listing if p.startswith(prefix)]
        return names

    def read_text(self, remote_path: str) -> str | None:
        self._check()
        if remote_path == STATUS_PATH:
            self.m.tick()
        data = self.m.fs.get(remote_path)
        return None if data is None else data.decode()

    def launch(self, command: str, env: dict[str, str]) -> None:
        self._check()
        self.m.launched.append((command, dict(env)))
        self.m.checkpoints_at_launch = sorted(
            p.rsplit("/", 1)[1] for p in self.m.fs if p.startswith(CKPT_DIR + "/")
        )

    def close(self) -> None:
        pass


class FakeVast:
    def __init__(self, world: "World") -> None:
        self.world = world
        self.offers = [OFFER]
        self.balance = 100.0
        self.instances: dict[int, Instance] = {}
        self.next_id = 1000
        self.created: list[dict[str, object]] = []
        self.destroyed: list[int] = []
        self.boot_status = "running"
        self.create_error: VastError | None = None
        self.create_error_after_renting = False
        self.attach_failures = 0
        self.attach_calls = 0
        self.destroy_failures = 0
        self.destroy_always_fails = False
        self.list_calls = 0

    def search_offers(self, settings: Settings | None = None) -> list[Offer]:
        assert settings is not None
        return list(self.offers)

    def get_balance(self) -> float:
        return self.balance

    def create_instance(
        self, offer_id: int, image: str, label: str, disk_gb: int, onstart: str, env: dict
    ) -> int:
        self.created.append({"offer_id": offer_id, "label": label, "onstart": onstart, "env": env})
        if self.create_error is not None and not self.create_error_after_renting:
            raise self.create_error
        instance_id = self.add_instance(label, self.boot_status)
        self.world.machine_for(instance_id)
        if self.create_error is not None:
            raise self.create_error
        return instance_id

    def add_instance(self, label: str | None, status: str = "running") -> int:
        self.next_id += 1
        self.instances[self.next_id] = Instance(self.next_id, status, "h", 22, PRICE, label)
        return self.next_id

    def get_instance(self, instance_id: int) -> Instance | None:
        return self.instances.get(instance_id)

    def list_instances(self, label_prefix: str) -> list[Instance]:
        self.list_calls += 1
        return [i for i in self.instances.values() if i.label and i.label.startswith(label_prefix)]

    def destroy_instance(self, instance_id: int) -> None:
        if self.destroy_always_fails:
            raise VastError("Vast.ai reported an error (HTTP 500).", status_code=500)
        if self.destroy_failures > 0:
            self.destroy_failures -= 1
            raise VastError("Vast.ai reported an error (HTTP 500).", status_code=500)
        if instance_id not in self.instances:
            raise VastError("not found", status_code=404)
        del self.instances[instance_id]
        self.destroyed.append(instance_id)

    def attach_ssh_key(self, instance_id: int, public_key: str) -> None:
        self.attach_calls += 1
        if self.attach_failures > 0:
            self.attach_failures -= 1
            raise VastError("instance not ready (HTTP 400)", status_code=400)


class World:
    """Everything one test needs, wired into an Orchestrator."""

    def __init__(self, tmp_path: Path, prepare_error: str | None = None) -> None:
        self.clock = FakeClock()
        self.store = JobStore(tmp_path / "jobs")
        self.vast = FakeVast(self)
        self.plans: list[FakeMachine] = []
        self.machines: dict[int, FakeMachine] = {}
        self.connected: list[int] = []
        self.prepare_error = prepare_error
        self.prepare_calls = 0
        self.updates: list[Job] = []
        self.orch = Orchestrator(
            SETTINGS,
            self.store,
            FakeCreds({"vast_api_key": VAST_KEY, "hf_token": HF_TOKEN}),  # type: ignore[arg-type]
            self.vast,
            self.session_factory,
            self.prepare,
            clock=self.clock,
            sleep=self.clock.sleep,
            ssh_public_key="ssh-ed25519 AAAA test",
            poll_interval_s=POLL_S,
        )

    def machine_for(self, instance_id: int) -> FakeMachine:
        if instance_id not in self.machines:
            machine = self.plans.pop(0) if self.plans else FakeMachine()
            if machine.vanish_on_die:
                machine.on_die = lambda: self.vast.instances.pop(instance_id, None)
            self.machines[instance_id] = machine
        return self.machines[instance_id]

    def session_factory(self, instance: Instance) -> FakeSession:
        self.connected.append(instance.id)
        return FakeSession(self.machine_for(instance.id))

    def prepare(self, job: Job) -> PreparedAudio:
        self.prepare_calls += 1
        if self.prepare_error is not None:
            raise MediaPrepError(self.prepare_error)
        path = self.store.job_dir(job.id) / "audio" / "audio.opus"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opus-bytes")
        return PreparedAudio(path=path, title="My Talk", duration_s=3600.0, size_bytes=10)

    def submit_and_run(
        self, cap: float | None = None, cancel: threading.Event | None = None
    ) -> Job:
        job = self.orch.submit("C:/audio/talk.mp3", is_youtube=False)
        if cap is not None:
            self.store.save(job.with_(cost_cap=cap))
        return self.orch.run(job.id, self.updates.append, cancel)

    def ckpt_names(self, job_id: str) -> list[str]:
        folder = self.store.job_dir(job_id) / "checkpoints"
        return sorted(p.name for p in folder.iterdir())


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def assert_no_secrets(job: Job) -> None:
    for text in (job.message, job.error or ""):
        assert VAST_KEY not in text
        assert HF_TOKEN not in text


# --- the brief's tests ---------------------------------------------------------------------------


def test_happy_path_downloads_outputs_and_destroys_instance(world: World) -> None:
    job = world.submit_and_run()

    assert job.state is JobState.DONE
    out = world.store.job_dir(job.id) / "output"
    assert sorted(p.name for p in out.iterdir()) == sorted(OUTPUT_FILES)
    assert all((out / name).stat().st_size > 0 for name in OUTPUT_FILES)
    assert world.vast.instances == {}
    assert job.instance_id is None
    assert job.output_dir == str(out)
    assert world.store.get(job.id) == job
    assert world.updates and world.updates[-1] == job
    machine = next(iter(world.machines.values()))
    command, env = machine.launched[0]
    assert env == {"HF_TOKEN": HF_TOKEN}
    assert HF_TOKEN not in command
    assert "run_job.py" in command and REMOTE_DIR in command
    spec = json.loads(machine.fs[f"{REMOTE_DIR}/job.json"])
    assert spec["max_runtime_s"] >= 3600.0
    assert machine.fs[f"{REMOTE_DIR}/input/audio.opus"] == b"opus-bytes"
    created = world.vast.created[0]
    assert created["label"] == f"gputx-{job.id}"
    assert created["env"] == {}
    assert job.cost_so_far == pytest.approx(world.clock.now * PRICE / 3600)
    assert job.cost_cap > 0


def test_preparation_failure_never_rents_a_machine(tmp_path: Path) -> None:
    world = World(tmp_path, prepare_error="This file has no audio to transcribe.")
    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert job.error == "This file has no audio to transcribe."
    assert world.vast.created == []


def test_no_affordable_offer_fails_before_renting(world: World) -> None:
    world.vast.offers = [dataclasses.replace(OFFER, dph_total=50.0)]
    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert "No suitable machine" in (job.error or "")
    assert "Nothing was rented" in (job.error or "")
    assert world.vast.created == []


def test_low_balance_fails_before_renting(world: World) -> None:
    world.vast.balance = 0.5
    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert "balance" in (job.error or "")
    assert world.vast.created == []


def test_machine_failure_rents_replacement_and_resumes_with_local_checkpoints(
    world: World,
) -> None:
    world.plans = [FakeMachine(chunks=3, die_after=2), FakeMachine(chunks=3)]
    job = world.submit_and_run()

    assert job.state is JobState.DONE
    assert job.retries == 1
    first, second = (world.machines[i] for i in sorted(world.machines))
    assert first.computed == ["words.json", "diar_000.json"]
    assert second.checkpoints_at_launch == ["diar_000.json", "words.json"]
    assert second.computed == ["diar_001.json", "diar_002.json"]
    assert len(world.vast.created) == 2
    assert world.vast.instances == {}
    # Both machines' time is billed: the machines ran back to back from t=0.
    assert job.banked_cost == pytest.approx(world.clock.now * PRICE / 3600)
    assert job.banked_cost > second.ticks * POLL_S * PRICE / 3600


def test_machine_failures_beyond_the_retry_limit_fail_the_job(world: World) -> None:
    world.plans = [FakeMachine(die_after=1) for _ in range(SETTINGS.max_machine_retries + 1)]
    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert len(world.vast.created) == SETTINGS.max_machine_retries + 1
    assert world.vast.instances == {}


def test_unreachable_polls_count_as_machine_lost(world: World) -> None:
    # Unreachable over SSH, but Vast.ai still lists it as running.
    world.plans = [FakeMachine(chunks=1, die_after=1, vanish_on_die=False), FakeMachine(chunks=1)]
    job = world.submit_and_run()

    assert job.state is JobState.DONE
    assert job.retries == 1
    assert len(world.vast.destroyed) == 2
    assert world.vast.instances == {}


def test_cap_reached_stops_destroys_keeps_checkpoints(world: World) -> None:
    world.plans = [FakeMachine(chunks=20)]
    job = world.submit_and_run(cap=0.95)

    assert job.state is JobState.STOPPED_AT_CAP
    assert world.vast.instances == {}
    assert job.instance_id is None
    names = world.ckpt_names(job.id)
    assert "words.json" in names and "diar_000.json" in names
    assert job.cost_so_far >= 0.95
    assert job.banked_cost == job.cost_so_far
    assert "cap" in job.message.lower()


def test_raise_cap_and_continue_resumes_from_checkpoints(world: World) -> None:
    world.plans = [FakeMachine(chunks=20), FakeMachine(chunks=20)]
    stopped = world.submit_and_run(cap=0.95)
    assert stopped.state is JobState.STOPPED_AT_CAP
    first_done = list(world.machines[min(world.machines)].computed)

    with pytest.raises(ValueError):
        world.orch.raise_cap_and_continue(stopped.id, stopped.banked_cost)
    raised = world.orch.raise_cap_and_continue(stopped.id, 5.0)
    assert raised.state is JobState.PREPARING
    assert raised.cost_cap == 5.0
    job = world.orch.run(stopped.id, world.updates.append)

    assert job.state is JobState.DONE
    second = world.machines[max(world.machines)]
    assert second.checkpoints_at_launch == sorted(first_done)
    assert set(second.computed).isdisjoint(first_done)
    assert len(first_done) + len(second.computed) == 21
    assert world.prepare_calls == 1
    assert world.vast.instances == {}


def test_raise_cap_rejects_a_job_not_stopped_at_cap(world: World) -> None:
    job = world.submit_and_run()
    with pytest.raises(ValueError):
        world.orch.raise_cap_and_continue(job.id, 50.0)


def test_failed_remote_job_destroys_instance_and_surfaces_error(world: World) -> None:
    world.plans = [FakeMachine(fail_error=f"CUDA out of memory token={HF_TOKEN} key={VAST_KEY}")]
    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert "CUDA out of memory" in (job.error or "")
    assert "***" in (job.error or "")
    assert_no_secrets(job)
    assert world.vast.instances == {}


def test_destroy_failure_is_retried_and_reported(world: World) -> None:
    world.vast.destroy_failures = 2
    job = world.submit_and_run()
    assert job.state is JobState.DONE
    assert world.vast.instances == {}

    other = World(world.store._root.parent / "second")  # type: ignore[attr-defined]
    other.vast.destroy_always_fails = True
    failed = other.submit_and_run()
    instance_id = next(iter(other.vast.instances))
    assert "may still be running" in failed.message
    assert str(instance_id) in failed.message
    assert failed.instance_id == instance_id


def test_resume_pending_finds_instance_by_label_when_id_not_saved(world: World) -> None:
    job = world.orch.submit("C:/audio/talk.mp3", is_youtube=False)
    world.prepare(job)
    audio = world.store.job_dir(job.id) / "audio" / "audio.opus"
    saved = job.with_(
        state=JobState.STARTING_MACHINE,
        audio_path=str(audio),
        duration_s=3600.0,
        machine_started_at=0.0,
        price_per_hour=PRICE,
        cost_cap=5.0,
    )
    world.store.save(saved)
    instance_id = world.vast.add_instance(f"gputx-{job.id}")

    assert world.orch.resume_pending() == [job.id]
    assert world.store.get(job.id).instance_id == instance_id

    finished = world.orch.run(job.id, world.updates.append)
    assert finished.state is JobState.DONE
    assert world.vast.created == []
    assert world.vast.instances == {}


def test_resume_pending_marks_lost_machine_failed(world: World) -> None:
    job = world.orch.submit("C:/audio/talk.mp3", is_youtube=False)
    world.store.save(job.with_(state=JobState.TRANSCRIBING, instance_id=77, machine_started_at=0.0))
    (world.store.job_dir("broken")).mkdir(parents=True)
    (world.store.job_dir("broken") / "job.json").write_text("{bad", encoding="utf-8")

    assert world.orch.resume_pending() == []
    lost = world.store.get(job.id)
    assert lost.state is JobState.FAILED
    assert "machine was lost while the app was closed" in (lost.error or "")


def test_reap_orphans_destroys_unreferenced_labelled_instances_only(world: World) -> None:
    done = world.submit_and_run()
    active = world.orch.submit("C:/audio/b.mp3", is_youtube=False)
    world.store.save(active.with_(state=JobState.TRANSCRIBING))
    corrupt_dir = world.store.job_dir("corrupt")
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "job.json").write_text("{bad", encoding="utf-8")

    unlabelled = world.vast.add_instance(None)
    foreign = world.vast.add_instance("someone-else")
    of_done = world.vast.add_instance(f"gputx-{done.id}")
    of_active = world.vast.add_instance(f"gputx-{active.id}")
    unknown = world.vast.add_instance("gputx-nosuchjob")
    of_corrupt = world.vast.add_instance("gputx-corrupt")

    reaped = world.orch.reap_orphans()

    assert sorted(reaped) == sorted([of_done, unknown, of_corrupt])
    assert set(world.vast.instances) == {unlabelled, foreign, of_active}


def test_cancel_destroys_instance(world: World) -> None:
    cancel = threading.Event()
    world.plans = [FakeMachine(chunks=10)]

    def on_update(job: Job) -> None:
        if job.state is JobState.LABELING:
            cancel.set()

    job = world.orch.submit("C:/audio/talk.mp3", is_youtube=False)
    result = world.orch.run(job.id, on_update, cancel)

    assert result.state is JobState.CANCELLED
    assert world.vast.instances == {}
    assert world.vast.destroyed


# --- extra tests ---------------------------------------------------------------------------------


def test_attach_key_retries_then_succeeds(world: World) -> None:
    world.vast.attach_failures = 2
    job = world.submit_and_run()
    assert job.state is JobState.DONE
    assert world.vast.attach_calls == 3


def test_attach_key_persistent_failure_fails_and_destroys(world: World) -> None:
    world.vast.attach_failures = 99
    job = world.submit_and_run()
    assert job.state is JobState.FAILED
    assert world.vast.attach_calls == 6
    assert world.vast.instances == {}
    assert world.connected == []


def test_definite_create_refusal_skips_the_sweep(world: World) -> None:
    world.vast.create_error = VastError(f"refused (HTTP 400) {VAST_KEY}", status_code=400)
    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert world.vast.list_calls == 0
    assert job.machine_started_at is None
    assert job.cost_so_far == 0.0
    assert_no_secrets(job)


def test_ambiguous_create_sweeps_and_destroys_what_it_finds(world: World) -> None:
    world.vast.create_error = VastAmbiguousCreateError()
    world.vast.create_error_after_renting = True
    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert world.vast.list_calls >= 1
    assert len(world.vast.destroyed) == 1
    assert world.vast.instances == {}
    assert world.connected == []


def test_ambiguous_create_with_nothing_found_stops_after_the_window(world: World) -> None:
    world.vast.create_error = VastError("could not reach Vast.ai", status_code=None)
    job = world.submit_and_run()

    assert job.state is JobState.FAILED
    assert world.vast.list_calls > 1
    assert world.clock.now >= CREATE_SWEEP_WINDOW_S
    assert "gputx-" in (job.error or "")


def test_unparseable_status_is_treated_as_no_status(world: World) -> None:
    world.plans = [FakeMachine(chunks=1, garbage_ticks=2)]
    job = world.submit_and_run()
    assert job.state is JobState.DONE


def test_remote_path_traversal_is_never_written(world: World, tmp_path: Path) -> None:
    evil = {
        f"{CKPT_DIR}/../evil.json": b"x",
        f"{CKPT_DIR}/..\\diar_009.json": b"x",
        f"{CKPT_DIR}/diar_001.json/../../evil2.json": b"x",
        f"{CKPT_DIR}/notes.txt": b"x",
    }
    world.plans = [FakeMachine(chunks=1, extra_listing=evil)]
    job = world.submit_and_run()

    assert job.state is JobState.DONE
    machine = next(iter(world.machines.values()))
    assert not any(path in machine.downloads for path in evil)
    found = [p.name for p in tmp_path.rglob("*") if "evil" in p.name or p.name == "notes.txt"]
    assert found == []
    assert world.ckpt_names(job.id) == ["diar_000.json", "words.json"]


def test_spend_cap_during_startup_stops_at_cap(world: World) -> None:
    world.vast.boot_status = "loading"
    job = world.submit_and_run(cap=0.95)

    assert job.state is JobState.STOPPED_AT_CAP
    assert world.vast.instances == {}
    assert world.connected == []
    assert job.cost_so_far >= 0.95


def test_two_jobs_never_share_checkpoints(world: World) -> None:
    world.plans = [FakeMachine(chunks=20), FakeMachine(chunks=2)]
    first = world.submit_and_run(cap=0.95)
    assert first.state is JobState.STOPPED_AT_CAP
    first_names = world.ckpt_names(first.id)

    second = world.submit_and_run()

    assert second.state is JobState.DONE
    machine = world.machines[max(world.machines)]
    assert machine.checkpoints_at_launch == []
    assert machine.computed == ["words.json", "diar_000.json", "diar_001.json"]
    assert world.ckpt_names(first.id) == first_names
    assert world.ckpt_names(second.id) == ["diar_000.json", "diar_001.json", "words.json"]


def test_missing_hf_token_fails_before_renting(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.orch._creds = FakeCreds({"vast_api_key": VAST_KEY})  # type: ignore[attr-defined]
    job = world.submit_and_run()
    assert job.state is JobState.FAILED
    assert world.vast.created == []


def test_submit_saves_a_preparing_job_with_a_safe_name(world: World) -> None:
    job = world.orch.submit("C:/audio/talk.mp3", is_youtube=False, name='My: "Talk"?')
    assert job.state is JobState.PREPARING
    assert job.name == "My Talk"
    assert world.store.get(job.id) == job


def test_build_onstart_starts_the_watchdog_for_the_job_dir() -> None:
    script = build_onstart(SETTINGS)
    assert "watchdog.py" in script
    assert f"--job-dir {REMOTE_DIR}" in script
    assert "PYTHONPATH=/app:/app/remote" in script
