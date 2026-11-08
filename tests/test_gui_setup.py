"""Qt tests for the one-time setup dialog. Fake checks and an in-memory keyring only."""

import logging
import threading

import pytest

from app.credentials import HF_TOKEN, VAST_API_KEY, CredentialStore
from app.gui.setup_dialog import SetupDialog
from app.hf_check import DIARIZATION_MODEL, SEGMENTATION_MODEL, HfAccess, HfCheckError
from app.vast_client import VastError
from tests.gui_fakes import FakeBackend

WAIT_MS = 5000
VAST_KEY = "vastkey_0123456789abcdefWXYZ"
HF = "hf_secret_token_0123456789QRST"
SERVICE = "GpuTranscriber"


def granted(user: str = "tom") -> HfAccess:
    return HfAccess(
        valid=True, user=user, models={DIARIZATION_MODEL: True, SEGMENTATION_MODEL: True}
    )


def build(qtbot: object, backend: FakeBackend, check_vast: object, check_hf: object) -> SetupDialog:
    dialog = SetupDialog(
        CredentialStore(backend=backend), check_vast=check_vast, check_hf=check_hf  # type: ignore[arg-type]
    )
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.vast_edit.setText(VAST_KEY)
    dialog.hf_edit.setText(HF)
    return dialog


def submit(qtbot: object, dialog: SetupDialog) -> None:
    dialog.save_button.click()
    qtbot.waitUntil(lambda: not dialog.is_checking(), timeout=WAIT_MS)  # type: ignore[attr-defined]


def visible_text(dialog: SetupDialog) -> str:
    return " ".join([dialog.vast_status.text(), dialog.hf_status.text(), dialog.windowTitle()])


def test_instructions_explain_two_factor_permissions_credit_and_model_terms(qtbot: object) -> None:
    dialog = SetupDialog(
        CredentialStore(backend=FakeBackend()),
        check_vast=lambda k: 1.0,
        check_hf=lambda t: granted(),
    )
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    text = dialog.instructions_text()
    assert "two-factor" in text
    assert "Instance (read and write)" in text
    assert "User (read)" in text
    assert "Miscellaneous (read and write)" in text
    assert "Billing" in text and "off" in text.lower()
    assert "prepaid credit" in text.lower()
    assert "https://huggingface.co/pyannote/speaker-diarization-3.1" in text
    assert "https://huggingface.co/pyannote/segmentation-3.0" in text


def test_fields_are_masked(qtbot: object) -> None:
    dialog = build(qtbot, FakeBackend(), lambda k: 1.0, lambda t: granted())
    assert dialog.vast_edit.echoMode() == dialog.vast_edit.EchoMode.Password
    assert dialog.hf_edit.echoMode() == dialog.hf_edit.EchoMode.Password


def test_saves_only_after_both_checks_pass_and_masks_keys(
    qtbot: object, caplog: pytest.LogCaptureFixture
) -> None:
    backend = FakeBackend()
    threads: list[int] = []

    def check_vast(key: str) -> float:
        threads.append(threading.get_ident())
        return 12.5

    dialog = build(qtbot, backend, check_vast, lambda t: granted())
    with caplog.at_level(logging.DEBUG):
        submit(qtbot, dialog)
    assert backend.data == {(SERVICE, VAST_API_KEY): VAST_KEY, (SERVICE, HF_TOKEN): HF}
    assert dialog.result() == SetupDialog.DialogCode.Accepted
    assert dialog.balance == pytest.approx(12.5)
    assert "$12.50" in dialog.vast_status.text()
    assert "WXYZ" in dialog.vast_status.text()
    assert VAST_KEY not in visible_text(dialog)
    assert HF not in visible_text(dialog)
    assert VAST_KEY not in caplog.text and HF not in caplog.text
    assert threads and threads[0] != threading.get_ident()


def test_vast_error_saves_nothing(qtbot: object) -> None:
    backend = FakeBackend()

    def check_vast(key: str) -> float:
        raise VastError("Vast.ai did not accept the API key.", status_code=401)

    dialog = build(qtbot, backend, check_vast, lambda t: granted())
    submit(qtbot, dialog)
    assert backend.data == {}
    assert "did not accept the API key" in dialog.vast_status.text()
    assert dialog.result() != SetupDialog.DialogCode.Accepted


def test_models_not_accepted_are_named_and_nothing_saved(qtbot: object) -> None:
    backend = FakeBackend()
    partial = HfAccess(
        valid=True, user="tom", models={DIARIZATION_MODEL: True, SEGMENTATION_MODEL: False}
    )
    dialog = build(qtbot, backend, lambda k: 3.0, lambda t: partial)
    submit(qtbot, dialog)
    assert backend.data == {}
    status = dialog.hf_status.text()
    assert SEGMENTATION_MODEL in status
    assert DIARIZATION_MODEL not in status.replace(SEGMENTATION_MODEL, "")


def test_hf_check_error_saves_nothing(qtbot: object) -> None:
    backend = FakeBackend()

    def check_hf(token: str) -> HfAccess:
        raise HfCheckError("Could not reach Hugging Face. Check your internet connection.")

    dialog = build(qtbot, backend, lambda k: 3.0, check_hf)
    submit(qtbot, dialog)
    assert backend.data == {}
    assert "Could not reach Hugging Face" in dialog.hf_status.text()


def test_unexpected_error_in_check_is_plain_and_secret_free(qtbot: object) -> None:
    backend = FakeBackend()

    def check_vast(key: str) -> float:
        raise RuntimeError(f"weird failure with {key}")

    dialog = build(qtbot, backend, check_vast, lambda t: granted())
    submit(qtbot, dialog)
    assert backend.data == {}
    assert VAST_KEY not in visible_text(dialog)
    assert "went wrong" in dialog.vast_status.text().lower()


def test_empty_fields_ask_for_both_values(qtbot: object) -> None:
    backend = FakeBackend()
    dialog = build(qtbot, backend, lambda k: 1.0, lambda t: granted())
    dialog.vast_edit.setText("  ")
    dialog.save_button.click()
    assert not dialog.is_checking()
    assert backend.data == {}
    assert "Paste" in dialog.vast_status.text()
