"""Qt tests for the one-time setup dialog. Fake checks and an in-memory keyring only."""

import logging
import threading

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit

from app.credentials import HF_TOKEN, VAST_API_KEY, CredentialStore
from app.gui.setup_dialog import SetupDialog
from app.hf_check import DIARIZATION_MODEL, SEGMENTATION_MODEL, HfAccess, HfCheckError
from app.vast_client import VastError
from tests.gui_fakes import FakeBackend

WAIT_MS = 5000
VAST_KEY = "QZXJ9W7KQV3PZJXQ8VKW"
HF = "hf_ZQJX8V6WKQPZ4JXQ"
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
    """Every text the dialog can show: labels, title, and what the input fields display."""
    labels = [label.text() for label in dialog.findChildren(QLabel)]
    fields = [edit.displayText() for edit in dialog.findChildren(QLineEdit)]
    return " ".join([*labels, *fields, dialog.windowTitle()])


def assert_no_key_fragment(dialog: SetupDialog) -> None:
    """No piece of either key (any 3 characters in a row) appears anywhere in the dialog.

    The keys use runs like ``QZX`` that occur in no English text, so any match is a leak.
    """
    text = visible_text(dialog)
    for secret in (VAST_KEY, HF):
        for start in range(len(secret) - 2):
            assert secret[start : start + 3] not in text


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
    assert dialog.vast_status.text().startswith("Saved")
    assert "$12.50" in dialog.vast_status.text()
    assert dialog.hf_status.text().startswith("Saved")
    assert "tom" in dialog.hf_status.text()
    assert_no_key_fragment(dialog)
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
    assert not status.startswith("Saved")
    assert not dialog.vast_status.text().startswith("Saved")  # nothing was saved
    assert_no_key_fragment(dialog)


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
    assert_no_key_fragment(dialog)
    assert "went wrong" in dialog.vast_status.text().lower()


def test_status_and_error_text_is_plain_text(qtbot: object) -> None:
    def check_vast(key: str) -> float:
        raise VastError('<a href="https://evil">Vast.ai said no</a>')

    dialog = build(qtbot, FakeBackend(), check_vast, lambda t: granted())
    submit(qtbot, dialog)
    for label in (dialog.vast_status, dialog.hf_status, dialog.form_status):
        assert label.textFormat() == Qt.TextFormat.PlainText
    assert dialog.vast_status.text() == '<a href="https://evil">Vast.ai said no</a>'


def test_empty_fields_ask_for_both_values(qtbot: object) -> None:
    backend = FakeBackend()
    dialog = build(qtbot, backend, lambda k: 1.0, lambda t: granted())
    dialog.vast_edit.setText("  ")
    dialog.save_button.click()
    assert not dialog.is_checking()
    assert backend.data == {}
    assert "Paste" in dialog.vast_status.text()


class SecondWriteFails(FakeBackend):
    """Saving the Hugging Face token fails; everything else works."""

    def set_password(self, service: str, name: str, value: str) -> None:
        if name == HF_TOKEN:
            raise OSError("Credential Manager is locked")
        super().set_password(service, name, value)


@pytest.mark.parametrize("had_old_key", [True, False], ids=["old-key-restored", "no-old-key"])
def test_failed_second_save_leaves_the_first_key_as_it_was(
    qtbot: object, caplog: pytest.LogCaptureFixture, had_old_key: bool
) -> None:
    backend = SecondWriteFails()
    if had_old_key:
        backend.data[(SERVICE, VAST_API_KEY)] = "OLDVASTKEY1234567890"
    dialog = build(qtbot, backend, check_vast=lambda key: 5.0, check_hf=lambda token: granted())

    with caplog.at_level(logging.DEBUG):
        submit(qtbot, dialog)

    expected = {(SERVICE, VAST_API_KEY): "OLDVASTKEY1234567890"} if had_old_key else {}
    assert backend.data == expected  # both or neither
    assert dialog.result() != SetupDialog.DialogCode.Accepted
    shown = dialog.form_status.text()
    assert shown and VAST_KEY not in shown and HF not in shown
    assert VAST_KEY not in caplog.text and HF not in caplog.text
    assert "OLDVASTKEY1234567890" not in shown
