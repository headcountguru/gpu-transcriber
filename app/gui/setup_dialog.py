"""One-time setup: the Vast.ai API key and the Hugging Face token, each checked before saving.

Nothing is saved unless both checks pass. A key is never shown or logged: the fields are
password fields and status lines never quote any part of a key.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.credentials import HF_TOKEN, VAST_API_KEY, CredentialError, CredentialStore
from app.gui import theme
from app.gui.worker import TaskWorker, plain_error
from app.hf_check import GATED_MODELS, HF_SITE, HfAccess

logger = logging.getLogger(__name__)

CheckVast = Callable[[str], float]
CheckHf = Callable[[str], HfAccess]

_MODEL_LINKS = " and ".join(
    f'<a href="{HF_SITE}/{model}">{HF_SITE}/{model}</a>' for model in GATED_MODELS
)
INSTRUCTIONS_HTML = f"""
<p><b>Vast.ai</b> rents the GPU machine for each job.</p>
<ol>
<li>First turn on two-factor sign-in for your Vast.ai account. Vast.ai only lets an API key
rent machines once two-factor sign-in is on, so create the key <i>after</i> turning it on.</li>
<li>Create an API key at <a href="https://cloud.vast.ai/">cloud.vast.ai</a> with these
permissions: Instance (read and write), User (read), Miscellaneous (read and write).
Leave Billing off.</li>
<li>Add prepaid credit to the account. The app reads your balance from that credit.</li>
</ol>
<p><b>Hugging Face</b> provides the speaker-labeling models. Use a token from the Hugging Face
account that accepted the terms on both model pages: {_MODEL_LINKS}.</p>
<p>Both are kept in Windows Credential Manager on this PC.</p>
"""

MSG_NEED_VAST = "Paste your Vast.ai API key."
MSG_NEED_HF = "Paste your Hugging Face token."
MSG_CHECKING = "Checking…"
MSG_HF_REFUSED = "Hugging Face did not accept this token. Check that you copied all of it."


@dataclass(frozen=True)
class CheckOutcome:
    """Result of checking both secrets; an error string is plain words and secret-free."""

    balance: float | None
    vast_error: str | None
    hf_access: HfAccess | None
    hf_error: str | None


def _attempt(check: Callable[[str], object], secret: str, what: str) -> tuple[object, str | None]:
    try:
        return check(secret), None
    except Exception as exc:
        logger.warning("%s check failed (%s)", what, type(exc).__name__)
        return None, plain_error(exc)


def run_checks(
    vast_key: str, hf_token: str, check_vast: CheckVast, check_hf: CheckHf
) -> CheckOutcome:
    """Check both secrets (blocking; run on a worker thread)."""
    balance, vast_error = _attempt(check_vast, vast_key, "Vast.ai key")
    access, hf_error = _attempt(check_hf, hf_token, "Hugging Face token")
    return CheckOutcome(
        balance=float(balance) if isinstance(balance, int | float) else None,
        vast_error=vast_error,
        hf_access=access if isinstance(access, HfAccess) else None,
        hf_error=hf_error,
    )


def _hf_message(access: HfAccess) -> tuple[bool, str]:
    if not access.valid:
        return False, MSG_HF_REFUSED
    account = f" (account {access.user})" if access.user else ""
    missing = [model for model, ok in access.models.items() if not ok]
    if missing:
        names = ", ".join(missing)
        return False, (
            f"This token works{account}, but the terms are not accepted on: {names}. Open each "
            "page while signed in to that account, accept the terms, then try again."
        )
    return True, f"This token works{account}; both models are accepted."


def _credit_text(balance: float) -> str:
    text = f"Credit on the account: ${balance:.2f}."
    if balance <= 0:
        text += " Add prepaid credit at cloud.vast.ai before starting a job."
    return text


class SetupDialog(QDialog):
    """Asks for both secrets, checks them on a worker thread, and saves only if both pass."""

    def __init__(
        self,
        creds: CredentialStore,
        check_vast: CheckVast,
        check_hf: CheckHf,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._creds = creds
        self._check_vast = check_vast
        self._check_hf = check_hf
        self._worker: TaskWorker | None = None
        self.balance: float | None = None
        self._secrets = ("", "")
        self._saved = False
        self.setWindowTitle("Connect your Vast.ai and Hugging Face accounts")
        self.setMinimumWidth(620)
        self._build()

    def _build(self) -> None:
        self.instructions_label = QLabel(INSTRUCTIONS_HTML)
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setTextFormat(Qt.TextFormat.RichText)
        self.instructions_label.setOpenExternalLinks(True)

        self.vast_edit = self._secret_field("Vast.ai API key")
        self.hf_edit = self._secret_field("Hugging Face token")
        self.vast_status = self._status_label()
        self.hf_status = self._status_label()
        self.form_status = self._status_label()
        form = QFormLayout()
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(12)
        form.addRow("Vast.ai API key", self.vast_edit)
        form.addRow("", self.vast_status)
        form.addRow("Hugging Face token", self.hf_edit)
        form.addRow("", self.hf_status)

        self.save_button = QPushButton("Check and save")
        self.save_button.setObjectName("primary")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._on_save)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(self.instructions_label)
        layout.addLayout(form)
        layout.addWidget(self.form_status)
        layout.addLayout(buttons)

    @staticmethod
    def _secret_field(name: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setEchoMode(QLineEdit.EchoMode.Password)
        edit.setAccessibleName(name)
        edit.setPlaceholderText(f"Paste your {name}")
        return edit

    @staticmethod
    def _status_label() -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)  # error text comes from servers
        label.hide()
        return label

    @staticmethod
    def _set_status(label: QLabel, text: str, kind: str) -> None:
        label.setText(text)
        label.setVisible(bool(text))
        theme.set_kind(label, kind)

    def instructions_text(self) -> str:
        return self.instructions_label.text()

    def is_checking(self) -> bool:
        return self._worker is not None

    def reject(self) -> None:
        if self.is_checking():
            return  # wait for the check to finish; it is bounded by request timeouts
        super().reject()

    # --- checking and saving --------------------------------------------------------------------

    def _on_save(self) -> None:
        if self.is_checking():
            return
        vast_key = self.vast_edit.text().strip()
        hf_token = self.hf_edit.text().strip()
        self._set_status(self.vast_status, "" if vast_key else MSG_NEED_VAST, theme.KIND_FAILED)
        self._set_status(self.hf_status, "" if hf_token else MSG_NEED_HF, theme.KIND_FAILED)
        self._set_status(self.form_status, "", "")
        if not vast_key or not hf_token:
            return
        self._set_status(self.vast_status, MSG_CHECKING, theme.KIND_MUTED)
        self._set_status(self.hf_status, MSG_CHECKING, theme.KIND_MUTED)
        self._set_busy(True)
        self._secrets = (vast_key, hf_token)
        check_vast, check_hf = self._check_vast, self._check_hf
        worker = TaskWorker(
            lambda: run_checks(vast_key, hf_token, check_vast, check_hf), "Key check", self
        )
        worker.succeeded.connect(self._on_checked)
        worker.failed.connect(self._on_check_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._worker = None
        self._set_busy(False)
        if self._saved:
            self.accept()  # only once the thread is over, so closing cannot outlive it

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.save_button, self.vast_edit, self.hf_edit):
            widget.setEnabled(not busy)

    def _on_check_failed(self, message: str) -> None:
        self._set_status(self.form_status, message, theme.KIND_FAILED)

    def _credit_kind(self, balance: float) -> str:
        return theme.KIND_CAP if balance <= 0 else theme.KIND_DONE

    def _vast_result(self, outcome: CheckOutcome) -> bool:
        if outcome.balance is None:
            self._set_status(self.vast_status, outcome.vast_error or "", theme.KIND_FAILED)
            return False
        text = f"This key works. {_credit_text(outcome.balance)}"
        self._set_status(self.vast_status, text, self._credit_kind(outcome.balance))
        return True

    def _hf_result(self, outcome: CheckOutcome) -> bool:
        if outcome.hf_access is None:
            self._set_status(self.hf_status, outcome.hf_error or "", theme.KIND_FAILED)
            return False
        ok, text = _hf_message(outcome.hf_access)
        self._set_status(self.hf_status, text, theme.KIND_DONE if ok else theme.KIND_FAILED)
        return ok

    def _show_saved(self, balance: float, access: HfAccess) -> None:
        """No piece of a key is ever echoed: only "Saved", the credit and the account name."""
        self._set_status(
            self.vast_status, f"Saved. {_credit_text(balance)}", self._credit_kind(balance)
        )
        account = f" Hugging Face account: {access.user}." if access.user else ""
        self._set_status(self.hf_status, f"Saved.{account}", theme.KIND_DONE)

    def _on_checked(self, outcome: object) -> None:
        if not isinstance(outcome, CheckOutcome):
            return
        vast_ok = self._vast_result(outcome)
        hf_ok = self._hf_result(outcome)
        if not (vast_ok and hf_ok) or outcome.balance is None or outcome.hf_access is None:
            return
        vast_key, hf_token = self._secrets
        try:
            self._creds.set_all({VAST_API_KEY: vast_key, HF_TOKEN: hf_token})  # both or neither
        except CredentialError as exc:
            self._set_status(self.form_status, str(exc), theme.KIND_FAILED)
            return
        logger.info("Vast.ai key and Hugging Face token checked and saved")
        self._show_saved(outcome.balance, outcome.hf_access)
        self.balance = outcome.balance
        self._saved = True
