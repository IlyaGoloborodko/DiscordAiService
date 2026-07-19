"""Logging setup: everything goes to the console, the important parts go to Telegram.

The code all over this project already writes plain `logger.warning(...)` /
`logger.exception(...)` lines. This module is the only place that decides where
those lines end up. Call `setup_logging()` once, when the service starts.

Two separate levels, both from `.env`:

    LOG_LEVEL=INFO           what you see in the console
    TELEGRAM_LOG_LEVEL=ERROR what gets sent to your phone

Telegram is meant for "something broke", not for a live feed — so by default it
only gets ERROR and above. Set it to WARNING if you want more.

Sending a message to Telegram takes a network round-trip. We never do that while
handling a request: log lines are dropped into a queue and a background thread
sends them. If Telegram is down, or slow, or the queue fills up, the service
keeps running and the messages are simply lost. Logging must never be the reason
something fails.
"""

import atexit
import logging
import os
import queue
import threading
import time

import httpx

logger = logging.getLogger(__name__)

# Telegram rejects anything longer than this.
_MAX_MESSAGE_CHARS = 4000

# Telegram starts refusing messages at roughly 20 per minute per chat, so we
# leave at least this long between two sends.
_SECONDS_BETWEEN_SENDS = 3.0

# If more than this many messages are waiting to be sent, we start dropping new
# ones. A burst of errors should not eat memory.
_QUEUE_LIMIT = 100

# Log lines from these libraries are never forwarded to Telegram. Without this,
# sending a message would log something, which would send a message, forever.
_NEVER_FORWARD = ("httpx", "httpcore", __name__)


def _level(name: str, default: int) -> int:
    """Read a log level like "WARNING" from the environment.

    Falls back to `default` if it is missing or misspelled — a typo in `.env`
    should not leave you with no logging at all.
    """
    raw = (os.getenv(name) or "").strip().upper()
    if not raw:
        return default
    return logging.getLevelNamesMapping().get(raw, default)


class TelegramHandler(logging.Handler):
    """Sends log records to a Telegram chat, from a background thread."""

    def __init__(self, token: str, chat_id: str):
        super().__init__()
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._chat_id = chat_id
        self._outbox: queue.Queue[str | None] = queue.Queue(maxsize=_QUEUE_LIMIT)
        self._sender = threading.Thread(target=self._send_forever, name="telegram-log", daemon=True)
        self._sender.start()

    def emit(self, record: logging.LogRecord) -> None:
        """Called by the logging machinery. Must be quick and must never raise."""
        if record.name.startswith(_NEVER_FORWARD):
            return
        try:
            self._outbox.put_nowait(self._compose(record))
        except queue.Full:
            pass  # too many errors at once; the console still has them all
        except Exception:  # noqa: BLE001 - a broken log line must not break the caller
            self.handleError(record)

    def _compose(self, record: logging.LogRecord) -> str:
        """Turn a log record into the text of one Telegram message."""
        text = f"{record.levelname} — {record.name}\n\n{self.format(record)}"
        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[:_MAX_MESSAGE_CHARS] + "\n[...truncated]"
        return text

    def _send_forever(self) -> None:
        """The background thread: take one message off the queue, send it, repeat."""
        with httpx.Client(timeout=10) as client:
            while True:
                text = self._outbox.get()
                if text is None:  # put there by close()
                    return
                self._send_one(client, text)
                time.sleep(_SECONDS_BETWEEN_SENDS)

    def _send_one(self, client: httpx.Client, text: str) -> None:
        try:
            client.post(self._url, json={"chat_id": self._chat_id, "text": text})
        except Exception:  # noqa: BLE001 - Telegram being unreachable is not our problem
            pass  # deliberately silent: logging about it here would loop

    def close(self) -> None:
        """Ask the sender thread to stop, and give it a moment to finish up."""
        try:
            self._outbox.put_nowait(None)
        except queue.Full:
            pass
        self._sender.join(timeout=5)
        super().close()


def _telegram_handler() -> TelegramHandler | None:
    """Build the Telegram handler, or None if it isn't configured.

    Telegram logging is optional on purpose: without a token and a chat id the
    service runs exactly as before, just without the phone notifications.
    """
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return None

    handler = TelegramHandler(token, chat_id)
    handler.setLevel(_level("TELEGRAM_LOG_LEVEL", logging.ERROR))
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def setup_logging() -> None:
    """Wire up console logging, and Telegram too if `.env` has the token and chat id.

    Safe to call more than once — the second call replaces the handlers rather
    than adding a second copy of each.
    """
    console_level = _level("LOG_LEVEL", logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))

    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
        old.close()

    handlers: list[logging.Handler] = [console]
    telegram = _telegram_handler()
    if telegram is not None:
        handlers.append(telegram)
        atexit.register(telegram.close)

    for handler in handlers:
        root.addHandler(handler)

    # The root level has to be the most verbose of the two, otherwise it would
    # filter records out before either handler ever sees them.
    root.setLevel(min(handler.level for handler in handlers))

    # These libraries are extremely chatty at DEBUG and drown out our own lines.
    for noisy in ("httpx", "httpcore", "openai", "asyncio"):
        logging.getLogger(noisy).setLevel(max(console_level, logging.INFO))

    logger.info(
        "logging ready: console=%s telegram=%s",
        logging.getLevelName(console_level),
        logging.getLevelName(telegram.level) if telegram else "off",
    )
