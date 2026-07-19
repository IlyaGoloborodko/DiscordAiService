"""Tests for the logging setup and the Telegram handler.

Run: .venv/Scripts/python.exe -m unittest test_logging

Nothing here talks to Telegram. The background sender thread is replaced, so we
can look at what *would* have been sent.
"""

import logging
import os
import unittest
from unittest import mock

from app import logging_setup
from app.logging_setup import TelegramHandler, setup_logging


def _record(level: int = logging.ERROR, name: str = "app.services.agent_service", message: str = "boom"):
    return logging.LogRecord(name, level, "file.py", 1, message, args=(), exc_info=None)


def _handler() -> TelegramHandler:
    """A handler whose background sender never starts, so the queue stays put."""
    with mock.patch.object(logging_setup.threading, "Thread"):
        handler = TelegramHandler("token", "chat")
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


class LevelReadingTests(unittest.TestCase):
    def test_reads_a_level_by_name(self):
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "warning"}):
            self.assertEqual(logging_setup._level("LOG_LEVEL", logging.INFO), logging.WARNING)

    def test_missing_value_falls_back(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(logging_setup._level("LOG_LEVEL", logging.INFO), logging.INFO)

    def test_typo_falls_back_instead_of_silencing_everything(self):
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "WARNINGG"}):
            self.assertEqual(logging_setup._level("LOG_LEVEL", logging.INFO), logging.INFO)


class TelegramHandlerTests(unittest.TestCase):
    def test_queues_a_message(self):
        handler = _handler()
        handler.emit(_record(message="search-service died"))
        self.assertIn("search-service died", handler._outbox.get_nowait())

    def test_message_says_the_level_and_where_it_came_from(self):
        handler = _handler()
        handler.emit(_record())
        text = handler._outbox.get_nowait()
        self.assertIn("ERROR", text)
        self.assertIn("app.services.agent_service", text)

    def test_http_library_lines_are_never_forwarded(self):
        # Sending to Telegram uses httpx, which logs, which would send again...
        handler = _handler()
        handler.emit(_record(name="httpx"))
        handler.emit(_record(name="httpcore.connection"))
        handler.emit(_record(name=logging_setup.__name__))
        self.assertTrue(handler._outbox.empty())

    def test_a_huge_message_is_truncated(self):
        handler = _handler()
        handler.emit(_record(message="x" * 99_999))
        self.assertLess(len(handler._outbox.get_nowait()), logging_setup._MAX_MESSAGE_CHARS + 100)

    def test_a_flood_is_dropped_rather_than_piling_up(self):
        handler = _handler()
        for _ in range(logging_setup._QUEUE_LIMIT + 50):
            handler.emit(_record())  # must not raise
        self.assertEqual(handler._outbox.qsize(), logging_setup._QUEUE_LIMIT)

    def test_telegram_being_down_is_not_our_problem(self):
        handler = _handler()
        client = mock.Mock()
        client.post.side_effect = OSError("no network")
        handler._send_one(client, "hello")  # must not raise


class SenderThreadTests(unittest.TestCase):
    """The real background thread, with Telegram itself replaced by a mock."""

    def test_a_logged_error_reaches_the_telegram_api(self):
        client = mock.MagicMock()
        client.__enter__.return_value = client
        with mock.patch.object(logging_setup.httpx, "Client", return_value=client), mock.patch.object(
            logging_setup, "_SECONDS_BETWEEN_SENDS", 0
        ):
            handler = TelegramHandler("secret-token", "12345")
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler.emit(_record(message="postgres is gone"))
            handler.close()  # waits for the thread to drain

        url, = client.post.call_args.args
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(url, "https://api.telegram.org/botsecret-token/sendMessage")
        self.assertEqual(payload["chat_id"], "12345")
        self.assertIn("postgres is gone", payload["text"])


class SetupTests(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        self.addCleanup(root.setLevel, root.level)
        self.addCleanup(setattr, root, "handlers", list(root.handlers))

    def test_no_telegram_handler_without_a_token(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            setup_logging()
        self.assertEqual([type(h) for h in logging.getLogger().handlers], [logging.StreamHandler])

    def test_telegram_handler_is_added_when_configured(self):
        with mock.patch.object(logging_setup.threading, "Thread"), mock.patch.dict(
            os.environ, {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1", "TELEGRAM_LOG_LEVEL": "WARNING"}
        ):
            setup_logging()
        telegram = [h for h in logging.getLogger().handlers if isinstance(h, TelegramHandler)]
        self.assertEqual(len(telegram), 1)
        self.assertEqual(telegram[0].level, logging.WARNING)

    def test_calling_twice_does_not_double_the_handlers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            setup_logging()
            setup_logging()
        self.assertEqual(len(logging.getLogger().handlers), 1)

    def test_root_is_verbose_enough_for_the_quieter_handler(self):
        # Console at WARNING must not swallow the DEBUG lines Telegram asked for.
        with mock.patch.object(logging_setup.threading, "Thread"), mock.patch.dict(
            os.environ,
            {
                "LOG_LEVEL": "WARNING",
                "TELEGRAM_BOT_TOKEN": "t",
                "TELEGRAM_CHAT_ID": "1",
                "TELEGRAM_LOG_LEVEL": "DEBUG",
            },
        ):
            setup_logging()
        self.assertEqual(logging.getLogger().level, logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
