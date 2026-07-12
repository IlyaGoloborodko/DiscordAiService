import json
import logging
import os
from typing import Any

import tiktoken
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    SystemPromptPart,
    UserPromptPart,
)
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.storage import SessionMemory, get_redis, get_sessionmaker

logger = logging.getLogger(__name__)

# A widely-used encoding; the exact model tokenizer isn't critical here — we only
# need a stable, reasonable estimate to budget history size.
HISTORY_ENCODING = "cl100k_base"
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(HISTORY_ENCODING)
    return _encoder


class MemoryStore:
    """Per-session conversation memory for the agent.

    Stores the serialized pydantic-ai message history. Postgres (via SQLAlchemy,
    JSONB column) is the durable source of truth; Redis is a hot cache with a
    sliding TTL. Both are best-effort — if either is down the agent still answers,
    just without memory."""

    TTL_SECONDS = 6 * 60 * 60

    def __init__(self) -> None:
        # Token budget for the INTERMEDIATE history only. The system prompt (added
        # fresh by the agent) and the latest user message are never part of this
        # and are never trimmed.
        self.token_limit = int(os.getenv("HISTORY_TOKEN_LIMIT", "20000"))

    @staticmethod
    def _rkey(session_key: str) -> str:
        return f"sess:{session_key}"

    async def load(self, session_key: str) -> list[ModelMessage]:
        raw = await self._load_redis(session_key)
        if raw is not None:
            return self._decode(raw, session_key)

        payload = await self._load_pg(session_key)
        if payload is None:
            return []
        # Re-warm the Redis cache from the durable copy.
        await self._cache_redis(session_key, json.dumps(payload).encode())
        return self._decode(payload, session_key)

    async def save(self, session_key: str, messages: list[ModelMessage]) -> None:
        # Strip system prompts: the agent injects the current one fresh each run,
        # so persisting it would only let a stale copy accumulate.
        trimmed = self._trim(self._strip_system(messages))
        raw = ModelMessagesTypeAdapter.dump_json(trimmed)
        payload = json.loads(raw)  # JSON-native structure for the JSONB column
        await self._cache_redis(session_key, raw)
        await self._save_pg(session_key, payload)

    async def clear(self, session_key: str) -> None:
        """Forget a session's history in both stores (best-effort)."""
        try:
            await get_redis().delete(self._rkey(session_key))
        except Exception:
            logger.warning("redis clear failed for %s", session_key, exc_info=True)
        try:
            async with get_sessionmaker()() as session:
                await session.execute(
                    delete(SessionMemory).where(SessionMemory.session_key == session_key)
                )
                await session.commit()
        except Exception:
            logger.warning("postgres clear failed for %s", session_key, exc_info=True)

    # --- decoding ------------------------------------------------------------

    @staticmethod
    def _decode(data: bytes | list[Any], session_key: str) -> list[ModelMessage]:
        try:
            if isinstance(data, (bytes, bytearray, str)):
                return list(ModelMessagesTypeAdapter.validate_json(data))
            return list(ModelMessagesTypeAdapter.validate_python(data))
        except Exception:
            logger.exception("failed to decode message history for %s", session_key)
            return []

    # --- trimming ------------------------------------------------------------

    @staticmethod
    def _strip_system(messages: list[ModelMessage]) -> list[ModelMessage]:
        """Drop SystemPromptParts (and any request left empty by that)."""
        out: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                parts = [p for p in msg.parts if not isinstance(p, SystemPromptPart)]
                if not parts:
                    continue
                if len(parts) != len(msg.parts):
                    msg = ModelRequest(parts=parts)
            out.append(msg)
        return out

    def _trim(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Keep the newest messages that fit within the token budget, dropping
        whole messages that don't. Then align the start to a user prompt so the
        history never begins with a dangling tool-return."""
        kept: list[ModelMessage] = []
        total = 0
        for msg in reversed(messages):
            tokens = self._count_tokens(msg)
            if kept and total + tokens > self.token_limit:
                break
            total += tokens
            kept.append(msg)
        kept.reverse()

        for idx, msg in enumerate(kept):
            if isinstance(msg, ModelRequest) and any(
                isinstance(part, UserPromptPart) for part in msg.parts
            ):
                return kept[idx:]
        return []

    @staticmethod
    def _count_tokens(message: ModelMessage) -> int:
        raw = ModelMessagesTypeAdapter.dump_json([message])
        return len(_get_encoder().encode(raw.decode("utf-8")))

    # --- redis ---------------------------------------------------------------

    async def _load_redis(self, session_key: str) -> bytes | None:
        try:
            return await get_redis().get(self._rkey(session_key))
        except Exception:
            logger.warning("redis load failed for %s", session_key, exc_info=True)
            return None

    async def _cache_redis(self, session_key: str, raw: bytes) -> None:
        try:
            await get_redis().set(self._rkey(session_key), raw, ex=self.TTL_SECONDS)
        except Exception:
            logger.warning("redis save failed for %s", session_key, exc_info=True)

    # --- postgres ------------------------------------------------------------

    async def _load_pg(self, session_key: str) -> list[Any] | None:
        try:
            async with get_sessionmaker()() as session:
                row = await session.get(SessionMemory, session_key)
            return row.messages if row is not None else None
        except Exception:
            logger.warning("postgres load failed for %s", session_key, exc_info=True)
            return None

    async def _save_pg(self, session_key: str, payload: list[Any]) -> None:
        try:
            stmt = pg_insert(SessionMemory).values(session_key=session_key, messages=payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[SessionMemory.session_key],
                set_={"messages": payload, "updated_at": func.now()},
            )
            async with get_sessionmaker()() as session:
                await session.execute(stmt)
                await session.commit()
        except Exception:
            logger.warning("postgres save failed for %s", session_key, exc_info=True)
