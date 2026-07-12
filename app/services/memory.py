import json
import logging
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.storage import SessionMemory, get_redis, get_sessionmaker

logger = logging.getLogger(__name__)


class MemoryStore:
    """Per-session conversation memory for the agent.

    Stores the serialized pydantic-ai message history. Postgres (via SQLAlchemy,
    JSONB column) is the durable source of truth; Redis is a hot cache with a
    sliding TTL. Both are best-effort — if either is down the agent still answers,
    just without memory."""

    TTL_SECONDS = 6 * 60 * 60
    # Keep only the last few conversational turns. A "turn" starts at a user
    # prompt and runs through its tool round-trips and final answer, so counting
    # turns (not raw messages) keeps history small without ever cutting a turn in
    # half and leaving a dangling tool-return. 2 turns ~= the last 4 user/assistant
    # messages. Small on purpose: long history let stale answers poison the model.
    MAX_TURNS = 2

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
        trimmed = self._trim(messages)
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

    def _trim(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Keep only the last MAX_TURNS turns. Cutting at a user-prompt boundary
        keeps each turn whole (its tool round-trips stay paired) and guarantees
        the history starts with a user prompt, never a dangling tool-return."""
        turn_starts = [
            idx
            for idx, msg in enumerate(messages)
            if isinstance(msg, ModelRequest)
            and any(isinstance(part, UserPromptPart) for part in msg.parts)
        ]
        if len(turn_starts) <= self.MAX_TURNS:
            return messages
        return messages[turn_starts[-self.MAX_TURNS]:]

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
