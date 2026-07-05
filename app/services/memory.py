import logging

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)

from app.storage import get_pool, get_redis

logger = logging.getLogger(__name__)


class MemoryStore:
    """Per-session conversation memory for the agent.

    Stores the serialized pydantic-ai message history as a single JSON blob:
    Postgres is the durable source of truth, Redis is a hot cache with a sliding
    TTL. Both are best-effort — if either is down the agent still answers, just
    without memory."""

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
        if raw is None:
            raw = await self._load_pg(session_key)
            if raw is not None:
                await self._cache_redis(session_key, raw)
        if not raw:
            return []
        try:
            return list(ModelMessagesTypeAdapter.validate_json(raw))
        except Exception:
            logger.exception("failed to decode message history for %s", session_key)
            return []

    async def save(self, session_key: str, messages: list[ModelMessage]) -> None:
        trimmed = self._trim(messages)
        raw = ModelMessagesTypeAdapter.dump_json(trimmed)
        await self._cache_redis(session_key, raw)
        await self._save_pg(session_key, raw)

    async def clear(self, session_key: str) -> None:
        """Forget a session's history in both stores (best-effort)."""
        try:
            await get_redis().delete(self._rkey(session_key))
        except Exception:
            logger.warning("redis clear failed for %s", session_key, exc_info=True)
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM agent_sessions WHERE session_key = $1", session_key)
        except Exception:
            logger.warning("postgres clear failed for %s", session_key, exc_info=True)

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

    async def _load_pg(self, session_key: str) -> bytes | None:
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchval(
                    "SELECT messages FROM agent_sessions WHERE session_key = $1",
                    session_key,
                )
            return row.encode() if row is not None else None
        except Exception:
            logger.warning("postgres load failed for %s", session_key, exc_info=True)
            return None

    async def _save_pg(self, session_key: str, raw: bytes) -> None:
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_sessions (session_key, messages, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (session_key)
                    DO UPDATE SET messages = $2::jsonb, updated_at = now()
                    """,
                    session_key,
                    raw.decode(),
                )
        except Exception:
            logger.warning("postgres save failed for %s", session_key, exc_info=True)
