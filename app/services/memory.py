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
    MAX_MESSAGES = 40

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

    # --- trimming ------------------------------------------------------------

    def _trim(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Bound history growth. Keep the last MAX_MESSAGES, then start at a user
        prompt so we never hand the model a dangling tool-return."""
        if len(messages) <= self.MAX_MESSAGES:
            return messages
        tail = messages[-self.MAX_MESSAGES:]
        for idx, msg in enumerate(tail):
            if isinstance(msg, ModelRequest) and any(
                isinstance(part, UserPromptPart) for part in msg.parts
            ):
                return tail[idx:]
        return tail

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
