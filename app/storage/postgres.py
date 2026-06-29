import asyncio
import os

import asyncpg

_pool: asyncpg.Pool | None = None
_lock = asyncio.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_key TEXT PRIMARY KEY,
    messages    JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def get_pool() -> asyncpg.Pool:
    """Lazily create a process-wide asyncpg pool and ensure the schema exists.
    Double-checked under a lock so concurrent first requests create it once."""
    global _pool
    if _pool is None:
        async with _lock:
            if _pool is None:
                pool = await asyncpg.create_pool(
                    dsn=os.getenv("POSTGRES_DSN"),
                    min_size=1,
                    max_size=5,
                )
                async with pool.acquire() as conn:
                    await conn.execute(_SCHEMA)
                _pool = pool
    return _pool
