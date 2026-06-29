import os

from redis.asyncio import Redis

_redis: Redis | None = None


def get_redis() -> Redis:
    """Lazily create a process-wide Redis client. The underlying connection pool
    is created on first use, so this is safe to call per request and per worker.
    Stores raw bytes (decode_responses stays False) — we cache JSON blobs."""
    global _redis
    if _redis is None:
        _redis = Redis.from_url(
            os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            decode_responses=False,
        )
    return _redis
