import os

import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

from app.api import prompt_router, tts_router, agent_router, playback_router
from app.logging_setup import setup_logging

load_dotenv()
setup_logging()  # after load_dotenv: the levels and the Telegram token come from .env
app = FastAPI()

app.include_router(prompt_router)
app.include_router(tts_router)
app.include_router(agent_router)
app.include_router(playback_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Deliberately checks nothing but the process itself —
    Redis, Postgres and the search service are all best-effort, so the service
    is still useful (and must stay up) when one of them is down."""
    return {"status": "ok"}


if __name__ == "__main__":
    # Only for running the service by hand. In Docker, uvicorn is called
    # directly from the Dockerfile, so nothing here applies there.
    #
    # APP_HOST defaults to localhost on purpose: a dev machine should not put
    # this on the network. The container overrides it with 0.0.0.0, because
    # binding to localhost inside a container means "nobody can reach me" —
    # the bot would get connection refused while these logs look perfectly fine.
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_RELOAD", "true").lower() == "true",
        workers=int(os.getenv("APP_WORKERS", "3")),
        # log_config=None keeps uvicorn from installing its own handlers, so its
        # errors travel up to ours and reach Telegram like everything else.
        log_config=None,
    )
