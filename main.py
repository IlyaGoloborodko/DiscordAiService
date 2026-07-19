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


if __name__ == "__main__":

    # log_config=None keeps uvicorn from installing its own handlers, so its
    # errors travel up to ours and reach Telegram like everything else.
    uvicorn.run("main:app", host='127.0.0.1', port=8066, reload=True, workers=3, log_config=None)