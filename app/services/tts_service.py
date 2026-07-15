import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

# Text-to-speech via the OpenAI Audio API (gpt-4o-mini-tts). Same public contract as
# the previous local-Piper implementation — an async generator of audio-byte chunks
# feeding the /tts StreamingResponse — only the backend changed.
#
# pydantic-ai is NOT used here: it wraps chat/completions only, and speech is a
# separate endpoint (POST /v1/audio/speech), so we call the openai SDK directly.
#
# All knobs are read from env at call time (env is loaded after import), matching the
# rest of the service. The API key is a SEPARATE constant from the text model's
# (`TM_API_KEY`) even though it currently holds the same value — so the two can
# diverge later without a code change.

_DEFAULT_MODEL = "gpt-4o-mini-tts"
_DEFAULT_VOICE = "nova"
# `pcm` is raw 24 kHz, 16-bit, mono, little-endian audio (no container) — the closest
# drop-in to Piper's raw PCM stream. NOTE: Piper emitted 22.05 kHz; the player must be
# told 24 kHz or set TTS_FORMAT to a self-describing container (opus/wav/mp3).
_DEFAULT_FORMAT = "pcm"


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("TTS_API_KEY"),
        base_url=os.getenv("TTS_BASE_URL", "https://api.openai.com/v1"),
    )


async def text_to_speech(text: str) -> AsyncGenerator[bytes, None]:
    """Stream synthesized speech for `text` as audio-byte chunks."""
    params: dict = {
        "model": os.getenv("TTS_MODEL", _DEFAULT_MODEL),
        "voice": os.getenv("TTS_VOICE", _DEFAULT_VOICE),
        "input": text,
        "response_format": os.getenv("TTS_FORMAT", _DEFAULT_FORMAT),
    }
    # Steerable tone/emotion — only gpt-4o-mini-tts honours it, so send it only when set.
    instructions = os.getenv("TTS_INSTRUCTIONS")
    if instructions:
        params["instructions"] = instructions

    async with _client().audio.speech.with_streaming_response.create(**params) as response:
        async for chunk in response.iter_bytes():
            if chunk:
                yield chunk
