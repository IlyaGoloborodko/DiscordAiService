from typing import AsyncGenerator
import asyncio

from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeVoice


HOST = "127.0.0.1"
PORT = 4215


async def text_to_speech(text: str) -> AsyncGenerator[bytes, None]:
    """Пока тестирование на локальном tts"""
    client = AsyncTcpClient(HOST, PORT)
    await client.connect()

    await client.write_event(
        Synthesize(
            text=text,
            voice=SynthesizeVoice(name="ru_RU-dmitri-medium"),
        ).event()
    )

    stream_started = False

    try:
        while True:
            event = await asyncio.wait_for(client.read_event(), timeout=15)
            if event is None:
                break

            if event.type == "audio-start":
                stream_started = True
                continue
            elif event.type == "audio-chunk":
                if stream_started:
                    raw = event.payload or b""
                    if raw:
                        yield raw

            elif event.type == "audio-stop":
                break

    finally:
        await client.disconnect()