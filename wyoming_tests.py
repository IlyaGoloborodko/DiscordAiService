import asyncio
import sounddevice as sd

from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeVoice


HOST = "127.0.0.1"
PORT = 4215


async def text_to_speech(text: str):
    client = AsyncTcpClient(HOST, PORT)
    await client.connect()

    await client.write_event(
        Synthesize(
            text=text,
            voice=SynthesizeVoice(name="ru_RU-dmitri-medium"),
        ).event()
    )

    stream = None

    try:
        while True:
            event = await asyncio.wait_for(client.read_event(), timeout=15)
            if event is None:
                break

            if event.type == "audio-start":
                rate = event.data["rate"]
                width = event.data["width"]
                channels = event.data["channels"]

                if width != 2:
                    raise RuntimeError(f"Expected 16-bit PCM (width=2), got width={width}")

                stream = sd.RawOutputStream(
                    samplerate=rate,
                    channels=channels,
                    dtype="int16",
                )
                stream.start()

            elif event.type == "audio-chunk":
                raw = event.payload or b""
                if raw and stream is not None:
                    stream.write(raw)

            elif event.type == "audio-stop":
                break

    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        await client.disconnect()