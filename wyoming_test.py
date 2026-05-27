from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeVoice
import asyncio


async def main():
    client = AsyncTcpClient("127.0.0.1", 4215)
    await client.connect()

    synth = Synthesize(
        text="Привет! Проверка Piper",
        voice=SynthesizeVoice(name="ru_RU-dmitri-medium"),
    )

    await client.write_event(synth.event())

    audio = bytearray()

    while True:
        event = await client.read_event()

        if event is None:
            break

        if event.type == "audio":
            audio.extend(event.data.get("audio", b""))

        if event.type == "done":
            break

    await client.disconnect()

    print(len(audio))


#asyncio.run(main())