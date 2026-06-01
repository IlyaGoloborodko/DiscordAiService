from pydantic import BaseModel, Field


class MusicAgentOutput(BaseModel):
    full_answer_for_tts: str = Field(
        description="Full response text for TTS. Should be natural, concise, conversational,"
                    " and free of JSON and unnecessary characters. This part must be on Russian"
    )
    search_string_for_music: str = Field(
        description="Short search query for YouTube (2-6 words), artist and track/genre only. Recommend specific compositions"
    )
