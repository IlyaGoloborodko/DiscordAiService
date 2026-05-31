from pydantic import BaseModel

class PromptInfo(BaseModel):
    user_name: str | None = None
    user_message: str


