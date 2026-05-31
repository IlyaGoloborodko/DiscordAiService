import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

from app.api import prompt_router

load_dotenv()
app = FastAPI()

app.include_router(prompt_router)



if __name__ == "__main__":

    uvicorn.run("main:app", host='127.0.0.1', port=8066, reload=True, workers=3)