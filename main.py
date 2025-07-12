import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_with_gpt(request: MessageRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": request.message}]
        )
        print(response)
        reply = response.choices[0].message.content.strip()
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

