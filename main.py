import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from retriever import Retriever
from generator import Generator
from langchain_openai import OpenAIEmbeddings


load_dotenv()

INDEX_PATH='faiss_index'
MODEL="gpt-4o-mini"
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

retriever = Retriever(INDEX_PATH, OpenAIEmbeddings())
generator = Generator(api_key=OPENAI_API_KEY, model=MODEL)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_with_gpt(request: MessageRequest):
    try:
        docs = retriever.retrive(request.message)
        texts = [doc.page_content for doc in docs]
        answer = generator.generate(request.message, texts)
        return {"reply": answer}
    
        #response = client.chat.completions.create(
        #    model="gpt-4.1-mini",
        #    messages=[{"role": "user", "content": request.message}]
        #)
        #reply = response.choices[0].message.content.strip()
        #return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

