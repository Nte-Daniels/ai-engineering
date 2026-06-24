import asyncio

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/ask")
async def ask(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        response = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": "You are a senior data engineer. Be concise."},
        {"role": "user", "content": question}
         ]
        )
        
        answer = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        print(f"Tokens — input: {input_tokens}, output: {output_tokens}")
        
        return {
            "answer": answer,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))