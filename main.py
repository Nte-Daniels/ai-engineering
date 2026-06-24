import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/ask")
@limiter.limit("5/minute")
async def ask(request: Request, question: str):
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