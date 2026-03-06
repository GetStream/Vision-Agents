"""
Simple token server for the Cricket Umpire frontend.
Run this separately: python token_server.py
It runs on port 8001 alongside the main agent server (port 8000).
"""
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from getstream import Stream as StreamClient

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/token")
async def get_token(user_id: str):
    try:
        client = StreamClient(
            api_key=os.environ["STREAM_API_KEY"],
            api_secret=os.environ["STREAM_API_SECRET"],
        )
        token = client.create_token(user_id)
        return JSONResponse({"token": token, "user_id": user_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("🏏 Token server running on http://localhost:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)