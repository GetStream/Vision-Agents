#!/usr/bin/env python3
"""
Nemotron ASR Server

Standalone FastAPI server that wraps NVIDIA Nemotron Speech model.
Run this in a separate environment with NeMo installed.

Usage:
    pip install nemo_toolkit[asr] fastapi uvicorn
    python nemotron_server.py

The server exposes:
    POST /transcribe - Transcribe audio bytes
    GET /health - Health check
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal, cast

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
executor = ThreadPoolExecutor(max_workers=2)


class TranscribeRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000


class TranscribeResponse(BaseModel):
    text: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model(
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
    device: Literal["cpu", "cuda"] = "cpu",
):
    import nemo.collections.asr as nemo_asr

    logger.info(f"Loading Nemotron model: {model_name}")
    m = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    if device == "cuda":
        m = m.cuda()
    logger.info("Model loaded")
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    import os

    device_str = os.environ.get("NEMOTRON_DEVICE", "cpu")
    device: Literal["cpu", "cuda"] = cast(
        Literal["cpu", "cuda"],
        device_str if device_str in ("cpu", "cuda") else "cpu",
    )
    model_name = os.environ.get(
        "NEMOTRON_MODEL", "nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model = load_model(model_name, device)
    yield
    executor.shutdown(wait=False)


app = FastAPI(title="Nemotron ASR Server", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import base64

    audio_bytes = base64.b64decode(request.audio_base64)
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

    if audio_array.size == 0:
        return TranscribeResponse(text="", processing_time_ms=0)

    start_time = time.time()

    def _transcribe():
        results = model.transcribe([audio_array])
        if isinstance(results, tuple):
            results = results[0]
        return results

    loop = asyncio.get_running_loop()
    transcripts = await loop.run_in_executor(executor, _transcribe)

    processing_time_ms = (time.time() - start_time) * 1000

    if isinstance(transcripts, list) and transcripts:
        text = " ".join(str(item) for item in transcripts).strip()
    elif isinstance(transcripts, str):
        text = transcripts.strip()
    else:
        text = str(transcripts).strip() if transcripts else ""

    return TranscribeResponse(text=text, processing_time_ms=processing_time_ms)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
