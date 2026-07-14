"""FastAPI application: the public face of the edge router.

Exposes one streaming endpoint. The caller POSTs financial text; the router
decides whether the local NIM node or AWS Bedrock serves it, and the tokens
stream back as Server-Sent Events regardless of which lane won.
"""

import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.clients.nim_client import check_local_node_health
from src.config import Settings, get_settings
from src.router.fallback import route_and_stream

load_dotenv()   # push .env into the process environment so boto3 can see it

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="Desoky Capital — Hybrid Edge Inference Router",
    version="1.0.0",
)


class FinancialPayload(BaseModel):
    """The request body. Only `text` is required."""

    text: str = Field(min_length=1, description="Financial text to analyze.")
    system_prompt: str | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


@app.get("/healthz")
async def healthz() -> dict:
    """Our own health, plus a live probe of the local edge node."""
    settings: Settings = get_settings()
    local_ready = await check_local_node_health(settings)
    return {
        "status": "ok",
        "local_nim_ready": local_ready,
        "active_route": "LOCAL_NIM" if local_ready else "AWS_BEDROCK",
    }


@app.post("/api/v1/process-financials")
async def process_financials(payload: FinancialPayload) -> StreamingResponse:
    """Route the request to the fastest healthy lane and stream the answer."""
    settings: Settings = get_settings()

    max_tokens = payload.max_tokens or settings.default_max_tokens
    temperature = (
        payload.temperature
        if payload.temperature is not None
        else settings.default_temperature
    )

    generator = route_and_stream(
        settings,
        text=payload.text,
        system_prompt=payload.system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # don't let a proxy buffer the stream
        },
    )