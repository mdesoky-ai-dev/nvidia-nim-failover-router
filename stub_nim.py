"""A stand-in for a real NVIDIA NIM container.

The router only ever asks NIM for two things, so that is all this serves:

  GET  /v1/health/ready      -> 200 when ready (drives check_local_node_health)
  POST /v1/chat/completions  -> OpenAI-format SSE stream (drives stream_nim)

There is no model here. This exists to exercise the LOCAL_NIM lane and the
failover path on hardware without an NVIDIA GPU. Kill this process and the
router should silently switch to AWS Bedrock on the very next request.

Run:  uvicorn stub_nim:app --port 8000
"""

import asyncio
import json
import time
from typing import AsyncIterator

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Stub NIM (protocol-compatible, no model)")

# Flip to False (or hit /admin/offline) to simulate VRAM exhaustion / node death.
READY = True

TOKEN_DELAY_S = 0.02

CANNED = (
    "Local edge node responding. Revenue grew 12 percent while operating costs "
    "grew 8 percent, so the 4-point spread expanded operating margin. This is "
    "positive operating leverage: incremental revenue is absorbing fixed cost."
)


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


@app.get("/v1/health/ready")
async def ready() -> Response:
    """The readiness probe. 503 when 'offline' so the router fails over."""
    if not READY:
        return Response(status_code=503)
    return Response(status_code=200)


@app.post("/admin/offline")
async def go_offline() -> dict:
    """Simulate the node dying without killing the process."""
    global READY
    READY = False
    return {"ready": READY}


@app.post("/admin/online")
async def go_online() -> dict:
    global READY
    READY = True
    return {"ready": READY}


def _chunk(content: str | None, finish: str | None = None) -> str:
    """One OpenAI-format SSE frame, shaped exactly as NIM would send it."""
    delta = {"content": content} if content is not None else {}
    payload = {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "stub/local-edge",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def _generate(max_tokens: int) -> AsyncIterator[str]:
    words = CANNED.split()[:max_tokens]
    for i, word in enumerate(words):
        await asyncio.sleep(TOKEN_DELAY_S)
        text = word if i == 0 else f" {word}"
        yield _chunk(text)
    yield _chunk(None, finish="stop")
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> StreamingResponse:
    """Stream a canned completion in the OpenAI-compatible SSE format."""
    max_tokens = req.max_tokens or 1024
    return StreamingResponse(
        _generate(max_tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )