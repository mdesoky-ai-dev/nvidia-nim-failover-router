"""The switchboard.

Decides which lane serves a request and streams the answer back:

  1. Probe the local NIM node's health.
  2. Healthy  -> stream from local NIM (LOCAL_NIM).
     Unhealthy -> stream from AWS Bedrock (AWS_BEDROCK).
  3. Resilience: if the local lane dies BEFORE its first token, fail over to
     Bedrock silently. (Once tokens have started flowing we can't rewind, so
     a mid-stream error is surfaced instead.)
  4. For every request, record latency-to-first-token (TTFT) and the lane,
     then emit a normalized SSE stream the caller can consume identically
     regardless of which backend won.
"""

import json
import logging
import time
from typing import AsyncIterator

from src.clients.bedrock_client import stream_bedrock
from src.clients.nim_client import check_local_node_health, stream_nim
from src.config import Settings

logger = logging.getLogger("edge.router")

LOCAL = "LOCAL_NIM"
CLOUD = "AWS_BEDROCK"


def _sse(payload: dict) -> str:
    """Wrap a dict as one Server-Sent Events frame."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _make_stream(settings, destination, *, text, system_prompt, max_tokens, temperature):
    """Return the right client's token generator for the chosen lane."""
    if destination == LOCAL:
        return stream_nim(
            settings, text=text, system_prompt=system_prompt,
            max_tokens=max_tokens, temperature=temperature,
        )
    return stream_bedrock(
        settings, text=text, system_prompt=system_prompt,
        max_tokens=max_tokens, temperature=temperature,
    )


async def route_and_stream(
    settings,
    *,
    text,
    system_prompt,
    max_tokens,
    temperature,
) -> AsyncIterator[str]:
    """Pick a lane, stream it back as normalized SSE, and log TTFT."""
    local_ready = await check_local_node_health(settings)
    destination = LOCAL if local_ready else CLOUD

    state = {"t_start": time.perf_counter(), "ttft_ms": None, "tokens": 0}

    async def _emit(dest):
        gen = _make_stream(
            settings, dest, text=text, system_prompt=system_prompt,
            max_tokens=max_tokens, temperature=temperature,
        )
        async for token in gen:
            if state["ttft_ms"] is None:
               state["ttft_ms"] = (time.perf_counter() - state["t_start"]) * 1000.0
               logger.info("TTFT route=%s ttft_ms=%.1f", dest, state["ttft_ms"])
            state["tokens"] += 1
            yield _sse({"type": "token", "content": token, "source": dest})

    try:
        async for frame in _emit(destination):
            yield frame
    except Exception as exc:
        if destination == LOCAL and state["ttft_ms"] is None:
            logger.warning(
                "Local lane failed before first token (%s). Failing over to %s.",
                exc.__class__.__name__, CLOUD,
            )
            destination = CLOUD
            try:
                async for frame in _emit(destination):
                    yield frame
            except Exception as exc2:
                logger.exception("Cloud failover also failed")
                yield _sse({"type": "error", "source": CLOUD, "message": str(exc2)})
                return
        else:
            logger.exception("Stream failed on route=%s", destination)
            yield _sse({"type": "error", "source": destination, "message": str(exc)})
            return

    total_ms = (time.perf_counter() - state["t_start"]) * 1000.0
    ttft = state["ttft_ms"]
    logger.info(
        "REQUEST_COMPLETE route=%s ttft_ms=%s total_ms=%.1f tokens=%d",
        destination, f"{ttft:.1f}" if ttft is not None else "n/a", total_ms, state["tokens"],
    )
    yield _sse({
        "type": "done",
        "source": destination,
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "total_ms": round(total_ms, 1),
        "tokens": state["tokens"],
    })
    yield "data: [DONE]\n\n"