"""Local NVIDIA NIM client.

NIM exposes an OpenAI-compatible surface, so the chat path speaks the standard
`/v1/chat/completions` SSE protocol and the readiness path is a plain GET. Two
public callables:

  * check_local_node_health() -> bool   (the hardware guard-rail)
  * stream_nim() -> AsyncIterator[str]   (text deltas only)
"""

import json
import logging
from typing import AsyncIterator

import httpx

from src.config import Settings

logger = logging.getLogger("edge.nim")


async def check_local_node_health(settings: Settings) -> bool:
    """Fast readiness probe against the local NIM container.

    Returns True only on an HTTP 200 from the readiness endpoint inside the
    configured sub-second budget. A timeout, a connection error, or any
    non-200 status all mean the edge node cannot serve this request and we
    must engage the cloud failover.
    """
    try:
        async with httpx.AsyncClient(timeout=settings.health_timeout_s) as client:
            resp = await client.get(settings.nim_health_url)

        if resp.status_code == 200:
            return True

        logger.warning(
            "Local NVIDIA NIM offline or VRAM limit reached. Engaging cloud "
            "failover. (health_status=%s)",
            resp.status_code,
        )
        return False

    except httpx.HTTPError as exc:
        # Covers TimeoutException, ConnectError, ReadError, etc.
        logger.warning(
            "Local NVIDIA NIM offline or VRAM limit reached. Engaging cloud "
            "failover. (%s)",
            exc.__class__.__name__,
        )
        return False


def _build_messages(text: str, system_prompt: str | None) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    return messages


async def stream_nim(
    settings: Settings,
    *,
    text: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
) -> AsyncIterator[str]:
    """Stream text deltas from the local NIM OpenAI-compatible endpoint."""
    payload = {
        "model": settings.nim_model,
        "messages": _build_messages(text, system_prompt),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if settings.nim_api_key:
        headers["Authorization"] = f"Bearer {settings.nim_api_key}"

    timeout = httpx.Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", settings.nim_chat_url, json=payload, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices")
                if choices:
                    content = choices[0].get("delta", {}).get("content")
                if content:
                    yield content