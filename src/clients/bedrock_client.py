"""AWS Bedrock failover client.

Talks to Bedrock using aioboto3, the async version of AWS's boto3 toolkit.
Async matters here: it lets the Bedrock call wait without freezing the app,
just like the NIM client, so one server can still handle many requests.

We call Bedrock's `converse_stream` method, which streams the model's reply
back one text chunk at a time. Each chunk arrives tagged `contentBlockDelta`;
we pull the text out of those and pass it along.

AWS credentials are not stored here. aioboto3 finds them automatically from
your environment (env vars, ~/.aws config, or an AWS IAM role).
"""

import logging
from typing import AsyncIterator

import aioboto3

from src.config import Settings

logger = logging.getLogger("edge.bedrock")


async def stream_bedrock(
    settings: Settings,
    *,
    text: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
) -> AsyncIterator[str]:
    """Stream text deltas from AWS Bedrock via the Converse streaming API."""
    kwargs: dict = {
        "modelId": settings.bedrock_model_id,
        "messages": [{"role": "user", "content": [{"text": text}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    session = aioboto3.Session()
    async with session.client(
        "bedrock-runtime", region_name=settings.aws_region
    ) as client:
        response = await client.converse_stream(**kwargs)
        stream = response.get("stream")
        if stream is None:
            return

        async for event in stream:
            if "contentBlockDelta" in event:
                token = event["contentBlockDelta"].get("delta", {}).get("text")
                if token:
                    yield token
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                logger.info(
                    "Bedrock usage input_tokens=%s output_tokens=%s",
                    usage.get("inputTokens"),
                    usage.get("outputTokens"),
                )