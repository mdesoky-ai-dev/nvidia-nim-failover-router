"""Fire a request at the router and print the SSE frames as they stream in."""

import asyncio
import json

import httpx

URL = "http://localhost:9000/api/v1/process-financials"

PAYLOAD = {
    "text": (
        "Q3 revenue rose 12% to $4.2M while operating costs grew 8%. "
        "Summarize the margin story in two sentences."
    )
}


async def main() -> None:
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", URL, json=PAYLOAD) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break

                frame = json.loads(data)

                if frame["type"] == "token":
                    print(frame["content"], end="", flush=True)
                elif frame["type"] == "done":
                    print("\n\n--- METRICS ---")
                    print(f"route:    {frame['source']}")
                    print(f"TTFT:     {frame['ttft_ms']} ms")
                    print(f"total:    {frame['total_ms']} ms")
                    print(f"tokens:   {frame['tokens']}")
                elif frame["type"] == "error":
                    print(f"\n[ERROR from {frame['source']}] {frame['message']}")


if __name__ == "__main__":
    asyncio.run(main())