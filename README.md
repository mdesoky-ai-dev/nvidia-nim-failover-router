# nvidia-nim-failover-router

A production-grade, low-latency inference router that serves LLM requests from a
**local NVIDIA NIM node** when available and **transparently fails over to AWS
Bedrock** when it is not. Built for Desoky Capital LLC.

The router speaks the OpenAI-compatible protocol that NIM exposes, streams tokens
back as Server-Sent Events, and instruments every request with
**latency-to-first-token (TTFT)** and the lane that served it — so the
performance case for edge inference is measured, not asserted.

`FastAPI` · `httpx` · `aioboto3` · fully async end to end.

---

## Why hybrid-edge

Local GPU inference avoids the cloud round trip, keeps sensitive financial text
on-premises, and collapses time-to-first-token. But a local node has finite VRAM
and can saturate or go down. A single-lane design would drop requests; this
router treats the cloud as an always-available safety net and pays cloud rates
only for overflow.

The measured gap, captured by the router's own metrics on this project:

| Lane          | Time to first token | Relative |
| ------------- | ------------------- | -------- |
| `LOCAL_NIM`   | **~44 ms**          | 1×       |
| `AWS_BEDROCK` | **~1,830 ms**       | ~42× slower |

That ~42× difference in TTFT *is* the business case for edge inference, and it is
the number the router exists to surface.

---

## How it works

```
                         POST /api/v1/process-financials
                                      │
                                      ▼
                            route_and_stream()
                                      │
                     1. GET :8000/v1/health/ready  (200 ms budget)
                                      │
                    ┌─────────────────┴─────────────────┐
                 200 OK                              non-200 / timeout / refused
                    │                                     │
                    ▼                                     ▼
              LOCAL_NIM                              AWS_BEDROCK
        stream_nim() → httpx                   stream_bedrock() → aioboto3
        OpenAI-format SSE                      Bedrock Converse stream
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      ▼
                    normalized SSE frames to the client
              {"type","content","source"}  +  final {"type":"done", ...}
```

1. **Health guard-rail.** Every request begins with a fast (200 ms budget) probe
   of the local node's readiness endpoint. A non-200, a timeout, or a refused
   connection all mean the edge lane is unavailable.
2. **Lane selection.** Healthy → local NIM. Otherwise → Bedrock.
3. **Normalized streaming.** Each backend's native chunk format is unwrapped in
   its client and re-emitted as one uniform SSE shape, so the caller consumes
   both lanes identically and only learns which one served it by reading
   `source`.
4. **Pre-first-token failover.** If the local lane passes its health check but
   dies *before emitting a token*, the router silently switches to Bedrock. Once
   tokens have started streaming it cannot rewind, so a mid-stream error is
   surfaced instead.
5. **Observability.** TTFT and the routing destination are logged server-side and
   returned to the caller in the closing `done` frame.

---

## Project layout

```
src/
  config.py              env-driven settings (endpoints, timeouts, model IDs)
  main.py                FastAPI app: endpoint, /healthz, serves the demo UI
  clients/
    nim_client.py        local health probe + OpenAI-format streaming
    bedrock_client.py    async Bedrock Converse streaming
  router/
    fallback.py          the switchboard: health → lane → TTFT → failover
static/
  index.html             single-file demo UI (streaming, route badge, metrics)
stub_nim.py              protocol-compatible stand-in for a real NIM container
test_client.py           async SSE client for terminal testing
```

---

## A note on the local lane (honest scope)

This project was developed on hardware **without an NVIDIA GPU**, so a real NIM
container cannot run locally here. The `LOCAL_NIM` lane is therefore validated
against `stub_nim.py` — a small server that implements the exact two endpoints
the router depends on:

* `GET /v1/health/ready` — the readiness probe
* `POST /v1/chat/completions` — an OpenAI-format SSE token stream

This validates the parts this project is actually about: **routing logic,
protocol handling, streaming, and failover.** It does **not** exercise
NIM-specific behavior (model loading, VRAM pressure, GPU throughput).

Because the router is configured entirely through environment variables, moving
to real hardware is a config change, not a code change: stand up a NIM container
on a GPU node and point `NIM_BASE_URL` at it. Nothing in `src/` changes.

---

## Running it

Requires Python 3.11+ and AWS credentials with Bedrock access.

**1. Install**

```bash
pip install -r requirements.txt
```

**2. Configure** — create a `.env` in the project root (gitignored):

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

Newer Bedrock Claude models require an inference profile — note the regional
`us.` prefix on the model ID.

**3. Start the router** (terminal 1):

```bash
uvicorn src.main:app --reload --port 9000
```

**4. (Optional) Start the stub NIM** to exercise the local lane (terminal 2):

```bash
uvicorn stub_nim:app --port 8000 --host 0.0.0.0
```

With the stub down, requests fail over to Bedrock. With it up, they route local.

**5. Try it.** Open the demo UI at **http://localhost:9000/**, or use the CLI:

```bash
python test_client.py
```

---

## Live failover demo

The stub exposes admin toggles so you can take the edge node down and bring it
back *without restarting anything* — the router re-probes on every request:

```bash
curl -X POST http://localhost:8000/admin/offline   # next request → AWS_BEDROCK
curl -X POST http://localhost:8000/admin/online    # next request → LOCAL_NIM
```

The demo UI wires these to buttons and shows the route badge and TTFT flip in
real time.

---

## Configuration reference

All values are environment variables (see `src/config.py`); every one has a
sensible default.

| Variable            | Default                                          | Purpose                          |
| ------------------- | ------------------------------------------------ | -------------------------------- |
| `NIM_BASE_URL`      | `http://localhost:8000`                          | Local NIM node base URL          |
| `NIM_MODEL`         | `meta/llama-3.1-8b-instruct`                      | Model the NIM container serves   |
| `HEALTH_TIMEOUT_MS` | `200`                                            | Readiness-probe budget           |
| `AWS_REGION`        | `us-east-1`                                       | Bedrock region                   |
| `BEDROCK_MODEL_ID`  | `anthropic.claude-sonnet-4-5-20250929-v1:0`       | Failover model                   |
| `DEFAULT_MAX_TOKENS`| `1024`                                           | Generation cap                   |
| `DEFAULT_TEMPERATURE`| `0.2`                                           | Sampling temperature             |

---

## API

**`POST /api/v1/process-financials`** — streams the answer as SSE.

```json
{ "text": "Q3 revenue rose 12% ...", "max_tokens": 1024, "temperature": 0.2 }
```

Frames: `{"type":"token","content":"...","source":"LOCAL_NIM"}` during streaming,
then `{"type":"done","source":...,"ttft_ms":...,"total_ms":...,"tokens":...}`.

**`GET /healthz`** — router liveness plus a live probe of the edge node and the
lane a request would currently take.