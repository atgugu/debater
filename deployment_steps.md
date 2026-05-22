# Web Deployment Roadmap — Debater App (Revised)

## Current State
- Single-process Python `ThreadingHTTPServer` (stdlib)
- Claude CLI subprocess calls (~51 per 5-round debate)
- Filesystem state (JSON + markdown per debate)
- Single HTML file frontend (1778 lines, Tailwind + Marked.js via CDN)
- No auth, no SSL, no rate limiting, no DB
- Cost: ~$0.33/debate (Haiku), ~$3.30/debate (Opus)

---

## Recommended Stack

| Component | Choice | Why | Monthly Cost |
|---|---|---|---|
| Hosting | **Railway** | Auto-deploy from git, Nixpacks, persistent volumes, $5 hobby tier | $5 |
| Auth | **Clerk** | Drop-in social login (Google/GitHub/Apple), JWT verification, vanilla JS SDK | $0 (10K MAU free) |
| Database | **SQLite + Litestream** | Embedded, zero latency, JSON column support, S3 backup | ~$0.10 (S3) |
| Payments | **Stripe Checkout + Tax** | Industry standard, hosted payment page, auto tax collection | 2.9% + $0.30/txn |
| Real-time | **Server-Sent Events** | Unidirectional push, native browser `EventSource`, no library needed | $0 |
| Job queue | **asyncio.create_task** | No Celery/Redis — overkill for 1-10 concurrent debates | $0 |
| Framework | **FastAPI** | Async-first, SSE via StreamingResponse, Pydantic validation, auto docs | $0 |
| Frontend build | **Vite** | Tree-shake Tailwind, bundle Marked.js, minify — keep vanilla JS | $0 |
| Claude API | **Anthropic Python SDK** | Eliminates CLI binary, 1-3s faster per call, enables streaming | Same API costs |
| Build/deploy | **Nixpacks (Railway)** | Auto-detects Python, no Dockerfile needed | $0 |
| **Total** | | | **~$5.10/mo + Stripe fees** |

---

## Implementation Order (optimized for dependencies)

### Step 1: Anthropic SDK Migration (prerequisite for deployment)
**File:** `debater.py` — `run_agent()` and `run_claude_cmd()`

Replace subprocess `claude` CLI calls with direct SDK:
```python
from anthropic import Anthropic
client = Anthropic()  # reads ANTHROPIC_API_KEY from env

def run_agent(agent_name, prompt, model, ...):
    response = client.messages.create(
        model=model, max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
```

Benefits:
- Eliminates `claude` binary dependency (can't install on cloud servers easily)
- ~1-3s faster per call (no subprocess overhead)
- Enables streaming responses later
- For WebSearch: use Brave Search API or Tavily as a tool

**Effort:** 1-2 days

### Step 2: FastAPI Migration
**File:** `debater.py` — replace `DebaterHandler` class

Replace stdlib HTTP with FastAPI:
- `do_GET` / `do_POST` → `@app.get()` / `@app.post()` decorators
- Manual JSON parsing → Pydantic models
- Manual headers → CORS middleware
- Thread-per-debate → `asyncio.create_task`
- Serve `index.html` via `StaticFiles`

**Effort:** 2-3 days

### Step 3: Server-Sent Events
**Files:** `debater.py` + `index.html`

Replace 10s polling with SSE:
```python
# Backend
@app.get("/api/debate/{id}/stream")
async def stream_debate(id: str):
    async def generate():
        while True:
            state = load_state(id)
            yield f"data: {json.dumps(state)}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(generate(), media_type="text/event-stream")
```
```javascript
// Frontend — replaces setInterval polling
const es = new EventSource(`/api/debate/${debateId}/stream`);
es.onmessage = (e) => updateUI(JSON.parse(e.data));
```

**Effort:** 1 day

### Step 4: SQLite Migration
**File:** `debater.py` — replace filesystem functions

Replace `load_state()`/`save_state()` with SQLite:
```sql
CREATE TABLE debates (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    claim TEXT,
    title TEXT,
    status TEXT,
    state JSON,
    created_at TEXT,
    updated_at TEXT
);
```
- `list_debates()` → `SELECT ... ORDER BY created_at DESC`
- `load_state()` → `SELECT state FROM debates WHERE id = ?`
- `save_state()` → `UPDATE debates SET state = ? WHERE id = ?`
- MD files stored as columns or kept on filesystem (less critical)
- Add Litestream for continuous S3 backup

**Effort:** 1-2 days

### Step 5: Railway Deployment
**Files:** `Procfile`, `requirements.txt`, Railway dashboard

```
# Procfile
web: uvicorn debater:app --host 0.0.0.0 --port $PORT

# requirements.txt
fastapi
uvicorn
anthropic
aiosqlite
pyjwt
cryptography
```

- Connect Railway to GitHub repo
- Set env vars: `ANTHROPIC_API_KEY`, `CLERK_SECRET_KEY`
- Nixpacks auto-builds, Railway auto-deploys on push
- Railway provides SSL, custom domain support

**Effort:** 1 hour

### Step 6: Clerk Auth
**Files:** `index.html` (Clerk JS SDK), `debater.py` (JWT middleware)

Frontend — add Clerk sign-in:
```html
<script src="https://cdn.clerk.dev/v1/clerk.browser.js"></script>
```

Backend — verify JWT on every request:
```python
from fastapi import Depends, HTTPException
import jwt

async def get_user(authorization: str = Header()):
    token = authorization.replace("Bearer ", "")
    payload = jwt.decode(token, CLERK_PUBLIC_KEY, algorithms=["RS256"])
    return payload["sub"]  # user_id
```

- Add `user_id` column to debates table
- Filter debates by user: `WHERE user_id = ?`

**Effort:** 1-2 days

### Step 7: Stripe Payments
**Files:** `debater.py` (checkout + webhook), `index.html` (pricing UI)

Tiers:
| Tier | Price | Limits |
|---|---|---|
| Free | $0 | 3 debates/month, Haiku only, 3 rounds |
| Pro | $9.99/mo | 50 debates/month, all models, 10 rounds |
| Enterprise | $29.99/mo | Unlimited, Opus, API access |

```python
@app.post("/api/checkout")
async def create_checkout(user_id: str = Depends(get_user)):
    session = stripe.checkout.Session.create(
        customer=get_or_create_stripe_customer(user_id),
        line_items=[{"price": PRICE_ID, "quantity": 1}],
        mode="subscription",
        success_url="...", cancel_url="...",
    )
    return {"url": session.url}
```

Enable Stripe Tax in dashboard for automatic tax collection.

**Effort:** 2-3 days

### Step 8: Vite Frontend Build
**Files:** `vite.config.js`, `package.json`, `index.html`

```bash
npm init -y
npm install tailwindcss marked vite
```

- Move CDN imports to npm packages
- `vite build` → tree-shaken CSS, minified JS
- Railway build step: `npm run build` before Python start

**Effort:** half day

---

## Timeline

| Step | What | Days |
|---|---|---|
| 1 | Anthropic SDK | 1-2 |
| 2 | FastAPI | 2-3 |
| 3 | SSE | 1 |
| 4 | SQLite | 1-2 |
| 5 | Railway deploy | 0.1 |
| **MVP live** | | **~6 days** |
| 6 | Clerk auth | 1-2 |
| 7 | Stripe payments | 2-3 |
| 8 | Vite build | 0.5 |
| **Full production** | | **~10-12 days** |

---

## Key Decisions Explained

**Why SQLite over Postgres?** The app stores JSON-heavy state (~143KB per 5-round debate). Every agent status update does load/save. SQLite is microseconds, Postgres over network is 10-50ms. No benefit from relational features — the state is a document.

**Why no Celery/Redis?** The app runs 1-10 concurrent debates. Each is a sequential pipeline. `asyncio.create_task` handles this natively. Celery adds a broker, a worker, and serialization overhead for a problem that doesn't exist.

**Why SSE over WebSockets?** Data flows one direction: server → client. SSE is simpler (no upgrade handshake, built-in browser reconnection, works through proxies). WebSockets are for bidirectional real-time (chat, gaming).

**Why Clerk over Supabase Auth?** Clerk is auth-only, works with any backend. Supabase Auth couples to Supabase ecosystem. Since we use SQLite not Supabase Postgres, their auth alone creates unnecessary dependency.

**Why Railway over Fly.io?** Railway auto-detects Python, deploys from `git push`, has persistent volumes. Fly.io requires more ops knowledge (`fly.toml`, Machines API, volume management). Same price, better DX for solo dev.

---

## Security Hardening

### Authentication & Authorization
- **Every API endpoint** behind Clerk JWT verification (no anonymous access except landing page)
- **Debate ownership**: `user_id` on every debate row, enforced in queries (`WHERE user_id = ?`)
- **API key rotation**: Anthropic key as Railway secret, rotatable without redeploy
- **CORS whitelist**: Only allow requests from your domain, not `*`

### Input Validation & Injection
- **Claim sanitization**: Pydantic model with `max_length=2000`, strip HTML tags
- **Message sanitization**: Same — prevent prompt injection payloads reaching Claude context
- **Path traversal**: FastAPI path parameters are type-validated, no raw filesystem access
- **SQL injection**: Parameterized queries only (`?` placeholders), never string interpolation
- **XSS**: Marked.js sanitizes by default; replace remaining `.innerHTML` with `.textContent` where not rendering markdown
- **Content Security Policy header**: restrict script sources to self + Clerk CDN

### Rate Limiting
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_user_id)

@app.post("/api/debate")
@limiter.limit("10/hour")  # Free tier
async def create_debate(...):
```
- **Per-user limits**: 10 debates/hour (free), 50/hour (pro)
- **Per-IP fallback**: 100 requests/minute for unauthenticated endpoints
- **Debate creation cooldown**: 30 seconds between debates per user
- **Message spam**: max 5 messages per minute per debate
- **Claude API budget**: track tokens per user per month, hard cutoff at tier limit

### Data Protection
- **SSL/TLS**: Railway provides automatic HTTPS with Let's Encrypt
- **Debate content encryption**: optional — encrypt `state` column with user-derived key (debates contain potentially sensitive claims)
- **GDPR compliance**: user data export endpoint, account deletion cascade (delete all debates + Stripe customer)
- **Debug logs**: strip from production or redact prompts/responses (contain full debate text)
- **Litestream backups encrypted**: S3 bucket with server-side encryption (SSE-S3)

### Abuse Prevention
- **Anthropic API cost cap**: set `max_tokens` per call, hard monthly budget per user
- **Research query limit**: already capped at 3-6 per agent per round — enforce globally too
- **Debate timeout**: kill debates running longer than 30 minutes
- **Content moderation**: Anthropic's built-in safety + optional pre-screen of claims via classifier

---

## Scaling Strategy

### Phase 1: Single Server (0-500 users)
**Current architecture works with the recommended stack changes.**
- Railway single instance handles ~50 concurrent debates
- SQLite handles ~1000 writes/second (debate updates are ~10/second peak)
- Anthropic API is the bottleneck, not the server
- **Cost: $5/month + API costs**

### Phase 2: Vertical Scaling (500-5,000 users)
- **Railway Pro plan** ($20/month): more CPU/RAM
- **SQLite still works**: WAL mode handles concurrent reads + single writer efficiently
- **Connection pooling**: Anthropic SDK with `max_retries` and connection reuse
- **CDN for frontend**: Cloudflare free tier for static assets (index.html, JS/CSS bundle)
- **Cost: $20/month + CDN (free) + API costs**

### Phase 3: Horizontal Scaling (5,000-50,000 users)
This is where architecture changes are needed:

- **Migrate SQLite → Postgres** (Neon serverless): multiple server instances need shared DB
- **Migrate debate execution to task queue**: Dramatiq + Redis for cross-instance job dispatch
- **Railway multi-instance**: 2-4 instances behind Railway's load balancer
- **SSE connection management**: each instance handles its own SSE streams; client reconnects to any instance
- **Sticky sessions**: not needed if debate state is in Postgres (any instance can serve any request)
- **Cost: $50-100/month + Neon ($19/month) + Redis ($10/month)**

### Phase 4: Global Scale (50,000+ users)
- **Multi-region**: Fly.io (replaces Railway) for edge deployment
- **Read replicas**: Neon automatic read replicas in each region
- **Queue per region**: Redis cluster
- **CDN**: Cloudflare Pro for global edge caching + DDoS protection
- **Monitoring**: Datadog or Grafana Cloud for distributed tracing
- **Cost: $200-500/month + API costs scale linearly**

### Scaling Bottleneck Analysis
| Component | Bottleneck At | Solution |
|---|---|---|
| Anthropic API | ~50 concurrent calls | Increase `max_concurrent_claude`, queue excess |
| SQLite | ~5,000 concurrent users | Migrate to Postgres |
| Railway single instance | ~500 concurrent connections | Add instances |
| SSE connections | ~10,000 per instance | Multiple instances + reconnection |
| Disk (debate files) | ~100GB | S3 for markdown files, DB for state only |
| Stripe webhooks | Not a bottleneck | Stripe handles scale |

### Key Insight
**The real bottleneck is always the Anthropic API.** A 5-round debate makes ~51 API calls over ~5-10 minutes. With `max_concurrent_claude=10`, the server can run ~10 debates simultaneously regardless of how many instances or how powerful the DB. Scaling the app server beyond the API throughput is wasted effort. The scaling strategy should focus on:
1. Queueing excess debates (not dropping them)
2. Giving users estimated wait times
3. Letting paying users skip the queue
