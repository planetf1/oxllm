# `oxllm` 🦀 (Oxide LLM Proxy)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/Rust-1.85.1%2B-orange.svg)](https://www.rust-lang.org/)
[![CI](https://github.com/planetf1/oxllm/actions/workflows/ci.yml/badge.svg)](https://github.com/planetf1/oxllm/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/oxllm.svg)](https://crates.io/crates/oxllm)

`oxllm` (Oxide LLM Proxy) is an ultra-minimalist, high-resilience adaptive routing LLM gateway written in Rust. It exposes an OpenAI-compatible interface, proxying requests to a tiered fallback pool of LLM providers with automatic rate-limit detection, circuit breakers, and failover.

Built to operate entirely in memory with zero local disk persistence, `oxllm` is optimized for resource-constrained edge devices (like OpenWrt routers), developer workstations, and background daemons. The **stripped release binary is ~2.6 MB** and idle RAM usage is **~14 MB**.

---

## 🚀 Key Features

* **Zero-Disk Dependency**: No SQLite, local caching, or file write operations during routing. State is strictly in memory.
* **&lt;2ms Routing Overhead**: Lock-free concurrency across routing loop, counters, and probe permits. Verified by CI benchmark.
* **Adaptive Circuit Breaker**: Strict `HalfOpen` state machine with lock-free `probe_in_flight` atomic check-and-set. Rate limits and server errors trip per-provider circuits with exponential backoff. Idle-based penalty decay automatically rehabilitates providers.
* **Tiered Failover**: Configure fallback chains across multiple providers. If the primary returns 429 or 5xx, the proxy transparently cascades to the next. Per-target `priority` states the preference explicitly instead of implying it from array order.
* **Decision-Model Endpoint**: `POST /v1/systemone` routes llama.cpp decision models (typed choice/score questions answered in one forward pass) through the same provider pool — model rewrite, circuit breaking, and failover included.
* **Routing Policies**: Per-virtual-model `priority` (configuration order), `fastest` (lowest measured time-to-first-byte), or `cheapest` (lowest configured cost) selection among healthy candidates.
* **Session Affinity**: Requests carrying a session key stay on one provider while it is healthy, keeping upstream prompt-prefix caches warm. Bounded LRU, idle TTL, automatic break-and-re-pin on failure.
* **Optional Inbound Auth**: `server.api_keys` gates the public `/v1/*` routes behind constant-time `Authorization: Bearer` checks. Admin routes are always localhost-only.
* **Anthropic Messages API**: `POST /v1/messages` (SSE streaming) and `/v1/messages/count_tokens` pass through to providers marked `api = "anthropic"`, with the same circuit breaking, failover, and session affinity as the OpenAI routes. See [Anthropic Messages API](#-anthropic-messages-api).
* **Raw Completions Route**: `POST /v1/completions` (with SSE streaming) for fill-in-the-middle clients such as editor autocomplete.
* **Hot Config Reloading**: the daemon watches `config.toml` and hot-reloads on change (invalid edits are rejected, the old config keeps serving). Also available on demand via `POST /reload` or `SIGHUP` — hot-swaps the provider pool via `tokio::sync::watch` without dropping connections, preserving circuit state, counters, and session pins for providers that keep their name.
* **Model Aliases & Default Route**: `[aliases]` maps client-facing names (`gpt-4o = "smart"`) onto virtual models, and `server.default_model` routes unknown names instead of 400 — point unmodified OpenAI clients straight at it.
* **Local Stats Dashboard**: Every provider tracks request count, success count, token volumes, and last request time via lock-free atomics. Query via `oxllm status` or `curl /status` — no external collector needed.
* **Traffic Visibility**: a bounded in-memory ring of recent request records (metadata only) plus `oxllm tail` for a live stream — route, provider, attempts, status, latency, TTFT per request, no collector needed.
* **OOM-Proof Telemetry**: Bounded OTel event channel (1024 cap) with non-blocking `try_send` drops. If `otelite` is offline, telemetry degrades gracefully and the proxy keeps running.
* **W3C Trace Context Propagation**: Extracts and injects `traceparent` headers for continuous trace spans.
* **Dual-Stack IPv4/IPv6**: Configurable via `bind_family`: `"ipv4"` (default), `"ipv6"`, or `"dual"` for both.
* **Unix-Style Environment Expansion**: Shell-style `${VAR}` replacement in TOML config values.
* **Musl Cross-Compilation**: Pure-Rust `rustls-tls` stack avoids native OpenSSL linking on edge routers.
* **OpenAI SDK Compatible** — JSON error format, CORS headers, and `x-request-id`
  correlation ID on every response. Works with official OpenAI Python and JavaScript
  SDKs, including browser-based usage.

---

## 🔐 Authentication

Outbound provider credentials live in each provider's `api_key`. Inbound auth is
opt-in: set `server.api_keys` to a list of keys and the public `/v1/*` routes will
require a matching `Authorization: Bearer <key>` header, compared in constant time.

```toml
[server]
host = "127.0.0.1"
port = 8080
otel_endpoint = "http://127.0.0.1:4318"
api_keys = ["${OXLLM_CLIENT_KEY}"]   # empty list (default) leaves the proxy open
```

Leaving it empty is appropriate for loopback-only deployments. Administrative
routes (`/status`, `/reload`, `/health`, `/admin/*`) are localhost-restricted
regardless of this setting, and a request rejected for a bad key never reaches an
upstream provider.

## 🧠 Anthropic Messages API

Clients that speak the Anthropic wire format (the Anthropic SDKs, `pi`, Claude
Code) can use `POST /v1/messages` and `POST /v1/messages/count_tokens`. Requests
are forwarded unchanged apart from the `model` rewrite, so the upstream must
speak Anthropic natively (Anthropic itself, or a LiteLLM proxy exposing
`/v1/messages`).

Mark such providers with `api = "anthropic"` (default `"openai"`, so existing
configs are unaffected). A request only ever reaches providers whose `api`
matches its route; non-matching targets in a virtual model are skipped, so one
virtual model can hold both kinds.

```toml
[[providers]]
name = "anthropic-direct"
enabled = true
base_url = "https://api.anthropic.com/v1/"   # keep the trailing slash
api_key = "${ANTHROPIC_API_KEY}"
models = ["claude-sonnet-5-5"]
api = "anthropic"

[[providers]]
name = "litellm-claude"
enabled = true
base_url = "https://litellm.example.com/v1/"
api_key = "${LITELLM_KEY}"
models = ["claude-sonnet-5-5"]
api = "anthropic"

[virtual_models]
claude = [
  { provider = "anthropic-direct", model = "claude-sonnet-5-5" },
  { provider = "litellm-claude", model = "claude-sonnet-5-5" },
]
```

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "x-api-key: $OXLLM_CLIENT_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model": "claude", "max_tokens": 256, "messages": [{"role": "user", "content": "Hello"}]}'
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="unused-if-no-api_keys")
msg = client.messages.create(
    model="claude", max_tokens=256, messages=[{"role": "user", "content": "Hello"}]
)
```

Behaviour specific to this route:

* **Upstream auth**: `x-api-key: <provider api_key>`. `anthropic-version` is
  forwarded from the client (default `2023-06-01`); `anthropic-beta` is forwarded
  when present.
* **Inbound auth**: with `server.api_keys` set, `/v1/messages*` accepts the key
  as `x-api-key` or `Authorization: Bearer` (constant-time compare). The OpenAI
  routes still take Bearer only.
* **Failover**: 429 puts the provider on cooldown (honouring `retry-after`) and
  fails over at once. 5xx (including 529 `overloaded_error`) and connection
  errors are retried against the same provider until the circuit threshold, so a
  one-off blip is absorbed; a persistent failure opens the circuit and fails over. Other 4xx (auth,
  `invalid_request_error`) are returned to the caller untouched, with no failover
  and no health penalty.
* **Errors oxllm generates** (unknown model, no `anthropic` candidate, all
  providers down) use the Anthropic shape
  `{"type":"error","error":{"type":...,"message":...}}`.
* **Streaming**: SSE passes through byte-for-byte. A failure mid-stream counts
  against the provider when the stream ends; there is no hot-swap.
* **Usage**: `/status` and `oxllm tail` read `usage` from the response (or from
  `message_start`/`message_delta` events when streaming). Input tokens include
  `cache_read_input_tokens` and `cache_creation_input_tokens`; cache reads also
  feed the cached-token counter.
* **Session affinity**: `x-session-id` as usual, falling back to
  `metadata.user_id` from the body.
* **`GET /v1/models`** returns a superset of the OpenAI and Anthropic list shapes
  (`object`/`created`/`owned_by` plus `type`/`display_name`/`created_at` and
  `has_more`/`first_id`/`last_id`), so either SDK family can list models.

OpenAI-to-Anthropic body translation is not implemented; it is a possible
follow-up.

## 🧵 OpenAI Responses and Conversations APIs (stateful)

The full OpenAI Responses and Conversations surface is proxied to
`api = "openai"` providers:

* `POST /v1/responses` (SSE streaming supported), `POST /v1/responses/compact`,
  `POST /v1/responses/input_tokens`
* `GET`/`DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input_items`,
  `POST /v1/responses/{id}/cancel`
* `POST /v1/conversations`, `GET`/`POST`/`DELETE /v1/conversations/{id}`
* `GET`/`POST /v1/conversations/{id}/items`,
  `GET`/`DELETE /v1/conversations/{id}/items/{item_id}`

These APIs keep conversation state upstream: a follow-up sends
`previous_response_id` (or `conversation`) instead of resending the history.
Only the provider that created that id can resolve it, so oxllm records which
provider returned each response `id`, `conversation.id` and conversation `id`,
and sends everything that names one of those ids to that provider.

* **New state** (a response with no `previous_response_id`/`conversation`, or
  a new conversation) routes and fails over like chat completions.
* **Follow-ups naming a known id** go only to the provider holding it. If that
  provider is down, the request fails with 502. **There is no failover**: no
  other provider could continue the conversation.
* **Ids oxllm has no record of** (after a restart, created through another
  proxy, or evicted) are looked up: oxllm asks each of the route's OpenAI
  providers for the object (`GET /responses/{id}` or `/conversations/{id}`)
  and pins the first that has it. If none does, a generation routes by policy
  and the upstream's own error comes back; a sub-resource call gets 404.
* **Upstream 4xx** (other than 408 and 429) pass through verbatim, with no
  failover or health penalty, as on `/v1/messages`.
* **Choosing a model for `POST /v1/conversations`**: its body has no `model`,
  so name the virtual model in an `x-oxllm-model` header, or set
  `server.default_model`. The body is forwarded untouched. The same header is
  the fallback for `compact` and `input_tokens`, where `model` is optional.
* **Deleting** a response or conversation drops its record; deleting an item
  does not.
* **Record table**: in memory, kept across reloads, rebuilt by lookup after a
  restart. `server.response_affinity_ttl_secs` (default 30 days, matching
  OpenAI's response retention) and `server.response_affinity_capacity`
  (default 4096, LRU). Conversations don't expire upstream, so a long-idle
  conversation falls out of the table and is found again by lookup on next use.

If you need failover for multi-turn conversations, keep the client stateless:
send the full history each turn with `store: false`, or use chat completions.
The Assistants API (`/v1/assistants`, `/v1/threads`) is not proxied: OpenAI
shut it down on 2026-08-26, with Responses and Conversations as the
replacement.

## 🌐 CORS Support

All public endpoints return `Access-Control-Allow-Origin: *`
headers. Browser-based applications can call the proxy directly.

---

## 📦 Project Layout

```
oxllm/
├── Cargo.toml              # Workspace root
├── config.toml             # Multi-tier cloud provider config (6 providers)
├── crates/
│   ├── oxllm-core/         # Core: config parsing, circuit breaker, router, telemetry
│   └── oxllm/              # CLI: Axum server, routes, signal handling, admin API
├── docs/
│   ├── architecture.md     # Concurrency model, circuit breaker rules, telemetry
│   └── providers.md        # Free-tier provider guide (snapshot: 2026-05-30)
├── .github/workflows/      # CI, security, release, crates.io publish
└── dist-workspace.toml     # cargo-dist release config
```

---

## 🛠️ Installation

### 1. Homebrew (easiest — pre-compiled binary)

```bash
brew tap planetf1/homebrew-tap
brew install oxllm
```

Pre-compiled for macOS and Linux (aarch64 + x86_64). No Rust toolchain needed. Binary size: ~2.6 MB stripped.

### 2. Cargo (compiled from source)

```bash
cargo install oxllm
```

Builds from [crates.io](https://crates.io/crates/oxllm). Requires Rust 1.85.1+.

### 3. From source (latest main)

```bash
git clone https://github.com/planetf1/oxllm.git
cd oxllm
cargo build --release
./target/release/oxllm serve --config config.toml
```

### Default Config Location

`oxllm serve` looks for config in this order:
1. `--config <path>` if provided
2. `~/.config/oxllm/config.toml` (XDG base directory)
3. `./config.toml` (current directory, for development)

```bash
# Quick start with local Ollama (no API keys needed):
cp config.toml ~/.config/oxllm/config.toml
oxllm serve

# Or with cloud providers (set env vars first):
export GROQ_API_KEY="gsk_..."
export GOOGLE_API_KEY="AIza..."
cp config.toml ~/.config/oxllm/config.toml
oxllm serve

# Or install it as a persistent user service (launchd on macOS,
# systemd on Linux): starts now, at login, and restarts on failure.
oxllm install
oxllm status
# ...and when you want it gone:
oxllm uninstall
```


## 🚀 Quick Start

The primary use case is routing across **multiple free-tier cloud providers** with automatic failover.
Ollama can be added as a local fallback for testing or as a last resort.

### 1. Set up providers

The repo includes two ready-to-use configs:

- **`config.toml`** — free-tier cloud providers in two virtual model tiers (`smart`, `basic`), with a local Ollama fallback

For a local-only trial, delete the cloud providers and keep the Ollama one (no API keys needed).

For the cloud config, set your API keys (see [Provider Guide](docs/providers.md) for sign-up links):

```bash
export GROQ_API_KEY="gsk_..."
export GOOGLE_API_KEY="AIza..."
export SAMBANOVA_API_KEY="..."
export OPENROUTER_API_KEY="sk-or-..."
```

### 2. Start the proxy

```bash
oxllm serve --config config.toml
```

### 3. Test it

```bash
# Smart model (strongest available — cascades through providers on failure)
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "smart", "messages": [{"role": "user", "content": "Hello"}]}'

# Basic model (fast, cheap, high rate limits)
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "basic", "messages": [{"role": "user", "content": "Hello"}]}'

# Embeddings
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "basic", "input": "hello world"}'

# Raw completions (fill-in-the-middle, e.g. editor autocomplete)
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "basic", "prompt": "def add(a, b):\n    ", "max_tokens": 16}'

# Live dashboard (no external collector needed)
curl http://127.0.0.1:8080/status
```


## ⚙️ Configuration

### Server Options

| Field | Default | Description |
|---|---|---|
| `host` | `"127.0.0.1"` | Bind address (not used when `bind_family` is `ipv6`/`dual`) |
| `port` | `8080` | Listen port |
| `otel_endpoint` | — | OTLP HTTP endpoint (e.g. `http://127.0.0.1:4318`). If unreachable, proxy starts without telemetry. Records spans with GenAI semantic attributes, 3 metrics (provider status gauge, request duration histogram, token counter), and W3C trace context propagation. See [architecture docs](docs/architecture.md#4-telemetry-layer--trace-context-propagation). |
| `upstream_timeout_secs` | `5` | Upstream request timeout in seconds |
| `bind_family` | `"ipv4"` | Address family: `"ipv4"`, `"ipv6"`, or `"dual"` (both) |
| `shutdown_timeout_secs` | `30` | Graceful-shutdown deadline after SIGTERM/SIGINT: in-flight requests drain until it expires, then the process exits anyway. `0` waits indefinitely. |
| `watch_config` | `true` | Watch the config file and hot-reload on change. Disable on flash-storage edge targets, or to make reloads strictly manual. |
| `request_log_capacity` | `256` | Recent request records kept in memory for `oxllm status` / `oxllm tail` (metadata only). `0` disables. Applies at startup. |
| `circuit_failure_threshold` | `3` | Consecutive failures before a provider's circuit opens. Lower it for twitchy free-tier upstreams. |
| `circuit_base_cooldown_secs` | `60` | Base circuit-open cooldown; doubles per failure beyond the threshold. |
| `rate_limit_fallback_secs` | `30` | Rate-limit cooldown when a 429 carries no `Retry-After`. |
| `default_model` | — | Virtual model that serves requests naming an unknown model, instead of returning 400. |

### Provider Definition

Each provider requires `name`, `enabled`, `base_url` (with trailing `/v1/`), `api_key` (or `${VAR}` env reference), and `models` list. Optional per-provider keys: `upstream_timeout_secs` (overrides `server.upstream_timeout_secs`, so a fast-failing local provider can fail over sooner than a slow global timeout) and `cost_per_mtok` (cost weight used by the `cheapest` policy). Per-provider circuit tuning: `circuit_failure_threshold` and `circuit_base_cooldown_secs` override the server-wide values for that provider. `first_byte_timeout_secs` applies to streaming requests only: a provider that sends no response headers in that time is abandoned and its circuit opens at once, so a hung endpoint costs one short wait instead of the full `upstream_timeout_secs`. Non-streaming requests ignore it, because their headers only arrive after the whole generation.

### Virtual Models (Fallback Chains)

Virtual models define the routing order. If a provider returns 429 or 5xx, the proxy transparently tries the next. Optional per-target fields state the routing intent explicitly:

- `priority` (lower = preferred): targets without one keep their configured order after all prioritized targets.
- `weight` (default policy only): proportional sharing among healthy targets in the same priority tier — `{ weight = 3 }` and `{ weight = 1 }` split traffic ~75/25 per request via a weighted lottery, instead of the first healthy target taking everything. Ignored by `fastest`/`cheapest`.

Configs that set neither behave exactly as before:

```toml
[virtual_models]
smart = [
  { provider = "groq-strong",  model = "llama-3.3-70b-versatile" },
  { provider = "groq-basic",   model = "meta-llama/llama-4-scout-17b-16e-instruct" },
  { provider = "ollama-fallback", model = "granite4.1:3b" },
]
```

#### Layered virtual models

Virtual models can be layered: a target of `{ virtual = "other" }` splices in another virtual model's targets, in their own order.

```toml
[virtual_models]
general-med = [{ virtual = "gemma12b" }, { virtual = "granite8b" }]   # quality order
gemma12b    = [{ provider = "omlx-local", model = "gemma-12b" }, { provider = "llamacpp-local", model = "gemma-12b-Q4" }]
granite8b   = [{ provider = "omlx-local", model = "granite-8b" }, { provider = "llamacpp-local", model = "granite-8b-Q4" }]
```

Clients ask for `general-med`; the runtime behind each model is a leaf detail you can swap without touching the tiers. A `priority` on a reference orders the whole group; a leaf reached by several paths appears once; cycles, undefined references and nesting deeper than 16 are rejected at load, `oxllm validate` and reload (a rejected reload keeps the running config). `/status` and counters show the flattened leaf targets.

### How the Routing Algorithm Works

1. A request names a virtual model (or an alias, or falls to `default_model`). Its pool is a list of **entries**: one (provider, upstream model) pair each, in routing order (see **Routing Policies**).
2. Entries are tried in order, **each at most once per request**. An entry is skipped when its provider or its model is unavailable: circuit open, rate-limited, or admin-offline. If none can take the request, the client gets a 502 (Anthropic-shaped `api_error` body on `/v1/messages`).
3. Health is tracked at two scopes. The **provider** scope is reachability: connection failures and 429s (a rate limit applies to the key, so it covers every model on the provider). The **model** scope is everything a response can say about one model: timeouts, 5xx/529, and a stream that breaks. One hanging model therefore does not take its healthy siblings on the same gateway out of rotation.
4. On success the entry's circuit closes and its failure count resets.
5. On 429 the provider is rate-limited for `retry-after` (else `rate_limit_fallback_secs`) and the request fails over immediately.
6. On a timeout the model's circuit opens **at once**: a hang costs every caller a full timeout, so waiting for `circuit_failure_threshold` victims would multiply it. Other failures (5xx, 529, connection errors) count towards the threshold (default 3); at the threshold the circuit opens for `circuit_base_cooldown_secs` x 2^(failures - threshold).
7. **HalfOpen probes**: after the cooldown a single probe request is allowed through; concurrent requests skip the entry (an atomic `compare_exchange`, no lock). A successful probe closes the circuit.
8. **Idle decay**: every 5 minutes without traffic reduces the failure count by 1; below the threshold an open circuit closes again.
9. Anthropic routes (`/v1/messages`) return 4xx other than 408/429 to the caller untouched: another provider would not fix a bad request or key, and it says nothing about provider health.

### Routing Policies

By default a virtual model uses `priority`: strict configuration order with the
health checks above. Alternatives, set per virtual model:

```toml
[policies]
smart = "fastest"     # lowest measured EWMA time-to-first-byte
budget = "cheapest"   # lowest cost_per_mtok, latency breaking ties
```

Under `fastest`, each provider's time-to-first-byte is tracked as an EWMA and
candidates are ordered by it in 20% bands — a provider must be meaningfully
faster to move up, so near-ties keep their configured order and do not flap on
noise. Every 20th request rotates the order so a demoted provider is still
sampled, and unmeasured providers sort first so a newly added one is tried
immediately rather than starved. Health checks (circuit breaker, rate limits,
manual overrides) apply before ordering in all cases.

### Model Aliases

Point unmodified OpenAI clients at oxllm without editing their model lists:
requests for an alias name route to its target virtual model, and `GET
/v1/models` lists the aliases alongside their targets.

```toml
[aliases]
gpt-4o = "smart"
gpt-4o-mini = "fast"
```

`server.default_model` goes one step further and routes *any* unknown model
name to a chosen virtual model instead of returning 400 — useful for clients
whose model names you do not control.

### Hot-Reload State Preservation

A reload (config-file change, `oxllm reload`, or SIGHUP) keeps runtime state
for every provider that keeps its name: circuit-breaker position, failure and
rate-limit windows, request/token counters, admin offline overrides, and
session-affinity pins. Editing the config mid-incident no longer re-enables a
provider you took offline or resets failover history. Server-level fields
that cannot be hot-swapped (`port`, `host`, `bind_family`, `otel_endpoint`,
affinity TTL/capacity, `shutdown_timeout_secs`, `watch_config`) produce an
explicit "restart required" warning in the reload response and logs.

### Session Affinity

Requests that carry a session key stick to one provider while it stays
healthy, so upstream prompt-prefix caches stay warm (a cold provider re-pays
prompt processing on every turn). The key is taken from the `x-session-id`
header (configurable via `server.session_affinity_header`, or `""` to disable
header affinity) with the request body's `prompt_cache_key` as fallback. Pins
expire after 15 minutes idle (`session_affinity_ttl_secs`) and the table is
bounded (`session_affinity_capacity`, default 1024, LRU eviction). When the
pinned provider's circuit trips, the session re-routes by policy immediately
and re-pins to whatever serves it. Keyless traffic is unaffected and keeps
rotating by policy.

### Traffic Visibility

A bounded ring of recent request records (metadata only: timestamp, endpoint,
route, provider, model, status, attempts, latency, TTFT, error, request id —
never bodies) is kept in memory (`server.request_log_capacity`, default 256):

- `oxllm status -vv` renders the most recent records; the header carries the daemon
  version and any failed reload or pending restart-required warning.
- `oxllm tail [-r N]` prints the last N records and then follows live traffic
  over a localhost-only SSE stream; plain lines, so `grep`/`jq` compose.
- Admin-offline providers show as "Offline (admin)" instead of looking healthy.

### Decision Endpoints (/v1/systemone)

llama.cpp decision models (typed choice/score questions answered in one
forward pass) route through the provider pool like chat. Point a virtual
model at decision-capable providers and POST to `/v1/systemone`:

```toml
[virtual_models]
route-decisions = [
  { provider = "llamacpp-local", model = "Kev-4B-Q4_K_M" },
  { provider = "hosted-jev",     model = "jev" },
]
```

```sh
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"route-decisions","state":"router","questions":[...]}'
```

Model rewrite per target, circuit breaking, and failover all apply; the
response passes through verbatim.

### Status & Metrics

`oxllm status` answers "is anything wrong?" first, then lets you drill down:

| Command | Shows |
|---|---|
| `oxllm status` | health, a list of problems, one row per route (state, strategy, available/total entries, requests, failures, p50/p95) |
| `oxllm status -v` | + each route's entries: state (with retry countdown), API, priority/weight tier, traffic, TTFT |
| `oxllm status -vv` | + provider reachability, token and latency percentiles, the last 15 requests |
| `--route NAME` | narrows any level to one virtual model |
| `--json` | the same tiers as JSON for scripts, e.g. `oxllm status --json \| jq '.health.problems'` |

A route is **ok** (every entry available), **degraded** (some), or **down** (none can take a request). Entry states are `healthy`, `probing`, `rate_limited`, `open`, and `offline` (admin). Traffic is attributed per `(provider, model)`, so two routes sharing a provider do not show each other's counts, and `cached_tokens` (upstream prompt-cache reuse) is reported per route. Tables drop optional columns to fit the terminal (`$COLUMNS` or `--width`).

The same counters feed the OpenTelemetry spans and metrics (see Telemetry below).

### Example Configs

- `config.toml` — cloud providers across 2 tiers (smart + basic) plus a local Ollama fallback

---

## 📟 CLI Subcommands

```bash
# Run
oxllm serve -c config.toml           # foreground; -v debug, -vv trace
oxllm validate -c config.toml        # syntax, references, cycles; run before reload
oxllm install -c config.toml         # run as a login service (launchd / systemd)

# Observe
oxllm status                         # problems first, then one row per route
oxllm status -v                      # + each route's entries
oxllm status -vv --json              # everything, as JSON
oxllm status --host router.lan       # a remote daemon (OXLLM_HOST / OXLLM_PORT)
oxllm tail -r 20                     # last 20 requests, then follow live (Ctrl-C to stop)

# Operate
oxllm provider list                  # providers and their state
oxllm provider offline <name>        # take a provider out of rotation (manual hold)
oxllm provider online <name>         # put it back
oxllm provider reset <name>          # clear circuit, failures, rate limit
oxllm reload                         # apply config edits (validated first); or --pid <pid> for SIGHUP
oxllm stop                           # graceful stop (drains in-flight streams)
oxllm uninstall                      # remove the service
```

`oxllm reload` and `oxllm stop` exit non-zero when no daemon can be reached or the new config is invalid, so `edit && oxllm reload` chains fail loudly instead of silently doing nothing. Every command has `--help` with examples and a suggested next step.

### Example `oxllm status` output

A pool with one provider down and a rate-limited sibling (default level):

```
oxllm 0.1.24  up 3h 02m  412 requests  DEGRADED

  ! pool-docs              degraded      1 of 2 entries available
  ! claude-ext             down          aws/claude-sonnet-5-5 (open), retry in 41s

Routes
+───────────────+──────────+──────────+───────+─────+──────+────────+────────+
| Route         | State    | Strategy | Avail | Req | Fail | p50    | p95    |
+───────────────+──────────+──────────+───────+─────+──────+────────+────────+
| pool-docs     | degraded | priority |   1/2 | 118 |    2 |  912ms | 3410ms |
| pool-easy     | ok       | priority |   3/3 | 204 |    0 |  655ms | 1203ms |
+───────────────+──────────+──────────+───────+─────+──────+────────+────────+

More: oxllm status -v (targets per route)  |  -vv (providers, recent requests)  |  --route NAME  |  --json
```

`-v` adds, per route, which entry is where and why (the Tier column shows `p1`/`w3` for an explicit priority or weight):

```
pool-docs  (priority: configured order, first healthy wins; failover on error)
+───+────────────+───────────────────────+───────────+───────────+──────+─────+
| # | Provider   | Model                 | State     | API       | Tier | Req |
+───+────────────+───────────────────────+───────────+───────────+──────+─────+
| 1 | claude-ext | aws/claude-sonnet-5-5 | open 41s  | anthropic | -    |  96 |
| 2 | claude-int | aws/claude-sonnet-5-5 | healthy   | anthropic | -    |  22 |
+───+────────────+───────────────────────+───────────+───────────+──────+─────+
```

All admin endpoints (`/health`, `/status`, `/reload`, `/admin/*`) are restricted to localhost — external callers receive `403 Forbidden`.

---

## 📊 Telemetry

oxllm exports OpenTelemetry (OTel) traces and metrics via OTLP/HTTP JSON to a collector like [otelite](https://github.com/planetf1/otelite).

### Configuration

Set `otel_endpoint` in `[server]` to point at your OTLP HTTP collector:

```toml
[server]
otel_endpoint = "http://127.0.0.1:4318"
```

If the endpoint is unreachable or not configured, oxllm logs a warning and starts
degraded — telemetry events are silently discarded. The proxy always works
without a collector.

### Span Attributes (Traces)

Every request produces one `oxllm.transaction` span with GenAI semantic conventions:

| Attribute | Example | Description |
|---|---|---|
| `gen_ai.operation.name` | `chat` / `messages` | Endpoint family |
| `gen_ai.provider.name` | `claude-ext` | Provider that served it (the last one tried, on failure) |
| `gen_ai.request.model` | `aws/claude-sonnet-5-5` | Upstream model used |
| `gen_ai.usage.input_tokens` | `1420` | Input tokens (Anthropic: includes cache reads and creations) |
| `gen_ai.usage.output_tokens` | `312` | Output tokens |
| `proxy.route` | `pool-docs` | Virtual model the client asked for |
| `proxy.attempts_required` | `2` | Pool entries tried |
| `proxy.initial_failure_reason` | `all_providers_failed` | Set on a failed request (`all_providers_failed`, `no_available_provider`, `stream_failed`) |
| `proxy.request_id` | `oxllm-3dc0...` | Matches the `x-request-id` response header |

A request that fails after every entry (or whose stream breaks) gets span status `ERROR`, so `otelite traces list --status ERROR` finds it. Spans link to an incoming W3C `traceparent` when present. Find them by service: `otelite traces list --since 1h` and look for service `oxllm`.

### Metrics

| Metric | Type | Description |
|---|---|---|
| `llm_proxy.provider.status` | Gauge | `0` = healthy, `1` = rate-limited, `2` = circuit tripped (per provider) |
| `llm_proxy.request.duration` | Histogram | Request lifecycle (ms), buckets from 5 ms to 10 min so slow LLM calls are not lost in an overflow bucket |
| `llm_proxy.tokens.consumed` | Counter | Cumulative tokens by provider, model and type |

Metrics are exported every 60 s by default. `otel_endpoint` is the collector's base URL; oxllm appends `/v1/traces` and `/v1/metrics` itself.

### Logging

Logs are emitted via `tracing` to stdout with `EnvFilter` support:

- **Default**: `info` — server start/stop, circuit transitions, errors
- **`-v`**: `debug` — adds per-request routing info
- **`-vv`**: `trace` — full request/response details

Override via `RUST_LOG` env var:
```bash
export RUST_LOG=oxllm=debug,oxllm_core=info
oxllm serve
```

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
