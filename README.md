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

OpenAI-to-Anthropic body translation is not implemented; it is a possible
follow-up.

## 🌐 CORS Support

All public endpoints return `Access-Control-Allow-Origin: *`
headers. Browser-based applications can call the proxy directly.

---

## 📦 Project Layout

```
oxllm/
├── Cargo.toml              # Workspace root
├── config.toml             # Multi-tier cloud provider config (6 providers)
├── config-local-test.toml  # Local-only Ollama config for testing
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
./target/release/oxllm serve --config config-local-test.toml
```

### Default Config Location

`oxllm serve` looks for config in this order:
1. `--config <path>` if provided
2. `~/.config/oxllm/config.toml` (XDG base directory)
3. `./config.toml` (current directory, for development)

```bash
# Quick start with local Ollama (no API keys needed):
cp config-local-test.toml ~/.config/oxllm/config.toml
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

- **`config.toml`** — 6 free-tier cloud providers with 2 virtual model tiers
- **`config-local-test.toml`** — local Ollama only (for testing)

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

For local testing with Ollama instead of cloud providers:
```bash
oxllm serve --config config-local-test.toml
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

Each provider requires `name`, `enabled`, `base_url` (with trailing `/v1/`), `api_key` (or `${VAR}` env reference), and `models` list. Optional per-provider keys: `upstream_timeout_secs` (overrides `server.upstream_timeout_secs`, so a fast-failing local provider can fail over sooner than a slow global timeout) and `cost_per_mtok` (cost weight used by the `cheapest` policy).

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

### How the Routing Algorithm Works

1. When a request arrives, the proxy iterates the virtual model's provider list in order (or by measured latency — see **Routing Policies** below).
2. For each provider, it checks: **circuit breaker state** (Closed? Open? HalfOpen?), **rate-limit window** (cooling down?), **manual override** (admin-disabled?).
3. The first healthy provider is selected for the request.
4. On success: circuit resets to Closed, failure count drops to 0.
5. On 429 (rate limit): sets a cooldown timer based on `retry-after` header (default `rate_limit_fallback_secs`, 30s). After `circuit_failure_threshold` failures, circuit opens.
6. On 5xx: increments failure counter. After `circuit_failure_threshold` (default 3) failures, circuit opens for **`circuit_base_cooldown_secs` × 2^(failures−threshold)** seconds (default 60s base).
7. **HalfOpen probes**: After cooldown expires, a single probe request is allowed. Only one concurrent probe — others bypass via atomic `compare_exchange`.
8. **Idle decay**: Every 5 minutes without a request, failure count decreases by 1. Below 3 failures, Open circuits automatically rehabilitate to Closed.

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

- `oxllm status` renders the most recent records, plus the daemon version and
  the last reload outcome (failures and pending restart-required warnings).
- `oxllm tail [-n N]` prints the last N records and then follows live traffic
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

`oxllm status` prints per-virtual-model routing tables, a per-route rollup with
latency distributions (p50/p95/p99 of total latency and time-to-first-byte), and
per-provider totals. Traffic is attributed per `(provider, model)` pair, so two
virtual models sharing one provider do not show each other's traffic, and
`cached_tokens` (upstream prompt-cache reuse) is reported per route so warm-vs-cold
latency differences can be explained. Tables
drop optional columns to fit the terminal width (`$COLUMNS`, or `--width`), and
`--json` emits the raw `/status` payload for scripting. The same counters feed
the OpenTelemetry spans (`gen_ai.usage.*`, `proxy.attempts_required`) and
metrics exported to the configured OTLP endpoint.

### Example Configs

- `config.toml` — 6 cloud providers across 2 tiers (smart + basic)
- `config-local-test.toml` — local Ollama only, zero API keys

---

## 📟 CLI Subcommands

```bash
# Start the proxy
oxllm serve                          # default: ~/.config/oxllm/config.toml
oxllm serve -v                       # verbose: per-request routing info
oxllm serve -vv                      # trace: full request/response dump

# Validate config syntax
oxllm validate                       # checks env vars, provider cross-refs

# Live dashboard (no external collector needed)
oxllm status                         # version, routing tables, counters, recent requests
oxllm status --host router.lan       # against a remote daemon (OXLLM_HOST/OXLLM_PORT)

# Watch traffic as it happens (Ctrl-C to stop)
oxllm tail                           # live stream of request records
oxllm tail -n 20                     # last 20 records, then follow live traffic

# Manage providers at runtime
oxllm provider list                  # condensed provider status table
oxllm provider offline <name>        # take a provider out of rotation
oxllm provider online <name>         # re-enable a disabled provider
oxllm provider reset <name>          # clear circuit breaker, failures, rate limit

# Config hot-reload (automatic on file change; manual trigger below)
oxllm reload                         # POST /reload on -p PORT (default 8080),
                                     # or --pid <pid> to SIGHUP directly

# Graceful stop (drains in-flight SSE streams)
oxllm stop
```

`oxllm reload` and `oxllm stop` exit non-zero when no daemon can be reached, so
`edit && oxllm reload` chains fail loudly instead of silently doing nothing.

### Example `oxllm status` Output (after ~5 hours of real use)

```
Uptime: 311m 3s  |  Total Requests: 150

Virtual Model: smart
-------------------------------------------------------------------------------------------------------------------------------
| Provider             | Model                                         | Circuit                        | Requests |  Success |
-------------------------------------------------------------------------------------------------------------------------------
| groq-strong          | llama-3.3-70b-versatile                       | Open (197s cooldown)           |       16 |        0 |
| sambanova-strong     | Llama-4-Maverick-17B-128E-Instruct            | Closed (Healthy)               |       30 |        8 |
| groq-basic           | meta-llama/llama-4-scout-17b-16e-instruct     | Open (225s cooldown)           |       30 |       17 |
| google-basic         | gemini-2.5-flash                              | Closed (Healthy)               |       32 |       22 |
| sambanova-basic      | DeepSeek-V3.1                                 | Closed (Healthy)               |       15 |       10 |
| openrouter-basic     | ibm-granite/granite-4.1-8b                    | Closed (Healthy)               |       27 |       27 |
| ollama-fallback      | granite4.1:3b                                 | Closed (Healthy)               |        0 |        0 |

Virtual Model: basic
-------------------------------------------------------------------------------------------------------------------------------
| Provider             | Model                                         | Circuit                        | Requests |  Success |
-------------------------------------------------------------------------------------------------------------------------------
| groq-basic           | meta-llama/llama-4-scout-17b-16e-instruct     | Open (225s cooldown)           |       30 |       17 |
| google-basic         | gemini-2.5-flash                              | Closed (Healthy)               |       32 |       22 |
| sambanova-basic      | DeepSeek-V3.1                                 | Closed (Healthy)               |       15 |       10 |
| openrouter-basic     | ibm-granite/granite-4.1-8b                    | Closed (Healthy)               |       27 |       27 |
| ollama-fallback      | granite4.1:3b                                 | Closed (Healthy)               |        0 |        0 |

Use 'oxllm provider offline <name>' to take a provider out of rotation.
Use 'oxllm provider reset <name>' to clear circuit breaker state.
```

Piping through `cat` or a pager adds the full per-provider counter table with failure counts, token volumes, and last-request timestamps:

```
+--------------------+-----------------------------------------------+--------------------------------+----------+---------------+----------+-----------+--------------+---------------+--------------+
| Provider Name      | Models                                                | Circuit Breaker State          | Failures | Rate Limited? | Requests | Successes | Tokens Input | Tokens Output | Last Request|
+--------------------+-----------------------------------------------+--------------------------------+----------+---------------+----------+-----------+--------------+---------------+--------------+
| groq-strong        | llama-3.3-70b-versatile                       | Open (Cooldown: 197s left)     | 5        | No            | 16       | 0         | 0            | 0             | Just now     |
| sambanova-strong   | Llama-4-Maverick-17B-128E-Instruct            | Closed (Healthy)               | 13       | No            | 30       | 8         | 94           | 4             | Just now     |
| groq-basic         | meta-llama/llama-4-scout-17b-16e-instruct     | Open (Cooldown: 225s left)     | 5        | No            | 30       | 17        | 232          | 10            | Just now     |
| google-basic       | gemini-2.5-flash                              | Closed (Healthy)               | 8        | No            | 32       | 22        | 0            | 0             | Just now     |
| sambanova-basic    | DeepSeek-V3.1                                 | Closed (Healthy)               | 1        | Yes           | 15       | 10        | 0            | 0             | Just now     |
| openrouter-basic   | ibm-granite/granite-4.1-8b                    | Closed (Healthy)               | 0        | No            | 27       | 27        | 0            | 0             | Just now     |
| ollama-fallback    | granite4.1:3b                                 | Closed (Healthy)               | 0        | No            | 0        | 0         | 0            | 0             | Never        |
+--------------------+-----------------------------------------------+--------------------------------+----------+---------------+----------+-----------+--------------+---------------+--------------+
```

This example — captured after 5 hours of real use — shows:
- **groq-strong**: Circuit is *Open* (197s cooldown remaining) after 5 failures with 0 successes across 16 requests, meaning all attempts hit rate limits or errors.
- **groq-basic**: Also *Open* (225s cooldown) after 5 failures, but 17 of 30 requests succeeded before the circuit tripped.
- **sambanova-strong**: *Closed* and healthy but with 13 failures — it's been reliable enough to stay open despite a high error rate.
- **openrouter-basic**: Perfect record — 27/27 requests succeeded, 0 failures, circuit Closed.
- **sambanova-basic**: Currently *rate-limited* (1 failure, marked "Yes"), but the circuit remains Closed.
- **ollama-fallback**: Never used (0 requests), sitting idle as the last-resort local model.

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

Every routed transaction generates a span with GenAI semantic conventions:

| Attribute | Example | Description |
|---|---|---|
| `gen_ai.operation.name` | `chat` / `embeddings` | Operation type |
| `gen_ai.provider.name` | `groq-strong` | Provider selected |
| `gen_ai.request.model` | `llama-3.3-70b-versatile` | Model used |
| `gen_ai.usage.input_tokens` | `1420` | Input token count |
| `gen_ai.usage.output_tokens` | `312` | Output token count |
| `proxy.attempts_required` | `2` | How many providers were tried |
| `proxy.initial_failure_reason` | `429_too_many_requests` | First failure cause (if any) |

Spans are linked to incoming W3C `traceparent` headers when present.

### Metrics

| Metric | Type | Description |
|---|---|---|
| `llm_proxy.provider.status` | Gauge | `0` = healthy, `1` = rate-limited, `2` = circuit tripped |
| `llm_proxy.request.duration` | Histogram | Request lifecycle duration (ms) |
| `llm_proxy.tokens.consumed` | Counter | Cumulative tokens by provider, model, type |

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
