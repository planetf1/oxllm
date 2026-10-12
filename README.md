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
* **Tiered Failover**: Configure fallback chains across multiple providers. If the primary returns 429 or 5xx, the proxy transparently cascades to the next.
* **Hot Config Reloading**: `SIGHUP` signal or `POST /reload` HTTP endpoint — parses updated `config.toml` and hot-swaps the provider pool via `tokio::sync::watch` without dropping connections.
* **Local Stats Dashboard**: Every provider tracks request count, success count, token volumes, and last request time via lock-free atomics. Query via `oxllm status` or `curl /status` — no external collector needed.
* **OOM-Proof Telemetry**: Bounded OTel event channel (1024 cap) with non-blocking `try_send` drops. If `otelite` is offline, telemetry degrades gracefully and the proxy keeps running.
* **W3C Trace Context Propagation**: Extracts and injects `traceparent` headers for continuous trace spans.
* **Custom Headers & Client Header Passthrough**: Configure custom upstream HTTP headers per provider (`headers = { ... }` or `[providers.headers]`) for custom `User-Agent`, vendor-specific session IDs, or custom authentication schemes (e.g. Azure `api-key`), with automatic transparent passthrough for incoming client `x-*` headers.
* **Dual-Stack IPv4/IPv6**: Configurable via `bind_family`: `"ipv4"` (default), `"ipv6"`, or `"dual"` for both.
* **Unix-Style Environment Expansion**: Shell-style `${VAR}` replacement in TOML config values.
* **Musl Cross-Compilation**: Pure-Rust `rustls-tls` stack avoids native OpenSSL linking on edge routers.
* **OpenAI SDK Compatible** — JSON error format, CORS headers, and `x-request-id`
  correlation ID on every response. Works with official OpenAI Python and JavaScript
  SDKs, including browser-based usage.

---

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

### Provider Definition

Each provider requires `name`, `enabled`, `base_url` (with trailing `/v1/`), `api_key` (or `${VAR}` env reference), and `models` list.

### Virtual Models (Fallback Chains)

Virtual models define the routing order. If a provider returns 429 or 5xx, the proxy transparently tries the next:

```toml
[virtual_models]
smart = [
  { provider = "groq-strong",  model = "llama-3.3-70b-versatile" },
  { provider = "groq-basic",   model = "meta-llama/llama-4-scout-17b-16e-instruct" },
  { provider = "ollama-fallback", model = "granite4.1:3b" },
]
```

### How the Routing Algorithm Works

1. When a request arrives, the proxy iterates the virtual model's provider list in order.
2. For each provider, it checks: **circuit breaker state** (Closed? Open? HalfOpen?), **rate-limit window** (cooling down?), **manual override** (admin-disabled?).
3. The first healthy provider is selected for the request.
4. On success: circuit resets to Closed, failure count drops to 0.
5. On 429 (rate limit): sets a cooldown timer based on `retry-after` header (default 30s). After 3 failures, circuit opens.
6. On 5xx: increments failure counter. After 3 failures, circuit opens for **60 × 2^(failures-3)** seconds.
7. **HalfOpen probes**: After cooldown expires, a single probe request is allowed. Only one concurrent probe — others bypass via atomic `compare_exchange`.
8. **Idle decay**: Every 5 minutes without a request, failure count decreases by 1. Below 3 failures, Open circuits automatically rehabilitate to Closed.

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
oxllm status                         # virtual model routing table + per-provider counters

# Manage providers at runtime
oxllm provider list                  # condensed provider status table
oxllm provider offline <name>        # take a provider out of rotation
oxllm provider online <name>         # re-enable a disabled provider
oxllm provider reset <name>          # clear circuit breaker, failures, rate limit

# Config hot-reload (SIGHUP)
oxllm reload

# Graceful stop (drains in-flight SSE streams)
oxllm stop
```

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
