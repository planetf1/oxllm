# Technical Architecture & Design Document: `oxllm`

`oxllm` (Oxide LLM Proxy) is a minimalist, ultra-low-footprint adaptive routing gateway written in Rust. It exposes a single OpenAI-compatible interface, proxying requests to a tiered fallback pool of free-tier LLM providers. It operates entirely in memory with zero local disk persistence, making it highly optimized for edge devices, routers, and developer workstations.

---

## 1. System Philosophy & Target Constraints

* **Zero-Disk Dependency:** No SQLite or local caching. State is maintained strictly in memory using thread-safe primitives.
* **Resource Ceilings:**
    * **Binary Size:** < 15 MB (stripped, release build targeting musl/macOS).
    * **Memory Footprint:** < 25 MB RAM at idle; < 40 MB under concurrent load.
    * **Latency Overhead:** < 2 ms internal routing overhead (excluding upstream network request).
* **Platforms:** Native execution on macOS (Apple Silicon/Intel) and Linux (including openwrt/musl edge router targets).
* **Execution Modes:** Interactive foreground CLI tool or persistent background service (`systemd` / `launchd`).

---

## 2. Architecture & Request Pipeline

```
[Agent Framework] 
       │
       ▼ (OpenAI Spec: :8080/v1/*)
┌────────────────────────────────────────────────────────┐
│ oxllm Core Engine                                      │
│                                                        │
│  ┌──────────────────┐      ┌────────────────────────┐  │
│  │   Axum Router    │ ────>│  In-Memory State Pool  │  │
│  └─────────┬────────┘      │  (Providers & Limits)  │  │
│            │               └────────────────────────┘  │
│            ▼                                           │
│  ┌──────────────────┐      ┌────────────────────────┐  │
│  │ Adaptive Failure │<──── │ OTel Exporter Backend  │  │
│  │    Loop (SSE)    │      └───────────┬────────────┘  │
│  └─────────┬────────┘                  │               │
└────────────┼───────────────────────────┼───────────────┘
             │                           │ (JSON over HTTP/4318)
             ▼                           ▼
   [Upstream Providers]          [otelite Collector]
(Groq, SambaNova, Cerebras...)
```

### Supported Endpoints
1.  `POST /v1/chat/completions` – Supports both standard JSON payloads and Server-Sent Events (SSE) streaming (`stream: true`).
2.  `POST /v1/embeddings` – Standard non-streaming batch vectors.
3.  `GET /v1/models` – Returns a consolidated virtual array of all models exposed by currently active and healthy upstream providers.
4.  `GET /status` – Administrative endpoint displaying per-provider circuit state, request counters, token volumes, and last request time. Localhost-restricted.
5. `POST /reload` – Administrative endpoint triggering config hot-reload via HTTP. Same effect as SIGHUP. Localhost-restricted.
6. `GET /health` – Lightweight health-check endpoint. Localhost-restricted.
7. `POST /admin/providers/{name}/offline` – Manually mark a provider as disabled. Localhost-restricted.
8. `POST /admin/providers/{name}/online` – Re-enable a manually disabled provider. Localhost-restricted.
9. `POST /admin/providers/{name}/reset` – Reset a provider's circuit-breaker state and failure counters. Localhost-restricted.

All public endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) return CORS headers (`Access-Control-Allow-Origin: *`) and include an `x-request-id` correlation header on every response.

---

## 3. Technical Specification

### 3.1. In-Memory State & Circuit Breaker Logic
The core application state wraps an active provider pool in an asynchronous read/write lock (`Arc<RwLock<Vec<ProviderState>>>`).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,                  // Healthy, accepting traffic
    Open { until: Instant }, // Tripped due to 429/5xx, bypassing provider
    HalfOpen,                // Cooldown expired, sending a single probe request
}

pub struct ProviderState {
    pub name: String,
    pub base_url: Url,
    pub api_key: String,
    pub models: Vec<String>,
    pub headers: HashMap<String, String>,
    pub circuit: Arc<RwLock<CircuitState>>,
    pub rate_limited_until: Arc<RwLock<Option<Instant>>>,
    pub consecutive_failures: Arc<RwLock<u32>>,
    pub last_attempt_time: Arc<RwLock<Option<Instant>>>,
    pub probe_in_flight: Arc<AtomicBool>,
    pub requests: AtomicU64,
    pub successes: AtomicU64,
    pub tokens_input: AtomicU64,
    pub tokens_output: AtomicU64,
    pub manual_disabled: AtomicBool, // v0.1.6: set via admin endpoints to manually disable a provider
}
```

#### Adaptive Logic & Circuit Breaker Rules:
1.  **The Handshake Guard**: When a request hits `/v1/chat/completions`, the engine loops through the configured provider array under a fast `RwLock::read` lock, selecting the first provider where `circuit` is `Closed` or is in `HalfOpen` (with no active `probe_in_flight`), and `rate_limited_until` is either `None` or in the past.
2.  **Thundering Herd Defense**: When a provider's `Open` cooldown expires:
    * The provider transitions to `HalfOpen`.
    * The first request that selects it flags `probe_in_flight = true` atomically via `.compare_exchange()`.
    * Any concurrent requests that hit the gateway *while* the probe is in-flight will bypass this provider and fall back to the next available provider.
    * If the probe succeeds: transition `circuit = CircuitState::Closed`, reset `consecutive_failures = 0`, and reset `probe_in_flight = false`.
    * If the probe fails: increment `consecutive_failures`, trip `circuit = CircuitState::Open { until: Instant::now() + longer_cooldown }`, and reset `probe_in_flight = false`.
3.  **Connection & Handshake Timeouts**: A strict upstream timeout (defaulting to 5 seconds) is established. If a provider takes longer than this to complete its initial connection or headers handshake, it is treated as a `5xx` connection failure, triggering immediate failover.
4.  **Initial Failure Handling (Reactive)**: If the initial connection handshake or headers return a `429 Too Many Requests` or `5xx Server Error`:
    * Parse the upstream response headers for `retry-after`.
    * Set `rate_limited_until = Instant::now() + extracted_duration`.
    * Increment `consecutive_failures`. If failures >= 3, set `circuit = CircuitState::Open { until: Instant::now() + Duration::from_secs(60) }`.
    * Immediately drop the connection to this upstream, advance the loop index, and dispatch to the next available provider.
5. **Mid-Stream Fallback**: If an upstream accepts the connection with a `200 OK` and fails mid-stream during an SSE event transfer, the proxy will transparently forward the disconnect to the downstream agent. As of v0.1.9, streaming success feedback is deferred until stream completion, and mid-stream failures ARE counted as failures (incrementing `consecutive_failures`). The proxy does not attempt to hot-swap upstreams mid-flight to avoid corrupted JSON token streams.

### 3.2. Configuration Schema & Hot Reloading
Configuration is declared in a single `config.toml` file (avoiding deprecated YAML dependencies).

```toml
# config.toml

[server]
host = "127.0.0.1"
port = 8080
otel_endpoint = "http://127.0.0.1:4318"
upstream_timeout_secs = 5 # Strict timeout for connecting & receiving upstream headers

[[providers]]
name = "google-ai-studio"
enabled = true
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key = "${AI_STUDIO_KEY}" # Supports raw string or environment variable mapping
models = ["gemini-2.5-flash"]

[[providers]]
name = "groq"
enabled = true
base_url = "https://api.groq.com/openai/v1"
api_key = "${GROQ_KEY}"
models = ["llama-4-scout", "deepseek-r1-distill"]

[[providers]]
name = "sambanova"
enabled = true
base_url = "https://api.sambanova.ai/v1"
api_key = "${SAMBANOVA_KEY}"
models = ["llama-3.3-70b-instruct"]

[virtual_models]
llama-3.3-70b = [
  { provider = "sambanova", model = "llama-3.3-70b-instruct" },
  { provider = "groq", model = "llama-3.3-70b-versatile" }
]
complex-free = [
  { provider = "groq", model = "deepseek-r1-distill" },
  { provider = "sambanova", model = "llama-3.3-70b-instruct" },
  { provider = "google-ai-studio", model = "gemini-2.5-flash" }
]
```

#### Custom Headers & Client Header Passthrough
Each provider can specify custom HTTP headers attached to upstream requests (`headers = { ... }` or `[providers.headers]`):

```toml
[[providers]]
name = "custom-provider"
enabled = true
base_url = "https://api.example.com/v1"
api_key = "${API_KEY}"
models = ["model-name"]
headers = { "User-Agent" = "custom-agent/1.0", "x-session-id" = "${SESSION_ID}" }
```

* **Provider Overrides**: Configured headers in `headers` take precedence over default headers (such as `Authorization` or `Content-Type`) and client-supplied headers.
* **Client Header Passthrough**: Incoming client headers starting with `x-` (e.g., `x-session-id`, `x-request-id`, `x-opencode-session`) are transparently forwarded to upstream providers unless explicitly overridden by provider configuration.
* **Environment Variable Expansion**: Header values support standard `${ENV_VAR}` expansion.
* **Validation**: Header names and values are verified during startup and via `oxllm validate`.

#### Reloading Mechanism
To keep the binary free of intensive filesystem polling threads:
* **POSIX Signal Handling:** The application listens for a `SIGHUP` signal.
* **HTTP Reload Endpoint:** A `POST /reload` endpoint (localhost-restricted) triggers the same logic.
* **Action:** Upon intercepting either trigger, the configuration file is re-parsed. New keys or target providers are mapped into a fresh `Vec<ProviderState>`, and the pointer is safely updated using a `tokio::sync::watch` channel, maintaining current uptime for connected clients.

---

## 4. Telemetry Layer & Trace Context Propagation

`oxllm` interacts with the `otelite` collector by pushing standard asynchronous OTLP/HTTP JSON payloads over port `4318`. 

### 4.1. W3C Trace Context Propagation
To enable seamless end-to-end trace auditing, `oxllm` participates in trace propagation:
* Extracts the incoming `traceparent` and `tracestate` HTTP headers from downstream client requests.
* **Root Context Synthesis** *(planned, not yet implemented)*: If the incoming request has no active `traceparent`, `oxllm` should generate a valid root trace context before forwarding to `otelite` to maintain absolute continuous tracking.
* Safely parses trace metadata via hex-decoding trace IDs (`[u8; 16]`) and span IDs (`[u8; 8]`), attaching context using `with_parent_context()`.
* Inject W3C Trace Context headers into upstream requests to the selected provider.
* Pushes standard parented spans to `otelite` so developers get continuous trace chains.

### 4.1.1. Request Correlation (`x-request-id`)
Every response includes an `x-request-id` header containing a unique correlation ID. This header is generated at the earliest point in the request pipeline and is present even when no upstream provider is reached (e.g., on validation errors or when all providers are unavailable). Clients can use this ID to correlate logs, traces, and responses.

### 4.2. Bounded Backpressure Safety
To protect against Out-Of-Memory (OOM) situations on memory-constrained edge routers, telemetry is funneled through a **bounded channel** (size `1024`). Telemetry is sent via a non-blocking `try_send` strategy. If the collector is down or lagging and the buffer fills up, telemetry payloads are gracefully dropped to prioritize routing stability over complete logs.

### 4.3. Metrics Framework
The agent must configure the following standard metrics:

* `llm_proxy.provider.status` (Gauge): Emits `0` for healthy, `1` for rate-limited cooldown, `2` for circuit-breaker tripped. Attributes: `provider.name`.
* `llm_proxy.request.duration` (Histogram): Total lifecycle duration of the transaction. Attributes: `provider.name`, `client.endpoint`.
* `llm_proxy.tokens.consumed` (Counter): Cumulative count of tokens processed. Attributes: `provider.name`, `model.name`, `type` (`input` or `output`).

### 4.4. Semantic Span Attributes
Every routed transaction generates an OpenTelemetry Span containing the official GenAI Semantic Conventions:

```json
{
  "trace_id": "...",
  "span_id": "...",
  "name": "oxllm.chat.completions",
  "attributes": {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "groq",
    "gen_ai.request.model": "llama-4-scout",
    "gen_ai.usage.input_tokens": 1420,
    "gen_ai.usage.output_tokens": 312,
    "proxy.attempts_required": 2,
    "proxy.initial_failure_reason": "429_tokens_exhausted"
  }
}
```

---

## 5. Execution & Lifecycle Management

### 5.1. Command Line Interface (CLI)
The binary supports clean POSIX subcommands for daemon management and operational debugging:
* `oxllm serve --config /path/to/config.toml` (Starts the proxy. Use `-v` for per-request routing info, `-vv` for full trace.)
* `oxllm validate --config /path/to/config.toml` (Parses configuration syntax, resolves environment variables, checks upstream network paths, then exits)
* `oxllm status` (Queries the running daemon locally over loopback and prints the virtual model routing table plus per-provider counters)
* `oxllm stop` (Gracefully stops the daemon via SIGTERM — drains in-flight SSE streams before exiting)
* `oxllm reload` (Finds the running `oxllm` daemon process and triggers SIGHUP immediately, or use `POST /reload` HTTP endpoint)

### 5.2. Admin Route Protection
To prevent external network actors from auditing provider credentials or configurations, all administrative endpoints (like `/status`, `/health`, `/reload`, and `/admin/providers/*`) are **localhost-restricted**. The `localhost_only` middleware accepts both IPv4 loopback (`127.0.0.1`) and IPv6-mapped IPv4 addresses (e.g., `::ffff:127.0.0.1`, added in v0.1.8). External callers receive an immediate `403 Forbidden`.

### 5.3. Cross-Origin Resource Sharing (CORS)
All public endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) return
CORS headers allowing cross-origin requests from any origin. This enables browser-based
OpenAI SDKs (JavaScript, Vercel AI SDK) to call the proxy directly.

### 5.4. Daemon Configuration (Service Mode)

#### macOS Lifecycle (`/Library/LaunchDaemons/org.oxllm.plist`)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>org.oxllm.proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/oxllm</string>
        <string>serve</string>
        <string>--config</string>
        <string>/etc/oxllm/config.toml</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/oxllm.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/oxllm.err</string>
</dict>
</plist>
```

#### Linux Lifecycle (`/etc/systemd/system/oxllm.service`)
```ini
[Unit]
Description=Oxide LLM Proxy
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/oxllm serve --config /etc/oxllm/config.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5s
MemoryMax=50M
CPUWeight=100

[Install]
WantedBy=multi-user.target
```
