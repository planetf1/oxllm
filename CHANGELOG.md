# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.17] - 2026-10-28

### Fixed
- **All 11 open RUSTSEC advisories cleared** via `cargo update`: quinn-proto
  0.11.18 (zero-length DATAGRAM/STREAM panics, unbounded connection-ID
  retirement, ACK-frequency panic, reassembly memory-exhaustion bypass),
  rustls 0.23.46+ (cross-level handshake acceptance, session-secret `Debug`
  leaks, Brotli certificate decompression allocation, KEM downgrade),
  rustls-webpki 0.103.16 (distrusted intermediates), and tokio 1.54.1
  (`AsyncFd` I/O-safety unsoundness).

## [0.1.16] - 2026-10-28

### Added
- **Session affinity**: requests with a session key (`x-session-id` header,
  configurable, or the body's `prompt_cache_key`) stick to one provider while
  it is healthy, keeping upstream prefix caches warm. Bounded LRU with idle
  TTL; pins break automatically on circuit trips and re-route by policy.

### Fixed
- **Route counters counted attempts, not requests**: a request that failed
  over three providers reported three requests, and the `failures` counter was
  never written at all. Requests are now counted once per client request and a
  terminal 502 increments the route's failures.

## [0.1.15] - 2026-10-28

### Added
- **Inbound API keys** (`server.api_keys`): public `/v1/*` routes can require
  `Authorization: Bearer <key>` (constant-time comparison). Empty list keeps
  the proxy open; admin routes stay localhost-only either way.
- **`cheapest` routing policy**: orders healthy candidates by configured
  `cost_per_mtok` (absent = free), latency breaking ties.
- **Per-provider `upstream_timeout_secs`**: overrides the global timeout so a
  fast-failing provider can fail over sooner.
- **Weighted rotation within latency bands**: equal-speed providers now share
  load instead of always hitting the first configured.
- **Cache-hit tracking**: `cached_tokens` from upstream usage is counted per
  provider/model and surfaced in `/status` (`oxllm status --json`).
- **`oxllm provider list --json`** emits the raw `/status` payload.

## [0.1.14] - 2026-10-28

- **Latency-aware routing** (`[policies]`, `"fastest"`): virtual models can
  order healthy candidates by measured EWMA time-to-first-byte instead of
  strict configuration order. 20% bands prevent flapping on noise, every 20th
  request rotates the order so demoted providers are still sampled, and
  unmeasured providers sort first so new additions are tried immediately.
- **Per-(provider, model) traffic attribution** in `/status`: two virtual
  models sharing one provider no longer show each other's counters. Route
  rows also carry TTFT and total-latency distributions (p50/p95/p99) from
  fixed-bucket histograms (relaxed atomics, no hot-path locking).
- **`oxllm status --json`** emits the raw `/status` payload for scripting;
  tables now adapt to terminal width (`$COLUMNS` or `--width`), dropping
  optional columns rather than wrapping.

### Fixed
- **OTLP export was silently dead**: the exporter had no HTTP client feature
  enabled, so spans and metrics were never sent (logged as a warning at
  startup since 0.1.12). Now uses `reqwest-blocking-client`, which is required
  because OTel's batch processor runs off the Tokio runtime.
- **Chat non-streaming successes were double-counted**: the per-provider
  counter was incremented before the response body was read and again after,
  which could report more successes than requests.

## [0.1.12] - 2026-06-01

### Fixed
- **x-request-id correlation fixed**: Request ID is now generated once in the
  middleware and propagated to route handlers via Axum extensions. The response
  header, log lines, and OTel span attribute now carry the same ID, enabling
  end-to-end request tracing.
- **502 fallback returns JSON error format**: The "all providers failed"
  response now uses `json_error_response()` with `{"error": ...}` shape,
  matching the OpenAI error format for the 502 case. Previously it returned
  plain text.
- **`json_error_response` no longer panics**: Changed signature to accept
  `StatusCode` directly instead of `u16`, and handles serialization failure
  gracefully with a plain-text fallback. Eliminates `.expect()` and `.unwrap()`
  in user-facing error paths.
- **Homebrew formula publish fixed**: `publish-homebrew-formula` job now depends
  on `build-global-artifacts` where cargo-dist generates the `.rb` formula file.

## [0.1.11] - 2026-06-01

### Added
- **OpenAI-compatible JSON error format**: All error responses (400, 502, 403) now
  return `{"error": {"message": ..., "type": ..., "code": ...}}` with
  `Content-Type: application/json`. Official OpenAI SDKs can parse errors correctly.
- **CORS headers**: `Access-Control-Allow-Origin: *` on all public endpoints
  (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`). Enables browser-based
  OpenAI SDKs (JavaScript, Vercel AI SDK) to call the proxy directly.
- **`x-request-id` correlation**: Every response now includes an `oxllm-<hex>`
  `x-request-id` header, visible in both success and error responses. The ID is
  also attached to all log lines and OTel spans (`proxy.request_id` attribute)
  for end-to-end request tracing.
- **Provider status OTel gauge**: `llm_proxy.provider.status` (0=Healthy, 1=Cooldown,
  2=Tripped) now emits on every circuit state transition. Previously the gauge was
  defined but never populated.
- **Unit & integration tests**: 7 new tests covering CORS preflight, JSON error
  format parsing, and `x-request-id` presence on both success and error responses.

### Changed
- `localhost_only` middleware returns JSON error body instead of empty 403.
- CORS layer registered globally, before all per-route middleware.
- All upstream-failure `warn!` log lines now include `request_id` field.

### Documentation
- Added CORS support, `x-request-id`, and JSON error format to README features list.
- Updated architecture doc with CORS subsection, request correlation docs, and
  updated middleware diagram.

## [0.1.10] - 2026-06-01

### Documentation
- Updated `docs/providers.md` to match the actual config.toml — removed Cerebras
  (not in config), updated Groq/SambaNova/OpenRouter model names to verified IDs,
  excluded Gemini 2.5 Pro (paid-only), added second SambaNova tier.
- Updated `docs/architecture.md` with missing endpoints (`/admin/providers/*`,
  `/health`), corrected rate-limit header parsing claim (only `Retry-After`),
  documented v0.1.9 mid-stream feedback deferral, marked root-context-synthesis
  as planned-not-implemented, added `manual_disabled` field to struct diagram,
  and updated admin-route protection to mention dual-stack IPv6 support.

## [0.1.9] - 2026-06-01

### Fixed
- Hot-reload race: replaced all `.find().unwrap()` calls on provider state lookups
  with graceful `let Some(...) else { continue; }` patterns. Concurrent SIGHUP reloads
  no longer panic when removing an in-flight provider.
- Upstream error body passthrough: when a provider returns a non-2xx status, the
  upstream error message is now captured and included in the final 502 response.
  Clients see actionable errors instead of a generic "all providers failed" message.
- Mid-stream feedback deferred: circuit breaker success feedback for streaming
  chat completions now fires AFTER the stream completes rather than on 200 OK.
  Mid-stream disconnections are counted as failures, preventing broken providers
  from staying marked healthy.

## [0.1.8] - 2026-06-01

### Fixed
- `oxllm serve` now uses the `host` config field for IPv4 binding instead of
  hardcoding `127.0.0.1`. Set `host = "0.0.0.0"` to accept connections from
  other machines. Admin and status routes remain protected by `localhost_only`
  middleware regardless of bind address.
- `localhost_only` middleware now correctly recognizes IPv4-mapped IPv6
  loopback addresses (`::ffff:127.0.0.0/104`). This fixes CLI `oxllm status`
  failures when the server is bound to a dual-stack `[::]` socket.

## [0.1.7] - 2026-06-01

### Fixed
- `oxllm --version` now reports the actual crate version from Cargo.toml
  instead of a hardcoded `0.1.0` string (broken since v0.1.5).
  Uses `env!("CARGO_PKG_VERSION")` via clap derive.

### Added
- Admin CLI commands now include `oxllm provider list|offline|online|reset`
  for runtime provider management without curl.
- `localhost_only` middleware protects admin/status/health/reload endpoints
  from non-loopback callers (403 Forbidden).


## [0.1.6] - 2026-05-30

### Added
- Admin API: `POST /admin/providers/{name}/offline|online|reset` — runtime provider management.
- CLI: `oxllm provider list|offline|online|reset` subcommands — manage providers without curl.
- `oxllm provider list` — condensed provider status table.
- Friendly error messages when server not running (all CLI commands).
- XDG config path support (`~/.config/oxllm/config.toml` with `./config.toml` fallback).
- Provider guide: `docs/providers.md` — free-tier services, model names verified live (2026-05-30).
- Token counting from upstream JSON responses (non-streaming).

### Fixed
- Corrected model names for all 6 providers (verified via each provider's `/v1/models` endpoint).
- Gemini 2.5 Pro excluded (paid-only on free tier, quota = 0).
- Table column widths widened to fit 45-character model names.
- Last Request column width fixed (12 chars).
- Cleaned up debug `println!` statements from error paths.

### Changed
- Model names: updated to verified values (e.g. `llama-4-scout` → `meta-llama/llama-4-scout-17b-16e-instruct`).
- Ollama defaults: `granite4:micro` → `granite4.1:3b`.
- Default config searches XDG path before current directory.

### Documentation
- Full README overhaul: binary size, routing algorithm, CLI examples, telemetry section.
- Provider guide with snapshot date and research methodology.
- API endpoint table includes admin routes.

## [0.1.5] - 2026-05-30

### Added
- `oxllm stop` subcommand — gracefully stops the daemon via SIGTERM.
- `oxllm serve -v` / `-vv` — verbosity flags for per-request routing info or full trace.
- `POST /reload` HTTP endpoint — trigger config reload without shell access.
- `bind_family` config option (`"ipv4"`, `"ipv6"`, `"dual"`) — dual-stack IPv4/IPv6 binding.
- Last request time per provider — shown in `/status` and `oxllm status` ("Just now", "5m ago", etc.).
- Virtual model routing table in `/status` — shows each virtual model's fallback chain with per-hop health and counters.
- Circuit transition logging at `info!` level — see when circuits open, close, or rate-limit.

### Changed
- Per-request routing logs demoted from `info!` to `debug!` — default output is now quiet (errors and circuit transitions only). Use `-v` to see routing decisions.
- PID file cleaned up on graceful shutdown.

### Documentation
- Installation section restructured: Homebrew first (easiest), then `cargo install`, then source build.
- Full `oxllm status` output sample in README showing virtual model routing table.

## [0.1.4] - 2026-05-30

### Added
- Local per-provider request/success/token counters visible via `GET /status` and `oxllm status` — no external collector needed.
- `upstream_timeout_secs` config field in `[server]` section (default 5 seconds).
- Multi-tier `config.toml` with `smart`/`basic` virtual models and local Ollama fallback.
- Token counting from upstream JSON responses (non-streaming).
- crates.io publish workflow (tag-triggered, idempotent).

### Fixed
- Removed `println!` debug statements from error paths.
- Removed invalid `crates-io` value from `dist-workspace.toml`.

### Documentation
- Overhauled README with endpoint table, status output example, and quick start guide.

## [0.1.3] - 2026-05-30

### Fixed
- Proxy no longer crashes at startup when the OTLP collector is unreachable — telemetry exporter failure now logs a `WARN` and falls back to a silent no-op drain loop.
- `base_url` must end with a trailing `/v1/` path so relative URL joins produce correct endpoints (e.g. `http://localhost:11434/v1/` for Ollama).

### Documentation
- Added Ollama local-only example config to README with full self-contained setup instructions.
- Documented `base_url` trailing-slash convention and optional telemetry behaviour.


## [0.1.2] - 2026-05-30

### Fixed
- Calibrated CI coverage thresholds to match actual coverage (workspace 43%, oxllm-core 55%, oxllm 36%) — thresholds now set ~3pp below measured values so regressions are caught without false failures.

## [0.1.1] - 2026-05-30

### Added
- Configured automated security audit compliance scanning in workflows.

## [0.1.0] - 2026-05-30

### Added
- Core OpenAI-compatible chat completions proxy with full SSE streaming support.
- Embeddings proxy route with automatic reactive failover.
- Adaptive Priority Routing Strategy supporting circuit breakers, exponential backoffs, and idle-based decay aging.
- Strict lock-free thundering herd permit shielding using atomic operations.
- POSIX signal SIGHUP reloader watcher to hot-swap configuration on the fly.
- Graceful shutdown logic that drains active streaming clients on SIGINT/SIGTERM.
- Backpressure-safe, bounded OpenTelemetry span and metrics pipeline.

## [0.1.13] - 2026-10-28

### Added
- **`POST /v1/completions`**: raw (non-chat) completions route for
  fill-in-the-middle clients such as editor autocomplete. Resolves virtual
  models, fails over across providers, accounts token usage, and honours
  `stream: true`.
- The SSE forwarding path is now shared by `/v1/chat/completions` and
  `/v1/completions` (`stream_upstream_body`), so both stream chunk-by-chunk
  with circuit-breaker feedback deferred until the stream ends.
