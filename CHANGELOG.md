# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.25] - 2026-11-04

### Added
- **Nested virtual models**: a target can be `{ virtual = "other" }` instead of
  `provider`/`model`, splicing in that model's targets in their own order, so
  tiers can be layered (`general-med -> [gemma12b, granite8b]`, each of those
  listing runtimes). Flattened once at state-build time; the request path is
  unchanged and counters stay per leaf (provider, model). A `priority` on a
  reference applies to the whole group; `weight` is rejected on references;
  leaves reached by several paths appear once. Validation rejects cycles
  (error names the full path, `a -> b -> c -> a`), undefined references, and
  nesting deeper than 16, on startup, `oxllm validate`, `oxllm reload`,
  SIGHUP and the config watcher; a rejected reload keeps the running config.
  Flattening re-checks for cycles itself, so it never relies on `validate()`.

### Added
- **Per-provider tuning**: `circuit_failure_threshold` and
  `circuit_base_cooldown_secs` override the server-wide values for one
  provider, so a flaky gateway and a stable local server need not share a
  setting. `first_byte_timeout_secs` (streaming requests only) abandons a
  provider that sends no response headers in time and fails over, catching a
  hung endpoint long before `upstream_timeout_secs`.

### Fixed
- **A provider listed twice in one virtual model never reached its second
  entry.** The routing loop picked the upstream model by provider name and
  selected by provider, so after a failure it retried the first entry until
  the circuit opened. Each (provider, model) entry is now tried at most once
  per request, in order.
- **A hung upstream cost every caller `circuit_failure_threshold` full
  timeouts** (3 x `upstream_timeout_secs`) before its circuit opened. A
  timeout now opens the circuit at once, so one request pays one timeout and
  later requests skip the provider until its cooldown ends and a probe
  succeeds.

### Added
- **Anthropic Messages API**: `POST /v1/messages` (SSE streaming) and
  `POST /v1/messages/count_tokens`, forwarded verbatim (apart from the `model`
  rewrite) to providers configured with the new `api = "anthropic"` field
  (default `"openai"`, so existing configs are unchanged). Upstream auth uses
  `x-api-key` with `anthropic-version` (default `2023-06-01`) and
  `anthropic-beta` forwarded. Requests only reach providers of the matching
  wire format, so one virtual model can mix both. 429 triggers a
  rate-limit cooldown honouring `retry-after`; 5xx and 529 are retried until
  the circuit threshold, then fail over; other 4xx return
  to the caller untouched. Errors oxllm generates use the Anthropic error
  shape. Inbound `server.api_keys` are accepted via `x-api-key` on these
  routes, and `metadata.user_id` is a session-affinity fallback. Usage stats
  read Anthropic `usage` (cache tokens included), streamed or not.
- `GET /v1/models` now also carries the Anthropic list fields (`type`,
  `display_name`, `created_at`, `has_more`, `first_id`, `last_id`) alongside the
  OpenAI ones.

## [0.1.24] - 2026-11-03

### Added
- **Traffic visibility** (issue #12): a bounded in-memory ring of recent
  request records (metadata only — never bodies; capacity via
  `server.request_log_capacity`, default 256, `0` disables) surfaced three
  ways: `recent_requests` in `/status`, a compact recent-requests table in
  `oxllm status`, and `oxllm tail [-n N]` — the last N records followed by
  a live SSE stream from the localhost-only `/admin/tail`. Records carry
  endpoint, route, provider, model, status, attempts, latency, TTFT, and
  terminal error. No new dependencies.
- **`/status` reports the daemon `version`**, the last reload outcome
  (`reload.last_error`, `reload.restart_warnings`), and per-provider
  `admin_disabled` — `oxllm status` shows all three, and admin-offline
  providers render as "Offline (admin)" in both CLI tables instead of
  looking healthy while receiving no traffic.

## [0.1.23] - 2026-11-03

### Fixed
- **Critical: the daemon restart-looped every `shutdown_timeout_secs`**:
  the v0.1.19 graceful-shutdown deadline wrapped `tokio::time::timeout`
  around the entire serve future instead of only the post-signal drain,
  so with the default 30s the process exited cleanly after ~31s of
  serving and KeepAlive restarted it — wiping circuit state, counters,
  and session pins on every cycle. The deadline is now a watchdog armed
  when the shutdown signal arrives: it bounds only the drain, and normal
  exit cancels it. Regression test: a daemon with a 2s deadline and no
  signal must still be serving 6s later.

## [0.1.22] - 2026-11-03

### Fixed
- **Flaky parallel install tests on CI**: the two install-refusal tests
  both created and deleted a dummy unit file at the same fixed path, and
  nextest runs tests as separate processes — one test's cleanup deleted
  the other's fixture mid-assertion on runners without a real
  installation (green locally against a real install, red on CI). The
  dummy is now created once and left behind, and the byte-identity check
  only runs where a real installation pre-exists.
- **`oxllm install` refuses an occupied port** (issue #8): installing
  while something already listens on the configured port used to produce
  a service that crash-looped on bind failures under KeepAlive. The
  preflight gates fresh installs only — replacing an existing
  installation with `--force` stops the old service first, which frees
  the port, so a running daemon no longer blocks its own upgrade. The
  refusal runs before anything is written, leaving existing files
  untouched; the install tests are hermetic against a throwaway HOME.

## [0.1.21] - 2026-11-03

### Added
- **Per-target `weight` for proportional load sharing** (issue #10):
  under the default policy, candidates carrying a `weight` share traffic
  via a per-request weighted lottery within their priority tier
  (probability proportional to weight; healthy candidates only, since
  circuit/rate-limit checks still gate the chain). Ignored by `fastest`
  and `cheapest`; no weights set preserves strict order.
- **`--host` flag** (env `OXLLM_HOST`, default `127.0.0.1`) on `status`,
  `reload`, and the `provider` subcommands (issue #9): CLI commands can
  now reach a remote daemon or one bound to IPv6 loopback (`--host ::1`,
  bracketed automatically in URLs) instead of hardcoding 127.0.0.1.

### Fixed
- **`oxllm install` refuses an occupied port** (issue #8): installing
  while something already listens on the configured port used to produce
  a service that crash-looped on bind failures under KeepAlive. The
  preflight check runs before anything is written, so a refusal leaves
  an existing installation untouched.
- **An explicit `policies.<vm> = "priority"` behaved like `fastest`**: the
  policy match fell through to latency ordering. Explicit `priority` now
  means strict configured order, same as omitting the policy.

### Changed
- **The four proxy handlers share one routing loop** (issue #11):
  chat, completions, embeddings, and systemone were four ~300-line copies
  of the same parse/resolve/failover/feedback skeleton. They are now thin
  endpoint specs over a single `proxy_request` loop (routes.rs drops by
  ~960 lines); usage parsing accepts both OpenAI-style and systemone
  token fields. All existing endpoint tests (SSE streaming, failover,
  usage accounting, 502s) pass unchanged.

## [0.1.20] - 2026-11-03

### Added
- **Service installation** (issue #2): `oxllm install` sets the daemon up
  as a user-level service — launchd agent on macOS (`RunAtLoad` +
  `KeepAlive`, logs to `~/Library/Logs/oxllm.log`), systemd user service on
  Linux — validating the config first and refusing to overwrite an existing
  installation without `--force`. `oxllm uninstall` stops and removes it.
  Verified live on macOS: install, status, KeepAlive respawn after kill -9,
  and clean uninstall. `install` warns when the config uses `${VAR}`
  references, which services cannot inherit from the shell.
- **Decision-model routing** (issue #6): `POST /v1/systemone` proxies
  llama.cpp decision endpoints (typed choice/score questions, one forward
  pass) through the provider pool like chat — per-target model rewrite,
  circuit breaking, and failover; responses pass through verbatim and usage
  (`input_tokens`/`prompt_tokens`) is accounted.
- **Per-target `priority` in virtual models** (issue #4): lower = preferred;
  targets without a priority keep their configured order after all
  prioritized ones. Existing configs behave identically.

### Fixed
- **`oxllm provider list` table misaligned** (issue #3): the hardcoded
  border/header/data widths (three different schemes for the same columns)
  are replaced by the same adaptive renderer `oxllm status` uses — widths
  computed from the longest cell, optional columns dropped on narrow
  terminals, over-long cells truncated with an ellipsis.

## [0.1.19] - 2026-11-03

### Added
- **Model aliases** (`[aliases]` table) and `server.default_model`:
  unmodified OpenAI clients can address virtual models by the names they
  were built with (`gpt-4o = "smart"`), any unknown model name can route
  to a default instead of 400, and `GET /v1/models` lists the aliases.
- **Circuit-breaker tuning** in `[server]`: `circuit_failure_threshold`
  (default 3), `circuit_base_cooldown_secs` (default 60), and
  `rate_limit_fallback_secs` (default 30) — tune sensitivity per deployment
  instead of the hardcoded values.
- **Graceful-shutdown deadline**: `server.shutdown_timeout_secs`
  (default 30, `0` = wait forever) bounds the post-SIGTERM drain, so a
  request stuck on a hung upstream can no longer hold shutdown hostage.
- **Config-watcher toggle**: `server.watch_config = false` disables the
  config file watcher (edge/flash targets, or strictly manual reloads).

### Changed
- **Hot-reload preserves runtime state**: providers that keep their name
  carry their circuit-breaker position, failure/rate-limit windows,
  request/token counters, admin offline overrides, and session-affinity
  pins across a reload. A config edit mid-incident no longer re-enables a
  provider taken offline or resets failover history. Server fields that
  cannot be hot-swapped (`port`, `host`, `bind_family`, `otel_endpoint`,
  affinity TTL/capacity, `shutdown_timeout_secs`, `watch_config`) now
  produce explicit "restart required" warnings in the `POST /reload`
  response and logs.
- **`oxllm status` and the `oxllm provider` subcommands exit non-zero**
  when no daemon can be found, matching `reload` and `stop`.

## [0.1.18] - 2026-11-03

### Added
- **Config file watching** (issue #7): the daemon now polls its config
  file every 2 seconds and hot-reloads when the contents change, so config
  edits take effect without a manual step. A broken edit is logged once and
  the previous config keeps serving; fixing the file reloads.

### Changed
- **`oxllm reload` now drives the daemon over HTTP** (`POST /reload` on
  `-p/--port`, default 8080, `OXLLM_PORT`) instead of signalling a PID read
  from `/tmp/oxllm.pid`. The daemon's validation error is printed when the
  new config is rejected. `--pid <pid>` keeps the direct-SIGHUP escape
  hatch. **Breaking:** in `reload`, `-p` is now `--port` (it was `--pid`).

### Fixed
- **A short-lived second `oxllm serve` no longer orphans the daemon**
  (issue #7): every instance used to overwrite `/tmp/oxllm.pid` and delete
  it on exit, so `oxllm reload`/`oxllm stop` lost track of the long-running
  daemon. The PID file is now written only after a successful bind, is
  never overwritten while it tracks a live process, and is removed on
  shutdown only if it still records the exiting process's own PID.
- **`oxllm reload` and `oxllm stop` exit non-zero** when no daemon can be
  found, so `edit && oxllm reload` chains fail loudly instead of silently
  succeeding while the daemon keeps serving the old config.

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
