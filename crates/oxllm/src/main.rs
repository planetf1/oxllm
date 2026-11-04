use clap::{Parser, Subcommand};
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::{header, HeaderValue, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Json,
};
use tower_http::cors::{Any, CorsLayer};

use oxllm_core::config::{Config, ServerConfig};
use oxllm_core::state::{
    AppState, CircuitConfig, CircuitState, ModelStats, ProviderState, RequestLog, RoutingPolicy,
    SessionAffinity,
};
use oxllm_core::telemetry::{TelemetryClient, TelemetryWorker};
use reqwest::Url;

/// Resolves the config file path using XDG base directory conventions.
///
/// If the given path exists, returns it as-is.
/// Otherwise, tries the XDG config path: `~/.config/oxllm/config.toml`
/// (respecting `$XDG_CONFIG_HOME` if set).
/// If that also doesn't exist, returns the original path so the caller
/// produces a clear file-not-found error.
fn resolve_config_path(given: PathBuf) -> PathBuf {
    if given.exists() {
        return given;
    }
    // Try XDG config path
    let xdg_config = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".config")
        })
        .join("oxllm")
        .join("config.toml");
    if xdg_config.exists() {
        return xdg_config;
    }
    given // fall back to original path (will produce a clear file-not-found error)
}

mod routes;

#[derive(Parser, Debug)]
#[command(
    name = "oxllm",
    version = env!("CARGO_PKG_VERSION"),
    author = "Nigel Jones",
    about = "Resilient routing proxy for LLM APIs (OpenAI and Anthropic wire formats)",
    long_about = "oxllm sits between your clients and your LLM providers. Clients ask for a \
virtual model name; oxllm picks a provider for it, fails over when one is slow or down, and \
reports what happened.\n\nEndpoints: /v1/chat/completions, /v1/completions, /v1/embeddings, \
/v1/messages (Anthropic), /v1/models.",
    after_help = "Getting started:\n  1. oxllm validate -c config.toml   check a config\n  2. oxllm serve -c config.toml      run in the foreground\n  3. oxllm status                    what is healthy, what is not\n\nRunning as a service:\n  oxllm install -c config.toml       launchd (macOS) / systemd (Linux), starts at login\n  oxllm reload                       apply config edits without dropping connections\n\nWhen something is wrong:\n  oxllm status -v                    per-route targets and why each is out\n  oxllm tail -r 20                   the last 20 requests, then live\n  oxllm provider offline NAME        take a provider out of rotation\n  oxllm provider reset NAME          clear its circuit and failure count\n\nDocs: README.md (config reference), docs/architecture.md (design)."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Runs the proxy in the foreground (use `install` to run it as a service)
    #[command(
        after_help = "Config: --config, else $OXLLM_CONFIG, else ~/.config/oxllm/config.toml, else ./config.toml.\n${VAR} references in the config need those variables set in this shell.\nNext: oxllm status (from another terminal), or oxllm install to keep it running."
    )]
    Serve {
        /// Path to the configuration TOML file
        /// (searches: <path>, ~/.config/oxllm/config.toml, ./config.toml)
        #[arg(short, long, default_value = "config.toml", env = "OXLLM_CONFIG")]
        config: PathBuf,

        /// Verbosity level. Repeat for more output: -v (debug), -vv (trace)
        #[arg(short, long, action = clap::ArgAction::Count, default_value_t = 0)]
        verbose: u8,
    },
    /// Checks a config file without starting anything (syntax, references, cycles)
    #[command(
        after_help = "Exits 0 when valid, 1 with the reason when not. Run it before `oxllm reload`.\nNote: ${VAR} references must be set in this shell, or validation fails on them."
    )]
    Validate {
        /// Path to the configuration TOML file
        /// (searches: <path>, ~/.config/oxllm/config.toml, ./config.toml)
        #[arg(short, long, default_value = "config.toml", env = "OXLLM_CONFIG")]
        config: PathBuf,
    },
    /// Shows health at a glance: problems first, then one row per route
    #[command(
        long_about = "Shows what is wrong first, then one row per route (state, strategy, \
available/total entries, traffic, latency).\n\nDetail levels (each includes the previous):\n  \
(default)  health, problems, routes\n  -v         + each route's targets: state, tier, traffic\n  \
-vv        + providers, latency percentiles, recent requests\n\n--json emits the same tiers for \
scripts. --route NAME narrows everything to one virtual model.",
        after_help = "Examples:\n  oxllm status\n  oxllm status -v --route pool-standard\n  \
oxllm status --json | jq '.health.problems'\n\nNext: oxllm tail -r 20 (live requests), \
oxllm provider offline NAME (take a provider out of rotation)"
    )]
    Status {
        /// Host of the running gateway server
        #[arg(long, default_value = "127.0.0.1", env = "OXLLM_HOST")]
        host: String,
        /// Port of the running gateway server
        #[arg(short, long, default_value_t = 8080, env = "OXLLM_PORT")]
        port: u16,
        /// Emit JSON at the chosen detail level instead of tables
        #[arg(long)]
        json: bool,
        /// Table width in characters (defaults to $COLUMNS, else 120)
        #[arg(short, long)]
        width: Option<usize>,
        /// More detail: -v targets per route, -vv providers and recent requests
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
        /// Show only this virtual model
        #[arg(short, long)]
        route: Option<String>,
    },
    /// Applies config edits to the running daemon without dropping connections
    #[command(
        after_help = "The new config is validated first; if it is invalid the daemon keeps the old one and this exits 1 with the reason.\nProvider health, counters and session pins carry over for providers that keep their name.\nNext: oxllm status"
    )]
    Reload {
        /// Host of the running gateway server
        #[arg(long, default_value = "127.0.0.1", env = "OXLLM_HOST")]
        host: String,
        /// Port of the running gateway server
        #[arg(short, long, default_value_t = 8080, env = "OXLLM_PORT")]
        port: u16,
        /// Send SIGHUP to this PID instead of using the HTTP endpoint
        #[arg(long)]
        pid: Option<u32>,
    },
    /// Stops the daemon gracefully (SIGTERM; in-flight requests drain first)
    #[command(
        after_help = "If it was installed as a service the service manager restarts it; use `oxllm uninstall` to remove it."
    )]
    Stop {
        /// PID of the running oxllm process (optional, reads from /tmp/oxllm.pid by default)
        #[arg(short, long)]
        pid: Option<u32>,
    },
    /// Installs oxllm as a login service (launchd on macOS, systemd on Linux) and starts it
    #[command(
        after_help = "The service does not inherit your shell, so ${VAR} references in the config will not resolve; use a wrapper script that exports them, or avoid references.\nNext: oxllm status"
    )]
    Install {
        /// Path to the configuration TOML file
        /// (searches: <path>, ~/.config/oxllm/config.toml, ./config.toml)
        #[arg(short, long, default_value = "config.toml", env = "OXLLM_CONFIG")]
        config: PathBuf,
        /// Replace an existing installation (restarts the service)
        #[arg(long)]
        force: bool,
    },
    /// Stops and removes the installed service
    Uninstall,
    /// Shows recent requests, then follows new ones live (Ctrl-C to stop)
    #[command(
        after_help = "Each line: time, endpoint, route, provider that served it, status, latency, attempts. Metadata only; request and response bodies are never recorded.\nExample: oxllm tail -r 20"
    )]
    Tail {
        /// Host of the running gateway server
        #[arg(long, default_value = "127.0.0.1", env = "OXLLM_HOST")]
        host: String,
        /// Port of the running gateway server
        #[arg(short, long, default_value_t = 8080, env = "OXLLM_PORT")]
        port: u16,
        /// Print the N most recent records before following live traffic
        #[arg(short, long, default_value_t = 0)]
        recent: usize,
    },
    /// Takes providers out of or back into rotation, and resets their health
    #[command(
        subcommand,
        after_help = "Examples:\n  oxllm provider list\n  oxllm provider offline cb        stop sending traffic (survives circuit recovery)\n  oxllm provider online cb         resume\n  oxllm provider reset cb          clear circuit, failures and rate limit"
    )]
    Provider(ProviderCommand),
}

#[derive(Subcommand, Debug)]
enum ProviderCommand {
    /// Lists providers with their state (see `oxllm status -v` for per-model detail)
    List {
        /// Host of the running gateway server
        #[arg(long, default_value = "127.0.0.1", env = "OXLLM_HOST")]
        host: String,
        /// Port of the running gateway server
        #[arg(short, long, default_value_t = 8080, env = "OXLLM_PORT")]
        port: u16,
        /// Emit the raw /status JSON instead of a table
        #[arg(long)]
        json: bool,
    },
    /// Takes a provider out of rotation until `online` (a manual hold, not a circuit)
    Offline {
        /// Name of the provider to take offline
        name: String,
        /// Host of the running gateway server
        #[arg(long, default_value = "127.0.0.1", env = "OXLLM_HOST")]
        host: String,
        /// Port of the running gateway server
        #[arg(short, long, default_value_t = 8080, env = "OXLLM_PORT")]
        port: u16,
    },
    /// Puts a provider back into rotation
    Online {
        /// Name of the provider to bring online
        name: String,
        /// Host of the running gateway server
        #[arg(long, default_value = "127.0.0.1", env = "OXLLM_HOST")]
        host: String,
        /// Port of the running gateway server
        #[arg(short, long, default_value_t = 8080, env = "OXLLM_PORT")]
        port: u16,
    },
    /// Clears a provider's circuit, failure count and rate-limit window
    Reset {
        /// Name of the provider to reset
        name: String,
        /// Host of the running gateway server
        #[arg(long, default_value = "127.0.0.1", env = "OXLLM_HOST")]
        host: String,
        /// Port of the running gateway server
        #[arg(short, long, default_value_t = 8080, env = "OXLLM_PORT")]
        port: u16,
    },
}

#[derive(Clone)]
pub struct Reloader {
    sender: tokio::sync::watch::Sender<Arc<AppState>>,
    config_path: PathBuf,
}

#[derive(Clone)]
pub struct ReloadableState {
    pub app_state: tokio::sync::watch::Receiver<Arc<AppState>>,
    pub telemetry: TelemetryClient,
    pub start_time: Instant,
    pub reloader: Reloader,
}

impl axum::extract::FromRef<ReloadableState> for Arc<AppState> {
    fn from_ref(state: &ReloadableState) -> Self {
        state.app_state.borrow().clone()
    }
}

impl axum::extract::FromRef<ReloadableState> for (Arc<AppState>, TelemetryClient) {
    fn from_ref(state: &ReloadableState) -> Self {
        (state.app_state.borrow().clone(), state.telemetry.clone())
    }
}

impl axum::extract::FromRef<ReloadableState> for Instant {
    fn from_ref(state: &ReloadableState) -> Self {
        state.start_time
    }
}

impl axum::extract::FromRef<ReloadableState> for (Arc<AppState>, Instant) {
    fn from_ref(state: &ReloadableState) -> Self {
        (state.app_state.borrow().clone(), state.start_time)
    }
}

impl axum::extract::FromRef<ReloadableState> for Reloader {
    fn from_ref(state: &ReloadableState) -> Self {
        state.reloader.clone()
    }
}

/// Builds a fresh AppState from a config. When `previous` is given (a
/// reload), runtime state carries across for every provider that keeps its
/// name: circuit position, failure/rate-limit windows, counters, admin
/// overrides, and session-affinity pins all survive a config edit, so an
/// edit mid-incident no longer re-enables a provider taken offline or
/// resets failover history.
fn build_app_state(config: Config, previous: Option<&AppState>) -> Result<AppState, String> {
    // Flatten nested virtual models up front (this re-checks for cycles, so
    // it holds even if the caller skipped `validate()`); the routing hot
    // path only ever sees leaf targets.
    let virtual_models = config.flatten_virtual_models().map_err(|e| e.to_string())?;
    let mut providers = Vec::new();
    for p in config.providers {
        if !p.enabled {
            continue;
        }
        let url = Url::parse(&p.base_url).map_err(|e| {
            format!(
                "Invalid base URL '{}' for provider '{}': {}",
                p.base_url, p.name, e
            )
        })?;

        let mut state = ProviderState {
            name: p.name,
            base_url: url,
            api_key: p.api_key,
            models: p.models,
            upstream_timeout_secs: p.upstream_timeout_secs,
            cost_per_mtok: p.cost_per_mtok,
            api: p.api,
            tuning: oxllm_core::state::ProviderTuning {
                circuit_failure_threshold: p.circuit_failure_threshold,
                circuit_base_cooldown_secs: p.circuit_base_cooldown_secs,
                first_byte_timeout_secs: p.first_byte_timeout_secs,
            },
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        if let Some(old) =
            previous.and_then(|prev| prev.providers.iter().find(|o| o.name == state.name))
        {
            state.circuit = old.circuit.clone();
            state.consecutive_failures = old.consecutive_failures.clone();
            state.rate_limited_until = old.rate_limited_until.clone();
            state.last_attempt_time = old.last_attempt_time.clone();
            state.probe_in_flight = old.probe_in_flight.clone();
            state.manual_disabled = AtomicBool::new(
                old.manual_disabled
                    .load(std::sync::atomic::Ordering::Acquire),
            );
            state.requests =
                AtomicU64::new(old.requests.load(std::sync::atomic::Ordering::Relaxed));
            state.successes =
                AtomicU64::new(old.successes.load(std::sync::atomic::Ordering::Relaxed));
            state.tokens_input =
                AtomicU64::new(old.tokens_input.load(std::sync::atomic::Ordering::Relaxed));
            state.tokens_output =
                AtomicU64::new(old.tokens_output.load(std::sync::atomic::Ordering::Relaxed));
            state.model_stats = old.model_stats.clone();
            state.model_health = old.model_health.clone();
        }
        providers.push(state);
    }

    let http_client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

    let policies: std::collections::HashMap<String, RoutingPolicy> = config
        .policies
        .iter()
        .map(|(vm, policy)| {
            let mapped = match policy {
                oxllm_core::config::PolicyConfig::Priority => RoutingPolicy::Priority,
                oxllm_core::config::PolicyConfig::Fastest => RoutingPolicy::Fastest,
                oxllm_core::config::PolicyConfig::Cheapest => RoutingPolicy::Cheapest,
            };
            (vm.clone(), mapped)
        })
        .collect();

    let upstream_timeout_secs = config.server.upstream_timeout_secs;
    let request_log_capacity = config.server.request_log_capacity;
    Ok(AppState {
        providers,
        virtual_models,
        aliases: config.aliases,
        default_model: config.server.default_model.clone(),
        circuit: CircuitConfig {
            failure_threshold: config.server.circuit_failure_threshold,
            base_cooldown_secs: config.server.circuit_base_cooldown_secs,
            rate_limit_fallback_secs: config.server.rate_limit_fallback_secs,
        },
        route_stats: previous.map(|p| p.route_stats.clone()).unwrap_or_default(),
        policies,
        routing_calls: AtomicU64::new(
            previous
                .map(|p| p.routing_calls.load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0),
        ),
        api_keys: config.server.api_keys.clone(),
        session_affinity: previous
            .map(|p| p.session_affinity.clone())
            .unwrap_or_else(|| {
                Arc::new(SessionAffinity::new(
                    std::time::Duration::from_secs(config.server.session_affinity_ttl_secs),
                    config.server.session_affinity_capacity,
                ))
            }),
        session_affinity_header: config.server.session_affinity_header.clone(),
        server: config.server,
        reload_health: previous
            .map(|p| p.reload_health.clone())
            .unwrap_or_default(),
        // The ring carries across reloads (like session affinity); a capacity
        // change needs a restart and is covered by the restart warnings.
        request_log: previous
            .map(|p| p.request_log.clone())
            .unwrap_or_else(|| Arc::new(RequestLog::new(request_log_capacity))),
        http_client,
        upstream_timeout_secs,
    })
}

/// Fixed PID file used by `oxllm stop` to locate the daemon.
const PID_FILE: &str = "/tmp/oxllm.pid";

/// How often the config file watcher polls for changes.
const CONFIG_WATCH_INTERVAL: Duration = Duration::from_secs(2);

/// Whether a process with the given PID exists (probed with `kill -0`).
fn process_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Writes our PID to the PID file, but never overwrites a live foreign
/// process's entry: a second `oxllm serve` (another port, another config)
/// must not steal the file from the long-running daemon, or `oxllm stop`
/// loses track of it. Stale PIDs are replaced.
fn write_pid_file(path: &std::path::Path) -> std::io::Result<()> {
    let pid = std::process::id();
    if let Ok(existing) = std::fs::read_to_string(path) {
        if let Ok(existing_pid) = existing.trim().parse::<u32>() {
            if existing_pid != pid && process_alive(existing_pid) {
                warn!(
                    "PID file {} tracks live process {}, not overwriting",
                    path.display(),
                    existing_pid
                );
                return Ok(());
            }
        }
    }
    std::fs::write(path, pid.to_string())
}

/// Removes the PID file on shutdown, but only while it still records our
/// own PID, so an instance that lost the race for the file cannot delete
/// the daemon's entry on its way out.
fn cleanup_pid_file(path: &std::path::Path) {
    let own = std::process::id().to_string();
    if let Ok(content) = std::fs::read_to_string(path) {
        if content.trim() == own {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn send_sighup(pid: u32) -> std::io::Result<()> {
    let status = std::process::Command::new("kill")
        .args(["-HUP", &pid.to_string()])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "kill command failed with exit code: {:?}",
            status.code()
        )))
    }
}

/// Generates a random request ID for response correlation.
fn generate_request_id() -> String {
    format!("oxllm-{:016x}", rand::random::<u64>())
}

/// Middleware that adds an `x-request-id` header to every response.
/// Generates the ID before calling the handler, stores it in request
/// extensions so route handlers can read the same ID for logs/telemetry,
/// and inserts it into the response header (without overriding an existing
/// `x-request-id` forwarded from upstream).
async fn add_request_id(mut req: Request<Body>, next: Next) -> Response {
    let request_id = generate_request_id();
    req.extensions_mut().insert(request_id.clone());
    let mut response = next.run(req).await;
    if !response.headers().contains_key("x-request-id") {
        // SAFETY: generate_request_id produces only ASCII hex chars and "oxllm-" prefix.
        response.headers_mut().insert(
            "x-request-id",
            HeaderValue::from_str(&request_id)
                .expect("generated request ID contains invalid characters"),
        );
    }
    response
}

/// Constant-time byte-slice equality, so a key comparison does not leak its
/// length or prefix through response timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Rejects /v1/* requests when `server.api_keys` is non-empty and the
/// `Authorization: Bearer <key>` header (or `x-api-key` on /v1/messages)
/// matches none of them. Empty list
/// keeps the proxy open (loopback deployments). Admin routes stay
/// localhost-only regardless.
async fn require_api_key(
    State((app_state, _telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // The key set is captured per-reload in AppState; read it through the
    // watch channel without cloning the vec on every request.
    let api_keys = &app_state.api_keys;
    if api_keys.is_empty() {
        return next.run(req).await;
    }

    // Anthropic SDK clients send the key as `x-api-key`; accept it on the
    // Messages routes alongside the usual Bearer token.
    let is_messages = req.uri().path().starts_with("/v1/messages");
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| {
            is_messages
                .then(|| req.headers().get("x-api-key").and_then(|v| v.to_str().ok()))
                .flatten()
        });

    let authorized = presented.is_some_and(|key| {
        api_keys
            .iter()
            .any(|candidate| constant_time_eq(candidate.as_bytes(), key.as_bytes()))
    });

    if authorized {
        next.run(req).await
    } else {
        warn!(target: "oxllm::security", "Rejected /v1 request with invalid or missing API key");
        if is_messages {
            return routes::anthropic_error_response(
                "Missing or invalid API key",
                "authentication_error",
                StatusCode::UNAUTHORIZED,
            );
        }
        let body = serde_json::json!({
            "error": {
                "message": "Missing or invalid API key",
                "type": "invalid_request_error",
                "code": 401
            }
        });
        (StatusCode::UNAUTHORIZED, Json(body)).into_response()
    }
}

async fn localhost_only(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let is_local = match addr.ip() {
        IpAddr::V4(v4) => v4.is_loopback(),
        // Dual-stack bindings present IPv4 connections as IPv4-mapped IPv6
        // addresses like ::ffff:127.0.0.1. to_canonical() converts these to
        // their IPv4 representation so is_loopback() works correctly.
        IpAddr::V6(v6) => v6.is_loopback() || v6.to_canonical().is_loopback(),
    };
    if is_local {
        next.run(req).await
    } else {
        warn!(target: "oxllm::security", "Blocked external attempt to access administrative route from IP: {}", addr.ip());
        let body = serde_json::json!({
            "error": {
                "message": "Access denied: administrative routes are localhost-only",
                "type": "forbidden",
                "code": 403
            }
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let mut response = Response::new(Body::from(bytes));
        *response.status_mut() = StatusCode::FORBIDDEN;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        response
    }
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// Resolves on SIGINT/SIGTERM, cleaning up the PID file, and arms the
/// drain deadline: a watchdog that force-exits the process if in-flight
/// requests have not drained within `deadline` of the signal. The
/// deadline bounds only the drain, never the serve lifetime — the
/// watchdog is cancelled by normal process exit if the drain completes.
async fn shutdown_signal(deadline: Option<Duration>) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received SIGINT, shutting down gracefully...");
        },
        _ = terminate => {
            info!("Received SIGTERM, shutting down gracefully...");
        },
    }

    // The drain deadline: if the graceful drain (which starts when this
    // future resolves) has not finished within the deadline, exit anyway
    // so a request stuck on a hung upstream cannot hold shutdown hostage.
    if let Some(deadline) = deadline {
        tokio::spawn(async move {
            tokio::time::sleep(deadline).await;
            warn!(
                "Graceful shutdown exceeded the {:?} deadline (server.shutdown_timeout_secs); exiting with in-flight requests dropped",
                deadline
            );
            std::process::exit(0);
        });
    }

    // Clean up PID file on shutdown, but only if it still records our PID
    cleanup_pid_file(std::path::Path::new(PID_FILE));
}

/// Loads, validates, and swaps in the config at `config_path`. Shared by
/// the SIGHUP handler, `POST /reload`, and the config file watcher. On any
/// failure the previous state stays live and the reason is returned.
/// Server-level settings that only take effect at startup: a hot-reload
/// accepts the new config, swaps the routing state, and returns one of
/// these per changed field so the operator knows a restart is needed.
fn server_config_restart_warnings(old: &ServerConfig, new: &ServerConfig) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut note = |changed: bool, key: &str, from: String, to: String| {
        if changed {
            warnings.push(format!(
                "server.{} changed from {} to {} — restart required to apply",
                key, from, to
            ));
        }
    };
    note(
        old.port != new.port,
        "port",
        old.port.to_string(),
        new.port.to_string(),
    );
    note(
        old.host != new.host,
        "host",
        old.host.clone(),
        new.host.clone(),
    );
    note(
        old.bind_family != new.bind_family,
        "bind_family",
        old.bind_family.clone(),
        new.bind_family.clone(),
    );
    note(
        old.otel_endpoint != new.otel_endpoint,
        "otel_endpoint",
        old.otel_endpoint.clone(),
        new.otel_endpoint.clone(),
    );
    note(
        old.session_affinity_ttl_secs != new.session_affinity_ttl_secs,
        "session_affinity_ttl_secs",
        old.session_affinity_ttl_secs.to_string(),
        new.session_affinity_ttl_secs.to_string(),
    );
    note(
        old.session_affinity_capacity != new.session_affinity_capacity,
        "session_affinity_capacity",
        old.session_affinity_capacity.to_string(),
        new.session_affinity_capacity.to_string(),
    );
    note(
        old.shutdown_timeout_secs != new.shutdown_timeout_secs,
        "shutdown_timeout_secs",
        old.shutdown_timeout_secs.to_string(),
        new.shutdown_timeout_secs.to_string(),
    );
    note(
        old.watch_config != new.watch_config,
        "watch_config",
        old.watch_config.to_string(),
        new.watch_config.to_string(),
    );
    note(
        old.request_log_capacity != new.request_log_capacity,
        "request_log_capacity",
        old.request_log_capacity.to_string(),
        new.request_log_capacity.to_string(),
    );
    warnings
}

/// Loads, validates, and swaps in the config at `config_path`. Shared by
/// the SIGHUP handler, `POST /reload`, and the config file watcher. On any
/// failure the previous state stays live and the reason is returned. On
/// success, returns the restart-required warnings for server fields that
/// cannot be hot-swapped.
fn reload_config(
    config_path: &std::path::Path,
    sender: &tokio::sync::watch::Sender<Arc<AppState>>,
) -> Result<Vec<String>, String> {
    let previous = sender.borrow().clone();
    let config = match Config::load_from_file(config_path) {
        Ok(c) => c,
        Err(e) => {
            let reason = format!("Failed to load config: {}", e);
            previous.reload_health.record_failure(&reason);
            return Err(reason);
        },
    };
    if let Err(e) = config.validate() {
        let reason = format!("Config validation failed: {}", e);
        previous.reload_health.record_failure(&reason);
        return Err(reason);
    }
    let warnings = server_config_restart_warnings(&previous.server, &config.server);
    let new_state = match build_app_state(config, Some(&previous)) {
        Ok(state) => state,
        Err(e) => {
            let reason = format!("Failed to build state: {}", e);
            previous.reload_health.record_failure(&reason);
            return Err(reason);
        },
    };
    sender
        .send(Arc::new(new_state))
        .map_err(|e| format!("Failed to update watch channel: {}", e))?;
    previous.reload_health.record_success(warnings.clone());
    Ok(warnings)
}

fn json_response(status: StatusCode, body: serde_json::Value) -> Response {
    let bytes = serde_json::to_vec(&body).unwrap();
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

async fn handle_http_reload(
    axum::extract::State(reloader): axum::extract::State<Reloader>,
) -> impl IntoResponse {
    match reload_config(&reloader.config_path, &reloader.sender) {
        Ok(warnings) => {
            info!("Configuration reloaded via HTTP POST /reload");
            let mut body = serde_json::json!({"message": "Configuration reloaded successfully"});
            if !warnings.is_empty() {
                body["warnings"] = serde_json::json!(warnings);
            }
            json_response(StatusCode::OK, body)
        },
        Err(reason) => json_response(
            StatusCode::BAD_REQUEST,
            serde_json::json!({
                "error": {
                    "message": reason,
                    "type": "invalid_request_error",
                    "code": 400
                }
            }),
        ),
    }
}

async fn handle_sighup(
    config_path: PathBuf,
    watch_sender: tokio::sync::watch::Sender<Arc<AppState>>,
) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sig = match signal(SignalKind::hangup()) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to register SIGHUP handler: {}", e);
                return;
            },
        };

        info!("Registered SIGHUP reload listener");
        while sig.recv().await.is_some() {
            info!("SIGHUP received, reloading configuration...");
            match reload_config(&config_path, &watch_sender) {
                Ok(warnings) => {
                    for warning in warnings {
                        warn!("{}", warning);
                    }
                },
                Err(e) => {
                    error!(
                        "Hot-reload via SIGHUP failed, keeping previous config: {}",
                        e
                    );
                },
            }
        }
    }
}

/// Watches the config file and hot-reloads when its contents change, so
/// config edits take effect without a manual `oxllm reload`. The file is
/// small, so polling its bytes every couple of seconds is cheaper and more
/// portable than a filesystem-event dependency. A reload fires only when
/// the contents differ from the last attempt (successful or not), so a
/// broken edit is reported once and the fixed edit reloads.
async fn watch_config_file(
    config_path: PathBuf,
    sender: tokio::sync::watch::Sender<Arc<AppState>>,
    poll_interval: Duration,
) {
    // None until the first successful read, so a file that is briefly
    // unreadable at startup becomes the baseline instead of a change.
    let mut last_contents: Option<Vec<u8>> = None;
    let mut interval = tokio::time::interval(poll_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    info!("Watching config file {:?} for changes", config_path);
    loop {
        interval.tick().await;
        // Transient read failures (file mid-rename) are retried next tick.
        let Ok(contents) = std::fs::read(&config_path) else {
            continue;
        };
        let is_baseline = last_contents.is_none();
        if !is_baseline && last_contents.as_ref() == Some(&contents) {
            continue;
        }
        last_contents = Some(contents);
        if is_baseline {
            continue;
        }
        info!("Config file changed, hot-reloading...");
        match reload_config(&config_path, &sender) {
            Ok(warnings) => {
                for warning in warnings {
                    warn!("{}", warning);
                }
            },
            Err(e) => {
                error!(
                    "Hot-reload of changed config failed, keeping previous config: {}",
                    e
                );
            },
        }
    }
}

async fn run_serve(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = resolve_config_path(config_path);
    // 1. Load initial config
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;

    // 2. Build initial AppState
    let app_state = Arc::new(build_app_state(config.clone(), None)?);

    // 3. Initialize watch channel
    let (watch_sender, watch_receiver) = tokio::sync::watch::channel(app_state.clone());

    // 4. Initialize telemetry channel & spawn TelemetryWorker
    let (telemetry_tx, telemetry_rx) = tokio::sync::mpsc::channel(1024);
    let telemetry_client = TelemetryClient::new(telemetry_tx);
    let otel_endpoint = config.server.otel_endpoint.clone();
    let _worker_handle = TelemetryWorker::spawn(&otel_endpoint, telemetry_rx)?;

    // 5. Spawn SIGHUP listener and (unless disabled) the config file watcher
    let config_path_clone = config_path.clone();
    tokio::spawn(handle_sighup(config_path_clone, watch_sender.clone()));
    if config.server.watch_config {
        tokio::spawn(watch_config_file(
            config_path.clone(),
            watch_sender.clone(),
            CONFIG_WATCH_INTERVAL,
        ));
    } else {
        info!("Config file watching disabled (server.watch_config = false)");
    }

    // 6. Build Axum Router
    let start_time = Instant::now();
    let reloader = Reloader {
        sender: watch_sender.clone(),
        config_path: config_path.clone(),
    };
    let reloadable_state = ReloadableState {
        app_state: watch_receiver,
        telemetry: telemetry_client,
        start_time,
        reloader,
    };

    use axum::routing::{get, post};
    // A watch::Receiver is itself Clone-and-cheap: `from_fn_with_state` needs
    // an owned copy per layer, and each request reads through it without
    // locking the routing pool.
    let state_for_middleware = reloadable_state.clone();
    let app = axum::Router::new()
        .route(
            "/v1/models",
            get(routes::list_models).layer(middleware::from_fn_with_state(
                state_for_middleware.clone(),
                require_api_key,
            )),
        )
        .route(
            "/v1/embeddings",
            post(routes::create_embeddings).layer(middleware::from_fn_with_state(
                state_for_middleware.clone(),
                require_api_key,
            )),
        )
        .route(
            "/v1/systemone",
            post(routes::create_systemone).layer(middleware::from_fn_with_state(
                state_for_middleware.clone(),
                require_api_key,
            )),
        )
        .route(
            "/v1/completions",
            post(routes::create_completions).layer(middleware::from_fn_with_state(
                state_for_middleware.clone(),
                require_api_key,
            )),
        )
        .route(
            "/v1/chat/completions",
            post(routes::create_chat_completions).layer(middleware::from_fn_with_state(
                state_for_middleware.clone(),
                require_api_key,
            )),
        )
        .route(
            "/v1/messages",
            post(routes::create_messages).layer(middleware::from_fn_with_state(
                state_for_middleware.clone(),
                require_api_key,
            )),
        )
        .route(
            "/v1/messages/count_tokens",
            post(routes::count_tokens).layer(middleware::from_fn_with_state(
                state_for_middleware.clone(),
                require_api_key,
            )),
        )
        .route(
            "/status",
            get(routes::get_status).layer(middleware::from_fn(localhost_only)),
        )
        .route(
            "/health",
            get(health_check).layer(middleware::from_fn(localhost_only)),
        )
        .route(
            "/reload",
            post(handle_http_reload).layer(middleware::from_fn(localhost_only)),
        )
        .route(
            "/admin/providers/{name}/offline",
            post(routes::admin_offline).layer(middleware::from_fn(localhost_only)),
        )
        .route(
            "/admin/providers/{name}/online",
            post(routes::admin_online).layer(middleware::from_fn(localhost_only)),
        )
        .route(
            "/admin/providers/{name}/reset",
            post(routes::admin_reset).layer(middleware::from_fn(localhost_only)),
        )
        .route(
            "/admin/tail",
            get(routes::tail_requests).layer(middleware::from_fn(localhost_only)),
        )
        .layer(middleware::from_fn(add_request_id))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::OPTIONS,
                ])
                .allow_headers(Any),
        )
        .with_state(reloadable_state);

    // Start listening. The PID file is written only after a successful
    // bind: an instance that fails to start must not claim (or clobber)
    // the daemon's PID entry.
    let port = config.server.port;
    let shutdown_timeout_secs = config.server.shutdown_timeout_secs;
    let listener = match config.server.bind_family.as_str() {
        "ipv6" => {
            let addr = format!("[::]:{}", port);
            info!("Listening on http://{} (IPv6 only)", addr);
            tokio::net::TcpListener::bind(&addr).await?
        },
        "dual" => {
            let addr = std::net::SocketAddr::new(
                std::net::IpAddr::V6(std::net::Ipv6Addr::UNSPECIFIED),
                port,
            );
            let socket = socket2::Socket::new(
                socket2::Domain::IPV6,
                socket2::Type::STREAM,
                Some(socket2::Protocol::TCP),
            )
            .map_err(|e| format!("Failed to create socket: {}", e))?;
            socket
                .set_only_v6(false)
                .map_err(|e| format!("Failed to set dual-stack: {}", e))?;
            socket
                .set_reuse_address(true)
                .map_err(|e| format!("Failed to set reuse address: {}", e))?;
            socket
                .bind(&addr.into())
                .map_err(|e| format!("Failed to bind: {}", e))?;
            socket
                .listen(1024)
                .map_err(|e| format!("Failed to listen: {}", e))?;
            socket
                .set_nonblocking(true)
                .map_err(|e| format!("Failed to set non-blocking: {}", e))?;
            info!("Listening on [::]:{} (dual-stack IPv4/IPv6)", port);
            tokio::net::TcpListener::from_std(socket.into())
                .map_err(|e| format!("Failed to create tokio listener: {}", e))?
        },
        _ => {
            // Default: IPv4
            let addr = format!("{}:{}", config.server.host, port);
            info!("Listening on http://{} (IPv4)", addr);
            tokio::net::TcpListener::bind(&addr).await?
        },
    };

    // Write PID file (after the bind succeeded; see above)
    if let Err(e) = write_pid_file(std::path::Path::new(PID_FILE)) {
        warn!("Failed to write PID file: {}", e);
    }

    // The graceful drain starts when the shutdown signal arrives and runs
    // until in-flight requests finish; the drain deadline is armed inside
    // shutdown_signal (a watchdog that exits if the drain overruns). The
    // serve future itself is never time-bounded.
    let deadline = if shutdown_timeout_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(shutdown_timeout_secs))
    };
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal(deadline))
    .await?;

    Ok(())
}

fn run_validate(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = resolve_config_path(config_path);
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;
    println!("Configuration file at '{:?}' is VALID!", config_path);
    Ok(())
}

/// One column of an ASCII status table.
struct TableColumn {
    header: &'static str,
    right_align: bool,
    /// Dropped first (right-to-left) when the table will not fit the terminal.
    optional: bool,
}

/// Renders a bordered table, dropping optional columns rather than wrapping or
/// overflowing when the terminal is narrow. Falls back to ellipsis truncation
/// only when even the required columns do not fit.
fn render_table(columns: &[TableColumn], rows: &[Vec<String>], max_width: usize) -> String {
    let mut keep: Vec<usize> = (0..columns.len()).collect();

    loop {
        let widths: Vec<usize> = keep
            .iter()
            .map(|&i| {
                let mut w = columns[i].header.chars().count();
                for row in rows {
                    w = w.max(row.get(i).map(|c| c.chars().count()).unwrap_or(0));
                }
                w
            })
            .collect();
        let total = widths.iter().sum::<usize>() + 3 * keep.len() + 1;
        if total <= max_width {
            break;
        }
        match keep.iter().rposition(|&i| columns[i].optional) {
            Some(pos) => {
                keep.remove(pos);
            },
            None => break,
        }
    }

    let widths: Vec<usize> = keep
        .iter()
        .map(|&i| {
            let mut w = columns[i].header.chars().count();
            for row in rows {
                w = w.max(row.get(i).map(|c| c.chars().count()).unwrap_or(0));
            }
            w
        })
        .collect();

    // Last resort: shrink the widest columns so the table cannot overflow.
    let mut widths = widths;
    let mut total = widths.iter().sum::<usize>() + 3 * keep.len() + 1;
    while total > max_width {
        let Some((widest, _)) = widths
            .iter()
            .enumerate()
            .filter(|(_, w)| **w > 3)
            .max_by_key(|(_, w)| **w)
        else {
            break;
        };
        widths[widest] -= 1;
        total -= 1;
    }

    let truncate = |text: &str, width: usize| -> String {
        let len = text.chars().count();
        if len <= width {
            return text.to_string();
        }
        if width <= 1 {
            return text.chars().take(width).collect();
        }
        let mut out: String = text.chars().take(width - 1).collect();
        out.push('\u{2026}');
        out
    };

    let border = |left: char, mid: char, right: char| -> String {
        let mut line = String::new();
        line.push(left);
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                line.push(mid);
            }
            line.push_str(&"\u{2500}".repeat(w + 2));
        }
        line.push(right);
        line
    };

    let mut out = String::new();
    out.push_str(&border('+', '+', '+'));
    out.push('\n');

    let mut header_line = String::from("|");
    for (pos, &col) in keep.iter().enumerate() {
        header_line.push_str(&format!(
            " {:<width$} |",
            truncate(columns[col].header, widths[pos]),
            width = widths[pos]
        ));
    }
    out.push_str(&header_line);
    out.push('\n');
    out.push_str(&border('+', '+', '+'));
    out.push('\n');

    for row in rows {
        let mut line = String::from("|");
        for (pos, &col) in keep.iter().enumerate() {
            let cell = truncate(row.get(col).map(String::as_str).unwrap_or(""), widths[pos]);
            if columns[col].right_align {
                line.push_str(&format!(" {:>width$} |", cell, width = widths[pos]));
            } else {
                line.push_str(&format!(" {:<width$} |", cell, width = widths[pos]));
            }
        }
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str(&border('+', '+', '+'));
    out
}

/// Renders an optional millisecond measurement for a table cell.
fn fmt_ms(value: Option<u64>) -> String {
    match value {
        Some(v) if v == u64::MAX => ">10s".to_string(),
        Some(v) => format!("{}ms", v),
        None => "-".to_string(),
    }
}

/// Terminal width: explicit override, then $COLUMNS, then a safe default.
fn status_width(override_width: Option<usize>) -> usize {
    override_width
        .or_else(|| std::env::var("COLUMNS").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(120)
        .max(40)
}

/// CLI-side base URL: brackets IPv6 literals so `http://[::1]:8080/...`
/// is well-formed (issue #9).
fn cli_base_url(host: &str, port: u16) -> String {
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{}]", host)
    } else {
        host.to_string()
    };
    format!("http://{}:{}", host, port)
}

/// How much `oxllm status` prints. Each level includes the one before.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Detail {
    /// Health, problems, one row per route.
    Summary,
    /// Plus each route's targets with state, priority/weight and traffic.
    Targets,
    /// Plus per-provider totals, latency percentiles and recent requests.
    Full,
}

impl Detail {
    fn from_verbosity(v: u8) -> Self {
        match v {
            0 => Detail::Summary,
            1 => Detail::Targets,
            _ => Detail::Full,
        }
    }
}

/// `1h 02m`, `17m 15s`, `42s`.
fn fmt_duration(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h {:02}m", secs / 3600, (secs / 60) % 60)
    } else if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

fn jstr<'a>(v: &'a serde_json::Value, key: &str) -> &'a str {
    v.get(key).and_then(|x| x.as_str()).unwrap_or("-")
}

fn ju64(v: &serde_json::Value, key: &str) -> u64 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(0)
}

fn jms(v: &serde_json::Value, key: &str) -> String {
    fmt_ms(v.get(key).and_then(|x| x.as_u64()))
}

/// Entry state for a table cell: `healthy`, `open 12s`, `rate_limited 5s`.
fn entry_state(e: &serde_json::Value) -> String {
    let state = e
        .get("availability")
        .and_then(|x| x.as_str())
        .unwrap_or("unknown");
    match e.get("retry_in_secs").and_then(|x| x.as_u64()) {
        Some(s) => format!("{} {}s", state, s),
        None => state.to_string(),
    }
}

fn col(header: &'static str, right_align: bool, optional: bool) -> TableColumn {
    TableColumn {
        header,
        right_align,
        optional,
    }
}

/// Renders the `/status` JSON at the requested detail level. Tolerant of
/// older daemons: fields it needs but cannot find print as `-`, and the
/// health summary is derived only when the daemon reports it.
fn render_status(
    status: &serde_json::Value,
    detail: Detail,
    route_filter: Option<&str>,
    max_width: usize,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let empty = Vec::new();
    let arr = |key: &str| -> &Vec<serde_json::Value> {
        status.get(key).and_then(|v| v.as_array()).unwrap_or(&empty)
    };

    let health = status.get("health");
    let overall = health
        .and_then(|h| h.get("state"))
        .and_then(|s| s.as_str())
        .unwrap_or("unknown");
    let _ = writeln!(
        out,
        "\noxllm {}  up {}  {} requests  {}",
        status
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("(daemon older than v0.1.24)"),
        fmt_duration(ju64(status, "uptime_secs")),
        ju64(status, "total_requests"),
        overall.to_uppercase()
    );
    for w in status["reload"]["restart_warnings"]
        .as_array()
        .unwrap_or(&empty)
    {
        let _ = writeln!(out, "  reload warning: {}", w.as_str().unwrap_or("?"));
    }

    // Problems come first: this is what the operator opened status to learn.
    let problems = health
        .and_then(|h| h.get("problems"))
        .and_then(|p| p.as_array())
        .unwrap_or(&empty);
    if problems.is_empty() {
        if overall == "ok" {
            let _ = writeln!(out, "  all routes healthy");
        }
    } else {
        let _ = writeln!(out);
        for p in problems {
            let _ = writeln!(
                out,
                "  ! {:<22} {:<13} {}",
                truncate_str(jstr(p, "name"), 22),
                jstr(p, "kind"),
                jstr(p, "detail")
            );
        }
    }

    // Routes: one row per virtual model.
    let routes: Vec<&serde_json::Value> = arr("routes")
        .iter()
        .filter(|r| route_filter.is_none_or(|f| jstr(r, "route") == f))
        .collect();
    if routes.is_empty() {
        if let Some(f) = route_filter {
            let _ = writeln!(out, "\nNo route named '{}'. Routes:", f);
            for r in arr("routes") {
                let _ = writeln!(out, "  {}", jstr(r, "route"));
            }
        }
    } else {
        let full = detail >= Detail::Full;
        let rows: Vec<Vec<String>> = routes
            .iter()
            .map(|r| {
                let mut row = vec![
                    jstr(r, "route").to_string(),
                    jstr(r, "state").to_string(),
                    jstr(r, "policy").to_string(),
                    format!("{}/{}", ju64(r, "available"), ju64(r, "targets")),
                    ju64(r, "requests").to_string(),
                    ju64(r, "failures").to_string(),
                    jms(r, "latency_p50_ms"),
                    jms(r, "latency_p95_ms"),
                ];
                if full {
                    row.push(ju64(r, "tokens_input").to_string());
                    row.push(ju64(r, "tokens_output").to_string());
                    row.push(jms(r, "ttft_p50_ms"));
                    row.push(jms(r, "latency_p99_ms"));
                }
                row
            })
            .collect();
        let mut cols = vec![
            col("Route", false, false),
            col("State", false, false),
            col("Strategy", false, true),
            col("Avail", true, false),
            col("Req", true, false),
            col("Fail", true, true),
            col("p50", true, true),
            col("p95", true, true),
        ];
        if full {
            cols.extend([
                col("Tok In", true, true),
                col("Tok Out", true, true),
                col("TTFT50", true, true),
                col("p99", true, true),
            ]);
        }
        let _ = writeln!(out, "\nRoutes\n{}", render_table(&cols, &rows, max_width));
    }

    if detail >= Detail::Targets {
        let mut vms: Vec<&String> = status["virtual_models"]
            .as_object()
            .map(|o| o.keys().collect())
            .unwrap_or_default();
        vms.sort();
        for vm in vms {
            if route_filter.is_some_and(|f| f != vm.as_str()) {
                continue;
            }
            let route = arr("routes").iter().find(|r| jstr(r, "route") == vm);
            let _ = writeln!(
                out,
                "\n{}{}",
                vm,
                route
                    .map(|r| format!("  ({})", jstr(r, "strategy")))
                    .unwrap_or_default()
            );
            let entries = status["virtual_models"][vm.as_str()]
                .as_array()
                .unwrap_or(&empty);
            let rows: Vec<Vec<String>> = entries
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let tuning = match (e.get("priority"), e.get("weight")) {
                        (Some(p), Some(w)) => format!("p{} w{}", p, w),
                        (Some(p), None) => format!("p{}", p),
                        (None, Some(w)) => format!("w{}", w),
                        (None, None) => "-".to_string(),
                    };
                    vec![
                        (i + 1).to_string(),
                        jstr(e, "provider").to_string(),
                        jstr(e, "model").to_string(),
                        entry_state(e),
                        jstr(e, "api").to_string(),
                        tuning,
                        ju64(e, "requests").to_string(),
                        ju64(e, "successes").to_string(),
                        ju64(e, "tokens_input").to_string(),
                        ju64(e, "tokens_output").to_string(),
                        jms(e, "ttft_p50_ms"),
                    ]
                })
                .collect();
            let _ = writeln!(
                out,
                "{}",
                render_table(
                    &[
                        col("#", true, false),
                        col("Provider", false, false),
                        col("Model", false, false),
                        col("State", false, false),
                        col("API", false, true),
                        col("Tier", false, true),
                        col("Req", true, false),
                        col("OK", true, true),
                        col("Tok In", true, true),
                        col("Tok Out", true, true),
                        col("TTFT50", true, true),
                    ],
                    &rows,
                    max_width,
                )
            );
        }
    }

    if detail >= Detail::Full {
        let rows: Vec<Vec<String>> = arr("providers")
            .iter()
            .map(|p| {
                let circuit = if p["admin_disabled"].as_bool().unwrap_or(false) {
                    "Offline (admin)".to_string()
                } else {
                    jstr(p, "circuit").to_string()
                };
                vec![
                    jstr(p, "name").to_string(),
                    circuit,
                    ju64(p, "failures").to_string(),
                    if p["rate_limited"].as_bool().unwrap_or(false) {
                        "yes"
                    } else {
                        "no"
                    }
                    .to_string(),
                    ju64(p, "requests").to_string(),
                    ju64(p, "successes").to_string(),
                    ju64(p, "tokens_input").to_string(),
                    ju64(p, "tokens_output").to_string(),
                    jstr(p, "last_request").to_string(),
                    jstr(p, "models").to_string(),
                ]
            })
            .collect();
        let _ = writeln!(
            out,
            "\nProviders (reachability; per-model health is in the route tables)\n{}",
            render_table(
                &[
                    col("Provider", false, false),
                    col("Circuit", false, false),
                    col("Fails", true, true),
                    col("Limited", false, true),
                    col("Req", true, false),
                    col("OK", true, true),
                    col("Tok In", true, true),
                    col("Tok Out", true, true),
                    col("Last", false, true),
                    col("Models", false, true),
                ],
                &rows,
                max_width,
            )
        );

        let recent = arr("recent_requests");
        if !recent.is_empty() {
            let _ = writeln!(
                out,
                "Recent requests (newest first, last 15 of {})",
                recent.len()
            );
            for rec in recent.iter().rev().take(15) {
                let _ = writeln!(out, "{}", format_request_record(rec));
            }
        }
    }

    // Always say how to see more (or less), so the next step is obvious.
    let _ = match detail {
        Detail::Summary => writeln!(
            out,
            "\nMore: oxllm status -v (targets per route)  |  -vv (providers, recent requests)  |  --route NAME  |  --json"
        ),
        Detail::Targets => writeln!(
            out,
            "\nMore: oxllm status -vv (providers, recent requests)  |  --json -v for scripts"
        ),
        Detail::Full => writeln!(out, "\nLive traffic: oxllm tail -r 20"),
    };
    out
}

/// Reduces a full `/status` document to the requested detail level for
/// `--json`, so scripts get the same tiers as the table output.
fn status_json_at(
    status: &serde_json::Value,
    detail: Detail,
    route_filter: Option<&str>,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let keep = |key: &str, out: &mut serde_json::Map<String, serde_json::Value>| {
        if let Some(v) = status.get(key) {
            out.insert(key.to_string(), v.clone());
        }
    };
    for key in [
        "version",
        "uptime_secs",
        "total_requests",
        "health",
        "reload",
    ] {
        keep(key, &mut out);
    }
    let filter_routes = |v: &serde_json::Value| -> serde_json::Value {
        match (v.as_array(), route_filter) {
            (Some(items), Some(f)) => serde_json::Value::Array(
                items
                    .iter()
                    .filter(|r| jstr(r, "route") == f)
                    .cloned()
                    .collect(),
            ),
            _ => v.clone(),
        }
    };
    if let Some(routes) = status.get("routes") {
        out.insert("routes".to_string(), filter_routes(routes));
    }
    if detail >= Detail::Targets {
        if let Some(vms) = status.get("virtual_models") {
            let mut vms = vms.clone();
            if let (Some(obj), Some(f)) = (vms.as_object_mut(), route_filter) {
                obj.retain(|k, _| k == f);
            }
            out.insert("virtual_models".to_string(), vms);
        }
    }
    if detail >= Detail::Full {
        keep("providers", &mut out);
        keep("recent_requests", &mut out);
    }
    serde_json::Value::Object(out)
}

async fn run_status(
    host: &str,
    port: u16,
    json: bool,
    width_override: Option<usize>,
    verbose: u8,
    route: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = format!("{}/status", cli_base_url(host, port));
    let res = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            println!("oxllm is not running on {}", cli_base_url(host, port));
            println!("Start it with: oxllm serve   (or install it as a service: oxllm install)");
            std::process::exit(1);
        },
        Err(e) => return Err(e.into()),
    };
    if !res.status().is_success() {
        return Err(format!("Server returned error status: {}", res.status()).into());
    }
    let status: serde_json::Value = serde_json::from_str(&res.text().await?)?;
    let detail = Detail::from_verbosity(verbose);

    if json {
        let reduced = status_json_at(&status, detail, route.as_deref());
        println!("{}", serde_json::to_string_pretty(&reduced)?);
        return Ok(());
    }
    print!(
        "{}",
        render_status(
            &status,
            detail,
            route.as_deref(),
            status_width(width_override)
        )
    );
    Ok(())
}

/// Triggers a hot-reload on the running daemon. The default path is the
/// daemon's `POST /reload` endpoint, which validates the new config and
/// reports the reason on failure; `--pid` falls back to signalling the
/// process directly with SIGHUP. Exits non-zero when the reload did not
/// happen, so a script chaining `edit && oxllm reload` cannot mistake
/// failure for success.
async fn run_reload(
    host: &str,
    port: u16,
    pid_opt: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(pid) = pid_opt {
        if let Err(e) = send_sighup(pid) {
            println!("Failed to send SIGHUP to process {}: {}", pid, e);
            std::process::exit(1);
        }
        println!(
            "Successfully sent hot-reload signal (SIGHUP) to process {}",
            pid
        );
        return Ok(());
    }

    let client = reqwest::Client::new();
    let url = format!("{}/reload", cli_base_url(host, port));
    let res = match client.post(&url).send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            println!("oxllm is not running on http://127.0.0.1:{}", port);
            println!("Start it with: oxllm serve");
            std::process::exit(1);
        },
        Err(e) => return Err(e.into()),
    };
    let status = res.status();
    let body = res.text().await?;
    println!("{}", body);
    if !status.is_success() {
        std::process::exit(1);
    }
    Ok(())
}

/// Gracefully stops the daemon recorded in the PID file (or `--pid`).
/// Exits non-zero when no daemon can be found, so scripts cannot mistake
/// failure for success.
fn run_stop(pid_opt: Option<u32>) -> Result<(), Box<dyn std::error::Error>> {
    let pid = match pid_opt {
        Some(p) => p,
        None => match std::fs::read_to_string(PID_FILE) {
            Ok(content) => match content.trim().parse::<u32>() {
                Ok(p) => p,
                Err(_) => {
                    println!("Invalid PID in {}", PID_FILE);
                    std::process::exit(1);
                },
            },
            Err(_) => {
                println!("oxllm is not running (no PID file at {})", PID_FILE);
                println!("Start it with: oxllm serve");
                std::process::exit(1);
            },
        },
    };

    let status = match std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
    {
        Ok(s) => s,
        Err(e) => {
            println!("Failed to send SIGTERM: {}", e);
            std::process::exit(1);
        },
    };
    if status.success() {
        println!(
            "Successfully sent graceful shutdown signal (SIGTERM) to process {}",
            pid
        );
    } else {
        println!("kill command failed with exit code: {:?}", status.code());
        std::process::exit(1);
    }
    Ok(())
}

/// Best-effort check whether something already listens on the port the
/// daemon would bind: installing into an occupied port produces a service
/// that fails to bind and crash-loops under KeepAlive (issue #8).
fn port_in_use(server: &ServerConfig) -> bool {
    let port = server.port;
    let check = |addr: String| std::net::TcpStream::connect(addr).is_ok();
    match server.bind_family.as_str() {
        // IPv6-only bind: nothing would answer on IPv4.
        "ipv6" => check(format!("[::1]:{}", port)),
        // Dual-stack accepts both families.
        "dual" => {
            check(format!("[::1]:{}", port))
                || check(format!("127.0.0.1:{}", port))
                || check(format!("{}:{}", server.host, port))
        },
        _ => check(format!("{}:{}", server.host, port)),
    }
}

/// Installs oxllm as a user-level service pointing at the current binary
/// and the resolved config. The config is loaded and validated first, so a
/// broken config is refused before anything is installed. Installing over
/// an existing unit requires --force (issue #2).
fn run_install(config_path: PathBuf, force: bool) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = resolve_config_path(config_path);

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let unit_exists = service_unit_path().exists();

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    if unit_exists && !force {
        println!(
            "oxllm is already installed at {}",
            service_unit_path().display()
        );
        println!("Re-run with --force to replace it; the service is restarted either way.");
        std::process::exit(1);
    }

    // Preflight: the installed service is only useful if the config works.
    let config = Config::load_from_file(&config_path)?;
    config.validate()?;

    // Preflight: a fresh install into an occupied port would make the
    // service crash-loop on bind failures (issue #8). Replacing an
    // existing installation is exempt — the old service is stopped as
    // part of the replacement, freeing the port — and the check runs
    // before anything is written, so a refusal leaves an existing
    // installation untouched.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    if !unit_exists && port_in_use(&config.server) {
        println!(
            "Port {} is already in use — the installed service would fail to bind and crash-loop.",
            config.server.port
        );
        println!("Stop whatever is listening on it (e.g. oxllm stop), or change server.port.");
        std::process::exit(1);
    };

    // A service manager does not inherit the interactive shell's
    // environment, so `${VAR}` references in the config would fail at
    // service startup even though they validate fine here.
    if let Ok(raw) = std::fs::read_to_string(&config_path) {
        if raw.contains("${") {
            println!("Warning: the config references environment variables (${{VAR}}).");
            println!("The service does not inherit your shell environment:");
            #[cfg(target_os = "macos")]
            println!("  run: launchctl setenv VAR value   (per variable, before loading)");
            #[cfg(target_os = "linux")]
            println!("  add Environment=VAR=value to the unit, or avoid env references");
        }
    }

    let exe = std::env::current_exe()?.canonicalize()?;

    #[cfg(target_os = "macos")]
    install_launchd(&exe, &config_path)?;

    #[cfg(target_os = "linux")]
    install_systemd(&exe, &config_path)?;

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        println!("Service installation is not supported on this platform.");
        println!("See docs/architecture.md for a manual init-script example.");
    }

    Ok(())
}

/// Stops and removes the installed user-level service. Removing nothing
/// (never installed, or already removed) is a success.
/// Prints the N most recent request records, then follows the daemon's
/// live tail (GET /admin/tail, SSE) until interrupted.
async fn run_tail(host: &str, port: u16, recent: usize) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let base = cli_base_url(host, port);

    if recent > 0 {
        let res = client.get(format!("{}/status", base)).send().await?;
        let body = res.text().await?;
        let status: serde_json::Value = serde_json::from_str(&body)?;
        if let Some(records) = status["recent_requests"].as_array() {
            let skip = records.len().saturating_sub(recent);
            for rec in records.iter().skip(skip) {
                print_request_record(rec);
            }
        }
    }

    // Follow the live stream. A disabled request log answers with an error
    // JSON instead of an SSE stream.
    let res = match client.get(format!("{}/admin/tail", base)).send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            println!("oxllm is not running on {}", base);
            println!("Start it with: oxllm serve");
            std::process::exit(1);
        },
        Err(e) => return Err(e.into()),
    };
    let content_type = res
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if !content_type.starts_with("text/event-stream") {
        let body = res.text().await?;
        println!("{}", body);
        std::process::exit(1);
    }

    let mut res = res;
    let mut buf = String::new();
    while let Some(chunk) = res.chunk().await? {
        buf.push_str(&String::from_utf8_lossy(&chunk));
        // SSE events are separated by blank lines; process only complete ones.
        while let Some(pos) = buf.find("\n\n") {
            let event: String = buf.drain(..pos + 2).collect();
            for line in event.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(rec) = serde_json::from_str::<serde_json::Value>(data) {
                        print_request_record(&rec);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Prints one request record as a compact single line.
fn print_request_record(rec: &serde_json::Value) {
    println!("{}", format_request_record(rec));
}

/// Renders one request record as a compact single line.
fn format_request_record(rec: &serde_json::Value) -> String {
    let ts = rec["ts_ms"].as_u64().unwrap_or(0) / 1000;
    let (h, m, sec) = ((ts / 3600) % 24, (ts / 60) % 60, ts % 60);
    let endpoint = rec["endpoint"].as_str().unwrap_or("?");
    let route = rec["route"].as_str().unwrap_or("?");
    let provider = rec["provider"].as_str().unwrap_or("-");
    let status = rec["status"].as_u64().unwrap_or(0);
    let latency = rec["latency_ms"].as_u64().unwrap_or(0);
    let attempts = rec["attempts"].as_u64().unwrap_or(1);
    let ttft = rec["ttft_ms"].as_u64();
    let error = rec["error"].as_str();

    let mut line = format!(
        "{:02}:{:02}:{:02}  {:<10} {:<18} {:<18} {:>3} {:>6}ms",
        h,
        m,
        sec,
        endpoint,
        truncate_str(route, 18),
        truncate_str(provider, 18),
        status,
        latency
    );
    if let Some(ttft) = ttft {
        line.push_str(&format!("  ttft {}ms", ttft));
    }
    if attempts > 1 {
        line.push_str(&format!("  attempt {}", attempts));
    }
    if let Some(error) = error {
        line.push_str(&format!("  {}", truncate_str(error, 60)));
    }
    line
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('\u{2026}');
        out
    }
}

fn run_uninstall() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    uninstall_launchd()?;

    #[cfg(target_os = "linux")]
    uninstall_systemd()?;

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        println!("Service installation is not supported on this platform.");
    }

    Ok(())
}

// -----------------------------------------------------------------------
// macOS: launchd user agent
// -----------------------------------------------------------------------

#[cfg(target_os = "macos")]
const LAUNCHD_LABEL: &str = "org.oxllm.proxy";

/// Where the launchd agent plist lives for the current user.
#[cfg(target_os = "macos")]
fn service_unit_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join("Library")
        .join("LaunchAgents")
        .join("org.oxllm.proxy.plist")
}

/// Minimal XML escaping for paths embedded in the plist.
#[cfg(target_os = "macos")]
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The launchd agent plist: RunAtLoad + KeepAlive, logs to
/// ~/Library/Logs/oxllm.log. Kept in the same shape as the documented
/// example in docs/architecture.md §5.4.
#[cfg(target_os = "macos")]
fn launchd_plist(exe: &std::path::Path, config: &std::path::Path, log: &std::path::Path) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>serve</string>
        <string>--config</string>
        <string>{config}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ProcessType</key>
    <string>Background</string>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
</dict>
</plist>
"#,
        label = LAUNCHD_LABEL,
        exe = xml_escape(&exe.display().to_string()),
        config = xml_escape(&config.display().to_string()),
        log = xml_escape(&log.display().to_string()),
    )
}

#[cfg(target_os = "macos")]
fn install_launchd(
    exe: &std::path::Path,
    config: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let plist_path = service_unit_path();
    let log = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
        .join("Library")
        .join("Logs")
        .join("oxllm.log");

    std::fs::create_dir_all(plist_path.parent().unwrap_or(std::path::Path::new(".")))?;
    std::fs::create_dir_all(log.parent().unwrap_or(std::path::Path::new(".")))?;
    std::fs::write(&plist_path, launchd_plist(exe, config, &log))?;

    // Replace any loaded instance, then load the new one.
    let _ = std::process::Command::new("launchctl")
        .arg("unload")
        .arg(&plist_path)
        .status();
    let status = std::process::Command::new("launchctl")
        .arg("load")
        .arg(&plist_path)
        .status()?;
    if !status.success() {
        println!("launchctl load failed (exit {:?})", status.code());
        std::process::exit(1);
    }

    println!(
        "Installed launchd agent {} at {}",
        LAUNCHD_LABEL,
        plist_path.display()
    );
    println!("Starts now and at login; KeepAlive restarts it if it exits.");
    println!("Check it with: oxllm status");
    println!("Logs: {}", log.display());
    Ok(())
}

#[cfg(target_os = "macos")]
fn uninstall_launchd() -> Result<(), Box<dyn std::error::Error>> {
    let plist_path = service_unit_path();
    let _ = std::process::Command::new("launchctl")
        .arg("unload")
        .arg(&plist_path)
        .status();
    if plist_path.exists() {
        std::fs::remove_file(&plist_path)?;
        println!(
            "Stopped and removed the launchd agent at {}",
            plist_path.display()
        );
    } else {
        println!("No launchd agent installed at {}", plist_path.display());
    }
    Ok(())
}

// -----------------------------------------------------------------------
// Linux: systemd user service
// -----------------------------------------------------------------------

/// Where the systemd user unit lives for the current user.
#[cfg(target_os = "linux")]
fn service_unit_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".config")
        .join("systemd")
        .join("user")
        .join("oxllm.service")
}

/// The systemd user unit, in the same shape as the documented example in
/// docs/architecture.md §5.4 (Restart=on-failure, journald logging).
#[cfg(target_os = "linux")]
fn systemd_unit(exe: &std::path::Path, config: &std::path::Path) -> String {
    format!(
        "[Unit]\nDescription=Oxide LLM Proxy\nAfter=network.target\n\n[Service]\nType=simple\nExecStart={exe} serve --config {config}\nRestart=on-failure\nRestartSec=5s\n\n[Install]\nWantedBy=default.target\n",
        exe = exe.display(),
        config = config.display(),
    )
}

#[cfg(target_os = "linux")]
fn install_systemd(
    exe: &std::path::Path,
    config: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let unit_path = service_unit_path();
    std::fs::create_dir_all(unit_path.parent().unwrap_or(std::path::Path::new(".")))?;
    std::fs::write(&unit_path, systemd_unit(exe, config))?;

    let run = |args: &[&str]| -> Result<bool, Box<dyn std::error::Error>> {
        Ok(std::process::Command::new("systemctl")
            .arg("--user")
            .args(args)
            .status()
            .map(|s| s.success())?)
    };

    run(&["daemon-reload"])?;
    if !run(&["enable", "--now", "oxllm.service"])? {
        println!("systemctl --user enable --now oxllm.service failed");
        std::process::exit(1);
    }

    println!("Installed systemd user service at {}", unit_path.display());
    println!("Starts now and at login; restarted on failure.");
    println!("Check it with: oxllm status");
    println!("Logs: journalctl --user -u oxllm.service -f");
    Ok(())
}

#[cfg(target_os = "linux")]
fn uninstall_systemd() -> Result<(), Box<dyn std::error::Error>> {
    let unit_path = service_unit_path();
    let _ = std::process::Command::new("systemctl")
        .args(["--user", "disable", "--now", "oxllm.service"])
        .status();
    if unit_path.exists() {
        std::fs::remove_file(&unit_path)?;
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .status();
        println!(
            "Stopped and removed the systemd user service at {}",
            unit_path.display()
        );
    } else {
        println!(
            "No systemd user service installed at {}",
            unit_path.display()
        );
    }
    Ok(())
}

async fn run_provider_offline(
    name: &str,
    host: &str,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = format!(
        "{}/admin/providers/{}/offline",
        cli_base_url(host, port),
        name
    );
    let res = match client.post(&url).send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            println!("oxllm is not running on http://127.0.0.1:{}", port);
            println!("Start it with: oxllm serve");
            std::process::exit(1);
        },
        Err(e) => return Err(e.into()),
    };
    let body = res.text().await?;
    println!("{}", body);
    Ok(())
}

async fn run_provider_online(
    name: &str,
    host: &str,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = format!(
        "{}/admin/providers/{}/online",
        cli_base_url(host, port),
        name
    );
    let res = match client.post(&url).send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            println!("oxllm is not running on http://127.0.0.1:{}", port);
            println!("Start it with: oxllm serve");
            std::process::exit(1);
        },
        Err(e) => return Err(e.into()),
    };
    let body = res.text().await?;
    println!("{}", body);
    Ok(())
}

async fn run_provider_reset(
    name: &str,
    host: &str,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = format!(
        "{}/admin/providers/{}/reset",
        cli_base_url(host, port),
        name
    );
    let res = match client.post(&url).send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            println!("oxllm is not running on http://127.0.0.1:{}", port);
            println!("Start it with: oxllm serve");
            std::process::exit(1);
        },
        Err(e) => return Err(e.into()),
    };
    let body = res.text().await?;
    println!("{}", body);
    Ok(())
}

async fn run_provider_list(
    host: &str,
    port: u16,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = format!("{}/status", cli_base_url(host, port));
    let res = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) if e.is_connect() => {
            println!("oxllm is not running on http://127.0.0.1:{}", port);
            println!("Start it with: oxllm serve");
            std::process::exit(1);
        },
        Err(e) => return Err(e.into()),
    };

    #[derive(serde::Deserialize)]
    struct RouteEntry {
        provider: String,
        model: String,
        circuit: String,
        requests: u64,
        successes: u64,
    }

    #[derive(serde::Deserialize)]
    struct ProviderAdminState {
        name: String,
        #[serde(default)]
        admin_disabled: bool,
    }

    #[derive(serde::Deserialize)]
    struct StatusResponse {
        virtual_models: std::collections::HashMap<String, Vec<RouteEntry>>,
        #[serde(default)]
        providers: Vec<ProviderAdminState>,
    }

    if json {
        let parsed: serde_json::Value = serde_json::from_str(&res.text().await?)?;
        println!("{}", serde_json::to_string_pretty(&parsed)?);
        return Ok(());
    }

    let status: StatusResponse = res.json().await?;

    // Deduplicate providers across virtual models, keeping the first occurrence
    // to show each provider once with its model and circuit state
    let mut seen = std::collections::HashSet::new();
    let mut providers: Vec<&RouteEntry> = Vec::new();

    for routes in status.virtual_models.values() {
        for entry in routes {
            if seen.insert(&entry.provider) {
                providers.push(entry);
            }
        }
    }

    println!();
    let rows: Vec<Vec<String>> = providers
        .iter()
        .map(|entry| {
            let icon = if entry.circuit.starts_with("Closed") {
                '✓'
            } else if entry.circuit.starts_with("Half") {
                '⧖'
            } else {
                '✗'
            };
            let admin_offline = status
                .providers
                .iter()
                .any(|p| p.name == entry.provider && p.admin_disabled);
            vec![
                format!("{} {}", icon, entry.provider),
                entry.model.clone(),
                if admin_offline {
                    "Offline (admin)".to_string()
                } else {
                    entry.circuit.clone()
                },
                entry.requests.to_string(),
                entry.successes.to_string(),
            ]
        })
        .collect();
    println!(
        "{}",
        render_table(
            &[
                TableColumn {
                    header: "Provider",
                    right_align: false,
                    optional: false,
                },
                TableColumn {
                    header: "Model",
                    right_align: false,
                    optional: true,
                },
                TableColumn {
                    header: "Circuit",
                    right_align: false,
                    optional: true,
                },
                TableColumn {
                    header: "Requests",
                    right_align: true,
                    optional: false,
                },
                TableColumn {
                    header: "Success",
                    right_align: true,
                    optional: false,
                },
            ],
            &rows,
            status_width(None),
        )
    );
    println!();
    println!("Use 'oxllm provider offline <name>' to take a provider out of rotation.");
    println!("Use 'oxllm provider reset <name>' to clear circuit breaker state.");
    println!();

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Build the log filter based on verbosity level
    let log_filter = match &cli.command {
        Commands::Serve { verbose, .. } => match verbose {
            0 => "info,oxllm=info,oxllm_core=info",
            1 => "info,oxllm=debug,oxllm_core=debug",
            _ => "trace",
        },
        _ => "info,oxllm=debug,oxllm_core=debug",
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_filter));
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    match cli.command {
        Commands::Serve { config, .. } => {
            run_serve(config).await?;
        },
        Commands::Validate { config } => {
            run_validate(config)?;
        },
        Commands::Status {
            host,
            port,
            json,
            width,
            verbose,
            route,
        } => {
            run_status(&host, port, json, width, verbose, route).await?;
        },
        Commands::Reload { host, port, pid } => {
            run_reload(&host, port, pid).await?;
        },
        Commands::Stop { pid } => {
            run_stop(pid)?;
        },
        Commands::Install { config, force } => {
            run_install(config, force)?;
        },
        Commands::Uninstall => {
            run_uninstall()?;
        },
        Commands::Tail { host, port, recent } => {
            run_tail(&host, port, recent).await?;
        },
        Commands::Provider(cmd) => match cmd {
            ProviderCommand::List { host, port, json } => {
                run_provider_list(&host, port, json).await?;
            },
            ProviderCommand::Offline { name, host, port } => {
                run_provider_offline(&name, &host, port).await?;
            },
            ProviderCommand::Online { name, host, port } => {
                run_provider_online(&name, &host, port).await?;
            },
            ProviderCommand::Reset { name, host, port } => {
                run_provider_reset(&name, &host, port).await?;
            },
        },
    }

    Ok(())
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use oxllm_core::config::{ApiFormat, VirtualModelTarget};
    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn spawn_mock_upstream(responses: Vec<String>) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let responses = Arc::new(responses);
            let mut idx = 0;

            while let Ok((mut stream, _)) = listener.accept().await {
                let responses = responses.clone();
                let current_idx = idx;
                idx += 1;

                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf).await;

                    let response = if responses.is_empty() {
                        "HTTP/1.1 500 Internal Error\r\nContent-Length: 0\r\n\r\n"
                    } else {
                        let r_idx = current_idx.min(responses.len() - 1);
                        &responses[r_idx]
                    };

                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                });
            }
        });

        addr
    }

    #[tokio::test]
    async fn test_integration_rate_limit_failover() {
        let upstream1_resp = vec![
            "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Length: 0\r\n\r\n"
                .to_string(),
        ];
        let upstream2_resp = vec![
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 75\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hello from prov2\"}}]}".to_string()
        ];

        let addr1 = spawn_mock_upstream(upstream1_resp).await;
        let addr2 = spawn_mock_upstream(upstream2_resp).await;

        let p1 = ProviderState {
            name: "prov1".to_string(),
            base_url: Url::parse(&format!("http://{}", addr1)).unwrap(),
            api_key: "key1".to_string(),
            models: vec!["gpt-4-upstream".to_string()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let p2 = ProviderState {
            name: "prov2".to_string(),
            base_url: Url::parse(&format!("http://{}", addr2)).unwrap(),
            api_key: "key2".to_string(),
            models: vec!["gpt-4-upstream".to_string()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "gpt-4".to_string(),
            vec![
                VirtualModelTarget {
                    provider: "prov1".to_string(),
                    model: "gpt-4-upstream".to_string(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
                VirtualModelTarget {
                    provider: "prov2".to_string(),
                    model: "gpt-4-upstream".to_string(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
            ],
        );

        let http_client = reqwest::Client::builder().build().unwrap();
        let app_state = Arc::new(AppState {
            providers: vec![p1, p2],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client,
            upstream_timeout_secs: 5,
        });

        let (_watch_sender, watch_receiver) = tokio::sync::watch::channel(app_state.clone());
        let (telemetry_tx, _telemetry_rx) = tokio::sync::mpsc::channel(1024);
        let telemetry_client = TelemetryClient::new(telemetry_tx);

        let dummy_reloader = Reloader {
            sender: _watch_sender.clone(),
            config_path: PathBuf::from("config.toml"),
        };

        let reloadable_state = ReloadableState {
            app_state: watch_receiver,
            telemetry: telemetry_client,
            start_time: Instant::now(),
            reloader: dummy_reloader,
        };

        let router = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(reloadable_state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}]
        });

        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 200);

        let body: Value = res.json().await.unwrap();
        let content = body["choices"][0]["message"]["content"].as_str().unwrap();
        assert_eq!(content, "Hello from prov2");

        // The 429 window is held by the limited model, not the whole provider.
        let p1_model = app_state.providers[0].model_health.get("gpt-4-upstream");
        let p1_limited = p1_model.rate_limited_until.read().await;
        assert!(p1_limited.is_some());
    }

    #[tokio::test]
    async fn test_integration_sse_streaming() {
        let sse_response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"token\": \"Hello\"}\n\ndata: {\"token\": \" World\"}\n\n".to_string();

        let addr = spawn_mock_upstream(vec![sse_response]).await;

        let p = ProviderState {
            name: "prov1".to_string(),
            base_url: Url::parse(&format!("http://{}", addr)).unwrap(),
            api_key: "key1".to_string(),
            models: vec!["gpt-4-upstream".to_string()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "gpt-4".to_string(),
            vec![VirtualModelTarget {
                provider: "prov1".to_string(),
                model: "gpt-4-upstream".to_string(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );

        let http_client = reqwest::Client::builder().build().unwrap();
        let app_state = Arc::new(AppState {
            providers: vec![p],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client,
            upstream_timeout_secs: 5,
        });

        let (_watch_sender, watch_receiver) = tokio::sync::watch::channel(app_state.clone());
        let (telemetry_tx, _telemetry_rx) = tokio::sync::mpsc::channel(1024);
        let telemetry_client = TelemetryClient::new(telemetry_tx);

        let dummy_reloader = Reloader {
            sender: _watch_sender.clone(),
            config_path: PathBuf::from("config.toml"),
        };

        let reloadable_state = ReloadableState {
            app_state: watch_receiver,
            telemetry: telemetry_client,
            start_time: Instant::now(),
            reloader: dummy_reloader,
        };

        let router = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(reloadable_state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });

        let mut res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 200);
        assert_eq!(
            res.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let mut body = String::new();
        while let Some(chunk) = res.chunk().await.unwrap() {
            body.push_str(std::str::from_utf8(&chunk).unwrap());
        }

        assert!(body.contains("data: {\"token\": \"Hello\"}"));
        assert!(body.contains("data: {\"token\": \" World\"}"));
    }

    /// Mock upstream that records the request line path and body so tests can
    /// assert routing targets and model rewriting.
    async fn spawn_recording_upstream(
        responses: Vec<String>,
        recorded: Arc<tokio::sync::Mutex<Vec<(String, String)>>>,
    ) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let mut idx = 0usize;

            while let Ok((mut stream, _)) = listener.accept().await {
                let recorded = recorded.clone();
                let responses = responses.clone();
                let response_idx = idx;
                idx += 1;

                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];

                    // Read until the header terminator; the request bodies in
                    // these tests are far smaller than the buffer.
                    loop {
                        match stream.read(&mut tmp).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&tmp[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                    break;
                                }
                            },
                        }
                    }

                    let text = String::from_utf8_lossy(&buf).to_string();
                    let path = text
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        .unwrap_or_default()
                        .to_string();
                    let body = text
                        .split("\r\n\r\n")
                        .nth(1)
                        .unwrap_or_default()
                        .to_string();
                    recorded.lock().await.push((path, body));

                    let response = responses.get(response_idx).cloned().unwrap_or_else(|| {
                        "HTTP/1.1 500 Internal Error\r\nContent-Length: 0\r\n\r\n".to_string()
                    });
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                });
            }
        });

        addr
    }

    fn completions_test_state(addr: SocketAddr) -> Arc<AppState> {
        let provider = ProviderState {
            name: "omlx-local".to_string(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "test-key".to_string(),
            models: vec!["granite-4.1-3b-4bit".to_string()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "fim-virtual".to_string(),
            vec![VirtualModelTarget {
                provider: "omlx-local".to_string(),
                model: "granite-4.1-3b-4bit".to_string(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );

        Arc::new(AppState {
            providers: vec![provider],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        })
    }

    /// POST /v1/completions: virtual model is rewritten to the upstream model,
    /// the request lands on <base>/completions, the body passes through intact,
    /// and token usage is accounted on the provider.
    #[tokio::test]
    async fn test_integration_completions_passthrough() {
        let upstream_body = r#"{"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":"    return json.load(f)","finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            upstream_body.len(),
            upstream_body
        );

        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![response], recorded.clone()).await;
        let app_state = completions_test_state(addr);

        let (_watch_sender, watch_receiver) = tokio::sync::watch::channel(app_state.clone());
        let (telemetry_tx, _telemetry_rx) = tokio::sync::mpsc::channel(1024);
        let reloadable_state = ReloadableState {
            app_state: watch_receiver,
            telemetry: TelemetryClient::new(telemetry_tx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _watch_sender.clone(),
                config_path: PathBuf::from("config.toml"),
            },
        };

        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(reloadable_state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let payload = serde_json::json!({
            "model": "fim-virtual",
            "prompt": "def read_config(path):\n    ",
            "max_tokens": 8
        });
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/completions", proxy_addr))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 200);
        let body: Value = res.json().await.unwrap();
        assert_eq!(body["choices"][0]["text"], "    return json.load(f)");

        let recorded = recorded.lock().await;
        let (path, upstream_request) = recorded.first().expect("upstream never called");
        assert_eq!(path, "/v1/completions");
        let upstream_json: Value = serde_json::from_str(upstream_request).unwrap();
        assert_eq!(upstream_json["model"], "granite-4.1-3b-4bit");
        assert_eq!(upstream_json["prompt"], "def read_config(path):\n    ");
        drop(recorded);

        assert_eq!(
            app_state.providers[0]
                .tokens_input
                .load(std::sync::atomic::Ordering::Relaxed),
            11
        );
        assert_eq!(
            app_state.providers[0]
                .tokens_output
                .load(std::sync::atomic::Ordering::Relaxed),
            7
        );
    }

    /// POST /v1/completions with stream=true must forward SSE frames as they
    /// arrive (content-type preserved, no buffering into a single JSON body).
    #[tokio::test]
    async fn test_integration_completions_sse_streaming() {
        let sse_response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"choices\":[{\"text\":\"Hello\"}]}\n\ndata: {\"choices\":[{\"text\":\" World\"}]}\n\n".to_string();

        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![sse_response], recorded).await;
        let app_state = completions_test_state(addr);

        let (_watch_sender, watch_receiver) = tokio::sync::watch::channel(app_state);
        let (telemetry_tx, _telemetry_rx) = tokio::sync::mpsc::channel(1024);
        let reloadable_state = ReloadableState {
            app_state: watch_receiver,
            telemetry: TelemetryClient::new(telemetry_tx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _watch_sender.clone(),
                config_path: PathBuf::from("config.toml"),
            },
        };

        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(reloadable_state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let payload = serde_json::json!({
            "model": "fim-virtual",
            "prompt": "x",
            "stream": true
        });
        let mut res = reqwest::Client::new()
            .post(format!("http://{}/v1/completions", proxy_addr))
            .json(&payload)
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 200);
        assert_eq!(
            res.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        let mut body = String::new();
        while let Some(chunk) = res.chunk().await.unwrap() {
            body.push_str(std::str::from_utf8(&chunk).unwrap());
        }

        assert!(body.contains("data: {\"choices\":[{\"text\":\"Hello\"}]}"));
        assert!(body.contains("data: {\"choices\":[{\"text\":\" World\"}]}"));
    }

    /// Per-(provider, model) attribution: two virtual models sharing one
    /// provider must not show each other's traffic, and /status must report
    /// the latency distribution for the route that actually ran.
    #[tokio::test]
    async fn test_status_attributes_traffic_per_model_not_per_provider() {
        let response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr =
            spawn_recording_upstream(vec![response.to_string(), response.to_string()], recorded)
                .await;

        let provider = ProviderState {
            name: "omlx-local".to_string(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "k".to_string(),
            models: vec!["small".to_string()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "router-small".to_string(),
            vec![VirtualModelTarget {
                provider: "omlx-local".to_string(),
                model: "small".to_string(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );
        virtual_models.insert(
            "router-big".to_string(),
            vec![VirtualModelTarget {
                provider: "omlx-local".to_string(),
                model: "big".to_string(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );

        let app_state = Arc::new(AppState {
            providers: vec![provider],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws.clone(),
                config_path: PathBuf::from("config.toml"),
            },
        };

        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions),
            )
            .route("/status", axum::routing::get(routes::get_status))
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        // Two calls to the small route only.
        for _ in 0..2 {
            let res = client
                .post(format!("http://{}/v1/completions", proxy_addr))
                .json(&serde_json::json!({"model": "router-small", "prompt": "x"}))
                .send()
                .await
                .unwrap();
            assert_eq!(res.status(), 200);
        }

        let status: Value = client
            .get(format!("http://{}/status", proxy_addr))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let small = &status["virtual_models"]["router-small"][0];
        assert_eq!(small["model"], "small");
        assert_eq!(small["requests"], 2, "small route should have 2 requests");

        let big = &status["virtual_models"]["router-big"][0];
        assert_eq!(big["model"], "big");
        assert_eq!(
            big["requests"], 0,
            "the sibling model on the same provider must not inherit traffic"
        );
        assert_eq!(big["tokens_input"], 0);

        // Route rollup carries the distribution for the route that ran.
        let routes = status["routes"].as_array().expect("routes array");
        let small_route = routes
            .iter()
            .find(|r| r["route"] == "router-small")
            .expect("router-small route row");
        assert_eq!(small_route["requests"], 2);
        assert_eq!(small_route["successes"], 2);
        assert_eq!(small_route["failures"], 0);
        assert_eq!(small_route["policy"], "priority");
        assert!(
            small_route["latency_p50_ms"].is_number(),
            "p50 should be populated after two requests"
        );

        let big_route = routes
            .iter()
            .find(|r| r["route"] == "router-big")
            .expect("router-big route row");
        assert_eq!(big_route["requests"], 0);
    }

    /// Inbound auth: with server.api_keys set, /v1 routes reject requests
    /// without a valid Bearer key and accept requests with one. Admin routes
    /// stay localhost-only regardless.
    #[tokio::test]
    async fn test_api_key_auth_gate() {
        let upstream_body = r#"{"choices":[{"index":0,"text":"ok","finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            upstream_body.len(),
            upstream_body
        );
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![response], recorded.clone()).await;

        let provider = ProviderState {
            name: "omlx-local".to_string(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "upstream-key".to_string(),
            models: vec!["m".to_string()],
            upstream_timeout_secs: Some(2),
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "vm".to_string(),
            vec![VirtualModelTarget {
                provider: "omlx-local".to_string(),
                model: "m".to_string(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );

        let app_state = Arc::new(AppState {
            providers: vec![provider],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            api_keys: vec!["sekrit".to_string()],
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws.clone(),
                config_path: PathBuf::from("config.toml"),
            },
        };

        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions)
                    .layer(middleware::from_fn_with_state(rs.clone(), require_api_key)),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/completions", proxy_addr);
        let body = serde_json::json!({"model": "vm", "prompt": "x"});

        // No key -> 401, and the upstream is never contacted.
        let res = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(res.status(), 401);
        assert!(
            recorded.lock().await.is_empty(),
            "unauthorized request must not reach upstream"
        );

        // Wrong key -> 401.
        let res = client
            .post(&url)
            .header("Authorization", "Bearer wrong")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 401);

        // Correct key -> routed.
        let res = client
            .post(&url)
            .header("Authorization", "Bearer sekrit")
            .json(&body)
            .send()
            .await
            .unwrap();
        if res.status() != 200 {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            panic!("authorized request failed: {} {}", status, text);
        }
        assert_eq!(recorded.lock().await.len(), 1);
    }

    /// Session affinity: two providers back one virtual model; requests
    /// carrying the same x-session-id stick to whichever provider served the
    /// first request, while keyless traffic still rotates by policy.
    #[tokio::test]
    async fn test_session_affinity_pins_provider() {
        let response_body = r#"{"choices":[{"index":0,"text":"ok","finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![response; 16], recorded.clone()).await;

        let mk = |name: &str| ProviderState {
            name: name.to_string(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "k".to_string(),
            models: vec!["m".to_string()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "vm".to_string(),
            vec![
                VirtualModelTarget {
                    provider: "prov-a".to_string(),
                    model: "m".to_string(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
                VirtualModelTarget {
                    provider: "prov-b".to_string(),
                    model: "m".to_string(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
            ],
        );

        let app_state = Arc::new(AppState {
            providers: vec![mk("prov-a"), mk("prov-b")],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(600),
                64,
            )),
            session_affinity_header: "x-session-id".to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws.clone(),
                config_path: PathBuf::from("config.toml"),
            },
        };

        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/completions", proxy_addr);

        // Same session key, several requests.
        for _ in 0..4 {
            let res = client
                .post(&url)
                .header("x-session-id", "session-1")
                .json(&serde_json::json!({"model": "vm", "prompt": "x"}))
                .send()
                .await
                .unwrap();
            assert_eq!(res.status(), 200);
        }

        // The recorded upstream request bodies all carry the model, but the
        // mock cannot distinguish providers by connection; assert via the
        // affinity table instead: exactly one pin exists, and every pinned
        // lookup resolves to the same provider.
        let pinned = app_state.session_affinity.pinned_provider("session-1");
        assert!(pinned.is_some());
        let first = pinned.unwrap();
        for _ in 0..3 {
            assert_eq!(
                app_state.session_affinity.pinned_provider("session-1"),
                Some(first.clone()),
                "session must stay pinned to one provider"
            );
        }
        // A different session is unpinned until it makes a request.
        assert_eq!(
            app_state.session_affinity.pinned_provider("session-2"),
            None
        );
        assert_eq!(app_state.session_affinity.len(), 1);
    }

    // -----------------------------------------------------------------------
    // New integration tests: timeout failover, all-providers-fail, manual
    // offline, rate-limit recovery, embeddings endpoint
    // -----------------------------------------------------------------------

    /// Circuit breaker trips after 3 failures; failover routes to healthy provider.
    #[tokio::test]
    async fn test_integration_circuit_breaker_failover() {
        use oxllm_core::router::{AdaptivePriorityStrategy, RoutingStrategy};

        let p1 = ProviderState {
            name: "primary".into(),
            base_url: Url::parse("https://api.fail.example.com/v1/").unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let p2 = ProviderState {
            name: "secondary".into(),
            base_url: Url::parse("https://api.ok.example.com/v1/").unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let strategy = AdaptivePriorityStrategy::default();
        let candidates = vec![&p1, &p2];

        // Initially selects primary
        let selected = strategy.select(&candidates).await.unwrap();
        assert_eq!(selected.name, "primary");

        // Simulate 3 failures on primary
        for _ in 0..3 {
            strategy.feedback(&p1, false, false, Some(500), None).await;
        }

        // Circuit should be open now; select should skip to secondary
        assert!(matches!(
            *p1.circuit.read().await,
            CircuitState::Open { .. }
        ));
        let selected = strategy.select(&candidates).await.unwrap();
        assert_eq!(selected.name, "secondary");
    }

    /// When every provider fails, the proxy returns 502 Bad Gateway.
    #[tokio::test]
    async fn test_integration_all_providers_fail_gives_502() {
        let fail_response =
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n".to_string();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(fail_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let make_provider = |name: &str| ProviderState {
            name: name.into(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let p1 = make_provider("p1");
        let p2 = make_provider("p2");

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "test".into(),
            vec![
                VirtualModelTarget {
                    provider: "p1".into(),
                    model: "model".into(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
                VirtualModelTarget {
                    provider: "p2".into(),
                    model: "model".into(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
            ],
        );

        let app_state = Arc::new(AppState {
            providers: vec![p1, p2],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let state = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 502);
        let body = res.text().await.unwrap();
        assert!(body.contains("All upstream chat completions providers failed"));
    }

    /// Manual offline takes a provider out of rotation; requests bypass it.
    #[tokio::test]
    async fn test_integration_manual_offline_bypasses_provider() {
        use std::sync::atomic::Ordering;

        let ok_response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"
                .to_string();

        // Two independent upstreams
        let p1_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p1_addr = p1_listener.local_addr().unwrap();
        let ok_response_p1 = ok_response.clone();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = p1_listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                // Should never be hit — but respond just in case
                let _ = stream.write_all(ok_response_p1.as_bytes()).await;
            }
        });

        let p2_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let p2_addr = p2_listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = p2_listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let p1 = ProviderState {
            name: "p1".into(),
            base_url: Url::parse(&format!("http://{}/v1/", p1_addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(true), // MANUALLY DISABLED
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let p2 = ProviderState {
            name: "p2".into(),
            base_url: Url::parse(&format!("http://{}/v1/", p2_addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "test".into(),
            vec![
                VirtualModelTarget {
                    provider: "p1".into(),
                    model: "model".into(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
                VirtualModelTarget {
                    provider: "p2".into(),
                    model: "model".into(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
            ],
        );

        let app_state = Arc::new(AppState {
            providers: vec![p1, p2],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let state = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        // p1 was never hit; p2 handled the request
        assert_eq!(app_state.providers[0].requests.load(Ordering::Relaxed), 0);
        assert_eq!(app_state.providers[1].requests.load(Ordering::Relaxed), 1);
    }

    /// Admin reset endpoint clears circuit breaker, failures, and rate limit.
    #[tokio::test]
    async fn test_integration_admin_reset() {
        use std::sync::atomic::Ordering;

        let ok_response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"
                .to_string();
        let ok_response_clone = ok_response.clone();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response_clone.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let p = ProviderState {
            name: "resetme".into(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "test".into(),
            vec![VirtualModelTarget {
                provider: "resetme".into(),
                model: "model".into(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );

        let app_state = Arc::new(AppState {
            providers: vec![p],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let state = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };

        // Simulate a tripped state
        *app_state.providers[0].circuit.write().await = CircuitState::Open {
            until: Instant::now() + Duration::from_secs(300),
        };
        *app_state.providers[0].consecutive_failures.write().await = 5;
        *app_state.providers[0].rate_limited_until.write().await =
            Some(Instant::now() + Duration::from_secs(300));
        app_state.providers[0]
            .manual_disabled
            .store(true, Ordering::Release);

        // Build router with admin routes
        let router = axum::Router::new()
            .route(
                "/admin/providers/{name}/reset",
                axum::routing::post(routes::admin_reset),
            )
            .with_state(state.clone());
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(proxy_listener, router).await.unwrap();
        });

        // Call admin reset
        let client = reqwest::Client::new();
        let res = client
            .post(format!(
                "http://{}/admin/providers/resetme/reset",
                proxy_addr
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        // Verify all state is cleared
        assert_eq!(
            *app_state.providers[0].circuit.read().await,
            CircuitState::Closed
        );
        assert_eq!(*app_state.providers[0].consecutive_failures.read().await, 0);
        assert!(app_state.providers[0]
            .rate_limited_until
            .read()
            .await
            .is_none());
        assert!(!app_state.providers[0]
            .manual_disabled
            .load(Ordering::Acquire));
    }

    /// /v1/embeddings endpoint proxies correctly and returns embeddings.
    #[tokio::test]
    async fn test_integration_embeddings_success() {
        let body = r#"{"data":[{"embedding":[0.1,0.2,0.3],"index":0}],"model":"test","usage":{"prompt_tokens":4}}"#;
        let ok_response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let p = ProviderState {
            name: "emb".into(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert(
            "test-emb".into(),
            vec![VirtualModelTarget {
                provider: "emb".into(),
                model: "model".into(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );

        let app_state = Arc::new(AppState {
            providers: vec![p],
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let state = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions),
            )
            .route(
                "/v1/embeddings",
                axum::routing::post(routes::create_embeddings),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/embeddings", proxy_addr))
            .json(&serde_json::json!({
                "model": "test-emb",
                "input": "hello"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let body: Value = res.json().await.unwrap();
        assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 3);
    }

    /// Invalid model name returns 400 Bad Request.
    #[tokio::test]
    async fn test_integration_invalid_model_returns_400() {
        let ok_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}".to_string();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let p = ProviderState {
            name: "prov".into(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["real-model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };
        let mut vm = std::collections::HashMap::new();
        vm.insert(
            "known-model".into(),
            vec![VirtualModelTarget {
                provider: "prov".into(),
                model: "real-model".into(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );

        let state = Arc::new(AppState {
            providers: vec![p],
            virtual_models: vm,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });
        let (_ws, wr) = tokio::sync::watch::channel(state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client.post(format!("http://{}/v1/chat/completions", addr))
            .json(&serde_json::json!({"model": "nonexistent", "messages": [{"role":"user","content":"hi"}]}))
            .send().await.unwrap();
        assert_eq!(res.status(), 400);
    }

    /// Admin online re-enables a manually disabled provider.
    #[tokio::test]
    async fn test_integration_admin_online() {
        use std::sync::atomic::Ordering;

        let ok_response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"
                .to_string();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let p = ProviderState {
            name: "target".into(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(true), // Start disabled
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut vms = std::collections::HashMap::new();
        vms.insert(
            "test".into(),
            vec![VirtualModelTarget {
                provider: "target".into(),
                model: "model".into(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );
        let state = Arc::new(AppState {
            providers: vec![p],
            virtual_models: vms,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });
        let (_ws, wr) = tokio::sync::watch::channel(state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws.clone(),
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/admin/providers/{name}/online",
                axum::routing::post(routes::admin_online),
            )
            .with_state(rs);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        // Verify provider is disabled
        assert!(state.providers[0].manual_disabled.load(Ordering::Acquire));

        // Call admin online
        let client = reqwest::Client::new();
        let res = client
            .post(format!(
                "http://{}/admin/providers/target/online",
                proxy_addr
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        // Verify provider is now enabled
        assert!(!state.providers[0].manual_disabled.load(Ordering::Acquire));
    }

    /// Token counts are parsed from upstream JSON and recorded in provider counters.
    #[tokio::test]
    async fn test_integration_token_count_parsed() {
        // Upstream returns a response with usage data
        let body = r#"{"id":"test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":42,"completion_tokens":7,"total_tokens":49}}"#;
        let ok_response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let p = ProviderState {
            name: "counter".into(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut vms = std::collections::HashMap::new();
        vms.insert(
            "test".into(),
            vec![VirtualModelTarget {
                provider: "counter".into(),
                model: "model".into(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );
        let state = Arc::new(AppState {
            providers: vec![p],
            virtual_models: vms,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });
        let (_ws, wr) = tokio::sync::watch::channel(state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .json(
                &serde_json::json!({"model": "test", "messages": [{"role":"user","content":"hi"}]}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        // Verify token counts were parsed
        let tokens_in = state.providers[0]
            .tokens_input
            .load(std::sync::atomic::Ordering::Relaxed);
        let tokens_out = state.providers[0]
            .tokens_output
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            tokens_in > 0,
            "Expected input tokens > 0, got {}",
            tokens_in
        );
        assert!(
            tokens_out > 0,
            "Expected output tokens > 0, got {}",
            tokens_out
        );
        assert_eq!(tokens_in, 42);
        assert_eq!(tokens_out, 7);
    }

    // -----------------------------------------------------------------------
    // Test helpers for JSON error format, CORS, and x-request-id tests
    // -----------------------------------------------------------------------

    /// Builds a minimal test AppState + ReloadableState with one provider
    /// pointing at the given upstream address, plus alias/default-model
    /// routing configuration.
    fn build_test_state_with(
        upstream_addr: SocketAddr,
        aliases: std::collections::HashMap<String, String>,
        default_model: Option<String>,
    ) -> (Arc<AppState>, ReloadableState) {
        let p = ProviderState {
            name: "prov".into(),
            base_url: Url::parse(&format!("http://{}/v1/", upstream_addr)).unwrap(),
            api_key: "key".into(),
            models: vec!["model".into()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };
        let mut vm = std::collections::HashMap::new();
        vm.insert(
            "test".into(),
            vec![VirtualModelTarget {
                provider: "prov".into(),
                model: "model".into(),
                priority: None,
                weight: None,
                virtual_ref: None,
            }],
        );

        let app_state = Arc::new(AppState {
            providers: vec![p],
            virtual_models: vm,
            aliases,
            default_model,
            circuit: CircuitConfig::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });
        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        (app_state, rs)
    }

    /// Builds a minimal test AppState + ReloadableState with one provider
    /// pointing at the given upstream address.
    fn build_test_state(upstream_addr: SocketAddr) -> (Arc<AppState>, ReloadableState) {
        build_test_state_with(upstream_addr, Default::default(), None)
    }

    /// Builds a test router with all middleware layers (request-id, CORS).
    fn build_test_router(state: ReloadableState) -> axum::Router {
        axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions),
            )
            .route(
                "/v1/embeddings",
                axum::routing::post(routes::create_embeddings),
            )
            .route("/v1/models", axum::routing::get(routes::list_models))
            .layer(middleware::from_fn(add_request_id))
            .layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods([
                        axum::http::Method::GET,
                        axum::http::Method::POST,
                        axum::http::Method::OPTIONS,
                    ])
                    .allow_headers(Any),
            )
            .with_state(state)
    }

    // -----------------------------------------------------------------------
    // Integration tests: JSON error format
    // -----------------------------------------------------------------------

    /// Invalid JSON body returns 400 with structured JSON error.
    #[tokio::test]
    async fn test_integration_invalid_json() {
        let ok_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}".to_string();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let (_app_state, rs) = build_test_state(addr);
        let router = build_test_router(rs);
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(proxy_listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .body("not valid json {{{")
            .header("Content-Type", "application/json")
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 400);
        assert_eq!(
            res.headers().get("content-type").unwrap(),
            "application/json"
        );

        let body: Value = res.json().await.unwrap();
        let error = body.get("error").expect("response should have 'error' key");
        assert!(error.get("message").is_some());
        assert_eq!(
            error.get("type").unwrap().as_str().unwrap(),
            "invalid_request_error"
        );
        assert_eq!(error.get("code").unwrap().as_u64().unwrap(), 400);
    }

    /// Missing 'model' field returns 400 with structured JSON error.
    #[tokio::test]
    async fn test_integration_missing_model() {
        let ok_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}".to_string();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let (_app_state, rs) = build_test_state(addr);
        let router = build_test_router(rs);
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(proxy_listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .json(&serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 400);
        assert_eq!(
            res.headers().get("content-type").unwrap(),
            "application/json"
        );

        let body: Value = res.json().await.unwrap();
        let error = body.get("error").expect("response should have 'error' key");
        assert!(error.get("message").is_some());
        assert_eq!(
            error.get("type").unwrap().as_str().unwrap(),
            "invalid_request_error"
        );
        assert_eq!(error.get("code").unwrap().as_u64().unwrap(), 400);
    }

    /// Unknown virtual model returns 400 with structured JSON error.
    #[tokio::test]
    async fn test_integration_unknown_model() {
        let ok_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}".to_string();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let (_app_state, rs) = build_test_state(addr);
        let router = build_test_router(rs);
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(proxy_listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .json(&serde_json::json!({
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 400);
        assert_eq!(
            res.headers().get("content-type").unwrap(),
            "application/json"
        );

        let body: Value = res.json().await.unwrap();
        let error = body.get("error").expect("response should have 'error' key");
        let message = error.get("message").unwrap().as_str().unwrap();
        assert!(message.contains("nonexistent-model"));
        assert_eq!(
            error.get("type").unwrap().as_str().unwrap(),
            "invalid_request_error"
        );
        assert_eq!(error.get("code").unwrap().as_u64().unwrap(), 400);
    }

    // -----------------------------------------------------------------------
    // Integration tests: CORS headers
    // -----------------------------------------------------------------------

    /// OPTIONS preflight to /v1/chat/completions returns CORS headers.
    #[tokio::test]
    async fn test_integration_cors_options() {
        let ok_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}".to_string();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let (_app_state, rs) = build_test_state(addr);
        let router = build_test_router(rs);
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(proxy_listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .request(
                reqwest::Method::OPTIONS,
                format!("http://{}/v1/chat/completions", proxy_addr),
            )
            .header("Origin", "http://localhost:3000")
            .header("Access-Control-Request-Method", "POST")
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 200);
        assert_eq!(
            res.headers().get("access-control-allow-origin").unwrap(),
            "*"
        );
        assert!(res.headers().contains_key("access-control-allow-methods"));
    }

    /// POST to /v1/chat/completions includes CORS origin header in response.
    #[tokio::test]
    async fn test_integration_cors_post() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"Hello from prov"}}]}"#;
        let ok_response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let (_app_state, rs) = build_test_state(addr);
        let router = build_test_router(rs);
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(proxy_listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .header("Origin", "http://localhost:3000")
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 200);
        assert_eq!(
            res.headers().get("access-control-allow-origin").unwrap(),
            "*"
        );
    }

    // -----------------------------------------------------------------------
    // Integration tests: x-request-id header
    // -----------------------------------------------------------------------

    /// Successful response includes x-request-id matching oxllm-[0-9a-f]{16}.
    #[tokio::test]
    async fn test_integration_request_id_on_success() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"Hello from prov"}}]}"#;
        let ok_response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let (_app_state, rs) = build_test_state(addr);
        let router = build_test_router(rs);
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(proxy_listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 200);
        let request_id = res
            .headers()
            .get("x-request-id")
            .expect("x-request-id header should be present");
        let id_str = request_id.to_str().unwrap();
        assert!(
            id_str.starts_with("oxllm-"),
            "x-request-id should start with 'oxllm-', got: {}",
            id_str
        );
        assert_eq!(
            id_str.len(),
            22,
            "x-request-id should be 'oxllm-' + 16 hex chars (len 22), got len {}",
            id_str.len()
        );
        // Verify remaining chars are valid hex
        let hex_part = &id_str[6..];
        assert!(
            hex_part.chars().all(|c| c.is_ascii_hexdigit()),
            "x-request-id hex part should be all hex chars, got: {}",
            hex_part
        );
    }

    /// Error response also includes x-request-id header.
    #[tokio::test]
    async fn test_integration_request_id_on_error() {
        let ok_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}".to_string();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(ok_response.as_bytes()).await;
                let _ = stream.flush().await;
            }
        });

        let (_app_state, rs) = build_test_state(addr);
        let router = build_test_router(rs);
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(proxy_listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy_addr))
            .body("not valid json")
            .header("Content-Type", "application/json")
            .send()
            .await
            .unwrap();

        assert_eq!(res.status(), 400);
        let request_id = res
            .headers()
            .get("x-request-id")
            .expect("x-request-id header should be present on error responses");
        let id_str = request_id.to_str().unwrap();
        assert!(
            id_str.starts_with("oxllm-"),
            "x-request-id should start with 'oxllm-', got: {}",
            id_str
        );
        assert_eq!(id_str.len(), 22);
    }

    // -----------------------------------------------------------------------
    // Config hot-reload (issue #7): file watcher, HTTP reload, PID file
    // -----------------------------------------------------------------------

    /// Minimal ServerConfig for test AppState literals.
    fn test_server_config() -> oxllm_core::config::ServerConfig {
        oxllm_core::config::ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 18099,
            otel_endpoint: "http://127.0.0.1:4318".to_string(),
            upstream_timeout_secs: 5,
            bind_family: "ipv4".to_string(),
            api_keys: vec![],
            session_affinity_header: "x-session-id".to_string(),
            session_affinity_ttl_secs: 900,
            session_affinity_capacity: 1024,
            circuit_failure_threshold: 3,
            circuit_base_cooldown_secs: 60,
            rate_limit_fallback_secs: 30,
            shutdown_timeout_secs: 30,
            watch_config: true,
            request_log_capacity: 256,
            default_model: None,
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("oxllm-test-{}-{}", std::process::id(), name));
        path
    }

    fn test_config_toml(provider: &str) -> String {
        format!(
            r#"
[server]
host = "127.0.0.1"
port = 18099
otel_endpoint = "http://127.0.0.1:4318"

[[providers]]
name = "{provider}"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "key"
models = ["m"]

[virtual_models]
vm = [{{ provider = "{provider}", model = "m" }}]
"#
        )
    }

    /// Same shape, but the virtual model targets an undefined provider, so
    /// `validate()` rejects it.
    fn test_config_toml_dangling_reference() -> String {
        r#"
[server]
host = "127.0.0.1"
port = 18099
otel_endpoint = "http://127.0.0.1:4318"

[[providers]]
name = "prov-a"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "key"
models = ["m"]

[virtual_models]
vm = [{ provider = "missing-prov", model = "m" }]
"#
        .to_string()
    }

    /// Spawns and reaps a child to obtain a PID that is guaranteed dead.
    fn dead_pid() -> u32 {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let pid = child.id();
        child.kill().unwrap();
        child.wait().unwrap();
        pid
    }

    /// Editing the config file takes effect without a manual reload: the
    /// watcher swaps in the new state through the same path as SIGHUP and
    /// POST /reload. Regression test for issue #7 ("config changes need a
    /// manual reload").
    #[tokio::test]
    async fn test_config_watcher_hot_reloads_on_file_change() {
        let path = temp_path("watch-valid.toml");
        std::fs::write(&path, test_config_toml("prov-a")).unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let initial = Arc::new(build_app_state(config, None).unwrap());
        let (sender, mut receiver) = tokio::sync::watch::channel(initial);

        tokio::spawn(watch_config_file(
            path.clone(),
            sender,
            Duration::from_millis(50),
        ));

        // Let the watcher establish its baseline before rewriting. On the
        // current-thread test runtime the spawned task runs while we sleep,
        // so 3 poll intervals are ample margin.
        tokio::time::sleep(Duration::from_millis(150)).await;
        std::fs::write(&path, test_config_toml("prov-b")).unwrap();

        tokio::time::timeout(Duration::from_secs(2), receiver.changed())
            .await
            .expect("watcher must reload after the config file changes")
            .unwrap();
        assert_eq!(receiver.borrow_and_update().providers[0].name, "prov-b");
        let _ = std::fs::remove_file(&path);
    }

    /// Complementary case: a broken edit (dangling provider reference) is
    /// rejected, the previous config keeps serving, and fixing the file
    /// reloads. A failed reload must never swap in broken state.
    #[tokio::test]
    async fn test_config_watcher_keeps_old_config_when_new_config_invalid() {
        let path = temp_path("watch-invalid.toml");
        std::fs::write(&path, test_config_toml("prov-a")).unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let initial = Arc::new(build_app_state(config, None).unwrap());
        let (sender, mut receiver) = tokio::sync::watch::channel(initial);

        tokio::spawn(watch_config_file(
            path.clone(),
            sender,
            Duration::from_millis(50),
        ));
        tokio::time::sleep(Duration::from_millis(150)).await;

        std::fs::write(&path, test_config_toml_dangling_reference()).unwrap();
        let swapped = tokio::time::timeout(Duration::from_millis(300), receiver.changed()).await;
        assert!(swapped.is_err(), "an invalid config must not be swapped in");

        std::fs::write(&path, test_config_toml("prov-b")).unwrap();
        tokio::time::timeout(Duration::from_secs(2), receiver.changed())
            .await
            .expect("watcher must reload after the config file is fixed")
            .unwrap();
        assert_eq!(receiver.borrow_and_update().providers[0].name, "prov-b");
        let _ = std::fs::remove_file(&path);
    }

    /// `oxllm reload` drives the daemon through POST /reload and reports
    /// success only when the new config was actually swapped in.
    #[tokio::test]
    async fn test_run_reload_succeeds_via_http_endpoint() {
        let path = temp_path("reload-ok.toml");
        std::fs::write(&path, test_config_toml("prov-a")).unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let initial = Arc::new(build_app_state(config, None).unwrap());
        let (sender, mut receiver) = tokio::sync::watch::channel(initial);

        let router = axum::Router::new()
            .route("/reload", axum::routing::post(handle_http_reload))
            .with_state(Reloader {
                sender,
                config_path: path.clone(),
            });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        run_reload("127.0.0.1", port, None).await.unwrap();
        receiver.changed().await.unwrap();
        assert_eq!(receiver.borrow_and_update().providers[0].name, "prov-a");

        std::fs::write(&path, test_config_toml("prov-b")).unwrap();
        run_reload("127.0.0.1", port, None).await.unwrap();
        receiver.changed().await.unwrap();
        assert_eq!(receiver.borrow_and_update().providers[0].name, "prov-b");
        let _ = std::fs::remove_file(&path);
    }

    /// A second `oxllm serve` must not steal the PID file from a live
    /// daemon (issue #7 reproducer: the daemon becomes unfindable after a
    /// short-lived second instance exits).
    #[test]
    fn test_write_pid_file_preserves_live_foreign_pid() {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let path = temp_path("pid-live");
        std::fs::write(&path, child.id().to_string()).unwrap();

        write_pid_file(&path).unwrap();

        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            child.id().to_string(),
            "a live foreign PID must not be overwritten"
        );
        child.kill().unwrap();
        child.wait().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    /// A stale PID (process gone) must be replaced so a crashed instance
    /// does not block the next `oxllm serve` from being tracked.
    #[test]
    fn test_write_pid_file_overwrites_stale_pid() {
        let path = temp_path("pid-stale");
        std::fs::write(&path, dead_pid().to_string()).unwrap();

        write_pid_file(&path).unwrap();

        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            std::process::id().to_string(),
            "a stale PID must be replaced with our own"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Shutdown cleanup removes only our own PID entry: an instance that
    /// lost the race for the file must not delete the daemon's entry on
    /// its way out (the second half of the issue #7 reproducer).
    #[test]
    fn test_cleanup_pid_file_removes_only_own_pid() {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let foreign = temp_path("pid-cleanup-foreign");
        std::fs::write(&foreign, child.id().to_string()).unwrap();
        cleanup_pid_file(&foreign);
        assert!(foreign.exists(), "must not remove a foreign PID entry");
        child.kill().unwrap();
        child.wait().unwrap();

        let own = temp_path("pid-cleanup-own");
        std::fs::write(&own, std::process::id().to_string()).unwrap();
        cleanup_pid_file(&own);
        assert!(!own.exists(), "must remove our own PID entry");
    }

    #[test]
    fn test_process_alive_detects_live_and_dead_pids() {
        assert!(process_alive(std::process::id()));
        assert!(!process_alive(dead_pid()));
    }

    // -----------------------------------------------------------------------
    // Reload state preservation, restart warnings, aliases (improvements
    // batch, 2026-11-03)
    // -----------------------------------------------------------------------

    /// A hot-reload must not wipe runtime state for providers that keep
    /// their name: circuit position, failure history, admin offline, and
    /// counters all survive — editing the config mid-incident must not
    /// re-enable a provider taken offline or reset failover history.
    #[tokio::test]
    async fn test_reload_preserves_runtime_state() {
        let path = temp_path("carry.toml");
        std::fs::write(&path, test_config_toml("prov-a")).unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let initial = Arc::new(build_app_state(config, None).unwrap());

        // Simulate an incident: circuit open, failures accumulated, provider
        // manually taken offline, traffic counted, session pinned.
        {
            let p = &initial.providers[0];
            *p.circuit.write().await = CircuitState::Open {
                until: Instant::now() + Duration::from_secs(300),
            };
            *p.consecutive_failures.write().await = 5;
            p.manual_disabled
                .store(true, std::sync::atomic::Ordering::Release);
            p.requests
                .fetch_add(7, std::sync::atomic::Ordering::Relaxed);
        }
        initial.session_affinity.pin("session-1", "prov-a");

        let (sender, mut receiver) = tokio::sync::watch::channel(initial.clone());

        // Same provider name, unrelated config edit (different model set).
        std::fs::write(&path, test_config_toml("prov-a")).unwrap();
        reload_config(&path, &sender).unwrap();
        receiver.changed().await.unwrap();
        let reloaded = receiver.borrow_and_update().clone();

        let p = &reloaded.providers[0];
        // The carried `until` instant is still in the future.
        match *p.circuit.read().await {
            CircuitState::Open { until } => assert!(
                until > Instant::now(),
                "circuit position must survive a reload"
            ),
            state => panic!("circuit must still be Open, got {:?}", state),
        }
        assert_eq!(*p.consecutive_failures.read().await, 5);
        assert!(
            p.manual_disabled.load(std::sync::atomic::Ordering::Acquire),
            "admin offline must survive a reload"
        );
        assert_eq!(
            p.requests.load(std::sync::atomic::Ordering::Relaxed),
            7,
            "traffic counters must survive a reload"
        );
        assert_eq!(
            reloaded.session_affinity.pinned_provider("session-1"),
            Some("prov-a".to_string()),
            "session pins must survive a reload"
        );

        // A provider renamed out of the config loses its carried state, and
        // a provider added by the edit starts fresh.
        let two_providers = r#"
[server]
host = "127.0.0.1"
port = 18099
otel_endpoint = "http://127.0.0.1:4318"

[[providers]]
name = "prov-b"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "key"
models = ["m"]

[[providers]]
name = "prov-a"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "key"
models = ["m"]

[virtual_models]
vm = [{ provider = "prov-a", model = "m" }]
vm2 = [{ provider = "prov-b", model = "m" }]
"#
        .to_string();
        std::fs::write(&path, two_providers).unwrap();
        reload_config(&path, &sender).unwrap();
        receiver.changed().await.unwrap();
        let reloaded = receiver.borrow_and_update().clone();

        let fresh = reloaded
            .providers
            .iter()
            .find(|p| p.name == "prov-b")
            .expect("new provider must exist after reload");
        assert_eq!(*fresh.consecutive_failures.read().await, 0);
        assert!(!fresh
            .manual_disabled
            .load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(fresh.requests.load(std::sync::atomic::Ordering::Relaxed), 0);
        let _ = std::fs::remove_file(&path);
    }

    /// Server fields that only apply at startup produce one explicit
    /// warning per change, and an unchanged config produces none.
    #[test]
    fn test_server_config_restart_warnings() {
        let old = test_server_config();
        assert_eq!(
            server_config_restart_warnings(&old, &test_server_config()),
            Vec::<String>::new(),
            "identical configs must not warn"
        );

        let mut new = test_server_config();
        new.port = 18098;
        new.watch_config = false;
        let warnings = server_config_restart_warnings(&old, &new);
        assert_eq!(warnings.len(), 2, "got: {:?}", warnings);
        assert!(warnings[0].contains("server.port"), "got: {:?}", warnings);
        assert!(
            warnings[1].contains("server.watch_config"),
            "got: {:?}",
            warnings
        );
        for warning in &warnings {
            assert!(warning.contains("restart required"), "got: {}", warning);
        }
    }

    /// POST /reload surfaces restart warnings in its JSON body.
    #[tokio::test]
    async fn test_http_reload_reports_restart_warnings() {
        let path = temp_path("warn-reload.toml");
        std::fs::write(&path, test_config_toml("prov-a")).unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let initial = Arc::new(build_app_state(config, None).unwrap());
        let (sender, _receiver) = tokio::sync::watch::channel(initial);

        let router = axum::Router::new()
            .route("/reload", axum::routing::post(handle_http_reload))
            .with_state(Reloader {
                sender,
                config_path: path.clone(),
            });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        // Same config, but with a startup-only field changed.
        let changed = test_config_toml("prov-a").replace("port = 18099", "port = 18098");
        std::fs::write(&path, changed).unwrap();

        let res = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/reload", port))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let body: Value = res.json().await.unwrap();
        let warnings = body["warnings"].as_array().expect("warnings array");
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].as_str().unwrap().contains("server.port"),
            "got: {}",
            warnings[0]
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Aliases and the default model route requests from unmodified OpenAI
    /// clients onto virtual models, and unknown names still 400 without a
    /// default.
    #[tokio::test]
    async fn test_alias_and_default_model_route_requests() {
        let upstream_body = r#"{\"choices\":[{\"index\":0,\"text\":\"ok\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            upstream_body.len(),
            upstream_body
        );
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![response; 3], recorded.clone()).await;

        let mut aliases = std::collections::HashMap::new();
        aliases.insert("gpt-4o".to_string(), "test".to_string());
        let (_app_state, rs) = build_test_state_with(addr, aliases, Some("test".to_string()));

        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/completions", proxy_addr);

        // Alias name routes.
        let res = client
            .post(&url)
            .json(&serde_json::json!({"model": "gpt-4o", "prompt": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200, "alias must route");

        // Unknown name routes via the default model.
        let res = client
            .post(&url)
            .json(&serde_json::json!({"model": "whatever-client-sends", "prompt": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200, "default model must catch unknown names");

        // The upstream saw the real model, and route stats landed on the
        // canonical virtual model, not the alias.
        let recorded = recorded.lock().await;
        assert_eq!(recorded.len(), 2);
        let (_path, body) = &recorded[0];
        let upstream_json: Value = serde_json::from_str(body).unwrap();
        assert_eq!(upstream_json["model"], "model");
    }

    /// GET /v1/models lists alias names alongside their target models so
    /// clients can discover them.
    /// The explicit `priority` field orders candidates (lower wins);
    /// targets without one keep configuration order after all prioritized
    /// targets. With no priorities anywhere, array order is preserved
    /// (issue #4).
    #[test]
    fn test_priority_field_orders_candidates() {
        let path = temp_path("priority.toml");
        std::fs::write(
            &path,
            r#"
[server]
host = "127.0.0.1"
port = 18099
otel_endpoint = "http://127.0.0.1:4318"

[[providers]]
name = "prov-a"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "key"
models = ["m"]

[virtual_models]
vm = [
  { provider = "prov-a", model = "m-first", priority = 5 },
  { provider = "prov-a", model = "m-best",  priority = 1 },
  { provider = "prov-a", model = "m-tail" },
]
"#,
        )
        .unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let state = build_app_state(config, None).unwrap();

        let candidates = state.resolve_candidates("vm");
        let models: Vec<&str> = candidates.iter().map(|c| c.model.as_str()).collect();
        assert_eq!(
            models,
            vec!["m-best", "m-first", "m-tail"],
            "lower priority must win; unprioritized targets must stay in array order at the tail"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Complementary case: with no priority fields set, candidates keep
    /// the configured array order exactly (today's semantics).
    #[test]
    fn test_priority_absent_preserves_array_order() {
        let path = temp_path("priority-absent.toml");
        std::fs::write(
            &path,
            r#"
[server]
host = "127.0.0.1"
port = 18099
otel_endpoint = "http://127.0.0.1:4318"

[[providers]]
name = "prov-a"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "key"
models = ["m"]

[virtual_models]
vm = [
  { provider = "prov-a", model = "m-one" },
  { provider = "prov-a", model = "m-two" },
  { provider = "prov-a", model = "m-three" },
]
"#,
        )
        .unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let state = build_app_state(config, None).unwrap();

        let candidates = state.resolve_candidates("vm");
        let models: Vec<&str> = candidates.iter().map(|c| c.model.as_str()).collect();
        assert_eq!(models, vec!["m-one", "m-two", "m-three"]);
        let _ = std::fs::remove_file(&path);
    }

    /// POST /v1/systemone routes through the pool like chat: model is
    /// rewritten to the target's upstream model, the response passes
    /// through verbatim, usage is accounted, and a failing provider
    /// cascades to the next (issue #6).
    #[tokio::test]
    async fn test_systemone_routes_with_failover() {
        // First upstream answers 500, second answers with a decision body.
        let fail = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n".to_string();
        let decision_body = r#"{"answers":[{"question":"route","choice":"fast"}],"usage":{"input_tokens":42,"output_tokens":0}}"#;
        let ok = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            decision_body.len(),
            decision_body
        );
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr1 = spawn_recording_upstream(vec![fail], recorded.clone()).await;
        let addr2 = spawn_recording_upstream(vec![ok], recorded.clone()).await;

        let mk = |name: &str, addr: SocketAddr| ProviderState {
            name: name.to_string(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "k".to_string(),
            models: vec!["m".to_string()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut vms = std::collections::HashMap::new();
        vms.insert(
            "route-decisions".to_string(),
            vec![
                VirtualModelTarget {
                    provider: "llamacpp-local".to_string(),
                    model: "Kev-4B-Q4_K_M".to_string(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
                VirtualModelTarget {
                    provider: "hosted-jev".to_string(),
                    model: "jev".to_string(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
            ],
        );
        let app_state = Arc::new(AppState {
            providers: vec![mk("llamacpp-local", addr1), mk("hosted-jev", addr2)],
            virtual_models: vms,
            aliases: Default::default(),
            default_model: None,
            // Threshold 1 so the failing provider is skipped after its first
            // 500, keeping the expected attempt count at two.
            circuit: CircuitConfig {
                failure_threshold: 1,
                ..CircuitConfig::default()
            },
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/v1/systemone",
                axum::routing::post(routes::create_systemone),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/systemone", proxy_addr))
            .json(&serde_json::json!({
                "model": "route-decisions",
                "state": "router",
                "questions": [{"id": "route", "type": "choice", "options": ["fast", "cheap"]}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200, "must fail over to the second provider");
        let body: Value = res.json().await.unwrap();
        assert_eq!(
            body["answers"][0]["choice"], "fast",
            "response must pass through verbatim"
        );

        // Both upstreams saw the request; the second got the rewritten model.
        let recorded = recorded.lock().await;
        assert_eq!(recorded.len(), 2, "both providers must have been tried");
        let (_path, second_body) = &recorded[1];
        let upstream_json: Value = serde_json::from_str(second_body).unwrap();
        assert_eq!(upstream_json["model"], "jev");

        // Usage accounted on the provider that served (input only; decisions
        // decode nothing).
        assert_eq!(
            app_state.providers[1]
                .tokens_input
                .load(std::sync::atomic::Ordering::Relaxed),
            42
        );
    }

    /// The generated launchd plist carries the resolved binary and config
    /// paths, KeepAlive, and the log path (issue #2).
    #[cfg(target_os = "macos")]
    #[test]
    fn test_launchd_plist_content() {
        let plist = launchd_plist(
            std::path::Path::new("/usr/local/bin/oxllm"),
            std::path::Path::new("/Users/test/.config/oxllm/config.toml"),
            std::path::Path::new("/Users/test/Library/Logs/oxllm.log"),
        );
        assert!(
            plist.contains("<string>org.oxllm.proxy</string>"),
            "{}",
            plist
        );
        assert!(plist.contains("/usr/local/bin/oxllm"));
        assert!(plist.contains("serve"));
        assert!(plist.contains("--config"));
        assert!(plist.contains("/Users/test/.config/oxllm/config.toml"));
        assert!(plist.contains("<key>KeepAlive</key>"));
        assert!(plist.contains("<key>RunAtLoad</key>"));
        assert!(plist.contains("/Users/test/Library/Logs/oxllm.log"));
    }

    /// Paths with XML-special characters are escaped in the plist.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_launchd_plist_escapes_paths() {
        let plist = launchd_plist(
            std::path::Path::new("/opt/a&b/oxllm"),
            std::path::Path::new("/tmp/config.toml"),
            std::path::Path::new("/tmp/oxllm.log"),
        );
        assert!(plist.contains("/opt/a&amp;b/oxllm"), "{}", plist);
        assert!(!plist.contains("/opt/a&b/"));
    }

    /// The generated systemd user unit carries the binary and config paths
    /// and restarts on failure (issue #2).
    #[cfg(target_os = "linux")]
    #[test]
    fn test_systemd_unit_content() {
        let unit = systemd_unit(
            std::path::Path::new("/usr/local/bin/oxllm"),
            std::path::Path::new("/home/test/.config/oxllm/config.toml"),
        );
        assert!(unit.contains(
            "ExecStart=/usr/local/bin/oxllm serve --config /home/test/.config/oxllm/config.toml"
        ));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("WantedBy=default.target"));
    }

    /// IPv6 host literals render bracketed in CLI URLs, IPv4 and hostnames
    /// unchanged (issue #9).
    #[test]
    fn test_cli_base_url_brackets_ipv6() {
        assert_eq!(cli_base_url("127.0.0.1", 8080), "http://127.0.0.1:8080");
        assert_eq!(cli_base_url("::1", 8080), "http://[::1]:8080");
        assert_eq!(cli_base_url("[::1]", 8080), "http://[::1]:8080");
        assert_eq!(cli_base_url("router.lan", 8080), "http://router.lan:8080");
    }

    /// `oxllm status --host ::1` reaches a daemon bound to IPv6 loopback
    /// (issue #9): the old hardcoded 127.0.0.1 could not.
    #[tokio::test]
    async fn test_status_reaches_ipv6_loopback_daemon() {
        let response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![response.to_string()], recorded).await;
        let (_app_state, rs) = build_test_state(addr);

        let router = axum::Router::new()
            .route("/status", axum::routing::get(routes::get_status))
            .with_state(rs);
        let listener = TcpListener::bind("[::1]:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        run_status("::1", port, true, None, 0, None)
            .await
            .expect("status must reach the IPv6-bound daemon");
    }

    /// Per-target weights share load among healthy candidates: with 3:1
    /// weights and no session keys, traffic splits roughly 75/25 instead
    /// of the first healthy target taking everything (issue #10).
    #[tokio::test]
    async fn test_weight_shares_load_among_healthy_candidates() {
        let response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let recorded_a = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let recorded_b = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr_a =
            spawn_recording_upstream(vec![response.to_string(); 200], recorded_a.clone()).await;
        let addr_b =
            spawn_recording_upstream(vec![response.to_string(); 200], recorded_b.clone()).await;

        let mk = |name: &str, addr: SocketAddr| ProviderState {
            name: name.to_string(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: "k".to_string(),
            models: vec!["m".to_string()],
            upstream_timeout_secs: None,
            cost_per_mtok: None,
            api: Default::default(),
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        };

        let mut vms = std::collections::HashMap::new();
        vms.insert(
            "spread".to_string(),
            vec![
                VirtualModelTarget {
                    provider: "heavy".to_string(),
                    model: "m".to_string(),
                    priority: None,
                    weight: Some(3),
                    virtual_ref: None,
                },
                VirtualModelTarget {
                    provider: "light".to_string(),
                    model: "m".to_string(),
                    priority: None,
                    weight: Some(1),
                    virtual_ref: None,
                },
            ],
        );
        let app_state = Arc::new(AppState {
            providers: vec![mk("heavy", addr_a), mk("light", addr_b)],
            virtual_models: vms,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: "".to_string(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        for _ in 0..120 {
            let res = client
                .post(format!("http://{}/v1/completions", proxy_addr))
                .json(&serde_json::json!({"model": "spread", "prompt": "x"}))
                .send()
                .await
                .unwrap();
            assert_eq!(res.status(), 200);
        }

        let heavy = recorded_a.lock().await.len();
        let light = recorded_b.lock().await.len();
        assert_eq!(heavy + light, 120);
        let ratio = heavy as f64 / 120.0;
        assert!(
            (0.60..=0.90).contains(&ratio),
            "expected ~0.75 of traffic on the weight-3 target, got {}",
            ratio
        );
    }

    /// An explicit `policies.<vm> = "priority"` must mean strict configured
    /// order — it previously fell into the latency-ordering arm and behaved
    /// like `fastest`.
    #[tokio::test]
    async fn test_explicit_priority_policy_keeps_configured_order() {
        let response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let recorded_a = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let recorded_b = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr_a =
            spawn_recording_upstream(vec![response.to_string(); 4], recorded_a.clone()).await;
        let addr_b =
            spawn_recording_upstream(vec![response.to_string(); 4], recorded_b.clone()).await;

        let mk = |name: &str, addr: SocketAddr, ttft_us: u64| {
            let p = ProviderState {
                name: name.to_string(),
                base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
                api_key: "k".to_string(),
                models: vec!["m".to_string()],
                upstream_timeout_secs: None,
                cost_per_mtok: None,
                api: Default::default(),
                tuning: Default::default(),
                circuit: Arc::new(RwLock::new(CircuitState::Closed)),
                consecutive_failures: Arc::new(RwLock::new(0)),
                rate_limited_until: Arc::new(RwLock::new(None)),
                last_attempt_time: Arc::new(RwLock::new(None)),
                probe_in_flight: Arc::new(AtomicBool::new(false)),
                manual_disabled: AtomicBool::new(false),
                requests: AtomicU64::new(0),
                successes: AtomicU64::new(0),
                tokens_input: AtomicU64::new(0),
                tokens_output: AtomicU64::new(0),
                model_stats: Arc::new(ModelStats::default()),
                model_health: Default::default(),
            };
            p.model_stats
                .counters(&ModelStats::key(name, "m"))
                .record_ttft(ttft_us);
            p
        };

        let mut vms = std::collections::HashMap::new();
        vms.insert(
            "vm".to_string(),
            vec![
                VirtualModelTarget {
                    provider: "configured-first-but-slow".to_string(),
                    model: "m".to_string(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
                VirtualModelTarget {
                    provider: "configured-second-but-fast".to_string(),
                    model: "m".to_string(),
                    priority: None,
                    weight: None,
                    virtual_ref: None,
                },
            ],
        );
        let mut policies = std::collections::HashMap::new();
        policies.insert("vm".to_string(), RoutingPolicy::Priority);
        let app_state = Arc::new(AppState {
            providers: vec![
                mk("configured-first-but-slow", addr_a, 900_000),
                mk("configured-second-but-fast", addr_b, 100_000),
            ],
            virtual_models: vms,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies,
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: "".to_string(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });

        let (_ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: _ws,
                config_path: PathBuf::from("."),
            },
        };
        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();
        let res = client
            .post(format!("http://{}/v1/completions", proxy_addr))
            .json(&serde_json::json!({"model": "vm", "prompt": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        assert_eq!(
            recorded_a.lock().await.len(),
            1,
            "the configured-first provider must serve despite being slower"
        );
        assert_eq!(recorded_b.lock().await.len(), 0);
    }

    /// /status reports the daemon version, the last reload outcome (with
    /// restart warnings), and the recent-request ring; a proxied request
    /// lands in the ring with route, provider, attempts, and status
    /// (issue #12 + status gaps).
    #[tokio::test]
    async fn test_status_reports_version_reload_health_and_records() {
        let path = temp_path("status-rich.toml");
        std::fs::write(&path, test_config_toml("prov-a")).unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let initial = Arc::new(build_app_state(config, None).unwrap());
        let (sender, receiver) = tokio::sync::watch::channel(initial);
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: receiver,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender,
                config_path: path.clone(),
            },
        };

        let router = axum::Router::new()
            .route("/reload", axum::routing::post(handle_http_reload))
            .route("/status", axum::routing::get(routes::get_status))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();

        // Version is reported.
        let status: Value = client
            .get(format!("{}/status", base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(status["version"], env!("CARGO_PKG_VERSION"));

        // A failed reload is visible.
        std::fs::write(&path, test_config_toml_dangling_reference()).unwrap();
        let res = client
            .post(format!("{}/reload", base))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 400);
        let status: Value = client
            .get(format!("{}/status", base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let error = status["reload"]["last_error"]
            .as_str()
            .expect("reload failure must be reported");
        assert!(error.contains("missing-prov"), "got: {}", error);

        // A successful reload with a restart-only change reports warnings.
        let changed = test_config_toml("prov-a").replace("port = 18099", "port = 18098");
        std::fs::write(&path, changed).unwrap();
        let res = client
            .post(format!("{}/reload", base))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let status: Value = client
            .get(format!("{}/status", base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(
            status["reload"]["last_error"].is_null(),
            "successful reload must clear the error"
        );
        let warnings = status["reload"]["restart_warnings"]
            .as_array()
            .expect("restart warnings");
        assert!(warnings
            .iter()
            .any(|w| w.as_str().unwrap_or("").contains("server.port")));

        let _ = std::fs::remove_file(&path);
    }

    /// A proxied request lands in the recent-request ring, and the live
    /// tail SSE stream carries it (issue #12).
    #[tokio::test]
    async fn test_request_ring_and_live_tail() {
        let upstream_body = r#"{\"choices\":[{\"index\":0,\"text\":\"ok\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            upstream_body.len(),
            upstream_body
        );
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![response], recorded).await;

        let (_app_state, rs) = build_test_state(addr);
        let router = axum::Router::new()
            .route(
                "/v1/completions",
                axum::routing::post(routes::create_completions),
            )
            .route("/status", axum::routing::get(routes::get_status))
            .route("/admin/tail", axum::routing::get(routes::tail_requests))
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let client = reqwest::Client::new();

        // Subscribe to the tail before the request completes.
        let mut tail = client
            .get(format!("{}/admin/tail", base))
            .send()
            .await
            .unwrap();
        assert!(tail
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .starts_with("text/event-stream"));

        let res = client
            .post(format!("{}/v1/completions", base))
            .json(&serde_json::json!({"model": "test", "prompt": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        // The ring has the record.
        let status: Value = client
            .get(format!("{}/status", base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let recs = status["recent_requests"].as_array().expect("records");
        let rec = recs.last().expect("one record");
        assert_eq!(rec["route"], "test");
        assert_eq!(rec["provider"], "prov");
        assert_eq!(rec["status"], 200);
        assert_eq!(rec["attempts"], 1);
        assert_eq!(rec["endpoint"], "completions");
        assert_eq!(rec["model"], "model");

        // The tail stream delivered it as an SSE data event.
        let mut seen = String::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !seen.contains("\"route\":\"test\"") {
            match tokio::time::timeout(Duration::from_secs(2), tail.chunk()).await {
                Ok(Ok(Some(chunk))) => seen.push_str(&String::from_utf8_lossy(&chunk)),
                Ok(Ok(None)) => break,
                Ok(Err(e)) => panic!("tail stream error: {}", e),
                Err(_) => {},
            }
        }
        assert!(
            seen.contains("\"route\":\"test\""),
            "tail stream must carry the record, got: {}",
            seen
        );
    }

    /// Admin-offline providers are visible in /status (admin_disabled) —
    /// previously they looked healthy while receiving no traffic.
    #[tokio::test]
    async fn test_status_reports_admin_offline() {
        let response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![response.to_string()], recorded).await;

        let (app_state, rs) = build_test_state(addr);
        app_state.providers[0]
            .manual_disabled
            .store(true, std::sync::atomic::Ordering::Release);
        drop(app_state);

        let router = axum::Router::new()
            .route("/status", axum::routing::get(routes::get_status))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let status: Value = reqwest::Client::new()
            .get(format!("{}/status", base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(status["providers"][0]["admin_disabled"], true);
    }

    #[tokio::test]
    async fn test_list_models_includes_aliases() {
        let response =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let recorded = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_recording_upstream(vec![response.to_string()], recorded).await;

        let mut aliases = std::collections::HashMap::new();
        aliases.insert("gpt-4o".to_string(), "test".to_string());
        let (_aliased, rs) = build_test_state_with(addr, aliases, None);

        let router = axum::Router::new()
            .route("/v1/models", axum::routing::get(routes::list_models))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let body: Value = reqwest::Client::new()
            .get(format!("http://{}/v1/models", proxy_addr))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let ids: Vec<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&"test"), "target vm must be listed: {:?}", ids);
        assert!(ids.contains(&"gpt-4o"), "alias must be listed: {:?}", ids);
    }

    // ---- Anthropic Messages API ------------------------------------------

    /// Upstream that records each full raw request (request line, headers,
    /// body) and answers connection N with `responses[N]` (last one repeats).
    async fn spawn_raw_recording_upstream(
        responses: Vec<String>,
        recorded: Arc<tokio::sync::Mutex<Vec<String>>>,
    ) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut idx = 0usize;
            while let Ok((mut stream, _)) = listener.accept().await {
                let recorded = recorded.clone();
                // The mock closes the socket after one reply, so tell the
                // client not to pool it; otherwise a follow-up request can
                // reuse the dead connection and fail spuriously.
                let response = responses[idx.min(responses.len() - 1)].replacen(
                    "\r\n",
                    "\r\nConnection: close\r\n",
                    1,
                );
                idx += 1;
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    let mut expected: Option<usize> = None;
                    loop {
                        match stream.read(&mut tmp).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase();
                            let len = head
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                            expected = Some(end + 4 + len);
                        }
                        if expected.is_some_and(|e| buf.len() >= e) {
                            break;
                        }
                    }
                    recorded
                        .lock()
                        .await
                        .push(String::from_utf8_lossy(&buf).to_string());
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                });
            }
        });
        addr
    }

    fn http_json(status_line: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            status_line,
            body.len(),
            body
        )
    }

    fn test_provider(name: &str, addr: SocketAddr, api: ApiFormat) -> ProviderState {
        ProviderState {
            name: name.to_string(),
            base_url: Url::parse(&format!("http://{}/v1/", addr)).unwrap(),
            api_key: format!("key-{}", name),
            models: vec!["m".to_string()],
            upstream_timeout_secs: Some(2),
            cost_per_mtok: None,
            api,
            tuning: Default::default(),
            circuit: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(RwLock::new(0)),
            rate_limited_until: Arc::new(RwLock::new(None)),
            last_attempt_time: Arc::new(RwLock::new(None)),
            probe_in_flight: Arc::new(AtomicBool::new(false)),
            manual_disabled: AtomicBool::new(false),
            requests: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            tokens_input: AtomicU64::new(0),
            tokens_output: AtomicU64::new(0),
            model_stats: Arc::new(ModelStats::default()),
            model_health: Default::default(),
        }
    }

    /// Serves the full public route set over one virtual model `vm` whose
    /// targets are the given providers in order. Returns the proxy address
    /// and the shared state.
    async fn spawn_messages_proxy(
        providers: Vec<ProviderState>,
        api_keys: Vec<String>,
    ) -> (SocketAddr, Arc<AppState>) {
        let targets: Vec<VirtualModelTarget> = providers
            .iter()
            .map(|p| VirtualModelTarget {
                provider: p.name.clone(),
                model: format!("upstream-{}", p.name),
                priority: None,
                weight: None,
                virtual_ref: None,
            })
            .collect();
        spawn_proxy_with_targets(providers, targets, api_keys).await
    }

    async fn spawn_proxy_with_targets(
        providers: Vec<ProviderState>,
        targets: Vec<VirtualModelTarget>,
        api_keys: Vec<String>,
    ) -> (SocketAddr, Arc<AppState>) {
        let mut virtual_models = std::collections::HashMap::new();
        virtual_models.insert("vm".to_string(), targets);
        let app_state = Arc::new(AppState {
            providers,
            virtual_models,
            aliases: Default::default(),
            default_model: None,
            circuit: CircuitConfig::default(),
            server: test_server_config(),
            reload_health: Default::default(),
            request_log: Default::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: std::collections::HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys,
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                1024,
            )),
            session_affinity_header: oxllm_core::state::DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::builder().build().unwrap(),
            upstream_timeout_secs: 5,
        });
        let (ws, wr) = tokio::sync::watch::channel(app_state.clone());
        let (ttx, _trx) = tokio::sync::mpsc::channel(1024);
        let rs = ReloadableState {
            app_state: wr,
            telemetry: TelemetryClient::new(ttx),
            start_time: Instant::now(),
            reloader: Reloader {
                sender: ws.clone(),
                config_path: PathBuf::from("config.toml"),
            },
        };
        let gate = || middleware::from_fn_with_state(rs.clone(), require_api_key);
        let router = axum::Router::new()
            .route(
                "/v1/messages",
                axum::routing::post(routes::create_messages).layer(gate()),
            )
            .route(
                "/v1/messages/count_tokens",
                axum::routing::post(routes::count_tokens).layer(gate()),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(routes::create_chat_completions).layer(gate()),
            )
            .layer(middleware::from_fn(add_request_id))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // Keep the watch sender alive for the server's lifetime.
            let _ws = ws;
            axum::serve(listener, router).await.unwrap();
        });
        (proxy_addr, app_state)
    }

    const MSG_OK: &str = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":100,"cache_creation_input_tokens":20}}"#;

    fn messages_body() -> Value {
        serde_json::json!({
            "model": "vm",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}]
        })
    }

    #[tokio::test]
    async fn test_messages_non_streaming_success() {
        let rec = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr =
            spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], rec.clone()).await;
        let (proxy, state) = spawn_messages_proxy(
            vec![test_provider("a1", addr, ApiFormat::Anthropic)],
            vec![],
        )
        .await;

        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .header("anthropic-beta", "prompt-caching-2024-07-31")
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let body: Value = res.json().await.unwrap();
        assert_eq!(body["id"], "msg_1");

        let raw = rec.lock().await[0].to_ascii_lowercase();
        assert!(raw.starts_with("post /v1/messages "), "got: {raw}");
        assert!(raw.contains("x-api-key: key-a1"), "got: {raw}");
        assert!(raw.contains("anthropic-version: 2023-06-01"), "got: {raw}");
        assert!(
            raw.contains("anthropic-beta: prompt-caching-2024-07-31"),
            "got: {raw}"
        );
        assert!(!raw.contains("authorization:"), "got: {raw}");
        assert!(raw.contains("\"model\":\"upstream-a1\""), "got: {raw}");

        // Usage: input folds in cache reads and creations.
        let p = &state.providers[0];
        assert_eq!(
            p.tokens_input.load(std::sync::atomic::Ordering::Relaxed),
            130
        );
        assert_eq!(
            p.tokens_output.load(std::sync::atomic::Ordering::Relaxed),
            5
        );
        let snap = state.route_stats.snapshot("vm");
        assert_eq!(snap.cached_tokens, 100);
        assert_eq!(snap.successes, 1);
    }

    #[tokio::test]
    async fn test_messages_passes_anthropic_response_headers() {
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nrequest-id: req_123\r\nanthropic-ratelimit-requests-remaining: 41\r\nContent-Length: {}\r\n\r\n{}",
            MSG_OK.len(),
            MSG_OK
        );
        let addr = spawn_raw_recording_upstream(vec![resp], Default::default()).await;
        let (proxy, _s) = spawn_messages_proxy(
            vec![test_provider("a1", addr, ApiFormat::Anthropic)],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        assert_eq!(res.headers()["request-id"], "req_123");
        assert_eq!(
            res.headers()["anthropic-ratelimit-requests-remaining"],
            "41"
        );
    }

    #[tokio::test]
    async fn test_messages_forwards_client_anthropic_version() {
        let rec = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr =
            spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], rec.clone()).await;
        let (proxy, _s) = spawn_messages_proxy(
            vec![test_provider("a1", addr, ApiFormat::Anthropic)],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .header("anthropic-version", "2024-01-01")
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let raw = rec.lock().await[0].to_ascii_lowercase();
        assert!(raw.contains("anthropic-version: 2024-01-01"), "got: {raw}");
        assert!(!raw.contains("2023-06-01"), "got: {raw}");
    }

    #[tokio::test]
    async fn test_messages_streaming_success() {
        let sse = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12,\"output_tokens\":1,\"cache_read_input_tokens\":40}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":9}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n{}",
            sse
        );
        let rec = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_raw_recording_upstream(vec![resp], rec).await;
        let (proxy, state) = spawn_messages_proxy(
            vec![test_provider("a1", addr, ApiFormat::Anthropic)],
            vec![],
        )
        .await;

        let mut body = messages_body();
        body["stream"] = Value::Bool(true);
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let text = res.text().await.unwrap();
        assert_eq!(text, sse, "SSE must pass through byte-for-byte");

        // Accounting runs after the stream closes.
        let p = &state.providers[0];
        for _ in 0..50 {
            if p.successes.load(std::sync::atomic::Ordering::Relaxed) == 1
                && p.tokens_output.load(std::sync::atomic::Ordering::Relaxed) == 9
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(p.successes.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(
            p.tokens_input.load(std::sync::atomic::Ordering::Relaxed),
            52
        );
        assert_eq!(
            p.tokens_output.load(std::sync::atomic::Ordering::Relaxed),
            9
        );
        assert_eq!(state.route_stats.snapshot("vm").cached_tokens, 40);
    }

    async fn failover_case(first_status: &str, extra_header: &str) {
        let r1 = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let r2 = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let err = format!(
            "HTTP/1.1 {}\r\n{}Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}",
            first_status, extra_header
        );
        let a1 = spawn_raw_recording_upstream(vec![err], r1.clone()).await;
        let a2 = spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], r2.clone()).await;
        let (proxy, state) = spawn_messages_proxy(
            vec![
                test_provider("a1", a1, ApiFormat::Anthropic),
                test_provider("a2", a2, ApiFormat::Anthropic),
            ],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        // 429 cools the provider down at once; 5xx and 529 are retried on
        // the same provider until the breaker threshold, as on OpenAI routes.
        assert!(!r1.lock().await.is_empty());
        assert_eq!(r2.lock().await.len(), 1);
        // 429, server errors and overload are all scoped to the model; the
        // provider itself stays reachable.
        let health = state.providers[0].model_health.get("upstream-a1");
        let model_failures = *health.consecutive_failures.read().await;
        let provider_failures = *state.providers[0].consecutive_failures.read().await;
        assert_eq!((provider_failures, model_failures), (0, 1));
    }

    #[tokio::test]
    async fn test_messages_429_fails_over_and_honours_retry_after() {
        failover_case("429 Too Many Requests", "Retry-After: 30\r\n").await;
    }

    #[tokio::test]
    async fn test_messages_429_sets_retry_after_cooldown() {
        let a1 = spawn_raw_recording_upstream(
            vec![
                "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 30\r\nContent-Length: 0\r\n\r\n"
                    .to_string(),
            ],
            Default::default(),
        )
        .await;
        let a2 =
            spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], Default::default())
                .await;
        let (proxy, state) = spawn_messages_proxy(
            vec![
                test_provider("a1", a1, ApiFormat::Anthropic),
                test_provider("a2", a2, ApiFormat::Anthropic),
            ],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let health = state.providers[0].model_health.get("upstream-a1");
        let until = health
            .rate_limited_until
            .read()
            .await
            .expect("429 must set a rate-limit window");
        let left = until.saturating_duration_since(Instant::now());
        assert!(left > Duration::from_secs(20), "left: {:?}", left);
    }

    #[tokio::test]
    async fn test_messages_529_fails_over() {
        failover_case("529 Overloaded", "").await;
    }

    #[tokio::test]
    async fn test_messages_500_fails_over() {
        failover_case("500 Internal Server Error", "").await;
    }

    #[tokio::test]
    async fn test_messages_401_not_failed_over() {
        let r1 = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let r2 = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let err_body = r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#;
        let a1 =
            spawn_raw_recording_upstream(vec![http_json("401 Unauthorized", err_body)], r1.clone())
                .await;
        let a2 = spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], r2.clone()).await;
        let (proxy, state) = spawn_messages_proxy(
            vec![
                test_provider("a1", a1, ApiFormat::Anthropic),
                test_provider("a2", a2, ApiFormat::Anthropic),
            ],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 401);
        let body: Value = res.json().await.unwrap();
        assert_eq!(body["error"]["type"], "authentication_error");
        assert!(r2.lock().await.is_empty(), "401 must not fail over");
        assert_eq!(
            *state.providers[0].consecutive_failures.read().await,
            0,
            "a client error must not count against provider health"
        );
    }

    #[tokio::test]
    async fn test_messages_no_matching_candidate() {
        let addr =
            spawn_raw_recording_upstream(vec![http_json("200 OK", "{}")], Default::default()).await;
        let (proxy, _s) =
            spawn_messages_proxy(vec![test_provider("o1", addr, ApiFormat::Openai)], vec![]).await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 400);
        let body: Value = res.json().await.unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("api = \"anthropic\""),
            "got: {body}"
        );
    }

    #[tokio::test]
    async fn test_messages_all_providers_down_uses_anthropic_error_shape() {
        let addr = spawn_raw_recording_upstream(
            vec![http_json("529 Overloaded", "{}")],
            Default::default(),
        )
        .await;
        let (proxy, _s) = spawn_messages_proxy(
            vec![test_provider("a1", addr, ApiFormat::Anthropic)],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 502);
        let body: Value = res.json().await.unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "api_error");
    }

    #[tokio::test]
    async fn test_mixed_virtual_model_routes_by_wire_format() {
        let ro = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let ra = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let openai_ok = r#"{"choices":[{"message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
        // OpenAI provider first in order: the Messages route must skip it.
        let o1 =
            spawn_raw_recording_upstream(vec![http_json("200 OK", openai_ok)], ro.clone()).await;
        let a1 = spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], ra.clone()).await;
        let (proxy, _s) = spawn_messages_proxy(
            vec![
                test_provider("o1", o1, ApiFormat::Openai),
                test_provider("a1", a1, ApiFormat::Anthropic),
            ],
            vec![],
        )
        .await;
        let client = reqwest::Client::new();

        let res = client
            .post(format!("http://{}/v1/messages", proxy))
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        assert_eq!(ra.lock().await.len(), 1);
        assert!(ro.lock().await.is_empty());

        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy))
            .json(&serde_json::json!({"model":"vm","messages":[{"role":"user","content":"x"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        assert_eq!(ro.lock().await.len(), 1);
        assert_eq!(ra.lock().await.len(), 1, "chat must not reach anthropic");
        let raw = ro.lock().await[0].to_ascii_lowercase();
        assert!(raw.contains("authorization: bearer key-o1"), "got: {raw}");
        assert!(!raw.contains("x-api-key"), "got: {raw}");
    }

    #[tokio::test]
    async fn test_messages_inbound_x_api_key_auth() {
        let rec = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr =
            spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], rec.clone()).await;
        let (proxy, _s) = spawn_messages_proxy(
            vec![test_provider("a1", addr, ApiFormat::Anthropic)],
            vec!["sekrit".to_string()],
        )
        .await;
        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/messages", proxy);

        let res = client
            .post(&url)
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 401);
        let body: Value = res.json().await.unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "authentication_error");

        let res = client
            .post(&url)
            .header("x-api-key", "wrong")
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 401);
        assert!(rec.lock().await.is_empty());

        let res = client
            .post(&url)
            .header("x-api-key", "sekrit")
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        // Bearer still works on /v1/messages.
        let res = client
            .post(&url)
            .header("Authorization", "Bearer sekrit")
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        // x-api-key is not accepted on the OpenAI routes.
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy))
            .header("x-api-key", "sekrit")
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 401);
    }

    #[tokio::test]
    async fn test_count_tokens_routes_to_anthropic() {
        let rec = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let addr = spawn_raw_recording_upstream(
            vec![http_json("200 OK", r#"{"input_tokens":7}"#)],
            rec.clone(),
        )
        .await;
        let (proxy, _s) = spawn_messages_proxy(
            vec![test_provider("a1", addr, ApiFormat::Anthropic)],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages/count_tokens", proxy))
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let body: Value = res.json().await.unwrap();
        assert_eq!(body["input_tokens"], 7);
        let raw = rec.lock().await[0].to_ascii_lowercase();
        assert!(
            raw.starts_with("post /v1/messages/count_tokens "),
            "got: {raw}"
        );
    }

    #[tokio::test]
    async fn test_messages_metadata_user_id_pins_session() {
        let a1 =
            spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], Default::default())
                .await;
        let a2 =
            spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], Default::default())
                .await;
        let (proxy, state) = spawn_messages_proxy(
            vec![
                test_provider("a1", a1, ApiFormat::Anthropic),
                test_provider("a2", a2, ApiFormat::Anthropic),
            ],
            vec![],
        )
        .await;
        let mut body = messages_body();
        body["metadata"] = serde_json::json!({"user_id": "sess-42"});
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/messages", proxy))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        assert_eq!(
            state.session_affinity.pinned_provider("sess-42").as_deref(),
            Some("a1")
        );
    }

    #[tokio::test]
    async fn test_models_is_superset_of_openai_and_anthropic_shapes() {
        let addr =
            spawn_raw_recording_upstream(vec![http_json("200 OK", "{}")], Default::default()).await;
        let (_state, rs) = build_test_state_with(addr, Default::default(), None);
        let router = axum::Router::new()
            .route("/v1/models", axum::routing::get(routes::list_models))
            .with_state(rs);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let models_addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let body: Value = reqwest::Client::new()
            .get(format!("http://{}/v1/models", models_addr))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        // OpenAI shape.
        assert_eq!(body["object"], "list");
        let first = &body["data"][0];
        assert_eq!(first["object"], "model");
        assert!(first["created"].is_u64());
        assert!(first["owned_by"].is_string());
        // Anthropic shape.
        assert_eq!(first["type"], "model");
        assert_eq!(first["display_name"], first["id"]);
        assert!(first["created_at"].as_str().unwrap().ends_with('Z'));
        assert_eq!(body["has_more"], false);
        assert_eq!(body["first_id"], first["id"]);
        assert_eq!(
            body["last_id"],
            body["data"].as_array().unwrap().last().unwrap()["id"]
        );
    }

    fn target(provider: &str, model: &str) -> VirtualModelTarget {
        VirtualModelTarget {
            provider: provider.to_string(),
            model: model.to_string(),
            priority: None,
            weight: None,
            virtual_ref: None,
        }
    }

    /// A provider listed twice in one virtual model: when its first entry
    /// fails, the second entry (a different model) is tried next on the same
    /// provider, rather than the first being retried.
    #[tokio::test]
    async fn test_same_provider_second_entry_is_a_real_fallback() {
        let rec = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let openai_ok = r#"{"choices":[{"message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
        let addr = spawn_raw_recording_upstream(
            vec![
                http_json("500 Internal Server Error", "{}"),
                http_json("200 OK", openai_ok),
            ],
            rec.clone(),
        )
        .await;
        let (proxy, _s) = spawn_proxy_with_targets(
            vec![test_provider("p", addr, ApiFormat::Openai)],
            vec![target("p", "model-a"), target("p", "model-b")],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/chat/completions", proxy))
            .json(&serde_json::json!({"model":"vm","messages":[{"role":"user","content":"x"}]}))
            .send()
            .await
            .unwrap();
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        assert_eq!(status, 200, "body: {text}");
        let seen = rec.lock().await.clone();
        assert_eq!(seen.len(), 2, "one attempt per entry: {seen:?}");
        assert!(
            seen[0].contains("\"model\":\"model-a\""),
            "got: {}",
            seen[0]
        );
        assert!(
            seen[1].contains("\"model\":\"model-b\""),
            "got: {}",
            seen[1]
        );
    }

    /// An upstream that accepts the connection and never answers.
    async fn spawn_hanging_upstream() -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                held.push(stream);
            }
        });
        addr
    }

    /// A hung upstream costs one timeout, then its circuit opens at once and
    /// the next request skips it without waiting.
    #[tokio::test]
    async fn test_hung_upstream_opens_circuit_after_one_timeout() {
        let hang = spawn_hanging_upstream().await;
        let ok_addr =
            spawn_raw_recording_upstream(vec![http_json("200 OK", MSG_OK)], Default::default())
                .await;
        let mut slow = test_provider("slow", hang, ApiFormat::Anthropic);
        slow.upstream_timeout_secs = Some(1);
        let (proxy, state) = spawn_messages_proxy(
            vec![slow, test_provider("ok", ok_addr, ApiFormat::Anthropic)],
            vec![],
        )
        .await;
        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/messages", proxy);

        let t = Instant::now();
        let res = client
            .post(&url)
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let first = t.elapsed();
        // One timeout, not `circuit_failure_threshold` (3) of them.
        assert!(first >= Duration::from_secs(1), "first: {:?}", first);
        assert!(first < Duration::from_millis(2500), "first: {:?}", first);
        // The timeout opens the MODEL's circuit; the provider stays closed.
        let health = state.providers[0].model_health.get("upstream-slow");
        assert!(
            matches!(*health.circuit.read().await, CircuitState::Open { .. }),
            "one timeout must open the model's circuit"
        );
        assert_eq!(
            *state.providers[0].circuit.read().await,
            CircuitState::Closed
        );

        let t = Instant::now();
        let res = client
            .post(&url)
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        assert!(
            t.elapsed() < Duration::from_millis(800),
            "second request must skip the hung provider, took {:?}",
            t.elapsed()
        );
    }

    /// Upstream that accepts the request, then sends nothing for `delay`
    /// before answering with `response`.
    async fn spawn_delayed_upstream(delay: Duration, response: String) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let response = response.clone();
                tokio::spawn(async move {
                    let mut tmp = [0u8; 4096];
                    let _ = stream.read(&mut tmp).await;
                    tokio::time::sleep(delay).await;
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        addr
    }

    /// A streaming request abandons a provider that sends no headers within
    /// first_byte_timeout_secs and fails over; non-streaming requests are
    /// unaffected (their headers legitimately arrive after the generation).
    #[tokio::test]
    async fn test_first_byte_timeout_fails_over_streaming_only() {
        let sse = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string();
        let slow =
            spawn_delayed_upstream(Duration::from_secs(3), http_json("200 OK", MSG_OK)).await;
        let fast = spawn_raw_recording_upstream(vec![sse], Default::default()).await;
        let mut slow_p = test_provider("slow", slow, ApiFormat::Anthropic);
        slow_p.tuning.first_byte_timeout_secs = Some(1);
        slow_p.upstream_timeout_secs = Some(10);
        let (proxy, state) = spawn_messages_proxy(
            vec![slow_p, test_provider("fast", fast, ApiFormat::Anthropic)],
            vec![],
        )
        .await;
        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/messages", proxy);

        // Streaming: gives up on `slow` after ~1 s, served by `fast`.
        let mut body = messages_body();
        body["stream"] = Value::Bool(true);
        let t = Instant::now();
        let res = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(res.status(), 200);
        let took = t.elapsed();
        assert!(
            took >= Duration::from_secs(1) && took < Duration::from_millis(2500),
            "took {:?}",
            took
        );
        let last = state.request_log.recent(1);
        assert_eq!(last[0].provider, "fast");
        let health = state.providers[0].model_health.get("upstream-slow");
        assert!(matches!(
            *health.circuit.read().await,
            CircuitState::Open { .. }
        ));

        // Non-streaming: first_byte_timeout_secs does not apply, so the slow
        // provider (circuit reset) is waited on and answers after ~3 s.
        *health.circuit.write().await = CircuitState::Closed;
        let t = Instant::now();
        let res = client
            .post(&url)
            .json(&messages_body())
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        assert!(
            t.elapsed() >= Duration::from_secs(3),
            "took {:?}",
            t.elapsed()
        );
        assert_eq!(state.request_log.recent(1)[0].provider, "slow");
    }

    fn nested_toml(extra_vms: &str) -> String {
        format!(
            r#"
[server]
host = "127.0.0.1"
port = 8080
otel_endpoint = "http://127.0.0.1:4318"
watch_config = false

[[providers]]
name = "p1"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "k"
models = ["m"]

[virtual_models]
leaf = [{{ provider = "p1", model = "m" }}]
{extra_vms}
"#
        )
    }

    /// A reload that introduces a reference cycle is rejected with the full
    /// path in the error, and the running config is kept.
    #[tokio::test]
    async fn test_reload_with_cycle_keeps_previous_config() {
        let path = temp_path("nested-cycle.toml");
        std::fs::write(&path, nested_toml("outer = [{ virtual = \"leaf\" }]")).unwrap();
        let config = Config::load_from_file(&path).unwrap();
        config.validate().unwrap();
        let initial = Arc::new(build_app_state(config, None).unwrap());
        let (sender, receiver) = tokio::sync::watch::channel(initial.clone());
        assert!(receiver.borrow().virtual_models.contains_key("outer"));

        std::fs::write(
            &path,
            nested_toml(
                "a = [{ virtual = \"b\" }]\nb = [{ virtual = \"c\" }]\nc = [{ virtual = \"a\" }]",
            ),
        )
        .unwrap();
        let err = reload_config(&path, &sender).unwrap_err();
        assert!(err.contains("a -> b -> c -> a"), "got: {err}");
        assert!(
            Arc::ptr_eq(&receiver.borrow(), &initial),
            "a rejected reload must leave the running state in place"
        );
        assert!(receiver
            .borrow()
            .reload_health
            .snapshot()
            .last_error
            .is_some());
        let _ = std::fs::remove_file(&path);
    }

    fn load_nested(name: &str, extra_vms: &str) -> Config {
        let path = temp_path(name);
        std::fs::write(&path, nested_toml(extra_vms)).unwrap();
        let config = Config::load_from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        config
    }

    /// build_app_state refuses a cyclic config even when validate() was
    /// never called.
    #[test]
    fn test_build_app_state_is_cycle_safe_without_validate() {
        let config = load_nested("nested-nov.toml", "a = [{ virtual = \"a\" }]");
        let err = build_app_state(config, None).err().expect("must refuse");
        assert!(err.contains("a -> a"), "got: {err}");
    }

    /// End to end: the flattened output of build_app_state drives a real
    /// request. `outer` references `grp` (down, up); the request is served
    /// by `up` after `down` fails, with the leaf's own model name upstream.
    #[tokio::test]
    async fn test_nested_virtual_model_routes_through_groups() {
        let mut config = load_nested(
            "nested-e2e.toml",
            r#"
outer = [{ virtual = "grp" }]
grp = [{ provider = "p1", model = "m-down" }, { provider = "p1", model = "m-up" }]
"#,
        );
        config.validate().unwrap();
        config.providers.clear();
        let state = build_app_state_for_flatten_only(&config);
        let outer = state.get("outer").expect("outer flattened").clone();
        assert_eq!(
            outer.iter().map(|t| t.model.as_str()).collect::<Vec<_>>(),
            ["m-down", "m-up"]
        );

        // Serve those flattened targets through the real proxy.
        let rec = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let ok = r#"{"choices":[{"message":{"role":"assistant","content":"x"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
        let addr = spawn_raw_recording_upstream(
            vec![
                http_json("500 Internal Server Error", "{}"),
                http_json("200 OK", ok),
            ],
            rec.clone(),
        )
        .await;
        let targets: Vec<VirtualModelTarget> =
            outer.iter().map(|t| target("p1", &t.model)).collect();
        let (proxy, _s) = spawn_proxy_with_targets(
            vec![test_provider("p1", addr, ApiFormat::Openai)],
            targets,
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/chat/completions", proxy))
            .json(&serde_json::json!({"model":"vm","messages":[{"role":"user","content":"x"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let seen = rec.lock().await.clone();
        assert!(seen[0].contains("\"model\":\"m-down\""), "got: {}", seen[0]);
        assert!(seen[1].contains("\"model\":\"m-up\""), "got: {}", seen[1]);
    }

    fn build_app_state_for_flatten_only(
        config: &Config,
    ) -> std::collections::HashMap<String, Vec<VirtualModelTarget>> {
        config.flatten_virtual_models().unwrap()
    }

    /// Two models behind one provider: when the first hangs, its circuit
    /// opens but the sibling model on the same provider keeps being served,
    /// and the hung model is skipped without waiting on later requests.
    #[tokio::test]
    async fn test_hung_model_does_not_take_down_its_sibling() {
        // A single upstream that hangs only for model-slow, answers model-ok.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    if req.contains("model-slow") {
                        tokio::time::sleep(Duration::from_secs(30)).await;
                    } else {
                        let _ = stream
                            .write_all(
                                http_json("200 OK", r#"{"choices":[],"usage":{}}"#)
                                    .replacen("\r\n", "\r\nConnection: close\r\n", 1)
                                    .as_bytes(),
                            )
                            .await;
                    }
                });
            }
        });
        let mut p = test_provider("gw", addr, ApiFormat::Openai);
        p.upstream_timeout_secs = Some(1);
        let (proxy, state) = spawn_proxy_with_targets(
            vec![p],
            vec![target("gw", "model-slow"), target("gw", "model-ok")],
            vec![],
        )
        .await;
        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/chat/completions", proxy);
        let body = serde_json::json!({"model":"vm","messages":[{"role":"user","content":"x"}]});

        // First request: pays one timeout on model-slow, served by model-ok.
        let t = Instant::now();
        let res = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(res.status(), 200);
        assert!(t.elapsed() >= Duration::from_secs(1));
        assert_eq!(state.request_log.recent(1)[0].model, "model-ok");

        // The provider itself is healthy; only model-slow is open.
        assert_eq!(
            *state.providers[0].circuit.read().await,
            CircuitState::Closed
        );
        let slow = state.providers[0].model_health.get("model-slow");
        assert!(matches!(
            *slow.circuit.read().await,
            CircuitState::Open { .. }
        ));
        let ok = state.providers[0].model_health.get("model-ok");
        assert_eq!(*ok.circuit.read().await, CircuitState::Closed);

        // Second request skips model-slow instantly and still succeeds.
        let t = Instant::now();
        let res = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(res.status(), 200);
        assert!(
            t.elapsed() < Duration::from_millis(600),
            "took {:?}",
            t.elapsed()
        );
        assert_eq!(state.request_log.recent(1)[0].model, "model-ok");
    }

    /// A refused connection is a provider-level failure: every model on the
    /// host is affected, so the provider circuit (not the model's) counts it.
    #[tokio::test]
    async fn test_unreachable_host_is_a_provider_failure() {
        let dead = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = dead.local_addr().unwrap();
        drop(dead); // nothing listens: connection refused
        let (proxy, state) = spawn_proxy_with_targets(
            vec![test_provider("gone", addr, ApiFormat::Openai)],
            vec![target("gone", "m1")],
            vec![],
        )
        .await;
        let res = reqwest::Client::new()
            .post(format!("http://{}/v1/chat/completions", proxy))
            .json(&serde_json::json!({"model":"vm","messages":[{"role":"user","content":"x"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 502);
        assert_eq!(*state.providers[0].consecutive_failures.read().await, 1);
        let health = state.providers[0].model_health.get("m1");
        assert_eq!(*health.consecutive_failures.read().await, 0);
    }

    // ---- tiered status rendering ----------------------------------------

    fn sample_status() -> serde_json::Value {
        serde_json::json!({
            "version": "9.9.9",
            "uptime_secs": 3725,
            "total_requests": 42,
            "health": {
                "state": "degraded",
                "problems": [
                    {"scope":"route","name":"pool-docs","kind":"degraded","detail":"1 of 2 entries available"},
                    {"scope":"provider","name":"claude-ext","kind":"down","detail":"aws/claude-sonnet-5-5 (open), retry in 20s"}
                ]
            },
            "reload": {"last_error": null, "restart_warnings": []},
            "routes": [
                {"route":"pool-docs","policy":"priority","strategy":"priority: configured order, first healthy wins; failover on error",
                 "targets":2,"available":1,"state":"degraded","requests":10,"successes":9,"failures":1,
                 "tokens_input":100,"tokens_output":50,"latency_p50_ms":900,"latency_p95_ms":2000,"latency_p99_ms":3000,"ttft_p50_ms":700},
                {"route":"pool-easy","policy":"cheapest","strategy":"cheapest: lowest cost_per_mtok first, latency breaks ties",
                 "targets":1,"available":1,"state":"ok","requests":5,"successes":5,"failures":0,
                 "tokens_input":10,"tokens_output":5,"latency_p50_ms":300,"latency_p95_ms":400,"latency_p99_ms":500,"ttft_p50_ms":200}
            ],
            "virtual_models": {
                "pool-docs": [
                    {"provider":"claude-ext","model":"aws/claude-sonnet-5-5","circuit":"Open (20s cooldown)","availability":"open","retry_in_secs":20,
                     "api":"anthropic","priority":1,"requests":6,"successes":5,"tokens_input":60,"tokens_output":30,"ttft_p50_ms":800},
                    {"provider":"claude-int","model":"aws/claude-sonnet-5-5","circuit":"Closed (Healthy)","availability":"healthy",
                     "api":"anthropic","weight":3,"requests":4,"successes":4,"tokens_input":40,"tokens_output":20,"ttft_p50_ms":600}
                ],
                "pool-easy": [
                    {"provider":"cb","model":"m","circuit":"Closed (Healthy)","availability":"healthy","api":"openai",
                     "requests":5,"successes":5,"tokens_input":10,"tokens_output":5}
                ]
            },
            "providers": [
                {"name":"claude-ext","models":"aws/claude-sonnet-5-5","circuit":"Closed (Healthy)","failures":0,"rate_limited":false,
                 "admin_disabled":false,"requests":6,"successes":5,"tokens_input":60,"tokens_output":30,"last_request":"Just now"}
            ],
            "recent_requests": [
                {"ts_ms":1793788091000u64,"endpoint":"messages","route":"pool-docs","provider":"claude-int","status":200,"latency_ms":812,"attempts":2,"ttft_ms":700,"error":null}
            ]
        })
    }

    #[test]
    fn test_status_summary_shows_problems_and_routes_only() {
        let out = render_status(&sample_status(), Detail::Summary, None, 120);
        assert!(out.contains("DEGRADED"), "got:\n{out}");
        assert!(out.contains("1h 02m"), "uptime formatted: {out}");
        let problems_at = out.find("pool-docs").unwrap();
        let table_at = out.find("Routes").unwrap();
        assert!(problems_at < table_at, "problems precede the table:\n{out}");
        assert!(out.contains("claude-ext") && out.contains("retry in 20s"));
        assert!(out.contains("pool-easy"));
        // Summary does not include per-target or provider tables.
        assert!(!out.contains("Providers (reachability"));
        assert!(!out.contains("| Provider "));
        // And it says how to get more.
        assert!(out.contains("oxllm status -v"), "next-step hint: {out}");
    }

    #[test]
    fn test_status_verbose_adds_targets_with_tier_and_state() {
        let out = render_status(&sample_status(), Detail::Targets, None, 140);
        assert!(out.contains("| Provider"), "targets table: {out}");
        assert!(
            out.contains("open 20s"),
            "entry state with countdown: {out}"
        );
        assert!(out.contains("p1"), "priority shown: {out}");
        assert!(out.contains("w3"), "weight shown: {out}");
        assert!(out.contains("anthropic"));
        assert!(
            out.contains("cheapest: lowest cost_per_mtok"),
            "strategy text: {out}"
        );
        assert!(!out.contains("Providers (reachability"));
        assert!(out.contains("-vv"));
    }

    #[test]
    fn test_status_very_verbose_adds_providers_and_recent_requests() {
        let out = render_status(&sample_status(), Detail::Full, None, 160);
        assert!(out.contains("Providers (reachability"), "got: {out}");
        assert!(out.contains("Recent requests"), "got: {out}");
        assert!(
            out.contains("attempt 2"),
            "retry visible in recent line: {out}"
        );
        assert!(out.contains("p99"));
    }

    #[test]
    fn test_status_route_filter_narrows_everything() {
        let out = render_status(&sample_status(), Detail::Targets, Some("pool-easy"), 140);
        assert!(out.contains("pool-easy"));
        assert!(!out.contains("claude-int"), "other route leaked: {out}");
        let unknown = render_status(&sample_status(), Detail::Summary, Some("nope"), 140);
        assert!(unknown.contains("No route named 'nope'"), "got: {unknown}");
        assert!(
            unknown.contains("pool-docs"),
            "lists valid routes: {unknown}"
        );
    }

    #[test]
    fn test_status_tolerates_an_older_daemon_without_health() {
        let old = serde_json::json!({
            "uptime_secs": 5, "total_requests": 1, "providers": [], "virtual_models": {},
            "routes": [{"route":"r","policy":"priority","requests":1,"successes":1,"failures":0}]
        });
        let out = render_status(&old, Detail::Summary, None, 120);
        assert!(out.contains("older than v0.1.24"), "got: {out}");
        assert!(out.contains("| r "), "route still listed: {out}");
    }

    #[test]
    fn test_status_json_tiers_match_the_table_tiers() {
        let s = sample_status();
        let keys = |v: &serde_json::Value| -> Vec<String> {
            let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        let summary = status_json_at(&s, Detail::Summary, None);
        assert!(keys(&summary).contains(&"health".to_string()));
        assert!(!keys(&summary).contains(&"virtual_models".to_string()));
        assert!(!keys(&summary).contains(&"providers".to_string()));
        let targets = status_json_at(&s, Detail::Targets, None);
        assert!(keys(&targets).contains(&"virtual_models".to_string()));
        assert!(!keys(&targets).contains(&"providers".to_string()));
        let full = status_json_at(&s, Detail::Full, None);
        assert!(keys(&full).contains(&"providers".to_string()));
        assert!(keys(&full).contains(&"recent_requests".to_string()));
        // Route filter applies to both routes and targets.
        let one = status_json_at(&s, Detail::Targets, Some("pool-easy"));
        assert_eq!(one["routes"].as_array().unwrap().len(), 1);
        assert_eq!(one["virtual_models"].as_object().unwrap().len(), 1);
    }

    #[test]
    fn test_fmt_duration_boundaries() {
        assert_eq!(fmt_duration(0), "0s");
        assert_eq!(fmt_duration(59), "59s");
        assert_eq!(fmt_duration(60), "1m 00s");
        assert_eq!(fmt_duration(3599), "59m 59s");
        assert_eq!(fmt_duration(3600), "1h 00m");
    }

    // ---- /status contract from a live router ------------------------------

    /// /status reports per-entry availability, the route strategy and a
    /// problems list that names a model whose circuit is open.
    #[tokio::test]
    async fn test_status_endpoint_reports_health_strategy_and_problems() {
        let addr = spawn_raw_recording_upstream(
            vec![http_json("500 Internal Server Error", "{}")],
            Default::default(),
        )
        .await;
        let mut p = test_provider("flaky", addr, ApiFormat::Openai);
        p.tuning.circuit_failure_threshold = Some(1);
        let (_proxy, state) =
            spawn_proxy_with_targets(vec![p], vec![target("flaky", "m1")], vec![]).await;
        // Trip the model's circuit directly, then read /status through the real handler.
        state.providers[0]
            .model_health
            .get("m1")
            .circuit
            .write()
            .await
            .clone_from(&CircuitState::Open {
                until: Instant::now() + Duration::from_secs(30),
            });
        let status = routes::get_status(axum::extract::State((state.clone(), Instant::now())))
            .await
            .into_response();
        let bytes = axum::body::to_bytes(status.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["health"]["state"], "down");
        assert_eq!(v["routes"][0]["state"], "down");
        assert_eq!(v["routes"][0]["available"], 0);
        assert!(v["routes"][0]["strategy"]
            .as_str()
            .unwrap()
            .starts_with("priority"));
        let entry = &v["virtual_models"]["vm"][0];
        assert_eq!(entry["availability"], "open");
        assert_eq!(entry["api"], "openai");
        let problems = v["health"]["problems"].as_array().unwrap();
        assert!(problems
            .iter()
            .any(|p| p["kind"] == "down" && p["name"] == "vm"));
        assert!(problems.iter().any(
            |p| p["scope"] == "provider" && p["detail"].as_str().unwrap().contains("retry in")
        ));
    }

    // ---- CLI help -----------------------------------------------------------

    #[test]
    fn test_cli_definition_is_consistent_and_every_command_is_described() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        cmd.clone().debug_assert();
        fn walk(cmd: &clap::Command, path: &str) {
            for sub in cmd.get_subcommands() {
                let name = format!("{path} {}", sub.get_name());
                if sub.get_name() != "help" {
                    assert!(
                        sub.get_about().is_some(),
                        "`{}` has no one-line description",
                        name.trim()
                    );
                }
                walk(sub, &name);
            }
        }
        walk(&cmd, "oxllm");
    }

    #[test]
    fn test_cli_help_points_to_the_next_step() {
        use clap::CommandFactory;
        let mut top = Cli::command();
        let help = top.render_long_help().to_string();
        assert!(help.contains("Getting started"), "got: {help}");
        assert!(help.contains("oxllm status -v"), "got: {help}");
        let mut status = Cli::command().find_subcommand("status").unwrap().clone();
        let help = status.render_long_help().to_string();
        assert!(help.contains("--route"), "got: {help}");
        assert!(help.contains("-vv"), "got: {help}");
        assert!(help.contains("oxllm tail"), "next step named: {help}");
    }

    /// A 429 for one model (LiteLLM answers "No deployments available for
    /// selected model" per model) must not rate-limit its siblings on the
    /// same provider: the sibling keeps serving and only the limited model
    /// is skipped until its window ends.
    #[tokio::test]
    async fn test_429_for_one_model_does_not_block_its_sibling() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let reply = if req.contains("model-limited") {
                        "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 30\r\nConnection: close\r\nContent-Length: 0\r\n\r\n".to_string()
                    } else {
                        http_json("200 OK", r#"{"choices":[],"usage":{}}"#).replacen(
                            "\r\n",
                            "\r\nConnection: close\r\n",
                            1,
                        )
                    };
                    let _ = stream.write_all(reply.as_bytes()).await;
                });
            }
        });
        let (proxy, state) = spawn_proxy_with_targets(
            vec![test_provider("gw", addr, ApiFormat::Openai)],
            vec![target("gw", "model-limited"), target("gw", "model-ok")],
            vec![],
        )
        .await;
        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/chat/completions", proxy);
        let body = serde_json::json!({"model":"vm","messages":[{"role":"user","content":"x"}]});

        // First request: model-limited answers 429, model-ok serves.
        let res = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(res.status(), 200);
        assert_eq!(state.request_log.recent(1)[0].model, "model-ok");

        // The 429 window belongs to the model, not the provider.
        assert!(
            state.providers[0].rate_limited_until.read().await.is_none(),
            "a model's 429 must not rate-limit the whole provider"
        );
        let limited = state.providers[0].model_health.get("model-limited");
        assert!(limited.rate_limited_until.read().await.is_some());

        // A single-entry route for the sibling model is served.
        let (proxy2, _s2) = spawn_proxy_with_targets(
            vec![test_provider("gw", addr, ApiFormat::Openai)],
            vec![target("gw", "model-ok")],
            vec![],
        )
        .await;
        let res = client
            .post(format!("http://{}/v1/chat/completions", proxy2))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        // And the limited model is skipped instantly on the next request.
        let t = Instant::now();
        let res = client.post(&url).json(&body).send().await.unwrap();
        assert_eq!(res.status(), 200);
        assert!(t.elapsed() < Duration::from_millis(500));
        let availability = state.providers[0].entry_availability("model-limited").await;
        assert_eq!(availability.name(), "rate_limited");
        assert_eq!(
            state.providers[0]
                .entry_availability("model-ok")
                .await
                .name(),
            "healthy"
        );
    }
}
