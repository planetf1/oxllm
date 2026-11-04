use axum::{
    body::Body,
    extract::{Extension, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures_util::{stream::unfold, StreamExt};
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, warn};

use oxllm_core::config::ApiFormat;
use oxllm_core::router::{
    latency_band, order_by_cost, order_by_latency, rotate_within_band, weighted_lottery,
    AdaptivePriorityStrategy, LatencyCandidate, RoutingStrategy,
};
use oxllm_core::state::{
    unix_millis, AppState, CircuitState, ModelStats, ProviderState, RequestRecord, RouteCandidate,
    RoutingPolicy,
};
use oxllm_core::telemetry::{TelemetryClient, TelemetryEvent};

/// Every Nth request ignores measured latency for one routing decision, so a
/// provider demoted by a transient blip is still re-sampled.
const LATENCY_EXPLORE_EVERY: u64 = 20;

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

/// GET /v1/models
pub async fn list_models(State(app_state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut data = Vec::new();
    let now = Instant::now();

    for (vm_name, targets) in &app_state.virtual_models {
        let mut is_healthy = false;

        for target in targets {
            if let Some(provider) = app_state
                .providers
                .iter()
                .find(|p| p.name == target.provider)
            {
                // Check if provider is healthy (Closed, or expired cooldown ready for HalfOpen)
                let circuit = *provider.circuit.read().await;
                let is_tripped = match circuit {
                    CircuitState::Closed | CircuitState::HalfOpen => false,
                    CircuitState::Open { until } => now < until,
                };

                let rl = *provider.rate_limited_until.read().await;
                let is_rate_limited = match rl {
                    Some(until) => now < until,
                    None => false,
                };

                if !is_tripped && !is_rate_limited {
                    is_healthy = true;
                    break;
                }
            }
        }

        if is_healthy {
            data.push(ModelObject {
                id: vm_name.clone(),
                object: "model",
                created: 1717070400, // May 30, 2026 constant
                owned_by: "oxllm-virtual",
            });
        }
    }

    // Alias names ride along with their (healthy) target virtual models, so
    // unmodified OpenAI clients can discover and use the names they were
    // built with.
    let mut aliases: Vec<(&String, &String)> = app_state.aliases.iter().collect();
    aliases.sort();
    for (alias, vm) in aliases {
        if data.iter().any(|m| m.id == *vm) {
            data.push(ModelObject {
                id: alias.clone(),
                object: "model",
                created: 1717070400, // May 30, 2026 constant
                owned_by: "oxllm-alias",
            });
        }
    }

    Json(ModelsResponse {
        object: "list",
        data,
    })
}

#[derive(Serialize)]
struct RouteEntry {
    provider: String,
    model: String,
    circuit: String,
    requests: u64,
    successes: u64,
    tokens_input: u64,
    tokens_output: u64,
    ttft_p50_ms: Option<u64>,
    latency_p50_ms: Option<u64>,
    cached_tokens: u64,
}

/// GET /status (localhost restricted in main.rs routing)
pub async fn get_status(
    State((app_state, start_time)): State<(Arc<AppState>, Instant)>,
) -> impl IntoResponse {
    #[derive(Serialize)]
    struct ProviderStatus {
        name: String,
        models: String,
        circuit: String,
        failures: u32,
        rate_limited: bool,
        admin_disabled: bool,
        requests: u64,
        successes: u64,
        tokens_input: u64,
        tokens_output: u64,
        last_request: String,
    }

    /// Per-virtual-model totals, independent of how many providers back it.
    #[derive(Serialize)]
    struct RouteSummary {
        route: String,
        policy: String,
        requests: u64,
        successes: u64,
        failures: u64,
        tokens_input: u64,
        tokens_output: u64,
        latency_p50_ms: Option<u64>,
        latency_p95_ms: Option<u64>,
        latency_p99_ms: Option<u64>,
        latency_mean_ms: Option<f64>,
        ttft_p50_ms: Option<u64>,
        ttft_p95_ms: Option<u64>,
        cached_tokens: u64,
    }

    #[derive(Serialize)]
    struct StatusResponse {
        version: &'static str,
        uptime_secs: u64,
        total_requests: u64,
        reload: oxllm_core::state::ReloadHealthSnapshot,
        providers: Vec<ProviderStatus>,
        virtual_models: std::collections::HashMap<String, Vec<RouteEntry>>,
        routes: Vec<RouteSummary>,
        recent_requests: Vec<oxllm_core::state::RequestRecord>,
    }

    let mut status_list = Vec::new();
    let mut total_requests: u64 = 0;
    let now = Instant::now();

    for provider in &app_state.providers {
        let circ = *provider.circuit.read().await;
        let circuit_str = match circ {
            CircuitState::Closed => "Closed (Healthy)".to_string(),
            CircuitState::HalfOpen => "Half-Open (Probing)".to_string(),
            CircuitState::Open { until } => {
                let left = until.saturating_duration_since(now).as_secs();
                format!("Open (Cooldown: {}s left)", left)
            },
        };

        let rl = *provider.rate_limited_until.read().await;
        let is_limited = match rl {
            Some(until) => now < until,
            None => false,
        };

        let failures = *provider.consecutive_failures.read().await;

        let requests = provider.requests.load(std::sync::atomic::Ordering::Relaxed);
        let successes = provider
            .successes
            .load(std::sync::atomic::Ordering::Relaxed);
        let tokens_input = provider
            .tokens_input
            .load(std::sync::atomic::Ordering::Relaxed);
        let tokens_output = provider
            .tokens_output
            .load(std::sync::atomic::Ordering::Relaxed);

        let last_request = {
            let last = provider.last_attempt_time.read().await;
            match *last {
                Some(instant) => {
                    let elapsed = now.saturating_duration_since(instant);
                    if elapsed.as_secs() < 60 {
                        "Just now".to_string()
                    } else if elapsed.as_secs() < 3600 {
                        format!("{}m ago", elapsed.as_secs() / 60)
                    } else if elapsed.as_secs() < 86400 {
                        format!("{}h ago", elapsed.as_secs() / 3600)
                    } else {
                        format!("{}d ago", elapsed.as_secs() / 86400)
                    }
                },
                None => "Never".to_string(),
            }
        };

        total_requests += requests;

        status_list.push(ProviderStatus {
            name: provider.name.clone(),
            models: provider.models.join(", "),
            circuit: circuit_str,
            failures,
            rate_limited: is_limited,
            admin_disabled: provider
                .manual_disabled
                .load(std::sync::atomic::Ordering::Relaxed),
            requests,
            successes,
            tokens_input,
            tokens_output,
            last_request,
        });
    }

    // Build virtual model routing table
    let mut virtual_models: std::collections::HashMap<String, Vec<RouteEntry>> =
        std::collections::HashMap::new();

    for (vm_name, targets) in &app_state.virtual_models {
        let mut entries = Vec::new();
        for target in targets {
            let provider_state = app_state
                .providers
                .iter()
                .find(|p| p.name == target.provider);
            let circuit_str = match provider_state {
                Some(provider) => {
                    let circ = *provider.circuit.read().await;
                    let now = std::time::Instant::now();
                    match circ {
                        CircuitState::Closed => "Closed (Healthy)".to_string(),
                        CircuitState::HalfOpen => "Half-Open (Probing)".to_string(),
                        CircuitState::Open { until } => {
                            let left = until.saturating_duration_since(now).as_secs();
                            format!("Open ({}s cooldown)", left)
                        },
                    }
                },
                None => "Unknown".to_string(),
            };
            // Traffic attributed to this (provider, model) pair, not to the
            // provider as a whole: one provider may serve several routes.
            let stats = provider_state
                .map(|provider| {
                    provider
                        .model_stats
                        .snapshot(&oxllm_core::state::ModelStats::key(
                            &target.provider,
                            &target.model,
                        ))
                })
                .unwrap_or_default();
            entries.push(RouteEntry {
                provider: target.provider.clone(),
                model: target.model.clone(),
                circuit: circuit_str,
                requests: stats.requests,
                successes: stats.successes,
                tokens_input: stats.tokens_input,
                tokens_output: stats.tokens_output,
                ttft_p50_ms: stats.ttft_p50_ms,
                latency_p50_ms: stats.latency_p50_ms,
                cached_tokens: stats.cached_tokens,
            });
        }
        virtual_models.insert(vm_name.clone(), entries);
    }

    let mut routes: Vec<RouteSummary> = app_state
        .route_stats_keys()
        .into_iter()
        .map(|route_name| {
            let snap = app_state.route_stats.snapshot(&route_name);
            RouteSummary {
                policy: match app_state.policies.get(&route_name) {
                    Some(RoutingPolicy::Fastest) => "fastest".to_string(),
                    Some(RoutingPolicy::Cheapest) => "cheapest".to_string(),
                    _ => "priority".to_string(),
                },
                requests: snap.requests,
                successes: snap.successes,
                failures: snap.failures,
                tokens_input: snap.tokens_input,
                tokens_output: snap.tokens_output,
                latency_p50_ms: snap.latency_p50_ms,
                latency_p95_ms: snap.latency_p95_ms,
                latency_p99_ms: snap.latency_p99_ms,
                latency_mean_ms: snap.latency_mean_ms,
                ttft_p50_ms: snap.ttft_p50_ms,
                ttft_p95_ms: snap.ttft_p95_ms,
                cached_tokens: snap.cached_tokens,
                route: route_name,
            }
        })
        .collect();
    routes.sort_by(|a, b| a.route.cmp(&b.route));

    Json(StatusResponse {
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: start_time.elapsed().as_secs(),
        total_requests,
        reload: app_state.reload_health.snapshot(),
        providers: status_list,
        virtual_models,
        routes,
        recent_requests: app_state.request_log.recent(50),
    })
}

/// GET /admin/tail (localhost-only): live SSE stream of request records as
/// they complete. Returns a clear error when recording is disabled.
pub async fn tail_requests(
    State(app_state): State<Arc<AppState>>,
) -> axum::response::sse::Sse<
    impl futures_util::Stream<Item = Result<axum::response::sse::Event, axum::Error>>,
> {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use futures_util::StreamExt;
    use tokio_stream::wrappers::BroadcastStream;

    let rx = app_state
        .request_log
        .subscribe()
        .expect("tail_requests route checks enablement");
    let stream = BroadcastStream::new(rx).filter_map(|item| async move {
        // Lagged receivers just skip missed records; the ring is the source
        // of truth for history.
        item.ok()
            .map(|rec| Ok(Event::default().json_data(rec).unwrap()))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Applies the configured routing policy to a resolved candidate list.
/// `Priority` keeps configuration order untouched; `Fastest` sorts by observed
/// EWMA TTFT in 20% bands (with periodic exploration), so the order only
/// changes when one provider is meaningfully quicker.
/// Extracts the routing session key for a request: the configured header
/// first (default `x-session-id`), then the body's `prompt_cache_key` field,
/// then (Anthropic Messages bodies) `metadata.user_id`. `None` means the request is keyless and follows policy ordering only.
fn session_key(app_state: &AppState, headers: &HeaderMap, payload: &Value) -> Option<String> {
    let from_header = (!app_state.session_affinity_header.is_empty())
        .then(|| {
            headers
                .get(&app_state.session_affinity_header)
                .and_then(|v| v.to_str().ok())
                .filter(|s| !s.is_empty())
                .map(String::from)
        })
        .flatten();
    from_header
        .or_else(|| {
            payload
                .get("prompt_cache_key")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from)
        })
        .or_else(|| {
            payload
                .get("metadata")
                .and_then(|m| m.get("user_id"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from)
        })
}

/// Applies sticky affinity (when a session key is present and its pin is
/// healthy) and then the configured routing policy. Affinity wins over
/// policy ordering; keyless requests rotate within policy as before.
fn ordered_candidates<'a>(
    app_state: &AppState,
    route_name: &str,
    session_key: Option<&str>,
    candidates: &[RouteCandidate<'a>],
) -> Vec<RouteCandidate<'a>> {
    let mut ordered: Vec<RouteCandidate<'a>> = candidates.to_vec();
    if ordered.len() < 2 {
        return ordered;
    }

    // Sticky routing: keep a session on its provider while it is healthy so
    // the upstream prefix cache stays warm. Policy ordering still governs
    // keyless traffic and sessions whose pin is unhealthy or expired.
    if let Some(key) = session_key {
        if let Some(pinned) = app_state.session_affinity.pinned_provider(key) {
            if let Some(pos) = ordered.iter().position(|c| {
                c.provider.name == pinned
                    && !c
                        .provider
                        .manual_disabled
                        .load(std::sync::atomic::Ordering::Relaxed)
            }) {
                let chosen = ordered.remove(pos);
                ordered.insert(0, chosen);
                return ordered;
            }
        }
    }

    let policy = app_state.policies.get(route_name).copied();
    if matches!(policy, None | Some(RoutingPolicy::Priority)) {
        // Strict order (priority-sorted, session-pinned), plus optional
        // weighted sharing within priority tiers.
        apply_weighted_sharing(&mut ordered);
        return ordered;
    }

    let call_index = app_state
        .routing_calls
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let described: Vec<LatencyCandidate> = ordered
        .iter()
        .map(|c| LatencyCandidate {
            provider: c.provider.name.clone(),
            model: c.model.clone(),
            ewma_ttft_ms: c
                .provider
                .model_stats
                .ewma_ttft_ms(&c.provider.name, &c.model),
            cost_per_mtok: c.provider.cost_per_mtok,
        })
        .collect();

    let order = match policy {
        Some(RoutingPolicy::Cheapest) => order_by_cost(&described, call_index),
        _ => {
            let bands: Vec<u32> = described
                .iter()
                .map(|c| latency_band(c.ewma_ttft_ms))
                .collect();
            let base = order_by_latency(&described, call_index, LATENCY_EXPLORE_EVERY);
            // Spread load across equal-speed providers, then let the periodic
            // exploration rotation still surface the demoted ones.
            rotate_within_band(base, &bands, call_index / LATENCY_EXPLORE_EVERY)
        },
    };
    order.into_iter().map(|i| ordered[i].clone()).collect()
}

/// Proportional load sharing under the default (priority) policy: within
/// each priority tier, candidates carrying a `weight` are drawn per request
/// as a lottery with probability proportional to it, so healthy targets
/// share traffic instead of the first one taking everything. Unweighted
/// candidates keep their configured order after them, and tiers with no
/// weights at all are untouched — no weights anywhere means today's strict
/// order. The `fastest` and `cheapest` policies never call this.
fn apply_weighted_sharing(ordered: &mut [RouteCandidate<'_>]) {
    if !ordered.iter().any(|c| c.weight.is_some()) {
        return;
    }
    let mut rng = rand::rng();
    let mut result: Vec<usize> = Vec::with_capacity(ordered.len());
    // Candidates arrive priority-sorted from resolve_candidates; tiers are
    // maximal runs of equal effective priority.
    let mut start = 0;
    while start < ordered.len() {
        let tier_priority = ordered[start].priority;
        let mut end = start + 1;
        while end < ordered.len() && ordered[end].priority == tier_priority {
            end += 1;
        }
        if ordered[start..end].iter().any(|c| c.weight.is_some()) {
            let weights: Vec<u64> = ordered[start..end]
                .iter()
                .map(|c| c.weight.unwrap_or(0) as u64)
                .collect();
            result.extend(
                weighted_lottery(&weights, &mut rng)
                    .into_iter()
                    .map(|i| start + i),
            );
        } else {
            result.extend(start..end);
        }
        start = end;
    }
    // Apply the permutation in place.
    let source: Vec<RouteCandidate<'_>> = ordered.to_vec();
    for (dest, src) in result.into_iter().enumerate() {
        ordered[dest] = source[src].clone();
    }
}

/// Per-endpoint configuration for the shared proxy loop.
struct EndpointSpec {
    /// Path appended to the provider base URL ("embeddings",
    /// "chat/completions", ...).
    path: &'static str,
    /// Telemetry operation name and log label.
    operation: &'static str,
    /// Noun used in the terminal 502 message ("chat completions",
    /// "embeddings", ...).
    failure_noun: &'static str,
    /// Whether `stream: true` requests forward as SSE.
    streaming: bool,
    /// Wire format of this endpoint. Only providers with a matching `api`
    /// are candidates; it also selects upstream auth headers, the error
    /// body shape and usage parsing.
    api: ApiFormat,
}

/// The shared proxy loop behind every public endpoint (issue #11): JSON
/// parse, model resolution (aliases, default model), trace-context
/// extraction, policy-ordered candidates, then the failover loop — model
/// rewrite per target, per-provider circuit breaking and feedback, usage
/// accounting, session affinity — ending in a terminal 502. Streaming
/// endpoints delegate to `stream_upstream_body`; everything else passes
/// the response through verbatim.
#[allow(clippy::too_many_arguments)]
async fn proxy_request(
    spec: &EndpointSpec,
    app_state: Arc<AppState>,
    telemetry: TelemetryClient,
    request_id: String,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut payload: Value = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            return error_response(
                spec.api,
                &format!("Invalid JSON payload: {}", e),
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            )
        },
    };

    let requested_model = match payload.get("model").and_then(|m| m.as_str()) {
        Some(m) => m,
        None => {
            return error_response(
                spec.api,
                "Missing required 'model' field",
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            )
        },
    };

    // Owned copy: `payload` is mutated per attempt, so the &str borrow cannot
    // live across the routing loop. Names resolve through aliases and the
    // configured default model before being rejected.
    let route_name = match app_state.resolve_route(requested_model) {
        Some(route) => route.to_string(),
        None => {
            return error_response(
                spec.api,
                &format!("Invalid or unmapped virtual model: {}", requested_model),
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            )
        },
    };
    // Providers speaking another wire format are skipped, not errored on.
    let candidates = app_state.resolve_candidates_for(&route_name, Some(spec.api));
    if candidates.is_empty() {
        let message = if app_state.resolve_candidates(&route_name).is_empty() {
            format!("Invalid or unmapped virtual model: {}", requested_model)
        } else {
            format!(
                "Virtual model '{}' has no provider configured with api = \"{}\" for this endpoint",
                route_name,
                spec.api.as_str()
            )
        };
        return error_response(
            spec.api,
            &message,
            "invalid_request_error",
            StatusCode::BAD_REQUEST,
        );
    }

    // Extract W3C traceparent headers if present for tracing
    let (trace_id, parent_span_id) = extract_traceparent(&headers);

    let is_streaming = payload
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let strategy = AdaptivePriorityStrategy {
        circuit: app_state.circuit,
    };
    let mut attempts = 0;
    let start_time = Instant::now();
    let mut last_upstream_error = String::new();
    let mut last_failed_provider = String::new();
    let mut last_failed_model = String::new();
    let mut last_failed_status: u16 = 0;

    // Map candidate provider references for select strategy
    let candidates = ordered_candidates(
        &app_state,
        &route_name,
        session_key(&app_state, &headers, &payload).as_deref(),
        &candidates,
    );
    let candidate_states: Vec<&ProviderState> = candidates.iter().map(|c| c.provider).collect();

    // Route-level counters: one row per virtual model, one request each,
    // regardless of how many provider attempts it takes.
    let route_counters = app_state.route_stats.counters(&route_name);
    route_counters
        .requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    while let Some(selected) = strategy.select(&candidate_states).await {
        attempts += 1;
        let target_model = match candidates
            .iter()
            .find(|c| c.provider.name == selected.name)
            .map(|c| c.model.clone())
        {
            Some(m) => m,
            None => {
                warn!(
                    request_id = %request_id,
                    "Provider {} removed from candidates during routing",
                    selected.name
                );
                continue;
            },
        };

        // 1. Rewrite model field in JSON body
        payload["model"] = Value::String(target_model.clone());
        let rewritten_body = Bytes::from(serde_json::to_vec(&payload).unwrap());

        // 2. Safely parse base URL and join embeddings path (Url::joinTrailingSlashes mitigation)
        let endpoint_url = match selected.base_url.join(spec.path) {
            Ok(url) => url,
            Err(e) => {
                error!("Invalid base URL path join for {}: {}", selected.name, e);
                continue;
            },
        };

        // 3. Construct upstream request
        let mut req = app_state
            .http_client
            .post(endpoint_url.as_str())
            .body(rewritten_body)
            .timeout(Duration::from_secs(
                selected
                    .upstream_timeout_secs
                    .unwrap_or(app_state.upstream_timeout_secs),
            ))
            .header("Content-Type", "application/json");
        req = match spec.api {
            ApiFormat::Openai => {
                req.header("Authorization", format!("Bearer {}", selected.api_key))
            },
            ApiFormat::Anthropic => {
                let version = headers
                    .get("anthropic-version")
                    .cloned()
                    .unwrap_or_else(|| HeaderValue::from_static(DEFAULT_ANTHROPIC_VERSION));
                let mut r = req
                    .header("x-api-key", selected.api_key.as_str())
                    .header("anthropic-version", version);
                if let Some(beta) = headers.get("anthropic-beta") {
                    r = r.header("anthropic-beta", beta);
                }
                r
            },
        };

        // Propagate tracing headers if present
        if let Some(traceparent) = headers.get("traceparent") {
            req = req.header("traceparent", traceparent);
        }

        debug!(
            "{} request routing to {} (attempt {})",
            spec.operation, selected.name, attempts
        );

        // Increment request counter
        if let Some(target) = app_state.providers.iter().find(|p| p.name == selected.name) {
            target
                .requests
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            target
                .model_stats
                .counters(&ModelStats::key(&selected.name, &target_model))
                .requests
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let res = req.send().await;

        match res {
            Ok(res) if res.status().is_success() => {
                let status_code = res.status().as_u16();
                let upstream_headers = res.headers().clone();

                // Streaming endpoints forward the upstream SSE body as it
                // arrives, never buffered.
                if spec.streaming && is_streaming {
                    return stream_upstream_body(
                        res,
                        upstream_headers,
                        status_code,
                        app_state.clone(),
                        selected.name.clone(),
                        selected.is_probe,
                        spec.operation,
                        route_name.clone(),
                        target_model,
                        telemetry.clone(),
                        trace_id.clone(),
                        parent_span_id.clone(),
                        request_id.clone(),
                        start_time,
                        attempts,
                        spec.api,
                    )
                    .into_response();
                }

                let res_body = match res.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        warn!(
                            request_id = %request_id,
                            "Failed to read success response body from {}: {}",
                            selected.name, e
                        );
                        let target_provider_state = app_state
                            .providers
                            .iter()
                            .find(|p| p.name == selected.name)
                            .unwrap();
                        strategy
                            .feedback(
                                target_provider_state,
                                false,
                                selected.is_probe,
                                Some(status_code),
                                None,
                            )
                            .await;
                        telemetry.emit(TelemetryEvent::UpdateStatus {
                            provider: selected.name.clone(),
                            status: circuit_status(target_provider_state).await,
                        });
                        continue;
                    },
                };

                // Parse token counts from the upstream response; field names
                // differ by wire format and endpoint family.
                let usage = serde_json::from_slice::<Value>(&res_body)
                    .map(|v| parse_usage(&v, spec.api))
                    .unwrap_or_default();
                let (input_tokens, output_tokens) = (usage.input, usage.output);

                // Report Success feedback
                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback(
                        target_provider_state,
                        true,
                        selected.is_probe,
                        Some(status_code),
                        None,
                    )
                    .await;
                telemetry.emit(TelemetryEvent::UpdateStatus {
                    provider: selected.name.clone(),
                    status: circuit_status(target_provider_state).await,
                });

                // Increment local counters
                target_provider_state
                    .successes
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                target_provider_state
                    .tokens_input
                    .fetch_add(input_tokens, std::sync::atomic::Ordering::Relaxed);
                target_provider_state
                    .tokens_output
                    .fetch_add(output_tokens, std::sync::atomic::Ordering::Relaxed);
                {
                    let elapsed_ms = start_time.elapsed().as_millis() as u64;
                    let model_counters = target_provider_state
                        .model_stats
                        .counters(&ModelStats::key(&selected.name, &target_model));
                    if let Some(key) = session_key(&app_state, &headers, &payload).as_deref() {
                        app_state.session_affinity.pin(key, &selected.name);
                    }
                    for counters in [&model_counters, &route_counters] {
                        counters
                            .successes
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        counters
                            .tokens_input
                            .fetch_add(input_tokens, std::sync::atomic::Ordering::Relaxed);
                        counters
                            .tokens_output
                            .fetch_add(output_tokens, std::sync::atomic::Ordering::Relaxed);
                        counters.latency.record(elapsed_ms);
                        // Non-streaming: first byte is indistinguishable from the
                        // whole response, so total latency is the TTFT sample.
                        counters.record_ttft(elapsed_ms * 1000);
                        counters
                            .cached_tokens
                            .fetch_add(usage.cached, std::sync::atomic::Ordering::Relaxed);
                    }
                }

                // Push Telemetry Metrics
                telemetry.emit(TelemetryEvent::RecordTransaction {
                    operation: spec.operation.to_string(),
                    provider: selected.name.clone(),
                    model: target_model.clone(),
                    input_tokens,
                    output_tokens,
                    duration: start_time.elapsed(),
                    attempts,
                    failure_reason: None,
                    trace_id: trace_id.clone(),
                    parent_span_id: parent_span_id.clone(),
                    request_id: request_id.clone(),
                });

                // Construct clean Axum response copying headers cleanly via bytes
                app_state.request_log.record(RequestRecord {
                    ts_ms: unix_millis(),
                    endpoint: spec.operation.to_string(),
                    route: route_name.clone(),
                    model: target_model.clone(),
                    provider: selected.name.clone(),
                    status: status_code,
                    attempts,
                    latency_ms: start_time.elapsed().as_millis() as u64,
                    // Non-streaming: the whole response is the first byte.
                    ttft_ms: Some(start_time.elapsed().as_millis() as u64),
                    streamed: false,
                    error: None,
                    request_id: request_id.clone(),
                });

                let mut response = Response::new(Body::from(res_body));
                *response.status_mut() = StatusCode::OK;
                copy_response_headers(&upstream_headers, response.headers_mut());
                return response.into_response();
            },
            Ok(res) => {
                let status_code = res.status().as_u16();
                warn!(
                    request_id = %request_id,
                    "{} request upstream {} failed with status {}",
                    spec.operation,
                    selected.name, status_code
                );

                // Extract Retry-After before consuming the response body
                let retry_after = extract_retry_after(res.headers());

                // Anthropic client errors (auth, invalid_request, ...) are
                // the caller's to see: another provider would not fix them,
                // and they say nothing about this provider's health. Pass
                // them through verbatim without failover or breaker penalty.
                if spec.api == ApiFormat::Anthropic && is_passthrough_client_error(status_code) {
                    let upstream_headers = res.headers().clone();
                    let error_body = res.bytes().await.unwrap_or_default();
                    if selected.is_probe {
                        // Release the lock-free probe permit; the circuit
                        // stays HalfOpen so the next request probes again.
                        if let Some(p) =
                            app_state.providers.iter().find(|p| p.name == selected.name)
                        {
                            p.probe_in_flight
                                .store(false, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                    app_state.request_log.record(RequestRecord {
                        ts_ms: unix_millis(),
                        endpoint: spec.operation.to_string(),
                        route: route_name.clone(),
                        model: target_model.clone(),
                        provider: selected.name.clone(),
                        status: status_code,
                        attempts,
                        latency_ms: start_time.elapsed().as_millis() as u64,
                        ttft_ms: None,
                        streamed: false,
                        error: Some(String::from_utf8_lossy(&error_body).to_string()),
                        request_id: request_id.clone(),
                    });
                    let mut response = Response::new(Body::from(error_body));
                    *response.status_mut() =
                        StatusCode::from_u16(status_code).unwrap_or(StatusCode::BAD_REQUEST);
                    copy_response_headers(&upstream_headers, response.headers_mut());
                    return response;
                }

                // Best-effort: read upstream error body for final 502 response
                let error_body = match res.bytes().await {
                    Ok(b) => String::from_utf8_lossy(&b).to_string(),
                    Err(e) => {
                        warn!(
                            request_id = %request_id,
                            "Failed to read error body from {}: {}", selected.name, e
                        );
                        String::new()
                    },
                };
                let parsed_error = if !error_body.is_empty() {
                    serde_json::from_str::<Value>(&error_body)
                        .ok()
                        .and_then(|v| {
                            v.get("error")
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str().map(String::from))
                        })
                        .unwrap_or(error_body)
                } else {
                    String::new()
                };
                last_upstream_error = parsed_error;
                last_failed_provider = selected.name.clone();
                last_failed_model = target_model.clone();
                last_failed_status = status_code;

                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback(
                        target_provider_state,
                        false,
                        selected.is_probe,
                        Some(status_code),
                        retry_after,
                    )
                    .await;
                telemetry.emit(TelemetryEvent::UpdateStatus {
                    provider: selected.name.clone(),
                    status: circuit_status(target_provider_state).await,
                });
            },
            Err(e) => {
                warn!(
                    request_id = %request_id,
                    "{} request upstream {} connection failed: {}",
                    spec.operation,
                    selected.name, e
                );
                last_upstream_error = e.to_string();
                last_failed_provider = selected.name.clone();
                last_failed_model = target_model.clone();
                last_failed_status = 0;
                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback(target_provider_state, false, selected.is_probe, None, None)
                    .await;
                telemetry.emit(TelemetryEvent::UpdateStatus {
                    provider: selected.name.clone(),
                    status: circuit_status(target_provider_state).await,
                });
            },
        }
    }

    let message = if !last_upstream_error.is_empty() {
        format!(
            "All upstream {} providers failed or are rate-limited. Last error from {} ({}): {}",
            spec.failure_noun, last_failed_provider, last_failed_status, last_upstream_error
        )
    } else {
        format!(
            "All upstream {} providers failed or are rate-limited",
            spec.failure_noun
        )
    };
    route_counters
        .failures
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Some(key) = session_key(&app_state, &headers, &payload).as_deref() {
        app_state.session_affinity.unpin(key);
    }
    app_state.request_log.record(RequestRecord {
        ts_ms: unix_millis(),
        endpoint: spec.operation.to_string(),
        route: route_name.clone(),
        model: last_failed_model,
        provider: last_failed_provider,
        status: StatusCode::BAD_GATEWAY.as_u16(),
        attempts,
        latency_ms: start_time.elapsed().as_millis() as u64,
        ttft_ms: None,
        streamed: false,
        error: Some(last_upstream_error),
        request_id: request_id.clone(),
    });
    error_response(spec.api, &message, "server_error", StatusCode::BAD_GATEWAY)
}

const EMBEDDINGS_SPEC: EndpointSpec = EndpointSpec {
    path: "embeddings",
    operation: "embeddings",
    failure_noun: "embeddings",
    streaming: false,
    api: ApiFormat::Openai,
};

/// POST /v1/embeddings
///
/// OpenAI-compatible embeddings, routed through the provider pool with
/// failover and usage accounting.
pub async fn create_embeddings(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &EMBEDDINGS_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

const SYSTEMONE_SPEC: EndpointSpec = EndpointSpec {
    path: "systemone",
    operation: "systemone",
    failure_noun: "systemone",
    streaming: false,
    api: ApiFormat::Openai,
};

/// POST /v1/systemone
///
/// llama.cpp decision-model endpoints (typed single-choice / ordinal /
/// yes-no questions answered in one forward pass, ggml-org/llama.cpp
/// PR #29818). Routed through the provider pool like chat completions:
/// model rewrite per target, per-provider circuit breaking, failover.
/// No streaming; the response passes through verbatim.
pub async fn create_systemone(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &SYSTEMONE_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

const COMPLETIONS_SPEC: EndpointSpec = EndpointSpec {
    path: "completions",
    operation: "completions",
    failure_noun: "completions",
    streaming: true,
    api: ApiFormat::Openai,
};

/// POST /v1/completions
///
/// Raw (non-chat) completions, used by fill-in-the-middle clients such as
/// editor autocomplete. Adds no buffering: streaming requests are forwarded
/// as SSE, non-streaming responses pass through verbatim.
pub async fn create_completions(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &COMPLETIONS_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

const CHAT_SPEC: EndpointSpec = EndpointSpec {
    path: "chat/completions",
    operation: "chat",
    failure_noun: "chat completions",
    streaming: true,
    api: ApiFormat::Openai,
};

/// POST /v1/chat/completions
///
/// OpenAI-compatible chat completions with SSE streaming passthrough,
/// routed through the provider pool with failover and usage accounting.
pub async fn create_chat_completions(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(&CHAT_SPEC, app_state, telemetry, request_id, headers, body).await
}
const MESSAGES_SPEC: EndpointSpec = EndpointSpec {
    path: "messages",
    operation: "messages",
    failure_noun: "messages",
    streaming: true,
    api: ApiFormat::Anthropic,
};

/// POST /v1/messages
///
/// Anthropic Messages API passthrough (no body translation) with SSE
/// streaming, routed to `api = "anthropic"` providers only.
pub async fn create_messages(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &MESSAGES_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

const COUNT_TOKENS_SPEC: EndpointSpec = EndpointSpec {
    path: "messages/count_tokens",
    operation: "count_tokens",
    failure_noun: "count_tokens",
    streaming: false,
    api: ApiFormat::Anthropic,
};

/// POST /v1/messages/count_tokens
///
/// Anthropic token counting; same routing and failover as `/v1/messages`,
/// response passed through verbatim.
pub async fn count_tokens(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &COUNT_TOKENS_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

/// for telemetry: 0 = Healthy, 1 = Cooldown, 2 = Tripped.
/// Forwards an upstream SSE body to the client chunk-by-chunk (never buffered),
/// and defers circuit-breaker feedback until the stream ends so that a failure
/// part-way through is still attributed to the provider that served it.
#[allow(clippy::too_many_arguments)]
fn stream_upstream_body(
    res: reqwest::Response,
    upstream_headers: HeaderMap,
    status_code: u16,
    app_state: Arc<AppState>,
    provider_name: String,
    is_probe: bool,
    operation: &'static str,
    route_name: String,
    model_name: String,
    telemetry: TelemetryClient,
    trace_id: Option<String>,
    parent_span_id: Option<String>,
    request_id: String,
    start_time: Instant,
    attempts: u32,
    api: ApiFormat,
) -> Response {
    let duration = start_time.elapsed();

    let (tx, rx) =
        tokio::sync::mpsc::channel::<Result<Bytes, Box<dyn std::error::Error + Send + Sync>>>(32);

    tokio::spawn(async move {
        let mut reqwest_stream = res.bytes_stream();
        let mut stream_success = true;
        let mut usage_scan = SseUsageScanner::default();

        while let Some(chunk_result) = reqwest_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if api == ApiFormat::Anthropic {
                        usage_scan.feed(&chunk);
                    }
                    if tx.send(Ok(chunk)).await.is_err() {
                        // Client disconnected
                        stream_success = false;
                        break;
                    }
                },
                Err(e) => {
                    stream_success = false;
                    let _ = tx
                        .send(Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>))
                        .await;
                    break;
                },
            }
        }

        // Deferred circuit-breaker feedback after stream completes
        if let Some(provider_state) = app_state.providers.iter().find(|p| p.name == provider_name) {
            let strategy = AdaptivePriorityStrategy {
                circuit: app_state.circuit,
            };
            strategy
                .feedback(
                    provider_state,
                    stream_success,
                    is_probe,
                    Some(status_code),
                    None,
                )
                .await;
            telemetry.emit(TelemetryEvent::UpdateStatus {
                provider: provider_name.clone(),
                status: circuit_status(provider_state).await,
            });

            if stream_success {
                provider_state
                    .successes
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            provider_state
                .tokens_input
                .fetch_add(usage_scan.usage.input, std::sync::atomic::Ordering::Relaxed);
            provider_state.tokens_output.fetch_add(
                usage_scan.usage.output,
                std::sync::atomic::Ordering::Relaxed,
            );
            let model_counters = provider_state
                .model_stats
                .counters(&ModelStats::key(&provider_name, &model_name));
            let route_counters = app_state.route_stats.counters(&route_name);
            for counters in [&model_counters, &route_counters] {
                counters
                    .tokens_input
                    .fetch_add(usage_scan.usage.input, std::sync::atomic::Ordering::Relaxed);
                counters.tokens_output.fetch_add(
                    usage_scan.usage.output,
                    std::sync::atomic::Ordering::Relaxed,
                );
                counters.cached_tokens.fetch_add(
                    usage_scan.usage.cached,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }

        // Push Telemetry Metrics
        telemetry.emit(TelemetryEvent::RecordTransaction {
            operation: operation.to_string(),
            provider: provider_name.clone(),
            model: model_name.clone(),
            input_tokens: usage_scan.usage.input,
            output_tokens: usage_scan.usage.output,
            duration,
            attempts,
            failure_reason: if stream_success {
                None
            } else {
                Some("stream_failed".to_string())
            },
            trace_id,
            parent_span_id,
            request_id: request_id.clone(),
        });

        // Record for the recent-request ring and live tails.
        app_state.request_log.record(RequestRecord {
            ts_ms: unix_millis(),
            endpoint: operation.to_string(),
            route: route_name,
            model: model_name,
            provider: provider_name,
            status: if stream_success { status_code } else { 0 },
            attempts,
            latency_ms: duration.as_millis() as u64,
            ttft_ms: None,
            streamed: true,
            error: if stream_success {
                None
            } else {
                Some("stream failed".to_string())
            },
            request_id,
        });
    });

    // Convert mpsc receiver to a Stream for axum
    let axum_stream = unfold(Some(rx), |state| async move {
        let mut rx = state?;
        rx.recv().await.map(|item| (item, Some(rx)))
    });

    let mut response = Response::new(Body::from_stream(axum_stream));
    *response.status_mut() = StatusCode::OK;
    copy_response_headers(&upstream_headers, response.headers_mut());
    response
}

async fn circuit_status(provider: &ProviderState) -> u64 {
    let circuit = *provider.circuit.read().await;
    let now = Instant::now();
    match circuit {
        CircuitState::Closed => {
            let rl = *provider.rate_limited_until.read().await;
            if rl.is_some() && now < rl.unwrap() {
                1 // Cooldown (rate-limited)
            } else {
                0 // Healthy
            }
        },
        CircuitState::HalfOpen => 1, // Cooldown (probing)
        CircuitState::Open { until } => {
            if now < until {
                2 // Tripped
            } else {
                1 // Cooldown expired, about to probe
            }
        },
    }
}

/// Extracts W3C traceparent segments: 00-{trace_id}-{span_id}-{flags}
fn extract_traceparent(headers: &HeaderMap) -> (Option<String>, Option<String>) {
    if let Some(val) = headers.get("traceparent").and_then(|v| v.to_str().ok()) {
        let segments: Vec<&str> = val.split('-').collect();
        if segments.len() >= 3 {
            return (Some(segments[1].to_string()), Some(segments[2].to_string()));
        }
    }
    (None, None)
}

/// Copies upstream headers safely to prevent cross-crate type conflicts (Gotcha #2)
fn copy_response_headers(src: &HeaderMap, dest: &mut HeaderMap) {
    for (key, value) in src.iter() {
        // Only copy standard metadata headers, ignore compression or transfer encodings
        let name_str = key.as_str();
        if name_str.starts_with("x-")
            || name_str == "content-type"
            || name_str == "cache-control"
            || name_str == "openai-version"
        {
            if let Ok(name) = HeaderName::from_bytes(name_str.as_bytes()) {
                if let Ok(val) = HeaderValue::from_bytes(value.as_bytes()) {
                    dest.insert(name, val);
                }
            }
        }
    }
}

/// Helper to parse Retry-After backoff duration
fn extract_retry_after(headers: &HeaderMap) -> Option<Duration> {
    if let Some(retry_after) = headers.get("retry-after").and_then(|h| h.to_str().ok()) {
        if let Ok(seconds) = retry_after.parse::<u64>() {
            return Some(Duration::from_secs(seconds));
        }
    }
    None
}

/// `anthropic-version` sent upstream when the client supplies none.
const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// Client errors that are returned to the caller as-is on Anthropic routes:
/// 4xx except 408 (timeout) and 429 (rate limit), which fail over.
fn is_passthrough_client_error(status: u16) -> bool {
    (400..500).contains(&status) && status != 408 && status != 429
}

/// Token usage extracted from an upstream response.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Usage {
    input: u64,
    output: u64,
    /// Prompt tokens served from the upstream cache.
    cached: u64,
}

/// Reads token usage from a response body (or SSE event payload). OpenAI
/// style reports `prompt_tokens`/`completion_tokens` (systemone uses
/// `input_tokens`/`output_tokens`). Anthropic reports `input_tokens`
/// excluding cache traffic, so cache reads and creations are added to the
/// input total and the reads are also surfaced as `cached`.
fn parse_usage(v: &Value, api: ApiFormat) -> Usage {
    let usage = v.get("usage");
    let field = |name: &str| -> u64 {
        usage
            .and_then(|u| u.get(name))
            .and_then(|t| t.as_u64())
            .unwrap_or(0)
    };
    match api {
        ApiFormat::Openai => Usage {
            input: usage
                .and_then(|u| u.get("prompt_tokens").or_else(|| u.get("input_tokens")))
                .and_then(|t| t.as_u64())
                .unwrap_or(0),
            output: usage
                .and_then(|u| {
                    u.get("completion_tokens")
                        .or_else(|| u.get("output_tokens"))
                })
                .and_then(|t| t.as_u64())
                .unwrap_or(0),
            cached: 0,
        },
        ApiFormat::Anthropic => {
            let cached = field("cache_read_input_tokens");
            Usage {
                input: field("input_tokens") + cached + field("cache_creation_input_tokens"),
                output: field("output_tokens"),
                cached,
            }
        },
    }
}

/// Incremental scanner over an Anthropic SSE stream that picks usage out of
/// `message_start` (input and cache tokens) and `message_delta` (cumulative
/// output tokens) events while the bytes are forwarded untouched.
#[derive(Default)]
struct SseUsageScanner {
    pending: Vec<u8>,
    usage: Usage,
}

impl SseUsageScanner {
    fn feed(&mut self, chunk: &[u8]) {
        self.pending.extend_from_slice(chunk);
        while let Some(pos) = self.pending.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=pos).collect();
            self.scan_line(&line);
        }
        // A line that never ends must not grow without bound.
        if self.pending.len() > 1 << 20 {
            self.pending.clear();
        }
    }

    fn scan_line(&mut self, line: &[u8]) {
        let Ok(text) = std::str::from_utf8(line) else {
            return;
        };
        let Some(data) = text.trim_end().strip_prefix("data:") else {
            return;
        };
        let Ok(event) = serde_json::from_str::<Value>(data.trim_start()) else {
            return;
        };
        match event.get("type").and_then(|t| t.as_str()) {
            Some("message_start") => {
                if let Some(message) = event.get("message") {
                    let start = parse_usage(message, ApiFormat::Anthropic);
                    self.usage.input = start.input;
                    self.usage.cached = start.cached;
                    self.usage.output = start.output;
                }
            },
            Some("message_delta") => {
                let delta = parse_usage(&event, ApiFormat::Anthropic);
                if delta.output > 0 {
                    self.usage.output = delta.output;
                }
            },
            _ => {},
        }
    }
}

/// Error response in the shape of the endpoint's wire format: OpenAI's
/// `{"error": {...}}` or Anthropic's `{"type": "error", "error": {...}}`.
fn error_response(api: ApiFormat, message: &str, error_type: &str, status: StatusCode) -> Response {
    match api {
        ApiFormat::Openai => json_error_response(message, error_type, status),
        ApiFormat::Anthropic => anthropic_error_response(message, error_type, status),
    }
}

/// Anthropic error body. OpenAI-flavoured type names map to Anthropic's
/// (`server_error` -> `api_error`).
pub(crate) fn anthropic_error_response(
    message: &str,
    error_type: &str,
    status: StatusCode,
) -> Response {
    let error_type = match error_type {
        "server_error" => "api_error",
        other => other,
    };
    let body = serde_json::json!({
        "type": "error",
        "error": {"type": error_type, "message": message}
    });
    let bytes = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(_) => return (status, message.to_string()).into_response(),
    };
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// Returns a structured JSON error response matching the OpenAI error format.
fn json_error_response(message: &str, error_type: &str, status: StatusCode) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": status.as_u16()
        }
    });
    let bytes = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(_) => return (status, message.to_string()).into_response(),
    };
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

// ---------------------------------------------------------------------------
// Admin handlers — must be mounted behind localhost_only middleware
// ---------------------------------------------------------------------------

/// POST /admin/providers/{name}/offline
pub async fn admin_offline(
    State(app_state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    match app_state.providers.iter().find(|p| p.name == name) {
        Some(provider) => {
            provider
                .manual_disabled
                .store(true, std::sync::atomic::Ordering::Release);
            (StatusCode::OK, format!("Provider '{}' taken offline", name))
        },
        None => (
            StatusCode::NOT_FOUND,
            format!("Provider '{}' not found", name),
        ),
    }
}

/// POST /admin/providers/{name}/online
pub async fn admin_online(
    State(app_state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    match app_state.providers.iter().find(|p| p.name == name) {
        Some(provider) => {
            provider
                .manual_disabled
                .store(false, std::sync::atomic::Ordering::Release);
            (
                StatusCode::OK,
                format!("Provider '{}' brought online", name),
            )
        },
        None => (
            StatusCode::NOT_FOUND,
            format!("Provider '{}' not found", name),
        ),
    }
}

/// POST /admin/providers/{name}/reset — resets circuit breaker to Closed,
/// failures to 0, rate limit cleared, manual disabled cleared.
pub async fn admin_reset(
    State(app_state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    match app_state.providers.iter().find(|p| p.name == name) {
        Some(provider) => {
            // Reset circuit state
            *provider.circuit.write().await = CircuitState::Closed;
            // Reset consecutive failures
            *provider.consecutive_failures.write().await = 0;
            // Clear rate limit
            *provider.rate_limited_until.write().await = None;
            // Clear manual disabled
            provider
                .manual_disabled
                .store(false, std::sync::atomic::Ordering::Release);
            (
                StatusCode::OK,
                format!("Provider '{}' reset to healthy", name),
            )
        },
        None => (
            StatusCode::NOT_FOUND,
            format!("Provider '{}' not found", name),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_usage_openai_shapes() {
        let chat = serde_json::json!({"usage": {"prompt_tokens": 3, "completion_tokens": 4}});
        assert_eq!(
            parse_usage(&chat, ApiFormat::Openai),
            Usage {
                input: 3,
                output: 4,
                cached: 0
            }
        );
        let systemone = serde_json::json!({"usage": {"input_tokens": 5, "output_tokens": 6}});
        let u = parse_usage(&systemone, ApiFormat::Openai);
        assert_eq!((u.input, u.output), (5, 6));
        assert_eq!(
            parse_usage(&serde_json::json!({}), ApiFormat::Openai),
            Usage::default()
        );
    }

    #[test]
    fn test_parse_usage_anthropic_folds_cache_into_input() {
        let v = serde_json::json!({"usage": {
            "input_tokens": 10, "output_tokens": 5,
            "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20
        }});
        assert_eq!(
            parse_usage(&v, ApiFormat::Anthropic),
            Usage {
                input: 130,
                output: 5,
                cached: 100
            }
        );
    }

    #[test]
    fn test_sse_scanner_handles_events_split_across_chunks() {
        let mut scan = SseUsageScanner::default();
        let stream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n";
        // Feed in awkward 5-byte slices.
        for piece in stream.as_bytes().chunks(5) {
            scan.feed(piece);
        }
        assert_eq!(scan.usage.input, 7);
        assert_eq!(scan.usage.output, 42);
    }

    #[test]
    fn test_passthrough_client_errors_exclude_retryable_4xx() {
        for ok in [400, 401, 403, 404, 413, 422] {
            assert!(is_passthrough_client_error(ok), "{ok}");
        }
        for retry in [408, 429, 500, 502, 529] {
            assert!(!is_passthrough_client_error(retry), "{retry}");
        }
    }

    #[test]
    fn test_anthropic_error_shape() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let res = anthropic_error_response("boom", "server_error", StatusCode::BAD_GATEWAY);
            let bytes = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
            let v: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(v["type"], "error");
            assert_eq!(v["error"]["type"], "api_error");
            assert_eq!(v["error"]["message"], "boom");
        });
    }
}
