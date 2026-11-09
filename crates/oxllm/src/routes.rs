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
    AdaptivePriorityStrategy, LatencyCandidate, Outcome,
};
use oxllm_core::state::{
    unix_millis, AppState, CircuitState, EntryAvailability, ModelStats, ProviderState,
    RequestRecord, RouteCandidate, RoutingPolicy,
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
    // Anthropic `GET /v1/models` fields. OpenAI clients ignore them, so one
    // response satisfies both SDK families.
    #[serde(rename = "type")]
    kind: &'static str,
    display_name: String,
    created_at: &'static str,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
    // Anthropic list pagination fields (the list is never paginated).
    has_more: bool,
    first_id: Option<String>,
    last_id: Option<String>,
}

/// `created` (1717070400) as RFC 3339, for Anthropic's `created_at`.
const MODEL_CREATED_AT: &str = "2024-05-30T12:00:00Z";

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
                kind: "model",
                display_name: vm_name.clone(),
                created_at: MODEL_CREATED_AT,
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
                kind: "model",
                display_name: alias.clone(),
                created_at: MODEL_CREATED_AT,
            });
        }
    }

    Json(ModelsResponse {
        object: "list",
        has_more: false,
        first_id: data.first().map(|m| m.id.clone()),
        last_id: data.last().map(|m| m.id.clone()),
        data,
    })
}

#[derive(Serialize)]
struct RouteEntry {
    provider: String,
    model: String,
    circuit: String,
    /// Machine-readable availability: healthy, probing, rate_limited, open
    /// or offline. `circuit` is the same fact as display text.
    availability: &'static str,
    /// Seconds until an open or rate-limited entry is tried again.
    #[serde(skip_serializing_if = "Option::is_none")]
    retry_in_secs: Option<u64>,
    /// Wire format of the provider (openai / anthropic).
    api: &'static str,
    /// Explicit `priority` of the target within the route, if configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    priority: Option<u32>,
    /// Load-sharing `weight` of the target within its priority tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    weight: Option<u32>,
    /// Configured cost per 1M output tokens (used by `cheapest`).
    #[serde(skip_serializing_if = "Option::is_none")]
    cost_per_mtok: Option<f64>,
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
        /// Plain-language ordering rule, including weights and priorities.
        strategy: String,
        /// Pool entries after flattening nested virtual models.
        targets: usize,
        /// Entries that can take a request right now.
        available: usize,
        /// ok (all available), degraded (some), or down (none).
        state: &'static str,
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

    /// One thing that needs attention, for the summary view and scripts.
    #[derive(Serialize)]
    struct Problem {
        /// route, provider or reload
        scope: &'static str,
        name: String,
        /// down, degraded, offline, rate_limited, or reload_failed
        kind: &'static str,
        detail: String,
    }

    #[derive(Serialize)]
    struct Health {
        /// ok, degraded, or down (any route with no available entry).
        state: &'static str,
        problems: Vec<Problem>,
    }

    #[derive(Serialize)]
    struct StatusResponse {
        health: Health,
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
        // Show entries in the order routing tries them (stable by priority),
        // not raw config order, so the table matches behaviour.
        let mut ordered: Vec<&oxllm_core::config::VirtualModelTarget> = targets.iter().collect();
        ordered.sort_by_key(|t| t.priority.unwrap_or(u32::MAX));
        for target in ordered {
            let provider_state = app_state
                .providers
                .iter()
                .find(|p| p.name == target.provider);
            let availability = match provider_state {
                Some(provider) => Some(provider.entry_availability(&target.model).await),
                None => None,
            };
            let circuit_str = match availability {
                Some(EntryAvailability::Healthy) => "Closed (Healthy)".to_string(),
                Some(EntryAvailability::Probing) => "Half-Open (Probing)".to_string(),
                Some(EntryAvailability::Open { secs }) => format!("Open ({}s cooldown)", secs),
                Some(EntryAvailability::RateLimited { secs }) => {
                    format!("Rate-limited ({}s)", secs)
                },
                Some(EntryAvailability::Offline) => "Offline (admin)".to_string(),
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
                availability: availability.map_or("unknown", |a| a.name()),
                retry_in_secs: availability.and_then(|a| a.retry_in_secs()),
                api: provider_state.map_or("openai", |p| p.api.as_str()),
                priority: target.priority,
                weight: target.weight,
                cost_per_mtok: provider_state.and_then(|p| p.cost_per_mtok),
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
            let entries = virtual_models.get(&route_name);
            let targets = entries.map_or(0, |e| e.len());
            let available = entries.map_or(0, |e| {
                e.iter()
                    .filter(|x| matches!(x.availability, "healthy" | "probing"))
                    .count()
            });
            let strategy = describe_strategy(
                app_state.policies.get(&route_name).copied(),
                app_state
                    .virtual_models
                    .get(&route_name)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            );
            RouteSummary {
                strategy,
                targets,
                available,
                state: if available == 0 {
                    "down"
                } else if available < targets {
                    "degraded"
                } else {
                    "ok"
                },
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

    // Problems, most severe first: routes with nothing available, a failed
    // reload, then degraded routes and individual entries that are out.
    let mut problems: Vec<Problem> = Vec::new();
    for r in &routes {
        if r.state == "down" {
            problems.push(Problem {
                scope: "route",
                name: r.route.clone(),
                kind: "down",
                detail: format!("0 of {} entries available", r.targets),
            });
        }
    }
    if let Some(err) = app_state.reload_health.snapshot().last_error {
        problems.push(Problem {
            scope: "reload",
            name: "config".to_string(),
            kind: "reload_failed",
            detail: format!("previous config still serving: {}", err),
        });
    }
    for r in &routes {
        if r.state == "degraded" {
            problems.push(Problem {
                scope: "route",
                name: r.route.clone(),
                kind: "degraded",
                detail: format!("{} of {} entries available", r.available, r.targets),
            });
        }
    }
    let mut seen_entries = std::collections::HashSet::new();
    let mut vm_names: Vec<&String> = virtual_models.keys().collect();
    vm_names.sort();
    for vm in vm_names {
        for e in &virtual_models[vm] {
            if matches!(e.availability, "healthy" | "probing") {
                continue;
            }
            if !seen_entries.insert((e.provider.clone(), e.model.clone(), e.availability)) {
                continue;
            }
            let kind = match e.availability {
                "offline" => "offline",
                "rate_limited" => "rate_limited",
                _ => "down",
            };
            let detail = match e.retry_in_secs {
                Some(s) => format!("{} ({}), retry in {}s", e.model, e.availability, s),
                None => format!("{} ({})", e.model, e.availability),
            };
            problems.push(Problem {
                scope: "provider",
                name: e.provider.clone(),
                kind,
                detail,
            });
        }
    }
    let overall = if routes.iter().any(|r| r.state == "down") {
        "down"
    } else if !problems.is_empty() {
        "degraded"
    } else {
        "ok"
    };

    Json(StatusResponse {
        health: Health {
            state: overall,
            problems,
        },
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

/// Plain-language description of how a route orders its entries, for
/// `/status`: the policy plus any explicit priorities and weights.
fn describe_strategy(
    policy: Option<RoutingPolicy>,
    targets: &[oxllm_core::config::VirtualModelTarget],
) -> String {
    match policy {
        Some(RoutingPolicy::Fastest) => {
            "fastest: lowest measured time-to-first-byte first (20% bands, re-sampled every 20th request)"
                .to_string()
        },
        Some(RoutingPolicy::Cheapest) => {
            "cheapest: lowest cost_per_mtok first, latency breaks ties".to_string()
        },
        _ => {
            let weighted = targets.iter().any(|t| t.weight.is_some());
            let prioritised = targets.iter().any(|t| t.priority.is_some());
            let mut s = if prioritised {
                "priority: lowest `priority` first, then configured order".to_string()
            } else {
                "priority: configured order, first healthy wins".to_string()
            };
            if weighted {
                s.push_str("; entries with a weight share load within their tier");
            }
            s.push_str("; failover on error");
            s
        },
    }
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
    /// Stateful endpoint (OpenAI Responses / Conversations): ids the
    /// upstream returns are recorded, follow-ups naming one are pinned to its
    /// provider without failover, and upstream client errors pass through
    /// verbatim.
    stateful: bool,
    /// Where the virtual model comes from.
    model: ModelField,
}

/// How an endpoint names its virtual model.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelField {
    /// Body `model` is required and rewritten per target.
    Required,
    /// Body `model` if present, else the `x-oxllm-model` header, else the
    /// default model; written into the body per target.
    Optional,
    /// The body has no `model` field (Conversations create): the header or
    /// default model chooses the route and the body is not touched.
    Absent,
}

/// Header naming the virtual model for endpoints whose body has none.
const MODEL_HEADER: &str = "x-oxllm-model";

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

    if spec.model != ModelField::Required && !payload.is_object() {
        return error_response(
            spec.api,
            "Request body must be a JSON object",
            "invalid_request_error",
            StatusCode::BAD_REQUEST,
        );
    }
    let body_model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .map(String::from);
    let header_model = || {
        headers
            .get(MODEL_HEADER)
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
            .map(String::from)
    };
    let requested_model = match spec.model {
        ModelField::Required => match body_model {
            Some(m) => m,
            None => {
                return error_response(
                    spec.api,
                    "Missing required 'model' field",
                    "invalid_request_error",
                    StatusCode::BAD_REQUEST,
                )
            },
        },
        ModelField::Optional => body_model.or_else(header_model).unwrap_or_default(),
        ModelField::Absent => header_model().unwrap_or_default(),
    };

    // Owned copy: `payload` is mutated per attempt, so the &str borrow cannot
    // live across the routing loop. Names resolve through aliases and the
    // configured default model before being rejected.
    let route_name = match app_state.resolve_route(&requested_model) {
        Some(route) => route.to_string(),
        None if requested_model.is_empty() => {
            return error_response(
                spec.api,
                &format!(
                    "No model for this request: name a virtual model in the {} header, \
                     or configure server.default_model",
                    MODEL_HEADER
                ),
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            )
        },
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
    let mut candidates = ordered_candidates(
        &app_state,
        &route_name,
        session_key(&app_state, &headers, &payload).as_deref(),
        &candidates,
    );

    // Stateful follow-ups can only be served where the state lives. An id
    // oxllm has no record of (created elsewhere, or forgotten on restart) is
    // looked up on the route's providers; if none claims it, the request
    // routes by policy and the upstream decides.
    let refs = if spec.stateful {
        referenced_state_ids(&payload)
    } else {
        Vec::new()
    };
    let mut pinned = refs.iter().find_map(|r| {
        app_state
            .response_affinity
            .pinned_provider(&r.id)
            .map(|provider| (r.id.clone(), provider))
    });
    if pinned.is_none() {
        if let Some(r) = refs.first() {
            let mut providers: Vec<&ProviderState> = Vec::new();
            for c in &candidates {
                if !providers.iter().any(|p| p.name == c.provider.name) {
                    providers.push(c.provider);
                }
            }
            if let Some(owner) = discover_state_owner(&app_state, r, &providers).await {
                pinned = Some((r.id.clone(), owner));
            }
        }
    }
    if let Some((id, provider)) = &pinned {
        candidates.retain(|c| &c.provider.name == provider);
        if candidates.is_empty() {
            return error_response(
                spec.api,
                &format!(
                    "'{}' is held by provider '{}', which does not serve model '{}'",
                    id, provider, route_name
                ),
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            );
        }
    }

    // Route-level counters: one row per virtual model, one request each,
    // regardless of how many provider attempts it takes.
    let route_counters = app_state.route_stats.counters(&route_name);
    route_counters
        .requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Each pool entry (provider + model) is tried at most once per request,
    // so a provider listed twice with different models gets its second entry
    // as a genuine fallback instead of re-hitting the first.
    let mut tried = vec![false; candidates.len()];
    loop {
        let remaining: Vec<usize> = (0..candidates.len()).filter(|i| !tried[*i]).collect();
        let entries: Vec<(&ProviderState, &str)> = remaining
            .iter()
            .map(|&i| (candidates[i].provider, candidates[i].model.as_str()))
            .collect();
        let Some((picked, selected)) = strategy.select_entry(&entries).await else {
            break;
        };
        attempts += 1;
        let entry = remaining[picked];
        tried[entry] = true;
        let target_model = candidates[entry].model.clone();
        let held = (selected.is_probe, selected.model_probe);

        // 1. Rewrite model field in JSON body
        if spec.model != ModelField::Absent {
            payload["model"] = Value::String(target_model.clone());
        }
        let rewritten_body = Bytes::from(serde_json::to_vec(&payload).unwrap());

        // 2. Safely parse base URL and join embeddings path (Url::joinTrailingSlashes mitigation)
        let endpoint_url = match selected.base_url.join(spec.path) {
            Ok(url) => url,
            Err(e) => {
                error!("Invalid base URL path join for {}: {}", selected.name, e);
                strategy.release_probes(candidates[entry].provider, &target_model, held);
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
                with_provider_auth(req, selected.auth_header.as_ref(), &selected.api_key)
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

        // Streaming requests may also carry a first-byte deadline, so a
        // hung endpoint is abandoned long before the total timeout.
        let first_byte = app_state
            .providers
            .iter()
            .find(|p| p.name == selected.name)
            .and_then(|p| p.tuning.first_byte_timeout_secs)
            .filter(|_| is_streaming && spec.streaming)
            .map(Duration::from_secs);
        let res = match first_byte {
            Some(limit) => match tokio::time::timeout(limit, req.send()).await {
                Ok(r) => r,
                Err(_) => {
                    warn!(
                        request_id = %request_id,
                        "{} request upstream {} sent no response headers within {}s",
                        spec.operation,
                        selected.name,
                        limit.as_secs()
                    );
                    last_upstream_error = format!(
                        "no response headers within first_byte_timeout_secs ({}s)",
                        limit.as_secs()
                    );
                    last_failed_provider = selected.name.clone();
                    last_failed_model = target_model.clone();
                    last_failed_status = 0;
                    if let Some(state) =
                        app_state.providers.iter().find(|p| p.name == selected.name)
                    {
                        strategy
                            .feedback_entry(state, &target_model, held, Outcome::ModelTimeout, None)
                            .await;
                        telemetry.emit(TelemetryEvent::UpdateStatus {
                            provider: selected.name.clone(),
                            status: circuit_status(state).await,
                        });
                    }
                    continue;
                },
            },
            None => req.send().await,
        };

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
                        (selected.is_probe, selected.model_probe),
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
                        spec.stateful,
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
                            .feedback_entry(
                                target_provider_state,
                                &target_model,
                                held,
                                Outcome::ModelError,
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
                let parsed_body = serde_json::from_slice::<Value>(&res_body).ok();
                let usage = parsed_body
                    .as_ref()
                    .map(|v| parse_usage(v, spec.api))
                    .unwrap_or_default();
                if spec.stateful {
                    if let Some(v) = &parsed_body {
                        for id in created_state_ids(v) {
                            app_state.response_affinity.pin(&id, &selected.name);
                        }
                    }
                }
                let (input_tokens, output_tokens) = (usage.input, usage.output);

                // Report Success feedback
                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback_entry(
                        target_provider_state,
                        &target_model,
                        held,
                        Outcome::Success,
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
                    route: route_name.clone(),
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
                if (spec.api == ApiFormat::Anthropic || spec.stateful)
                    && is_passthrough_client_error(status_code)
                {
                    let upstream_headers = res.headers().clone();
                    let error_body = res.bytes().await.unwrap_or_default();
                    // Release any lock-free probe permits; circuits stay as
                    // they were so the next request probes again.
                    strategy.release_probes(candidates[entry].provider, &target_model, held);
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
                let outcome = if status_code == 429 {
                    Outcome::RateLimited
                } else {
                    Outcome::ModelError
                };
                strategy
                    .feedback_entry(
                        target_provider_state,
                        &target_model,
                        held,
                        outcome,
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
                let outcome = if e.is_timeout() {
                    Outcome::ModelTimeout
                } else {
                    Outcome::Unreachable
                };
                strategy
                    .feedback_entry(target_provider_state, &target_model, held, outcome, None)
                    .await;
                telemetry.emit(TelemetryEvent::UpdateStatus {
                    provider: selected.name.clone(),
                    status: circuit_status(target_provider_state).await,
                });
            },
        }
    }

    let message = if let Some((id, provider)) = &pinned {
        format!(
            "Provider '{}' holds the state for '{}' and failed or is unavailable; \
             stateful follow-ups cannot fail over. Last error ({}): {}",
            provider, id, last_failed_status, last_upstream_error
        )
    } else if !last_upstream_error.is_empty() {
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
    // A request that exhausts every entry is the most interesting one to
    // trace, so it gets a span and a duration sample like any other.
    telemetry.emit(TelemetryEvent::RecordTransaction {
        operation: spec.operation.to_string(),
        route: route_name.clone(),
        provider: last_failed_provider.clone(),
        model: last_failed_model.clone(),
        input_tokens: 0,
        output_tokens: 0,
        duration: start_time.elapsed(),
        attempts,
        failure_reason: Some(if attempts == 0 {
            "no_available_provider".to_string()
        } else {
            "all_providers_failed".to_string()
        }),
        trace_id: trace_id.clone(),
        parent_span_id: parent_span_id.clone(),
        request_id: request_id.clone(),
    });
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
    stateful: false,
    model: ModelField::Required,
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
    stateful: false,
    model: ModelField::Required,
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
    stateful: false,
    model: ModelField::Required,
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
    stateful: false,
    model: ModelField::Required,
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
    stateful: false,
    model: ModelField::Required,
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
    stateful: false,
    model: ModelField::Required,
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
    held: (bool, bool),
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
    stateful: bool,
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
                    if api == ApiFormat::Anthropic || stateful {
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

        // The upstream holds state for any id it announced, even if the
        // stream later broke, so pin regardless of stream success.
        for id in [&usage_scan.response_id, &usage_scan.conversation_id]
            .into_iter()
            .flatten()
        {
            app_state.response_affinity.pin(id, &provider_name);
        }

        // Deferred circuit-breaker feedback after stream completes
        if let Some(provider_state) = app_state.providers.iter().find(|p| p.name == provider_name) {
            let strategy = AdaptivePriorityStrategy {
                circuit: app_state.circuit,
            };
            strategy
                .feedback_entry(
                    provider_state,
                    &model_name,
                    held,
                    if stream_success {
                        Outcome::Success
                    } else {
                        Outcome::ModelError
                    },
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
            route: route_name.clone(),
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
            // Anthropic: request id, rate-limit state and backoff hints that
            // SDKs read for debugging and retries.
            || name_str == "request-id"
            || name_str == "retry-after"
            || name_str.starts_with("anthropic-")
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
    /// Responses API: ids announced in `response.created`.
    response_id: Option<String>,
    conversation_id: Option<String>,
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
            // OpenAI Responses streaming events.
            Some("response.created") => {
                if let Some(response) = event.get("response") {
                    let mut ids = created_state_ids(response).into_iter();
                    self.response_id = ids.next();
                    self.conversation_id = ids.next();
                }
            },
            Some("response.completed" | "response.incomplete" | "response.failed") => {
                if let Some(response) = event.get("response") {
                    self.usage = parse_usage(response, ApiFormat::Openai);
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
// OpenAI Responses + Conversations APIs (stateful, provider-pinned)
// ---------------------------------------------------------------------------

const RESPONSES_SPEC: EndpointSpec = EndpointSpec {
    path: "responses",
    operation: "responses",
    failure_noun: "responses",
    streaming: true,
    api: ApiFormat::Openai,
    stateful: true,
    model: ModelField::Required,
};

/// `model` is optional on compact and input_tokens; when absent oxllm
/// takes the route from `x-oxllm-model` or the default model.
const RESPONSES_COMPACT_SPEC: EndpointSpec = EndpointSpec {
    path: "responses/compact",
    operation: "responses.compact",
    failure_noun: "responses compact",
    streaming: false,
    api: ApiFormat::Openai,
    stateful: true,
    model: ModelField::Optional,
};

const RESPONSES_INPUT_TOKENS_SPEC: EndpointSpec = EndpointSpec {
    path: "responses/input_tokens",
    operation: "responses.input_tokens",
    failure_noun: "responses input_tokens",
    streaming: false,
    api: ApiFormat::Openai,
    stateful: true,
    model: ModelField::Optional,
};

/// A new conversation has no state yet, so it routes and fails over like
/// any generation; the body has no `model`, so the route comes from
/// `x-oxllm-model` or the default model.
const CONVERSATIONS_SPEC: EndpointSpec = EndpointSpec {
    path: "conversations",
    operation: "conversations.create",
    failure_noun: "conversations",
    streaming: false,
    api: ApiFormat::Openai,
    stateful: true,
    model: ModelField::Absent,
};

/// POST /v1/responses
///
/// OpenAI Responses API passthrough. New conversations route and fail over
/// like chat; the returned `id` (and `conversation.id`) is recorded so that
/// follow-ups carrying `previous_response_id` or `conversation` go to the
/// provider that holds the state, with no failover.
pub async fn create_responses(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &RESPONSES_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

/// POST /v1/responses/compact
pub async fn compact_responses(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &RESPONSES_COMPACT_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

/// POST /v1/responses/input_tokens
pub async fn count_response_input_tokens(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &RESPONSES_INPUT_TOKENS_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

/// POST /v1/conversations
pub async fn create_conversation(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy_request(
        &CONVERSATIONS_SPEC,
        app_state,
        telemetry,
        request_id,
        headers,
        body,
    )
    .await
}

/// Which upstream collection a state id lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateKind {
    Response,
    Conversation,
}

impl StateKind {
    fn collection(self) -> &'static str {
        match self {
            StateKind::Response => "responses",
            StateKind::Conversation => "conversations",
        }
    }
}

/// A state id named by a request.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StateRef {
    id: String,
    kind: StateKind,
}

/// Ids a Responses request refers to: `previous_response_id`, then
/// `conversation` (a string or `{"id": ...}`).
fn referenced_state_ids(payload: &Value) -> Vec<StateRef> {
    let conversation = payload.get("conversation").and_then(|c| {
        c.as_str()
            .or_else(|| c.get("id").and_then(|id| id.as_str()))
    });
    [
        (
            payload.get("previous_response_id").and_then(|v| v.as_str()),
            StateKind::Response,
        ),
        (conversation, StateKind::Conversation),
    ]
    .into_iter()
    .filter_map(|(id, kind)| {
        id.filter(|id| !id.is_empty()).map(|id| StateRef {
            id: id.to_string(),
            kind,
        })
    })
    .collect()
}

/// Ids a Responses (or Conversations) object creates or extends: its `id`,
/// then `conversation.id` when present.
fn created_state_ids(response: &Value) -> Vec<String> {
    [
        response.get("id").and_then(|v| v.as_str()),
        response
            .get("conversation")
            .and_then(|c| c.get("id"))
            .and_then(|v| v.as_str()),
    ]
    .into_iter()
    .flatten()
    .filter(|id| !id.is_empty())
    .map(String::from)
    .collect()
}

/// Upstream ids are opaque but path-safe (`resp_...`, `conv_...`,
/// `msg_...`); anything else is refused rather than spliced into an
/// upstream URL.
fn valid_state_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Finds which provider holds an id oxllm has no record of (created before
/// a restart, by another proxy, or evicted) by asking each candidate for
/// it with a cheap `GET /{collection}/{id}`. The first provider answering
/// 2xx is pinned and returned. Providers that are offline, or whose
/// provider-level circuit is open, are skipped; lookups give no circuit
/// feedback because a 404 says nothing about health.
async fn discover_state_owner(
    app_state: &AppState,
    state: &StateRef,
    providers: &[&ProviderState],
) -> Option<String> {
    if !valid_state_id(&state.id) {
        return None;
    }
    for provider in providers {
        if provider
            .manual_disabled
            .load(std::sync::atomic::Ordering::Relaxed)
            || provider.api != ApiFormat::Openai
        {
            continue;
        }
        if matches!(
            *provider.circuit.read().await,
            CircuitState::Open { until } if Instant::now() < until
        ) {
            continue;
        }
        let Ok(url) = provider
            .base_url
            .join(&format!("{}/{}", state.kind.collection(), state.id))
        else {
            continue;
        };
        let req = app_state
            .http_client
            .get(url.as_str())
            .timeout(Duration::from_secs(
                provider
                    .upstream_timeout_secs
                    .unwrap_or(app_state.upstream_timeout_secs),
            ));
        let res = with_provider_auth(req, provider.tuning.auth_header.as_ref(), &provider.api_key)
            .send()
            .await;
        match res {
            Ok(r) if r.status().is_success() => {
                debug!("state id {} found on provider {}", state.id, provider.name);
                app_state.response_affinity.pin(&state.id, &provider.name);
                return Some(provider.name.clone());
            },
            Ok(r) => debug!(
                "state id {} lookup on {}: {}",
                state.id,
                provider.name,
                r.status()
            ),
            Err(e) => debug!(
                "state id {} lookup on {} failed: {}",
                state.id, provider.name, e
            ),
        }
    }
    None
}

/// GET or DELETE /v1/responses/{id}
pub async fn response_by_id(
    State((app_state, _telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    method: axum::http::Method,
    uri: axum::http::Uri,
    axum::extract::Path(id): axum::extract::Path<String>,
    body: Bytes,
) -> Response {
    let target = OwnerTarget {
        kind: StateKind::Response,
        id: &id,
        rest: &[],
    };
    forward_to_state_owner(&app_state, &request_id, method, &uri, target, body).await
}

/// GET /v1/responses/{id}/input_items, POST /v1/responses/{id}/cancel
pub async fn response_action(
    State((app_state, _telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    method: axum::http::Method,
    uri: axum::http::Uri,
    axum::extract::Path((id, action)): axum::extract::Path<(String, String)>,
    body: Bytes,
) -> Response {
    let allowed = matches!(
        (&method, action.as_str()),
        (&axum::http::Method::GET, "input_items") | (&axum::http::Method::POST, "cancel")
    );
    if !allowed {
        return unknown_endpoint(&method, &format!("/v1/responses/{{id}}/{}", action));
    }
    let target = OwnerTarget {
        kind: StateKind::Response,
        id: &id,
        rest: &[action.as_str()],
    };
    forward_to_state_owner(&app_state, &request_id, method, &uri, target, body).await
}

/// GET, POST (update metadata) or DELETE /v1/conversations/{id}
pub async fn conversation_by_id(
    State((app_state, _telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    method: axum::http::Method,
    uri: axum::http::Uri,
    axum::extract::Path(id): axum::extract::Path<String>,
    body: Bytes,
) -> Response {
    let target = OwnerTarget {
        kind: StateKind::Conversation,
        id: &id,
        rest: &[],
    };
    forward_to_state_owner(&app_state, &request_id, method, &uri, target, body).await
}

/// GET (list) or POST (append) /v1/conversations/{id}/items
pub async fn conversation_items(
    State((app_state, _telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    method: axum::http::Method,
    uri: axum::http::Uri,
    axum::extract::Path(id): axum::extract::Path<String>,
    body: Bytes,
) -> Response {
    let target = OwnerTarget {
        kind: StateKind::Conversation,
        id: &id,
        rest: &["items"],
    };
    forward_to_state_owner(&app_state, &request_id, method, &uri, target, body).await
}

/// GET or DELETE /v1/conversations/{id}/items/{item_id}
pub async fn conversation_item(
    State((app_state, _telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    method: axum::http::Method,
    uri: axum::http::Uri,
    axum::extract::Path((id, item_id)): axum::extract::Path<(String, String)>,
    body: Bytes,
) -> Response {
    if !valid_state_id(&item_id) {
        return json_error_response(
            "Invalid item id",
            "invalid_request_error",
            StatusCode::BAD_REQUEST,
        );
    }
    let target = OwnerTarget {
        kind: StateKind::Conversation,
        id: &id,
        rest: &["items", item_id.as_str()],
    };
    forward_to_state_owner(&app_state, &request_id, method, &uri, target, body).await
}

fn unknown_endpoint(method: &axum::http::Method, path: &str) -> Response {
    json_error_response(
        &format!("Unknown endpoint: {} {}", method, path),
        "invalid_request_error",
        StatusCode::NOT_FOUND,
    )
}

/// The upstream object a sub-resource call addresses:
/// `{collection}/{id}/{rest...}`.
#[derive(Clone, Copy)]
struct OwnerTarget<'a> {
    kind: StateKind,
    id: &'a str,
    /// Path segments after the id, already validated or fixed literals.
    rest: &'a [&'a str],
}

/// Forwards a Responses / Conversations sub-resource call to the provider
/// holding the id (found by lookup when oxllm has no record). There is no
/// failover and no circuit feedback: only that provider can answer, and
/// these are cheap metadata calls rather than generations.
async fn forward_to_state_owner(
    app_state: &AppState,
    request_id: &str,
    method: axum::http::Method,
    uri: &axum::http::Uri,
    target: OwnerTarget<'_>,
    body: Bytes,
) -> Response {
    let id = target.id;
    if !valid_state_id(id) {
        return json_error_response(
            "Invalid id",
            "invalid_request_error",
            StatusCode::BAD_REQUEST,
        );
    }
    let provider_name = match app_state.response_affinity.pinned_provider(id) {
        Some(name) => name,
        None => {
            let openai: Vec<&ProviderState> = app_state
                .providers
                .iter()
                .filter(|p| p.api == ApiFormat::Openai)
                .collect();
            let state = StateRef {
                id: id.to_string(),
                kind: target.kind,
            };
            match discover_state_owner(app_state, &state, &openai).await {
                Some(name) => name,
                None => {
                    return json_error_response(
                        &format!("No configured provider holds '{}'", id),
                        "invalid_request_error",
                        StatusCode::NOT_FOUND,
                    )
                },
            }
        },
    };
    let Some(provider) = app_state.providers.iter().find(|p| p.name == provider_name) else {
        return json_error_response(
            &format!(
                "Provider '{}' holding '{}' is no longer configured",
                provider_name, id
            ),
            "server_error",
            StatusCode::BAD_GATEWAY,
        );
    };
    if provider
        .manual_disabled
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        return json_error_response(
            &format!("Provider '{}' holding '{}' is offline", provider_name, id),
            "server_error",
            StatusCode::SERVICE_UNAVAILABLE,
        );
    }

    let mut path = format!("{}/{}", target.kind.collection(), id);
    for segment in target.rest {
        path.push('/');
        path.push_str(segment);
    }
    let mut url = match provider.base_url.join(&path) {
        Ok(url) => url,
        Err(e) => {
            error!("Invalid base URL path join for {}: {}", provider.name, e);
            return json_error_response(
                "Invalid upstream URL",
                "server_error",
                StatusCode::BAD_GATEWAY,
            );
        },
    };
    url.set_query(uri.query());

    let start = Instant::now();
    let mut req = app_state
        .http_client
        .request(method.clone(), url.as_str())
        .timeout(Duration::from_secs(
            provider
                .upstream_timeout_secs
                .unwrap_or(app_state.upstream_timeout_secs),
        ));
    req = with_provider_auth(req, provider.tuning.auth_header.as_ref(), &provider.api_key);
    if !body.is_empty() {
        req = req.header("Content-Type", "application/json").body(body);
    }
    let mut operation = target.kind.collection().to_string();
    for segment in target.rest.iter().take(1) {
        operation.push('.');
        operation.push_str(segment);
    }
    operation.push('.');
    operation.push_str(&method.as_str().to_ascii_lowercase());
    let (status, upstream_headers, res_body, error) = match req.send().await {
        Ok(res) => {
            let status = res.status();
            let headers = res.headers().clone();
            match res.bytes().await {
                Ok(b) => {
                    let error =
                        (!status.is_success()).then(|| String::from_utf8_lossy(&b).to_string());
                    (status, headers, b, error)
                },
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    HeaderMap::new(),
                    Bytes::new(),
                    Some(e.to_string()),
                ),
            }
        },
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            HeaderMap::new(),
            Bytes::new(),
            Some(e.to_string()),
        ),
    };
    app_state.request_log.record(RequestRecord {
        ts_ms: unix_millis(),
        endpoint: operation,
        route: String::new(),
        model: String::new(),
        provider: provider.name.clone(),
        status: status.as_u16(),
        attempts: 1,
        latency_ms: start.elapsed().as_millis() as u64,
        ttft_ms: None,
        streamed: false,
        error: error.clone(),
        request_id: request_id.to_string(),
    });
    if res_body.is_empty() && status == StatusCode::BAD_GATEWAY {
        return json_error_response(
            &format!(
                "Provider '{}' holding '{}' failed: {}",
                provider.name,
                id,
                error.unwrap_or_default()
            ),
            "server_error",
            StatusCode::BAD_GATEWAY,
        );
    }
    // Deleting the object itself (not one of its items) ends its life.
    if method == axum::http::Method::DELETE && target.rest.is_empty() && status.is_success() {
        app_state.response_affinity.unpin(id);
    }
    let mut response = Response::new(Body::from(res_body));
    *response.status_mut() = status;
    copy_response_headers(&upstream_headers, response.headers_mut());
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

/// Attaches an OpenAI-format provider's key: raw in its configured
/// `auth_header`, else as `Authorization: Bearer`.
fn with_provider_auth(
    req: reqwest::RequestBuilder,
    auth_header: Option<&reqwest::header::HeaderName>,
    api_key: &str,
) -> reqwest::RequestBuilder {
    match auth_header {
        Some(name) => req.header(name, api_key),
        None => req.header("Authorization", format!("Bearer {api_key}")),
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
    fn test_responses_state_id_extraction() {
        let req =
            serde_json::json!({"previous_response_id": "resp_1", "conversation": {"id": "conv_1"}});
        let refs = |v: &Value| -> Vec<(String, StateKind)> {
            referenced_state_ids(v)
                .into_iter()
                .map(|r| (r.id, r.kind))
                .collect()
        };
        assert_eq!(
            refs(&req),
            [
                ("resp_1".to_string(), StateKind::Response),
                ("conv_1".to_string(), StateKind::Conversation)
            ]
        );
        let req = serde_json::json!({"conversation": "conv_2"});
        assert_eq!(
            refs(&req),
            [("conv_2".to_string(), StateKind::Conversation)]
        );
        assert!(referenced_state_ids(&serde_json::json!({"input": "x"})).is_empty());
        let res = serde_json::json!({"id": "resp_9", "conversation": {"id": "conv_9"}});
        assert_eq!(created_state_ids(&res), ["resp_9", "conv_9"]);
    }

    #[test]
    fn test_valid_state_id_rejects_path_tricks() {
        assert!(valid_state_id("resp_abc-123"));
        for bad in ["", "resp/..", "..", "resp?x=1", "resp%2F", "a b"] {
            assert!(!valid_state_id(bad), "{bad}");
        }
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
