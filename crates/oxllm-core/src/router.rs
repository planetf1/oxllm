use crate::state::{CircuitConfig, CircuitState, ModelHealth, ProviderState, SelectedProvider};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::info;

/// Modular strategy trait for selecting providers and updating their health state.
#[allow(async_fn_in_trait)]
pub trait RoutingStrategy: Send + Sync {
    /// Selects a candidate provider from the candidate list for a given virtual model.
    /// Performs dynamic idle-based penalty decay check prior to candidate evaluation.
    async fn select(&self, candidates: &[&ProviderState]) -> Option<SelectedProvider>;

    /// Records routing feedback (success or failure) to adjust candidate backoffs and circuits.
    async fn feedback(
        &self,
        provider: &ProviderState,
        success: bool,
        is_probe: bool,
        status_code: Option<u16>,
        retry_after: Option<Duration>,
    );
}

/// The default routing strategy: orders by priority, failover with exponential backoffs,
/// strict HalfOpen single-probe guards, and dynamic idle-based penalty decay.
/// Circuit sensitivity comes from the carried [`CircuitConfig`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AdaptivePriorityStrategy {
    pub circuit: CircuitConfig,
}

/// Borrowed view over the circuit-breaker fields that a provider and a
/// (provider, model) pair both carry, so one implementation of the breaker
/// serves both scopes.
pub struct CircuitView<'a> {
    pub name: &'a str,
    pub circuit: &'a RwLock<CircuitState>,
    pub failures: &'a RwLock<u32>,
    pub rate_limited_until: &'a RwLock<Option<Instant>>,
    pub last_attempt: &'a RwLock<Option<Instant>>,
    pub probe: &'a AtomicBool,
}

impl<'a> CircuitView<'a> {
    fn of_provider(p: &'a ProviderState) -> Self {
        Self {
            name: &p.name,
            circuit: &p.circuit,
            failures: &p.consecutive_failures,
            rate_limited_until: &p.rate_limited_until,
            last_attempt: &p.last_attempt_time,
            probe: &p.probe_in_flight,
        }
    }

    fn of_model(name: &'a str, h: &'a ModelHealth) -> Self {
        Self {
            name,
            circuit: &h.circuit,
            failures: &h.consecutive_failures,
            rate_limited_until: &h.rate_limited_until,
            last_attempt: &h.last_attempt_time,
            probe: &h.probe_in_flight,
        }
    }
}

/// Breaker limits for one provider: failures that open a circuit, and the
/// first cooldown (which doubles per failure beyond the threshold).
#[derive(Clone, Copy)]
struct Limits {
    threshold: u32,
    base_cooldown_secs: u64,
}

/// Result of checking one breaker scope for a request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Admit {
    Skip,
    Go,
    /// Admitted as the single HalfOpen probe; the caller holds the permit.
    Probe,
}

/// What happened to an attempt against one pool entry, as far as health
/// accounting is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The upstream answered successfully.
    Success,
    /// The host could not be reached (refused, DNS, TLS, reset): counted
    /// against the provider, since every model on it is equally affected.
    Unreachable,
    /// The upstream answered 429: a rate-limit window for this model,
    /// honouring `retry_after`. Gateways such as LiteLLM rate-limit per
    /// model ("No deployments available for selected model"), so a sibling
    /// model on the same provider keeps serving; a key-wide limit costs one
    /// 429 per model rather than hiding the siblings of a single model.
    RateLimited,
    /// The upstream answered with a server error, or a body or stream broke:
    /// counted against this model only.
    ModelError,
    /// The upstream did not answer in time: opens this model's circuit at
    /// once, since a hang costs every caller a full timeout.
    ModelTimeout,
}

impl AdaptivePriorityStrategy {
    /// Consecutive failures that open this provider's circuit: its own
    /// override, else the server-wide value.
    fn threshold(&self, provider: &ProviderState) -> u32 {
        provider
            .tuning
            .circuit_failure_threshold
            .unwrap_or(self.circuit.failure_threshold)
    }

    /// Feedback for an upstream that did not answer within its timeout. A
    /// hung endpoint costs every caller a full timeout, so unlike an
    /// ordinary failure it opens the circuit immediately instead of waiting
    /// for `failure_threshold` victims. Provider-scoped; the request path
    /// uses [`Self::feedback_entry`], which scopes a timeout to the model.
    pub async fn feedback_timeout(&self, provider: &ProviderState, is_probe: bool) {
        {
            let mut failures = provider.consecutive_failures.write().await;
            *failures = (*failures).max(self.threshold(provider).saturating_sub(1));
        }
        self.feedback(provider, false, is_probe, None, None).await;
    }

    /// Decides whether a breaker scope may take a request now: ages old
    /// failures, expires rate-limit windows, and hands out the single
    /// HalfOpen probe permit lock-free.
    async fn admit(&self, v: &CircuitView<'_>, threshold: u32) -> Admit {
        let now = Instant::now();

        // 1. Dynamic Idle Penalty Decay (Aging)
        {
            let mut failures = v.failures.write().await;
            let mut last_attempt = v.last_attempt.write().await;
            if *failures > 0 {
                if let Some(last_time) = *last_attempt {
                    let elapsed = now.saturating_duration_since(last_time);
                    let decay_steps = elapsed.as_secs() / 300; // 5 minutes = 300 seconds
                    if decay_steps > 0 {
                        *failures = failures.saturating_sub(decay_steps as u32);
                        *last_attempt = Some(last_time + Duration::from_secs(decay_steps * 300));

                        // If failures age below the threshold, rehabilitate open circuits
                        if *failures < threshold {
                            let mut circ = v.circuit.write().await;
                            if let CircuitState::Open { .. } = *circ {
                                *circ = CircuitState::Closed;
                            }
                        }
                    }
                }
            }
        }

        // 2. Rate Limit Window Check
        {
            let mut rl = v.rate_limited_until.write().await;
            if let Some(until) = *rl {
                if now >= until {
                    *rl = None;
                } else {
                    return Admit::Skip;
                }
            }
        }

        // 3. Circuit Breaker Checks
        let current_state = *v.circuit.read().await;
        let claim_probe = || {
            // Lock-free: flip false -> true to claim the probe slot. If it
            // is already true another request is probing this scope, and we
            // pass through to protect it from a thundering herd.
            v.probe
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        };
        match current_state {
            CircuitState::Closed => Admit::Go,
            CircuitState::Open { until } => {
                if now >= until {
                    // Cooldown expired: transition circuit state to HalfOpen
                    *v.circuit.write().await = CircuitState::HalfOpen;
                    if claim_probe() {
                        return Admit::Probe;
                    }
                }
                Admit::Skip
            },
            CircuitState::HalfOpen => {
                if claim_probe() {
                    Admit::Probe
                } else {
                    Admit::Skip
                }
            },
        }
    }

    async fn record_success(&self, v: &CircuitView<'_>, is_probe: bool) {
        let now = Instant::now();
        *v.failures.write().await = 0;
        *v.circuit.write().await = CircuitState::Closed;
        *v.last_attempt.write().await = Some(now);
        *v.rate_limited_until.write().await = None;

        if is_probe {
            v.probe.store(false, Ordering::SeqCst);
            info!(
                target: "oxllm_core::router",
                "HalfOpen probe succeeded for {} — circuit closed",
                v.name
            );
        } else {
            info!(
                target: "oxllm_core::router",
                "Circuit closed for {} after successful request",
                v.name
            );
        }
    }

    async fn record_failure(
        &self,
        v: &CircuitView<'_>,
        limits: Limits,
        is_probe: bool,
        status_code: Option<u16>,
        retry_after: Option<Duration>,
    ) {
        let Limits {
            threshold,
            base_cooldown_secs,
        } = limits;
        let now = Instant::now();
        let mut failures = v.failures.write().await;
        *failures += 1;
        *v.last_attempt.write().await = Some(now);

        if status_code == Some(429) {
            // Rate-limited: extract or apply default backoff
            let cooldown = retry_after
                .unwrap_or_else(|| Duration::from_secs(self.circuit.rate_limit_fallback_secs));
            *v.rate_limited_until.write().await = Some(now + cooldown);
            info!(
                target: "oxllm_core::router",
                "Rate-limited {} for {}s",
                v.name, cooldown.as_secs()
            );
        } else if *failures >= threshold || is_probe {
            // Tripped or failed probe: trigger exponential circuit cooldown
            let exponent = failures.saturating_sub(threshold);
            let cooldown = Duration::from_secs(base_cooldown_secs * 2u64.pow(exponent));
            *v.circuit.write().await = CircuitState::Open {
                until: now + cooldown,
            };
            info!(
                target: "oxllm_core::router",
                "Circuit opened for {} for {}s ({} consecutive failures)",
                v.name, cooldown.as_secs(), *failures
            );
        }

        if is_probe {
            v.probe.store(false, Ordering::SeqCst);
        }
    }

    fn limits(&self, provider: &ProviderState) -> Limits {
        Limits {
            threshold: self.threshold(provider),
            base_cooldown_secs: provider
                .tuning
                .circuit_base_cooldown_secs
                .unwrap_or(self.circuit.base_cooldown_secs),
        }
    }

    /// Picks the first usable pool entry. An entry is usable when the
    /// provider is reachable and admitted AND the (provider, model) pair is
    /// admitted, so one misbehaving model does not hide its siblings.
    /// Returns the index into `entries` with the selection.
    pub async fn select_entry(
        &self,
        entries: &[(&ProviderState, &str)],
    ) -> Option<(usize, SelectedProvider)> {
        for (idx, (provider, model)) in entries.iter().enumerate() {
            if provider.manual_disabled.load(Ordering::Acquire) {
                continue;
            }
            let provider_admit = self
                .admit(
                    &CircuitView::of_provider(provider),
                    self.threshold(provider),
                )
                .await;
            if provider_admit == Admit::Skip {
                continue;
            }
            let health = provider.model_health.get(model);
            let model_view = CircuitView::of_model(model, &health);
            let model_admit = self.admit(&model_view, self.threshold(provider)).await;
            if model_admit == Admit::Skip {
                // Do not strand a probe permit we took on the provider.
                if provider_admit == Admit::Probe {
                    provider.probe_in_flight.store(false, Ordering::SeqCst);
                }
                continue;
            }
            return Some((
                idx,
                SelectedProvider {
                    name: provider.name.clone(),
                    base_url: provider.base_url.clone(),
                    api_key: provider.api_key.clone(),
                    is_probe: provider_admit == Admit::Probe,
                    model_probe: model_admit == Admit::Probe,
                    upstream_timeout_secs: provider.upstream_timeout_secs,
                    auth_header: provider.tuning.auth_header.clone(),
                },
            ));
        }
        None
    }

    /// Records the outcome of an attempt against one pool entry, scoping the
    /// failure to the provider or to the model as described on [`Outcome`].
    /// Probe permits held by the selection are always resolved or released.
    pub async fn feedback_entry(
        &self,
        provider: &ProviderState,
        model: &str,
        selected_probe: (bool, bool),
        outcome: Outcome,
        retry_after: Option<Duration>,
    ) {
        let (provider_probe, model_probe) = selected_probe;
        let health = provider.model_health.get(model);
        let pv = CircuitView::of_provider(provider);
        let mv = CircuitView::of_model(model, &health);
        let limits = self.limits(provider);
        let threshold = limits.threshold;
        let release = |view: &CircuitView<'_>, held: bool| {
            if held {
                view.probe.store(false, Ordering::SeqCst);
            }
        };

        match outcome {
            Outcome::Success => {
                self.record_success(&pv, provider_probe).await;
                self.record_success(&mv, model_probe).await;
            },
            Outcome::Unreachable => {
                self.record_failure(&pv, limits, provider_probe, None, None)
                    .await;
                release(&mv, model_probe);
            },
            Outcome::ModelError | Outcome::ModelTimeout | Outcome::RateLimited => {
                if outcome == Outcome::ModelTimeout {
                    let mut failures = mv.failures.write().await;
                    *failures = (*failures).max(threshold.saturating_sub(1));
                }
                let (status, retry) = if outcome == Outcome::RateLimited {
                    (Some(429), retry_after)
                } else {
                    (None, None)
                };
                self.record_failure(&mv, limits, model_probe, status, retry)
                    .await;
                // The host answered, so the provider itself is reachable.
                if provider_probe {
                    self.record_success(&pv, true).await;
                }
            },
        }
    }

    /// Releases the probe permits of a selection that never produced an
    /// outcome (client error passed through, bad URL), leaving circuits as
    /// they were so the next request probes again.
    pub fn release_probes(&self, provider: &ProviderState, model: &str, held: (bool, bool)) {
        if held.0 {
            provider.probe_in_flight.store(false, Ordering::SeqCst);
        }
        if held.1 {
            provider
                .model_health
                .get(model)
                .probe_in_flight
                .store(false, Ordering::SeqCst);
        }
    }
}

impl RoutingStrategy for AdaptivePriorityStrategy {
    async fn select(&self, candidates: &[&ProviderState]) -> Option<SelectedProvider> {
        for provider in candidates {
            // Check for manual admin override
            if provider.manual_disabled.load(Ordering::Acquire) {
                continue;
            }
            let admit = self
                .admit(
                    &CircuitView::of_provider(provider),
                    self.threshold(provider),
                )
                .await;
            if admit != Admit::Skip {
                return Some(SelectedProvider {
                    name: provider.name.clone(),
                    base_url: provider.base_url.clone(),
                    api_key: provider.api_key.clone(),
                    is_probe: admit == Admit::Probe,
                    model_probe: false,
                    upstream_timeout_secs: provider.upstream_timeout_secs,
                    auth_header: provider.tuning.auth_header.clone(),
                });
            }
        }
        None
    }

    async fn feedback(
        &self,
        provider: &ProviderState,
        success: bool,
        is_probe: bool,
        status_code: Option<u16>,
        retry_after: Option<Duration>,
    ) {
        let view = CircuitView::of_provider(provider);
        if success {
            self.record_success(&view, is_probe).await;
        } else {
            self.record_failure(
                &view,
                self.limits(provider),
                is_probe,
                status_code,
                retry_after,
            )
            .await;
        }
    }
}

/// A routing candidate with the upstream model name it maps to.
#[derive(Debug, Clone)]
pub struct LatencyCandidate {
    pub provider: String,
    pub model: String,
    /// Observed EWMA time-to-first-byte; `None` means never measured yet.
    pub ewma_ttft_ms: Option<f64>,
    /// Configured cost per 1M output tokens; `None` means free/unknown.
    pub cost_per_mtok: Option<f64>,
}

/// Latency band width. Providers whose EWMA sits within this ratio of each
/// other share a band and therefore keep configuration order between them.
/// 1.2 means "20% apart before we reorder", which is wider than the jitter of a
/// warm local upstream, so the order does not flap on noise.
pub const LATENCY_BAND_RATIO: f64 = 1.20;

/// Maps an EWMA TTFT onto a band index. Unmeasured candidates (band 0) sort
/// ahead of measured ones, so a newly added provider is sampled promptly and
/// then lands in its real band — this doubles as the exploration mechanism.
pub fn latency_band(ms: Option<f64>) -> u32 {
    match ms {
        None => 0,
        Some(v) if !v.is_finite() || v <= 0.0 => 1,
        Some(v) => 1 + (v.ln() / LATENCY_BAND_RATIO.ln()).floor().max(0.0) as u32,
    }
}

/// Orders candidate indices for cost-aware routing: cheapest configured cost
/// per 1M output tokens first, EWMA TTFT breaking ties within the same cost.
/// Providers without a configured cost are treated as free and sort first,
/// which keeps the policy useful in mixed local/remote pools where local is
/// both free and fast.
pub fn order_by_cost(candidates: &[LatencyCandidate], call_index: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|&a, &b| {
        let cost_a = candidates[a].cost_per_mtok.unwrap_or(0.0);
        let cost_b = candidates[b].cost_per_mtok.unwrap_or(0.0);
        cost_a
            .partial_cmp(&cost_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let ta = candidates[a].ewma_ttft_ms.unwrap_or(f64::MAX);
                let tb = candidates[b].ewma_ttft_ms.unwrap_or(f64::MAX);
                ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    // Same exploration contract as `fastest`: rotate every Nth call so a
    // costlier provider still gets sampled (its price may have dropped).
    if call_index % 20 == 0 && indices.len() > 1 {
        indices.rotate_left(1);
    }
    indices
}

/// Weighted rotation inside one latency band: spreads load across equal-speed
/// providers instead of always hitting the first. Deterministic from the call
/// index, so it needs no shared state beyond the counter `order_by_latency`
/// already consumes.
pub fn rotate_within_band(order: Vec<usize>, bands: &[u32], call_index: u64) -> Vec<usize> {
    if bands.is_empty() {
        return order;
    }
    // Group indices by band, preserving the band ordering of `order`.
    let mut result = Vec::with_capacity(order.len());
    let mut i = 0;
    while i < order.len() {
        let band = bands[order[i]];
        let mut group: Vec<usize> = Vec::new();
        while i < order.len() && bands[order[i]] == band {
            group.push(order[i]);
            i += 1;
        }
        // Rotate this equal-band group by the call index modulo group size.
        let len = group.len();
        if len > 1 {
            group.rotate_right((call_index % len as u64) as usize);
        }
        result.extend(group);
    }
    result
}

/// Orders candidate indices for latency-aware routing.
///
/// The sort is stable, so candidates in the same latency band keep their
/// configured order — a fast provider must be meaningfully faster (one whole
/// band, i.e. 20%) to displace one above it. Every `explore_every`-th call the
/// order is rotated, so a provider that was demoted by a temporary blip is
/// re-sampled even while a better one is winning.
pub fn order_by_latency(
    candidates: &[LatencyCandidate],
    call_index: u64,
    explore_every: u64,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by_key(|&i| latency_band(candidates[i].ewma_ttft_ms));
    if explore_every > 1 && call_index % explore_every == 0 && indices.len() > 1 {
        indices.rotate_left(1);
    }
    indices
}

/// Draws indices in weighted-lottery order without replacement: each draw
/// picks a remaining index with probability proportional to its weight.
/// Zero-weight entries are never drawn while any positive weight remains
/// and keep their original relative order at the tail. Used for
/// proportional load sharing among healthy candidates (per-target
/// `weight`, default policy only).
pub fn weighted_lottery(weights: &[u64], rng: &mut impl rand::Rng) -> Vec<usize> {
    let mut remaining: Vec<(usize, u64)> = weights.iter().copied().enumerate().collect();
    let mut total: u64 = remaining.iter().map(|(_, w)| w).sum();
    let mut order = Vec::with_capacity(weights.len());
    while !remaining.is_empty() {
        if total == 0 {
            // Only zero-weight entries are left: configured order.
            order.extend(remaining.iter().map(|(i, _)| *i));
            break;
        }
        let mut pick = rng.random_range(0..total);
        let mut chosen = remaining.len() - 1;
        for (pos, (_, w)) in remaining.iter().enumerate() {
            if pick < *w {
                chosen = pos;
                break;
            }
            pick -= *w;
        }
        let (index, weight) = remaining.remove(chosen);
        total -= weight;
        order.push(index);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use reqwest::Url;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn setup_provider(name: &str) -> ProviderState {
        ProviderState {
            name: name.to_string(),
            base_url: Url::parse("https://api.example.com").unwrap(),
            api_key: "test_key".to_string(),
            models: vec!["gpt-4".to_string()],
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
            model_stats: Arc::new(crate::state::ModelStats::default()),
            model_health: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_router_select_healthy_first() {
        let p1 = setup_provider("prov1");
        let p2 = setup_provider("prov2");
        let candidates = vec![&p1, &p2];

        let strategy = AdaptivePriorityStrategy::default();
        let selected = strategy.select(&candidates).await.unwrap();

        assert_eq!(selected.name, "prov1");
        assert!(!selected.is_probe);
    }

    #[tokio::test]
    async fn test_router_feedback_success() {
        let p = setup_provider("prov1");
        {
            let mut failures = p.consecutive_failures.write().await;
            *failures = 2;
            let mut circ = p.circuit.write().await;
            *circ = CircuitState::Open {
                until: Instant::now() + Duration::from_secs(60),
            };
        }

        let strategy = AdaptivePriorityStrategy::default();
        strategy.feedback(&p, true, false, Some(200), None).await;

        assert_eq!(*p.consecutive_failures.read().await, 0);
        assert_eq!(*p.circuit.read().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_router_feedback_failure_trips_circuit() {
        let p = setup_provider("prov1");
        let strategy = AdaptivePriorityStrategy::default();

        // First failure
        strategy.feedback(&p, false, false, Some(500), None).await;
        assert_eq!(*p.consecutive_failures.read().await, 1);
        assert_eq!(*p.circuit.read().await, CircuitState::Closed);

        // Second failure
        strategy.feedback(&p, false, false, Some(500), None).await;
        assert_eq!(*p.consecutive_failures.read().await, 2);
        assert_eq!(*p.circuit.read().await, CircuitState::Closed);

        // Third failure - should trip
        strategy.feedback(&p, false, false, Some(500), None).await;
        assert_eq!(*p.consecutive_failures.read().await, 3);
        match *p.circuit.read().await {
            CircuitState::Open { .. } => {},
            _ => panic!("Expected Open circuit state"),
        };
    }

    #[tokio::test]
    async fn test_router_feedback_rate_limited() {
        let p = setup_provider("prov1");
        let strategy = AdaptivePriorityStrategy::default();

        strategy
            .feedback(&p, false, false, Some(429), Some(Duration::from_secs(10)))
            .await;

        assert!(p.rate_limited_until.read().await.is_some());
    }

    #[tokio::test]
    async fn test_router_half_open_single_probe() {
        let p = setup_provider("prov1");
        {
            let mut failures = p.consecutive_failures.write().await;
            *failures = 3;
            // Cooldown expired
            let mut circ = p.circuit.write().await;
            *circ = CircuitState::Open {
                until: Instant::now() - Duration::from_secs(10),
            };
        }

        let strategy = AdaptivePriorityStrategy::default();
        let candidates = vec![&p];

        // First request should get a probe
        let selected1 = strategy.select(&candidates).await.unwrap();
        assert_eq!(selected1.name, "prov1");
        assert!(selected1.is_probe);
        assert!(p.probe_in_flight.load(Ordering::SeqCst));

        // Second concurrent request should be bypassed since probe is in flight
        let selected2 = strategy.select(&candidates).await;
        assert!(selected2.is_none());

        // Feedback success on probe resolves the circuit
        strategy.feedback(&p, true, true, Some(200), None).await;
        assert_eq!(*p.circuit.read().await, CircuitState::Closed);
        assert!(!p.probe_in_flight.load(Ordering::SeqCst));
    }

    #[test]
    fn test_latency_band_separates_by_twenty_percent() {
        // 100ms vs 110ms is inside the band -> same bucket, order preserved.
        assert_eq!(latency_band(Some(100.0)), latency_band(Some(110.0)));
        // 100ms vs 200ms is a different band.
        assert_ne!(latency_band(Some(100.0)), latency_band(Some(200.0)));
        // Unmeasured sorts ahead of any measured value.
        assert!(latency_band(None) < latency_band(Some(1.0)));
    }

    #[test]
    fn test_order_by_latency_prefers_faster_band_and_keeps_ties_stable() {
        let candidates = vec![
            LatencyCandidate {
                provider: "slow".into(),
                model: "m".into(),
                cost_per_mtok: None,
                ewma_ttft_ms: Some(900.0),
            },
            LatencyCandidate {
                provider: "fast".into(),
                model: "m".into(),
                cost_per_mtok: None,
                ewma_ttft_ms: Some(120.0),
            },
            LatencyCandidate {
                provider: "peers".into(),
                model: "m".into(),
                cost_per_mtok: None,
                ewma_ttft_ms: Some(130.0),
            },
        ];

        // call_index 1 avoids the exploration rotation at index 0.
        let order = order_by_latency(&candidates, 1, 20);
        let names: Vec<&str> = order
            .iter()
            .map(|&i| candidates[i].provider.as_str())
            .collect();
        assert_eq!(names, vec!["fast", "peers", "slow"]);
    }

    #[test]
    fn test_order_by_latency_explores_on_schedule() {
        let candidates = vec![
            LatencyCandidate {
                provider: "slow".into(),
                model: "m".into(),
                cost_per_mtok: None,
                ewma_ttft_ms: Some(900.0),
            },
            LatencyCandidate {
                provider: "fast".into(),
                model: "m".into(),
                cost_per_mtok: None,
                ewma_ttft_ms: Some(120.0),
            },
        ];

        let steady = order_by_latency(&candidates, 1, 20);
        assert_eq!(candidates[steady[0]].provider, "fast");

        // The scheduled exploration call rotates, so the demoted provider is
        // deliberately sampled instead of being starved forever.
        let exploring = order_by_latency(&candidates, 0, 20);
        assert_eq!(candidates[exploring[0]].provider, "slow");
        assert_ne!(steady, exploring, "exploration must rotate the order");
    }

    #[test]
    fn test_order_by_cost_prefers_cheap_then_fast() {
        let candidates = vec![
            LatencyCandidate {
                provider: "expensive-fast".into(),
                model: "m".into(),
                ewma_ttft_ms: Some(100.0),
                cost_per_mtok: Some(5.0),
            },
            LatencyCandidate {
                provider: "cheap-slow".into(),
                model: "m".into(),
                ewma_ttft_ms: Some(900.0),
                cost_per_mtok: Some(0.5),
            },
            LatencyCandidate {
                provider: "free".into(),
                model: "m".into(),
                ewma_ttft_ms: Some(300.0),
                cost_per_mtok: None,
            },
        ];

        let order = order_by_cost(&candidates, 1);
        let names: Vec<&str> = order
            .iter()
            .map(|&i| candidates[i].provider.as_str())
            .collect();
        // Free sorts first (None = free), then by cost, then latency.
        assert_eq!(names, vec!["free", "cheap-slow", "expensive-fast"]);
    }

    #[test]
    fn test_rotate_within_band_spreads_equal_speed_providers() {
        let candidates = vec![
            LatencyCandidate {
                provider: "a".into(),
                model: "m".into(),
                ewma_ttft_ms: Some(100.0),
                cost_per_mtok: None,
            },
            LatencyCandidate {
                provider: "b".into(),
                model: "m".into(),
                ewma_ttft_ms: Some(105.0),
                cost_per_mtok: None,
            },
            LatencyCandidate {
                provider: "c".into(),
                model: "m".into(),
                ewma_ttft_ms: Some(5000.0),
                cost_per_mtok: None,
            },
        ];
        let bands: Vec<u32> = candidates
            .iter()
            .map(|c| latency_band(c.ewma_ttft_ms))
            .collect();
        assert_eq!(bands[0], bands[1], "a and b share a band");
        assert_ne!(bands[2], bands[0], "c is in a slower band");

        // Baseline order from latency ordering.
        let base = order_by_latency(&candidates, 1, 20);
        assert_eq!(candidates[base[0]].provider, "a");

        // Rotating within bands must keep the slow provider last while
        // alternating between the two equal-speed providers.
        let r0 = rotate_within_band(base.clone(), &bands, 0);
        let r1 = rotate_within_band(base.clone(), &bands, 1);
        assert_eq!(candidates[*r0.last().unwrap()].provider, "c");
        assert_eq!(candidates[*r1.last().unwrap()].provider, "c");
        assert_ne!(r0, r1, "equal-band providers should rotate between calls");
    }

    #[tokio::test]
    async fn test_router_decay() {
        let p = setup_provider("prov1");
        {
            let mut failures = p.consecutive_failures.write().await;
            *failures = 4;
            let mut last_attempt = p.last_attempt_time.write().await;
            // 11 minutes ago
            *last_attempt = Some(Instant::now() - Duration::from_secs(660));
        }

        let strategy = AdaptivePriorityStrategy::default();
        let candidates = vec![&p];

        // Select triggers decay check
        let _ = strategy.select(&candidates).await;

        // Should decay by 2 failures (660s / 300s = 2)
        assert_eq!(*p.consecutive_failures.read().await, 2);
    }

    /// Weighted lottery: zero weights keep configured order; with a seeded
    /// RNG the draw is deterministic; zero-weight entries always sort after
    /// positive ones; and over many draws the distribution tracks the
    /// weights (issue #10).
    #[test]
    fn test_weighted_lottery_zero_weights_keep_order() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        assert_eq!(weighted_lottery(&[0, 0, 0], &mut rng), vec![0, 1, 2]);
        assert_eq!(weighted_lottery(&[], &mut rng), Vec::<usize>::new());
    }

    #[test]
    fn test_weighted_lottery_zero_weight_sorts_last() {
        // 100 draws: the zero-weight entry must never precede a weighted one.
        for seed in 0..100u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let order = weighted_lottery(&[3, 0, 1], &mut rng);
            assert_eq!(order.len(), 3);
            assert_ne!(order[0], 1, "zero weight drawn first: {:?}", order);
            assert!(
                order.iter().position(|&i| i == 1).unwrap() == 2,
                "zero weight not last: {:?}",
                order
            );
        }
    }

    #[test]
    fn test_weighted_lottery_distribution_tracks_weights() {
        // 3:1 weights over 2000 first-draws should land near 75%.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut firsts = [0usize; 2];
        for _ in 0..2000 {
            let order = weighted_lottery(&[3, 1], &mut rng);
            firsts[order[0]] += 1;
        }
        let ratio = firsts[0] as f64 / 2000.0;
        assert!(
            (0.70..=0.80).contains(&ratio),
            "expected ~0.75, got {}",
            ratio
        );
    }
    /// A configured threshold of 1 opens the circuit on the first failure
    /// (twitchy free-tier upstream), while the default still needs three.
    #[tokio::test]
    async fn test_circuit_threshold_is_configurable() {
        let twitchy = setup_provider("twitchy");
        let strategy = AdaptivePriorityStrategy {
            circuit: CircuitConfig {
                failure_threshold: 1,
                ..CircuitConfig::default()
            },
        };
        strategy
            .feedback(&twitchy, false, false, Some(500), None)
            .await;
        assert!(
            matches!(*twitchy.circuit.read().await, CircuitState::Open { .. }),
            "threshold=1 must open on the first failure"
        );

        let patient = setup_provider("patient");
        let default_strategy = AdaptivePriorityStrategy::default();
        default_strategy
            .feedback(&patient, false, false, Some(500), None)
            .await;
        assert_eq!(*patient.circuit.read().await, CircuitState::Closed);
    }

    /// The open-circuit cooldown scales from the configured base, not the
    /// hardcoded 60s, and a 429 without Retry-After uses the configured
    /// fallback instead of 30s.
    #[tokio::test]
    async fn test_circuit_cooldown_and_rate_limit_fallback_are_configurable() {
        let p = setup_provider("prov1");
        {
            let mut failures = p.consecutive_failures.write().await;
            *failures = 3;
        }
        let strategy = AdaptivePriorityStrategy {
            circuit: CircuitConfig {
                base_cooldown_secs: 10,
                rate_limit_fallback_secs: 5,
                ..CircuitConfig::default()
            },
        };
        strategy.feedback(&p, false, false, Some(500), None).await;
        let state = *p.circuit.read().await;
        match state {
            CircuitState::Open { until } => {
                let cooldown = until.saturating_duration_since(Instant::now());
                assert!(
                    cooldown >= Duration::from_secs(9),
                    "cooldown should start from the 10s base, got {:?}",
                    cooldown
                );
            },
            state => panic!("Expected Open circuit state, got {:?}", state),
        }

        let q = setup_provider("prov2");
        strategy.feedback(&q, false, false, Some(429), None).await;
        let rl = *q.rate_limited_until.read().await;
        match rl {
            Some(until) => {
                let backoff = until.saturating_duration_since(Instant::now());
                assert!(
                    backoff >= Duration::from_secs(4),
                    "no-Retry-After 429 should use the 5s fallback, got {:?}",
                    backoff
                );
            },
            None => panic!("expected a rate-limit window"),
        }
    }

    #[tokio::test]
    async fn test_per_provider_threshold_and_cooldown_override_server_values() {
        let mut p = setup_provider("flaky");
        p.tuning.circuit_failure_threshold = Some(1);
        p.tuning.circuit_base_cooldown_secs = Some(7);
        // Server-wide default is 3 failures / 60 s; the override wins.
        let strategy = AdaptivePriorityStrategy::default();
        strategy.feedback(&p, false, false, Some(500), None).await;
        let state = *p.circuit.read().await;
        match state {
            CircuitState::Open { until } => {
                let left = until.saturating_duration_since(Instant::now());
                assert!(left <= Duration::from_secs(7), "cooldown {:?}", left);
            },
            other => panic!("expected Open after one failure, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_timeout_feedback_respects_per_provider_threshold() {
        let mut p = setup_provider("slow");
        p.tuning.circuit_failure_threshold = Some(5);
        let strategy = AdaptivePriorityStrategy::default();
        strategy.feedback_timeout(&p, false).await;
        // A timeout still opens the circuit at once, however high the threshold.
        assert!(matches!(*p.circuit.read().await, CircuitState::Open { .. }));
    }
}
