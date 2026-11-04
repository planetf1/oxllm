use crate::config::{ApiFormat, ServerConfig, VirtualModelTarget};
use reqwest::Url;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Wall-clock milliseconds since the Unix epoch.
pub fn unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open { until: Instant },
    HalfOpen,
}

/// One completed (or terminally failed) client request, for the recent
/// traffic log and live tail. Metadata only — request and response bodies
/// are deliberately never captured.
#[derive(Debug, Clone, Serialize)]
pub struct RequestRecord {
    /// Unix epoch milliseconds.
    pub ts_ms: u64,
    /// Endpoint family: "chat", "completions", "embeddings", "systemone".
    pub endpoint: String,
    /// Virtual model the request was routed through.
    pub route: String,
    /// Upstream model on the provider that served (or last failed) it.
    pub model: String,
    /// Provider that served the request, or the last one that failed.
    pub provider: String,
    /// Response status code; 0 when no response was produced.
    pub status: u16,
    /// Provider attempts used, including the successful/last one.
    pub attempts: u32,
    pub latency_ms: u64,
    /// Time to first byte, when measured (non-streaming: full latency).
    pub ttft_ms: Option<u64>,
    pub streamed: bool,
    /// Terminal failure reason, when the request failed.
    pub error: Option<String>,
    pub request_id: String,
}

/// Bounded ring of recent request records plus a live-tail broadcast.
/// Recording is a short Mutex critical section over tiny, bounded data;
/// the broadcast costs nothing when nobody is subscribed.
#[derive(Debug)]
pub struct RequestLog {
    capacity: usize,
    inner: std::sync::Mutex<VecDeque<RequestRecord>>,
    tail: tokio::sync::broadcast::Sender<RequestRecord>,
}

impl Default for RequestLog {
    fn default() -> Self {
        Self::new(default_request_log_capacity())
    }
}

fn default_request_log_capacity() -> usize {
    256
}

impl RequestLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            inner: std::sync::Mutex::new(VecDeque::with_capacity(capacity.min(1024))),
            tail: tokio::sync::broadcast::channel(capacity.max(16)).0,
        }
    }

    /// A log with recording disabled (`server.request_log_capacity = 0`).
    pub fn disabled() -> Self {
        Self::new(0)
    }

    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    /// Appends a record to the ring and broadcasts it to live tails.
    pub fn record(&self, rec: RequestRecord) {
        if self.capacity == 0 {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.len() == self.capacity {
            inner.pop_front();
        }
        inner.push_back(rec.clone());
        // Lagged or absent receivers are fine: the ring is the source of truth.
        let _ = self.tail.send(rec);
    }

    /// The most recent `limit` records, oldest first.
    pub fn recent(&self, limit: usize) -> Vec<RequestRecord> {
        if self.capacity == 0 || limit == 0 {
            return Vec::new();
        }
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let skip = inner.len().saturating_sub(limit);
        inner.iter().skip(skip).cloned().collect()
    }

    /// Live tail subscription; `None` when recording is disabled.
    pub fn subscribe(&self) -> Option<tokio::sync::broadcast::Receiver<RequestRecord>> {
        (self.capacity > 0).then(|| self.tail.subscribe())
    }
}

/// Outcome of the last config reload attempt, for /status: operators need
/// to see whether an edit applied, was rejected, or still needs a restart.
#[derive(Debug)]
pub struct ReloadHealth {
    inner: std::sync::Mutex<ReloadHealthSnapshot>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct ReloadHealthSnapshot {
    /// Unix epoch milliseconds of the last reload attempt.
    pub last_attempt_ms: Option<u64>,
    /// Unix epoch milliseconds of the last successful reload.
    pub last_success_ms: Option<u64>,
    /// Rejection reason of the last failed reload, if any.
    pub last_error: Option<String>,
    /// Restart-required warnings carried by the last successful reload.
    pub restart_warnings: Vec<String>,
}

impl Default for ReloadHealth {
    fn default() -> Self {
        Self {
            inner: std::sync::Mutex::new(ReloadHealthSnapshot::default()),
        }
    }
}

impl ReloadHealth {
    pub fn record_success(&self, restart_warnings: Vec<String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let now = unix_millis();
        inner.last_attempt_ms = Some(now);
        inner.last_success_ms = Some(now);
        inner.last_error = None;
        inner.restart_warnings = restart_warnings;
    }

    pub fn record_failure(&self, reason: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.last_attempt_ms = Some(unix_millis());
        inner.last_error = Some(reason.to_string());
    }

    pub fn snapshot(&self) -> ReloadHealthSnapshot {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Per-provider overrides of the server-wide circuit and timeout settings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderTuning {
    pub circuit_failure_threshold: Option<u32>,
    pub circuit_base_cooldown_secs: Option<u64>,
    pub first_byte_timeout_secs: Option<u64>,
}

#[derive(Debug)]
pub struct ProviderState {
    pub name: String,
    pub base_url: Url, // Parsed reqwest::Url to handle safe path joins and trailing slashes
    pub api_key: String,
    pub models: Vec<String>,
    /// Per-provider timeout override (seconds); `None` = server default.
    pub upstream_timeout_secs: Option<u64>,
    /// Cost per 1M output tokens; `None` = free / unknown.
    pub cost_per_mtok: Option<f64>,
    /// Wire format this provider speaks.
    pub api: ApiFormat,
    /// Per-provider circuit/timeout overrides.
    pub tuning: ProviderTuning,

    // Protects volatile metrics without needing a write lock on the entire pool vector
    pub circuit: Arc<RwLock<CircuitState>>,
    pub consecutive_failures: Arc<RwLock<u32>>,
    pub rate_limited_until: Arc<RwLock<Option<Instant>>>,
    pub last_attempt_time: Arc<RwLock<Option<Instant>>>,

    // Lock-free thundering-herd permit
    pub probe_in_flight: Arc<AtomicBool>,

    // Manual admin override — skips provider in routing regardless of circuit state
    pub manual_disabled: AtomicBool,

    // Local request/token counters (visible via /status without otel collector)
    pub requests: AtomicU64,
    pub successes: AtomicU64,
    pub tokens_input: AtomicU64,
    pub tokens_output: AtomicU64,

    // Per-(provider, upstream-model) counters, so /status can attribute
    // traffic correctly when one provider serves several virtual models.
    // Keyed by "provider\u{0}model" to avoid ambiguity when either side can
    // contain the separator character.
    pub model_stats: Arc<ModelStats>,
}

/// Bounded LRU of session-key -> provider pins, with TTL.
///
/// Stickiness exists to keep a conversation on one provider so the upstream
/// prefix cache stays warm (measured at ~6x on prompt processing). The map is
/// capped so a hostile or high-cardinality client cannot grow it without
/// bound; on overflow the least-recently-used pin is dropped and that session
/// simply re-routes by policy.
///
/// Interior mutability: std Mutex over a small VecDeque plus a timestamp map.
/// Held only for O(len) operations on request arrival (len <= cap), never
/// across await points.
#[derive(Debug)]
pub struct SessionAffinity {
    inner: std::sync::Mutex<AffinityInner>,
    ttl: std::time::Duration,
    capacity: usize,
}

#[derive(Debug, Default)]
struct AffinityInner {
    /// (session_key, provider, last_seen) in most-recent-first order.
    entries: std::collections::VecDeque<(String, String, std::time::Instant)>,
}

impl SessionAffinity {
    pub fn new(ttl: std::time::Duration, capacity: usize) -> Self {
        Self {
            inner: std::sync::Mutex::new(AffinityInner::default()),
            ttl,
            capacity: capacity.max(1),
        }
    }

    /// The provider a session is pinned to, if the pin exists and is fresh.
    /// Refreshes the pin's recency on hit.
    pub fn pinned_provider(&self, session_key: &str) -> Option<String> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let now = std::time::Instant::now();
        let idx = inner
            .entries
            .iter()
            .position(|(k, _, seen)| k == session_key && now.duration_since(*seen) < self.ttl)?;
        let (key, provider, _) = inner.entries.remove(idx).unwrap();
        inner.entries.push_front((key, provider.clone(), now));
        Some(provider)
    }

    /// Pins (or re-pins) a session to a provider, evicting stale and
    /// least-recently-used entries to stay within capacity.
    pub fn pin(&self, session_key: &str, provider: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let now = std::time::Instant::now();
        inner
            .entries
            .retain(|(_, _, seen)| now.duration_since(*seen) < self.ttl);
        inner.entries.retain(|(k, _, _)| k != session_key);
        inner
            .entries
            .push_front((session_key.to_string(), provider.to_string(), now));
        while inner.entries.len() > self.capacity {
            inner.entries.pop_back();
        }
    }

    /// Drops a pin (e.g. after the pinned provider failed terminally).
    pub fn unpin(&self, session_key: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.entries.retain(|(k, _, _)| k != session_key);
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.entries.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Default header inspected for a session key, before the body's
/// `prompt_cache_key` field.
pub const DEFAULT_SESSION_HEADER: &str = "x-session-id";

/// Upper bounds (in milliseconds) of the latency histogram buckets.
/// Fibonacci-spaced: ~1.6x resolution, dense where proxy latencies actually
/// live (sub-second) and still bounded for pathological cases.
pub const BUCKET_BOUNDS_MS: [u64; 21] = [
    1,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55,
    89,
    144,
    233,
    377,
    610,
    987,
    1597,
    2584,
    4181,
    6765,
    10946,
    u64::MAX,
];

/// Fixed-bucket latency histogram. Every record is a handful of relaxed atomic
/// adds and a bucket lookup, so it is safe on the request hot path; percentiles
/// are computed only when /status is scraped.
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; BUCKET_BOUNDS_MS.len()],
    count: AtomicU64,
    sum: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
        }
    }
}

impl Histogram {
    pub fn record(&self, ms: u64) {
        let idx = BUCKET_BOUNDS_MS
            .iter()
            .position(|bound| ms <= *bound)
            .unwrap_or(BUCKET_BOUNDS_MS.len() - 1);
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(ms, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn mean(&self) -> Option<f64> {
        let count = self.count();
        (count > 0).then(|| self.sum.load(Ordering::Relaxed) as f64 / count as f64)
    }

    /// Nearest-rank percentile, reported as the bucket's upper bound.
    pub fn percentile(&self, p: f64) -> Option<u64> {
        let total = self.count();
        if total == 0 {
            return None;
        }
        let target = ((p / 100.0) * total as f64).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (idx, bucket) in self.buckets.iter().enumerate() {
            seen += bucket.load(Ordering::Relaxed);
            if seen >= target {
                return Some(BUCKET_BOUNDS_MS[idx]);
            }
        }
        Some(*BUCKET_BOUNDS_MS.last().unwrap_or(&u64::MAX))
    }
}

/// Everything tracked per routing target (provider+model, or virtual model).
#[derive(Debug, Default)]
pub struct Counters {
    pub requests: AtomicU64,
    pub successes: AtomicU64,
    pub failures: AtomicU64,
    pub tokens_input: AtomicU64,
    pub tokens_output: AtomicU64,
    /// Total request latency (request received -> body fully forwarded).
    pub latency: Histogram,
    /// Time to first byte from the upstream, i.e. generation start for
    /// streaming and full response for non-streaming.
    pub ttft: Histogram,
    /// Fixed-point (microseconds) EWMA of TTFT, used by latency-aware routing.
    pub ewma_ttft_us: AtomicU64,
    /// Prompt tokens served from the upstream prefix cache (from usage
    /// `cached_tokens`), for cache hit-rate reporting.
    pub cached_tokens: AtomicU64,
}

impl Counters {
    const EWMA_ALPHA_NUM: u64 = 1;
    const EWMA_ALPHA_DEN: u64 = 5;

    fn ewma_update(slot: &AtomicU64, sample_us: u64) {
        let prev = slot.load(Ordering::Relaxed);
        let next = if prev == 0 {
            sample_us
        } else {
            (prev * (Self::EWMA_ALPHA_DEN - Self::EWMA_ALPHA_NUM)
                + sample_us * Self::EWMA_ALPHA_NUM)
                / Self::EWMA_ALPHA_DEN
        };
        slot.store(next, Ordering::Relaxed);
    }

    pub fn record_ttft(&self, us: u64) {
        self.ttft.record(us / 1000);
        Self::ewma_update(&self.ewma_ttft_us, us);
    }

    pub fn ewma_ttft_ms(&self) -> Option<f64> {
        let v = self.ewma_ttft_us.load(Ordering::Relaxed);
        (v > 0).then(|| v as f64 / 1000.0)
    }
}

/// A snapshot of one target's counters, for /status.
#[derive(Debug, Default, Clone, Copy)]
pub struct CountersSnapshot {
    pub requests: u64,
    pub successes: u64,
    pub failures: u64,
    pub tokens_input: u64,
    pub tokens_output: u64,
    pub latency_p50_ms: Option<u64>,
    pub latency_p95_ms: Option<u64>,
    pub latency_p99_ms: Option<u64>,
    pub latency_mean_ms: Option<f64>,
    pub ttft_p50_ms: Option<u64>,
    pub ttft_p95_ms: Option<u64>,
    pub cached_tokens: u64,
}

/// Shared counter table keyed by an opaque string. Reads take a read lock and
/// clone an `Arc`, so the hot path never allocates after the first request for
/// a key.
#[derive(Debug, Default)]
pub struct ModelStats {
    inner: std::sync::RwLock<HashMap<String, Arc<Counters>>>,
}

impl ModelStats {
    pub fn key(provider: &str, model: &str) -> String {
        format!("{}\u{0}{}", provider, model)
    }

    pub fn counters(&self, key: &str) -> Arc<Counters> {
        if let Some(existing) = self
            .inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
        {
            return existing;
        }
        let mut map = self.inner.write().unwrap_or_else(|e| e.into_inner());
        map.entry(key.to_string()).or_default().clone()
    }

    pub fn snapshot(&self, key: &str) -> CountersSnapshot {
        let Some(counters) = self
            .inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
        else {
            return CountersSnapshot::default();
        };
        CountersSnapshot {
            requests: counters.requests.load(Ordering::Relaxed),
            successes: counters.successes.load(Ordering::Relaxed),
            failures: counters.failures.load(Ordering::Relaxed),
            tokens_input: counters.tokens_input.load(Ordering::Relaxed),
            tokens_output: counters.tokens_output.load(Ordering::Relaxed),
            latency_p50_ms: counters.latency.percentile(50.0),
            latency_p95_ms: counters.latency.percentile(95.0),
            latency_p99_ms: counters.latency.percentile(99.0),
            latency_mean_ms: counters.latency.mean(),
            ttft_p50_ms: counters.ttft.percentile(50.0),
            ttft_p95_ms: counters.ttft.percentile(95.0),
            cached_tokens: counters.cached_tokens.load(Ordering::Relaxed),
        }
    }

    /// EWMA TTFT in milliseconds for a routing target, if any sample exists.
    pub fn ewma_ttft_ms(&self, provider: &str, model: &str) -> Option<f64> {
        let key = Self::key(provider, model);
        let counters = self
            .inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned()?;
        counters.ewma_ttft_ms()
    }
}

/// Circuit-breaker tuning, resolved from `[server]` config. Defaults
/// preserve the historical behaviour: trip after 3 consecutive failures,
/// 60s base cooldown doubling per failure beyond the threshold, 30s
/// rate-limit cooldown when a 429 carries no Retry-After.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircuitConfig {
    /// Consecutive failures before the circuit opens.
    pub failure_threshold: u32,
    /// Base cooldown (seconds) when the circuit opens.
    pub base_cooldown_secs: u64,
    /// Rate-limit cooldown (seconds) without a Retry-After header.
    pub rate_limit_fallback_secs: u64,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            base_cooldown_secs: 60,
            rate_limit_fallback_secs: 30,
        }
    }
}

#[derive(Clone)]
pub struct SelectedProvider {
    pub name: String,
    pub base_url: Url,
    pub api_key: String,
    pub is_probe: bool,
    pub upstream_timeout_secs: Option<u64>,
}

pub struct AppState {
    pub providers: Vec<ProviderState>,
    pub virtual_models: HashMap<String, Vec<VirtualModelTarget>>,
    /// Client-facing alias -> virtual model (validated at config time).
    pub aliases: HashMap<String, String>,
    /// Virtual model serving requests that name an unknown model.
    pub default_model: Option<String>,
    /// Circuit-breaker tuning applied by the routing strategy.
    pub circuit: CircuitConfig,
    /// Per-virtual-model counters (one entry per route, independent of how
    /// many providers back it).
    pub route_stats: Arc<ModelStats>,
    /// Routing policy per virtual model; absent means strict priority order.
    pub policies: HashMap<String, RoutingPolicy>,
    /// Monotonic counter driving periodic exploration under `Fastest`.
    pub routing_calls: AtomicU64,
    /// Inbound API keys; empty means the public /v1 routes are open.
    pub api_keys: Vec<String>,
    /// Session -> provider pins for sticky routing.
    pub session_affinity: Arc<SessionAffinity>,
    /// Header consulted for the session key (body `prompt_cache_key` is the
    /// fallback). Empty string disables header-based affinity.
    pub session_affinity_header: String,
    pub http_client: reqwest::Client,
    pub upstream_timeout_secs: u64,
    /// The server section this state was built from. Hot-reload diffs
    /// against it to warn when a change cannot apply without a restart.
    pub server: ServerConfig,
    /// Outcome of the last reload, surfaced via /status.
    pub reload_health: Arc<ReloadHealth>,
    /// Bounded ring of recent requests plus the live-tail broadcast.
    pub request_log: Arc<RequestLog>,
}

/// How a virtual model picks among its healthy candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingPolicy {
    /// Strict configuration order (circuit breaker still applies).
    #[default]
    Priority,
    /// Lowest measured EWMA TTFT first, with hysteresis and exploration.
    Fastest,
    /// Lowest configured cost per 1M output tokens first; latency breaks ties.
    Cheapest,
}

/// One routable candidate for a virtual model: the provider, the upstream
/// model name, and the target's tuning fields.
#[derive(Clone)]
pub struct RouteCandidate<'a> {
    pub provider: &'a ProviderState,
    pub model: String,
    /// Explicit priority (lower = preferred); `None` sorts after all
    /// prioritized targets.
    pub priority: Option<u32>,
    /// Proportional-sharing weight within the candidate's priority tier.
    pub weight: Option<u32>,
}

impl AppState {
    /// Virtual model names known from configuration, sorted for stable output.
    pub fn route_stats_keys(&self) -> Vec<String> {
        let mut names: Vec<String> = self.virtual_models.keys().cloned().collect();
        names.sort();
        names
    }

    /// Maps a client-requested model name onto the virtual model that
    /// serves it: exact match first, then `[aliases]`, then
    /// `server.default_model` for any unknown name. `None` means the
    /// request must fail with 400.
    pub fn resolve_route<'a>(&'a self, requested: &'a str) -> Option<&'a str> {
        if self.virtual_models.contains_key(requested) {
            return Some(requested);
        }
        if let Some(vm) = self.aliases.get(requested) {
            return Some(vm);
        }
        match &self.default_model {
            // Guarded at config-validation time; the filter keeps a stale
            // default from turning into a panic if state diverges.
            Some(default) if self.virtual_models.contains_key(default) => Some(default),
            _ => None,
        }
    }

    /// Resolves the candidate list for a given virtual model, ordered by
    /// the targets' explicit `priority` (lower first). Targets without a
    /// priority keep their configured order and sort after all prioritized
    /// ones; with no priorities set anywhere this is exactly the
    /// configured array order. Weights ride along for the routing policies.
    pub fn resolve_candidates(&self, virtual_model: &str) -> Vec<RouteCandidate<'_>> {
        self.resolve_candidates_for(virtual_model, None)
    }

    /// Like `resolve_candidates`, but when `api` is set keeps only providers
    /// speaking that wire format, so one virtual model can mix OpenAI and
    /// Anthropic providers without either route seeing the other's.
    pub fn resolve_candidates_for(
        &self,
        virtual_model: &str,
        api: Option<ApiFormat>,
    ) -> Vec<RouteCandidate<'_>> {
        let targets = match self.virtual_models.get(virtual_model) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut candidates: Vec<RouteCandidate> = Vec::new();
        for target in targets {
            if let Some(provider) = self.providers.iter().find(|p| p.name == target.provider) {
                if api.is_some_and(|a| a != provider.api) {
                    continue;
                }
                candidates.push(RouteCandidate {
                    provider,
                    model: target.model.clone(),
                    priority: target.priority,
                    weight: target.weight,
                });
            }
        }
        // Stable sort: equal or absent priorities keep configuration order.
        candidates.sort_by_key(|c| c.priority.unwrap_or(u32::MAX));
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_percentiles_and_mean() {
        let h = Histogram::default();
        assert_eq!(h.count(), 0);
        assert_eq!(h.percentile(50.0), None);

        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
            h.record(ms);
        }
        assert_eq!(h.count(), 10);
        assert_eq!(h.mean(), Some(55.0));
        // Nearest-rank p50 lands in a bucket whose upper bound is >= 50.
        let p50 = h.percentile(50.0).expect("p50");
        assert!((50..=55).contains(&p50), "p50 was {p50}");
        let p99 = h.percentile(99.0).expect("p99");
        assert!(p99 >= 100, "p99 was {p99}");
    }

    #[test]
    fn test_histogram_bucket_boundaries() {
        let h = Histogram::default();
        h.record(0);
        assert_eq!(h.percentile(100.0), Some(1)); // <=1ms bucket
        h.record(u64::MAX);
        assert_eq!(h.percentile(100.0), Some(u64::MAX)); // overflow bucket
    }

    #[test]
    fn test_model_stats_separates_models_on_one_provider() {
        let stats = ModelStats::default();
        let fast = stats.counters(&ModelStats::key("omlx", "small-model"));
        fast.requests.fetch_add(3, Ordering::Relaxed);
        fast.tokens_input.fetch_add(30, Ordering::Relaxed);

        let other = stats.counters(&ModelStats::key("omlx", "big-model"));
        other.requests.fetch_add(1, Ordering::Relaxed);

        let small = stats.snapshot(&ModelStats::key("omlx", "small-model"));
        let big = stats.snapshot(&ModelStats::key("omlx", "big-model"));
        assert_eq!(small.requests, 3);
        assert_eq!(small.tokens_input, 30);
        assert_eq!(big.requests, 1);
        assert_eq!(big.tokens_input, 0, "models must not share counters");
    }

    #[test]
    fn test_ewma_ttft_moves_towards_samples() {
        let stats = ModelStats::default();
        let c = stats.counters(&ModelStats::key("p", "m"));
        c.record_ttft(100_000); // 100ms
        assert_eq!(c.ewma_ttft_ms(), Some(100.0));
        for _ in 0..20 {
            c.record_ttft(200_000); // 200ms
        }
        let ewma = c.ewma_ttft_ms().expect("ewma");
        assert!(ewma > 190.0 && ewma <= 200.0, "ewma was {ewma}");
    }
}

#[cfg(test)]
mod affinity_tests {
    use super::*;

    #[test]
    fn test_pin_and_lookup() {
        let aff = SessionAffinity::new(std::time::Duration::from_secs(60), 8);
        assert_eq!(aff.pinned_provider("s1"), None);
        aff.pin("s1", "prov-a");
        assert_eq!(aff.pinned_provider("s1"), Some("prov-a".to_string()));
        // Re-pin moves the provider.
        aff.pin("s1", "prov-b");
        assert_eq!(aff.pinned_provider("s1"), Some("prov-b".to_string()));
    }

    #[test]
    fn test_capacity_evicts_least_recently_used() {
        let aff = SessionAffinity::new(std::time::Duration::from_secs(60), 2);
        aff.pin("s1", "a");
        aff.pin("s2", "b");
        // Touch s1 so s2 is the LRU entry.
        aff.pinned_provider("s1");
        aff.pin("s3", "c");
        assert_eq!(aff.pinned_provider("s1"), Some("a".to_string()));
        assert_eq!(aff.pinned_provider("s2"), None, "LRU pin must be evicted");
        assert_eq!(aff.pinned_provider("s3"), Some("c".to_string()));
        assert_eq!(aff.len(), 2);
    }

    #[test]
    fn test_ttl_expiry() {
        let aff = SessionAffinity::new(std::time::Duration::from_millis(20), 8);
        aff.pin("s1", "a");
        std::thread::sleep(std::time::Duration::from_millis(40));
        assert_eq!(
            aff.pinned_provider("s1"),
            None,
            "expired pin must not match"
        );
    }

    #[test]
    fn test_unpin() {
        let aff = SessionAffinity::new(std::time::Duration::from_secs(60), 8);
        aff.pin("s1", "a");
        aff.unpin("s1");
        assert_eq!(aff.pinned_provider("s1"), None);
    }

    fn provider_with_api(name: &str, api: ApiFormat) -> ProviderState {
        ProviderState {
            name: name.to_string(),
            base_url: Url::parse("http://127.0.0.1:9/v1/").unwrap(),
            api_key: "k".to_string(),
            models: vec!["m".to_string()],
            upstream_timeout_secs: None,
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
        }
    }

    #[test]
    fn test_resolve_candidates_for_filters_by_wire_format() {
        let target = |p: &str| VirtualModelTarget {
            provider: p.to_string(),
            model: "m".to_string(),
            priority: None,
            weight: None,
        };
        let mut virtual_models = HashMap::new();
        virtual_models.insert(
            "vm".to_string(),
            vec![target("o"), target("a1"), target("a2")],
        );
        let state = AppState {
            providers: vec![
                provider_with_api("o", ApiFormat::Openai),
                provider_with_api("a1", ApiFormat::Anthropic),
                provider_with_api("a2", ApiFormat::Anthropic),
            ],
            virtual_models,
            aliases: HashMap::new(),
            default_model: None,
            circuit: CircuitConfig::default(),
            route_stats: Arc::new(ModelStats::default()),
            policies: HashMap::new(),
            routing_calls: AtomicU64::new(0),
            api_keys: Vec::new(),
            session_affinity: Arc::new(SessionAffinity::new(
                std::time::Duration::from_secs(900),
                16,
            )),
            session_affinity_header: DEFAULT_SESSION_HEADER.to_string(),
            http_client: reqwest::Client::new(),
            upstream_timeout_secs: 5,
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 0,
                otel_endpoint: String::new(),
                upstream_timeout_secs: 5,
                bind_family: "ipv4".to_string(),
                api_keys: Vec::new(),
                session_affinity_header: DEFAULT_SESSION_HEADER.to_string(),
                session_affinity_ttl_secs: 900,
                session_affinity_capacity: 16,
                circuit_failure_threshold: 3,
                circuit_base_cooldown_secs: 60,
                rate_limit_fallback_secs: 30,
                shutdown_timeout_secs: 30,
                watch_config: false,
                request_log_capacity: 0,
                default_model: None,
            },
            reload_health: Default::default(),
            request_log: Default::default(),
        };
        let names = |api: Option<ApiFormat>| -> Vec<String> {
            state
                .resolve_candidates_for("vm", api)
                .iter()
                .map(|c| c.provider.name.clone())
                .collect()
        };
        assert_eq!(names(None), ["o", "a1", "a2"]);
        assert_eq!(names(Some(ApiFormat::Openai)), ["o"]);
        assert_eq!(names(Some(ApiFormat::Anthropic)), ["a1", "a2"]);
        // The unfiltered entry point is unchanged.
        assert_eq!(state.resolve_candidates("vm").len(), 3);
    }
}
