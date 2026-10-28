use crate::config::VirtualModelTarget;
use reqwest::Url;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open { until: Instant },
    HalfOpen,
}

#[derive(Debug)]
pub struct ProviderState {
    pub name: String,
    pub base_url: Url, // Parsed reqwest::Url to handle safe path joins and trailing slashes
    pub api_key: String,
    pub models: Vec<String>,

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

#[derive(Clone)]
pub struct SelectedProvider {
    pub name: String,
    pub base_url: Url,
    pub api_key: String,
    pub is_probe: bool,
}

pub struct AppState {
    pub providers: Vec<ProviderState>,
    pub virtual_models: HashMap<String, Vec<VirtualModelTarget>>,
    /// Per-virtual-model counters (one entry per route, independent of how
    /// many providers back it).
    pub route_stats: Arc<ModelStats>,
    /// Routing policy per virtual model; absent means strict priority order.
    pub policies: HashMap<String, RoutingPolicy>,
    /// Monotonic counter driving periodic exploration under `Fastest`.
    pub routing_calls: AtomicU64,
    pub http_client: reqwest::Client,
    pub upstream_timeout_secs: u64,
}

/// How a virtual model picks among its healthy candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingPolicy {
    /// Strict configuration order (circuit breaker still applies).
    #[default]
    Priority,
    /// Lowest measured EWMA TTFT first, with hysteresis and exploration.
    Fastest,
}

impl AppState {
    /// Virtual model names known from configuration, sorted for stable output.
    pub fn route_stats_keys(&self) -> Vec<String> {
        let mut names: Vec<String> = self.virtual_models.keys().cloned().collect();
        names.sort();
        names
    }

    /// Resolves the candidate list for a given virtual model.
    pub fn resolve_candidates(&self, virtual_model: &str) -> Vec<(&ProviderState, String)> {
        let targets = match self.virtual_models.get(virtual_model) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut candidates = Vec::new();
        for target in targets {
            if let Some(provider) = self.providers.iter().find(|p| p.name == target.provider) {
                candidates.push((provider, target.model.clone()));
            }
        }
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
