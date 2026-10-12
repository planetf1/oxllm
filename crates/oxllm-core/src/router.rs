use crate::state::{CircuitState, ProviderState, SelectedProvider};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
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
pub struct AdaptivePriorityStrategy;

impl RoutingStrategy for AdaptivePriorityStrategy {
    async fn select(&self, candidates: &[&ProviderState]) -> Option<SelectedProvider> {
        let now = Instant::now();

        for provider in candidates {
            // Check for manual admin override
            if provider.manual_disabled.load(Ordering::Acquire) {
                continue;
            }

            // 1. Dynamic Idle Penalty Decay (Aging)
            {
                let mut failures = provider.consecutive_failures.write().await;
                let mut last_attempt = provider.last_attempt_time.write().await;
                if *failures > 0 {
                    if let Some(last_time) = *last_attempt {
                        let elapsed = now.saturating_duration_since(last_time);
                        let decay_steps = elapsed.as_secs() / 300; // 5 minutes = 300 seconds
                        if decay_steps > 0 {
                            *failures = failures.saturating_sub(decay_steps as u32);
                            *last_attempt =
                                Some(last_time + Duration::from_secs(decay_steps * 300));

                            // If failures age below 3, rehabilitate open circuits
                            if *failures < 3 {
                                let mut circ = provider.circuit.write().await;
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
                let mut rl = provider.rate_limited_until.write().await;
                if let Some(until) = *rl {
                    if now >= until {
                        *rl = None;
                    } else {
                        continue; // Still rate-limited, skip candidate
                    }
                }
            }

            // 3. Circuit Breaker Checks
            let current_state = *provider.circuit.read().await;
            match current_state {
                CircuitState::Closed => {
                    // Fully healthy: Route immediately
                    return Some(SelectedProvider {
                        name: provider.name.clone(),
                        base_url: provider.base_url.clone(),
                        api_key: provider.api_key.clone(),
                        headers: provider.headers.clone(),
                        is_probe: false,
                    });
                },
                CircuitState::Open { until } => {
                    if now >= until {
                        // Cooldown expired: transition circuit state to HalfOpen
                        {
                            let mut state_guard = provider.circuit.write().await;
                            *state_guard = CircuitState::HalfOpen;
                        }

                        // Attempt to acquire the single flight probe slot lock-free
                        if provider
                            .probe_in_flight
                            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                        {
                            return Some(SelectedProvider {
                                name: provider.name.clone(),
                                base_url: provider.base_url.clone(),
                                api_key: provider.api_key.clone(),
                                headers: provider.headers.clone(),
                                is_probe: true,
                            });
                        }
                    }
                    // Still cooling down or another thread stole the HalfOpen probe permit
                    continue;
                },
                CircuitState::HalfOpen => {
                    // Lock-free check: If false, we flip it to true and claim the probe slot.
                    // If true, another concurrent thread is already probing this upstream.
                    // We instantly pass through to protect the upstream from a thundering herd!
                    if provider
                        .probe_in_flight
                        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                    {
                        return Some(SelectedProvider {
                            name: provider.name.clone(),
                            base_url: provider.base_url.clone(),
                            api_key: provider.api_key.clone(),
                            headers: provider.headers.clone(),
                            is_probe: true,
                        });
                    }
                    continue;
                },
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
        let now = Instant::now();

        if success {
            let mut failures = provider.consecutive_failures.write().await;
            *failures = 0;

            let mut state = provider.circuit.write().await;
            *state = CircuitState::Closed;

            let mut last_attempt = provider.last_attempt_time.write().await;
            *last_attempt = Some(now);

            let mut rl = provider.rate_limited_until.write().await;
            *rl = None;

            if is_probe {
                provider.probe_in_flight.store(false, Ordering::SeqCst);
                info!(
                    target: "oxllm_core::router",
                    "HalfOpen probe succeeded for {} — circuit closed",
                    provider.name
                );
            } else {
                info!(
                    target: "oxllm_core::router",
                    "Circuit closed for {} after successful request",
                    provider.name
                );
            }
        } else {
            let mut failures = provider.consecutive_failures.write().await;
            *failures += 1;

            let mut last_attempt = provider.last_attempt_time.write().await;
            *last_attempt = Some(now);

            // Handle Failures
            if status_code == Some(429) {
                // Rate-limited: extract or apply default backoff
                let cooldown = retry_after.unwrap_or_else(|| Duration::from_secs(30)); // 30s default fallback
                let mut rl = provider.rate_limited_until.write().await;
                *rl = Some(now + cooldown);
                info!(
                    target: "oxllm_core::router",
                    "Rate-limited {} for {}s",
                    provider.name, cooldown.as_secs()
                );
            } else if *failures >= 3 || is_probe {
                // Tripped or failed probe: trigger exponential circuit cooldown
                let exponent = failures.saturating_sub(3) as u32;
                let multiplier = 2u64.pow(exponent);
                let cooldown = Duration::from_secs(60 * multiplier);

                let mut state = provider.circuit.write().await;
                *state = CircuitState::Open {
                    until: now + cooldown,
                };
                info!(
                    target: "oxllm_core::router",
                    "Circuit opened for {} for {}s ({} consecutive failures)",
                    provider.name, cooldown.as_secs(), *failures
                );
            }

            if is_probe {
                provider.probe_in_flight.store(false, Ordering::SeqCst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            headers: std::collections::HashMap::new(),
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
        }
    }

    #[tokio::test]
    async fn test_router_select_healthy_first() {
        let p1 = setup_provider("prov1");
        let p2 = setup_provider("prov2");
        let candidates = vec![&p1, &p2];

        let strategy = AdaptivePriorityStrategy;
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

        let strategy = AdaptivePriorityStrategy;
        strategy.feedback(&p, true, false, Some(200), None).await;

        assert_eq!(*p.consecutive_failures.read().await, 0);
        assert_eq!(*p.circuit.read().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_router_feedback_failure_trips_circuit() {
        let p = setup_provider("prov1");
        let strategy = AdaptivePriorityStrategy;

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
        let strategy = AdaptivePriorityStrategy;

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

        let strategy = AdaptivePriorityStrategy;
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

        let strategy = AdaptivePriorityStrategy;
        let candidates = vec![&p];

        // Select triggers decay check
        let _ = strategy.select(&candidates).await;

        // Should decay by 2 failures (660s / 300s = 2)
        assert_eq!(*p.consecutive_failures.read().await, 2);
    }
}
