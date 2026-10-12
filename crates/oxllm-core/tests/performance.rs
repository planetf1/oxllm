use oxllm_core::router::{AdaptivePriorityStrategy, RoutingStrategy};
use oxllm_core::state::{CircuitState, ProviderState};
use reqwest::Url;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[tokio::test]
async fn test_routing_loop_latency_performance() {
    // 1. Create a typical 5-provider fallback pool
    let mut candidates = Vec::new();
    for i in 0..5 {
        candidates.push(ProviderState {
            name: format!("provider-{}", i),
            base_url: Url::parse("https://api.example.com").unwrap(),
            api_key: "key".to_string(),
            models: vec!["model-a".to_string()],
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
        });
    }

    let candidate_refs: Vec<&ProviderState> = candidates.iter().collect();
    let strategy = AdaptivePriorityStrategy;

    // 2. Warm up the CPU caches
    for _ in 0..100 {
        let _ = strategy.select(&candidate_refs).await;
    }

    // 3. Measure 10,000 select operations
    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let selected = strategy.select(&candidate_refs).await;
        assert!(selected.is_some());
    }
    let elapsed = start.elapsed();
    let avg_latency = elapsed / iterations;

    println!(
        "\n[BENCHMARK] Total elapsed for {} selects: {:?}",
        iterations, elapsed
    );
    println!("[BENCHMARK] Average latency per select: {:?}", avg_latency);

    // 4. Assert routing loop latency is strictly under 2ms (2,000,000 ns)
    assert!(
        avg_latency < Duration::from_millis(2),
        "Routing latency {:?} exceeded 2ms budget!",
        avg_latency
    );
}
