use crate::config::VirtualModelTarget;
use reqwest::Url;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64};
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
    pub headers: HashMap<String, String>,

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
}

#[derive(Clone)]
pub struct SelectedProvider {
    pub name: String,
    pub base_url: Url,
    pub api_key: String,
    pub headers: HashMap<String, String>,
    pub is_probe: bool,
}

pub struct AppState {
    pub providers: Vec<ProviderState>,
    pub virtual_models: HashMap<String, Vec<VirtualModelTarget>>,
    pub http_client: reqwest::Client,
    pub upstream_timeout_secs: u64,
}

impl AppState {
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
