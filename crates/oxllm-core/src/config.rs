use crate::error::{OxllmError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub otel_endpoint: String,
    #[serde(default = "default_upstream_timeout")]
    pub upstream_timeout_secs: u64,
    #[serde(default = "default_bind_family")]
    pub bind_family: String,
    /// Inbound API keys. When non-empty, requests to the public /v1/* routes
    /// must present `Authorization: Bearer <key>` matching one of these;
    /// admin routes remain localhost-only regardless. Keys are compared in
    /// constant time. An empty list (default) keeps the proxy open.
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Header inspected for the routing session key (default
    /// "x-session-id"; the request body's `prompt_cache_key` field is the
    /// fallback). Sessions stick to one provider while it stays healthy, so
    /// upstream prefix caches stay warm. Set to "" to disable header-based
    /// affinity.
    #[serde(default = "default_session_affinity_header")]
    pub session_affinity_header: String,
    /// How long a session pin survives without traffic (seconds).
    #[serde(default = "default_session_affinity_ttl_secs")]
    pub session_affinity_ttl_secs: u64,
    /// Maximum number of concurrent session pins.
    #[serde(default = "default_session_affinity_capacity")]
    pub session_affinity_capacity: usize,
    /// Consecutive failures before a provider's circuit opens.
    #[serde(default = "default_circuit_failure_threshold")]
    pub circuit_failure_threshold: u32,
    /// Base cooldown (seconds) when a circuit opens; doubles for each
    /// failure beyond the threshold.
    #[serde(default = "default_circuit_base_cooldown_secs")]
    pub circuit_base_cooldown_secs: u64,
    /// Rate-limit cooldown (seconds) applied when a 429 carries no
    /// `Retry-After` header.
    #[serde(default = "default_rate_limit_fallback_secs")]
    pub rate_limit_fallback_secs: u64,
    /// Deadline (seconds) for graceful shutdown after SIGTERM/SIGINT:
    /// in-flight requests are drained until it expires, then the process
    /// exits anyway. `0` waits indefinitely.
    #[serde(default = "default_shutdown_timeout_secs")]
    pub shutdown_timeout_secs: u64,
    /// Whether the daemon watches the config file and hot-reloads on
    /// change. Disable on flash-storage edge targets or to make reloads
    /// strictly manual (SIGHUP / POST /reload).
    #[serde(default = "default_watch_config")]
    pub watch_config: bool,
    /// Virtual model that serves requests naming an unknown model, instead
    /// of returning 400. `None` (default) keeps unknown models an error.
    #[serde(default)]
    pub default_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub enabled: bool,
    pub base_url: String,
    pub api_key: String,
    pub models: Vec<String>,
    /// Optional per-provider timeout override (seconds) for connecting and
    /// receiving upstream headers. Falls back to `server.upstream_timeout_secs`.
    #[serde(default)]
    pub upstream_timeout_secs: Option<u64>,
    /// Optional relative cost weight per 1M output tokens, used by the
    /// `cheapest` routing policy. Absent means "no cost signal" and such
    /// providers are treated as free.
    #[serde(default)]
    pub cost_per_mtok: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualModelTarget {
    pub provider: String,
    pub model: String,
    /// Routing preference within one virtual model: lower wins. Targets
    /// without a priority keep their configured order, after all targets
    /// that have one. Absent everywhere, this reproduces the historical
    /// strict array-order semantics.
    #[serde(default)]
    pub priority: Option<u32>,
}

/// Routing policy for one virtual model. Omitted virtual models keep strict
/// configuration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PolicyConfig {
    /// Strict configuration order (default): deterministic, and the circuit
    /// breaker still skips unhealthy providers.
    #[default]
    Priority,
    /// Lowest measured EWMA time-to-first-byte first, with hysteresis so a
    /// near-tie does not flap and periodic exploration so a demoted provider
    /// can win its place back.
    Fastest,
    /// Lowest cost per 1M output tokens first; latency breaks ties. Providers
    /// without a configured cost are treated as free.
    Cheapest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub providers: Vec<ProviderConfig>,
    pub virtual_models: HashMap<String, Vec<VirtualModelTarget>>,
    /// Optional per-virtual-model routing policy, e.g.
    /// ```toml
    /// [policies]
    /// my-model = "fastest"
    /// ```
    #[serde(default)]
    pub policies: HashMap<String, PolicyConfig>,
    /// Client-facing model name aliases, e.g.
    /// ```toml
    /// [aliases]
    /// gpt-4o = "router-small"
    /// ```
    /// Lets unmodified OpenAI clients address virtual models by the names
    /// they were built with. Targets must be defined virtual models.
    #[serde(default)]
    pub aliases: HashMap<String, String>,
}

impl Config {
    /// Loads a TOML configuration file and expands environment variables of style `${VAR_NAME}`.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| OxllmError::ConfigLoad(format!("Failed to read config file: {}", e)))?;

        let expanded = expand_env_vars(&content)?;
        let config: Config = toml::from_str(&expanded)?;
        Ok(config)
    }

    /// Validates the configuration syntax and cross-references virtual models with defined providers.
    pub fn validate(&self) -> Result<()> {
        let provider_map: HashMap<&str, &ProviderConfig> = self
            .providers
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();

        // 1. Validate that at least one provider is configured
        if self.providers.is_empty() {
            return Err(OxllmError::ConfigLoad(
                "At least one provider must be defined".into(),
            ));
        }

        // 2. Validate that virtual models target existing, enabled providers
        for (vm_name, targets) in &self.virtual_models {
            if targets.is_empty() {
                return Err(OxllmError::ConfigLoad(format!(
                    "Virtual model '{}' has no targets configured",
                    vm_name
                )));
            }

            for target in targets {
                match provider_map.get(target.provider.as_str()) {
                    Some(provider) => {
                        if !provider.enabled {
                            // Warn or skip: we allow referencing disabled providers,
                            // but the virtual model resolution loop will bypass them.
                        }
                    },
                    None => {
                        return Err(OxllmError::ConfigLoad(format!(
                            "Virtual model '{}' targets undefined provider '{}'",
                            vm_name, target.provider
                        )));
                    },
                }
            }
        }

        // 3. Aliases must point at defined virtual models
        for (alias, target) in &self.aliases {
            if !self.virtual_models.contains_key(target) {
                return Err(OxllmError::ConfigLoad(format!(
                    "Alias '{}' targets undefined virtual model '{}'",
                    alias, target
                )));
            }
        }

        // 4. The default model, when set, must be a defined virtual model
        if let Some(default) = &self.server.default_model {
            if !self.virtual_models.contains_key(default) {
                return Err(OxllmError::ConfigLoad(format!(
                    "server.default_model '{}' is not a defined virtual model",
                    default
                )));
            }
        }

        // 5. Circuit tuning sanity
        if self.server.circuit_failure_threshold == 0 {
            return Err(OxllmError::ConfigLoad(
                "server.circuit_failure_threshold must be at least 1".into(),
            ));
        }

        Ok(())
    }
}

fn default_session_affinity_header() -> String {
    "x-session-id".to_string()
}

fn default_session_affinity_ttl_secs() -> u64 {
    900
}

fn default_session_affinity_capacity() -> usize {
    1024
}

fn default_circuit_failure_threshold() -> u32 {
    3
}

fn default_circuit_base_cooldown_secs() -> u64 {
    60
}

fn default_rate_limit_fallback_secs() -> u64 {
    30
}

fn default_shutdown_timeout_secs() -> u64 {
    30
}

fn default_watch_config() -> bool {
    true
}

fn default_upstream_timeout() -> u64 {
    5
}

fn default_bind_family() -> String {
    "ipv4".to_string()
}

/// Helper function to perform Unix shell-style `${VAR_NAME}` environment variable expansions.
#[allow(clippy::while_let_on_iterator)]
pub fn expand_env_vars(raw_content: &str) -> Result<String> {
    let mut expanded = String::new();
    let mut chars = raw_content.char_indices().peekable();

    while let Some((idx, ch)) = chars.next() {
        if ch == '$' {
            if let Some(&(_, '{')) = chars.peek() {
                chars.next(); // Consume '{'

                let mut var_name = String::new();
                let mut found_close = false;

                while let Some((_, var_ch)) = chars.next() {
                    if var_ch == '}' {
                        found_close = true;
                        break;
                    }
                    var_name.push(var_ch);
                }

                if !found_close {
                    return Err(OxllmError::ConfigLoad(format!(
                        "Unclosed environment variable placeholder starting at index {}",
                        idx
                    )));
                }

                // Strictly resolve environment variable
                let val = std::env::var(&var_name)
                    .map_err(|_| OxllmError::EnvVarMissing(var_name.clone()))?;

                expanded.push_str(&val);
                continue;
            }
        }
        expanded.push(ch);
    }

    Ok(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_env_vars_success() {
        std::env::set_var("TEST_HOST", "127.0.0.1");
        std::env::set_var("TEST_KEY", "groq-key-123");

        let input = r#"
            host = "${TEST_HOST}"
            api_key = "${TEST_KEY}"
            other = "literal $10 string"
        "#;

        let result = expand_env_vars(input).unwrap();
        assert!(result.contains("host = \"127.0.0.1\""));
        assert!(result.contains("api_key = \"groq-key-123\""));
        assert!(result.contains("other = \"literal $10 string\""));
    }

    #[test]
    fn test_expand_env_vars_missing() {
        std::env::remove_var("MISSING_VAR_XYZ");
        let input = r#"api_key = "${MISSING_VAR_XYZ}""#;
        let result = expand_env_vars(input);
        assert!(
            matches!(result, Err(OxllmError::EnvVarMissing(ref name)) if name == "MISSING_VAR_XYZ")
        );
    }

    #[test]
    fn test_expand_env_vars_unclosed() {
        let input = r#"api_key = "${UNCLOSED"#;
        let result = expand_env_vars(input);
        assert!(matches!(result, Err(OxllmError::ConfigLoad(_))));
    }

    fn minimal_config() -> Config {
        toml::from_str(
            r#"
[server]
host = "127.0.0.1"
port = 8080
otel_endpoint = "http://127.0.0.1:4318"

[[providers]]
name = "p"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "k"
models = ["m"]

[virtual_models]
vm = [{ provider = "p", model = "m" }]
"#,
        )
        .unwrap()
    }

    #[test]
    fn test_validate_rejects_dangling_alias() {
        let mut config = minimal_config();
        config
            .aliases
            .insert("gpt-4o".to_string(), "no-such-vm".to_string());
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("gpt-4o"), "got: {}", err);
        assert!(err.to_string().contains("no-such-vm"), "got: {}", err);
    }

    #[test]
    fn test_validate_rejects_unknown_default_model() {
        let mut config = minimal_config();
        config.server.default_model = Some("no-such-vm".to_string());
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("default_model"), "got: {}", err);
    }

    #[test]
    fn test_validate_rejects_zero_circuit_threshold() {
        let mut config = minimal_config();
        config.server.circuit_failure_threshold = 0;
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("circuit_failure_threshold"),
            "got: {}",
            err
        );
    }

    #[test]
    fn test_validate_accepts_alias_and_default_model() {
        let mut config = minimal_config();
        config
            .aliases
            .insert("gpt-4o".to_string(), "vm".to_string());
        config.server.default_model = Some("vm".to_string());
        config.validate().unwrap();
    }

    #[test]
    fn test_new_server_fields_default_to_current_behaviour() {
        let config = minimal_config();
        let server = config.server;
        assert_eq!(server.shutdown_timeout_secs, 30);
        assert!(server.watch_config);
        assert_eq!(server.circuit_failure_threshold, 3);
        assert_eq!(server.circuit_base_cooldown_secs, 60);
        assert_eq!(server.rate_limit_fallback_secs, 30);
        assert_eq!(server.default_model, None);
    }
}
