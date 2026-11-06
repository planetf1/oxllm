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
    /// How long a Responses API id stays pinned to the provider that
    /// created it (seconds). Default 30 days, OpenAI's default retention.
    #[serde(default = "default_response_affinity_ttl_secs")]
    pub response_affinity_ttl_secs: u64,
    /// Maximum number of Responses API ids tracked (LRU beyond that).
    #[serde(default = "default_response_affinity_capacity")]
    pub response_affinity_capacity: usize,
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
    /// How many recent request records the daemon keeps in memory for
    /// /status and `oxllm tail` (metadata only, ~200 bytes each). `0`
    /// disables recording. Applies at startup.
    #[serde(default = "default_request_log_capacity")]
    pub request_log_capacity: usize,
    /// Virtual model that serves requests naming an unknown model, instead
    /// of returning 400. `None` (default) keeps unknown models an error.
    #[serde(default)]
    pub default_model: Option<String>,
}

/// Wire format a provider speaks. Requests only route to providers whose
/// format matches the inbound endpoint: the OpenAI routes use `openai`,
/// `/v1/messages` uses `anthropic`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ApiFormat {
    /// OpenAI-style (`Authorization: Bearer`, `chat/completions`, ...).
    #[default]
    Openai,
    /// Anthropic Messages (`x-api-key`, `anthropic-version`, `messages`).
    Anthropic,
}

impl ApiFormat {
    /// The config-file spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            ApiFormat::Openai => "openai",
            ApiFormat::Anthropic => "anthropic",
        }
    }
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
    /// Wire format the upstream speaks (`"openai"` default, or
    /// `"anthropic"`). Selects the upstream path and auth headers, and which
    /// inbound routes may use this provider.
    #[serde(default)]
    pub api: ApiFormat,
    /// Per-provider override of `server.circuit_failure_threshold`.
    #[serde(default)]
    pub circuit_failure_threshold: Option<u32>,
    /// Per-provider override of `server.circuit_base_cooldown_secs`.
    #[serde(default)]
    pub circuit_base_cooldown_secs: Option<u64>,
    /// Streaming requests only: give up on this provider if response
    /// headers have not arrived within this many seconds, and fail over.
    /// Catches a hung endpoint far sooner than the total
    /// `upstream_timeout_secs`. Not applied to non-streaming requests,
    /// whose headers arrive only after the whole generation.
    #[serde(default)]
    pub first_byte_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualModelTarget {
    /// Provider for a leaf target. Empty when `virtual` is set.
    #[serde(default)]
    pub provider: String,
    /// Upstream model name for a leaf target. Empty when `virtual` is set.
    #[serde(default)]
    pub model: String,
    /// Instead of `provider`/`model`: the name of another virtual model
    /// whose targets are spliced in here, in their own order. Resolved once
    /// at state-build time; cycles are rejected by validation. A `priority`
    /// on a reference applies to the whole group. `weight` is not allowed
    /// on a reference.
    #[serde(rename = "virtual", default)]
    pub virtual_ref: Option<String>,
    /// Routing preference within one virtual model: lower wins. Targets
    /// without a priority keep their configured order, after all targets
    /// that have one. Absent everywhere, this reproduces the historical
    /// strict array-order semantics.
    #[serde(default)]
    pub priority: Option<u32>,
    /// Proportional load sharing within one priority tier under the default
    /// policy: per request, candidates carrying a weight are drawn as a
    /// lottery with probability proportional to it. Ignored by the
    /// `fastest` and `cheapest` policies.
    #[serde(default)]
    pub weight: Option<u32>,
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
                if let Some(reference) = &target.virtual_ref {
                    if !target.provider.is_empty() || !target.model.is_empty() {
                        return Err(OxllmError::ConfigLoad(format!(
                            "Virtual model '{}' target references virtual model '{}' \
                             and also sets provider/model; use one or the other",
                            vm_name, reference
                        )));
                    }
                    if target.weight.is_some() {
                        return Err(OxllmError::ConfigLoad(format!(
                            "Virtual model '{}' target references virtual model '{}' \
                             with a weight; weights apply to provider targets, not groups",
                            vm_name, reference
                        )));
                    }
                    continue;
                }
                if target.provider.is_empty() || target.model.is_empty() {
                    return Err(OxllmError::ConfigLoad(format!(
                        "Virtual model '{}' has a target without provider and model \
                         (or `virtual = \"<name>\"`)",
                        vm_name
                    )));
                }
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

        // 2b. Virtual-model references: defined, acyclic, bounded depth
        self.check_virtual_model_references()?;

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

        // 5b. Per-provider tuning sanity
        for p in &self.providers {
            if p.circuit_failure_threshold == Some(0) {
                return Err(OxllmError::ConfigLoad(format!(
                    "Provider '{}': circuit_failure_threshold must be at least 1",
                    p.name
                )));
            }
            if p.first_byte_timeout_secs == Some(0) {
                return Err(OxllmError::ConfigLoad(format!(
                    "Provider '{}': first_byte_timeout_secs must be at least 1",
                    p.name
                )));
            }
        }

        // 6. Weights must be positive
        for (vm_name, targets) in &self.virtual_models {
            for target in targets {
                if target.weight == Some(0) {
                    return Err(OxllmError::ConfigLoad(format!(
                        "Virtual model '{}' target for provider '{}' has weight 0; \
                         weights must be at least 1",
                        vm_name, target.provider
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Maximum nesting of virtual-model references (a -> b -> c is depth 3).
pub const MAX_VIRTUAL_MODEL_DEPTH: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    InProgress,
    Done,
}

impl Config {
    /// Rejects undefined references, cycles (naming the full path, e.g.
    /// `a -> b -> c -> a`) and nesting deeper than
    /// [`MAX_VIRTUAL_MODEL_DEPTH`]. Depth-first search with three-colour
    /// marking; the depth guard fires before recursion can get deep, so a
    /// hostile chain cannot overflow the stack. Roots are visited in sorted
    /// order so the reported error is deterministic.
    pub fn check_virtual_model_references(&self) -> Result<()> {
        let mut marks: HashMap<&str, Mark> = HashMap::new();
        let mut roots: Vec<&str> = self.virtual_models.keys().map(String::as_str).collect();
        roots.sort_unstable();
        for root in roots {
            let mut path: Vec<&str> = Vec::new();
            self.visit_virtual_model(root, &mut marks, &mut path)?;
        }
        Ok(())
    }

    fn visit_virtual_model<'a>(
        &'a self,
        name: &'a str,
        marks: &mut HashMap<&'a str, Mark>,
        path: &mut Vec<&'a str>,
    ) -> Result<()> {
        match marks.get(name) {
            Some(Mark::Done) => return Ok(()),
            Some(Mark::InProgress) => {
                let start = path.iter().position(|n| *n == name).unwrap_or(0);
                let mut cycle: Vec<&str> = path[start..].to_vec();
                cycle.push(name);
                return Err(OxllmError::ConfigLoad(format!(
                    "Virtual model references form a cycle: {}",
                    cycle.join(" -> ")
                )));
            },
            None => {},
        }
        if path.len() >= MAX_VIRTUAL_MODEL_DEPTH {
            let mut chain: Vec<&str> = path.clone();
            chain.push(name);
            return Err(OxllmError::ConfigLoad(format!(
                "Virtual model references nest deeper than {} levels: {}",
                MAX_VIRTUAL_MODEL_DEPTH,
                chain.join(" -> ")
            )));
        }
        marks.insert(name, Mark::InProgress);
        path.push(name);
        for target in self.virtual_models.get(name).into_iter().flatten() {
            if let Some(reference) = target.virtual_ref.as_deref() {
                if !self.virtual_models.contains_key(reference) {
                    return Err(OxllmError::ConfigLoad(format!(
                        "Virtual model '{}' references undefined virtual model '{}'",
                        name, reference
                    )));
                }
                self.visit_virtual_model(reference, marks, path)?;
            }
        }
        path.pop();
        marks.insert(name, Mark::Done);
        Ok(())
    }

    /// Resolves every virtual model to a flat list of leaf (provider, model)
    /// targets. Runs the reference check itself, so it is safe to call
    /// without `validate()`. Order is preserved: a referenced group's own
    /// order stays contiguous, and the reference's `priority` applies to the
    /// whole group (weights inside a nested group are dropped, since a
    /// per-leaf lottery would break the group's order). Models containing
    /// references are de-duplicated on (provider, model), keeping the entry
    /// that sorts first; models without references are returned untouched.
    pub fn flatten_virtual_models(&self) -> Result<HashMap<String, Vec<VirtualModelTarget>>> {
        self.check_virtual_model_references()?;
        let mut memo: HashMap<String, Vec<VirtualModelTarget>> = HashMap::new();
        let names: Vec<String> = self.virtual_models.keys().cloned().collect();
        for name in &names {
            self.flatten_one(name, &mut memo);
        }
        Ok(memo)
    }

    /// Callers guarantee acyclicity and bounded depth via
    /// `check_virtual_model_references`, so this recursion terminates.
    fn flatten_one(
        &self,
        name: &str,
        memo: &mut HashMap<String, Vec<VirtualModelTarget>>,
    ) -> Vec<VirtualModelTarget> {
        if let Some(done) = memo.get(name) {
            return done.clone();
        }
        let targets = self.virtual_models.get(name).cloned().unwrap_or_default();
        let has_reference = targets.iter().any(|t| t.virtual_ref.is_some());
        let mut out: Vec<VirtualModelTarget> = Vec::new();
        for target in targets {
            match target.virtual_ref.clone() {
                None => out.push(target),
                Some(reference) => {
                    let mut group = self.flatten_one(&reference, memo);
                    // Settle the group's internal order before it takes the
                    // reference's single priority, so it stays contiguous.
                    group.sort_by_key(|t| t.priority.unwrap_or(u32::MAX));
                    for mut leaf in group {
                        leaf.priority = target.priority;
                        leaf.weight = None;
                        out.push(leaf);
                    }
                },
            }
        }
        if has_reference {
            out.sort_by_key(|t| t.priority.unwrap_or(u32::MAX));
            let mut seen: std::collections::HashSet<(String, String)> =
                std::collections::HashSet::new();
            out.retain(|t| seen.insert((t.provider.clone(), t.model.clone())));
        }
        memo.insert(name.to_string(), out.clone());
        out
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

fn default_response_affinity_ttl_secs() -> u64 {
    30 * 24 * 3600
}

fn default_response_affinity_capacity() -> usize {
    4096
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

fn default_request_log_capacity() -> usize {
    256
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
    fn test_validate_rejects_zero_weight() {
        let mut config = minimal_config();
        config.virtual_models.get_mut("vm").unwrap()[0].weight = Some(0);
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("weight"), "got: {}", err);
    }

    #[test]
    fn test_validate_accepts_positive_weight() {
        let mut config = minimal_config();
        config.virtual_models.get_mut("vm").unwrap()[0].weight = Some(3);
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

    #[test]
    fn test_provider_api_defaults_to_openai() {
        let config = minimal_config();
        assert_eq!(config.providers[0].api, ApiFormat::Openai);
        config.validate().unwrap();
    }

    #[test]
    fn test_provider_api_anthropic_parses() {
        let config: Config = toml::from_str(
            r#"
[server]
host = "127.0.0.1"
port = 8080
otel_endpoint = "http://127.0.0.1:4318"

[[providers]]
name = "claude"
enabled = true
base_url = "http://127.0.0.1:9/v1/"
api_key = "k"
models = ["m"]
api = "anthropic"

[virtual_models]
vm = [{ provider = "claude", model = "m" }]
"#,
        )
        .unwrap();
        assert_eq!(config.providers[0].api, ApiFormat::Anthropic);
        assert_eq!(config.providers[0].api.as_str(), "anthropic");
        config.validate().unwrap();
    }

    #[test]
    fn test_provider_api_rejects_unknown_value() {
        let err = toml::from_str::<Config>(
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
api = "gemini"

[virtual_models]
vm = [{ provider = "p", model = "m" }]
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("gemini"), "got: {}", err);
    }

    #[test]
    fn test_provider_tuning_parses_and_defaults_to_none() {
        let config = minimal_config();
        let p = &config.providers[0];
        assert_eq!(p.circuit_failure_threshold, None);
        assert_eq!(p.circuit_base_cooldown_secs, None);
        assert_eq!(p.first_byte_timeout_secs, None);

        let tuned: Config = toml::from_str(
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
circuit_failure_threshold = 1
circuit_base_cooldown_secs = 15
first_byte_timeout_secs = 20

[virtual_models]
vm = [{ provider = "p", model = "m" }]
"#,
        )
        .unwrap();
        let p = &tuned.providers[0];
        assert_eq!(p.circuit_failure_threshold, Some(1));
        assert_eq!(p.circuit_base_cooldown_secs, Some(15));
        assert_eq!(p.first_byte_timeout_secs, Some(20));
        tuned.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_zero_provider_tuning() {
        let mut config = minimal_config();
        config.providers[0].circuit_failure_threshold = Some(0);
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("circuit_failure_threshold"));

        let mut config = minimal_config();
        config.providers[0].first_byte_timeout_secs = Some(0);
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("first_byte_timeout_secs"));
    }

    // ---- nested virtual models ------------------------------------------

    /// Builds a config with providers `p1..p4` and the given virtual models,
    /// written as TOML array bodies (`{ virtual = "b" }`, `{ provider = ... }`).
    fn nested_config(vms: &[(&str, &str)]) -> Config {
        let mut toml = String::from(
            r#"
[server]
host = "127.0.0.1"
port = 8080
otel_endpoint = "http://127.0.0.1:4318"
"#,
        );
        for p in ["p1", "p2", "p3", "p4"] {
            toml.push_str(&format!(
                "\n[[providers]]\nname = \"{p}\"\nenabled = true\nbase_url = \"http://127.0.0.1:9/v1/\"\napi_key = \"k\"\nmodels = [\"m\"]\n"
            ));
        }
        toml.push_str("\n[virtual_models]\n");
        for (name, body) in vms {
            toml.push_str(&format!("{name} = [{body}]\n"));
        }
        toml::from_str(&toml).unwrap()
    }

    fn leaf(p: &str) -> String {
        format!("{{ provider = \"{p}\", model = \"m\" }}")
    }
    fn refer(v: &str) -> String {
        format!("{{ virtual = \"{v}\" }}")
    }
    fn flat_names(config: &Config, vm: &str) -> Vec<String> {
        config.flatten_virtual_models().unwrap()[vm]
            .iter()
            .map(|t| t.provider.clone())
            .collect()
    }

    #[test]
    fn test_nested_rejects_self_reference() {
        let c = nested_config(&[("a", &refer("a"))]);
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("a -> a"), "got: {err}");
    }

    #[test]
    fn test_nested_rejects_two_cycle_naming_the_path() {
        let c = nested_config(&[("a", &refer("b")), ("b", &refer("a"))]);
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("a -> b -> a"), "got: {err}");
    }

    #[test]
    fn test_nested_rejects_long_cycle_naming_the_full_path() {
        let c = nested_config(&[("a", &refer("b")), ("b", &refer("c")), ("c", &refer("a"))]);
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("a -> b -> c -> a"), "got: {err}");
    }

    #[test]
    fn test_nested_cycle_path_excludes_the_entry_chain() {
        // root -> a -> b -> a: the cycle is a -> b -> a, not root -> ...
        let c = nested_config(&[
            ("root", &refer("a")),
            ("a", &refer("b")),
            ("b", &format!("{}, {}", leaf("p1"), refer("a"))),
        ]);
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("a -> b -> a"), "got: {err}");
        assert!(!err.contains("root"), "got: {err}");
    }

    #[test]
    fn test_nested_accepts_diamond() {
        let c = nested_config(&[
            ("a", &format!("{}, {}", refer("b"), refer("c"))),
            ("b", &refer("d")),
            ("c", &refer("d")),
            ("d", &leaf("p1")),
        ]);
        c.validate().unwrap();
        // The shared leaf appears once.
        assert_eq!(flat_names(&c, "a"), ["p1"]);
    }

    #[test]
    fn test_nested_rejects_undefined_reference() {
        let c = nested_config(&[("a", &refer("ghost"))]);
        let err = c.validate().unwrap_err().to_string();
        assert!(
            err.contains("undefined virtual model 'ghost'"),
            "got: {err}"
        );
    }

    #[test]
    fn test_nested_rejects_excess_depth_without_overflow() {
        // A chain longer than the limit: must error, not recurse unboundedly.
        let n = MAX_VIRTUAL_MODEL_DEPTH + 5;
        let mut vms: Vec<(String, String)> = (0..n)
            .map(|i| (format!("v{i}"), refer(&format!("v{}", i + 1))))
            .collect();
        vms.push((format!("v{n}"), leaf("p1")));
        let refs: Vec<(&str, &str)> = vms.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let err = nested_config(&refs).validate().unwrap_err().to_string();
        assert!(err.contains("nest deeper than"), "got: {err}");
        // And a chain exactly at the limit is accepted.
        let n = MAX_VIRTUAL_MODEL_DEPTH - 1;
        let mut vms: Vec<(String, String)> = (0..n)
            .map(|i| (format!("v{i}"), refer(&format!("v{}", i + 1))))
            .collect();
        vms.push((format!("v{n}"), leaf("p1")));
        let refs: Vec<(&str, &str)> = vms.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        nested_config(&refs).validate().unwrap();
    }

    #[test]
    fn test_flatten_is_cycle_safe_without_validate() {
        let c = nested_config(&[("a", &refer("b")), ("b", &refer("a"))]);
        // Never called validate(): flattening must refuse, not loop.
        assert!(c.flatten_virtual_models().is_err());
    }

    #[test]
    fn test_flatten_preserves_group_order_and_dedups() {
        // general -> [gemma, granite]; each group has two runtimes; p2 is
        // shared by both groups and a direct leaf follows.
        let c = nested_config(&[
            (
                "general",
                &format!("{}, {}, {}", refer("gemma"), refer("granite"), leaf("p4")),
            ),
            ("gemma", &format!("{}, {}", leaf("p1"), leaf("p2"))),
            ("granite", &format!("{}, {}", leaf("p3"), leaf("p2"))),
        ]);
        c.validate().unwrap();
        assert_eq!(flat_names(&c, "general"), ["p1", "p2", "p3", "p4"]);
        // Referenced models stay usable on their own.
        assert_eq!(flat_names(&c, "gemma"), ["p1", "p2"]);
    }

    #[test]
    fn test_flatten_applies_reference_priority_to_whole_group() {
        let c = nested_config(&[
            (
                "x",
                &format!(
                    "{}, {}",
                    "{ virtual = \"late\", priority = 2 }", "{ virtual = \"early\", priority = 1 }"
                ),
            ),
            ("late", &format!("{}, {}", leaf("p1"), leaf("p2"))),
            ("early", &format!("{}, {}", leaf("p3"), leaf("p4"))),
        ]);
        c.validate().unwrap();
        assert_eq!(flat_names(&c, "x"), ["p3", "p4", "p1", "p2"]);
    }

    #[test]
    fn test_nested_rejects_weight_on_reference_and_mixed_target() {
        let c = nested_config(&[("a", "{ virtual = \"b\", weight = 2 }"), ("b", &leaf("p1"))]);
        assert!(c.validate().unwrap_err().to_string().contains("weight"));
        let c = nested_config(&[
            ("a", "{ virtual = \"b\", provider = \"p1\", model = \"m\" }"),
            ("b", &leaf("p1")),
        ]);
        assert!(c
            .validate()
            .unwrap_err()
            .to_string()
            .contains("use one or the other"));
        let c = nested_config(&[("a", "{ priority = 1 }")]);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_flat_configs_flatten_to_themselves() {
        let c = nested_config(&[("a", &format!("{}, {}", leaf("p2"), leaf("p1")))]);
        assert_eq!(flat_names(&c, "a"), ["p2", "p1"]);
    }
}
