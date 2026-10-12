use crate::error::{OxllmError, Result};
use reqwest::header::{HeaderName, HeaderValue};
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub enabled: bool,
    pub base_url: String,
    pub api_key: String,
    pub models: Vec<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualModelTarget {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub providers: Vec<ProviderConfig>,
    pub virtual_models: HashMap<String, Vec<VirtualModelTarget>>,
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

        // 2. Validate custom provider headers
        for provider in &self.providers {
            for (name, val) in &provider.headers {
                HeaderName::try_from(name.as_str()).map_err(|e| {
                    OxllmError::ConfigLoad(format!(
                        "Invalid header name '{}' in provider '{}': {}",
                        name, provider.name, e
                    ))
                })?;
                HeaderValue::try_from(val.as_str()).map_err(|e| {
                    OxllmError::ConfigLoad(format!(
                        "Invalid header value for '{}' in provider '{}': {}",
                        name, provider.name, e
                    ))
                })?;
            }
        }

        // 3. Validate that virtual models target existing, enabled providers
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

        Ok(())
    }
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

    #[test]
    fn test_provider_headers_parsing_and_env_expansion() {
        std::env::set_var("TEST_SESSION_ID", "sess-xyz-789");
        let toml_str = r#"
            [server]
            host = "127.0.0.1"
            port = 8080
            otel_endpoint = "http://localhost:4318"

            [[providers]]
            name = "opencode-go"
            enabled = true
            base_url = "https://opencode.ai/zen/go/v1/"
            api_key = "key123"
            models = ["longcat-2.5-preview-free"]
            headers = { "User-Agent" = "opencode/1.19.0", "x-opencode-session" = "${TEST_SESSION_ID}" }

            [virtual_models]
            "chat" = [{ provider = "opencode-go", model = "longcat-2.5-preview-free" }]
        "#;

        let expanded = expand_env_vars(toml_str).unwrap();
        let config: Config = toml::from_str(&expanded).unwrap();
        assert_eq!(config.providers.len(), 1);
        assert_eq!(
            config.providers[0].headers.get("User-Agent").unwrap(),
            "opencode/1.19.0"
        );
        assert_eq!(
            config.providers[0]
                .headers
                .get("x-opencode-session")
                .unwrap(),
            "sess-xyz-789"
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_provider_headers_invalid_name() {
        let toml_str = r#"
            [server]
            host = "127.0.0.1"
            port = 8080
            otel_endpoint = "http://localhost:4318"

            [[providers]]
            name = "bad-headers"
            enabled = true
            base_url = "https://example.com/v1"
            api_key = "key"
            models = ["model-1"]
            headers = { "Invalid Name Spaces" = "val" }

            [virtual_models]
            "test" = [{ provider = "bad-headers", model = "model-1" }]
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        let val_res = config.validate();
        assert!(val_res.is_err());
        assert!(val_res
            .unwrap_err()
            .to_string()
            .contains("Invalid header name"));
    }
}
