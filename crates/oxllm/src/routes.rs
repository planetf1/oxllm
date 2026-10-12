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

use oxllm_core::router::{AdaptivePriorityStrategy, RoutingStrategy};
use oxllm_core::state::{AppState, CircuitState, ProviderState, SelectedProvider};
use oxllm_core::telemetry::{TelemetryClient, TelemetryEvent};

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

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
            });
        }
    }

    Json(ModelsResponse {
        object: "list",
        data,
    })
}

#[derive(Serialize)]
struct RouteEntry {
    provider: String,
    model: String,
    circuit: String,
    requests: u64,
    successes: u64,
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
        requests: u64,
        successes: u64,
        tokens_input: u64,
        tokens_output: u64,
        last_request: String,
    }

    #[derive(Serialize)]
    struct StatusResponse {
        uptime_secs: u64,
        total_requests: u64,
        providers: Vec<ProviderStatus>,
        virtual_models: std::collections::HashMap<String, Vec<RouteEntry>>,
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
        for target in targets {
            let provider_state = app_state
                .providers
                .iter()
                .find(|p| p.name == target.provider);
            let (circuit_str, requests, successes) = match provider_state {
                Some(provider) => {
                    let circ = *provider.circuit.read().await;
                    let now = std::time::Instant::now();
                    let circuit_str = match circ {
                        CircuitState::Closed => "Closed (Healthy)".to_string(),
                        CircuitState::HalfOpen => "Half-Open (Probing)".to_string(),
                        CircuitState::Open { until } => {
                            let left = until.saturating_duration_since(now).as_secs();
                            format!("Open ({}s cooldown)", left)
                        },
                    };
                    let requests = provider.requests.load(std::sync::atomic::Ordering::Relaxed);
                    let successes = provider
                        .successes
                        .load(std::sync::atomic::Ordering::Relaxed);
                    (circuit_str, requests, successes)
                },
                None => ("Unknown".to_string(), 0, 0),
            };
            entries.push(RouteEntry {
                provider: target.provider.clone(),
                model: target.model.clone(),
                circuit: circuit_str,
                requests,
                successes,
            });
        }
        virtual_models.insert(vm_name.clone(), entries);
    }

    Json(StatusResponse {
        uptime_secs: start_time.elapsed().as_secs(),
        total_requests,
        providers: status_list,
        virtual_models,
    })
}

/// Attaches standard headers, client forwarded headers (e.g. x-* headers), and provider custom headers.
fn attach_request_headers(
    mut req: reqwest::RequestBuilder,
    selected: &SelectedProvider,
    client_headers: &HeaderMap,
) -> reqwest::RequestBuilder {
    let has_custom_content_type = selected
        .headers
        .keys()
        .any(|k| k.eq_ignore_ascii_case("content-type"));
    if !has_custom_content_type {
        req = req.header("Content-Type", "application/json");
    }

    let has_custom_auth = selected
        .headers
        .keys()
        .any(|k| k.eq_ignore_ascii_case("authorization"));
    if !has_custom_auth && !selected.api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", selected.api_key));
    }

    // Propagate W3C tracing header if present
    if let Some(traceparent) = client_headers.get("traceparent") {
        req = req.header("traceparent", traceparent);
    }

    // Forward incoming client headers starting with 'x-' if not overridden by provider headers
    for (header_name, header_val) in client_headers {
        let name_str = header_name.as_str();
        if name_str.starts_with("x-")
            && !selected
                .headers
                .keys()
                .any(|k| k.eq_ignore_ascii_case(name_str))
        {
            if let Ok(val_str) = header_val.to_str() {
                req = req.header(name_str, val_str);
            }
        }
    }

    // Apply provider custom headers (overriding client headers or default Authorization if specified)
    for (k, v) in &selected.headers {
        req = req.header(k.as_str(), v.as_str());
    }

    req
}

/// POST /v1/embeddings
pub async fn create_embeddings(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes, // Bounded ref-counted bytes for zero-cost routing retries
) -> impl IntoResponse {
    let mut payload: Value = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            return json_error_response(
                &format!("Invalid JSON payload: {}", e),
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            )
        },
    };

    let requested_model = match payload.get("model").and_then(|m| m.as_str()) {
        Some(m) => m,
        None => {
            return json_error_response(
                "Missing required 'model' field",
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            )
        },
    };

    let candidates = app_state.resolve_candidates(requested_model);
    if candidates.is_empty() {
        return json_error_response(
            &format!("Invalid or unmapped virtual model: {}", requested_model),
            "invalid_request_error",
            StatusCode::BAD_REQUEST,
        );
    }

    // Extract W3C traceparent headers if present for tracing
    let (trace_id, parent_span_id) = extract_traceparent(&headers);

    let strategy = AdaptivePriorityStrategy;
    let mut attempts = 0;
    let start_time = Instant::now();
    let mut last_upstream_error = String::new();
    let mut last_failed_provider = String::new();
    let mut last_failed_status: u16 = 0;

    // Map candidate provider references for select strategy
    let candidate_states: Vec<&ProviderState> = candidates.iter().map(|(p, _)| *p).collect();

    while let Some(selected) = strategy.select(&candidate_states).await {
        attempts += 1;
        let target_model = match candidates
            .iter()
            .find(|(p, _)| p.name == selected.name)
            .map(|(_, m)| m.clone())
        {
            Some(m) => m,
            None => {
                warn!(
                    request_id = %request_id,
                    "Provider {} removed from candidates during embeddings routing",
                    selected.name
                );
                continue;
            },
        };

        // 1. Rewrite model field in JSON body
        payload["model"] = Value::String(target_model.clone());
        let rewritten_body = Bytes::from(serde_json::to_vec(&payload).unwrap());

        // 2. Safely parse base URL and join embeddings path (Url::joinTrailingSlashes mitigation)
        let endpoint_url = match selected.base_url.join("embeddings") {
            Ok(url) => url,
            Err(e) => {
                error!("Invalid base URL path join for {}: {}", selected.name, e);
                continue;
            },
        };

        // 3. Construct upstream request
        let req = app_state
            .http_client
            .post(endpoint_url.as_str())
            .body(rewritten_body)
            .timeout(Duration::from_secs(app_state.upstream_timeout_secs));
        let req = attach_request_headers(req, &selected, &headers);

        debug!(
            "Embedding request routing to {} (attempt {})",
            selected.name, attempts
        );

        // Increment request counter
        if let Some(target) = app_state.providers.iter().find(|p| p.name == selected.name) {
            target
                .requests
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let res = req.send().await;

        match res {
            Ok(res) if res.status().is_success() => {
                let status_code = res.status().as_u16();
                let upstream_headers = res.headers().clone();
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
                            .feedback(
                                target_provider_state,
                                false,
                                selected.is_probe,
                                Some(status_code),
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

                // Parse token counts from upstream response
                let (input_tokens, output_tokens) = serde_json::from_slice::<Value>(&res_body)
                    .map(|v| {
                        let usage = v.get("usage");
                        let input = usage
                            .and_then(|u| u.get("prompt_tokens"))
                            .and_then(|t| t.as_u64())
                            .unwrap_or(0);
                        let output = usage
                            .and_then(|u| u.get("completion_tokens"))
                            .and_then(|t| t.as_u64())
                            .unwrap_or(0);
                        (input, output)
                    })
                    .unwrap_or((0, 0));

                // Report Success feedback
                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback(
                        target_provider_state,
                        true,
                        selected.is_probe,
                        Some(status_code),
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

                // Push Telemetry Metrics
                telemetry.emit(TelemetryEvent::RecordTransaction {
                    operation: "embeddings".to_string(),
                    provider: selected.name.clone(),
                    model: target_model,
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
                let mut response = Response::new(Body::from(res_body));
                *response.status_mut() = StatusCode::OK;
                copy_response_headers(&upstream_headers, response.headers_mut());
                return response.into_response();
            },
            Ok(res) => {
                let status_code = res.status().as_u16();
                warn!(
                    request_id = %request_id,
                    "Embedding request upstream {} failed with status {}",
                    selected.name, status_code
                );

                // Extract Retry-After before consuming the response body
                let retry_after = extract_retry_after(res.headers());

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
                last_failed_status = status_code;

                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback(
                        target_provider_state,
                        false,
                        selected.is_probe,
                        Some(status_code),
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
                    "Embedding request upstream {} connection failed: {}",
                    selected.name, e
                );
                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback(target_provider_state, false, selected.is_probe, None, None)
                    .await;
                telemetry.emit(TelemetryEvent::UpdateStatus {
                    provider: selected.name.clone(),
                    status: circuit_status(target_provider_state).await,
                });
            },
        }
    }

    let message = if !last_upstream_error.is_empty() {
        format!(
            "All upstream embeddings providers failed or are rate-limited. Last error from {} ({}): {}",
            last_failed_provider, last_failed_status, last_upstream_error
        )
    } else {
        "All upstream embeddings providers failed or are rate-limited".to_string()
    };
    json_error_response(&message, "server_error", StatusCode::BAD_GATEWAY)
}

/// POST /v1/chat/completions
pub async fn create_chat_completions(
    State((app_state, telemetry)): State<(Arc<AppState>, TelemetryClient)>,
    Extension(request_id): Extension<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let mut payload: Value = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            return json_error_response(
                &format!("Invalid JSON payload: {}", e),
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            )
        },
    };

    let requested_model = match payload.get("model").and_then(|m| m.as_str()) {
        Some(m) => m,
        None => {
            return json_error_response(
                "Missing required 'model' field",
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
            )
        },
    };

    let candidates = app_state.resolve_candidates(requested_model);
    if candidates.is_empty() {
        return json_error_response(
            &format!("Invalid or unmapped virtual model: {}", requested_model),
            "invalid_request_error",
            StatusCode::BAD_REQUEST,
        );
    }

    let is_streaming = payload
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    let (trace_id, parent_span_id) = extract_traceparent(&headers);

    let strategy = AdaptivePriorityStrategy;
    let mut attempts = 0;
    let start_time = Instant::now();
    let mut last_upstream_error = String::new();
    let mut last_failed_provider = String::new();
    let mut last_failed_status: u16 = 0;

    let candidate_states: Vec<&ProviderState> = candidates.iter().map(|(p, _)| *p).collect();

    while let Some(selected) = strategy.select(&candidate_states).await {
        attempts += 1;
        let target_model = match candidates
            .iter()
            .find(|(p, _)| p.name == selected.name)
            .map(|(_, m)| m.clone())
        {
            Some(m) => m,
            None => {
                warn!(
                    request_id = %request_id,
                    "Provider {} removed from candidates during chat routing",
                    selected.name
                );
                continue;
            },
        };

        payload["model"] = Value::String(target_model.clone());
        let rewritten_body = Bytes::from(serde_json::to_vec(&payload).unwrap());

        let endpoint_url = match selected.base_url.join("chat/completions") {
            Ok(url) => url,
            Err(e) => {
                error!("Invalid base URL path join for {}: {}", selected.name, e);
                continue;
            },
        };

        let req = app_state
            .http_client
            .post(endpoint_url.as_str())
            .body(rewritten_body)
            .timeout(Duration::from_secs(app_state.upstream_timeout_secs));
        let req = attach_request_headers(req, &selected, &headers);

        debug!(
            "Chat request routing to {} (attempt {})",
            selected.name, attempts
        );

        // Increment request counter
        if let Some(target) = app_state.providers.iter().find(|p| p.name == selected.name) {
            target
                .requests
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let res = req.send().await;

        match res {
            Ok(res) if res.status().is_success() => {
                let status_code = res.status().as_u16();
                let upstream_headers = res.headers().clone();
                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();

                if is_streaming {
                    let app_state_clone = app_state.clone();
                    let provider_name = selected.name.clone();
                    let is_probe = selected.is_probe;
                    let telemetry_clone = telemetry.clone();
                    let trace_id_clone = trace_id.clone();
                    let parent_span_id_clone = parent_span_id.clone();
                    let request_id_clone = request_id.clone();
                    let duration = start_time.elapsed();
                    let attempt_count = attempts;
                    let model_name = target_model;

                    let (tx, rx) = tokio::sync::mpsc::channel::<
                        Result<Bytes, Box<dyn std::error::Error + Send + Sync>>,
                    >(32);

                    tokio::spawn(async move {
                        let mut reqwest_stream = res.bytes_stream();
                        let mut stream_success = true;

                        while let Some(chunk_result) = reqwest_stream.next().await {
                            match chunk_result {
                                Ok(chunk) => {
                                    if tx.send(Ok(chunk)).await.is_err() {
                                        // Client disconnected
                                        stream_success = false;
                                        break;
                                    }
                                },
                                Err(e) => {
                                    stream_success = false;
                                    let _ = tx
                                        .send(Err(
                                            Box::new(e) as Box<dyn std::error::Error + Send + Sync>
                                        ))
                                        .await;
                                    break;
                                },
                            }
                        }

                        // Deferred circuit-breaker feedback after stream completes
                        if let Some(provider_state) = app_state_clone
                            .providers
                            .iter()
                            .find(|p| p.name == provider_name)
                        {
                            let strategy = AdaptivePriorityStrategy;
                            strategy
                                .feedback(
                                    provider_state,
                                    stream_success,
                                    is_probe,
                                    Some(status_code),
                                    None,
                                )
                                .await;
                            telemetry_clone.emit(TelemetryEvent::UpdateStatus {
                                provider: provider_name.clone(),
                                status: circuit_status(provider_state).await,
                            });

                            if stream_success {
                                provider_state
                                    .successes
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }

                        // Push Telemetry Metrics
                        telemetry_clone.emit(TelemetryEvent::RecordTransaction {
                            operation: "chat".to_string(),
                            provider: provider_name,
                            model: model_name,
                            input_tokens: 0,
                            output_tokens: 0,
                            duration,
                            attempts: attempt_count,
                            failure_reason: if stream_success {
                                None
                            } else {
                                Some("stream_failed".to_string())
                            },
                            trace_id: trace_id_clone,
                            parent_span_id: parent_span_id_clone,
                            request_id: request_id_clone,
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
                    return response.into_response();
                } else {
                    // Report Success feedback (non-streaming: immediate, as before)
                    strategy
                        .feedback(
                            target_provider_state,
                            true,
                            selected.is_probe,
                            Some(status_code),
                            None,
                        )
                        .await;
                    telemetry.emit(TelemetryEvent::UpdateStatus {
                        provider: selected.name.clone(),
                        status: circuit_status(target_provider_state).await,
                    });

                    // Increment local counters (streaming: token counts deferred)
                    target_provider_state
                        .successes
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let res_body = match res.bytes().await {
                        Ok(b) => b,
                        Err(e) => {
                            warn!(
                                request_id = %request_id,
                                "Failed to read success response body from {}: {}",
                                selected.name, e
                            );
                            strategy
                                .feedback(
                                    target_provider_state,
                                    false,
                                    selected.is_probe,
                                    Some(status_code),
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

                    // Parse token counts from upstream response
                    let (input_tokens, output_tokens) = serde_json::from_slice::<Value>(&res_body)
                        .map(|v| {
                            let usage = v.get("usage");
                            let input = usage
                                .and_then(|u| u.get("prompt_tokens"))
                                .and_then(|t| t.as_u64())
                                .unwrap_or(0);
                            let output = usage
                                .and_then(|u| u.get("completion_tokens"))
                                .and_then(|t| t.as_u64())
                                .unwrap_or(0);
                            (input, output)
                        })
                        .unwrap_or((0, 0));

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

                    // Push Telemetry Metrics
                    telemetry.emit(TelemetryEvent::RecordTransaction {
                        operation: "chat".to_string(),
                        provider: selected.name.clone(),
                        model: target_model,
                        input_tokens,
                        output_tokens,
                        duration: start_time.elapsed(),
                        attempts,
                        failure_reason: None,
                        trace_id: trace_id.clone(),
                        parent_span_id: parent_span_id.clone(),
                        request_id: request_id.clone(),
                    });

                    let mut response = Response::new(Body::from(res_body));
                    *response.status_mut() = StatusCode::OK;
                    copy_response_headers(&upstream_headers, response.headers_mut());
                    return response.into_response();
                }
            },
            Ok(res) => {
                let status_code = res.status().as_u16();
                warn!(
                    request_id = %request_id,
                    "Chat completions upstream {} failed with status {}",
                    selected.name, status_code
                );

                // Extract Retry-After before consuming the response body
                let retry_after = extract_retry_after(res.headers());

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
                last_failed_status = status_code;

                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback(
                        target_provider_state,
                        false,
                        selected.is_probe,
                        Some(status_code),
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
                    "Chat completions upstream {} connection failed: {}",
                    selected.name, e
                );
                let target_provider_state = app_state
                    .providers
                    .iter()
                    .find(|p| p.name == selected.name)
                    .unwrap();
                strategy
                    .feedback(target_provider_state, false, selected.is_probe, None, None)
                    .await;
                telemetry.emit(TelemetryEvent::UpdateStatus {
                    provider: selected.name.clone(),
                    status: circuit_status(target_provider_state).await,
                });
            },
        }
    }

    let message = if !last_upstream_error.is_empty() {
        format!(
            "All upstream chat completions providers failed or are rate-limited. Last error from {} ({}): {}",
            last_failed_provider, last_failed_status, last_upstream_error
        )
    } else {
        "All upstream chat completions providers failed or are rate-limited".to_string()
    };
    json_error_response(&message, "server_error", StatusCode::BAD_GATEWAY)
}

/// Reads the current circuit and rate-limit state to produce a status code
/// for telemetry: 0 = Healthy, 1 = Cooldown, 2 = Tripped.
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
