use opentelemetry::metrics::{Counter, Gauge, Histogram};
use opentelemetry::trace::{Span, TraceContextExt, TraceFlags, Tracer};
use opentelemetry::{global, Context, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::{RandomIdGenerator, SdkTracerProvider};
use opentelemetry_sdk::Resource;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::warn;

/// Defines the structured telemetry events allowed through our bounded channel.
#[derive(Debug)]
pub enum TelemetryEvent {
    RecordTransaction {
        operation: String, // "chat" or "embeddings"
        route: String,     // virtual model the client addressed
        provider: String,  // "groq", "sambanova", etc.
        model: String,
        input_tokens: u64,
        output_tokens: u64,
        duration: Duration,
        attempts: u32,
        failure_reason: Option<String>,
        trace_id: Option<String>,       // Extracted from W3C traceparent
        parent_span_id: Option<String>, // Extracted from W3C traceparent
        request_id: String,             // oxllm-generated request correlation ID
    },
    UpdateStatus {
        provider: String,
        status: u64, // 0 = Healthy, 1 = Cooldown, 2 = Tripped
    },
}

/// Lean, cloneable client handle exposed to the Axum router pipelines.
#[derive(Clone)]
pub struct TelemetryClient {
    sender: mpsc::Sender<TelemetryEvent>,
}

impl TelemetryClient {
    pub fn new(sender: mpsc::Sender<TelemetryEvent>) -> Self {
        Self { sender }
    }

    /// Emits events into the channel using a strict non-blocking bounded strategy.
    /// Defends memory pools against OOM spikes if the local collector (`otelite`) goes down.
    pub fn emit(&self, event: TelemetryEvent) {
        if let Err(_e) = self.sender.try_send(event) {
            // Graceful drop matching edge router constraints
            warn!(
                target: "oxllm::telemetry",
                "Telemetry queue full (1024 cap) or collector unreachable. Dropping event to prevent OOM."
            );
        }
    }
}

/// Background worker loop responsible for draining the channel and exporting to OTLP.
pub struct TelemetryWorker {
    rx: mpsc::Receiver<TelemetryEvent>,
    provider_status_gauge: Gauge<u64>,
    request_duration_histogram: Histogram<f64>,
    tokens_consumed_counter: Counter<u64>,
}

/// Full OTLP/HTTP URL for one signal. The exporter's `with_endpoint` uses the
/// URL verbatim (only the environment-variable route appends the signal
/// path), so the configured base endpoint must be extended here, or every
/// export POSTs to `/` and the collector answers 404.
fn signal_endpoint(base: &str, signal: &str) -> String {
    format!("{}/v1/{}", base.trim_end_matches('/'), signal)
}

impl TelemetryWorker {
    /// Spawns the worker thread alongside the runtime initialization.
    /// If the OTLP exporter cannot be initialised (e.g. collector offline or
    /// feature mismatch), the worker degrades gracefully to a no-op drain
    /// so the proxy still starts and serves requests. Telemetry is best-effort.
    pub fn spawn(
        otel_endpoint: &str,
        rx: mpsc::Receiver<TelemetryEvent>,
    ) -> Result<tokio::task::JoinHandle<()>, crate::error::OxllmError> {
        // Build Resource Metadata
        let resource = Resource::builder()
            .with_service_name("oxllm")
            .with_attributes(vec![KeyValue::new(
                "service.version",
                env!("CARGO_PKG_VERSION"),
            )])
            .build();

        let span_exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(signal_endpoint(otel_endpoint, "traces"))
            .build();

        let metric_exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .with_endpoint(signal_endpoint(otel_endpoint, "metrics"))
            .build();

        match (span_exporter, metric_exporter) {
            (Ok(se), Ok(me)) => {
                // 1. Full OTLP pipeline
                let tracer_provider = SdkTracerProvider::builder()
                    .with_resource(resource.clone())
                    .with_id_generator(RandomIdGenerator::default())
                    .with_batch_exporter(se)
                    .build();
                global::set_tracer_provider(tracer_provider);

                let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(me).build();
                let meter_provider = SdkMeterProvider::builder()
                    .with_resource(resource)
                    .with_reader(reader)
                    .build();
                global::set_meter_provider(meter_provider);

                let meter = global::meter("oxllm-metrics");
                let provider_status_gauge = meter
                    .u64_gauge("llm_proxy.provider.status")
                    .with_description("0=Healthy, 1=Cooldown, 2=Tripped")
                    .build();
                let request_duration_histogram = meter
                    .f64_histogram("llm_proxy.request.duration")
                    .with_description("Total transaction lifecycle duration")
                    .with_unit("ms")
                    // Default OTel buckets stop at 10 s, so slow LLM calls land
                    // in the +Inf overflow bucket, which some backends (otelite)
                    // refuse to store. Extend to 10 minutes.
                    .with_boundaries(vec![
                        5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0,
                        10_000.0, 30_000.0, 60_000.0, 120_000.0, 300_000.0, 600_000.0,
                    ])
                    .build();
                let tokens_consumed_counter = meter
                    .u64_counter("llm_proxy.tokens.consumed")
                    .with_description("Cumulative count of tokens processed")
                    .build();

                let mut worker = Self {
                    rx,
                    provider_status_gauge,
                    request_duration_histogram,
                    tokens_consumed_counter,
                };
                Ok(tokio::spawn(async move {
                    worker.run_loop().await;
                }))
            },
            (se_result, me_result) => {
                // Degraded mode: log the error(s) and drain the channel silently
                if let Err(e) = se_result {
                    warn!(
                        target: "oxllm::telemetry",
                        "OTLP span exporter failed to initialise (endpoint: {}): {}. \
                         Running without telemetry export.",
                        otel_endpoint, e
                    );
                }
                if let Err(e) = me_result {
                    warn!(
                        target: "oxllm::telemetry",
                        "OTLP metric exporter failed to initialise (endpoint: {}): {}. \
                         Running without telemetry export.",
                        otel_endpoint, e
                    );
                }
                // Drain the channel so senders never block
                Ok(tokio::spawn(async move {
                    let mut rx = rx;
                    while rx.recv().await.is_some() {} // silently discard
                }))
            },
        }
    }

    async fn run_loop(&mut self) {
        let tracer = global::tracer("oxllm-tracer");

        while let Some(event) = self.rx.recv().await {
            match event {
                TelemetryEvent::UpdateStatus { provider, status } => {
                    self.provider_status_gauge
                        .record(status, &[KeyValue::new("provider.name", provider)]);
                },
                TelemetryEvent::RecordTransaction {
                    operation,
                    route,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    duration,
                    attempts,
                    failure_reason,
                    trace_id,
                    parent_span_id,
                    request_id,
                } => {
                    let common_attributes = vec![
                        KeyValue::new("gen_ai.operation.name", operation.clone()),
                        KeyValue::new("gen_ai.provider.name", provider.clone()),
                        KeyValue::new("gen_ai.request.model", model.clone()),
                    ];

                    // 1. Export Metrics
                    let duration_ms = duration.as_secs_f64() * 1000.0;
                    self.request_duration_histogram
                        .record(duration_ms, &common_attributes);

                    if input_tokens > 0 {
                        let mut input_attrs = common_attributes.clone();
                        input_attrs.push(KeyValue::new("type", "input"));
                        self.tokens_consumed_counter.add(input_tokens, &input_attrs);
                    }
                    if output_tokens > 0 {
                        let mut output_attrs = common_attributes.clone();
                        output_attrs.push(KeyValue::new("type", "output"));
                        self.tokens_consumed_counter
                            .add(output_tokens, &output_attrs);
                    }

                    // 2. Build Propagated Trace Span Context if present
                    let span_builder = tracer.span_builder("oxllm.transaction");
                    let mut parent_ctx = None;

                    if let (Some(tid), Some(pid)) = (trace_id, parent_span_id) {
                        if let (Ok(t_bytes), Ok(s_bytes)) = (hex::decode(tid), hex::decode(pid)) {
                            if t_bytes.len() == 16 && s_bytes.len() == 8 {
                                let mut trace_id_arr = [0u8; 16];
                                let mut span_id_arr = [0u8; 8];
                                trace_id_arr.copy_from_slice(&t_bytes);
                                span_id_arr.copy_from_slice(&s_bytes);

                                let remote_span_context = opentelemetry::trace::SpanContext::new(
                                    opentelemetry::trace::TraceId::from_bytes(trace_id_arr),
                                    opentelemetry::trace::SpanId::from_bytes(span_id_arr),
                                    TraceFlags::SAMPLED,
                                    true,
                                    opentelemetry::trace::TraceState::default(),
                                );
                                parent_ctx = Some(
                                    Context::current()
                                        .with_remote_span_context(remote_span_context),
                                );
                            }
                        }
                    }

                    // Append Formal GenAI Semantic Convention Attributes
                    let mut attributes = common_attributes;
                    attributes.push(KeyValue::new(
                        "gen_ai.usage.input_tokens",
                        input_tokens as i64,
                    ));
                    attributes.push(KeyValue::new(
                        "gen_ai.usage.output_tokens",
                        output_tokens as i64,
                    ));
                    attributes.push(KeyValue::new("proxy.attempts_required", attempts as i64));

                    let terminal_failure = matches!(
                        failure_reason.as_deref(),
                        Some("all_providers_failed" | "no_available_provider" | "stream_failed")
                    );
                    if let Some(reason) = failure_reason {
                        attributes.push(KeyValue::new("proxy.initial_failure_reason", reason));
                    }

                    attributes.push(KeyValue::new("proxy.request_id", request_id));
                    attributes.push(KeyValue::new("proxy.route", route));

                    // Synthesize and explicitly end the span to flush to batch processor
                    let mut span = if let Some(ref cx) = parent_ctx {
                        tracer.build_with_context(span_builder, cx)
                    } else {
                        tracer.build(span_builder)
                    };
                    span.set_attributes(attributes);
                    if terminal_failure {
                        span.set_status(opentelemetry::trace::Status::error(
                            "request failed after all attempts",
                        ));
                    }
                    span.end_with_timestamp(std::time::SystemTime::now());
                },
            }
        }
    }
}

#[cfg(test)]
mod endpoint_tests {
    use super::signal_endpoint;

    #[test]
    fn test_signal_endpoint_appends_the_otlp_path() {
        assert_eq!(
            signal_endpoint("http://127.0.0.1:4318", "traces"),
            "http://127.0.0.1:4318/v1/traces"
        );
        assert_eq!(
            signal_endpoint("http://127.0.0.1:4318", "metrics"),
            "http://127.0.0.1:4318/v1/metrics"
        );
    }

    #[test]
    fn test_signal_endpoint_tolerates_trailing_slashes() {
        assert_eq!(
            signal_endpoint("http://collector:4318/", "traces"),
            "http://collector:4318/v1/traces"
        );
        assert_eq!(
            signal_endpoint("https://collector.example/otlp//", "metrics"),
            "https://collector.example/otlp/v1/metrics"
        );
    }
}
