//! Bounded multipart task surface for configured ONNX models.

use crate::{
    adapters::OutputResolution,
    ocr::Page,
    pdf::PdfOptions,
    recipes::Kind,
    registry::{Engine, Model, Registry},
    representation::Representation,
    requests::{self, RequestOptions},
    residency::EngineLease,
};
use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header::CONTENT_LENGTH, header::VARY},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{MethodRouter, get, post},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::Cursor,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

/// Default upload cap, replaced by `ONNX_MAX_BYTES` / `--max-bytes`.
pub const DEFAULT_MAX_BYTES: usize = 100 * 1024 * 1024;
/// Flat budget for reading and validating an upload before the per-request
/// page-scaled deadline is known.
const FLAT_DEADLINE_SECS: u64 = 120;
/// One JSON object per line; backpressure bounds pages in flight.
const DEFAULT_STREAM_CHANNEL: usize = 16;

type StreamItem = Result<axum::body::Bytes, std::convert::Infallible>;
/// Per-router admission limits. One plugin process serves one router, so
/// these are process-wide in production; living in the state instead of
/// module statics keeps independently spun-up test servers from sharing them.
#[derive(Clone)]
pub struct Limits {
    uploads: Arc<tokio::sync::Semaphore>,
    validation: Arc<tokio::sync::Semaphore>,
    text_pdfs: Arc<tokio::sync::Semaphore>,
}

impl Limits {
    fn new() -> Self {
        Self {
            uploads: Arc::new(tokio::sync::Semaphore::new(4)),
            validation: Arc::new(tokio::sync::Semaphore::new(4)),
            text_pdfs: Arc::new(tokio::sync::Semaphore::new(4)),
        }
    }
}

#[derive(Clone)]
pub struct ServerState {
    pub registry: Arc<Registry>,
    pub pdfium_library: Option<PathBuf>,
    pub max_bytes: usize,
    pub limits: Limits,
}

impl ServerState {
    pub fn new(registry: Arc<Registry>, pdfium_library: Option<PathBuf>, max_bytes: usize) -> Self {
        Self {
            registry,
            pdfium_library,
            max_bytes,
            limits: Limits::new(),
        }
    }
}

pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .merge(document_routes(post(ocr), post(legacy_ocr)))
        .route("/v1/audio/transcriptions", post(transcription))
        .route("/mesh/v1/images/similarity", post(similarity))
        .route("/mesh/v1/images/depth", post(depth))
        .layer(DefaultBodyLimit::max(
            state.max_bytes.saturating_add(64 * 1024),
        ))
        .with_state(state)
}

/// The canonical Mesh task route and the legacy direct-provider alias share
/// one OCR handler, gated so an unacceptable `Accept` header is rejected
/// before multipart data is read. On the legacy path only, and only when
/// `Accept` is absent, the multipart `response_format` names the body.
fn document_routes<S: Clone + Send + Sync + 'static>(
    canonical: MethodRouter<S>,
    legacy: MethodRouter<S>,
) -> Router<S> {
    Router::new()
        .route("/mesh/v1/documents/ocr", canonical)
        .route("/v1/documents/ocr", legacy)
        .route_layer(middleware::from_fn(accept_negotiation))
}

async fn accept_negotiation(request: Request, next: Next) -> Response {
    let mut response = match preflight_accept(request.headers()) {
        Ok(()) => next.run(request).await,
        Err(error) => error.into_response(),
    };
    response
        .headers_mut()
        .insert(VARY, HeaderValue::from_static("Accept"));
    response
}

fn preflight_accept(headers: &HeaderMap) -> std::result::Result<(), ApiError> {
    // NDJSON preflights with the canonical route; the handler then narrows the
    // legacy alias back to the v0.1 representations.
    const OCR_REPRESENTATIONS: &[Representation] = &[
        Representation::Json,
        Representation::Text,
        Representation::Markdown,
        Representation::NdJson,
    ];
    let accept = headers
        .get_all("accept")
        .iter()
        .map(|value| value.to_str())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(ApiError::bad_request)?
        .join(",");
    Representation::negotiate(Some(&accept), OCR_REPRESENTATIONS)
        .map(|_| ())
        .map_err(|error| ApiError::new(StatusCode::NOT_ACCEPTABLE, error))
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    let data = state.registry.models.values().map(|model| {
        let availability = model.state();
        json!({"id":model.config.model,"object":"model","variant":model.config.variant,
            "assets_ready":model.assets_ready(),"workload_class":model.recipe.kind.workload(),"operations":[model.recipe.kind.operation()],
            "availability":if availability == "unloaded" {"loadable"} else {availability},
            "max_upload_bytes":state.max_bytes})
    }).collect::<Vec<_>>();
    Json(json!({"object":"list","data":data}))
}

/// Largest whole unit so the configured cap reads naturally ("100 MiB").
fn human_bytes(bytes: usize) -> String {
    for (unit, name) in [(1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")] {
        if bytes >= unit && bytes.is_multiple_of(unit) {
            return format!("{} {name}", bytes / unit);
        }
    }
    format!("{bytes} bytes")
}

fn content_length(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok())
}

struct Upload {
    model: String,
    /// The spooled upload; deleted when the request completes or drops. The
    /// multipart body is never buffered as a whole-file `Vec<u8>`.
    file: tempfile::NamedTempFile,
    fields: BTreeMap<String, String>,
}

/// Multipart failures raised by the body limit keep their 413 status and
/// name the configured cap; every other multipart error is a 400.
fn multipart_error(error: axum::extract::multipart::MultipartError, max_bytes: usize) -> ApiError {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "upload exceeds the limit of {} (ONNX_MAX_BYTES)",
                human_bytes(max_bytes)
            ),
        );
    }
    ApiError::bad_request(error)
}

async fn read_upload(mut multipart: Multipart, max_bytes: usize) -> Result<Upload, ApiError> {
    let mut file = None;
    let mut fields = BTreeMap::new();
    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|error| multipart_error(error, max_bytes))?
    {
        let name = field
            .name()
            .ok_or_else(|| ApiError::bad_request("multipart field name required"))?
            .to_string();
        if name == "file" {
            if file.is_some() {
                return Err(ApiError::bad_request("duplicate file field"));
            }
            // Creating the spool file is blocking I/O; the writes below go
            // through tokio's async file. The NamedTempFile stays alive for
            // the whole read so the spool is deleted on success, error and
            // drop alike.
            let temp = tokio::task::spawn_blocking(|| {
                tempfile::Builder::new().prefix("mesh-onnx-").tempfile()
            })
            .await
            .map_err(ApiError::internal)?
            .map_err(ApiError::internal)?;
            let mut writer =
                tokio::fs::File::from_std(temp.as_file().try_clone().map_err(ApiError::internal)?);
            let mut size: u64 = 0;
            while let Some(chunk) = field
                .chunk()
                .await
                .map_err(|error| multipart_error(error, max_bytes))?
            {
                size = size.saturating_add(chunk.len() as u64);
                if size > max_bytes as u64 {
                    return Err(ApiError::new(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        format!(
                            "file is {size} bytes; the limit is {} (ONNX_MAX_BYTES)",
                            human_bytes(max_bytes)
                        ),
                    ));
                }
                writer.write_all(&chunk).await.map_err(ApiError::internal)?;
            }
            writer.flush().await.map_err(ApiError::internal)?;
            // The clone closes; the NamedTempFile owns the path from here.
            drop(writer);
            if size == 0 {
                return Err(ApiError::bad_request("file must contain at least 1 byte"));
            }
            file = Some(temp);
        } else {
            if fields.len() >= 32 || fields.contains_key(&name) {
                return Err(ApiError::bad_request(
                    "too many or duplicate multipart fields",
                ));
            }
            let mut value = Vec::new();
            while let Some(chunk) = field
                .chunk()
                .await
                .map_err(|error| multipart_error(error, max_bytes))?
            {
                if value.len().saturating_add(chunk.len()) > 64 * 1024 {
                    return Err(ApiError::bad_request("multipart control exceeds 64 KiB"));
                }
                value.extend_from_slice(&chunk);
            }
            let value = String::from_utf8(value).map_err(ApiError::bad_request)?;
            fields.insert(name, value);
        }
    }
    let model = fields
        .remove("model")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ApiError::bad_request("model field is required"))?;
    Ok(Upload {
        model,
        file: file.ok_or_else(|| ApiError::bad_request("file field is required"))?,
        fields,
    })
}

async fn ocr(
    State(state): State<ServerState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    task(state, headers, multipart, Kind::Indra, false).await
}
async fn legacy_ocr(
    State(state): State<ServerState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    task(state, headers, multipart, Kind::Indra, true).await
}
async fn transcription(
    State(state): State<ServerState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    task(state, headers, multipart, Kind::Whisper, false).await
}
async fn similarity(
    State(state): State<ServerState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    task(state, headers, multipart, Kind::Clip, false).await
}
async fn depth(
    State(state): State<ServerState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    task(state, headers, multipart, Kind::Depth, false).await
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

async fn task(
    state: ServerState,
    headers: HeaderMap,
    multipart: Multipart,
    kind: Kind,
    legacy: bool,
) -> Result<Response, ApiError> {
    let _upload_permit = state
        .limits
        .uploads
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "upload capacity is full"))?;
    let flat_deadline = tokio::time::Instant::now() + Duration::from_secs(FLAT_DEADLINE_SECS);
    let supported: &[Representation] = match kind {
        // The legacy alias keeps the v0.1 representations: it never streams.
        Kind::Indra if legacy => &[
            Representation::Json,
            Representation::Text,
            Representation::Markdown,
        ],
        Kind::Indra => &[
            Representation::Json,
            Representation::Text,
            Representation::Markdown,
            Representation::NdJson,
        ],
        Kind::Whisper => &[Representation::Json, Representation::Text],
        Kind::Depth => &[Representation::Json, Representation::Png],
        _ => &[Representation::Json],
    };
    let accept = headers
        .get_all("accept")
        .iter()
        .map(|value| value.to_str())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(ApiError::bad_request)?
        .join(",");
    // A known-length over-cap body fails before it is read. Chunked bodies
    // are caught by the byte count in `read_upload`.
    if let Some(length) = content_length(&headers)
        && length > state.max_bytes.saturating_add(64 * 1024) as u64
    {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "file is {length} bytes; the limit is {} (ONNX_MAX_BYTES)",
                human_bytes(state.max_bytes)
            ),
        ));
    }
    let mut upload =
        tokio::time::timeout_at(flat_deadline, read_upload(multipart, state.max_bytes))
            .await
            .map_err(|_| {
                ApiError::new(StatusCode::GATEWAY_TIMEOUT, "upload deadline exceeded")
            })??;
    drop(_upload_permit);
    // OpenAI-shaped surfaces name the body in the form. On the OpenAI
    // transcription route the field wins outright; on the legacy OCR alias it
    // applies only when `Accept` is absent, and any supplied `Accept` —
    // including `*/*` — overrides it. Remove it before validating the task's
    // own options so the alias does not leak the field into OCR defaults.
    let body_format = if kind == Kind::Whisper || legacy {
        upload.fields.remove("response_format")
    } else {
        None
    };
    let format = if kind == Kind::Whisper
        && let Some(requested) = body_format.as_deref()
    {
        Representation::from_openai_response_format(requested).map_err(ApiError::bad_request)?
    } else if accept.trim().is_empty()
        && let Some(requested) = body_format.as_deref()
    {
        match requested.trim().to_ascii_lowercase().as_str() {
            "json" => Representation::Json,
            "text" => Representation::Text,
            "markdown" => Representation::Markdown,
            _ => {
                return Err(ApiError::bad_request(
                    "response_format must be json, text, or markdown",
                ));
            }
        }
    } else {
        Representation::negotiate(Some(&accept), supported)
            .map_err(|error| ApiError::new(StatusCode::NOT_ACCEPTABLE, error))?
    };
    let model = state
        .registry
        .select(&upload.model, kind)
        .map_err(ApiError::bad_request)?;
    let mut fields = std::mem::take(&mut upload.fields);
    if legacy {
        // The alias serves v0.1 clients; the v0.2 fields stay canonical-only.
        for v2_only in ["pages", "pages_only", "stream", "deadline_secs"] {
            if fields.remove(v2_only).is_some() {
                return Err(ApiError::bad_request(format!(
                    "`{v2_only}` is only supported on the canonical /mesh/v1/documents/ocr route"
                )));
            }
        }
    }
    // The alias serves v0.1 clients: v0.2-only defaults must not reach their
    // merge at all, or a default `pages` would collide with a legacy
    // `max_pages` and reject a valid v0.1 request.
    let defaults = model
        .config
        .request_defaults
        .iter()
        .filter(|(key, _)| {
            !legacy
                || !matches!(
                    key.as_str(),
                    "pages" | "pages_only" | "stream" | "deadline_secs"
                )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    let options = requests::merge(kind, &defaults, fields).map_err(ApiError::bad_request)?;
    if let RequestOptions::Ocr(mut options) = options {
        if legacy {
            // The alias serves v0.1 clients: v0.2-only options never apply
            // there, even when `request_defaults` sets them. Without this a
            // default could turn a legacy request into metadata or NDJSON.
            options.pages = None;
            options.pages_only = false;
            options.stream = false;
            options.deadline_secs = None;
        }
        return ocr_task(state, model, upload, options, format, legacy, flat_deadline).await;
    }
    // Whisper, CLIP and Depth keep the v0.1 semantics: a flat 120-second
    // budget from the start of the request.
    let deadline = flat_deadline;
    let cancel = CancelOnDrop(Arc::new(AtomicBool::new(false)));
    let cancellation = cancel.0.clone();
    let result = tokio::time::timeout_at(deadline, async move {
        let permit = state
            .limits
            .validation
            .clone()
            .try_acquire_owned()
            .map_err(|_| {
                ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "validation capacity is full",
                )
            })?;
        let (upload, options, bytes) = tokio::time::timeout_at(
            deadline,
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let bytes = std::fs::read(upload.file.path()).map_err(ApiError::internal)?;
                match &options {
                    RequestOptions::Whisper(_) => {
                        crate::adapters::whisper::validate_wav(&bytes)
                            .map_err(ApiError::bad_request)?;
                    }
                    _ => {
                        decode_image(&bytes).map_err(ApiError::bad_request)?;
                    }
                }
                Ok::<_, ApiError>((upload, options, bytes))
            }),
        )
        .await
        .map_err(|_| ApiError::new(StatusCode::GATEWAY_TIMEOUT, "validation deadline exceeded"))?
        .map_err(ApiError::internal)??;
        let mut lease = model
            .acquire()
            .await
            .map_err(|error| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, error))?;
        tokio::task::spawn_blocking(move || {
            let engine = lease
                .engine()
                .as_mut()
                .context("model engine is not loaded")?;
            execute(
                engine,
                options,
                &bytes,
                &cancellation,
                &upload.model,
                format,
            )
        })
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::processing)
    })
    .await
    .map_err(|_| {
        ApiError::new(
            StatusCode::GATEWAY_TIMEOUT,
            "model request deadline exceeded",
        )
    })??;
    response(result, format)
}

/// Canonical-route OCR: a decoding-only pre-flight yields the page count
/// (metadata answers, deadline default, range validation) without touching
/// model admission, then the request either answers metadata, streams NDJSON
/// page events, or runs the aggregate path.
async fn ocr_task(
    state: ServerState,
    model: Arc<Model>,
    upload: Upload,
    options: PdfOptions,
    format: Representation,
    legacy: bool,
    flat_deadline: tokio::time::Instant,
) -> Result<Response, ApiError> {
    let permit = state
        .limits
        .validation
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "validation capacity is full",
            )
        })?;
    let source = crate::pdf::DocumentSource::Path(upload.file.path().to_path_buf());
    let pdfium = state.pdfium_library.clone();
    let inspect_options = options.clone();
    let metadata = tokio::time::timeout_at(
        tokio::time::Instant::now() + Duration::from_secs(FLAT_DEADLINE_SECS),
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            crate::pdf::inspect_document(&source, pdfium.as_deref(), &inspect_options)
        }),
    )
    .await
    .map_err(|_| ApiError::new(StatusCode::GATEWAY_TIMEOUT, "validation deadline exceeded"))?
    .map_err(ApiError::internal)
    .and_then(|result| result.map_err(ApiError::processing))?;
    if options.pages_only {
        // Decoding only: no engine, no admission, answered on a cold model.
        return Ok(Json(json!({
            "model": upload.model,
            "kind": "ocr",
            "pages": metadata.pages,
            "page_sizes": metadata.page_sizes,
        }))
        .into_response());
    }
    if let Some(range) = &options.pages {
        range
            .validate_against(metadata.pages)
            .map_err(ApiError::bad_request)?;
    }
    let selected = match &options.pages {
        Some(range) => range.selected_len(metadata.pages),
        None => metadata.pages.min(options.max_pages),
    };
    // The alias serves v0.1 clients: their contract is the flat 120-second
    // budget, whatever the page-scaled default says.
    let budget = if legacy {
        FLAT_DEADLINE_SECS
    } else {
        options
            .deadline_secs
            .unwrap_or_else(|| requests::default_deadline_secs(selected))
    };
    let deadline = if legacy {
        flat_deadline
    } else {
        tokio::time::Instant::now() + Duration::from_secs(budget)
    };
    if format == Representation::NdJson || options.stream {
        return ocr_stream(state, model, upload, options, budget, metadata.pages).await;
    }
    let cancel = CancelOnDrop(Arc::new(AtomicBool::new(false)));
    let cancellation = cancel.0.clone();
    let mut result = tokio::time::timeout_at(deadline, async move {
        if options.mode == crate::ocr::document::OcrMode::Text {
            let text_permit = state
                .limits
                .text_pdfs
                .clone()
                .try_acquire_owned()
                .map_err(|_| {
                    ApiError::new(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "PDF worker capacity is full",
                    )
                })?;
            return tokio::task::spawn_blocking(move || {
                let _text_permit = text_permit;
                let document = crate::pdf::process_cancellable_file(
                    upload.file.path(),
                    state.pdfium_library.as_deref(),
                    &options,
                    None,
                    &cancellation,
                )?;
                let mut value = serde_json::to_value(document)?;
                value["model"] = Value::String(upload.model);
                Ok::<_, anyhow::Error>(Payload::Value(value))
            })
            .await
            .map_err(ApiError::internal)?
            .map_err(ApiError::processing);
        }
        let mut lease = model
            .acquire()
            .await
            .map_err(|error| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, error))?;
        tokio::task::spawn_blocking(move || {
            let engine = lease
                .engine()
                .as_mut()
                .context("model engine is not loaded")?;
            let Engine::Indra(engine) = engine else {
                anyhow::bail!("model engine and request task do not match");
            };
            let engine: &mut crate::ocr::Engine = engine;
            let document = crate::pdf::process_cancellable_file(
                upload.file.path(),
                state.pdfium_library.as_deref(),
                &options,
                Some(engine),
                &cancellation,
            )?;
            let mut value = serde_json::to_value(document)?;
            value["model"] = Value::String(upload.model);
            Ok::<_, anyhow::Error>(Payload::Value(value))
        })
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::processing)
    })
    .await
    .map_err(|_| deadline_exceeded(budget))??;
    // The alias serves v0.1 clients: their JSON keeps the exact shape they
    // decoded before, without the v0.2 response bounds.
    if legacy
        && let Payload::Value(value) = &mut result
        && let Some(map) = value.as_object_mut()
    {
        map.remove("page_from");
        map.remove("page_to");
    }
    response(result, format)
}

/// 408-style timeout: name the budget and point at `pages` for chunking.
fn deadline_exceeded(budget: u64) -> ApiError {
    ApiError::new(
        StatusCode::REQUEST_TIMEOUT,
        format!(
            "model request deadline of {budget}s exceeded; \
             split the document into chunks with `pages` (for example pages=1-10) and retry"
        ),
    )
}

/// NDJSON wire events. Shapes are fixed so chunked callers can reassemble.
fn meta_event(model: &str, page_from: usize, page_to: usize) -> Value {
    json!({
        "type": "meta",
        "model": model,
        "pages": page_to - page_from + 1,
        "page_from": page_from,
        "page_to": page_to,
    })
}

fn page_event(page: &Page) -> Value {
    let serialized = serde_json::to_value(page).expect("page serializes");
    json!({
        "type": "page",
        "page": page.page_index + 1,
        "text": page.text,
        "markdown": page.markdown,
        "lines": serialized["lines"],
        "blocks": serialized["blocks"],
    })
}

fn done_event(pages: usize) -> Value {
    json!({"type": "done", "pages": pages})
}

fn error_event(message: &str) -> Value {
    json!({"type": "error", "error": {"message": message}})
}

/// NDJSON streaming on the canonical route. Acquisition happens before the
/// 200 is sent, so load failures stay normal HTTP errors; once the stream
/// starts, a failure is delivered as an `error` event and the stream ends.
/// Pages emit as each completes, in page order, from a spooled upload.
async fn ocr_stream(
    state: ServerState,
    model: Arc<Model>,
    upload: Upload,
    options: PdfOptions,
    budget: u64,
    page_count: usize,
) -> Result<Response, ApiError> {
    enum Worker {
        Text(tokio::sync::OwnedSemaphorePermit),
        Engine(EngineLease<Engine>),
    }
    let worker = if options.mode == crate::ocr::document::OcrMode::Text {
        // Text mode needs no engine, only the PDF worker pool.
        Worker::Text(
            state
                .limits
                .text_pdfs
                .clone()
                .try_acquire_owned()
                .map_err(|_| {
                    ApiError::new(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "PDF worker capacity is full",
                    )
                })?,
        )
    } else {
        Worker::Engine(
            model
                .acquire()
                .await
                .map_err(|error| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, error))?,
        )
    };
    let page_from = options.pages.as_ref().map_or(1, |range| range.from);
    let page_to = options
        .pages
        .as_ref()
        .map_or(page_count.min(options.max_pages), |range| {
            range.to.unwrap_or(page_count)
        });
    let (tx, rx) = mpsc::channel::<StreamItem>(DEFAULT_STREAM_CHANNEL);
    let cancel = Arc::new(AtomicBool::new(false));
    let stream_cancel = cancel.clone();
    let meta = serde_json::to_string(&meta_event(&upload.model, page_from, page_to))
        .map_err(ApiError::internal)?;
    let pdfium = state.pdfium_library.clone();
    let file = upload.file;
    tokio::task::spawn_blocking(move || {
        let send = |line: &str| -> Result<()> {
            // One JSON object per line: the framing NDJSON promises.
            tx.blocking_send(Ok(format!("{line}\n").into()))
                .map_err(|_| anyhow::anyhow!("stream receiver dropped"))?;
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("request cancelled");
            }
            Ok(())
        };
        let stream_result = (|| -> Result<()> {
            send(&meta)?;
            let started = Instant::now();
            let budget = Duration::from_secs(budget);
            let mut sink = |page: Page| -> Result<()> {
                if started.elapsed() > budget {
                    anyhow::bail!(
                        "model request deadline of {}s exceeded; \
                         split the document into chunks with `pages` (for example pages=1-10) and retry",
                        budget.as_secs()
                    );
                }
                let line = serde_json::to_string(&page_event(&page))?;
                send(&line)
            };
            let processed = match worker {
                Worker::Text(permit) => {
                    let _permit = permit;
                    crate::pdf::process_cancellable_streaming(
                        file.path(),
                        pdfium.as_deref(),
                        &options,
                        None,
                        &cancel,
                        &mut sink,
                    )
                }
                Worker::Engine(mut lease) => {
                    let engine = lease
                        .engine()
                        .as_mut()
                        .context("model engine is not loaded")?;
                    let Engine::Indra(engine) = engine else {
                        anyhow::bail!("model engine and request task do not match");
                    };
                    let engine: &mut crate::ocr::Engine = engine;
                    crate::pdf::process_cancellable_streaming(
                        file.path(),
                        pdfium.as_deref(),
                        &options,
                        Some(engine),
                        &cancel,
                        &mut sink,
                    )
                }
            };
            processed?;
            send(&serde_json::to_string(&done_event(
                page_to - page_from + 1,
            ))?)
        })();
        if let Err(error) = stream_result {
            // Best effort: the receiver may already be gone.
            stream_cancel.store(true, Ordering::Relaxed);
            if let Ok(line) = serde_json::to_string(&error_event(&error.to_string())) {
                let _ = tx.blocking_send(Ok(format!("{line}\n").into()));
            }
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", Representation::NdJson.content_type())
        .header(VARY, HeaderValue::from_static("Accept"))
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .map_err(ApiError::internal)
}

/// A task result in the representation the caller negotiated. Some tasks can
/// produce a body that is not JSON at all, so the payload carries the bytes
/// decision rather than forcing everything through a `Value`.
enum Payload {
    Value(Value),
    Png(Vec<u8>),
}

fn execute(
    engine: &mut Engine,
    options: RequestOptions,
    file: &[u8],
    cancel: &AtomicBool,
    model: &str,
    format: Representation,
) -> Result<Payload> {
    match (engine, options) {
        (Engine::Other(engine), RequestOptions::Clip(options)) => {
            let scores = engine.clip_rank(
                &decode_image(file)?,
                &options.labels,
                options.top_k,
                options.truncate_text,
                cancel,
            )?;
            Ok(Payload::Value(
                json!({"model":model,"scores":scores,"score_type":"cosine_similarity"}),
            ))
        }
        (Engine::Other(engine), RequestOptions::Whisper(options)) => {
            let mut result = serde_json::to_value(engine.transcribe_wav(
                file,
                &options.language,
                options.max_output_tokens,
                cancel,
            )?)?;
            result["model"] = Value::String(model.into());
            Ok(Payload::Value(result))
        }
        (Engine::Other(engine), RequestOptions::Depth(options)) => {
            let resolution = if options.output_resolution == "input" {
                OutputResolution::Input
            } else {
                OutputResolution::Native
            };
            let depth = engine.depth(&decode_image(file)?, resolution)?;
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("request cancelled");
            }
            // Asking for an image is itself the request for a preview, so the
            // representation decides rather than the `visualization` default.
            if format == Representation::Png {
                let preview = depth.normalized_preview()?;
                let mut png = Vec::new();
                preview.write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)?;
                return Ok(Payload::Png(png));
            }
            let preview = if options.visualization == "normalized" {
                Some(depth.normalized_preview()?.into_raw())
            } else {
                None
            };
            let mut value = serde_json::to_value(&depth)?;
            value["model"] = model.into();
            if let Some(preview) = preview {
                value["preview"] = json!({"encoding":"grayscale_u8","width":depth.width,"height":depth.height,"pixels":preview});
            }
            Ok(Payload::Value(value))
        }
        _ => anyhow::bail!("model engine and request task do not match"),
    }
}

fn decode_image(file: &[u8]) -> Result<image::DynamicImage> {
    let mut reader = image::ImageReader::new(Cursor::new(file)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(4096);
    limits.max_image_height = Some(4096);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    reader.decode().context("decode bounded image upload")
}

fn response(payload: Payload, format: Representation) -> Result<Response, ApiError> {
    let body = match payload {
        Payload::Png(png) => png,
        Payload::Value(value) => match format {
            Representation::Json | Representation::NdJson => {
                serde_json::to_vec(&value).map_err(ApiError::internal)?
            }
            representation => {
                let field = if representation == Representation::Markdown {
                    "markdown"
                } else {
                    "text"
                };
                let text = if let Some(pages) = value["pages"].as_array() {
                    pages
                        .iter()
                        .filter_map(|page| page[field].as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                } else {
                    value[field].as_str().unwrap_or("").to_string()
                };
                text.into_bytes()
            }
        },
    };
    Ok((
        [("content-type", format.content_type()), ("vary", "Accept")],
        body,
    )
        .into_response())
}

struct ApiError {
    status: StatusCode,
    message: String,
}
impl ApiError {
    fn new(status: StatusCode, error: impl std::fmt::Display) -> Self {
        Self {
            status,
            message: error.to_string(),
        }
    }
    fn bad_request(error: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::BAD_REQUEST, error)
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, error)
    }
    fn processing(error: anyhow::Error) -> Self {
        if error.is::<crate::adapters::InvalidRequest>() {
            return Self::bad_request(error);
        }
        if error.is::<crate::pdf::PasswordRequired>() {
            return Self::new(StatusCode::UNAUTHORIZED, error);
        }
        let status = match error.downcast_ref::<crate::pdf::InputError>() {
            Some(crate::pdf::InputError::Invalid(_)) => StatusCode::BAD_REQUEST,
            Some(crate::pdf::InputError::Unavailable(_)) => StatusCode::SERVICE_UNAVAILABLE,
            None => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, error)
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            [("vary", "Accept")],
            Json(json!({"error":{"message":self.message}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin::models::ManagedModelConfig;
    const INDRA: &str = "datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b";

    fn registry(model: &str) -> Arc<Registry> {
        let config: ManagedModelConfig = serde_json::from_value(json!({"model":model,"provider":"onnx","lifecycle":{"load":"on_demand","prefetch":false}})).unwrap();
        Arc::new(Registry::new(vec![config]).unwrap())
    }

    /// A registry whose model defaults carry a v0.2 selector: the legacy
    /// alias must ignore them, never collide with them.
    fn registry_with_pages_default(model: &str) -> Arc<Registry> {
        let config: ManagedModelConfig = serde_json::from_value(json!({"model":model,"provider":"onnx","lifecycle":{"load":"on_demand","prefetch":false},"request_defaults":{"pages":"1-10"}})).unwrap();
        Arc::new(Registry::new(vec![config]).unwrap())
    }

    async fn serve(state: ServerState) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, router(state)).await.unwrap();
        });
        (url, task)
    }

    async fn server() -> (String, tokio::task::JoinHandle<()>) {
        serve(ServerState::new(registry(INDRA), None, DEFAULT_MAX_BYTES)).await
    }

    async fn server_with(model: &str) -> (String, tokio::task::JoinHandle<()>) {
        serve(ServerState::new(registry(model), None, DEFAULT_MAX_BYTES)).await
    }

    fn tiny_png() -> Vec<u8> {
        let image = image::GrayImage::from_pixel(8, 8, image::Luma([0u8]));
        let mut png = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    fn tiny_wav() -> Vec<u8> {
        // A tenth of a second of 16 kHz mono silence: enough to pass audio
        // validation on a route whose model is deliberately not loaded.
        let frames: u32 = 1600;
        let data = vec![0u8; (frames * 2) as usize];
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        wav
    }

    fn ocr_form(extra: Vec<(&'static str, &'static str)>) -> reqwest::multipart::Form {
        let mut form = reqwest::multipart::Form::new()
            .text("model", INDRA)
            .part("file", reqwest::multipart::Part::bytes(tiny_png()));
        for (key, value) in extra {
            form = form.text(key, value);
        }
        form
    }

    #[tokio::test]
    async fn unloaded_models_remain_discoverable_without_loading_weights() {
        let (url, task) = server().await;
        let data: Value = reqwest::get(format!("{url}/v1/models"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(data["data"][0]["availability"], "loadable");
        assert_eq!(data["data"][0]["operations"][0], "document_ocr");
        assert_eq!(data["data"][0]["max_upload_bytes"], DEFAULT_MAX_BYTES);
        assert!(
            reqwest::get(format!("{url}/health"))
                .await
                .unwrap()
                .status()
                .is_success()
        );
        task.abort();
    }

    #[tokio::test]
    async fn depth_offers_an_image_body_and_ocr_refuses_it() {
        // The representation decision happens before weights are touched, so an
        // unloaded model still shows which bodies its task can produce: the
        // depth request gets past negotiation and stops at loading, while the
        // OCR request never gets past the media type it cannot produce.
        let (url, task) = server_with(
            "onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e",
        )
        .await;
        let form = reqwest::multipart::Form::new()
            .text("model", "onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e")
            .part("file", reqwest::multipart::Part::bytes(tiny_png()).file_name("scene.png"));
        let response = reqwest::Client::new()
            .post(format!("{url}/mesh/v1/images/depth"))
            .header("Accept", "image/png")
            .multipart(form)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get("vary")
                .and_then(|value| value.to_str().ok()),
            Some("Accept")
        );
        task.abort();
    }

    #[tokio::test]
    async fn rejects_bad_formats_fields_and_cross_task_routing_before_loading() {
        let (url, task) = server().await;
        let client = reqwest::Client::new();
        // Absolute paths: `/v1/` is the OpenAI-shaped surface, `/mesh/v1/` the
        // Mesh task surface, so the two cannot share a URL prefix helper.
        for (path, extra, accept, expected) in [
            (
                "/mesh/v1/documents/ocr",
                Some(("response_format", "json")),
                "application/json",
                StatusCode::BAD_REQUEST,
            ),
            (
                "/v1/audio/transcriptions",
                None,
                "application/json",
                StatusCode::BAD_REQUEST,
            ),
            (
                "/mesh/v1/documents/ocr",
                Some(("max_pages", "0")),
                "application/json",
                StatusCode::BAD_REQUEST,
            ),
            (
                "/mesh/v1/documents/ocr",
                None,
                "image/png",
                StatusCode::NOT_ACCEPTABLE,
            ),
        ] {
            let mut form = reqwest::multipart::Form::new().text("model", INDRA).part(
                "file",
                reqwest::multipart::Part::bytes(if path == "/v1/audio/transcriptions" {
                    tiny_wav()
                } else {
                    tiny_png()
                }),
            );
            if let Some((key, value)) = extra {
                form = form.text(key, value);
            }
            let response = client
                .post(format!("{url}{path}"))
                .header("Accept", accept)
                .multipart(form)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
        }
        task.abort();
    }

    #[tokio::test]
    async fn repeated_accept_headers_are_combined_before_negotiation() {
        let (url, task) = server().await;
        let form = reqwest::multipart::Form::new()
            .text("model", INDRA)
            .part("file", reqwest::multipart::Part::bytes(tiny_png()));
        let response = reqwest::Client::new()
            .post(format!("{url}/mesh/v1/documents/ocr"))
            .header("Accept", "image/png")
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "the second header permits JSON, then assets are unavailable"
        );
        task.abort();
    }

    #[tokio::test]
    async fn legacy_ocr_alias_honors_response_format_only_without_accept() {
        // The legacy direct-provider alias keeps serving older clients: with no
        // `Accept` header the multipart `response_format` names the body, any
        // supplied `Accept` overrides the field, and the canonical task route
        // keeps rejecting the field outright.
        let (url, task) = server().await;
        let client = reqwest::Client::new();
        let form = |format: Option<&str>| {
            let mut form = reqwest::multipart::Form::new()
                .text("model", INDRA)
                .part("file", reqwest::multipart::Part::bytes(tiny_png()));
            if let Some(format) = format {
                form = form.text("response_format", format.to_string());
            }
            form
        };
        for (accept, format, expected) in [
            // No preference: the legacy field selects the representation and
            // the unloaded model stops the request past negotiation.
            ("", Some("text"), StatusCode::SERVICE_UNAVAILABLE),
            // Any supplied Accept overrides the field, even `*/*`, and an
            // invalid field value is ignored rather than faulted.
            ("*/*", Some("invalid"), StatusCode::SERVICE_UNAVAILABLE),
            // An unacceptable Accept never falls back to the field.
            ("image/png", Some("text"), StatusCode::NOT_ACCEPTABLE),
        ] {
            let mut request = client
                .post(format!("{url}/v1/documents/ocr"))
                .multipart(form(format));
            if !accept.is_empty() {
                request = request.header("Accept", accept);
            }
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), expected, "accept={accept:?}");
            assert_eq!(
                response
                    .headers()
                    .get("vary")
                    .and_then(|value| value.to_str().ok()),
                Some("Accept")
            );
        }
        // The canonical task route rejects the field instead of ignoring it.
        let response = client
            .post(format!("{url}/mesh/v1/documents/ocr"))
            .header("Accept", "application/json")
            .multipart(form(Some("text")))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        task.abort();
    }

    #[tokio::test]
    async fn transcription_accepts_openais_response_format_field() {
        // `/v1/audio/transcriptions` is the OpenAI-shaped route, and there the
        // caller names the body in the form. An OpenAI SDK asking for `text`
        // must not be told it sent an unknown field.
        let (url, task) =
            server_with("onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7")
                .await;
        let client = reqwest::Client::new();
        for (format, expected) in [
            ("text", StatusCode::SERVICE_UNAVAILABLE),
            ("json", StatusCode::SERVICE_UNAVAILABLE),
            ("srt", StatusCode::BAD_REQUEST),
            ("verbose_json", StatusCode::BAD_REQUEST),
        ] {
            let form = reqwest::multipart::Form::new()
                .text(
                    "model",
                    "onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7",
                )
                .text("response_format", format)
                .part(
                    "file",
                    reqwest::multipart::Part::bytes(tiny_wav()).file_name("clip.wav"),
                );
            let response = client
                .post(format!("{url}/v1/audio/transcriptions"))
                .multipart(form)
                .send()
                .await
                .unwrap();
            // Supported shapes get past negotiation and stop at the unloaded
            // model; the timestamp-bearing shapes are refused with a reason.
            assert_eq!(response.status(), expected, "response_format={format}");
        }
        task.abort();
    }

    #[tokio::test]
    async fn upload_over_the_cap_is_413_with_the_limit_named() {
        // A tiny configured cap exercises the stream counter without the
        // default 100 MiB: the file field exceeds it mid-read and the error
        // names the limit and its configuration knob.
        let (url, task) = serve(ServerState::new(registry(INDRA), None, 1024)).await;
        let form = reqwest::multipart::Form::new().text("model", INDRA).part(
            "file",
            reqwest::multipart::Part::bytes(vec![7u8; 2048]).file_name("big.bin"),
        );
        let response = reqwest::Client::new()
            .post(format!("{url}/mesh/v1/documents/ocr"))
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body: Value = response.json().await.unwrap();
        assert_eq!(
            body["error"]["message"],
            "file is 2048 bytes; the limit is 1 KiB (ONNX_MAX_BYTES)"
        );
        task.abort();
    }

    #[tokio::test]
    async fn human_bytes_picks_the_largest_whole_unit() {
        assert_eq!(human_bytes(1024), "1 KiB");
        assert_eq!(human_bytes(104857600), "100 MiB");
        assert_eq!(human_bytes(1536), "1536 bytes");
        assert_eq!(human_bytes(2 << 30), "2 GiB");
    }

    #[tokio::test]
    async fn ndjson_selects_streaming_only_on_the_canonical_route() {
        // `Accept: application/x-ndjson` passes negotiation on the canonical
        // route and stops at the unloaded model; the legacy alias never
        // streams, so the same header is a 406 there.
        let (url, task) = server().await;
        let client = reqwest::Client::new();
        for (path, expected) in [
            ("/mesh/v1/documents/ocr", StatusCode::SERVICE_UNAVAILABLE),
            ("/v1/documents/ocr", StatusCode::NOT_ACCEPTABLE),
        ] {
            let response = client
                .post(format!("{url}{path}"))
                .header("Accept", "application/x-ndjson")
                .multipart(ocr_form(vec![]))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), expected, "{path}");
        }
        // The body field selects NDJSON on the canonical route too; the alias
        // rejects it as a v0.2-only field.
        for (path, expected) in [
            ("/mesh/v1/documents/ocr", StatusCode::SERVICE_UNAVAILABLE),
            ("/v1/documents/ocr", StatusCode::BAD_REQUEST),
        ] {
            let response = client
                .post(format!("{url}{path}"))
                .header("Accept", "application/json")
                .multipart(ocr_form(vec![("stream", "true")]))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), expected, "{path}");
        }
        task.abort();
    }

    #[tokio::test]
    async fn legacy_alias_rejects_v2_fields_and_pages_validation_fires_first() {
        let (url, task) = server().await;
        let client = reqwest::Client::new();
        // v0.2 fields are canonical-only; every one is a 400 on the alias.
        for field in ["pages", "pages_only", "stream", "deadline_secs"] {
            let value = match field {
                "pages" => "1",
                "deadline_secs" => "120",
                _ => "true",
            };
            let response = client
                .post(format!("{url}/v1/documents/ocr"))
                .header("Accept", "application/json")
                .multipart(ocr_form(vec![(field, value)]))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
            let body: Value = response.json().await.unwrap();
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("only supported on the canonical"),
                "{body}"
            );
        }
        // Syntax, range-shape and deadline bounds are 400s before any model
        // work; the unloaded model is never the reason these fail.
        for extra in [
            vec![("pages", "a-b")],
            vec![("pages", "5-2")],
            vec![("pages", "0-3")],
            vec![("pages", "1-10"), ("max_pages", "5")],
            vec![("deadline_secs", "29")],
            vec![("deadline_secs", "901")],
        ] {
            let response = client
                .post(format!("{url}/mesh/v1/documents/ocr"))
                .header("Accept", "application/json")
                .multipart(ocr_form(extra))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        // Valid v0.2 controls pass validation; the range request stops at the
        // unloaded model, while pages_only is answered without one.
        let response = client
            .post(format!("{url}/mesh/v1/documents/ocr"))
            .header("Accept", "application/json")
            .multipart(ocr_form(vec![("pages", " 1- "), ("deadline_secs", "30")]))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let response = client
            .post(format!("{url}/mesh/v1/documents/ocr"))
            .header("Accept", "application/json")
            .multipart(ocr_form(vec![("pages_only", "true")]))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["kind"], "ocr");
        assert_eq!(body["pages"], 1);
        task.abort();
    }

    #[tokio::test]
    async fn legacy_alias_ignores_v2_selectors_in_model_defaults() {
        // A default `pages` must not collide with a legacy `max_pages`: the
        // alias filters v0.2-only defaults before merging, so a valid v0.1
        // request passes validation and stops only at the unloaded model.
        let state = ServerState::new(registry_with_pages_default(INDRA), None, DEFAULT_MAX_BYTES);
        let (url, task) = serve(state).await;
        let response = reqwest::Client::new()
            .post(format!("{url}/v1/documents/ocr"))
            .header("Accept", "application/json")
            .multipart(ocr_form(vec![("max_pages", "5")]))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        task.abort();
    }

    #[tokio::test]
    async fn pages_only_answers_image_metadata_on_a_cold_model() {
        // Metadata is document decoding only: no engine, no admission, no
        // Pdfium for images. The fixture scan is a single known-size page.
        let (url, task) = server().await;
        let png = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scan.png"),
        )
        .unwrap();
        let form = reqwest::multipart::Form::new()
            .text("model", INDRA)
            .text("pages_only", "true")
            .part(
                "file",
                reqwest::multipart::Part::bytes(png).file_name("scan.png"),
            );
        let response = reqwest::Client::new()
            .post(format!("{url}/mesh/v1/documents/ocr"))
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["model"], INDRA);
        assert_eq!(body["kind"], "ocr");
        assert_eq!(body["pages"], 1);
        assert_eq!(body["page_sizes"][0]["width"], 1653);
        assert_eq!(body["page_sizes"][0]["height"], 2337);
        task.abort();
    }

    fn sample_page(page_index: usize, text: &str) -> Page {
        Page {
            page_index,
            rotation: 0,
            vertical: false,
            text: text.into(),
            markdown: text.into(),
            source: "ocr",
            lines: vec![],
            blocks: vec![],
        }
    }

    #[test]
    fn ndjson_events_carry_the_contract_shapes() {
        let meta = meta_event(INDRA, 3, 5);
        assert_eq!(
            meta,
            json!({"type":"meta","model":INDRA,"pages":3,"page_from":3,"page_to":5})
        );
        let page = page_event(&sample_page(2, "middle page"));
        assert_eq!(page["type"], "page");
        assert_eq!(page["page"], 3, "pages are 1-based document numbers");
        assert_eq!(page["text"], "middle page");
        assert_eq!(page["markdown"], "middle page");
        assert!(page["lines"].is_array());
        assert!(page["blocks"].is_array());
        let done = done_event(3);
        assert_eq!(done, json!({"type":"done","pages":3}));
        let error = error_event("boom");
        assert_eq!(error, json!({"type":"error","error":{"message":"boom"}}));
    }

    #[tokio::test]
    async fn ndjson_body_streams_events_in_order_one_json_object_per_line() {
        // The wire machinery a live engine drives: meta first, one page event
        // per completed page in page order, done last — or a single error
        // event when processing fails after the stream started.
        async fn collect(lines: Vec<String>) -> String {
            let (tx, rx) = mpsc::channel::<StreamItem>(8);
            tokio::spawn(async move {
                for line in lines {
                    tx.send(Ok(format!("{line}\n").into())).await.unwrap();
                }
            });
            let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
            let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
            String::from_utf8(bytes.to_vec()).unwrap()
        }
        let events = vec![
            serde_json::to_string(&meta_event(INDRA, 1, 2)).unwrap(),
            serde_json::to_string(&page_event(&sample_page(0, "first"))).unwrap(),
            serde_json::to_string(&page_event(&sample_page(1, "second"))).unwrap(),
            serde_json::to_string(&done_event(2)).unwrap(),
        ];
        let body = collect(events).await;
        let lines = body.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 4, "one JSON object per line: {body}");
        let parsed = lines
            .iter()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(parsed[0]["type"], "meta");
        assert_eq!(parsed[1]["type"], "page");
        assert_eq!(parsed[1]["page"], 1);
        assert_eq!(parsed[2]["page"], 2);
        assert_eq!(parsed[3]["type"], "done");

        let failed = collect(vec![
            serde_json::to_string(&meta_event(INDRA, 1, 10)).unwrap(),
            serde_json::to_string(&error_event("page 3 failed")).unwrap(),
        ])
        .await;
        let failed = failed
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(failed[0]["type"], "meta");
        assert_eq!(failed[1]["type"], "error");
        assert_eq!(failed[1]["error"]["message"], "page 3 failed");
        assert_eq!(
            failed.len(),
            2,
            "an error event ends the stream: {failed:?}"
        );
    }
}
