//! Bounded multipart task surface for configured ONNX models.

use crate::{
    adapters::OutputResolution,
    recipes::Kind,
    registry::{Engine, Registry},
    representation::Representation,
    requests::{self, RequestOptions},
};
use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
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
    time::Duration,
};

const MAX_FILE: usize = 64 * 1024 * 1024;
static UPLOAD_PERMITS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);
static VALIDATION_PERMITS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);
static TEXT_PDF_PERMITS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(4);
#[derive(Clone)]
pub struct ServerState {
    pub registry: Arc<Registry>,
    pub pdfium_library: Option<PathBuf>,
}

pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/mesh/v1/documents/ocr", post(ocr))
        .route("/v1/audio/transcriptions", post(transcription))
        .route("/mesh/v1/images/similarity", post(similarity))
        .route("/mesh/v1/images/depth", post(depth))
        .layer(DefaultBodyLimit::max(MAX_FILE + 64 * 1024))
        .with_state(state)
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    let data = state.registry.models.values().map(|model| {
        let state = model.state();
        json!({"id":model.config.model,"object":"model","variant":model.config.variant,
            "assets_ready":model.assets_ready(),"workload_class":model.recipe.kind.workload(),"operations":[model.recipe.kind.operation()],
            "availability":if state == "unloaded" {"loadable"} else {state}})
    }).collect::<Vec<_>>();
    Json(json!({"object":"list","data":data}))
}

struct Upload {
    model: String,
    file: Vec<u8>,
    fields: BTreeMap<String, String>,
}
async fn read_upload(mut multipart: Multipart) -> Result<Upload, ApiError> {
    let mut file = None;
    let mut fields = BTreeMap::new();
    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(ApiError::bad_request)?
    {
        let name = field
            .name()
            .ok_or_else(|| ApiError::bad_request("multipart field name required"))?
            .to_string();
        if name == "file" {
            if file.is_some() {
                return Err(ApiError::bad_request("duplicate file field"));
            }
            let bytes = field.bytes().await.map_err(ApiError::bad_request)?;
            if bytes.is_empty() || bytes.len() > MAX_FILE {
                return Err(ApiError::bad_request(
                    "file must contain between 1 byte and 64 MiB",
                ));
            }
            file = Some(bytes.to_vec());
        } else {
            if fields.len() >= 32 || fields.contains_key(&name) {
                return Err(ApiError::bad_request(
                    "too many or duplicate multipart fields",
                ));
            }
            let mut value = Vec::new();
            while let Some(chunk) = field.chunk().await.map_err(ApiError::bad_request)? {
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
    task(state, headers, multipart, Kind::Indra).await
}
async fn transcription(
    State(state): State<ServerState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    task(state, headers, multipart, Kind::Whisper).await
}
async fn similarity(
    State(state): State<ServerState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    task(state, headers, multipart, Kind::Clip).await
}
async fn depth(
    State(state): State<ServerState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    task(state, headers, multipart, Kind::Depth).await
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
) -> Result<Response, ApiError> {
    let _upload_permit = UPLOAD_PERMITS
        .try_acquire()
        .map_err(|_| ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "upload capacity is full"))?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    let supported: &[Representation] = match kind {
        Kind::Indra => &[
            Representation::Json,
            Representation::Text,
            Representation::Markdown,
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
    let mut upload = tokio::time::timeout_at(deadline, read_upload(multipart))
        .await
        .map_err(|_| ApiError::new(StatusCode::GATEWAY_TIMEOUT, "upload deadline exceeded"))??;
    // OpenAI's body field takes precedence even over an unacceptable Accept.
    // Remove it before validating the task's own options.
    drop(_upload_permit);
    let format = if kind == Kind::Whisper
        && let Some(requested) = upload.fields.remove("response_format")
    {
        Representation::from_openai_response_format(&requested).map_err(ApiError::bad_request)?
    } else {
        Representation::negotiate(Some(&accept), supported)
            .map_err(|error| ApiError::new(StatusCode::NOT_ACCEPTABLE, error))?
    };
    let model = state
        .registry
        .select(&upload.model, kind)
        .map_err(ApiError::bad_request)?;
    let options = requests::merge(
        kind,
        &model.config.request_defaults,
        std::mem::take(&mut upload.fields),
    )
    .map_err(ApiError::bad_request)?;
    let permit = VALIDATION_PERMITS.try_acquire().map_err(|_| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "validation capacity is full",
        )
    })?;
    let (upload, options) = tokio::time::timeout_at(
        deadline,
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            match &options {
                RequestOptions::Ocr(options) => crate::pdf::validate_input(&upload.file, options)
                    .map_err(ApiError::processing)?,
                RequestOptions::Whisper(_) => crate::adapters::whisper::validate_wav(&upload.file)
                    .map_err(ApiError::bad_request)?,
                _ => {
                    decode_image(&upload.file).map_err(ApiError::bad_request)?;
                }
            }
            Ok::<_, ApiError>((upload, options))
        }),
    )
    .await
    .map_err(|_| ApiError::new(StatusCode::GATEWAY_TIMEOUT, "validation deadline exceeded"))?
    .map_err(ApiError::internal)??;
    let cancel = CancelOnDrop(Arc::new(AtomicBool::new(false)));
    let cancellation = cancel.0.clone();
    let result = tokio::time::timeout_at(deadline, async move {
        if let RequestOptions::Ocr(pdf_options) = &options
            && pdf_options.mode == crate::ocr::document::OcrMode::Text
        {
            let text_permit = TEXT_PDF_PERMITS.try_acquire().map_err(|_| {
                ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "PDF worker capacity is full",
                )
            })?;
            return tokio::task::spawn_blocking(move || {
                let _text_permit = text_permit;
                let RequestOptions::Ocr(options) = options else {
                    unreachable!()
                };
                let mut result = serde_json::to_value(crate::pdf::process_cancellable(
                    &upload.file,
                    state.pdfium_library.as_deref(),
                    &options,
                    None,
                    &cancellation,
                )?)?;
                result["model"] = Value::String(upload.model);
                Ok::<_, anyhow::Error>(Payload::Value(result))
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
            execute(
                engine,
                options,
                &upload.file,
                state.pdfium_library.as_deref(),
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
    pdfium: Option<&std::path::Path>,
    cancel: &AtomicBool,
    model: &str,
    format: Representation,
) -> Result<Payload> {
    match (engine, options) {
        (Engine::Indra(engine), RequestOptions::Ocr(options)) => {
            let mut result = serde_json::to_value(crate::pdf::process_cancellable(
                file,
                pdfium,
                &options,
                Some(engine),
                cancel,
            )?)?;
            result["model"] = Value::String(model.into());
            Ok(Payload::Value(result))
        }
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
            Representation::Json => serde_json::to_vec(&value).map_err(ApiError::internal)?,
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
    async fn server() -> (String, tokio::task::JoinHandle<()>) {
        let config: ManagedModelConfig = serde_json::from_value(json!({"model":"datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b","provider":"onnx","lifecycle":{"load":"on_demand","prefetch":false}})).unwrap();
        let registry = Arc::new(Registry::new(vec![config]).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(
                listener,
                router(ServerState {
                    registry,
                    pdfium_library: None,
                }),
            )
            .await
            .unwrap();
        });
        (url, task)
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
        assert!(
            reqwest::get(format!("{url}/health"))
                .await
                .unwrap()
                .status()
                .is_success()
        );
        task.abort();
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

    async fn server_with(model: &str) -> (String, tokio::task::JoinHandle<()>) {
        let config: ManagedModelConfig = serde_json::from_value(json!({"model":model,"provider":"onnx","lifecycle":{"load":"on_demand","prefetch":false}})).unwrap();
        let registry = Arc::new(Registry::new(vec![config]).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(
                listener,
                router(ServerState {
                    registry,
                    pdfium_library: None,
                }),
            )
            .await
            .unwrap();
        });
        (url, task)
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
            let mut form = reqwest::multipart::Form::new()
                .text(
                    "model",
                    "datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b",
                )
                .part(
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
            .text(
                "model",
                "datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b",
            )
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
}
