//! Loopback HTTP implementation of the Mesh document_ocr_v1 provider contract.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use image::GrayImage;
use pdfium_render::prelude::*;
use serde_json::json;

use crate::ocr::{Document, Engine};

const MODEL: &str = "datalab/indra-ocr";
const MAX_FILE: usize = 64 * 1024 * 1024;
const MAX_BODY: usize = MAX_FILE + 64 * 1024;
const MAX_PAGES: usize = 1000;
const MAX_PAGE_DIMENSION: i32 = 4096;
static PDFIUM_BIND_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
enum DocumentError {
    ImageDecode(image::ImageError),
    PdfOpen(PdfiumError),
    PdfiumUnavailable(PdfiumError),
}

impl std::fmt::Display for DocumentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ImageDecode(_) => "decode document image",
            Self::PdfOpen(_) => "open uploaded PDF",
            Self::PdfiumUnavailable(_) => "load Pdfium library for PDF rendering",
        })
    }
}

impl std::error::Error for DocumentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::ImageDecode(error) => error,
            Self::PdfOpen(error) | Self::PdfiumUnavailable(error) => error,
        })
    }
}

#[derive(Clone)]
pub struct ServerState {
    engine: Arc<Mutex<Engine>>,
    pdfium_library: Option<PathBuf>,
}

impl ServerState {
    pub fn new(engine: Engine, pdfium_library: Option<PathBuf>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            pdfium_library,
        }
    }
}

pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/v1/documents/ocr", post(ocr))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state)
}

async fn models() -> impl IntoResponse {
    Json(json!({"object": "list", "data": [{"id": MODEL, "object": "model"}]}))
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

struct Upload {
    file: Vec<u8>,
    model: Option<String>,
    response_format: String,
    max_pages: usize,
}

async fn read_upload(mut multipart: Multipart) -> Result<Upload, ApiError> {
    let mut upload = Upload {
        file: Vec::new(),
        model: None,
        response_format: "json".into(),
        max_pages: 0,
    };
    let mut seen_file = false;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(ApiError::bad_request)?
    {
        match field.name() {
            Some("file") if !seen_file => {
                seen_file = true;
                upload.file = field.bytes().await.map_err(ApiError::bad_request)?.to_vec();
            }
            Some("model") if upload.model.is_none() => {
                upload.model = Some(field.text().await.map_err(ApiError::bad_request)?);
            }
            Some("response_format") => {
                upload.response_format = field.text().await.map_err(ApiError::bad_request)?;
            }
            Some("max_pages") => {
                upload.max_pages = field
                    .text()
                    .await
                    .map_err(ApiError::bad_request)?
                    .parse()
                    .map_err(ApiError::bad_request)?;
            }
            _ => {
                return Err(ApiError::bad_request(
                    "unexpected or duplicate multipart field",
                ));
            }
        }
    }
    if !seen_file || upload.file.is_empty() {
        return Err(ApiError::bad_request("file field is required"));
    }
    if upload.file.len() > MAX_FILE {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "document exceeds 64 MiB",
        ));
    }
    if upload.model.as_deref().unwrap_or(MODEL) != MODEL {
        return Err(ApiError::bad_request("unsupported OCR model"));
    }
    if !matches!(
        upload.response_format.as_str(),
        "json" | "text" | "markdown"
    ) {
        return Err(ApiError::bad_request(
            "response_format must be json, text, or markdown",
        ));
    }
    if upload.max_pages > MAX_PAGES {
        return Err(ApiError::bad_request(
            "max_pages must be between 0 and 1000",
        ));
    }
    Ok(upload)
}

async fn ocr(State(state): State<ServerState>, multipart: Multipart) -> Result<Response, ApiError> {
    let upload = read_upload(multipart).await?;
    let response_format = upload.response_format.clone();
    let document = tokio::task::spawn_blocking(move || process_upload(&state, upload))
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::processing)?;
    let joined = |markdown: bool| {
        document
            .pages
            .iter()
            .map(|page| {
                if markdown {
                    page.markdown.as_str()
                } else {
                    page.text.as_str()
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    Ok(match response_format.as_str() {
        "text" => (
            [("content-type", "text/plain; charset=utf-8")],
            joined(false),
        )
            .into_response(),
        "markdown" => (
            [("content-type", "text/markdown; charset=utf-8")],
            joined(true),
        )
            .into_response(),
        _ => Json(document).into_response(),
    })
}

fn process_upload(state: &ServerState, upload: Upload) -> Result<Document> {
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| anyhow::anyhow!("OCR engine lock poisoned"))?;
    decode_pages(
        &upload.file,
        upload.max_pages,
        state.pdfium_library.as_deref(),
        |pages| engine.recognize_document(pages),
    )
}

fn decode_pages<T>(
    data: &[u8],
    max_pages: usize,
    pdfium_library: Option<&std::path::Path>,
    recognize: impl FnOnce(&mut dyn Iterator<Item = Result<GrayImage>>) -> Result<T>,
) -> Result<T> {
    if data.starts_with(b"%PDF-") {
        return render_pdf(data, max_pages, pdfium_library, recognize);
    }
    let image = image::load_from_memory(data)
        .map_err(DocumentError::ImageDecode)?
        .into_luma8();
    recognize(&mut std::iter::once(Ok(image)))
}

fn render_pdf<T>(
    data: &[u8],
    max_pages: usize,
    library: Option<&std::path::Path>,
    recognize: impl FnOnce(&mut dyn Iterator<Item = Result<GrayImage>>) -> Result<T>,
) -> Result<T> {
    let _bind_lock = PDFIUM_BIND_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("Pdfium binding lock poisoned"))?;
    let bindings = match library {
        Some(path) => Pdfium::bind_to_library(path),
        None => Pdfium::bind_to_system_library(),
    };
    let pdfium = match bindings {
        Ok(bindings) => Pdfium::new(bindings),
        Err(PdfiumError::PdfiumLibraryBindingsAlreadyInitialized) => Pdfium::default(),
        Err(error) => return Err(DocumentError::PdfiumUnavailable(error).into()),
    };
    drop(_bind_lock);
    let document = pdfium
        .load_pdf_from_byte_vec(data.to_vec(), None)
        .map_err(DocumentError::PdfOpen)?;
    let count = page_limit(max_pages).min(document.pages().len() as usize);
    let config = PdfRenderConfig::new()
        .scale_page_by_factor(200.0 / 72.0)
        .set_maximum_width(MAX_PAGE_DIMENSION)
        .set_maximum_height(MAX_PAGE_DIMENSION);
    let mut pages = document.pages().iter().take(count).map(|page| {
        let bitmap = page
            .render_with_config(&config)
            .context("render PDF page")?;
        Ok(bitmap
            .as_image()
            .context("decode rendered PDF bitmap")?
            .into_luma8())
    });
    recognize(&mut pages)
}

fn page_limit(max_pages: usize) -> usize {
    if max_pages == 0 {
        MAX_PAGES
    } else {
        max_pages.min(MAX_PAGES)
    }
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn bad_request(error: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::BAD_REQUEST, error.to_string())
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }

    fn processing(error: anyhow::Error) -> Self {
        let status = match error.downcast_ref::<DocumentError>() {
            Some(DocumentError::ImageDecode(_) | DocumentError::PdfOpen(_)) => {
                StatusCode::BAD_REQUEST
            }
            Some(DocumentError::PdfiumUnavailable(_)) => StatusCode::SERVICE_UNAVAILABLE,
            None => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, format!("{error:#}"))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({"error": {"message": self.message}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processing_errors_have_distinct_statuses() {
        let error = decode_pages(b"invalid image", 0, None, |_| Ok(())).unwrap_err();
        let error = ApiError::processing(error.context("processing upload"));
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(error.message.contains("decode document image"));

        for (error, expected) in [
            (
                anyhow::Error::from(DocumentError::PdfOpen(PdfiumError::NoPagesInDocument)),
                StatusCode::BAD_REQUEST,
            ),
            (
                DocumentError::PdfiumUnavailable(PdfiumError::UnrecognizedPath).into(),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                PdfiumError::UnknownBitmapFormat.into(),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                anyhow::anyhow!("OCR engine lock poisoned"),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ] {
            assert_eq!(ApiError::processing(error).status, expected);
        }
    }

    #[test]
    fn image_pages_propagate_recognition_failures_as_server_errors() {
        let data = include_bytes!("../tests/fixtures/scan.png");
        let error = decode_pages(data, 0, None, |pages| -> Result<()> {
            let image = pages.next().expect("image page")?;
            assert!(image.width() > 100 && image.height() > 100);
            assert!(pages.next().is_none());
            anyhow::bail!("recognition runtime failed")
        })
        .unwrap_err();
        assert_eq!(
            ApiError::processing(error).status,
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    // A minimal blank PDF lets the rendering tests vary dimensions and page counts.
    fn blank_pdf(sizes: &[(u32, u32)]) -> Vec<u8> {
        use std::fmt::Write;

        let mut pdf = String::from("%PDF-1.7\n");
        let mut offsets = vec![0];
        let kids = (0..sizes.len())
            .map(|index| format!("{} 0 R", index + 3))
            .collect::<Vec<_>>()
            .join(" ");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            format!("<< /Type /Pages /Count {} /Kids [{kids}] >>", sizes.len()),
        ]
        .into_iter()
        .chain(sizes.iter().map(|(width, height)| {
            format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] >>")
        }));
        for (index, object) in objects.enumerate() {
            offsets.push(pdf.len());
            writeln!(pdf, "{} 0 obj\n{object}\nendobj", index + 1).unwrap();
        }
        let xref = pdf.len();
        writeln!(pdf, "xref\n0 {}\n0000000000 65535 f ", offsets.len()).unwrap();
        for offset in &offsets[1..] {
            writeln!(pdf, "{offset:010} 00000 n ").unwrap();
        }
        writeln!(
            pdf,
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF",
            offsets.len()
        )
        .unwrap();
        pdf.into_bytes()
    }

    #[test]
    #[ignore = "requires a Pdfium library"]
    fn pdfium_bounds_pages_and_stops_rendering_on_recognition_failure() {
        let library = PathBuf::from(std::env::var("ONNX_OCR_PDFIUM_LIBRARY").unwrap());
        let dimensions = decode_pages(
            &blank_pdf(&[(9000, 1000), (1000, 9000)]),
            0,
            Some(&library),
            |pages| {
                pages
                    .map(|page| Ok(page?.dimensions()))
                    .collect::<Result<Vec<_>>>()
            },
        )
        .unwrap();
        assert_eq!(dimensions.len(), 2);
        assert_eq!(dimensions[0].0, MAX_PAGE_DIMENSION as u32);
        assert_eq!(dimensions[1].1, MAX_PAGE_DIMENSION as u32);
        assert!(dimensions.iter().all(|&(width, height)| {
            width <= MAX_PAGE_DIMENSION as u32 && height <= MAX_PAGE_DIMENSION as u32
        }));

        for (limit, expected) in [(0, MAX_PAGES), (2, 2)] {
            let count = decode_pages(
                &blank_pdf(&vec![(1, 1); MAX_PAGES + 1]),
                limit,
                Some(&library),
                |pages| pages.map(|page| page.map(|_| 1)).sum::<Result<usize>>(),
            )
            .unwrap();
            assert_eq!(count, expected);
        }

        // Recognition can stop after consuming the first page.
        let error = decode_pages(
            &blank_pdf(&[(20, 20), (20, 20)]),
            0,
            Some(&library),
            |pages| -> Result<()> {
                pages.next().expect("first page")?;
                anyhow::bail!("stop recognition")
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "stop recognition");
        assert_eq!(
            ApiError::processing(error).status,
            StatusCode::INTERNAL_SERVER_ERROR
        );

        let error = decode_pages(b"%PDF-invalid", 0, Some(&library), |_| Ok(())).unwrap_err();
        assert_eq!(ApiError::processing(error).status, StatusCode::BAD_REQUEST);
    }

    #[test]
    #[ignore = "requires a Pdfium library"]
    fn pdfium_renders_uploaded_pdf() {
        let library = std::env::var("ONNX_OCR_PDFIUM_LIBRARY").expect("ONNX_OCR_PDFIUM_LIBRARY");
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/scan.pdf"
        ))
        .expect("read PDF");
        let dimensions = decode_pages(&data, 1, Some(std::path::Path::new(&library)), |pages| {
            pages
                .map(|page| Ok(page?.dimensions()))
                .collect::<Result<Vec<_>>>()
        })
        .expect("render PDF");
        assert_eq!(dimensions.len(), 1);
        assert!(dimensions[0].0 > 100 && dimensions[0].1 > 100);
    }

    #[tokio::test]
    #[ignore = "requires Indra v1.1.0 model assets and Pdfium"]
    async fn native_provider_serves_document_http_contract() {
        let model_dir = std::env::var("ONNX_OCR_MODEL_DIR").expect("ONNX_OCR_MODEL_DIR");
        let engine = Engine::load(std::path::Path::new(&model_dir)).expect("load ONNX sessions");
        let pdfium = std::fs::canonicalize(
            std::env::var("ONNX_OCR_PDFIUM_LIBRARY").expect("ONNX_OCR_PDFIUM_LIBRARY"),
        )
        .expect("Pdfium path");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        let task = tokio::spawn(async move {
            axum::serve(listener, router(ServerState::new(engine, Some(pdfium)))).await
        });
        for file in ["scan.png", "scan.pdf"] {
            let path = format!("{}/tests/fixtures/{file}", env!("CARGO_MANIFEST_DIR"));
            let form = reqwest::multipart::Form::new().text("model", MODEL).part(
                "file",
                reqwest::multipart::Part::bytes(std::fs::read(path).expect("sample"))
                    .file_name(file),
            );
            let response = reqwest::Client::new()
                .post(format!("http://{address}/v1/documents/ocr"))
                .multipart(form)
                .send()
                .await
                .expect("OCR response");
            let status = response.status();
            if status != StatusCode::OK {
                panic!(
                    "{file}: {status}: {}",
                    response.text().await.expect("error body")
                );
            }
            let body: serde_json::Value = response.json().await.expect("JSON document");
            assert_eq!(body["model"], MODEL, "{file}");
            assert_eq!(body["pages"][0]["page_index"], 0, "{file}");
            assert!(
                body["pages"][0]["text"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("대한성공회 서울교구"),
                "{file}"
            );
            assert!(
                body["pages"][0]["markdown"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("대한성공회 서울교구"),
                "{file}"
            );
        }
        task.abort();
    }
}
