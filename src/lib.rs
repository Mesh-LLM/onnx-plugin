//! Mesh plugin registration for the native Indra OCR provider.

mod ocr;
mod server;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use mesh_llm_plugin::{
    PluginMetadata, PluginRuntime, PluginStartupPolicy, capability, plugin_server_info,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const ENDPOINT: &str = "indra-ocr";

#[derive(Debug, Parser)]
struct Args {
    /// Folder containing Indra v1.1.0 det.onnx, rec.onnx, and meta.json.
    #[arg(long, env = "ONNX_OCR_MODEL_DIR")]
    model_dir: PathBuf,
    /// Pdfium dynamic library for PDF input. Image input needs no Pdfium.
    #[arg(long, env = "ONNX_OCR_PDFIUM_LIBRARY")]
    pdfium_library: Option<PathBuf>,
}

async fn run_async(args: Args) -> Result<()> {
    let engine = ocr::Engine::load(&args.model_dir).context("load Indra ONNX models")?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind OCR provider")?;
    let address = listener.local_addr()?;
    let url = format!("http://{address}");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(
            listener,
            server::router(server::ServerState::new(engine, args.pdfium_library)),
        )
        .await
        {
            eprintln!("onnx-ocr HTTP server: {error}");
        }
    });
    let health_url = format!("{url}/v1/models");
    let plugin = mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            "onnx-ocr", VERSION,
            plugin_server_info(
                "mesh-onnx-ocr", VERSION, "Indra ONNX document OCR",
                "Serves document OCR using Rust and ONNX Runtime through Mesh document_ocr_v1.",
                Some("Pass --model-dir with the Indra v1.1.0 model assets; --pdfium-library enables PDF input."),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/document_ocr_v1"),
            capability("backend:indra-ocr"),
        ],
        inference: [
            mesh_llm_plugin::inference::provider(ENDPOINT, url)
                .protocol("document_ocr_v1")
                .supports_streaming(false),
        ],
        health: move |_context| {
            let health_url = health_url.clone();
            Box::pin(async move {
                let response = reqwest::Client::builder()
                    .timeout(Duration::from_secs(2))
                    .build()?
                    .get(health_url)
                    .send()
                    .await?;
                Ok(format!("OCR HTTP status {}", response.status()))
            })
        },
    };
    PluginRuntime::run(plugin).await
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run_async(args))
}
