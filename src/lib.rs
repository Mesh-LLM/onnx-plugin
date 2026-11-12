//! Managed ONNX model providers for Mesh's typed inference tasks.

pub mod adapters;
pub mod ocr;
mod pdf;
mod recipes;
mod registry;
mod representation;
mod requests;
mod residency;
mod server;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use mesh_llm_plugin::{
    PluginMetadata, PluginRuntime, PluginStartupPolicy, capability, plugin_server_info,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Parser)]
struct Args {
    /// Pdfium dynamic library for PDF input. Image input needs no Pdfium.
    #[arg(long, env = "ONNX_PDFIUM_LIBRARY")]
    pdfium_library: Option<PathBuf>,
}

async fn run_async(args: Args) -> Result<()> {
    let registry = Arc::new(registry::Registry::new(
        mesh_llm_plugin::models::configured_models_from_env()?,
    )?);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind OCR provider")?;
    let address = listener.local_addr()?;
    let url = format!("http://{address}");
    let server_registry = registry.clone();
    tokio::spawn(async move {
        if let Err(error) = axum::serve(
            listener,
            server::router(server::ServerState {
                registry: server_registry,
                pdfium_library: args.pdfium_library,
            }),
        )
        .await
        {
            tracing::error!(%error, "ONNX HTTP server stopped");
        }
    });
    let health_url = format!("{url}/health");
    let plugin = mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            "onnx", VERSION,
            plugin_server_info(
                "mesh-onnx", VERSION, "Managed ONNX inference",
                "OCR, transcription, image/text similarity and relative depth using ONNX Runtime.",
                Some("Select provider = onnx under [[models]] in Mesh configuration; Pdfium enables PDF input."),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/document_ocr_v1"),
            capability("endpoint:inference/transcription_v1"),
            capability("endpoint:inference/image_text_similarity_v1"),
            capability("endpoint:inference/depth_estimation_v1"),
            capability("backend:onnx"),
        ],
        inference: [
            mesh_llm_plugin::inference::provider("ocr", url.clone())
                .protocol("document_ocr_v1")
                .supports_streaming(false),
            mesh_llm_plugin::inference::provider("transcription", url.clone())
                .protocol("transcription_v1").supports_streaming(false),
            mesh_llm_plugin::inference::provider("similarity", url.clone())
                .protocol("image_text_similarity_v1").supports_streaming(false),
            mesh_llm_plugin::inference::provider("depth", url)
                .protocol("depth_estimation_v1").supports_streaming(false),
        ],
        health: move |_context| {
            let health_url = health_url.clone();
            Box::pin(async move {
                let response = reqwest::Client::builder()
                    .timeout(Duration::from_secs(2))
                    .build()?
                    .get(health_url)
                    .send()
                    .await?.error_for_status()?;
                Ok(format!("ONNX HTTP status {}", response.status()))
            })
        },
        on_initialized: move |context| {
            let registry = registry.clone();
            Box::pin(async move { registry.initialize(context.owned()) })
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
