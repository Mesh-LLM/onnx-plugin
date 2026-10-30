//! Configured models, host-owned acquisition and independently leased engines.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{
    PluginContext,
    models::{ManagedModelConfig, ModelLoadPolicy, PreparedModelAssets},
};
use tokio::sync::OnceCell;

use crate::{
    adapters, ocr,
    recipes::{Kind, Recipe},
    residency::{EngineLease, Residency},
};

pub enum Engine {
    Indra(Box<ocr::Engine>),
    Other(adapters::Engine),
}

pub struct Model {
    pub config: ManagedModelConfig,
    pub recipe: Recipe,
    pub residency: Residency<Engine>,
    assets: OnceCell<PreparedModelAssets>,
    context: Arc<OnceLock<PluginContext<'static>>>,
    failure: Arc<Mutex<Option<Instant>>>,
}

pub struct Registry {
    pub models: BTreeMap<String, Arc<Model>>,
    context: Arc<OnceLock<PluginContext<'static>>>,
}

impl Registry {
    pub fn new(configs: Vec<ManagedModelConfig>) -> Result<Self> {
        let context = Arc::new(OnceLock::new());
        let mut models = BTreeMap::new();
        for config in configs {
            let recipe = Recipe::for_model(&config)?;
            crate::requests::validate_defaults(recipe.kind, &config.request_defaults)?;
            let residency =
                Residency::new(Duration::from_secs(config.lifecycle.idle_timeout_secs), 32);
            let id = config.model.clone();
            let model = Arc::new(Model {
                config,
                recipe,
                residency,
                assets: OnceCell::new(),
                context: context.clone(),
                failure: Arc::new(Mutex::new(None)),
            });
            if models.insert(id.clone(), model).is_some() {
                bail!("duplicate model id: {id}");
            }
        }
        Ok(Self { models, context })
    }

    pub fn initialize(self: &Arc<Self>, context: PluginContext<'static>) -> Result<()> {
        self.context
            .set(context)
            .map_err(|_| anyhow::anyhow!("model registry already initialized"))?;
        for model in self.models.values() {
            let model = model.clone();
            tokio::spawn(async move {
                let result = if model.config.lifecycle.load == ModelLoadPolicy::Startup {
                    model.acquire().await.map(drop)
                } else if model.config.lifecycle.prefetch {
                    model.assets().await.map(|_| ())
                } else {
                    Ok(())
                };
                if let Err(error) = result {
                    tracing::warn!(model = %model.config.model, error = %error, "model preparation failed");
                }
            });
        }
        let registry = self.clone();
        tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(1));
            loop {
                timer.tick().await;
                for model in registry.models.values() {
                    let model = model.clone();
                    tokio::task::spawn_blocking(move || model.residency.evict_idle(Instant::now()));
                }
            }
        });
        Ok(())
    }

    pub fn select(&self, id: &str, kind: Kind) -> Result<Arc<Model>> {
        let model = self
            .models
            .get(id)
            .with_context(|| format!("model is not configured: {id}"))?;
        if model.recipe.kind != kind {
            bail!("model does not support this task");
        }
        Ok(model.clone())
    }
}

impl Model {
    pub fn state(&self) -> &'static str {
        if self.in_backoff() {
            "failed"
        } else {
            self.residency.state()
        }
    }

    fn in_backoff(&self) -> bool {
        self.failure.lock().map_or(true, |failure| {
            failure.is_some_and(|at| at.elapsed() < Duration::from_secs(5))
        })
    }

    pub async fn assets(&self) -> Result<&PreparedModelAssets> {
        self.assets
            .get_or_try_init(|| async {
                if let Some(directory) = &self.config.source.local_dir {
                    let directory =
                        std::fs::canonicalize(directory).context("open local model directory")?;
                    let mut files = BTreeMap::new();
                    for relative in &self.recipe.plan.files {
                        let path =
                            std::fs::canonicalize(directory.join(relative)).with_context(|| {
                                format!("required model asset is missing: {relative}")
                            })?;
                        if !path.starts_with(&directory) || !path.is_file() {
                            bail!("model asset escapes its bundle: {relative}");
                        }
                        files.insert(relative.clone(), path);
                    }
                    return Ok(PreparedModelAssets {
                        directory,
                        files,
                        revision: self.recipe.plan.revision.clone(),
                    });
                }
                let context = self
                    .context
                    .get()
                    .context("Mesh asset service is not connected")?;
                context.prepare_model_assets(&self.recipe.plan).await
            })
            .await
    }

    pub async fn acquire(&self) -> Result<EngineLease<Engine>> {
        let mut lease = self.residency.acquire().await?;
        if lease.engine().is_none() {
            if self.in_backoff() {
                bail!("model load failed recently; retry after 5 seconds");
            }
            let directory: PathBuf = self
                .assets()
                .await
                .inspect_err(|_| {
                    if let Ok(mut failure) = self.failure.lock() {
                        *failure = Some(Instant::now());
                    }
                })?
                .directory
                .clone();
            let recipe = self.recipe.clone();
            let failure = self.failure.clone();
            lease = tokio::task::spawn_blocking(move || {
                let result = load_engine(&recipe, &directory);
                if let Ok(mut failure) = failure.lock() {
                    *failure = result.as_ref().err().map(|_| Instant::now());
                }
                let engine = result?;
                *lease.engine() = Some(engine);
                Ok::<_, anyhow::Error>(lease)
            })
            .await
            .context("model loading worker failed")??;
        }
        Ok(lease)
    }
}

fn load_engine(recipe: &Recipe, directory: &Path) -> Result<Engine> {
    let kind = match recipe.kind {
        Kind::Indra => {
            return Ok(Engine::Indra(Box::new(ocr::Engine::load_with_options(
                directory,
                recipe.threads,
                recipe.math,
            )?)));
        }
        Kind::Clip => adapters::ModelKind::Clip,
        Kind::Whisper => adapters::ModelKind::Whisper,
        Kind::Depth => adapters::ModelKind::Depth,
    };
    let variant = if recipe.int8 {
        adapters::Variant::Int8
    } else {
        adapters::Variant::Fp32
    };
    Ok(Engine::Other(adapters::Engine::load(
        kind,
        directory,
        recipe.threads,
        variant,
    )?))
}
