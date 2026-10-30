//! Explicit compatible model bundles. A model selection never downloads a repository snapshot.

use anyhow::{Result, bail};
use mesh_llm_plugin::models::{AssetPlan, ManagedModelConfig};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Indra,
    Clip,
    Whisper,
    Depth,
}

impl Kind {
    pub fn workload(self) -> &'static str {
        match self {
            Self::Indra => "document_ocr",
            Self::Clip => "image_text_similarity",
            Self::Whisper => "speech_recognition",
            Self::Depth => "depth_estimation",
        }
    }

    pub fn operation(self) -> &'static str {
        match self {
            Self::Indra => "document_ocr",
            Self::Clip => "image_text_similarity",
            Self::Whisper => "transcription",
            Self::Depth => "depth_estimation",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Recipe {
    pub kind: Kind,
    pub plan: AssetPlan,
    pub threads: usize,
    pub math: bool,
    pub int8: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeOptions {
    #[serde(default = "default_threads")]
    threads: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndraOptions {
    #[serde(default = "default_threads")]
    threads: usize,
    #[serde(default)]
    math: bool,
}

fn default_threads() -> usize {
    4
}

impl Recipe {
    pub fn for_model(config: &ManagedModelConfig) -> Result<Self> {
        let (repo, revision) = config
            .model
            .rsplit_once('@')
            .ok_or_else(|| anyhow::anyhow!("model must include an explicit @revision"))?;
        let kind = match repo {
            "datalab-to/indra-ocr" if revision == "v1.1.0" => Kind::Indra,
            "Xenova/clip-vit-base-patch32"
                if revision == "6b7c523703ae26e39be4b03774966f8571c7f523" =>
            {
                Kind::Clip
            }
            "onnx-community/whisper-tiny"
                if revision == "ff4177021cc41f7db950912b73ea4fdf7d01d8e7" =>
            {
                Kind::Whisper
            }
            "onnx-community/depth-anything-v2-small-ONNX"
                if revision == "c3b67641fd837b2368757101311e5d21e511441e" =>
            {
                Kind::Depth
            }
            _ => bail!("unsupported model or revision: {}", config.model),
        };
        let int8 = match config.variant.as_str() {
            "fp32" => false,
            "int8" if kind != Kind::Indra => true,
            _ => bail!("unsupported model variant: {}", config.variant),
        };
        let values = serde_json::to_value(&config.options)?;
        let (threads, math) = if kind == Kind::Indra {
            let options: IndraOptions = serde_json::from_value(values)?;
            crate::ocr::validate_load_options(options.threads, options.math)?;
            (options.threads, options.math)
        } else {
            let options: RuntimeOptions = serde_json::from_value(values)?;
            if !(1..=256).contains(&options.threads) {
                bail!("threads must be between 1 and 256");
            }
            (options.threads, false)
        };
        let mut asset_files = files(kind, int8);
        if math {
            asset_files.extend(
                [
                    "mathcls.onnx",
                    "math_enc.onnx",
                    "math_dec_init.onnx",
                    "math_dec_step.onnx",
                    "math_meta.json",
                ]
                .map(String::from),
            );
        }
        let plan = AssetPlan {
            model: config.model.clone(),
            repo: config.source.repo.clone().unwrap_or_else(|| repo.into()),
            revision: config
                .source
                .revision
                .clone()
                .unwrap_or_else(|| revision.into()),
            files: asset_files,
        };
        plan.validate()?;
        Ok(Self {
            kind,
            plan,
            threads,
            math,
            int8,
        })
    }
}

fn files(kind: Kind, int8: bool) -> Vec<String> {
    let suffix = if int8 { "_int8" } else { "" };
    match kind {
        Kind::Indra => [
            "meta.json",
            "det.onnx",
            "rec.onnx",
            "script.onnx",
            "order.onnx",
            "orient.onnx",
            "rec_cjk.onnx",
            "rec_rtl.onnx",
            "rec_world.onnx",
        ]
        .map(String::from)
        .to_vec(),
        Kind::Clip => vec![
            format!("onnx/vision_model{suffix}.onnx"),
            format!("onnx/text_model{suffix}.onnx"),
            "tokenizer.json".into(),
            "config.json".into(),
            "preprocessor_config.json".into(),
            "tokenizer_config.json".into(),
        ],
        Kind::Whisper => vec![
            format!("onnx/encoder_model{suffix}.onnx"),
            format!("onnx/decoder_model_merged{suffix}.onnx"),
            "tokenizer.json".into(),
            "config.json".into(),
            "preprocessor_config.json".into(),
            "tokenizer_config.json".into(),
            "generation_config.json".into(),
        ],
        Kind::Depth => {
            // The int8 export publishes 4-bit `MatMulNBits` weights as
            // `model_quantized`, not `model_int8`, and the graph refers to its
            // external-data file by that same name.
            let depth = if int8 { "model_quantized" } else { "model" };
            vec![
                format!("onnx/{depth}.onnx"),
                format!("onnx/{depth}.onnx_data"),
                "config.json".into(),
                "preprocessor_config.json".into(),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(model: &str) -> ManagedModelConfig {
        serde_json::from_value(serde_json::json!({"model":model,"provider":"onnx"})).unwrap()
    }

    #[test]
    fn depth_requires_graph_and_external_data_for_each_variant() {
        for int8 in [false, true] {
            let files = files(Kind::Depth, int8);
            let graph = files.iter().find(|file| file.ends_with(".onnx")).unwrap();
            assert!(files.contains(&format!("{graph}_data")));
            assert!(files.len() < 10);
        }
    }

    #[test]
    fn depth_plans_the_names_the_repository_actually_publishes() {
        // Planning a renamed graph would ask the host to download a file that
        // does not exist, and the external-data graph names its weights file
        // internally, so a renamed copy would not load either.
        assert!(files(Kind::Depth, false).contains(&"onnx/model.onnx".to_string()));
        assert!(files(Kind::Depth, false).contains(&"onnx/model.onnx_data".to_string()));
        assert!(files(Kind::Depth, true).contains(&"onnx/model_quantized.onnx".to_string()));
        assert!(files(Kind::Depth, true).contains(&"onnx/model_quantized.onnx_data".to_string()));
        assert!(
            !files(Kind::Depth, true)
                .iter()
                .any(|file| file.contains("model_int8"))
        );
    }

    #[test]
    fn every_other_task_keeps_the_int8_file_convention() {
        for kind in [Kind::Clip, Kind::Whisper] {
            assert!(
                files(kind, false)
                    .iter()
                    .all(|file| !file.contains("_int8"))
            );
            assert!(
                files(kind, true)
                    .iter()
                    .any(|file| file.contains("_int8.onnx"))
            );
        }
    }

    #[test]
    fn rejects_unsupported_revisions_variants_and_model_knobs_before_acquisition() {
        assert!(Recipe::for_model(&config("datalab-to/indra-ocr@main")).is_err());
        let mut config = config("datalab-to/indra-ocr@v1.1.0");
        assert_eq!(Recipe::for_model(&config).unwrap().threads, 4);
        config.variant = "int8".into();
        assert!(Recipe::for_model(&config).is_err());
        config.variant = "fp32".into();
        config.options.insert("temperature".into(), 1.into());
        assert!(Recipe::for_model(&config).is_err());
    }
}
