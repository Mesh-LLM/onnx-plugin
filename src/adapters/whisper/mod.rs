//! Whisper Tiny encoder/merged-decoder transcription with request-local KV state.

mod audio;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use ort::{
    session::Session,
    value::{DynValue, Tensor},
};
use serde::Serialize;
use tokenizers::Tokenizer;

use super::{Variant, graph, read_json, session};

/// A completed short-audio transcription. Streaming/timestamps are not advertised.
#[derive(Debug, Clone, Serialize)]
pub struct Transcription {
    pub text: String,
    pub language: String,
    pub generated_tokens: usize,
}

/// Validate a bounded supported WAV upload before acquiring model residency.
///
/// Supports 16 kHz PCM16 or float32, mono/stereo, up to 30 seconds.
pub fn validate_wav(bytes: &[u8]) -> Result<()> {
    audio::decode_wav(bytes).map(|_| ())
}

struct Generation {
    start: i64,
    eos: i64,
    transcribe: i64,
    no_timestamps: i64,
    languages: BTreeMap<String, i64>,
    suppress: BTreeSet<i64>,
    begin_suppress: BTreeSet<i64>,
    max_positions: usize,
    layers: usize,
    heads: usize,
    head_dim: usize,
}

pub struct Whisper {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    generation: Generation,
}

impl Whisper {
    pub fn load(directory: &Path, threads: usize, variant: Variant) -> Result<Self> {
        let config = read_json(directory, "config.json")?;
        let generation = read_json(directory, "generation_config.json")?;
        let preprocessor = read_json(directory, "preprocessor_config.json")?;
        if preprocessor["sampling_rate"].as_u64() != Some(16_000)
            || preprocessor["feature_size"].as_u64() != Some(80)
            || preprocessor["n_fft"].as_u64() != Some(400)
            || preprocessor["hop_length"].as_u64() != Some(160)
        {
            bail!(
                "unsupported Whisper feature extractor; this recipe requires 16 kHz/80 Mel/400 FFT/160 hop"
            );
        }
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("load Whisper tokenizer: {error}"))?;
        let decoder = session(&graph(directory, "decoder_model_merged", variant), threads)?;
        if !decoder
            .inputs()
            .iter()
            .any(|input| input.name() == "use_cache_branch")
        {
            bail!("Whisper recipe requires the merged decoder with use_cache_branch");
        }
        Ok(Self {
            encoder: session(&graph(directory, "encoder_model", variant), threads)?,
            decoder,
            tokenizer,
            generation: Generation::from_config(&config, &generation)?,
        })
    }

    /// Transcribe at most 30 seconds of PCM16/float32 WAV at 16 kHz.
    pub fn transcribe_wav(
        &mut self,
        wav: &[u8],
        language: &str,
        max_output_tokens: usize,
        cancel: &AtomicBool,
    ) -> Result<Transcription> {
        check_cancel(cancel)?;
        self.validate_request(language, max_output_tokens)?;
        let samples = audio::decode_wav(wav)?;
        let features = audio::log_mel(&samples, cancel)?;
        check_cancel(cancel)?;
        let encoder_outputs = self.encoder.run(ort::inputs!["input_features" => Tensor::from_array(([1, audio::MELS, audio::FRAMES], features))?])?;
        let hidden = encoder_outputs
            .get("last_hidden_state")
            .context("Whisper encoder last_hidden_state missing")?;
        let (shape, hidden) = hidden.try_extract_tensor::<f32>()?;
        if shape.as_ref() != [1, 1500, 384] || hidden.iter().any(|value| !value.is_finite()) {
            bail!("invalid Whisper Tiny encoder output: expected finite [1, 1500, 384]");
        }
        let hidden_tensor = Tensor::from_array((shape.to_vec(), hidden.to_vec()))?;
        drop(encoder_outputs);
        let mut state = DecodeState::new(&self.generation)?;
        let mut next = self.generation.start;
        let mut text_tokens = Vec::new();
        let mut detected_language = language.to_string();
        for step in 0..max_output_tokens + 3 {
            check_cancel(cancel)?;
            let logits = state.step(&mut self.decoder, next, &hidden_tensor)?;
            next = match step {
                0 => self
                    .generation
                    .language_token(language, &logits, &mut detected_language)?,
                1 => self.generation.transcribe,
                2 => self.generation.no_timestamps,
                _ => select_token(
                    &logits,
                    &self.generation.suppress,
                    (step == 3).then_some(&self.generation.begin_suppress),
                    self.generation.start - 1,
                )?,
            };
            if step >= 3 {
                if next == self.generation.eos {
                    break;
                }
                text_tokens.push(u32::try_from(next).context("invalid Whisper output token")?);
            }
        }
        check_cancel(cancel)?;
        let text = self
            .tokenizer
            .decode(&text_tokens, true)
            .map_err(|error| anyhow::anyhow!("decode Whisper tokens: {error}"))?;
        Ok(Transcription {
            text: text.trim().to_owned(),
            language: detected_language,
            generated_tokens: text_tokens.len(),
        })
    }

    fn validate_request(&self, language: &str, limit: usize) -> Result<()> {
        if limit == 0
            || limit
                .checked_add(4)
                .is_none_or(|total| total > self.generation.max_positions)
        {
            bail!(
                "max_output_tokens must be between 1 and {}",
                self.generation.max_positions - 4
            );
        }
        if language != "auto"
            && !self
                .generation
                .languages
                .contains_key(&format!("<|{language}|>"))
        {
            bail!("language must be auto or a language code supported by this Whisper model");
        }
        Ok(())
    }
}

impl Generation {
    fn from_config(config: &serde_json::Value, generation: &serde_json::Value) -> Result<Self> {
        let required = |name: &str| {
            generation[name]
                .as_i64()
                .or_else(|| config[name].as_i64())
                .with_context(|| format!("Whisper {name} missing"))
        };
        let heads = config["decoder_attention_heads"]
            .as_u64()
            .context("Whisper decoder heads missing")? as usize;
        let width = config["d_model"]
            .as_u64()
            .context("Whisper model width missing")? as usize;
        let layers = config["decoder_layers"]
            .as_u64()
            .context("Whisper decoder layers missing")? as usize;
        let max_positions = config["max_target_positions"]
            .as_u64()
            .context("Whisper decoder context missing")? as usize;
        if heads != 6 || width != 384 || layers != 4 || max_positions != 448 {
            bail!("invalid Whisper decoder configuration");
        }
        let languages = generation["lang_to_id"]
            .as_object()
            .context("Whisper language token map missing")?
            .iter()
            .map(|(name, id)| {
                Ok((
                    name.clone(),
                    id.as_i64().context("invalid Whisper language token")?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let tokens = |name: &str| -> Result<BTreeSet<i64>> {
            generation[name]
                .as_array()
                .into_iter()
                .flatten()
                .map(|id| id.as_i64().context("invalid suppressed Whisper token"))
                .collect()
        };
        let result = Self {
            start: required("decoder_start_token_id")?,
            eos: required("eos_token_id")?,
            transcribe: generation["task_to_id"]["transcribe"]
                .as_i64()
                .context("Whisper transcription task token missing")?,
            no_timestamps: required("no_timestamps_token_id")?,
            languages,
            suppress: tokens("suppress_tokens")?,
            begin_suppress: tokens("begin_suppress_tokens")?,
            max_positions,
            layers,
            heads,
            head_dim: width / heads,
        };
        let valid_token = |token: i64| (0..100_000).contains(&token);
        if !valid_token(result.start)
            || !valid_token(result.eos)
            || !valid_token(result.transcribe)
            || !valid_token(result.no_timestamps)
            || result.start <= result.eos
            || result.no_timestamps <= result.start
            || result.languages.is_empty()
            || result.languages.values().any(|token| !valid_token(*token))
        {
            bail!("invalid Whisper generation token configuration");
        }
        Ok(result)
    }

    fn language_token(&self, language: &str, logits: &[f32], detected: &mut String) -> Result<i64> {
        if language != "auto" {
            return self
                .languages
                .get(&format!("<|{language}|>"))
                .copied()
                .context("unsupported Whisper language");
        }
        let (name, token) = self
            .languages
            .iter()
            .filter(|(_, token)| **token >= 0)
            .filter_map(|(name, token)| {
                logits
                    .get(*token as usize)
                    .filter(|score| score.is_finite())
                    .map(|score| (name, *token, *score))
            })
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(name, token, _)| (name, token))
            .context("Whisper returned no usable language logits")?;
        *detected = name
            .trim_start_matches("<|")
            .trim_end_matches("|>")
            .to_owned();
        Ok(token)
    }
}

/// A zero-length cache tensor for the first decoder pass.
///
/// The merged decoder always declares `past_key_values.*` inputs, and the
/// no-cache branch reads them only through its `use_cache_branch` arm, so the
/// first pass must supply them with a zero-length sequence axis. `ort`'s tuple
/// and slice constructors reject that shape — `ToShape` refuses any dimension
/// below 1 — so the tensor is built from an owned `ndarray` array, whose
/// conversion passes the shape straight to `CreateTensorWithDataAsOrtValue`.
fn empty_cache_tensor(generation: &Generation) -> Result<Tensor<f32>> {
    Ok(Tensor::from_array(ndarray::Array4::<f32>::zeros((
        1,
        generation.heads,
        0,
        generation.head_dim,
    )))?)
}

struct DecodeState {
    cache: BTreeMap<String, DynValue>,
    has_cache: bool,
}

impl DecodeState {
    fn new(generation: &Generation) -> Result<Self> {
        let mut cache = BTreeMap::new();
        for layer in 0..generation.layers {
            for attention in ["decoder", "encoder"] {
                for kind in ["key", "value"] {
                    cache.insert(
                        format!("past_key_values.{layer}.{attention}.{kind}"),
                        empty_cache_tensor(generation)?.into_dyn(),
                    );
                }
            }
        }
        Ok(Self {
            cache,
            has_cache: false,
        })
    }

    fn step(
        &mut self,
        decoder: &mut Session,
        token: i64,
        hidden: &Tensor<f32>,
    ) -> Result<Vec<f32>> {
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array(([1, 1], vec![token]))?,
            "encoder_hidden_states" => hidden.view(),
            "use_cache_branch" => Tensor::from_array(([1], vec![self.has_cache]))?,
        ];
        for (name, value) in &self.cache {
            inputs.push((name.as_str().into(), value.view().into()));
        }
        let mut outputs = decoder.run(inputs)?;
        let (shape, logits) = outputs
            .get("logits")
            .context("Whisper decoder logits missing")?
            .try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] <= 0 || shape[2] > 100_000
        {
            bail!("invalid Whisper decoder logits shape");
        }
        let logits = logits.to_vec();
        for (name, value) in &mut self.cache {
            let suffix = name
                .strip_prefix("past_key_values.")
                .context("invalid internal cache key")?;
            let output_name = format!("present.{suffix}");
            let candidate = outputs
                .remove(&output_name)
                .with_context(|| format!("Whisper cache output {output_name} missing"))?;
            let (shape, _) = candidate.try_extract_tensor::<f32>()?;
            // The merged cached branch can emit empty encoder placeholders. Retain
            // the first-pass encoder KV instead of replacing it with that placeholder.
            if !suffix.contains(".encoder.")
                || !self.has_cache
                || shape.iter().all(|dimension| *dimension > 0)
            {
                *value = candidate;
            }
        }
        self.has_cache = true;
        Ok(logits)
    }
}

fn select_token(
    logits: &[f32],
    suppressed: &BTreeSet<i64>,
    begin_suppressed: Option<&BTreeSet<i64>>,
    max_text_token: i64,
) -> Result<i64> {
    logits
        .iter()
        .enumerate()
        .filter(|(index, value)| {
            value.is_finite()
                && (*index as i64) <= max_text_token
                && !suppressed.contains(&(*index as i64))
                && !begin_suppressed.is_some_and(|suppressed| suppressed.contains(&(*index as i64)))
        })
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index as i64)
        .context("Whisper has no usable unsuppressed output token")
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("transcription cancelled");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_decode_obeys_suppression_and_timestamp_boundary() {
        let logits = [0.1, 2.0, 3.0, f32::NAN, 10.0];
        assert_eq!(
            select_token(&logits, &BTreeSet::from([2]), None, 3).unwrap(),
            1
        );
        assert_eq!(
            select_token(&logits, &BTreeSet::from([2]), Some(&BTreeSet::from([1])), 3).unwrap(),
            0
        );
    }

    #[test]
    fn pre_cancelled_requests_fail_without_touching_sessions() {
        assert!(check_cancel(&AtomicBool::new(true)).is_err());
        assert!(check_cancel(&AtomicBool::new(false)).is_ok());
    }

    #[test]
    fn language_detection_uses_only_declared_language_tokens() {
        let config = serde_json::json!({
            "decoder_attention_heads": 6, "d_model": 384,
            "decoder_layers": 4, "max_target_positions": 448,
        });
        let generation = serde_json::json!({
            "decoder_start_token_id": 11, "eos_token_id": 10,
            "no_timestamps_token_id": 20, "task_to_id": {"transcribe": 19},
            "lang_to_id": {"<|en|>": 12, "<|fr|>": 13},
            "suppress_tokens": [], "begin_suppress_tokens": [],
        });
        let parsed = Generation::from_config(&config, &generation).unwrap();
        let mut logits = vec![100.0; 21];
        logits[12] = 0.1;
        logits[13] = 0.5;
        let mut language = "auto".to_string();
        assert_eq!(
            parsed
                .language_token("auto", &logits, &mut language)
                .unwrap(),
            13
        );
        assert_eq!(language, "fr");
        assert_eq!(
            parsed.language_token("en", &logits, &mut language).unwrap(),
            12
        );
        let mut malformed = generation;
        malformed["decoder_start_token_id"] = serde_json::json!(-1);
        assert!(Generation::from_config(&config, &malformed).is_err());
    }

    #[test]
    #[ignore = "requires a Whisper bundle and a short 16 kHz WAV; set ONNX_WHISPER_MODEL_DIR and/or ONNX_WHISPER_INT8_MODEL_DIR, plus ONNX_WHISPER_WAV"]
    fn whisper_transcribes_with_request_local_decoder_state() {
        let audio =
            std::fs::read(std::env::var("ONNX_WHISPER_WAV").expect("ONNX_WHISPER_WAV")).unwrap();
        let mut ran = 0;
        for (variable, variant) in [
            ("ONNX_WHISPER_MODEL_DIR", Variant::Fp32),
            ("ONNX_WHISPER_INT8_MODEL_DIR", Variant::Int8),
        ] {
            let Ok(dir) = std::env::var(variable) else {
                continue;
            };
            let mut engine = Whisper::load(Path::new(&dir), 1, variant).unwrap();
            let first = engine
                .transcribe_wav(&audio, "auto", 128, &AtomicBool::new(false))
                .unwrap();
            let second = engine
                .transcribe_wav(&audio, "auto", 128, &AtomicBool::new(false))
                .unwrap();
            // Request-local decoder state: a second pass over the same audio
            // must reproduce the first transcript exactly. A leaked cache drifts.
            assert_eq!(first.text, second.text, "{variable}");
            assert!(!first.text.is_empty(), "{variable}");
            // The fixture is spoken English and detection runs off the first
            // decoder step's logits, so this pins detection as well as decode.
            assert_eq!(first.language, "en", "{variable}");
            let text = first.text.to_lowercase();
            for phrase in ["quick brown fox", "lazy dog", "mesh"] {
                assert!(
                    text.contains(phrase),
                    "{variable} transcript missing {phrase:?}: {text}"
                );
            }
            assert!(first.generated_tokens > 4, "{variable}: no decoder steps");
            ran += 1;
        }
        assert!(
            ran > 0,
            "set ONNX_WHISPER_MODEL_DIR and/or ONNX_WHISPER_INT8_MODEL_DIR"
        );
    }
}
