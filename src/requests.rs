//! Per-task configuration and bounded request overrides.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{pdf::PdfOptions, recipes::Kind};

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClipOptions {
    pub top_k: usize,
    pub truncate_text: bool,
    pub labels: Vec<String>,
}

impl Default for ClipOptions {
    fn default() -> Self {
        Self {
            top_k: 3,
            truncate_text: false,
            labels: Vec::new(),
        }
    }
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WhisperOptions {
    pub language: String,
    pub max_output_tokens: usize,
}

impl Default for WhisperOptions {
    fn default() -> Self {
        Self {
            language: "auto".into(),
            max_output_tokens: 128,
        }
    }
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DepthOptions {
    pub output_resolution: String,
    pub visualization: String,
}

impl Default for DepthOptions {
    fn default() -> Self {
        Self {
            output_resolution: "input".into(),
            visualization: "none".into(),
        }
    }
}

pub enum RequestOptions {
    Ocr(PdfOptions),
    Clip(ClipOptions),
    Whisper(WhisperOptions),
    Depth(DepthOptions),
}

pub fn validate_defaults(
    kind: Kind,
    defaults: &BTreeMap<String, impl serde::Serialize>,
) -> Result<()> {
    let value = serde_json::to_value(defaults)?;
    if value.get("password").is_some() || value.get("labels").is_some() {
        bail!("password and labels are request-only inputs, not defaults");
    }
    decode(kind, value, false).map(|_| ())
}

pub fn merge(
    kind: Kind,
    defaults: &BTreeMap<String, impl serde::Serialize>,
    fields: BTreeMap<String, String>,
) -> Result<RequestOptions> {
    let mut values: Map<String, Value> = serde_json::from_value(serde_json::to_value(defaults)?)?;
    for (key, value) in fields {
        let parsed = match key.as_str() {
            "dpi" | "max_pages" | "top_k" | "max_output_tokens" | "image_ocr"
            | "keep_furniture" | "truncate_text" | "labels" => serde_json::from_str(&value)?,
            _ => Value::String(value),
        };
        values.insert(key, parsed);
    }
    decode(kind, Value::Object(values), true)
}

fn decode(kind: Kind, value: Value, request: bool) -> Result<RequestOptions> {
    Ok(match kind {
        Kind::Indra => {
            let options: PdfOptions = serde_json::from_value(value)?;
            options.validate()?;
            RequestOptions::Ocr(options)
        }
        Kind::Clip => {
            let options: ClipOptions = serde_json::from_value(value)?;
            if options.top_k == 0 || options.top_k > 100 {
                bail!("top_k must be between 1 and 100");
            }
            if options.labels.len() > 100
                || options
                    .labels
                    .iter()
                    .any(|label| label.is_empty() || label.len() > 4096)
            {
                bail!("labels must contain at most 100 nonempty texts of at most 4096 bytes");
            }
            if request && options.labels.is_empty() {
                bail!("labels are required for image/text similarity");
            }
            RequestOptions::Clip(options)
        }
        Kind::Whisper => {
            let options: WhisperOptions = serde_json::from_value(value)?;
            if !(1..=448).contains(&options.max_output_tokens) {
                bail!("max_output_tokens must be between 1 and 448");
            }
            if options.language != "auto"
                && (options.language.len() < 2
                    || options.language.len() > 3
                    || !options.language.bytes().all(|c| c.is_ascii_lowercase()))
            {
                bail!("language must be auto or a supported language code");
            }
            RequestOptions::Whisper(options)
        }
        Kind::Depth => {
            let options: DepthOptions = serde_json::from_value(value)?;
            if !matches!(options.output_resolution.as_str(), "input" | "native") {
                bail!("output_resolution must be input or native");
            }
            if !matches!(options.visualization.as_str(), "none" | "normalized") {
                bail!("visualization must be none or normalized");
            }
            RequestOptions::Depth(options)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolates_task_schemas_and_rejects_unknown_options() {
        for (kind, key, value) in [
            (Kind::Whisper, "dpi", "200"),
            (Kind::Depth, "language", "en"),
            (Kind::Clip, "temperature", "1"),
        ] {
            assert!(
                merge(
                    kind,
                    &BTreeMap::<String, Value>::new(),
                    BTreeMap::from([(key.into(), value.into())])
                )
                .is_err()
            );
        }
    }

    #[test]
    fn validates_defaults_and_bounded_request_overrides() {
        let defaults = BTreeMap::from([("max_output_tokens".into(), Value::from(64))]);
        let RequestOptions::Whisper(options) = merge(
            Kind::Whisper,
            &defaults,
            BTreeMap::from([("language".into(), "en".into())]),
        )
        .unwrap() else {
            panic!("whisper")
        };
        assert_eq!(options.max_output_tokens, 64);
        assert_eq!(options.language, "en");
        assert!(
            merge(
                Kind::Whisper,
                &defaults,
                BTreeMap::from([("max_output_tokens".into(), "449".into())])
            )
            .is_err()
        );
        assert!(
            validate_defaults(
                Kind::Indra,
                &BTreeMap::from([("password".into(), Value::from("secret"))])
            )
            .is_err()
        );
    }
}
