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
    reject_pages_and_max_pages(kind, &value)?;
    decode(kind, value, false).map(|_| ())
}

/// `pages` and `max_pages` select two different ways and never combine; the
/// check runs on the merged map so a default on one side cannot slip past a
/// field on the other.
fn reject_pages_and_max_pages(kind: Kind, value: &Value) -> Result<()> {
    if kind == Kind::Indra && value.get("pages").is_some() && value.get("max_pages").is_some() {
        bail!(
            "`pages` selects a page range and `max_pages` caps the page count; \
             they are mutually exclusive — use `pages` to select (for example 1-10) or `max_pages` to cap"
        );
    }
    Ok(())
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
            | "keep_furniture" | "truncate_text" | "labels" | "pages_only" | "stream"
            | "deadline_secs" => serde_json::from_str(&value)?,
            _ => Value::String(value),
        };
        values.insert(key, parsed);
    }
    let value = Value::Object(values);
    reject_pages_and_max_pages(kind, &value)?;
    decode(kind, value, true)
}

/// Per-request acquisition + inference budget when the caller does not name
/// one: grows with the selected page count, capped at ten minutes.
pub fn default_deadline_secs(pages: usize) -> u64 {
    std::cmp::min(60 + 2 * pages as u64, 600)
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
            if !(1..=444).contains(&options.max_output_tokens) {
                bail!("max_output_tokens must be between 1 and 444");
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

    #[test]
    fn ocr_v2_fields_parse_validate_and_stay_task_local() {
        // `pages` tolerates whitespace; `pages_only`, `stream` and
        // `deadline_secs` arrive as JSON literals like the other booleans.
        let RequestOptions::Ocr(options) = merge(
            Kind::Indra,
            &BTreeMap::<String, Value>::new(),
            BTreeMap::from([
                ("pages".into(), " 2-5 ".into()),
                ("pages_only".into(), "true".into()),
                ("stream".into(), "false".into()),
                ("deadline_secs".into(), "120".into()),
            ]),
        )
        .unwrap() else {
            panic!("ocr")
        };
        let range = options.pages.expect("range");
        assert_eq!((range.from, range.to), (2, Some(5)));
        assert!(options.pages_only);
        assert!(!options.stream);
        assert_eq!(options.deadline_secs, Some(120));

        // Syntax, bounds and exclusivity fail before any document or model work.
        for (key, value) in [
            ("pages", "a-b"),
            ("pages", "5-2"),
            ("pages", "0-3"),
            ("deadline_secs", "29"),
            ("deadline_secs", "901"),
        ] {
            let error = merge(
                Kind::Indra,
                &BTreeMap::<String, Value>::new(),
                BTreeMap::from([(key.into(), value.into())]),
            )
            .err()
            .expect("invalid field");
            assert!(error.to_string().contains(value.trim()), "{error}");
        }
        let error = merge(
            Kind::Indra,
            &BTreeMap::<String, Value>::new(),
            BTreeMap::from([
                ("pages".into(), "1-10".into()),
                ("max_pages".into(), "5".into()),
            ]),
        )
        .err()
        .expect("mutually exclusive");
        assert!(error.to_string().contains("mutually exclusive"), "{error}");

        // The v0.2 OCR fields stay on the OCR schema: other tasks still reject them.
        for kind in [Kind::Whisper, Kind::Clip, Kind::Depth] {
            for (key, value) in [
                ("pages", "1-10"),
                ("pages_only", "true"),
                ("stream", "true"),
                ("deadline_secs", "120"),
            ] {
                assert!(
                    merge(
                        kind,
                        &BTreeMap::<String, Value>::new(),
                        BTreeMap::from([(key.into(), value.into())])
                    )
                    .is_err(),
                    "{kind:?} must reject {key}"
                );
            }
        }
    }

    #[test]
    fn pages_and_max_pages_stay_exclusive_across_defaults_and_requests() {
        // The check runs on the merged map: a default setting one side and a
        // request field setting the other still lands on both keys.
        for (defaults, fields) in [
            (
                BTreeMap::from([("pages".to_string(), Value::from("1-10"))]),
                BTreeMap::from([("max_pages".to_string(), "5".to_string())]),
            ),
            (
                BTreeMap::from([("max_pages".to_string(), Value::from(5))]),
                BTreeMap::from([("pages".to_string(), "1-10".to_string())]),
            ),
        ] {
            let error = merge(Kind::Indra, &defaults, fields)
                .err()
                .expect("mutually exclusive across defaults and request");
            assert!(error.to_string().contains("mutually exclusive"), "{error}");
        }
        // Defaults alone must not set both either.
        let error = validate_defaults(
            Kind::Indra,
            &BTreeMap::from([
                ("pages".to_string(), Value::from("1-10")),
                ("max_pages".to_string(), Value::from(5)),
            ]),
        )
        .expect_err("defaults setting both is rejected");
        assert!(error.to_string().contains("mutually exclusive"), "{error}");
    }

    #[test]
    fn deadline_default_scales_with_pages_and_caps() {
        assert_eq!(default_deadline_secs(1), 62);
        assert_eq!(default_deadline_secs(10), 80);
        assert_eq!(default_deadline_secs(270), 600);
        assert_eq!(default_deadline_secs(1000), 600);
    }
}
