//! HTTP success representation negotiation, independent of model execution.

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Representation {
    Json,
    Text,
    Markdown,
    Png,
}

impl Representation {
    pub fn content_type(self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Text => "text/plain; charset=utf-8",
            Self::Markdown => "text/markdown; charset=utf-8",
            Self::Png => "image/png",
        }
    }

    /// Translate OpenAI's transcription `response_format` body field.
    ///
    /// `/v1/audio/transcriptions` is OpenAI-shaped, and there the caller picks
    /// the body with a form field rather than an `Accept` header. An OpenAI SDK
    /// client that asks for `text` would otherwise be rejected for an unknown
    /// field, so the field is translated into the same representation decision
    /// the header feeds — one mechanism, two ways in.
    ///
    /// `verbose_json`, `srt` and `vtt` all require word or segment timings this
    /// decoder does not produce. They are refused with a reason instead of
    /// being answered with a JSON body that silently lacks what was asked for.
    pub fn from_openai_response_format(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            other => bail!(
                "response_format '{other}' is not supported: this decoder produces no timestamps; use json or text"
            ),
        }
    }

    pub fn negotiate(accept: Option<&str>, supported: &[Self]) -> Result<Self> {
        // An absent or empty header means "no preference", not "nothing
        // acceptable": rejecting it would fail every client that sends the
        // header without a value.
        let accept = accept.map(str::trim).filter(|value| !value.is_empty());
        let ranges = parse_ranges(accept.unwrap_or("*/*"))?;
        let mut selected = None;
        for &representation in supported {
            let media = representation
                .content_type()
                .split(';')
                .next()
                .unwrap_or("");
            let mut preference = None;
            for (range, quality, parameters) in &ranges {
                if !parameters.iter().all(|(key, value)| {
                    representation
                        .content_type()
                        .split(';')
                        .skip(1)
                        .any(|part| {
                            part.trim().split_once('=').is_some_and(|(k, v)| {
                                k.eq_ignore_ascii_case(key)
                                    && v.trim_matches('"').eq_ignore_ascii_case(value)
                            })
                        })
                }) {
                    continue;
                }
                let specificity = if range.eq_ignore_ascii_case(media) {
                    2
                } else if range == "*/*" {
                    0
                } else if range.ends_with("/*")
                    && range.split('/').next() == media.split('/').next()
                {
                    1
                } else {
                    continue;
                };
                let specificity = (specificity, parameters.len());
                if preference.is_none_or(|(previous, _)| specificity > previous) {
                    preference = Some((specificity, *quality));
                }
            }
            if let Some((_, quality)) = preference
                && quality > 0.0
                && selected.is_none_or(|(_, previous)| quality > previous)
            {
                selected = Some((representation, quality));
            }
        }
        selected
            .map(|(value, _)| value)
            .ok_or_else(|| anyhow::anyhow!("no supported response representation is acceptable"))
    }
}

type MediaRange = (String, f32, Vec<(String, String)>);
fn parse_ranges(accept: &str) -> Result<Vec<MediaRange>> {
    let mut ranges = Vec::new();
    for entry in accept.split(',') {
        let mut parts = entry.trim().split(';');
        let media = parts.next().unwrap_or("").trim().to_ascii_lowercase();
        if !media.contains('/') {
            bail!("invalid Accept media range");
        }
        let mut quality = 1.0;
        let mut parameters = Vec::new();
        for parameter in parts {
            if let Some((name, value)) = parameter.trim().split_once('=') {
                if !name.trim().eq_ignore_ascii_case("q") {
                    parameters.push((
                        name.trim().to_ascii_lowercase(),
                        value.trim().trim_matches('"').to_ascii_lowercase(),
                    ));
                    continue;
                }
                quality = value.trim().parse::<f32>()?;
                if !quality.is_finite() || !(0.0..=1.0).contains(&quality) {
                    bail!("invalid Accept quality");
                }
            }
        }
        ranges.push((media, quality, parameters));
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;
    const ALL: &[Representation] = &[
        Representation::Json,
        Representation::Text,
        Representation::Markdown,
    ];

    #[test]
    fn media_parameters_constrain_the_representation_and_precedence() {
        assert!(Representation::negotiate(Some("text/plain; charset=iso-8859-1"), ALL).is_err());
        assert_eq!(
            Representation::negotiate(Some("text/plain; charset=\"UTF-8\""), ALL).unwrap(),
            Representation::Text
        );
        assert_eq!(
            Representation::negotiate(
                Some("text/plain; charset=utf-8;q=0, text/plain;q=1, application/json;q=0.5"),
                ALL
            )
            .unwrap(),
            Representation::Json
        );
        assert!(Representation::negotiate(Some("application/json;profile=unknown"), ALL).is_err());
    }

    #[test]
    fn defaults_and_quality_values_choose_a_supported_representation() {
        assert_eq!(
            Representation::negotiate(None, ALL).unwrap(),
            Representation::Json
        );
        assert_eq!(
            Representation::negotiate(Some("*/*"), ALL).unwrap(),
            Representation::Json
        );
        assert_eq!(
            Representation::negotiate(Some("application/json;q=0.2, text/markdown;q=0.8"), ALL)
                .unwrap(),
            Representation::Markdown
        );
        assert_eq!(
            Representation::negotiate(Some("text/*"), ALL).unwrap(),
            Representation::Text
        );
    }

    #[test]
    fn exact_exclusions_override_wildcards() {
        assert_eq!(
            Representation::negotiate(Some("application/json;q=0, */*;q=0.5"), ALL).unwrap(),
            Representation::Text
        );
        assert!(
            Representation::negotiate(
                Some("application/json;q=0, */*;q=1"),
                &[Representation::Json]
            )
            .is_err()
        );
        assert!(Representation::negotiate(Some("image/png"), ALL).is_err());
    }

    #[test]
    fn rejects_invalid_quality() {
        for accept in ["application/json;q=NaN", "*/*;q=2", "text/plain;q=-1"] {
            assert!(Representation::negotiate(Some(accept), ALL).is_err());
        }
    }

    #[test]
    fn an_empty_header_means_no_preference() {
        assert_eq!(
            Representation::negotiate(Some(""), ALL).unwrap(),
            Representation::Json
        );
        assert_eq!(
            Representation::negotiate(Some("   "), ALL).unwrap(),
            Representation::Json
        );
    }

    #[test]
    fn openai_response_format_names_the_same_representations() {
        assert_eq!(
            Representation::from_openai_response_format("json").unwrap(),
            Representation::Json
        );
        assert_eq!(
            Representation::from_openai_response_format(" TEXT ").unwrap(),
            Representation::Text
        );
        // A body shape this decoder cannot fill is refused, not approximated.
        for unsupported in ["verbose_json", "srt", "vtt", ""] {
            assert!(
                Representation::from_openai_response_format(unsupported).is_err(),
                "{unsupported} must not silently become JSON"
            );
        }
    }

    #[test]
    fn image_output_is_negotiated_without_affecting_text_tasks() {
        const DEPTH: &[Representation] = &[Representation::Json, Representation::Png];
        assert_eq!(
            Representation::negotiate(None, DEPTH).unwrap(),
            Representation::Json
        );
        assert_eq!(
            Representation::negotiate(Some("image/png"), DEPTH).unwrap(),
            Representation::Png
        );
        assert_eq!(
            Representation::negotiate(Some("*/*"), DEPTH).unwrap(),
            Representation::Json
        );
        // An endpoint that cannot produce an image refuses the request rather
        // than sending JSON labelled as a PNG.
        assert!(Representation::negotiate(Some("image/png"), ALL).is_err());
        assert_eq!(Representation::Png.content_type(), "image/png");
    }
}
