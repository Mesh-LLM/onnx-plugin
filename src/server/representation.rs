//! Accept negotiation and rendering for the document OCR task.

use axum::{
    Json,
    http::{HeaderMap, StatusCode, header::ACCEPT},
    response::{IntoResponse, Response},
};

use super::{ApiError, Document};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Representation {
    Json,
    Text,
    Markdown,
}

impl Representation {
    fn media_type(self) -> (&'static str, &'static str) {
        match self {
            Self::Json => ("application", "json"),
            Self::Text => ("text", "plain"),
            Self::Markdown => ("text", "markdown"),
        }
    }

    pub(super) fn render(self, document: Document) -> Response {
        if self == Self::Json {
            return Json(document).into_response();
        }
        let body = document
            .pages
            .iter()
            .map(|page| match self {
                Self::Markdown => page.markdown.as_str(),
                _ => page.text.as_str(),
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let content_type = match self {
            Self::Markdown => "text/markdown; charset=utf-8",
            _ => "text/plain; charset=utf-8",
        };
        ([("content-type", content_type)], body).into_response()
    }
}

pub(super) fn select(
    headers: &HeaderMap,
    legacy_format: Option<&str>,
    legacy_path: bool,
) -> Result<Representation, ApiError> {
    if !headers.contains_key(ACCEPT) {
        return match legacy_format.filter(|_| legacy_path).unwrap_or("json") {
            "json" => Ok(Representation::Json),
            "text" => Ok(Representation::Text),
            "markdown" => Ok(Representation::Markdown),
            _ => Err(ApiError::bad_request(
                "response_format must be json, text, or markdown",
            )),
        };
    }
    let mut best = None;
    for representation in [
        Representation::Json,
        Representation::Text,
        Representation::Markdown,
    ] {
        let mut preferred_range = None;
        for value in headers.get_all(ACCEPT) {
            for raw in accept_parts(value.to_str().map_err(|_| not_acceptable())?, b',')? {
                let range = MediaRange::parse(raw)?;
                if let Some(specificity) = range.specificity(representation) {
                    let preference = (specificity, range.parameter_count, range.quality);
                    if preferred_range.is_none_or(|previous| preference > previous) {
                        preferred_range = Some(preference);
                    }
                }
            }
        }
        let quality = preferred_range.map_or(0, |(_, _, quality)| quality);
        if quality > 0 && best.is_none_or(|(previous, _)| quality > previous) {
            best = Some((quality, representation));
        }
    }
    best.map(|(_, representation)| representation)
        .ok_or_else(not_acceptable)
}

struct MediaRange<'a> {
    kind: &'a str,
    subtype: &'a str,
    quality: u16,
    supported_parameters: bool,
    parameter_count: u8,
}

impl<'a> MediaRange<'a> {
    fn parse(raw: &'a str) -> Result<Self, ApiError> {
        let mut parts = accept_parts(raw, b';')?.into_iter();
        let (kind, subtype) = parts
            .next()
            .unwrap_or_default()
            .trim()
            .split_once('/')
            .ok_or_else(not_acceptable)?;
        if kind.is_empty() || subtype.is_empty() || (kind == "*" && subtype != "*") {
            return Err(not_acceptable());
        }
        let mut range = Self {
            kind,
            subtype,
            quality: 1000,
            supported_parameters: true,
            parameter_count: 0,
        };
        let mut weighted = false;
        for parameter in parts {
            let Some((name, value)) = parameter.trim().split_once('=') else {
                return Err(not_acceptable());
            };
            if name.trim().eq_ignore_ascii_case("q") {
                if weighted {
                    return Err(not_acceptable());
                }
                range.quality = quality(value.trim()).ok_or_else(not_acceptable)?;
                weighted = true;
            } else if !weighted {
                // Every emitted representation is UTF-8. Other media parameters
                // are unsupported; extensions after q do not constrain the type.
                let value = value.trim();
                let unquoted = value
                    .strip_prefix('"')
                    .and_then(|inner| inner.strip_suffix('"'))
                    .unwrap_or(value);
                if name.trim().eq_ignore_ascii_case("charset") && unquoted.contains('"') {
                    return Err(not_acceptable());
                }
                range.supported_parameters &= name.trim().eq_ignore_ascii_case("charset")
                    && unquoted.eq_ignore_ascii_case("utf-8");
                // Only charset is supported, and duplicate parameters are invalid.
                range.supported_parameters &= range.parameter_count == 0;
                range.parameter_count = 1;
            }
        }
        Ok(range)
    }

    fn specificity(&self, representation: Representation) -> Option<u8> {
        if !self.supported_parameters {
            return None;
        }
        let (kind, subtype) = representation.media_type();
        if self.kind == "*" && self.subtype == "*" {
            Some(0)
        } else if self.kind.eq_ignore_ascii_case(kind) && self.subtype == "*" {
            Some(1)
        } else if self.kind.eq_ignore_ascii_case(kind) && self.subtype.eq_ignore_ascii_case(subtype)
        {
            Some(2)
        } else {
            None
        }
    }
}

fn quality(value: &str) -> Option<u16> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    if !matches!(whole, "0" | "1")
        || fraction.len() > 3
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let fraction = fraction.parse::<u16>().unwrap_or(0) * 10u16.pow(3 - fraction.len() as u32);
    match whole {
        "0" => Some(fraction),
        "1" if fraction == 0 => Some(1000),
        _ => None,
    }
}

fn accept_parts(value: &str, separator: u8) -> Result<Vec<&str>, ApiError> {
    let mut parts = Vec::new();
    let (mut start, mut quoted, mut escaped) = (0, false, false);
    for (index, byte) in value.bytes().enumerate() {
        if escaped {
            escaped = false;
        } else if quoted && byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            quoted = !quoted;
        } else if !quoted && byte == separator {
            parts.push(&value[start..index]);
            start = index + 1;
        }
    }
    if quoted || escaped {
        return Err(not_acceptable());
    }
    parts.push(&value[start..]);
    Ok(parts)
}

fn not_acceptable() -> ApiError {
    ApiError::new(
        StatusCode::NOT_ACCEPTABLE,
        "Accept must allow application/json, text/plain, or text/markdown",
    )
}

#[cfg(test)]
mod tests;
