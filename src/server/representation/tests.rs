use super::*;
use crate::ocr::Page;
use axum::{body::to_bytes, http::HeaderValue};

fn headers(accept: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_str(accept).unwrap());
    headers
}

#[test]
fn canonical_accept_defaults_and_overrides_legacy_body() {
    for body in [None, Some("text"), Some("invalid")] {
        assert_eq!(
            select(&HeaderMap::new(), body, false).unwrap(),
            Representation::Json
        );
        assert_eq!(
            select(&headers("*/*"), body, false).unwrap(),
            Representation::Json
        );
        assert_eq!(
            select(&headers("text/markdown"), body, false).unwrap(),
            Representation::Markdown
        );
    }
}

#[test]
fn legacy_body_applies_only_without_any_accept_header() {
    for (body, expected) in [
        ("json", Representation::Json),
        ("text", Representation::Text),
        ("markdown", Representation::Markdown),
    ] {
        assert_eq!(
            select(&HeaderMap::new(), Some(body), true).unwrap(),
            expected
        );
        assert_eq!(
            select(&headers("*/*"), Some(body), true).unwrap(),
            Representation::Json
        );
    }
    assert_eq!(
        select(&HeaderMap::new(), Some("invalid"), true)
            .unwrap_err()
            .status,
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        select(&headers("text/plain"), Some("invalid"), true).unwrap(),
        Representation::Text
    );
    assert_eq!(
        select(&headers("image/png"), Some("json"), true)
            .unwrap_err()
            .status,
        StatusCode::NOT_ACCEPTABLE
    );
}

#[test]
fn weighted_ranges_respect_specific_exclusions_and_server_ties() {
    for (accept, expected) in [
        ("application/json", Representation::Json),
        ("text/plain", Representation::Text),
        ("text/markdown", Representation::Markdown),
        ("text/*", Representation::Text),
        (
            "application/json;q=0.2, text/markdown;q=0.8",
            Representation::Markdown,
        ),
        ("application/json;q=0, */*;q=1", Representation::Text),
        ("text/*;q=0.8, text/plain;q=0", Representation::Markdown),
        ("text/plain, application/json", Representation::Json),
        ("TEXT/PLAIN; charset=\"UTF-8\"", Representation::Text),
        ("text/plain;charset=latin-1,*/*", Representation::Json),
        (
            "text/plain;q=1;note=\"comma, semicolon; and \\\"quote\\\"\"",
            Representation::Text,
        ),
        (
            "text/plain;q=1, text/plain;charset=utf-8;q=0, application/json;q=0.5",
            Representation::Json,
        ),
    ] {
        assert_eq!(
            select(&headers(accept), None, false).unwrap(),
            expected,
            "{accept}"
        );
    }
    let mut multiple = headers("application/json;q=0.1");
    multiple.append(ACCEPT, HeaderValue::from_static("text/markdown;q=0.9"));
    assert_eq!(
        select(&multiple, None, false).unwrap(),
        Representation::Markdown
    );
}

#[test]
fn unacceptable_or_malformed_accept_cannot_fall_back_to_json() {
    for accept in [
        "image/png",
        "",
        "*/*;q=0",
        "application/json;q=0, text/*;q=0",
        "text/plain;charset=latin-1",
        "text/plain;charset=\"\"utf-8\"\"",
        "text/plain;charset=\"\"utf-8\"\",*/*",
        "text/plain;q=NaN",
        "text/plain;q=1.1",
        "text/plain;q=-1",
        "text/plain;q=0.0001",
        "*/plain",
        "text/plain;q=1;q=0",
        "text/plain;note=\"unfinished",
    ] {
        assert_eq!(
            select(&headers(accept), Some("json"), true)
                .unwrap_err()
                .status,
            StatusCode::NOT_ACCEPTABLE,
            "{accept}"
        );
    }
}

fn document() -> Document {
    let pages = [("first", "# first"), ("second", "# second")]
        .into_iter()
        .enumerate()
        .map(|(page_index, (text, markdown))| Page {
            page_index,
            rotation: 0,
            vertical: false,
            text: text.into(),
            markdown: markdown.into(),
            source: "ocr",
            lines: vec![],
            blocks: vec![],
        })
        .collect();
    Document {
        model: "datalab/indra-ocr",
        pages,
    }
}

#[tokio::test]
async fn representations_preserve_document_and_join_page_content() {
    for (representation, content_type, expected) in [
        (
            Representation::Text,
            "text/plain; charset=utf-8",
            "first\n\nsecond",
        ),
        (
            Representation::Markdown,
            "text/markdown; charset=utf-8",
            "# first\n\n# second",
        ),
    ] {
        let response = representation.render(document());
        assert_eq!(response.headers()["content-type"], content_type);
        assert_eq!(
            to_bytes(response.into_body(), 4096).await.unwrap().as_ref(),
            expected.as_bytes()
        );
    }
    let response = Representation::Json.render(document());
    assert_eq!(response.headers()["content-type"], "application/json");
    let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value, serde_json::to_value(document()).unwrap());
}
