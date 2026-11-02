//! Router admission tests are model-free; recognition tests remain opt-in.

use super::*;
use std::{net::SocketAddr, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct RunningServer(tokio::task::JoinHandle<()>);

impl Drop for RunningServer {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn serve(app: Router) -> (SocketAddr, RunningServer) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (address, RunningServer(task))
}

async fn must_not_recognize() -> Response {
    panic!("unacceptable requests must not reach the OCR handler")
}

#[tokio::test]
async fn both_routes_reject_unacceptable_headers_before_multipart_or_recognition() {
    let app = document_routes(post(must_not_recognize), post(must_not_recognize));
    let (address, _server) = serve(app).await;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    for route in ["/mesh/v1/documents/ocr", "/v1/documents/ocr"] {
        for accept in ["image/png", "application/json;q=0, text/*;q=0"] {
            let response = client
                .post(format!("http://{address}{route}"))
                .header("Accept", accept)
                .body("not multipart")
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_ACCEPTABLE);
            assert_eq!(response.headers()["vary"], "Accept");
            let error: serde_json::Value = response.json().await.unwrap();
            assert!(
                error["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("Accept")
            );
        }
    }
}

#[tokio::test]
async fn both_routes_admit_acceptable_headers() {
    async fn admitted() -> &'static str {
        "admitted"
    }

    let app = document_routes(post(admitted), post(admitted));
    let (address, _server) = serve(app).await;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    for route in ["/mesh/v1/documents/ocr", "/v1/documents/ocr"] {
        let response = client
            .post(format!("http://{address}{route}"))
            .header("Accept", "application/json")
            .body("not multipart")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["vary"], "Accept");
        assert_eq!(response.text().await.unwrap(), "admitted");
    }
}

#[tokio::test]
#[ignore = "requires Indra v1.1.0 model assets and ONNX Runtime"]
async fn native_image_negotiates_formats_and_preserves_legacy_precedence() {
    let model_dir = std::env::var("ONNX_OCR_MODEL_DIR").expect("ONNX_OCR_MODEL_DIR");
    let engine = Engine::load(std::path::Path::new(&model_dir)).expect("load ONNX sessions");
    let (address, _server) = serve(router(ServerState::new(engine, None))).await;
    for (route, accept, body_format, content_type) in [
        (
            "/mesh/v1/documents/ocr",
            None,
            "invalid",
            "application/json",
        ),
        (
            "/mesh/v1/documents/ocr",
            Some("text/plain"),
            "invalid",
            "text/plain; charset=utf-8",
        ),
        (
            "/mesh/v1/documents/ocr",
            Some("text/markdown"),
            "json",
            "text/markdown; charset=utf-8",
        ),
        (
            "/v1/documents/ocr",
            None,
            "text",
            "text/plain; charset=utf-8",
        ),
        (
            "/v1/documents/ocr",
            Some("*/*"),
            "invalid",
            "application/json",
        ),
    ] {
        let response = image_request(address, route, accept, body_format).await;
        assert_eq!(
            response.status,
            StatusCode::OK,
            "{route} {accept:?}: {}",
            response.body
        );
        assert_eq!(response.headers["vary"], "Accept");
        assert_eq!(response.headers["content-type"], content_type);
        assert!(
            response.body.contains("대한성공회 서울교구"),
            "{route} {accept:?}"
        );
        if content_type == "application/json" {
            let document: serde_json::Value = serde_json::from_str(&response.body).unwrap();
            assert_eq!(document["model"], MODEL);
            assert_eq!(document["pages"][0]["page_index"], 0);
        }
    }
    let response = image_request(address, "/v1/documents/ocr", None, "invalid").await;
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.headers["vary"], "Accept");
    let body: serde_json::Value = serde_json::from_str(&response.body).unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("response_format")
    );
}

struct ImageResponse {
    status: StatusCode,
    headers: HeaderMap,
    body: String,
}

// reqwest adds Accept: */* by default. Send the exact header set so None proves
// the absent-header compatibility case, rather than a wildcard request.
async fn image_request(
    address: SocketAddr,
    route: &str,
    accept: Option<&str>,
    format: &str,
) -> ImageResponse {
    let boundary = "mesh-ocr-contract";
    let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{MODEL}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n{format}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"scan.png\"\r\nContent-Type: image/png\r\n\r\n").into_bytes();
    body.extend_from_slice(include_bytes!("../../tests/fixtures/scan.png"));
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let accept = accept
        .map(|value| format!("Accept: {value}\r\n"))
        .unwrap_or_default();
    let head = format!(
        "POST {route} HTTP/1.1\r\nHost: {address}\r\n{accept}Content-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let response = tokio::time::timeout(Duration::from_secs(120), async {
        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        socket.write_all(head.as_bytes()).await.unwrap();
        socket.write_all(&body).await.unwrap();
        let mut response = Vec::new();
        socket
            .take(4 * 1024 * 1024)
            .read_to_end(&mut response)
            .await
            .unwrap();
        String::from_utf8(response).unwrap()
    })
    .await
    .expect("bounded OCR response");
    parse_image_response(response)
}

fn parse_image_response(response: String) -> ImageResponse {
    let (head, body) = response.split_once("\r\n\r\n").expect("HTTP response head");
    let mut lines = head.lines();
    let status = lines
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u16>()
        .unwrap();
    let mut headers = HeaderMap::new();
    for line in lines {
        let (name, value) = line.split_once(':').unwrap();
        headers.append(
            axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_str(value.trim()).unwrap(),
        );
    }
    assert!(
        !headers.contains_key("transfer-encoding"),
        "buffered document response must have a length"
    );
    assert_eq!(
        headers["content-length"]
            .to_str()
            .unwrap()
            .parse::<usize>()
            .unwrap(),
        body.len()
    );
    ImageResponse {
        status: StatusCode::from_u16(status).unwrap(),
        headers,
        body: body.into(),
    }
}
