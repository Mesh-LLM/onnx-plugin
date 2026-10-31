use serde_json::Value;
use std::{
    fs::File,
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

pub const IDS: [&str; 4] = [
    "datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b",
    "onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7",
    "Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523",
    "onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e",
];
/// The mesh workload class each pinned model advertises, in `IDS` order. A
/// client needs this to tell an OCR model from a depth model before it knows
/// anything else about them.
pub const CLASSES: [&str; 4] = [
    "document_ocr",
    "speech_recognition",
    "image_text_similarity",
    "depth_estimation",
];
pub const PATHS: [&str; 4] = [
    "/mesh/v1/documents/ocr",
    "/v1/audio/transcriptions",
    "/mesh/v1/images/similarity",
    "/mesh/v1/images/depth",
];
pub const DIR_VARS: [&str; 4] = [
    "ONNX_OCR_MODEL_DIR",
    "ONNX_WHISPER_MODEL_DIR",
    "ONNX_CLIP_MODEL_DIR",
    "ONNX_DEPTH_MODEL_DIR",
];

pub fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}
pub fn quote(value: &str) -> String {
    serde_json::to_string(value).unwrap()
}
pub fn directory(index: usize, int8: bool) -> PathBuf {
    let name = if int8 && index != 0 {
        DIR_VARS[index].replace("_MODEL_DIR", "_INT8_MODEL_DIR")
    } else {
        DIR_VARS[index].to_owned()
    };
    std::env::var_os(&name)
        .unwrap_or_else(|| panic!("set {name} to the pinned fixture bundle"))
        .into()
}
pub fn model_config(index: usize, int8: bool, load: &str, prefetch: bool, idle: u64) -> String {
    format!(
        "\n[[models]]\nmodel = {}\nprovider = \"onnx\"\nvariant = {}\n[models.source]\nlocal_dir = {}\n[models.options]\nthreads = 2\n[models.lifecycle]\nload = {}\nprefetch = {prefetch}\nidle_timeout_secs = {idle}\n",
        quote(IDS[index]),
        quote(if int8 && index != 0 { "int8" } else { "fp32" }),
        quote(directory(index, int8).to_str().unwrap()),
        quote(load)
    )
}
/// True when a mesh-wide model inventory names every pinned model together with
/// the workload class that decides which route can call it.
pub fn discovers_all(value: &Value) -> bool {
    value["mesh_models"].as_array().is_some_and(|models| {
        IDS.iter().zip(CLASSES).all(|(id, class)| {
            models
                .iter()
                .any(|model| model["name"] == *id && model["workload_class"] == class)
        })
    })
}

/// True when a client's `/api/status` reports a peer that accepts HTTP inference
/// and hosts every one of `models`. A peer advertises its models in the mesh
/// catalog before it claims the host role, so a client that wants to *call* a
/// model has to wait for this, not merely for the catalog entry.
pub fn serving_peer(value: &Value, models: &[&str]) -> bool {
    value["peers"].as_array().is_some_and(|peers| {
        peers.iter().any(|peer| {
            peer["role"] == "Host"
                && models.iter().all(|model| {
                    peer["hosted_models"].as_array().is_some_and(|hosted| {
                        hosted.iter().any(|hosted| hosted.as_str() == Some(model))
                    })
                })
        })
    })
}

/// The discovered entry for one pinned model, with the class it advertises.
pub fn discovered(value: &Value, index: usize) -> &Value {
    value["mesh_models"]
        .as_array()
        .expect("mesh model inventory is a list")
        .iter()
        .find(|model| model["name"] == IDS[index])
        .unwrap_or_else(|| panic!("client inventory does not name {}", IDS[index]))
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap()
}
pub fn form(index: usize) -> reqwest::multipart::Form {
    let file = if index == 1 { "speech.wav" } else { "scan.png" };
    let mut form = reqwest::multipart::Form::new()
        .text("model", IDS[index])
        .part(
            "file",
            reqwest::multipart::Part::bytes(fixture(file)).file_name(file),
        );
    if index == 2 {
        form = form.text(
            "labels",
            r#"["a scanned page of printed text","a photograph of a dog","a bar chart"]"#,
        );
    }
    form
}
pub async fn infer(client: &reqwest::Client, base: &str, index: usize) -> Value {
    let response = client
        .post(format!("{base}{}", PATHS[index]))
        .multipart(form(index))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(status.is_success(), "{} {status}: {body}", IDS[index]);
    let value: Value = serde_json::from_str(&body).unwrap();
    match index {
        0 => {
            assert_eq!(value["pages"][0]["source"], "ocr");
            assert!(
                value["pages"][0]["text"]
                    .as_str()
                    .unwrap()
                    .contains("SEOUL"),
                "{value}"
            );
        }
        1 => {
            assert_eq!(value["language"], "en");
            let words: String = value["text"]
                .as_str()
                .unwrap()
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .flat_map(char::to_lowercase)
                .collect();
            assert_eq!(
                words.split_whitespace().collect::<Vec<_>>().join(" "),
                "the quick brown fox jumps over the lazy dog mesh roots speech recognition to whichever note is serving the model"
            );
        }
        2 => {
            assert_eq!(
                value["scores"][0]["label"], "a scanned page of printed text",
                "{value}"
            );
            assert!(
                value["scores"][0]["score"].as_f64().unwrap() > 0.2,
                "{value}"
            );
        }
        3 => {
            assert_eq!(value["units"], "relative", "{value}");
            assert_eq!(value["width"], 1653);
            assert_eq!(value["height"], 2337);
        }
        _ => unreachable!(),
    }
    value
}

const MAX_START_ATTEMPTS: usize = 3;

fn can_retry_startup(attempt: usize, logs: &str) -> bool {
    attempt < MAX_START_ATTEMPTS
        && logs.lines().any(|line| {
            // Only an error carrying both the bind context and AddrInUse cause is
            // a port race. Missing assets, rejected config, and timeouts must fail.
            let message = match serde_json::from_str::<Value>(line) {
                Ok(value) if value["level"] == "error" || value["level"] == "fatal" => {
                    value["message"].as_str().map(|message| {
                        format!(
                            "{message}: {}",
                            value["context"].as_str().unwrap_or_default()
                        )
                    })
                }
                Ok(_) => None,
                Err(_) => line.strip_prefix("Error:").map(str::to_owned),
            };
            message.is_some_and(|message| {
                let message = message.to_ascii_lowercase();
                message.contains("bind")
                    && (message.contains("address already in use")
                        || message.contains("addrinuse")
                        || message.contains("os error 48")
                        || message.contains("os error 98")
                        || message.contains("os error 10048"))
            })
        })
}

struct StartupFailure {
    exited: bool,
    detail: String,
}

pub struct Server {
    child: Child,
    pub root: tempfile::TempDir,
    pub base: String,
    pub control: String,
}
impl Server {
    pub fn start(models: &str, join: Option<&str>) -> Self {
        Self::start_with_cache(models, join, None, false)
    }
    pub fn start_with_cache(
        models: &str,
        join: Option<&str>,
        cache: Option<&std::path::Path>,
        offline: bool,
    ) -> Self {
        let mut failures = Vec::new();
        for attempt in 1..=MAX_START_ATTEMPTS {
            let mut server = Self::spawn_once(models, join, cache, offline);
            match server.wait_for_startup() {
                Ok(()) => return server,
                Err(failure) => {
                    let logs = server.logs();
                    failures.push(format!("attempt {attempt}: {}\n{logs}", failure.detail));
                    let retry = failure.exited && can_retry_startup(attempt, &logs);
                    // Reap this owned child before releasing its directory or
                    // choosing fresh ports. Callers only see the final URLs.
                    drop(server);
                    assert!(retry, "Mesh startup failed:\n{}", failures.join("\n"));
                }
            }
        }
        unreachable!("the last failed attempt must report its diagnostics")
    }

    fn spawn_once(
        models: &str,
        join: Option<&str>,
        cache: Option<&std::path::Path>,
        offline: bool,
    ) -> Self {
        let root = tempfile::tempdir().unwrap();
        let api = TcpListener::bind("127.0.0.1:0").unwrap();
        let control = TcpListener::bind("127.0.0.1:0").unwrap();
        let api_port = api.local_addr().unwrap().port();
        let control_port = control.local_addr().unwrap().port();
        let config = root.path().join("config.toml");
        let plugin = if join.is_some() {
            String::new()
        } else {
            format!(
                "\n[[plugin]]\nname = \"onnx\"\ncommand = {}\nargs = [\"--pdfium-library\", {}]\n",
                quote(env!("CARGO_BIN_EXE_mesh-onnx")),
                quote(&std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY"))
            )
        };
        std::fs::write(&config, format!("[runtime]\nmode = {}\n[runtime.native_runtime]\ndownload = false\n{plugin}\n{models}",quote(if join.is_some(){"client"}else{"serve"}))).unwrap();
        let log = File::create(root.path().join("server.log")).unwrap();
        drop((api, control));
        let mut command = Command::new(
            std::env::var_os("MESH_ONNX_TEST_HOST")
                .expect("MESH_ONNX_TEST_HOST: build Mesh with just mesh"),
        );
        command
            .args([
                if join.is_some() { "client" } else { "serve" },
                "--headless",
                "--disable-iroh-relays",
                "--bind-ip",
                "127.0.0.1",
                "--bind-port",
                "0",
                "--log-format",
                "json",
            ])
            .arg("--config")
            .arg(config)
            .arg("--port")
            .arg(api_port.to_string())
            .arg("--console")
            .arg(control_port.to_string())
            .env("MESH_LLM_DATA_DIR", root.path().join("data"))
            .env(
                "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
                root.path().join("native-cache"),
            )
            .env_remove("MESH_LLM_CONFIG")
            .env_remove("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log);
        if let Some(cache) = cache {
            command.env("HF_HUB_CACHE", cache);
        }
        if offline {
            command.env("HF_HUB_OFFLINE", "1");
        }
        if let Some(token) = join {
            command.arg("--join").arg(token);
        }
        let child = command.spawn().expect("start Mesh host");
        Self {
            child,
            root,
            base: format!("http://127.0.0.1:{api_port}"),
            control: format!("http://127.0.0.1:{control_port}"),
        }
    }
    fn wait_for_startup(&mut self) -> Result<(), StartupFailure> {
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            if let Some(status) = self.child.try_wait().expect("check Mesh child status") {
                return Err(StartupFailure {
                    exited: true,
                    detail: format!("Mesh exited {status}"),
                });
            }
            // This event is emitted by our child after it has bound both the
            // API and console listeners. Probing a port alone could accept a
            // different process that won the reserve-close-spawn race.
            let ready = self.logs().lines().any(|line| {
                serde_json::from_str::<Value>(line)
                    .is_ok_and(|value| value["event"] == "api_ready" && value["url"] == self.base)
            });
            if ready {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(StartupFailure {
                    exited: false,
                    detail: "Mesh startup deadline".into(),
                });
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    pub fn logs(&self) -> String {
        std::fs::read_to_string(self.root.path().join("server.log")).unwrap_or_default()
    }
    pub fn assert_alive(&mut self) {
        let status = self.child.try_wait().unwrap();
        assert!(status.is_none(), "Mesh exited {status:?}: {}", self.logs());
    }
    pub async fn wait(
        &mut self,
        client: &reqwest::Client,
        url: &str,
        predicate: impl Fn(&Value) -> bool,
    ) -> Value {
        let deadline = Instant::now() + Duration::from_secs(90);
        let mut last = Value::Null;
        loop {
            self.assert_alive();
            if let Ok(response) = client.get(url).timeout(Duration::from_secs(3)).send().await
                && response.status().is_success()
                && let Ok(value) = response.json::<Value>().await
            {
                if predicate(&value) {
                    return value;
                }
                last = value;
            }
            assert!(
                Instant::now() < deadline,
                "state deadline {url}: {last}\n{}",
                self.logs()
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    pub async fn provider(&mut self, client: &reqwest::Client) -> String {
        let endpoints = self
            .wait(
                client,
                &format!("{}/api/plugins/endpoints", self.control),
                |v| {
                    v.as_array().is_some_and(|a| {
                        a.iter()
                            .any(|e| e["plugin_name"] == "onnx" && e["address"].as_str().is_some())
                    })
                },
            )
            .await;
        endpoints
            .as_array()
            .unwrap()
            .iter()
            .find(|e| {
                e["plugin_name"] == "onnx"
                    && e["address"]
                        .as_str()
                        .is_some_and(|a| a.starts_with("http://"))
            })
            .unwrap()["address"]
            .as_str()
            .unwrap()
            .to_owned()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod startup_tests {
    use super::*;

    #[test]
    fn retries_only_address_in_use_startup_failures_and_stops_at_the_bound() {
        let collision = r#"{"level":"error","event":"error","message":"Failed to bind Web console to port 1234: Address already in use (os error 48)"}"#;
        let fatal = r#"{"level":"fatal","event":"fatal","message":"Failed to bind Web console to port 1234","context":"Address already in use (os error 48)"}"#;
        assert!(can_retry_startup(1, fatal));
        assert!(can_retry_startup(1, collision));
        assert!(can_retry_startup(MAX_START_ATTEMPTS - 1, collision));
        assert!(!can_retry_startup(MAX_START_ATTEMPTS, collision));
        assert!(can_retry_startup(
            1,
            "Error: Failed to bind API: Address already in use (os error 98)"
        ));
        for logs in [
            r#"{"level":"error","message":"required model asset is missing"}"#,
            r#"{"level":"error","message":"Failed to bind Web console: Permission denied (os error 13)"}"#,
            r#"{"level":"warn","message":"Failed to bind: Address already in use"}"#,
            "Error: invalid configuration",
        ] {
            assert!(!can_retry_startup(1, logs), "{logs}");
        }
    }
}
