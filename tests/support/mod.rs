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
