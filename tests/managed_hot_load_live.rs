//! Opt-in certification against real Mesh and ONNX processes, never mock models.
//! Build Mesh with `just mesh`, set MESH_ONNX_TEST_HOST and the model/Pdfium
//! fixture variables from docs/TESTING.md, then run with --include-ignored.
mod support;
use serde_json::Value;
use std::{
    path::Path,
    time::{Duration, SystemTime},
};
use support::*;

fn cache_snapshot(root: &Path) -> Vec<(std::path::PathBuf, u64, SystemTime)> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(cache_snapshot(&path));
        } else {
            let meta = path.metadata().unwrap();
            files.push((path, meta.len(), meta.modified().unwrap()));
        }
    }
    files.sort();
    files
}
fn availability(value: &Value, model: &str, expected: &str) -> bool {
    value["data"].as_array().is_some_and(|a| {
        a.iter()
            .any(|m| m["id"] == model && m["availability"] == expected)
    })
}

#[tokio::test]
#[ignore = "requires MESH_ONNX_TEST_HOST, PDFIUM_LIBRARY and all seven pinned model bundles"]
async fn four_families_precisions_cold_concurrent_idle_reload_and_remote_routing() {
    let client = client();
    for int8 in [false, true] {
        let snapshots: Vec<_> = (0..4)
            .map(|i| cache_snapshot(&directory(i, int8)))
            .collect();
        let config: String = (0..4)
            .map(|i| model_config(i, int8, "on_demand", false, 3))
            .collect();
        let mut server = Server::start(&config, None);
        let provider = server.provider(&client).await;
        let inventory = format!("{provider}/v1/models");
        let cold = server
            .wait(&client, &inventory, |v| {
                IDS.iter().all(|id| availability(v, id, "loadable"))
            })
            .await;
        assert_eq!(cold["data"].as_array().unwrap().len(), 4);
        // First named requests race on each model; independent families also
        // progress concurrently, exercising request-local decoder caches.
        for i in 0..4 {
            let (a, b) = tokio::join!(
                infer(&client, &server.base, i),
                infer(&client, &server.base, i)
            );
            assert_eq!(a, b, "concurrent request-local decoders must agree");
        }
        let mut requests = tokio::task::JoinSet::new();
        for i in 0..4 {
            let c = client.clone();
            let base = server.base.clone();
            requests.spawn(async move {
                infer(&c, &base, i).await;
            });
        }
        while let Some(result) = requests.join_next().await {
            result.unwrap();
        }
        server
            .wait(&client, &inventory, |v| {
                IDS.iter().all(|id| availability(v, id, "loadable"))
            })
            .await;
        for (i, before) in snapshots.iter().enumerate() {
            assert_eq!(
                &cache_snapshot(&directory(i, int8)),
                before,
                "eviction must retain cached files"
            );
        }
        // Same provider process, same configured models, fresh engines.
        for i in 0..4 {
            infer(&client, &server.base, i).await;
        }
        server.assert_alive();
        representations(&client, &server.base).await;
        documents(&client, &server.base).await;
        // A second, client-only Mesh process has no plugin and cannot execute
        // inference locally. Each response must traverse the peer transport.
        let status = server
            .wait(&client, &format!("{}/api/status", server.control), |v| {
                v["token"].as_str().is_some()
            })
            .await;
        let token = status["token"].as_str().unwrap();
        let mut remote = Server::start("", Some(token));
        remote
            .wait(&client, &format!("{}/api/status", remote.control), |v| {
                v["peers"].as_array().is_some_and(|p| !p.is_empty())
            })
            .await;
        for i in 0..4 {
            infer(&client, &remote.base, i).await;
        }
        remote.assert_alive();
        server.assert_alive();
        eprintln!(
            "acceptance passed: {} (four families, concurrency, idle reload, remote HTTP)",
            if int8 { "INT8 + Indra FP32" } else { "FP32" }
        );
    }
}

async fn representations(client: &reqwest::Client, base: &str) {
    let response = client
        .post(format!("{base}{}", PATHS[3]))
        .header("Accept", "image/png")
        .multipart(form(3))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    assert_eq!(response.headers()["content-type"], "image/png");
    let png = response.bytes().await.unwrap();
    let image = image::load_from_memory(&png).unwrap();
    assert_eq!((image.width(), image.height()), (1653, 2337));
    assert!(matches!(image, image::DynamicImage::ImageLuma8(_)));
    let response = client
        .post(format!("{base}{}", PATHS[1]))
        .header("Accept", "image/png")
        .multipart(form(1).text("response_format", "text"))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    assert!(
        response.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/plain")
    );
    assert!(response.text().await.unwrap().contains("quick brown fox"));
    let response = client
        .post(format!("{base}{}", PATHS[1]))
        .multipart(form(1).text("language", "zz"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let response = client
        .post(format!("{base}{}", PATHS[0]))
        .header("Accept", "image/png")
        .multipart(form(0))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 406);
}

#[tokio::test]
#[ignore = "requires MESH_ONNX_TEST_HOST, PDFIUM_LIBRARY and ONNX_CLIP_MODEL_DIR"]
async fn startup_and_prefetch_policies_and_zero_disables_eviction() {
    let client = client();
    for (load, prefetch) in [("startup", false), ("on_demand", true)] {
        let mut server = Server::start(&model_config(2, false, load, prefetch, 0), None);
        let provider = server.provider(&client).await;
        let inventory = format!("{provider}/v1/models");
        let expected = if load == "startup" {
            "ready"
        } else {
            "loadable"
        };
        server
            .wait(&client, &inventory, |v| {
                availability(v, IDS[2], expected)
                    && (load != "on_demand" || v["data"][0]["assets_ready"] == true)
            })
            .await;
        if load == "on_demand" {
            // Inventory requests never create a session, including while
            // disk-only prefetch is enabled. First inference changes state.
            for _ in 0..10 {
                server
                    .wait(&client, &inventory, |v| availability(v, IDS[2], "loadable"))
                    .await;
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
        infer(&client, &server.base, 2).await;
        for _ in 0..20 {
            server
                .wait(&client, &inventory, |v| availability(v, IDS[2], "ready"))
                .await;
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

async fn documents(client: &reqwest::Client, base: &str) {
    for (file, mode, password, expected_status, source) in [
        ("scan.pdf", "auto", None, 200, "ocr"),
        ("text-with-image.pdf", "text", None, 200, "text"),
        ("hybrid.pdf", "auto", None, 200, "hybrid"),
        ("encrypted.pdf", "text", None, 400, ""),
        ("encrypted.pdf", "text", Some("wrong"), 400, ""),
        (
            "encrypted.pdf",
            "text",
            Some("fixture-password"),
            200,
            "text",
        ),
    ] {
        let mut form = reqwest::multipart::Form::new()
            .text("model", IDS[0])
            .text("mode", mode)
            .part(
                "file",
                reqwest::multipart::Part::bytes(fixture(file)).file_name(file),
            );
        if let Some(password) = password {
            form = form.text("password", password);
        }
        let response = client
            .post(format!("{base}{}", PATHS[0]))
            .multipart(form)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body = response.text().await.unwrap();
        assert_eq!(status.as_u16(), expected_status, "{file}: {body}");
        if expected_status == 200 {
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["pages"][0]["source"], source, "{file}: {value}");
        }
    }
}

#[tokio::test]
#[ignore = "requires MESH_ONNX_TEST_HOST and PDFIUM_LIBRARY"]
async fn concurrent_failed_loads_share_the_original_asset_error() {
    let empty = tempfile::tempdir().unwrap();
    let config = format!(
        "\n[[models]]\nmodel={}\nprovider=\"onnx\"\n[models.source]\nlocal_dir={}\n[models.lifecycle]\nload=\"on_demand\"\nprefetch=false\n",
        quote(IDS[2]),
        quote(empty.path().to_str().unwrap())
    );
    let mut server = Server::start(&config, None);
    let client = client();
    let provider = server.provider(&client).await;
    let request = || {
        client
            .post(format!("{provider}{}", PATHS[2]))
            .multipart(form(2))
            .send()
    };
    let (a, b) = tokio::join!(request(), request());
    let a = a.unwrap();
    let b = b.unwrap();
    assert_eq!(a.status(), 503);
    assert_eq!(b.status(), 503);
    let (a, b) = tokio::join!(a.text(), b.text());
    let a = a.unwrap();
    assert_eq!(a, b.unwrap());
    assert!(a.contains("required model asset is missing"), "{a}");
}

#[tokio::test]
#[ignore = "requires internet, MESH_ONNX_TEST_HOST, PDFIUM_LIBRARY and persistent MESH_ONNX_TEST_HF_CACHE"]
async fn host_asset_service_prefetches_without_sessions_and_reloads_offline() {
    let cache = std::path::PathBuf::from(
        std::env::var_os("MESH_ONNX_TEST_HF_CACHE")
            .expect("set persistent MESH_ONNX_TEST_HF_CACHE; assets are retained between runs"),
    );
    std::fs::create_dir_all(&cache).unwrap();
    let client = client();
    let config = |prefetch| {
        format!(
            "\n[[models]]\nmodel={}\nprovider=\"onnx\"\nvariant=\"int8\"\n[models.lifecycle]\nload=\"on_demand\"\nprefetch={prefetch}\nidle_timeout_secs=3\n",
            quote(IDS[2])
        )
    };
    let mut server = Server::start_with_cache(&config(true), None, Some(&cache), false);
    let provider = server.provider(&client).await;
    server
        .wait(&client, &format!("{provider}/v1/models"), |v| {
            availability(v, IDS[2], "loadable") && v["data"][0]["assets_ready"] == true
        })
        .await;
    let snapshot = cache_snapshot(&cache);
    assert!(!snapshot.is_empty(), "prefetch must prepare disk assets");
    infer(&client, &server.base, 2).await;
    server
        .wait(&client, &format!("{provider}/v1/models"), |v| {
            availability(v, IDS[2], "loadable")
        })
        .await;
    assert_eq!(
        cache_snapshot(&cache),
        snapshot,
        "idle eviction preserves prefetched weights"
    );
    drop(server);
    let mut offline = Server::start_with_cache(&config(false), None, Some(&cache), true);
    let provider = offline.provider(&client).await;
    offline
        .wait(&client, &format!("{provider}/v1/models"), |v| {
            availability(v, IDS[2], "loadable") && v["data"][0]["assets_ready"] == false
        })
        .await;
    infer(&client, &offline.base, 2).await;
    assert_eq!(
        cache_snapshot(&cache),
        snapshot,
        "offline cold load reuses the host-owned cache"
    );
}

#[tokio::test]
#[ignore = "requires MESH_ONNX_TEST_HOST, PDFIUM_LIBRARY and all seven pinned model bundles"]
async fn a_client_discovers_every_model_kind_and_calls_it_across_the_mesh() {
    let client = client();
    let config: String = (0..4)
        .map(|i| model_config(i, false, "on_demand", false, 3))
        .collect();
    let mut server = Server::start(&config, None);
    let provider = server.provider(&client).await;
    server
        .wait(&client, &format!("{provider}/v1/models"), |v| {
            IDS.iter().all(|id| availability(v, id, "loadable"))
        })
        .await;
    let status = server
        .wait(&client, &format!("{}/api/status", server.control), |v| {
            v["token"].as_str().is_some()
        })
        .await;
    let token = status["token"].as_str().unwrap();
    // A second Mesh node with no plugin of its own: everything it reports and
    // every request it accepts has to cross the peer transport.
    let mut remote = Server::start("", Some(token));
    // A serve node advertises its plugin models in the mesh catalog before it
    // claims the host role that makes them callable, so wait for a peer that
    // actually accepts inference for all four.
    remote
        .wait(&client, &format!("{}/api/status", remote.control), |v| {
            serving_peer(v, &IDS)
        })
        .await;
    // Discovery first, before a single request has woken an engine: a client
    // must be able to name each model and the kind of work it does.
    let inventory = remote
        .wait(&client, &format!("{}/api/models", remote.control), |v| {
            discovers_all(v)
        })
        .await;
    for (index, class) in CLASSES.iter().enumerate() {
        let entry = discovered(&inventory, index);
        assert_eq!(entry["workload_class"], *class, "{entry}");
        assert_eq!(entry["name"], IDS[index], "{entry}");
    }
    // Naming the models did not wake any of them: the provider still reports
    // every model as loadable with no session.
    server
        .wait(&client, &format!("{provider}/v1/models"), |v| {
            IDS.iter().all(|id| availability(v, id, "loadable"))
        })
        .await;
    // Each kind is then callable from the same client-only node.
    for index in 0..4 {
        infer(&client, &remote.base, index).await;
    }
    // Discovery survives the load, still naming all four with their classes.
    remote
        .wait(&client, &format!("{}/api/models", remote.control), |v| {
            discovers_all(v)
        })
        .await;
    remote.assert_alive();
    server.assert_alive();
    eprintln!("acceptance passed: four kinds discovered and called from a second node");
}
