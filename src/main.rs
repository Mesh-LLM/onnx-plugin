use std::process::ExitCode;

fn main() -> ExitCode {
    match mesh_onnx_ocr::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("onnx-ocr: {error:#}");
            ExitCode::FAILURE
        }
    }
}
