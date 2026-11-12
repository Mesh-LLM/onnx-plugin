use std::process::ExitCode;

fn main() -> ExitCode {
    match mesh_onnx::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("onnx: {error:#}");
            ExitCode::FAILURE
        }
    }
}
