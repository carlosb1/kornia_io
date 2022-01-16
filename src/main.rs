use kornia_rs::*;
use std::path::PathBuf;
use std::time::SystemTime;

fn main() {
    let PATH: PathBuf = [env!("CARGO_MANIFEST_DIR"), "clients", "test.jpg"]
        .iter()
        .collect();

    let str_path = PATH.into_os_string().into_string().unwrap();
    let start = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("get millis error");
    let info = kornia_rs::read_image(str_path.clone());
}
