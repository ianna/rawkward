use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let hip_path = env::var("HIP_PATH")
        .or_else(|_| env::var("ROCM_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());

    let hip_include = format!("{}/include", hip_path);
    let hip_lib = format!("{}/lib", hip_path);

    println!("cargo:rustc-link-search=native={}", hip_lib);
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    let mut hip_files = Vec::new();
    collect_hip_files("src/kernels/hip", &mut hip_files);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut object_files = Vec::new();

    for file in &hip_files {
        let obj_path = out_dir.join(
            Path::new(file)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .replace(".cpp", ".o"),
        );

        let status = Command::new("hipcc")
            .args([
                "-c",
                "-O3",
                "--std=c++17",
                "-fPIC",
                &format!("-I{}", hip_include),
                file,
                "-o",
                obj_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run hipcc");

        if !status.success() {
            panic!("hipcc failed on file {}", file);
        }

        object_files.push(obj_path);
    }

    // Archive all .o files into a static library
    let lib_path = out_dir.join("libawkward_hip_kernels.a");

    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib_path);

    for obj in &object_files {
        ar.arg(obj);
    }

    let status = ar.status().expect("Failed to run ar");
    if !status.success() {
        panic!("ar failed to create static library");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=awkward_hip_kernels");
}

fn collect_hip_files(dir: &str, out: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            collect_hip_files(path.to_str().unwrap(), out);
        } else if let Some(ext) = path.extension() {
            if ext == "cpp" || ext == "hip" {
                out.push(path.to_str().unwrap().to_string());
            }
        }
    }
}
