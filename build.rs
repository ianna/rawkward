use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let python_feature = env::var("CARGO_FEATURE_PYTHON").is_ok();
    let bench_cxx_feature = env::var("CARGO_FEATURE_BENCH_CXX").is_ok();
    let hip_feature = env::var("CARGO_FEATURE_HIP").is_ok();

    // 1. macOS dynamic Python linking
    if python_feature && target_os == "macos" {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }

    // 2. HIP detection
    let hip_available = detect_hip();
    println!("cargo:rustc-env=RAWKWARD_HIP_AVAILABLE={}", hip_available);

    if hip_feature && hip_available {
        println!("cargo:warning=ROCm/HIP detected — enabling GPU kernels");
        compile_hip_kernels();
    } else if hip_feature {
        println!("cargo:warning=HIP feature enabled but ROCm not found — building CPU-only");
    }

    // 3. Optional C++ benchmark kernels
    if bench_cxx_feature {
        compile_awkward_bench_cxx();
    }

    // 4. Rebuild triggers
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=benches/awkward_bench_wrappers.cpp");
    println!("cargo:rerun-if-env-changed=AWKWARD_CPP_PATH");
    println!("cargo:rerun-if-env-changed=AWKWARD_CPP_VERSION");
}

fn detect_hip() -> bool {
    let hipcc_exists = Command::new("which")
        .arg("hipcc")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let rocm_lib_exists =
        Path::new("/opt/rocm/lib").exists() ||
        Path::new("/opt/rocm-7.2.1/lib").exists() ||
        Path::new("/opt/rocm-7.1.0/lib").exists();

    hipcc_exists && rocm_lib_exists
}

fn compile_hip_kernels() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let hip_sources = [
        "src/kernels/hip/reduce/argmin.hip.cpp",
        "src/kernels/hip/reduce/argmax.hip.cpp",
        "src/kernels/hip/reduce/sum.hip.cpp",
        "src/kernels/hip/reduce/min.hip.cpp",
        "src/kernels/hip/reduce/max.hip.cpp",
    ];

    for src in hip_sources {
        let status = Command::new("hipcc")
            .args(&[
                "-O3",
                "--std=c++17",
                "-fPIC",
                "-c",
                src,
                "-o",
                &format!("{out_dir}/{}.o", Path::new(src).file_stem().unwrap().to_str().unwrap()),
            ])
            .status()
            .expect("Failed to run hipcc");

        if !status.success() {
            panic!("hipcc failed to compile {src}");
        }
    }

    let mut ar = Command::new("ar");
    ar.arg("crus")
        .arg(format!("{out_dir}/libawkward_hip_kernels.a"));

    for src in hip_sources {
        let obj = format!("{out_dir}/{}.o", Path::new(src).file_stem().unwrap().to_str().unwrap());
        ar.arg(obj);
    }

    let status = ar.status().expect("Failed to run ar");
    if !status.success() {
        panic!("Failed to archive HIP kernels");
    }

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=awkward_hip_kernels");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
