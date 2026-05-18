// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

    // 2. HIP kernel compilation + bindings
    if hip_feature {
        let hip_available = detect_hip();
        println!("cargo:rustc-env=RAWKWARD_HIP_AVAILABLE={}", hip_available);
        if hip_available {
            println!("cargo:warning=ROCm/HIP detected — enabling GPU kernels");
            compile_hip_kernels();
            generate_hip_bindings();
        } else {
            println!("cargo:warning=HIP feature enabled but ROCm not found — building CPU-only");
        }
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
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}

// ---------------------------------------------------------------------------
// HIP detection
// ---------------------------------------------------------------------------

fn detect_hip() -> bool {
    let hipcc_exists = Command::new("which")
        .arg("hipcc")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let rocm_lib_exists = Path::new("/opt/rocm/lib").exists()
        || glob::glob("/opt/rocm-*/lib").unwrap().next().is_some();

    hipcc_exists && rocm_lib_exists
}

// ---------------------------------------------------------------------------
// HIP kernel compilation (unity build → .hsaco)
// ---------------------------------------------------------------------------

fn compile_hip_kernels() {
    let hip_path = env::var("HIP_PATH")
        .or_else(|_| env::var("ROCM_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    let hip_include = format!("{}/include", hip_path);
    let hip_lib = format!("{}/lib", hip_path);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut hip_files: Vec<String> = Vec::new();
    collect_hip_files("src/kernels/hip", &mut hip_files);
    hip_files.sort(); // deterministic include order

    if hip_files.is_empty() {
        panic!("No .hip.cpp files found under src/kernels/hip/");
    }

    // Generate a unity translation unit so --genco can accept a single input.
    // --genco produces a device-only .hsaco (HSA code object) which is the
    // format hipModuleLoad / hipModuleLoadData require.
    let unity_path = out_dir.join("_all_hip_kernels.hip.cpp");
    {
        use std::io::Write;
        let mut f = fs::File::create(&unity_path).unwrap();
        for file in &hip_files {
            let abs = fs::canonicalize(file).unwrap_or_else(|_| PathBuf::from(file));
            writeln!(f, "#include \"{}\"", abs.display()).unwrap();
        }
    }

    let hsaco_path = out_dir.join("rawkward_kernels.hsaco");
    let status = Command::new("hipcc")
        .args([
            "--genco",
            "-O3",
            "--std=c++17",
            &format!("-I{}", hip_include),
        ])
        .arg(&unity_path)
        .args(["-o", hsaco_path.to_str().unwrap()])
        .status()
        .expect("hipcc --genco failed to launch");

    if !status.success() {
        panic!("hipcc --genco failed — check that all kernel sources compile cleanly");
    }

    println!(
        "cargo:rustc-env=RAWKWARD_HIP_MODULE_PATH={}",
        hsaco_path.display()
    );
    println!("cargo:rustc-link-search=native={}", hip_lib);
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

// ---------------------------------------------------------------------------
// HIP source discovery
// ---------------------------------------------------------------------------

/// Recursively collect .hip.cpp (and plain .hip) sources under `dir`.
/// Emits `cargo:rerun-if-changed` for each file so that Cargo rebuilds
/// when any kernel source is edited.
fn collect_hip_files(dir: &str, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_hip_files(path.to_str().unwrap(), out);
        } else if let Some(ext) = path.extension() {
            if ext == "cpp" || ext == "hip" {
                let s = path.to_str().unwrap().to_string();
                println!("cargo:rerun-if-changed={}", s);
                out.push(s);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bindgen: HIP runtime API bindings
// ---------------------------------------------------------------------------

fn generate_hip_bindings() {
    let hip_path = env::var("HIP_PATH")
        .or_else(|_| env::var("ROCM_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    let hip_include = format!("{}/include", hip_path);

    let bindings = bindgen::Builder::default()
        .header(format!("{}/hip/hip_runtime_api.h", hip_include))
        .clang_arg(format!("-I{}", hip_include))
        .clang_arg("-D__HIP_PLATFORM_AMD__=1")
        // Parse as C++ — ROCm headers contain inline C++
        .clang_arg("-x")
        .clang_arg("c++")
        // Only generate HIP symbols
        .allowlist_function("hip.*")
        .allowlist_type("hip.*")
        .allowlist_var("hip.*")
        // Avoid layout tests that break on ROCm's non-standard types
        .layout_tests(false)
        // Wrap everything in a module so the use-site can re-export selectively
        .raw_line("pub mod hip_bindings {")
        .raw_line("    #![allow(non_camel_case_types)]")
        .raw_line("    #![allow(non_snake_case)]")
        .raw_line("    #![allow(non_upper_case_globals)]")
        .raw_line("    #![allow(dead_code)]")
        .raw_line("    #![allow(unsafe_op_in_unsafe_fn)]")
        .raw_line("    #![allow(improper_ctypes)]")
        .raw_line("    #![allow(improper_ctypes_definitions)]")
        .raw_line("    #![allow(clippy::all)]")
        .generate_inline_functions(true)
        .trust_clang_mangling(false)
        .use_core()
        .generate()
        .expect("Unable to generate HIP bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("hip_bindings.rs");

    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write HIP bindings");

    // Append the closing brace for the pub mod hip_bindings { ... } wrapper
    // opened by the raw_line calls above.
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut f = OpenOptions::new().append(true).open(&out_path).unwrap();
        writeln!(f, "}}").unwrap();
    }

    println!(
        "cargo:rustc-env=RAWKWARD_HIP_BINDINGS={}",
        out_path.display()
    );
}

// ---------------------------------------------------------------------------
// Optional: awkward-cpp C++ benchmark kernels
// ---------------------------------------------------------------------------

fn compile_awkward_bench_cxx() {
    let awkward_cpp_path =
        env::var("AWKWARD_CPP_PATH").unwrap_or_else(|_| "/usr/local".to_string());
    let include = format!("{}/include", awkward_cpp_path);
    let lib = format!("{}/lib", awkward_cpp_path);

    let out_dir = env::var("OUT_DIR").unwrap();
    let obj = format!("{out_dir}/awkward_bench_wrappers.o");

    let status = Command::new("c++")
        .args([
            "-O2",
            "--std=c++17",
            "-fPIC",
            "-c",
            &format!("-I{}", include),
            "benches/awkward_bench_wrappers.cpp",
            "-o",
            &obj,
        ])
        .status()
        .expect("Failed to run c++");
    if !status.success() {
        panic!("c++ failed to compile awkward_bench_wrappers.cpp");
    }

    let lib_out = format!("{out_dir}/libawkward_bench.a");
    let status = Command::new("ar")
        .args(["crus", &lib_out, &obj])
        .status()
        .expect("Failed to run ar");
    if !status.success() {
        panic!("ar failed to archive awkward_bench_wrappers");
    }

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-lib=static=awkward_bench");
    println!("cargo:rustc-link-lib=dylib=awkward-cpp");
}
