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

    // 2. HIP detection and kernel compilation
    let hip_available = detect_hip();
    println!("cargo:rustc-env=RAWKWARD_HIP_AVAILABLE={}", hip_available);
    if hip_feature && hip_available {
        println!("cargo:warning=ROCm/HIP detected — enabling GPU kernels");
        compile_hip_kernels();
    } else if hip_feature {
        println!("cargo:warning=HIP feature enabled but ROCm not found — building CPU-only");
    }

    // 2b. Generate HIP bindings for the Rust backend
    if hip_feature && hip_available {
        generate_hip_bindings();
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

fn compile_hip_kernels() {
    let hip_path = env::var("HIP_PATH")
        .or_else(|_| env::var("ROCM_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    let hip_include = format!("{}/include", hip_path);
    let hip_lib = format!("{}/lib", hip_path);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut hip_files: Vec<String> = Vec::new();
    collect_hip_files("src/kernels/hip", &mut hip_files);
    hip_files.sort();

    if hip_files.is_empty() {
        panic!("No .hip.cpp files found under src/kernels/hip/");
    }

    // Generate a unity translation unit so --genco can produce one output.
    // --genco requires a single input when -o is specified.
    let unity_path = out_dir.join("_all_hip_kernels.hip.cpp");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&unity_path).unwrap();
        for file in &hip_files {
            let abs = std::fs::canonicalize(file).unwrap_or_else(|_| PathBuf::from(file));
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

    // .hsaco is a proper HSA code object — works with both
    // hipModuleLoad (path) and hipModuleLoadData (embedded bytes).
    println!(
        "cargo:rustc-env=RAWKWARD_HIP_MODULE_PATH={}",
        hsaco_path.display()
    );
    println!("cargo:rustc-link-search=native={}", hip_lib);
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

fn compile_hip_kernels_shared() {
    // Resolve HIP include/lib paths from env, falling back to /opt/rocm
    let hip_path = env::var("HIP_PATH")
        .or_else(|_| env::var("ROCM_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    let hip_include = format!("{}/include", hip_path);
    let hip_lib = format!("{}/lib", hip_path);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Collect all .hip/.cpp files recursively — picks up new kernels automatically
    let mut hip_files = Vec::new();
    collect_hip_files("src/kernels/hip", &mut hip_files);

    let mut object_files = Vec::new();
    for file in &hip_files {
        let obj_path = out_dir.join(format!(
            "{}.o",
            Path::new(file).file_stem().unwrap().to_string_lossy()
        ));
        let status = Command::new("hipcc")
            .args([
                "-O3",
                "--std=c++17",
                "-fPIC",
                "-c",
                &format!("-I{}", hip_include),
                file,
                "-o",
                obj_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run hipcc");
        if !status.success() {
            panic!("hipcc failed to compile {}", file);
        }
        object_files.push(obj_path);
    }

    let lib_path = out_dir.join("libawkward_hip_kernels.a");
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib_path);
    for obj in &object_files {
        ar.arg(obj);
    }
    if !ar.status().expect("Failed to run ar").success() {
        panic!("Failed to archive HIP kernels");
    }

    // Produce the runtime-loadable module for hipModuleLoadData.
    //let hsaco_path = out_dir.join("rawkward_kernels.hsaco");
    let module_path = out_dir.join("librawkward_kernels.so");
    let mut cmd = Command::new("hipcc");
    cmd.args([
        "-O3",
        "--std=c++17",
        "-fPIC",
        "-shared",
        &format!("-I{}", hip_include),
    ]);
    for file in &hip_files {
        cmd.arg(file);
    }
    cmd.args(["-o", module_path.to_str().unwrap()]);
    if !cmd
        .status()
        .expect("hipcc -shared failed to launch")
        .success()
    {
        panic!("hipcc -shared failed — check kernel sources compile cleanly");
    }
    println!(
        "cargo:rustc-env=RAWKWARD_HIP_MODULE_PATH={}",
        module_path.display()
    );
    println!("cargo:rustc-link-search=native={}", hip_lib);
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

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
                out.push(path.to_str().unwrap().to_string());
            }
        }
    }
}

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

fn generate_hip_bindings() {
    use bindgen;

    let hip_path = env::var("HIP_PATH")
        .or_else(|_| env::var("ROCM_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());

    let hip_include = format!("{}/include", hip_path);

    let bindings = bindgen::Builder::default()
        .header(format!("{}/hip/hip_runtime_api.h", hip_include))
        .clang_arg(format!("-I{}", hip_include))
        .clang_arg("-D__HIP_PLATFORM_AMD__=1")
        // Only generate HIP symbols
        .allowlist_function("hip.*")
        .allowlist_type("hip.*")
        .allowlist_var("hip.*")
        // ROCm headers contain inline C++ → needed for bindgen 0.69
        .clang_arg("-x")
        .clang_arg("c++")
        // Avoid layout tests (ROCm headers break them)
        .layout_tests(false)
        // Add allow attributes directly into generated file
        .raw_line("pub mod hip_bindings {")
        .raw_line("    #![allow(non_camel_case_types)]")
        .raw_line("    #![allow(non_snake_case)]")
        .raw_line("    #![allow(non_upper_case_globals)]")
        .raw_line("    #![allow(dead_code)]")
        .raw_line("    #![allow(unsafe_op_in_unsafe_fn)]")
        .raw_line("    #![allow(improper_ctypes)]")
        .raw_line("    #![allow(improper_ctypes_definitions)]")
        .raw_line("    #![allow(clippy::all)]")
        // This is the key: forces bindgen to mark all externs unsafe
        .generate_inline_functions(true)
        .trust_clang_mangling(false)
        .use_core()
        .generate()
        .expect("Unable to generate HIP bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("hip_bindings.rs");

    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write HIP bindings");
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
