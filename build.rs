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
    // Note: awkward_bench_wrappers.cpp and individual kernel sources are
    // registered inside compile_awkward_bench_cxx when bench-cxx is on.
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
//
// Two separate compilation units:
//
//   librawkward_argsort_cxx  — standalone argsort, no external headers.
//                              Always built.  Powers `argsort_cpu` bench.
//
//   libawkward_bench         — full awkward CPU-kernel wrappers.
//                              Only built when awkward/kernels.h is found.
//                              Powers `kernels` bench C++ comparisons.
//
// Header search order (for libawkward_bench only):
//   1. AWKWARD_CPP_INCLUDE env var (explicit path override, e.g.
//      ~/Projects/awkward.2.9.x/awkward/awkward-cpp/include)
//   2. `python3 -c "import awkward_cpp; …"` (pip / conda install)
//
// Source directory (must match the headers' failure() arity):
//   When headers are found via a local checkout, the sibling
//   src/cpu-kernels/ directory is used so sources and headers are
//   always co-versioned.  Falls back to src/kernels/awkward-cpp/
//   (in-tree copies) when the sibling dir doesn't exist.
//
// The pip wheel for awkward_cpp ≥ v50 does not ship C++ headers, so
// libawkward_bench may not compile on all machines.  librawkward_argsort_cxx
// has no such dependency and always works.

fn compile_awkward_bench_cxx() {
    // ── Part 1: standalone argsort (no headers, always works) ────────────
    let out_dir = env::var("OUT_DIR").unwrap();

    cc::Build::new()
        .cpp(true)
        .opt_level(2)
        .std("c++17")
        .flag("-fPIC")
        .file("benches/argsort_cxx_impl.cpp")
        .compile("rawkward_argsort_cxx");

    // Emit link directives explicitly — some cc versions only emit them via
    // the Library's Drop impl, which can be silently skipped.
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=rawkward_argsort_cxx");

    println!("cargo:rerun-if-changed=benches/argsort_cxx_impl.cpp");
    println!("cargo:rerun-if-env-changed=AWKWARD_CPP_INCLUDE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    // ── Part 2: full awkward bench lib (needs headers) ────────────────────
    let include_dir = match try_find_awkward_cpp_include() {
        Some(d) => d,
        None => {
            println!(
                "cargo:warning=bench-cxx: awkward/kernels.h not found — \
                 `kernels` bench C++ comparisons disabled. \
                 Set AWKWARD_CPP_INCLUDE to the directory containing \
                 awkward/kernels.h to enable them."
            );
            return; // argsort bench still works via Part 1
        }
    };

    // Prefer the cpu-kernels sources co-located with the found headers so that
    // the failure() call-site arity always matches what common.h declares.
    //
    // Checkout layout:
    //   awkward-cpp/include/   ← include_dir points here
    //   awkward-cpp/src/cpu-kernels/  ← co-versioned sources
    //
    // Fall back to the in-tree copies (src/kernels/awkward-cpp/) only when the
    // sibling directory doesn't exist (e.g. a pip include-only install).
    let ext_cpu_kernels = include_dir
        .parent() // …/awkward-cpp
        .map(|p| p.join("src/cpu-kernels"));

    let (src_dir, src_label): (PathBuf, &str) = match ext_cpu_kernels {
        Some(ref d) if d.is_dir() => (d.clone(), "checkout src/cpu-kernels"),
        _ => (
            PathBuf::from("src/kernels/awkward-cpp"),
            "in-tree src/kernels/awkward-cpp",
        ),
    };

    println!(
        "cargo:warning=bench-cxx: kernel sources → {} ({})",
        src_dir.display(),
        src_label,
    );

    let mut kernel_srcs: Vec<PathBuf> = fs::read_dir(&src_dir)
        .unwrap_or_else(|e| panic!("Cannot read kernel source dir {}: {}", src_dir.display(), e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |e| e == "cpp"))
        .collect();
    kernel_srcs.sort();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .opt_level(2)
        .std("c++17")
        .flag("-fPIC")
        .include(&include_dir)
        .file("benches/awkward_bench_wrappers.cpp");

    for src in &kernel_srcs {
        build.file(src);
        println!("cargo:rerun-if-changed={}", src.display());
    }

    build.compile("awkward_bench");

    println!(
        "cargo:rustc-link-search=native={}",
        env::var("OUT_DIR").unwrap()
    );
    println!("cargo:rustc-link-lib=static=awkward_bench");

    println!("cargo:rerun-if-changed=benches/awkward_bench_wrappers.cpp");
}

/// Try to locate the directory that contains `awkward/kernels.h`.
/// Returns `None` (never panics) so the caller can degrade gracefully.
fn try_find_awkward_cpp_include() -> Option<PathBuf> {
    // 1. Explicit override.
    if let Ok(path) = env::var("AWKWARD_CPP_INCLUDE") {
        let p = PathBuf::from(&path);
        if p.join("awkward/kernels.h").exists() {
            return Some(p);
        }
        println!(
            "cargo:warning=AWKWARD_CPP_INCLUDE={path:?} set but \
             awkward/kernels.h not found there"
        );
        return None;
    }

    // 2. Pip-installed package — ask Python for the package location.
    let python = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    if let Ok(out) = Command::new(&python)
        .args([
            "-c",
            "import awkward_cpp, os; \
             print(os.path.join(os.path.dirname(\
               os.path.abspath(awkward_cpp.__file__)), 'include'))",
        ])
        .output()
    {
        if out.status.success() {
            let inc = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let p = PathBuf::from(&inc);
            if p.join("awkward/kernels.h").exists() {
                println!("cargo:warning=awkward-cpp include: {inc}");
                return Some(p);
            }
        }
    }

    None
}
