// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::process::Command;

pub fn hip_available() -> bool {
    // `which hipcc` must both run *and* report success (exit 0). `.output()`
    // only errors when the process fails to spawn, so a "not found" result
    // (which exits non-zero) still returns Ok — check the status explicitly.
    let hipcc_found = Command::new("which")
        .arg("hipcc")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !hipcc_found {
        return false;
    }

    // Confirm the toolchain actually runs.
    Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
