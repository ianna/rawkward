// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::process::Command;

pub fn hip_available() -> bool {
    // Check if hipcc exists
    if Command::new("which").arg("hipcc").output().is_err() {
        return false;
    }

    // Try running a trivial HIP runtime call
    let out = Command::new("hipcc").arg("--version").output();

    out.map(|o| o.status.success()).unwrap_or(false)
}
