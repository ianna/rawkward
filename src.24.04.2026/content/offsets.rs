// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Offsets(pub Arc<[i64]>);
