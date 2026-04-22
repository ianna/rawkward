// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

pub trait Layout: Send + Sync + std::fmt::Debug {
    /// Returns the number of elements in the current layout level.
    fn len(&self) -> usize;

    /// Retrieves a single element (or child layout) at the specified index.
    fn getitem(&self, index: usize) -> Arc<dyn Layout>;

    /// Returns a new layout representing a view of the data from
    /// `start` (inclusive) to `stop` (exclusive).
    fn slice(&self, start: usize, stop: usize) -> Arc<dyn Layout>;

    /// Returns a "type-only" version of the layout.
    /// This is used for shape-checking and metadata operations
    /// without involving the actual underlying data buffers.
    fn typetracer(&self) -> Arc<dyn Layout>;
}
