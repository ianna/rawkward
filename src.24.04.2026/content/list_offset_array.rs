// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

use crate::content::Content;
use crate::layout::Layout;

#[derive(Clone, Debug)]
pub struct ListOffsetArray {
    pub offsets: Arc<[i64]>,
    pub content: Arc<Content>,
}

// ── Constructor ───────────────────────────────────────────────────────────────

impl ListOffsetArray {
    pub fn new(offsets: Arc<[i64]>, content: Arc<Content>) -> Self {
        assert!(
            !offsets.is_empty(),
            "ListOffsetArray offsets must have at least one element (the leading 0)"
        );
        ListOffsetArray { offsets, content }
    }

    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the sub-list at `index` as a Content, or None if out of bounds.
    pub fn get(&self, index: usize) -> Option<Content> {
        if index >= self.len() {
            return None;
        }
        let start = self.offsets[index] as usize;
        let stop = self.offsets[index + 1] as usize;
        Some((*self.content.slice_arc(start, stop)).clone())
    }
}

// ── Layout impl ───────────────────────────────────────────────────────────────

impl Layout for ListOffsetArray {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Return the sub-content for list at `index` as a zero-copy Arc slice.
    fn getitem(&self, index: usize) -> Arc<dyn Layout> {
        let start = self.offsets[index] as usize;
        let stop = self.offsets[index + 1] as usize;
        // Zero-copy: slice_arc shares the same Arc<[T]> backing buffer.
        self.content.slice_arc(start, stop)
    }

    /// Return a sub-array covering lists `start..stop`.
    /// Offsets are re-based to 0; content is a zero-copy Arc sub-slice.
    fn slice(&self, start: usize, stop: usize) -> Arc<dyn Layout> {
        Arc::new(self.slice_arc(start, stop))
    }

    /// Return a typetracer: same structure, empty data buffers.
    fn typetracer(&self) -> Arc<dyn Layout> {
        Arc::new(ListOffsetArray {
            // Single sentinel offset — no lists, no data.
            offsets: Arc::from([0_i64] as [i64; 1]),
            // Propagate typetracer down without cloning data.
            content: self.content.typetracer_arc(),
        })
    }
}

// ── Zero-copy helpers (return Arc<Content> for internal use) ──────────────────

impl ListOffsetArray {
    /// Like `Layout::slice` but returns a concrete `Arc<Content>` instead of
    /// `Arc<dyn Layout>`, so callers can store the result directly in struct
    /// fields without a downcast.
    ///
    /// Zero-copy: the new offsets are a fresh small Vec, but the content
    /// sub-slice shares the same Arc-backed buffer as `self.content`.
    pub fn slice_arc(&self, start: usize, stop: usize) -> ListOffsetArray {
        // Re-base offsets so the new array starts at 0.
        let base = self.offsets[start];
        let new_offsets: Arc<[i64]> = self.offsets[start..=stop]
            .iter()
            .map(|&o| o - base)
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .into();

        // Content sub-slice: zero-copy share of the backing buffer.
        let content_start = self.offsets[start] as usize;
        let content_stop = self.offsets[stop] as usize;
        let new_content = self.content.slice_arc(content_start, content_stop);

        ListOffsetArray {
            offsets: new_offsets,
            content: new_content,
        }
    }

    /// Convenience: wrap `slice_arc` result in `Arc<Content>`.
    pub fn slice_content_arc(&self, start: usize, stop: usize) -> Arc<Content> {
        Arc::new(Content::ListOffsetArray(self.slice_arc(start, stop)))
    }
}
