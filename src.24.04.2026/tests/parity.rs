// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::content::{Content, NumpyArray};
    use crate::kernels::{slice, Slice};

    #[test]
    fn test_slice_numpy_index() {
        let data: Arc<[f64]> = Arc::from(vec![1.0, 2.0, 3.0].into_boxed_slice());
        let arr = Content::NumpyArray(NumpyArray { data });
        let out = slice(&arr, &Slice::Index(1)).unwrap();
        if let Content::NumpyArray(a) = out {
            assert_eq!(&*a.data, &[2.0]);
        } else {
            panic!("unexpected content variant");
        }
    }
}
