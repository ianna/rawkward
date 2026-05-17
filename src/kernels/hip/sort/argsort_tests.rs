// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendKind, Backend};

    // CPU reference argsort for jagged lists
    fn cpu_argsort_jagged(values: &[f32], offsets: &[i64]) -> Vec<i64> {
        let mut out = vec![0i64; values.len()];

        for list_id in 0..offsets.len() - 1 {
            let start = offsets[list_id] as usize;
            let end   = offsets[list_id + 1] as usize;

            let mut pairs: Vec<(usize, f32)> =
                (start..end).map(|i| (i, values[i])).collect();

            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (rank, (idx, _)) in pairs.iter().enumerate() {
                out[start + rank] = *idx as i64;
            }
        }

        out
    }

    #[test]
    fn test_argsort_jagged_small_medium_large() {
        // Skip test if HIP backend is not available
        let backend = match Backend::new(BackendKind::Hip) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("HIP backend unavailable, skipping test");
                return;
            }
        };

        // Construct jagged data with small, medium, and large lists
        // small: 5 elements
        // medium: 120 elements
        // large: 1000 elements
        let mut values = Vec::new();
        let mut offsets = vec![0i64];

        // small list
        values.extend_from_slice(&[3.0, 1.0, 2.0, 5.0, 4.0]);
        offsets.push(values.len() as i64);

        // medium list
        for i in 0..120 {
            values.push((i as f32).sin());
        }
        offsets.push(values.len() as i64);

        // large list
        for i in 0..1000 {
            values.push(((i * 17) as f32).cos());
        }
        offsets.push(values.len() as i64);

        let mut gpu_out = vec![0i64; values.len()];
        let cpu_out = cpu_argsort_jagged(&values, &offsets);

        argsort_jagged(&backend, &values, &offsets, &mut gpu_out);

        assert_eq!(gpu_out, cpu_out, "HIP argsort_jagged mismatch");
    }

    #[test]
    fn test_argsort_jagged_empty_lists() {
        let backend = match Backend::new(BackendKind::Hip) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("HIP backend unavailable, skipping test");
                return;
            }
        };

        let values: Vec<f32> = vec![];
        let offsets: Vec<i64> = vec![0];

        let mut gpu_out = vec![];
        let cpu_out = cpu_argsort_jagged(&values, &offsets);

        argsort_jagged(&backend, &values, &offsets, &mut gpu_out);

        assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn test_argsort_jagged_singleton_lists() {
        let backend = match Backend::new(BackendKind::Hip) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("HIP backend unavailable, skipping test");
                return;
            }
        };

        let values = vec![10.0, 20.0, 30.0];
        let offsets = vec![0, 1, 2, 3];

        let mut gpu_out = vec![0i64; 3];
        let cpu_out = cpu_argsort_jagged(&values, &offsets);

        argsort_jagged(&backend, &values, &offsets, &mut gpu_out);

        assert_eq!(gpu_out, cpu_out);
    }
}

