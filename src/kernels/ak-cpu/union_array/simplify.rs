// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Simplify two nested UnionArrays into a single level.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_simplify.cpp`.

/// For each `i` where `outertags[i] == outerwhich` and
/// `innertags[outerindex[i]] == innerwhich`, write:
/// * `totags[i] = towhich`
/// * `toindex[i] = innerindex[outerindex[i]] + base`
pub fn union_array_simplify<OT, OI, IT, II>(
    totags: &mut [i8],
    toindex: &mut [i64],
    outertags: &[OT],
    outerindex: &[OI],
    innertags: &[IT],
    innerindex: &[II],
    towhich: i64,
    innerwhich: i64,
    outerwhich: i64,
    base: i64,
) where
    OT: Copy + Into<i64>,
    OI: Copy + Into<i64>,
    IT: Copy + Into<i64>,
    II: Copy + Into<i64>,
{
    for (i, (&ot, &oi)) in outertags.iter().zip(outerindex.iter()).enumerate() {
        if ot.into() == outerwhich {
            let j = oi.into() as usize;
            if innertags[j].into() == innerwhich {
                totags[i] = towhich as i8;
                toindex[i] = innerindex[j].into() + base;
            }
        }
    }
}

macro_rules! impl_simplify {
    ($fn_name:ident, $ot:ty, $oi:ty, $it:ty, $ii:ty) => {
        pub fn $fn_name(
            totags: &mut [i8],
            toindex: &mut [i64],
            outertags: &[$ot],
            outerindex: &[$oi],
            innertags: &[$it],
            innerindex: &[$ii],
            towhich: i64,
            innerwhich: i64,
            outerwhich: i64,
            base: i64,
        ) {
            union_array_simplify(
                totags, toindex, outertags, outerindex, innertags, innerindex, towhich, innerwhich,
                outerwhich, base,
            )
        }
    };
}

impl_simplify!(union_array8_32_simplify8_32_to8_64, i8, i32, i8, i32);
impl_simplify!(union_array8_32_simplify8_u32_to8_64, i8, i32, i8, u32);
impl_simplify!(union_array8_32_simplify8_64_to8_64, i8, i32, i8, i64);
impl_simplify!(union_array8_u32_simplify8_32_to8_64, i8, u32, i8, i32);
impl_simplify!(union_array8_u32_simplify8_u32_to8_64, i8, u32, i8, u32);
impl_simplify!(union_array8_u32_simplify8_64_to8_64, i8, u32, i8, i64);
impl_simplify!(union_array8_64_simplify8_32_to8_64, i8, i64, i8, i32);
impl_simplify!(union_array8_64_simplify8_u32_to8_64, i8, i64, i8, u32);
impl_simplify!(union_array8_64_simplify8_64_to8_64, i8, i64, i8, i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // outer: tags=[0,1,0], index=[0,0,1]; inner for tag=0: tags=[1,0], index=[10,20]
        // We want outerwhich=0, innerwhich=0, towhich=2, base=0
        let outertags = [0i8, 1, 0];
        let outerindex = [0i64, 0, 1];
        let innertags = [1i8, 0]; // position 1 has inner tag=0
        let innerindex = [10i64, 20];
        let mut totags = [0i8; 3];
        let mut toindex = [0i64; 3];
        union_array8_64_simplify8_64_to8_64(
            &mut totags,
            &mut toindex,
            &outertags,
            &outerindex,
            &innertags,
            &innerindex,
            2,
            0,
            0,
            0,
        );
        // i=0: outer[0]=0=outerwhich, inner[0] tag=1≠0 → skip
        // i=2: outer[2]=0=outerwhich, outer_idx=1, inner[1] tag=0=innerwhich → write
        assert_eq!(totags[2], 2);
        assert_eq!(toindex[2], 20);
    }

    #[test]
    fn with_base() {
        let outertags = [0i8];
        let outerindex = [0i32];
        let innertags = [0i8];
        let innerindex = [5i32];
        let mut totags = [0i8; 1];
        let mut toindex = [0i64; 1];
        union_array8_32_simplify8_32_to8_64(
            &mut totags,
            &mut toindex,
            &outertags,
            &outerindex,
            &innertags,
            &innerindex,
            1,
            0,
            0,
            100,
        );
        assert_eq!(totags[0], 1);
        assert_eq!(toindex[0], 105);
    }
}
