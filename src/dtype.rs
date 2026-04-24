// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 => 8,
        }
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }
}

pub trait ToDType {
    fn dtype(&self) -> DType;
}

impl ToDType for bool {
    fn dtype(&self) -> DType {
        DType::Bool
    }
}

impl ToDType for i8 {
    fn dtype(&self) -> DType {
        DType::I8
    }
}

impl ToDType for i16 {
    fn dtype(&self) -> DType {
        DType::I16
    }
}

impl ToDType for i32 {
    fn dtype(&self) -> DType {
        DType::I32
    }
}

impl ToDType for i64 {
    fn dtype(&self) -> DType {
        DType::I64
    }
}

impl ToDType for u8 {
    fn dtype(&self) -> DType {
        DType::U8
    }
}

impl ToDType for u16 {
    fn dtype(&self) -> DType {
        DType::U16
    }
}

impl ToDType for u32 {
    fn dtype(&self) -> DType {
        DType::U32
    }
}

impl ToDType for u64 {
    fn dtype(&self) -> DType {
        DType::U64
    }
}

impl ToDType for f32 {
    fn dtype(&self) -> DType {
        DType::F32
    }
}

impl ToDType for f64 {
    fn dtype(&self) -> DType {
        DType::F64
    }
}
