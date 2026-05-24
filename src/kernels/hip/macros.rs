// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#[macro_export]
macro_rules! hip_args {
    ($name:ident; $a0:expr, $a1:expr, $a2:expr $(,)?) => {
        let __h0 = $a0;
        let __h1 = $a1;
        let __h2 = $a2;
        let $name: [*mut ::std::ffi::c_void; 3] = [
            &__h0 as *const _ as *mut _,
            &__h1 as *const _ as *mut _,
            &__h2 as *const _ as *mut _,
        ];
    };
    ($name:ident; $a0:expr, $a1:expr, $a2:expr, $a3:expr $(,)?) => {
        let __h0 = $a0;
        let __h1 = $a1;
        let __h2 = $a2;
        let __h3 = $a3;
        let $name: [*mut ::std::ffi::c_void; 4] = [
            &__h0 as *const _ as *mut _,
            &__h1 as *const _ as *mut _,
            &__h2 as *const _ as *mut _,
            &__h3 as *const _ as *mut _,
        ];
    };
    ($name:ident; $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr $(,)?) => {
        let __h0 = $a0;
        let __h1 = $a1;
        let __h2 = $a2;
        let __h3 = $a3;
        let __h4 = $a4;
        let $name: [*mut ::std::ffi::c_void; 5] = [
            &__h0 as *const _ as *mut _,
            &__h1 as *const _ as *mut _,
            &__h2 as *const _ as *mut _,
            &__h3 as *const _ as *mut _,
            &__h4 as *const _ as *mut _,
        ];
    };
}
