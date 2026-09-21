// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// CUDA counterpart of `hip_args!`.  Materializes each argument into a named
// local before taking its address, so the `void**` kernel-parameter pointers
// stay valid through the `cuLaunchKernel` call in the same scope.  Backend-
// agnostic: it only builds a `[*mut c_void; N]` array of pointers-to-args.
#[macro_export]
macro_rules! cuda_args {
    ($name:ident; $a0:expr, $a1:expr, $a2:expr $(,)?) => {
        let __c0 = $a0;
        let __c1 = $a1;
        let __c2 = $a2;
        let $name: [*mut ::std::ffi::c_void; 3] = [
            &__c0 as *const _ as *mut _,
            &__c1 as *const _ as *mut _,
            &__c2 as *const _ as *mut _,
        ];
    };
    ($name:ident; $a0:expr, $a1:expr, $a2:expr, $a3:expr $(,)?) => {
        let __c0 = $a0;
        let __c1 = $a1;
        let __c2 = $a2;
        let __c3 = $a3;
        let $name: [*mut ::std::ffi::c_void; 4] = [
            &__c0 as *const _ as *mut _,
            &__c1 as *const _ as *mut _,
            &__c2 as *const _ as *mut _,
            &__c3 as *const _ as *mut _,
        ];
    };
    ($name:ident; $a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr $(,)?) => {
        let __c0 = $a0;
        let __c1 = $a1;
        let __c2 = $a2;
        let __c3 = $a3;
        let __c4 = $a4;
        let $name: [*mut ::std::ffi::c_void; 5] = [
            &__c0 as *const _ as *mut _,
            &__c1 as *const _ as *mut _,
            &__c2 as *const _ as *mut _,
            &__c3 as *const _ as *mut _,
            &__c4 as *const _ as *mut _,
        ];
    };
}
