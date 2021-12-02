#![recursion_limit="65536"]
#![feature(trivial_bounds)]
#![feature(allocator_api)]
#![feature(non_exhaustive)]
#![feature(wait_until)]
#![feature(maybe_uninit_ref)]

#[macro_use] extern crate gl_struct;

extern crate ar_engine;
extern crate maths_traits;
extern crate numerical_integration;
extern crate free_algebra;

pub mod soft_body;
pub mod profiler;


pub static mut LOGGING: bool = true;
pub(crate) static mut PROFILER: Option<crate::profiler::Profiler> = None;
