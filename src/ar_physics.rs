#![recursion_limit="65536"]
#![feature(weak_ptr_eq)]
#![feature(trivial_bounds)]
#![feature(allocator_api)]
#![feature(non_exhaustive)]


#[macro_use] extern crate macro_program;
#[macro_use] extern crate gl_struct;

extern crate ar_engine;
extern crate maths_traits;
extern crate numerical_integration;
extern crate free_algebra;

pub mod soft_body;
pub mod profiler;


pub(crate) static mut PROFILER: Option<crate::profiler::Profiler> = None;
