use gl_struct::*;

use std::mem::{swap, forget, MaybeUninit};
use std::ptr::copy_nonoverlapping;

pub struct BufVec<T:GPUCopy> {
    buf: Buffer<[T],ReadWrite>,
    len: usize
}

impl<T:GPUCopy> BufVec<T> {

    pub fn new(gl:&GLProvider) -> Self { Self::with_capacity(gl, 1) }

    pub fn with_capacity(gl:&GLProvider, cap:usize) -> Self {
        BufVec {
            buf: unsafe { Buffer::<[_],_>::uninitialized(gl, cap.max(1)) },
            len: 0
        }
    }

    pub fn len(&self) -> usize {self.len}
    pub fn capacity(&self) -> usize {self.len}

    pub fn as_slice(&self) -> BSlice<[T],ReadWrite> { self.buf.slice(0..self.len) }
    pub fn as_slice_mut(&mut self) -> BSliceMut<[T],ReadWrite> { self.buf.slice_mut(0..self.len) }

    fn ensure_capacity(&mut self, cap: usize) {
        if self.capacity() < cap {
            unsafe {
                let mut new_buf = Buffer::<[_],_>::uninitialized(&self.buf.gl_provider(), cap.max(2*self.capacity()));
                self.buf.copy_data(&mut new_buf);
                swap(&mut self.buf, &mut new_buf);
            }
        }
    }

    pub fn push(&mut self, item:T) {
        unsafe {
            self.ensure_capacity(self.len+1);
            copy_nonoverlapping(&item, &mut self.buf.map_mut()[self.len], 1);
            forget(item);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len() > 0 {
            unsafe {
                let mut item = MaybeUninit::uninit();
                copy_nonoverlapping(&self.buf.map()[self.len-1], item.assume_init_mut(), 1);
                self.len -= 1;
                Some(item.assume_init())
            }
        } else {
            None
        }

    }

    pub fn append_vec(&mut self, v: &mut Vec<T>) {
        unsafe {
            self.ensure_capacity(self.len()+v.len());
            copy_nonoverlapping(&v[0], &mut self.buf.map_mut()[self.len], v.len());
            self.len += v.len();
            v.set_len(0);
        }
    }

}
