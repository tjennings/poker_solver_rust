//! Lazily-committed memory buffer backed by anonymous mmap.
//!
//! Physical pages are only assigned RAM on first write (page fault).
//! Zero-initialized by the kernel -- matches `AtomicI16::new(0)` / `AtomicI32::new(0)`.

use std::ops::{Deref, DerefMut};

/// A lazily-committed buffer backed by anonymous mmap.
pub struct MmapBuffer<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: The buffer owns a contiguous allocation of T.
// If T is Send/Sync, the buffer is too.
unsafe impl<T: Send> Send for MmapBuffer<T> {}
unsafe impl<T: Sync> Sync for MmapBuffer<T> {}

impl<T> MmapBuffer<T> {
    /// Allocate a lazily-committed buffer of `len` elements.
    ///
    /// Virtual address space is reserved immediately, but physical pages
    /// are only committed on first access (write). All bytes are zero-initialized.
    ///
    /// # Panics
    /// Panics if mmap fails or if `len * size_of::<T>()` overflows.
    #[must_use]
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            };
        }
        let byte_len = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("MmapBuffer: size overflow");
        let ptr = alloc_mmap(byte_len).cast::<T>();
        Self { ptr, len }
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Deref for MmapBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for MmapBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for MmapBuffer<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            let byte_len = self.len * std::mem::size_of::<T>();
            free_mmap(self.ptr.cast::<u8>(), byte_len);
        }
    }
}

/// Platform-specific mmap allocation (unix).
#[cfg(unix)]
fn alloc_mmap(byte_len: usize) -> *mut u8 {
    #[cfg(target_os = "linux")]
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
    #[cfg(not(target_os = "linux"))]
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            byte_len,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };
    assert!(
        ptr != libc::MAP_FAILED,
        "mmap failed for {byte_len} bytes ({})",
        std::io::Error::last_os_error()
    );
    ptr.cast::<u8>()
}

#[cfg(unix)]
fn free_mmap(ptr: *mut u8, byte_len: usize) {
    unsafe {
        libc::munmap(ptr.cast::<libc::c_void>(), byte_len);
    }
}

// Fallback for non-unix: use Vec (no lazy allocation, but correct).
#[cfg(not(unix))]
fn alloc_mmap(byte_len: usize) -> *mut u8 {
    let mut v = vec![0u8; byte_len];
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(not(unix))]
fn free_mmap(ptr: *mut u8, byte_len: usize) {
    unsafe {
        drop(Vec::from_raw_parts(ptr, byte_len, byte_len));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI16, AtomicI32, Ordering};
    use test_macros::timed_test;

    #[timed_test]
    fn empty_buffer() {
        let buf: MmapBuffer<AtomicI16> = MmapBuffer::new(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[timed_test]
    fn zero_initialized() {
        let buf: MmapBuffer<AtomicI16> = MmapBuffer::new(1000);
        assert_eq!(buf.len(), 1000);
        for atom in buf.iter() {
            assert_eq!(atom.load(Ordering::Relaxed), 0);
        }
    }

    #[timed_test]
    fn read_write_round_trip() {
        let buf: MmapBuffer<AtomicI32> = MmapBuffer::new(100);
        buf[42].store(12345, Ordering::Relaxed);
        assert_eq!(buf[42].load(Ordering::Relaxed), 12345);
    }

    #[timed_test]
    fn large_allocation_succeeds() {
        // 10M elements x 2 bytes = 20 MB virtual
        let buf: MmapBuffer<AtomicI16> = MmapBuffer::new(10_000_000);
        assert_eq!(buf.len(), 10_000_000);
        // Only touch a few pages -- verify lazy allocation
        buf[0].store(1, Ordering::Relaxed);
        buf[5_000_000].store(2, Ordering::Relaxed);
        buf[9_999_999].store(3, Ordering::Relaxed);
        assert_eq!(buf[0].load(Ordering::Relaxed), 1);
        assert_eq!(buf[5_000_000].load(Ordering::Relaxed), 2);
        assert_eq!(buf[9_999_999].load(Ordering::Relaxed), 3);
    }

    #[timed_test]
    fn deref_allows_slice_ops() {
        let buf: MmapBuffer<AtomicI16> = MmapBuffer::new(5);
        buf[0].store(10, Ordering::Relaxed);
        buf[1].store(20, Ordering::Relaxed);
        let sum: i16 = buf.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        assert_eq!(sum, 30);
    }

    #[timed_test]
    fn par_chunks_works() {
        use rayon::prelude::*;
        let buf: MmapBuffer<AtomicI16> = MmapBuffer::new(1000);
        buf[500].store(42, Ordering::Relaxed);
        let max = buf
            .par_chunks(100)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|a| a.load(Ordering::Relaxed))
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert_eq!(max, 42);
    }

    #[timed_test]
    fn drop_releases_memory() {
        let buf: MmapBuffer<AtomicI32> = MmapBuffer::new(10_000);
        drop(buf);
    }
}
