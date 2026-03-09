use std::cell::UnsafeCell;
use std::fmt::{self, Debug, Formatter};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

// ---------------------------------------------------------------------------
// MutexLike
// ---------------------------------------------------------------------------

/// Mutex-like wrapper that does not actually perform any locking.
///
/// Use this wrapper when:
///   1. [`Send`], [`Sync`] and interior mutability are needed,
///   2. it is (manually) guaranteed that data races will not occur, and
///   3. performance is critical.
///
/// **Note**: This wrapper completely bypasses the "shared XOR mutable" rule.
/// Using it is **extremely unsafe** and should be avoided whenever possible.
#[derive(Debug)]
#[repr(transparent)]
pub struct MutexLike<T: ?Sized> {
    data: UnsafeCell<T>,
}

/// Smart-pointer wrapper returned when [`MutexLike`] is "locked".
#[derive(Debug)]
pub struct MutexGuardLike<'a, T: ?Sized + 'a> {
    mutex: &'a MutexLike<T>,
}

// SAFETY: MutexLike is only safe when the caller guarantees no concurrent
// mutable access. The Send/Sync impls mirror std::sync::Mutex's requirements.
unsafe impl<T: ?Sized + Send> Send for MutexLike<T> {}
unsafe impl<T: ?Sized + Send> Sync for MutexLike<T> {}
unsafe impl<'a, T: ?Sized + Sync + 'a> Sync for MutexGuardLike<'a, T> {}

impl<T> MutexLike<T> {
    /// Creates a new [`MutexLike`] with the given value.
    #[inline]
    pub fn new(val: T) -> Self {
        Self {
            data: UnsafeCell::new(val),
        }
    }

    /// Consumes the `MutexLike` and returns the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }
}

impl<T: ?Sized> MutexLike<T> {
    /// Acquires a mutex-like object **without** performing any locking.
    #[inline]
    pub fn lock(&self) -> MutexGuardLike<'_, T> {
        MutexGuardLike { mutex: self }
    }
}

impl<T: Default> Default for MutexLike<T> {
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<'a, T: ?Sized + 'a> Deref for MutexGuardLike<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: Caller guarantees no concurrent mutable access.
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for MutexGuardLike<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: Caller guarantees no concurrent mutable access.
        unsafe { &mut *self.mutex.data.get() }
    }
}

// ---------------------------------------------------------------------------
// AtomicF32
// ---------------------------------------------------------------------------

/// An atomically-accessible `f32`, backed by [`AtomicU32`].
///
/// All operations use [`Relaxed`] ordering.
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    #[inline]
    pub fn new(v: f32) -> Self {
        Self(AtomicU32::new(v.to_bits()))
    }

    #[inline]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Relaxed))
    }

    #[inline]
    pub fn store(&self, v: f32) {
        self.0.store(v.to_bits(), Relaxed);
    }
}

impl Debug for AtomicF32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.load().fmt(f)
    }
}

// ---------------------------------------------------------------------------
// AtomicF64
// ---------------------------------------------------------------------------

/// An atomically-accessible `f64`, backed by [`AtomicU64`].
///
/// All operations use [`Relaxed`] ordering.
pub struct AtomicF64(AtomicU64);

impl AtomicF64 {
    #[inline]
    pub fn new(v: f64) -> Self {
        Self(AtomicU64::new(v.to_bits()))
    }

    #[inline]
    pub fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Relaxed))
    }

    #[inline]
    pub fn store(&self, v: f64) {
        self.0.store(v.to_bits(), Relaxed);
    }

    /// Atomically adds `v` to the stored value via a CAS loop.
    #[inline]
    pub fn add(&self, v: f64) {
        let _ = self.0.fetch_update(Relaxed, Relaxed, |u| {
            Some((f64::from_bits(u) + v).to_bits())
        });
    }
}

impl Debug for AtomicF64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.load().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- MutexLike tests --

    #[test]
    fn mutex_like_basic() {
        let m = MutexLike::new(42);
        assert_eq!(*m.lock(), 42);
        *m.lock() = 99;
        assert_eq!(*m.lock(), 99);
    }

    #[test]
    fn mutex_like_default() {
        let m: MutexLike<i32> = MutexLike::default();
        assert_eq!(*m.lock(), 0);
    }

    #[test]
    fn mutex_like_vec() {
        let m = MutexLike::new(vec![1, 2, 3]);
        m.lock().push(4);
        assert_eq!(&*m.lock(), &[1, 2, 3, 4]);
    }

    // -- AtomicF32 tests --

    #[test]
    fn atomic_f32_store_load() {
        let a = AtomicF32::new(1.5);
        assert_eq!(a.load(), 1.5);
        a.store(3.25);
        assert_eq!(a.load(), 3.25);
    }

    #[test]
    fn atomic_f32_zero() {
        let a = AtomicF32::new(0.0);
        assert_eq!(a.load(), 0.0);
    }

    // -- AtomicF64 tests --

    #[test]
    fn atomic_f64_store_load() {
        let a = AtomicF64::new(1.5);
        assert_eq!(a.load(), 1.5);
        a.store(3.25);
        assert_eq!(a.load(), 3.25);
    }

    #[test]
    fn atomic_f64_add() {
        let a = AtomicF64::new(10.0);
        a.add(2.5);
        assert_eq!(a.load(), 12.5);
        a.add(-3.0);
        assert_eq!(a.load(), 9.5);
    }

    #[test]
    fn atomic_f64_debug() {
        let a = AtomicF64::new(42.0);
        let s = format!("{a:?}");
        assert_eq!(s, "42.0");
    }
}
