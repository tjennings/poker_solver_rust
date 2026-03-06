use std::mem::MaybeUninit;

/// `max(x, y)` without NaN propagation (branchless for finite inputs).
#[inline]
pub(crate) fn max(x: f32, y: f32) -> f32 {
    if x > y {
        x
    } else {
        y
    }
}

/// Returns `true` if `x` is positive zero (bit-exact).
#[inline]
pub(crate) fn is_zero(x: f32) -> bool {
    x.to_bits() == 0
}

#[inline]
pub(crate) fn sub_slice(lhs: &mut [f32], rhs: &[f32]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l -= *r);
}

#[inline]
pub(crate) fn mul_slice(lhs: &mut [f32], rhs: &[f32]) {
    lhs.iter_mut().zip(rhs).for_each(|(l, r)| *l *= *r);
}

#[inline]
pub(crate) fn div_slice(lhs: &mut [f32], rhs: &[f32], default: f32) {
    lhs.iter_mut()
        .zip(rhs)
        .for_each(|(l, r)| *l = if is_zero(*r) { default } else { *l / *r });
}

#[inline]
pub(crate) fn div_slice_uninit(
    dst: &mut [MaybeUninit<f32>],
    lhs: &[f32],
    rhs: &[f32],
    default: f32,
) {
    dst.iter_mut()
        .zip(lhs.iter().zip(rhs))
        .for_each(|(d, (l, r))| {
            d.write(if is_zero(*r) { default } else { *l / *r });
        });
}

#[inline]
pub(crate) fn mul_slice_scalar_uninit(dst: &mut [MaybeUninit<f32>], src: &[f32], scalar: f32) {
    dst.iter_mut().zip(src).for_each(|(d, s)| {
        d.write(*s * scalar);
    });
}

#[inline]
pub(crate) fn sum_slices_uninit<'a>(
    dst: &'a mut [MaybeUninit<f32>],
    src: &[f32],
) -> &'a mut [f32] {
    let len = dst.len();
    dst.iter_mut().zip(src).for_each(|(d, s)| {
        d.write(*s);
    });
    // SAFETY: All `len` elements have been initialized via `write` above.
    let dst = unsafe { &mut *(dst as *mut _ as *mut [f32]) };
    src[len..].chunks_exact(len).for_each(|s| {
        dst.iter_mut().zip(s).for_each(|(d, s)| {
            *d += *s;
        });
    });
    dst
}

#[inline]
pub(crate) fn sum_slices_f64_uninit<'a>(
    dst: &'a mut [MaybeUninit<f64>],
    src: &[f32],
) -> &'a mut [f64] {
    let len = dst.len();
    dst.iter_mut().zip(src).for_each(|(d, s)| {
        d.write(*s as f64);
    });
    // SAFETY: All `len` elements have been initialized via `write` above.
    let dst = unsafe { &mut *(dst as *mut _ as *mut [f64]) };
    src[len..].chunks_exact(len).for_each(|s| {
        dst.iter_mut().zip(s).for_each(|(d, s)| {
            *d += *s as f64;
        });
    });
    dst
}

#[inline]
pub(crate) fn fma_slices_uninit<'a>(
    dst: &'a mut [MaybeUninit<f32>],
    src1: &[f32],
    src2: &[f32],
) -> &'a mut [f32] {
    let len = dst.len();
    dst.iter_mut()
        .zip(src1.iter().zip(src2))
        .for_each(|(d, (s1, s2))| {
            d.write(*s1 * *s2);
        });
    // SAFETY: All `len` elements have been initialized via `write` above.
    let dst = unsafe { &mut *(dst as *mut _ as *mut [f32]) };
    src1[len..]
        .chunks_exact(len)
        .zip(src2[len..].chunks_exact(len))
        .for_each(|(s1, s2)| {
            dst.iter_mut()
                .zip(s1.iter().zip(s2))
                .for_each(|(d, (s1, s2))| {
                    *d += *s1 * *s2;
                });
        });
    dst
}

#[inline]
pub(crate) fn max_slices_uninit<'a>(
    dst: &'a mut [MaybeUninit<f32>],
    src: &[f32],
) -> &'a mut [f32] {
    let len = dst.len();
    dst.iter_mut().zip(src).for_each(|(d, s)| {
        d.write(*s);
    });
    // SAFETY: All `len` elements have been initialized via `write` above.
    let dst = unsafe { &mut *(dst as *mut _ as *mut [f32]) };
    src[len..].chunks_exact(len).for_each(|s| {
        dst.iter_mut().zip(s).for_each(|(d, s)| {
            *d = max(*d, *s);
        });
    });
    dst
}

#[inline]
pub(crate) fn max_fma_slices_uninit<'a>(
    dst: &'a mut [MaybeUninit<f32>],
    src1: &[f32],
    src2: &[f32],
) -> &'a mut [f32] {
    let len = dst.len();
    dst.iter_mut()
        .zip(src1.iter().zip(src2))
        .for_each(|(d, (s1, s2))| {
            d.write(if s2.is_sign_positive() {
                *s1 * *s2
            } else {
                *s1
            });
        });
    // SAFETY: All `len` elements have been initialized via `write` above.
    let dst = unsafe { &mut *(dst as *mut _ as *mut [f32]) };
    src1[len..]
        .chunks_exact(len)
        .zip(src2[len..].chunks_exact(len))
        .for_each(|(s1, s2)| {
            dst.iter_mut()
                .zip(s1.iter().zip(s2))
                .for_each(|(d, (s1, s2))| {
                    if s2.is_sign_positive() {
                        *d += *s1 * *s2;
                    } else {
                        *d = max(*d, *s1);
                    }
                });
        });
    dst
}

#[inline]
pub(crate) fn inner_product(src1: &[f32], src2: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 8;

    let len = src1.len();
    let len_chunk = len / CHUNK_SIZE * CHUNK_SIZE;
    let mut acc = [0.0f64; CHUNK_SIZE];

    for i in (0..len_chunk).step_by(CHUNK_SIZE) {
        for j in 0..CHUNK_SIZE {
            // SAFETY: `i + j < len_chunk <= len`, so the index is in bounds.
            unsafe {
                let x = *src1.get_unchecked(i + j);
                let y = *src2.get_unchecked(i + j);
                *acc.get_unchecked_mut(j) += (x * y) as f64;
            }
        }
    }

    for i in len_chunk..len {
        // SAFETY: `i < len`, so the index is in bounds.
        unsafe {
            let x = *src1.get_unchecked(i);
            let y = *src2.get_unchecked(i);
            *acc.get_unchecked_mut(0) += (x * y) as f64;
        }
    }

    acc.iter().sum::<f64>() as f32
}

#[inline]
pub(crate) fn row<T>(slice: &[T], index: usize, row_size: usize) -> &[T] {
    &slice[index * row_size..(index + 1) * row_size]
}

#[inline]
pub(crate) fn row_mut<T>(slice: &mut [T], index: usize, row_size: usize) -> &mut [T] {
    &mut slice[index * row_size..(index + 1) * row_size]
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max() {
        assert_eq!(max(1.0, 2.0), 2.0);
        assert_eq!(max(-1.0, -2.0), -1.0);
        assert_eq!(max(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_is_zero() {
        assert!(is_zero(0.0));
        assert!(!is_zero(-0.0));
        assert!(!is_zero(1.0));
        assert!(!is_zero(f32::NAN));
    }

    #[test]
    fn test_sub_slice() {
        let mut a = vec![3.0, 5.0, 7.0];
        let b = vec![1.0, 2.0, 3.0];
        sub_slice(&mut a, &b);
        assert_eq!(a, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mul_slice() {
        let mut a = vec![2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 0.5];
        mul_slice(&mut a, &b);
        assert_eq!(a, vec![2.0, 6.0, 2.0]);
    }

    #[test]
    fn test_div_slice() {
        let mut a = vec![6.0, 3.0, 0.0];
        let b = vec![2.0, 0.0, 5.0];
        div_slice(&mut a, &b, 99.0);
        assert_eq!(a, vec![3.0, 99.0, 0.0]);
    }

    #[test]
    fn test_sum_slices_uninit() {
        // 3 actions, 2 hands => src = [a0h0, a0h1, a1h0, a1h1, a2h0, a2h1]
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![MaybeUninit::uninit(); 2];
        let result = sum_slices_uninit(&mut dst, &src);
        assert_eq!(result, &[9.0, 12.0]);
    }

    #[test]
    fn test_fma_slices_uninit() {
        // strategy = [0.5, 0.5, 0.5, 0.5], cfv = [10, 20, 30, 40]
        // result[h] = sum_a (strategy[a,h] * cfv[a,h])
        let strategy = vec![0.5, 0.5, 0.5, 0.5];
        let cfv = vec![10.0, 20.0, 30.0, 40.0];
        let mut dst = vec![MaybeUninit::uninit(); 2];
        let result = fma_slices_uninit(&mut dst, &strategy, &cfv);
        assert_eq!(result, &[20.0, 30.0]);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = inner_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_row() {
        let data = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(row(&data, 0, 2), &[0, 1]);
        assert_eq!(row(&data, 1, 2), &[2, 3]);
        assert_eq!(row(&data, 2, 2), &[4, 5]);
    }

    #[test]
    fn test_row_mut() {
        let mut data = vec![0, 1, 2, 3, 4, 5];
        row_mut(&mut data, 1, 2)[0] = 99;
        assert_eq!(data, vec![0, 1, 99, 3, 4, 5]);
    }
}
