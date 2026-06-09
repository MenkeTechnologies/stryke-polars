//! src/nd.rs — numpy surface (ndarray, ufuncs, linalg, random, fft).
//!
//! Wire format for arrays:
//!   `{data: [v1, v2, ...], shape: [d1, d2, ...]}`
//! Row-major flattening. f64 only at this layer (i64/bool stay in the
//! polars DataFrame surface).
//!
//! Wire format for complex (fft output):
//!   `{real: [...], imag: [...], shape: [...]}`
//!
//! Backing crates:
//!   - `ndarray` for nd-array construct/reshape/sum/mean/etc.
//!   - `nalgebra` for inv/det/norm/solve (pure-Rust, no BLAS).
//!   - `rand` + `rand_distr` + `rand_chacha` for seeded distributions.
//!   - `rustfft` for forward + inverse FFT.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Context, Result};
use nalgebra::DMatrix;
use ndarray::{ArrayD, Axis, IxDyn};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, Uniform};
use rustfft::{num_complex::Complex, FftPlanner};
use serde_json::{json, Value};

use crate::ffi_call;

// ── JSON ↔ ndarray ─────────────────────────────────────────────────────────

fn parse_array(v: &Value) -> Result<ArrayD<f64>> {
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("array `data` missing or not an array"))?;
    let shape = v
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or_else(|| anyhow!("array `shape` missing or not an array"))?;
    let flat: Vec<f64> = data
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect();
    let shape_vec: Vec<usize> = shape
        .iter()
        .filter_map(|x| x.as_u64().map(|n| n as usize))
        .collect();
    let expected: usize = shape_vec.iter().product();
    if expected != flat.len() {
        bail!(
            "shape product ({}) ≠ data length ({})",
            expected,
            flat.len()
        );
    }
    ArrayD::from_shape_vec(IxDyn(&shape_vec), flat).context("ArrayD::from_shape_vec")
}

fn array_to_value(arr: &ArrayD<f64>) -> Value {
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<Value> = arr
        .iter()
        .map(|&x| {
            serde_json::Number::from_f64(x)
                .map(Value::Number)
                .unwrap_or(Value::Null)
        })
        .collect();
    json!({"data": data, "shape": shape})
}

fn scalar_to_value(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn get_array(args: &Value, key: &str) -> Result<ArrayD<f64>> {
    let a = args
        .get(key)
        .ok_or_else(|| anyhow!("missing argument `{key}`"))?;
    parse_array(a)
}

// ── ndarray construct ──────────────────────────────────────────────────────

fn shape_arg(args: &Value) -> Result<Vec<usize>> {
    let arr = args
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing argument `shape` (e.g. [3] or [2, 3])"))?;
    let shape: Vec<usize> = arr
        .iter()
        .filter_map(|x| x.as_u64().map(|n| n as usize))
        .collect();
    if shape.is_empty() {
        bail!("`shape` must contain at least one dimension");
    }
    Ok(shape)
}

/// All-zero array of given shape.
///
/// Args:   `{shape: [d1, d2, ...]}`
/// Result: `{array: {data, shape}}`
#[no_mangle]
pub extern "C" fn polars__arr_zeros(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let shape = shape_arg(&args)?;
        let arr = ArrayD::<f64>::zeros(IxDyn(&shape));
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// All-one array.
#[no_mangle]
pub extern "C" fn polars__arr_ones(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let shape = shape_arg(&args)?;
        let arr = ArrayD::<f64>::ones(IxDyn(&shape));
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `numpy.arange(start, stop, step)` — 1-D.
#[no_mangle]
pub extern "C" fn polars__arr_arange(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let start = args.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let stop = args
            .get("stop")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `stop`"))?;
        let step = args.get("step").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if step == 0.0 {
            bail!("`step` must be non-zero");
        }
        let mut data = Vec::new();
        let mut x = start;
        if step > 0.0 {
            while x < stop {
                data.push(x);
                x += step;
            }
        } else {
            while x > stop {
                data.push(x);
                x += step;
            }
        }
        let n = data.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("arange shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `numpy.linspace(start, stop, n)` — n evenly spaced f64 incl. endpoints.
#[no_mangle]
pub extern "C" fn polars__arr_linspace(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let start = args
            .get("start")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `start`"))?;
        let stop = args
            .get("stop")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `stop`"))?;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        if n == 0 {
            bail!("`n` must be ≥ 1");
        }
        let step = if n == 1 {
            0.0
        } else {
            (stop - start) / (n as f64 - 1.0)
        };
        let data: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("linspace shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Reshape (preserves data + length, changes shape).
///
/// Args:   `{array, shape: [...]}`
#[no_mangle]
pub extern "C" fn polars__arr_reshape(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let new_shape = shape_arg(&args)?;
        let total: usize = new_shape.iter().product();
        if total != arr.len() {
            bail!("reshape: shape product {} ≠ array len {}", total, arr.len());
        }
        let flat: Vec<f64> = arr.iter().copied().collect();
        let reshaped = ArrayD::from_shape_vec(IxDyn(&new_shape), flat).context("reshape")?;
        Ok(json!({"array": array_to_value(&reshaped)}))
    })
}

/// Transpose (full reversal of axes — 2-D only).
#[no_mangle]
pub extern "C" fn polars__arr_transpose(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 2 {
            bail!("transpose: only 2-D arrays supported in this slice");
        }
        let t = arr.t().to_owned();
        let t_dyn = t.into_dyn();
        Ok(json!({"array": array_to_value(&t_dyn)}))
    })
}

// ── ufunc helpers ──────────────────────────────────────────────────────────

fn unary_op<F: Fn(f64) -> f64>(args: &Value, f: F) -> Result<Value> {
    let arr = get_array(args, "array")?;
    let result = arr.mapv(f);
    Ok(json!({"array": array_to_value(&result)}))
}

fn binary_op<F: Fn(f64, f64) -> f64>(args: &Value, f: F) -> Result<Value> {
    let a = get_array(args, "a")?;
    let b = get_array(args, "b")?;
    if a.shape() != b.shape() {
        bail!(
            "binary op: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
    }
    let data: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    let result =
        ArrayD::from_shape_vec(IxDyn(a.shape()), data).context("binary op output shape")?;
    Ok(json!({"array": array_to_value(&result)}))
}

// ── ufuncs (unary) ─────────────────────────────────────────────────────────

/// Elementwise sin.
#[no_mangle]
pub extern "C" fn polars__np_sin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::sin))
}

/// Elementwise cos.
#[no_mangle]
pub extern "C" fn polars__np_cos(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::cos))
}

/// Elementwise tan.
#[no_mangle]
pub extern "C" fn polars__np_tan(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::tan))
}

/// Elementwise exp.
#[no_mangle]
pub extern "C" fn polars__np_exp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::exp))
}

/// Elementwise natural log.
#[no_mangle]
pub extern "C" fn polars__np_log(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::ln))
}

/// Elementwise sqrt.
#[no_mangle]
pub extern "C" fn polars__np_sqrt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::sqrt))
}

/// Elementwise abs.
#[no_mangle]
pub extern "C" fn polars__np_abs(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::abs))
}

// ── ufuncs (binary) ────────────────────────────────────────────────────────

/// Elementwise a + b. Shapes must match (no broadcasting in this slice).
#[no_mangle]
pub extern "C" fn polars__np_add(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x + y))
}

/// Elementwise a - b.
#[no_mangle]
pub extern "C" fn polars__np_subtract(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x - y))
}

/// Elementwise a * b.
#[no_mangle]
pub extern "C" fn polars__np_multiply(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x * y))
}

/// Elementwise a / b.
#[no_mangle]
pub extern "C" fn polars__np_divide(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x / y))
}

// ── reductions ─────────────────────────────────────────────────────────────

/// Sum of all elements (axis-collapsed scalar).
#[no_mangle]
pub extern "C" fn polars__arr_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let s = arr.iter().sum::<f64>();
        Ok(json!({"scalar": scalar_to_value(s)}))
    })
}

/// Mean of all elements.
#[no_mangle]
pub extern "C" fn polars__arr_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("mean of empty array");
        }
        let mean = arr.iter().sum::<f64>() / arr.len() as f64;
        Ok(json!({"scalar": scalar_to_value(mean)}))
    })
}

/// Min of all elements.
#[no_mangle]
pub extern "C" fn polars__arr_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let m = arr.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
        Ok(json!({"scalar": scalar_to_value(m)}))
    })
}

/// Max of all elements.
#[no_mangle]
pub extern "C" fn polars__arr_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let m = arr.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        Ok(json!({"scalar": scalar_to_value(m)}))
    })
}

/// Dot product (1-D arrays of matching length).
#[no_mangle]
pub extern "C" fn polars__arr_dot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape() != b.shape() || a.shape().len() != 1 {
            bail!(
                "dot: both arrays must be 1-D same length (got {:?} vs {:?})",
                a.shape(),
                b.shape()
            );
        }
        let d = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f64>();
        Ok(json!({"scalar": scalar_to_value(d)}))
    })
}

/// Concatenate two arrays along an axis (default 0).
#[no_mangle]
pub extern "C" fn polars__arr_concatenate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let axis = args.get("axis").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let result = ndarray::concatenate(Axis(axis), &[a.view(), b.view()])
            .context("ndarray::concatenate")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

// ── linalg (nalgebra) ──────────────────────────────────────────────────────

fn parse_matrix(v: &Value) -> Result<DMatrix<f64>> {
    let arr = parse_array(v)?;
    if arr.shape().len() != 2 {
        bail!("matrix must be 2-D, got shape {:?}", arr.shape());
    }
    let rows = arr.shape()[0];
    let cols = arr.shape()[1];
    let data: Vec<f64> = arr.iter().copied().collect();
    Ok(DMatrix::from_row_slice(rows, cols, &data))
}

fn matrix_to_value(m: &DMatrix<f64>) -> Value {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut flat = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            flat.push(m[(r, c)]);
        }
    }
    let data: Vec<Value> = flat.iter().map(|&x| scalar_to_value(x)).collect();
    json!({"data": data, "shape": [rows, cols]})
}

/// Matrix inverse. Errors if singular.
#[no_mangle]
pub extern "C" fn polars__linalg_inv(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        if !m.is_square() {
            bail!("inv: matrix must be square ({}x{})", m.nrows(), m.ncols());
        }
        let inv = m
            .try_inverse()
            .ok_or_else(|| anyhow!("matrix not invertible (singular)"))?;
        Ok(json!({"matrix": matrix_to_value(&inv)}))
    })
}

/// Matrix determinant.
#[no_mangle]
pub extern "C" fn polars__linalg_det(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        if !m.is_square() {
            bail!("det: matrix must be square");
        }
        Ok(json!({"scalar": scalar_to_value(m.determinant())}))
    })
}

/// Solve `A x = b` for x. `b` is a column vector (1-D array).
///
/// Args:   `{matrix: <A>, b: <vec>}`
/// Result: `{array: <x>}`
#[no_mangle]
pub extern "C" fn polars__linalg_solve(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        let b_arr = parse_array(
            args.get("b")
                .ok_or_else(|| anyhow!("missing argument `b`"))?,
        )?;
        if b_arr.shape().len() != 1 {
            bail!("b must be a 1-D vector");
        }
        if b_arr.len() != a.nrows() {
            bail!(
                "shape mismatch: A is {}×{}, b has length {}",
                a.nrows(),
                a.ncols(),
                b_arr.len()
            );
        }
        let b = nalgebra::DVector::from_vec(b_arr.iter().copied().collect());
        let lu = a.lu();
        let x = lu
            .solve(&b)
            .ok_or_else(|| anyhow!("linear system has no solution (singular matrix)"))?;
        let n = x.len();
        let result = ArrayD::from_shape_vec(IxDyn(&[n]), x.iter().copied().collect())
            .context("solve out")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// Frobenius / 2-norm of a vector or matrix.
#[no_mangle]
pub extern "C" fn polars__linalg_norm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let s = arr.iter().map(|x| x * x).sum::<f64>().sqrt();
        Ok(json!({"scalar": scalar_to_value(s)}))
    })
}

// ── random ─────────────────────────────────────────────────────────────────

fn rng_for(args: &Value) -> ChaCha8Rng {
    let seed = args.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);
    ChaCha8Rng::seed_from_u64(seed)
}

/// `np.random.normal(mean, std, n)` — n samples from Normal(mean, std).
///
/// Args:   `{n: u64, mean?: f64 (default 0), std?: f64 (default 1), seed?: u64}`
#[no_mangle]
pub extern "C" fn polars__rand_normal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let mean = args.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let std = args.get("std").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if std <= 0.0 {
            bail!("`std` must be > 0");
        }
        let dist = Normal::new(mean, std).context("Normal::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("rand_normal shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.uniform(low, high, n)`.
#[no_mangle]
pub extern "C" fn polars__rand_uniform(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let low = args.get("low").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let high = args.get("high").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if high <= low {
            bail!("`high` must be > `low`");
        }
        let dist = Uniform::new(low, high);
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("rand_uniform shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

// ── fft ────────────────────────────────────────────────────────────────────

/// Forward FFT of a 1-D real array. Result: complex array `{real, imag, shape}`.
#[no_mangle]
pub extern "C" fn polars__fft_fft(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("fft: only 1-D arrays supported in this slice");
        }
        let n = arr.len();
        let mut buf: Vec<Complex<f64>> = arr.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buf);
        let real: Vec<Value> = buf.iter().map(|c| scalar_to_value(c.re)).collect();
        let imag: Vec<Value> = buf.iter().map(|c| scalar_to_value(c.im)).collect();
        Ok(json!({"complex": {"real": real, "imag": imag, "shape": [n]}}))
    })
}

// ── P4b: more ndarray ──────────────────────────────────────────────────────

/// Identity matrix `n×n`.
///
/// Args:   `{n: u64}`
#[no_mangle]
pub extern "C" fn polars__arr_eye(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        if n == 0 {
            bail!("`n` must be ≥ 1");
        }
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, n]), data).context("eye shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Constant-fill array.
///
/// Args:   `{shape, value: f64}`
#[no_mangle]
pub extern "C" fn polars__arr_full(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let shape = shape_arg(&args)?;
        let value = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `value` (f64)"))?;
        let total: usize = shape.iter().product();
        let data = vec![value; total];
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data).context("full shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Index of minimum element across the flat iter.
#[no_mangle]
pub extern "C" fn polars__arr_argmin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("argmin of empty array");
        }
        let (idx, _) = arr
            .iter()
            .enumerate()
            .fold((0usize, f64::INFINITY), |(bi, bv), (i, &v)| {
                if v < bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        Ok(json!({"index": idx}))
    })
}

/// Index of maximum element across the flat iter.
#[no_mangle]
pub extern "C" fn polars__arr_argmax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("argmax of empty array");
        }
        let (idx, _) =
            arr.iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                });
        Ok(json!({"index": idx}))
    })
}

// ── P5b: more linalg ───────────────────────────────────────────────────────

/// Matrix trace (sum of diagonal). Square matrix required.
#[no_mangle]
pub extern "C" fn polars__linalg_trace(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        if !m.is_square() {
            bail!("trace: matrix must be square");
        }
        Ok(json!({"scalar": scalar_to_value(m.trace())}))
    })
}

/// Matrix rank (numerical rank via SVD singular-value count > 1e-10).
#[no_mangle]
pub extern "C" fn polars__linalg_rank(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        let r = m.rank(1e-10);
        Ok(json!({"rank": r}))
    })
}

/// Inverse FFT. Input is complex (real + imag arrays); output is real-part
/// 1-D array (drops imag).
///
/// Args:   `{complex: {real: [...], imag: [...], shape: [n]}}`
#[no_mangle]
pub extern "C" fn polars__fft_ifft(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let c = args
            .get("complex")
            .ok_or_else(|| anyhow!("missing argument `complex`"))?;
        let real = c
            .get("real")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("complex.real missing"))?;
        let imag = c
            .get("imag")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("complex.imag missing"))?;
        if real.len() != imag.len() {
            bail!("real / imag length mismatch");
        }
        let n = real.len();
        let mut buf: Vec<Complex<f64>> = real
            .iter()
            .zip(imag.iter())
            .map(|(r, i)| Complex::new(r.as_f64().unwrap_or(0.0), i.as_f64().unwrap_or(0.0)))
            .collect();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(n);
        fft.process(&mut buf);
        // rustfft does NOT normalize; divide by n.
        let data: Vec<f64> = buf.iter().map(|c| c.re / n as f64).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("ifft shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}
