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

/// Elementwise tanh.
#[no_mangle]
pub extern "C" fn polars__np_tanh(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::tanh))
}

/// Elementwise sinh.
#[no_mangle]
pub extern "C" fn polars__np_sinh(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::sinh))
}

/// Elementwise cosh.
#[no_mangle]
pub extern "C" fn polars__np_cosh(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::cosh))
}

/// Elementwise arctan.
#[no_mangle]
pub extern "C" fn polars__np_arctan(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::atan))
}

/// Elementwise arcsin.
#[no_mangle]
pub extern "C" fn polars__np_arcsin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::asin))
}

/// Elementwise arccos.
#[no_mangle]
pub extern "C" fn polars__np_arccos(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::acos))
}

/// Elementwise log2.
#[no_mangle]
pub extern "C" fn polars__np_log2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::log2))
}

/// Elementwise log10.
#[no_mangle]
pub extern "C" fn polars__np_log10(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::log10))
}

/// Elementwise exp2 (2^x).
#[no_mangle]
pub extern "C" fn polars__np_exp2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::exp2))
}

/// Elementwise floor (re-exposed at np_ prefix matching numpy convention).
#[no_mangle]
pub extern "C" fn polars__np_floor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::floor))
}

/// Elementwise ceil.
#[no_mangle]
pub extern "C" fn polars__np_ceil(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::ceil))
}

/// Elementwise sign (-1 / 0 / +1).
#[no_mangle]
pub extern "C" fn polars__np_sign(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        unary_op(&args, |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    })
}

/// Elementwise power: `a^b` per element. Shapes must match.
#[no_mangle]
pub extern "C" fn polars__np_power(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, f64::powf))
}

/// Elementwise mod (`a % b`). Shapes must match.
#[no_mangle]
pub extern "C" fn polars__np_mod(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x.rem_euclid(y)))
}

/// Elementwise max (per-pair). Shapes must match.
#[no_mangle]
pub extern "C" fn polars__np_maximum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, f64::max))
}

/// Elementwise min.
#[no_mangle]
pub extern "C" fn polars__np_minimum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, f64::min))
}

/// Elementwise `a % b` matching numpy (sign follows divisor).
#[no_mangle]
pub extern "C" fn polars__np_remainder(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, f64::rem_euclid))
}

/// Elementwise NaN-aware max (NaN treated as missing).
#[no_mangle]
pub extern "C" fn polars__np_fmax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |a, b| {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else {
                a.max(b)
            }
        })
    })
}

/// Elementwise NaN-aware min.
#[no_mangle]
pub extern "C" fn polars__np_fmin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |a, b| {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else {
                a.min(b)
            }
        })
    })
}

/// Heaviside step: x<0 ⇒ 0, x>0 ⇒ 1, x=0 ⇒ `h0`.
#[no_mangle]
pub extern "C" fn polars__np_heaviside(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, h0| {
            if x < 0.0 {
                0.0
            } else if x > 0.0 {
                1.0
            } else {
                h0
            }
        })
    })
}

// ── P4e: predicate ufuncs ──────────────────────────────────────────────────

/// Elementwise isnan (1.0/0.0 result).
#[no_mangle]
pub extern "C" fn polars__np_isnan(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        unary_op(&args, |x| if x.is_nan() { 1.0 } else { 0.0 })
    })
}

/// Elementwise isinf.
#[no_mangle]
pub extern "C" fn polars__np_isinf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        unary_op(&args, |x| if x.is_infinite() { 1.0 } else { 0.0 })
    })
}

/// Elementwise isfinite.
#[no_mangle]
pub extern "C" fn polars__np_isfinite(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        unary_op(&args, |x| if x.is_finite() { 1.0 } else { 0.0 })
    })
}

/// Round to nearest integer (banker's rounding via f64::round_ties_even).
#[no_mangle]
pub extern "C" fn polars__np_rint(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::round_ties_even))
}

/// Logical AND (treats non-zero as true). Output 1.0/0.0.
#[no_mangle]
pub extern "C" fn polars__np_logical_and(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |a, b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 })
    })
}

/// Logical OR.
#[no_mangle]
pub extern "C" fn polars__np_logical_or(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |a, b| if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 })
    })
}

/// Logical NOT.
#[no_mangle]
pub extern "C" fn polars__np_logical_not(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        unary_op(&args, |x| if x == 0.0 { 1.0 } else { 0.0 })
    })
}

/// Logical XOR.
#[no_mangle]
pub extern "C" fn polars__np_logical_xor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(
            &args,
            |a, b| {
                if (a != 0.0) ^ (b != 0.0) {
                    1.0
                } else {
                    0.0
                }
            },
        )
    })
}

// ── P4f: more unary ufuncs ─────────────────────────────────────────────────

/// numpy ufunc trunc.
#[no_mangle]
pub extern "C" fn polars__np_trunc(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::trunc))
}

/// numpy ufunc radians.
#[no_mangle]
pub extern "C" fn polars__np_radians(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::to_radians))
}

/// numpy ufunc degrees.
#[no_mangle]
pub extern "C" fn polars__np_degrees(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::to_degrees))
}

/// numpy ufunc negative.
#[no_mangle]
pub extern "C" fn polars__np_negative(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, |x| -x))
}

/// numpy ufunc reciprocal.
#[no_mangle]
pub extern "C" fn polars__np_reciprocal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, |x| 1.0 / x))
}

/// numpy ufunc square.
#[no_mangle]
pub extern "C" fn polars__np_square(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, |x| x * x))
}

/// numpy ufunc cbrt.
#[no_mangle]
pub extern "C" fn polars__np_cbrt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::cbrt))
}

/// numpy ufunc expm1.
#[no_mangle]
pub extern "C" fn polars__np_expm1(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::exp_m1))
}

/// numpy ufunc log1p.
#[no_mangle]
pub extern "C" fn polars__np_log1p(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::ln_1p))
}

/// numpy ufunc arctan2.
#[no_mangle]
pub extern "C" fn polars__np_arctan2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, f64::atan2))
}

/// numpy ufunc hypot.
#[no_mangle]
pub extern "C" fn polars__np_hypot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, f64::hypot))
}

/// numpy ufunc floor divide.
#[no_mangle]
pub extern "C" fn polars__np_floor_divide(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| (x / y).floor()))
}

/// numpy ufunc copysign.
#[no_mangle]
pub extern "C" fn polars__np_copysign(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, f64::copysign))
}

// ── retroactive: 4 fns whose exports landed in cfa02cc95a but bodies didn't ──

/// Reverse a flat array (preserves shape).
#[no_mangle]
pub extern "C" fn polars__arr_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let mut data: Vec<f64> = arr.iter().copied().collect();
        data.reverse();
        let out = ArrayD::from_shape_vec(IxDyn(arr.shape()), data).context("reverse shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Repeat each element `n` times in flat order (1-D).
#[no_mangle]
pub extern "C" fn polars__arr_repeat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("repeat: only 1-D arrays supported");
        }
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let mut data = Vec::with_capacity(arr.len() * n);
        for &v in arr.iter() {
            for _ in 0..n {
                data.push(v);
            }
        }
        let total = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[total]), data).context("repeat shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.random.geometric(p, n)` — int samples as f64.
#[no_mangle]
pub extern "C" fn polars__rand_geometric(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let p = args
            .get("p")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `p`"))?;
        if !(0.0..=1.0).contains(&p) || p == 0.0 {
            bail!("`p` must be in (0, 1]");
        }
        let dist = rand_distr::Geometric::new(p).context("Geometric::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("geometric shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Kronecker product of two matrices.
#[no_mangle]
pub extern "C" fn polars__linalg_kron(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_matrix(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_matrix(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        let k = a.kronecker(&b);
        Ok(json!({"matrix": matrix_to_value(&k)}))
    })
}

// ── P4g: cumulative axis ops ───────────────────────────────────────────────

/// ndarray cumprod.
#[no_mangle]
pub extern "C" fn polars__arr_cumprod(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let mut acc = 1.0;
        let data: Vec<f64> = arr
            .iter()
            .map(|&x| {
                acc *= x;
                acc
            })
            .collect();
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("cumprod shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// ndarray cummin.
#[no_mangle]
pub extern "C" fn polars__arr_cummin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let mut acc = f64::INFINITY;
        let data: Vec<f64> = arr
            .iter()
            .map(|&x| {
                acc = acc.min(x);
                acc
            })
            .collect();
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("cummin shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// ndarray cummax.
#[no_mangle]
pub extern "C" fn polars__arr_cummax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let mut acc = f64::NEG_INFINITY;
        let data: Vec<f64> = arr
            .iter()
            .map(|&x| {
                acc = acc.max(x);
                acc
            })
            .collect();
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("cummax shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

// ── P5h: fft + random + poly + linalg ──────────────────────────────────────

/// FFT rfft.
#[no_mangle]
pub extern "C" fn polars__fft_rfft(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("rfft: only 1-D arrays supported");
        }
        let n = arr.len();
        let mut buf: Vec<Complex<f64>> = arr.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buf);
        let m = n / 2 + 1;
        let real: Vec<Value> = buf[..m].iter().map(|c| scalar_to_value(c.re)).collect();
        let imag: Vec<Value> = buf[..m].iter().map(|c| scalar_to_value(c.im)).collect();
        Ok(json!({"complex": {"real": real, "imag": imag, "shape": [m]}}))
    })
}

/// FFT fftshift.
#[no_mangle]
pub extern "C" fn polars__fft_fftshift(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("fftshift: only 1-D arrays supported");
        }
        let data: Vec<f64> = arr.iter().copied().collect();
        let n = data.len();
        let mid = n / 2;
        let mut shifted = Vec::with_capacity(n);
        shifted.extend_from_slice(&data[mid..]);
        shifted.extend_from_slice(&data[..mid]);
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), shifted).context("fftshift shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.random.dirichlet(alpha, n)` — Dirichlet samples. Output: 2-D `[n, k]`.
#[no_mangle]
pub extern "C" fn polars__rand_dirichlet(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let alpha_arr = args
            .get("alpha")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `alpha` (array)"))?;
        let alpha: Vec<f64> = alpha_arr
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        let k = alpha.len();
        if k == 0 {
            bail!("`alpha` must be non-empty");
        }
        if alpha.iter().any(|&a| a <= 0.0) {
            bail!("`alpha` entries must be > 0");
        }
        let mut rng = rng_for(&args);
        let mut data = Vec::with_capacity(n * k);
        for _ in 0..n {
            let mut row = Vec::with_capacity(k);
            for &a in &alpha {
                let dist = rand_distr::Gamma::new(a, 1.0).context("Gamma::new")?;
                row.push(dist.sample(&mut rng));
            }
            let sum: f64 = row.iter().sum();
            for v in row.iter_mut() {
                *v /= sum;
            }
            data.extend(row);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, k]), data).context("dirichlet shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Polynomial fit via least squares (Vandermonde + normal equations).
#[no_mangle]
pub extern "C" fn polars__poly_polyfit(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_array(&args, "x")?;
        let y = get_array(&args, "y")?;
        if x.shape().len() != 1 || y.shape().len() != 1 {
            bail!("polyfit: x and y must be 1-D");
        }
        if x.len() != y.len() {
            bail!("polyfit: x and y must have same length");
        }
        let deg = args
            .get("deg")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `deg`"))? as usize;
        let m = x.len();
        let k = deg + 1;
        if m < k {
            bail!("polyfit: need ≥ deg+1 points");
        }
        let mut v_data = Vec::with_capacity(m * k);
        for &xi in x.iter() {
            let mut p = 1.0;
            for _ in 0..k {
                v_data.push(p);
                p *= xi;
            }
        }
        let vm = DMatrix::from_row_slice(m, k, &v_data);
        let yv = nalgebra::DVector::from_iterator(m, y.iter().copied());
        let vt = vm.transpose();
        let ata = &vt * &vm;
        let aty = &vt * yv;
        let lu = ata.lu();
        let coeffs = lu
            .solve(&aty)
            .ok_or_else(|| anyhow!("polyfit: normal-equations singular"))?;
        let out: Vec<Value> = coeffs.iter().map(|&v| scalar_to_value(v)).collect();
        Ok(json!({"coefficients": out}))
    })
}

/// Inverse real-FFT — accepts `n/2 + 1` complex bins, returns `n` real values.
///
/// Args:   `{complex: {real, imag, shape: [m]}, n: u64}` where `n` is the
/// original signal length (`m = n/2 + 1`).
#[no_mangle]
pub extern "C" fn polars__fft_irfft(args: *const c_char) -> *mut c_char {
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
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n` (original signal length)"))?
            as usize;
        let m = real.len();
        if m != n / 2 + 1 {
            bail!("complex length {m} ≠ n/2+1 ({})", n / 2 + 1);
        }
        // Reconstruct full conjugate-symmetric spectrum.
        let mut buf: Vec<Complex<f64>> = Vec::with_capacity(n);
        for i in 0..m {
            let r = real[i].as_f64().unwrap_or(0.0);
            let im = imag[i].as_f64().unwrap_or(0.0);
            buf.push(Complex::new(r, im));
        }
        for i in (1..n - m + 1).rev() {
            let c = buf[i];
            buf.push(c.conj());
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(n);
        fft.process(&mut buf);
        let data: Vec<f64> = buf.iter().map(|c| c.re / n as f64).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("irfft shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Add two polynomials (coefficient-wise; pad the shorter one).
#[no_mangle]
pub extern "C" fn polars__poly_polyadd(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `a` (array)"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `b` (array)"))?;
        let n = a.len().max(b.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let av = a.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let bv = b.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
            out.push(scalar_to_value(av + bv));
        }
        Ok(json!({"coefficients": out}))
    })
}

/// Multiply two polynomials (convolution of coefficient lists).
#[no_mangle]
pub extern "C" fn polars__poly_polymul(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a_arr = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `a`"))?;
        let b_arr = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `b`"))?;
        let a: Vec<f64> = a_arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
        let b: Vec<f64> = b_arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
        if a.is_empty() || b.is_empty() {
            return Ok(json!({"coefficients": Vec::<Value>::new()}));
        }
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        let result: Vec<Value> = out.iter().map(|&v| scalar_to_value(v)).collect();
        Ok(json!({"coefficients": result}))
    })
}

/// Split a 1-D array into `n_parts` contiguous chunks. Returns array of arrays.
/// If `len(array) % n_parts != 0`, leftover elements go to the last chunk.
#[no_mangle]
pub extern "C" fn polars__arr_split(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("split: only 1-D arrays supported");
        }
        let n_parts = args
            .get("n_parts")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n_parts`"))? as usize;
        if n_parts == 0 {
            bail!("`n_parts` must be ≥ 1");
        }
        let data: Vec<f64> = arr.iter().copied().collect();
        let total = data.len();
        let base = total / n_parts;
        let mut chunks: Vec<Value> = Vec::with_capacity(n_parts);
        let mut start = 0;
        for i in 0..n_parts {
            let end = if i == n_parts - 1 {
                total
            } else {
                start + base
            };
            let chunk_data: Vec<f64> = data[start..end].to_vec();
            let chunk_n = chunk_data.len();
            let chunk =
                ArrayD::from_shape_vec(IxDyn(&[chunk_n]), chunk_data).context("split chunk")?;
            chunks.push(array_to_value(&chunk));
            start = end;
        }
        Ok(json!({"chunks": chunks}))
    })
}

/// `np.meshgrid(x, y)` — produces two 2-D arrays `X` (rows of x) and `Y`
/// (columns of y) with shape `(len(y), len(x))`.
#[no_mangle]
pub extern "C" fn polars__arr_meshgrid(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_array(&args, "x")?;
        let y = get_array(&args, "y")?;
        if x.shape().len() != 1 || y.shape().len() != 1 {
            bail!("meshgrid: x and y must be 1-D");
        }
        let nx = x.len();
        let ny = y.len();
        let mut x_data = Vec::with_capacity(nx * ny);
        let mut y_data = Vec::with_capacity(nx * ny);
        for &yi in y.iter() {
            for &xi in x.iter() {
                x_data.push(xi);
                y_data.push(yi);
            }
        }
        let xx = ArrayD::from_shape_vec(IxDyn(&[ny, nx]), x_data).context("meshgrid x")?;
        let yy = ArrayD::from_shape_vec(IxDyn(&[ny, nx]), y_data).context("meshgrid y")?;
        Ok(json!({"x": array_to_value(&xx), "y": array_to_value(&yy)}))
    })
}

/// `np.random.pareto(shape, n)`.
#[no_mangle]
pub extern "C" fn polars__rand_pareto(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let shape = args
            .get("shape")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `shape`"))?;
        if shape <= 0.0 {
            bail!("`shape` must be > 0");
        }
        let dist = rand_distr::Pareto::new(1.0, shape).context("Pareto::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("pareto shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.weibull(shape, n)` (scale=1).
#[no_mangle]
pub extern "C" fn polars__rand_weibull(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let shape = args
            .get("shape")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `shape`"))?;
        if shape <= 0.0 {
            bail!("`shape` must be > 0");
        }
        let dist = rand_distr::Weibull::new(1.0, shape).context("Weibull::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("weibull shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.cauchy(n)` — standard Cauchy(0, 1).
#[no_mangle]
pub extern "C" fn polars__rand_cauchy(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let dist = rand_distr::Cauchy::new(0.0, 1.0).context("Cauchy::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("cauchy shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.zeros_like(array)`.
#[no_mangle]
pub extern "C" fn polars__arr_zeros_like(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let zeros = ArrayD::<f64>::zeros(IxDyn(arr.shape()));
        Ok(json!({"array": array_to_value(&zeros)}))
    })
}

/// `np.ones_like(array)`.
#[no_mangle]
pub extern "C" fn polars__arr_ones_like(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let ones = ArrayD::<f64>::ones(IxDyn(arr.shape()));
        Ok(json!({"array": array_to_value(&ones)}))
    })
}

/// `np.full_like(array, value)`.
#[no_mangle]
pub extern "C" fn polars__arr_full_like(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let value = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `value`"))?;
        let n = arr.len();
        let data = vec![value; n];
        let out = ArrayD::from_shape_vec(IxDyn(arr.shape()), data).context("full_like shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.fft.ifftshift(x)`.
#[no_mangle]
pub extern "C" fn polars__fft_ifftshift(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("ifftshift: only 1-D arrays supported");
        }
        let data: Vec<f64> = arr.iter().copied().collect();
        let n = data.len();
        let mid = n - n / 2;
        let mut shifted = Vec::with_capacity(n);
        shifted.extend_from_slice(&data[mid..]);
        shifted.extend_from_slice(&data[..mid]);
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), shifted).context("ifftshift shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.sum_along_axis` for a wider axis count — sum keeping reduced axis as 1.
#[no_mangle]
pub extern "C" fn polars__arr_normalize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let s: f64 = arr.iter().sum();
        if s == 0.0 {
            bail!("normalize: array sums to 0");
        }
        let result = arr.mapv(|x| x / s);
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// `np.linspace` style for log scale: n geometrically-spaced points.
#[no_mangle]
pub extern "C" fn polars__arr_logspace(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let start = args
            .get("start")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `start` (exponent)"))?;
        let stop = args
            .get("stop")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `stop` (exponent)"))?;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let base = args.get("base").and_then(|v| v.as_f64()).unwrap_or(10.0);
        if n == 0 {
            bail!("`n` must be ≥ 1");
        }
        let step = if n == 1 {
            0.0
        } else {
            (stop - start) / (n as f64 - 1.0)
        };
        let data: Vec<f64> = (0..n).map(|i| base.powf(start + step * i as f64)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("logspace shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Extract diagonal of a 2-D array (1-D output of length `min(rows, cols)`).
#[no_mangle]
pub extern "C" fn polars__arr_diagonal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 2 {
            bail!("diagonal: 2-D array required");
        }
        let rows = arr.shape()[0];
        let cols = arr.shape()[1];
        let n = rows.min(cols);
        let data: Vec<f64> = (0..n).map(|i| arr[[i, i]]).collect();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("diagonal shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Sum of diagonal (`np.trace`) — 2-D array required.
#[no_mangle]
pub extern "C" fn polars__arr_trace(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 2 {
            bail!("trace: 2-D array required");
        }
        let rows = arr.shape()[0];
        let cols = arr.shape()[1];
        let n = rows.min(cols);
        let s: f64 = (0..n).map(|i| arr[[i, i]]).sum();
        Ok(json!({"scalar": scalar_to_value(s)}))
    })
}

/// Roll elements of a 1-D array by `shift` positions (positive = right).
#[no_mangle]
pub extern "C" fn polars__arr_roll(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("roll: only 1-D arrays supported");
        }
        let shift = args
            .get("shift")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing argument `shift`"))?;
        let data: Vec<f64> = arr.iter().copied().collect();
        let n = data.len();
        if n == 0 {
            return Ok(json!({"array": array_to_value(&arr)}));
        }
        let s = ((shift % n as i64) + n as i64) as usize % n;
        let mut out = Vec::with_capacity(n);
        out.extend_from_slice(&data[n - s..]);
        out.extend_from_slice(&data[..n - s]);
        let result = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("roll shape")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// Flatten any-D to 1-D (preserves row-major order).
#[no_mangle]
pub extern "C" fn polars__arr_flatten(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let data: Vec<f64> = arr.iter().copied().collect();
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("flatten shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Remove all length-1 dims (`np.squeeze`).
#[no_mangle]
pub extern "C" fn polars__arr_squeeze(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let new_shape: Vec<usize> = arr.shape().iter().copied().filter(|&d| d != 1).collect();
        let final_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };
        let data: Vec<f64> = arr.iter().copied().collect();
        let out = ArrayD::from_shape_vec(IxDyn(&final_shape), data).context("squeeze shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.sinc(x)` — normalized sinc: `sin(πx) / (πx)` (1 at x=0).
#[no_mangle]
pub extern "C" fn polars__np_sinc(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        unary_op(&args, |x| {
            if x == 0.0 {
                1.0
            } else {
                let px = std::f64::consts::PI * x;
                px.sin() / px
            }
        })
    })
}

/// `np.sign_bit(x)` — 1.0 if negative (or -0.0), else 0.0.
#[no_mangle]
pub extern "C" fn polars__np_signbit(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        unary_op(&args, |x| if x.is_sign_negative() { 1.0 } else { 0.0 })
    })
}

/// `np.deg2rad(x)`.
#[no_mangle]
pub extern "C" fn polars__np_deg2rad(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::to_radians))
}

/// `np.rad2deg(x)`.
#[no_mangle]
pub extern "C" fn polars__np_rad2deg(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::to_degrees))
}

/// `np.cross(a, b)` for 3-D vectors. Returns 1-D length-3 array.
#[no_mangle]
pub extern "C" fn polars__arr_cross(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape() != b.shape() || a.shape().len() != 1 || a.len() != 3 {
            bail!("cross: both inputs must be 1-D length 3");
        }
        let ax = a[[0]];
        let ay = a[[1]];
        let az = a[[2]];
        let bx = b[[0]];
        let by = b[[1]];
        let bz = b[[2]];
        let data = vec![ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
        let arr = ArrayD::from_shape_vec(IxDyn(&[3]), data).context("cross shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Subtract polynomial b from a (pad shorter to longer).
#[no_mangle]
pub extern "C" fn polars__poly_polysub(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `b`"))?;
        let n = a.len().max(b.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let av = a.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let bv = b.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
            out.push(scalar_to_value(av - bv));
        }
        Ok(json!({"coefficients": out}))
    })
}

/// Eigenvalues (real part) of a general matrix (descending by magnitude).
#[no_mangle]
pub extern "C" fn polars__linalg_eigvals(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        if !m.is_square() {
            bail!("eigvals: matrix must be square");
        }
        let sch = nalgebra::Schur::new(m);
        let evs = sch.complex_eigenvalues();
        let mut vals: Vec<f64> = evs.iter().map(|c| c.re).collect();
        vals.sort_by(|a, b| {
            b.abs()
                .partial_cmp(&a.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = vals.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), vals).context("eigvals shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.percentile(a, q)` — q in `[0, 100]`, 1-D array required.
#[no_mangle]
pub extern "C" fn polars__arr_percentile(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("percentile: only 1-D arrays supported");
        }
        let q = args
            .get("q")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `q`"))?;
        if !(0.0..=100.0).contains(&q) {
            bail!("`q` must be in [0, 100]");
        }
        let mut data: Vec<f64> = arr.iter().copied().collect();
        if data.is_empty() {
            bail!("percentile of empty array");
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = data.len();
        let pos = (q / 100.0) * (n as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        let val = if lo == hi {
            data[lo]
        } else {
            data[lo] + frac * (data[hi] - data[lo])
        };
        Ok(json!({"scalar": scalar_to_value(val)}))
    })
}

/// `np.gradient(f)` — numerical gradient using central differences.
/// 1-D only this slice. Output length equals input length.
#[no_mangle]
pub extern "C" fn polars__arr_gradient(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("gradient: only 1-D arrays supported");
        }
        let data: Vec<f64> = arr.iter().copied().collect();
        let n = data.len();
        if n < 2 {
            bail!("gradient: need ≥ 2 points");
        }
        let mut grad = Vec::with_capacity(n);
        // Forward at first.
        grad.push(data[1] - data[0]);
        // Central in middle.
        for i in 1..n - 1 {
            grad.push((data[i + 1] - data[i - 1]) / 2.0);
        }
        // Backward at last.
        grad.push(data[n - 1] - data[n - 2]);
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), grad).context("gradient shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.trapezoid(y, dx)` — trapezoidal integration. Scalar result.
#[no_mangle]
pub extern "C" fn polars__arr_trapezoid(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("trapezoid: only 1-D arrays supported");
        }
        let dx = args.get("dx").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let data: Vec<f64> = arr.iter().copied().collect();
        let n = data.len();
        if n < 2 {
            return Ok(json!({"scalar": scalar_to_value(0.0)}));
        }
        let mut acc = 0.0;
        for i in 0..n - 1 {
            acc += 0.5 * (data[i] + data[i + 1]) * dx;
        }
        Ok(json!({"scalar": scalar_to_value(acc)}))
    })
}

/// `np.convolve(a, v, mode='full')` — discrete 1-D convolution.
#[no_mangle]
pub extern "C" fn polars__arr_convolve(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let v = get_array(&args, "v")?;
        if a.shape().len() != 1 || v.shape().len() != 1 {
            bail!("convolve: both inputs must be 1-D");
        }
        let a_data: Vec<f64> = a.iter().copied().collect();
        let v_data: Vec<f64> = v.iter().copied().collect();
        if a_data.is_empty() || v_data.is_empty() {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let n = a_data.len() + v_data.len() - 1;
        let mut out = vec![0.0; n];
        for (i, &ai) in a_data.iter().enumerate() {
            for (j, &vj) in v_data.iter().enumerate() {
                out[i + j] += ai * vj;
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("convolve shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Are two arrays elementwise close within tolerance? Returns array of 0.0/1.0.
#[no_mangle]
pub extern "C" fn polars__arr_isclose(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape() != b.shape() {
            bail!("isclose: shape mismatch");
        }
        let rtol = args.get("rtol").and_then(|v| v.as_f64()).unwrap_or(1e-5);
        let atol = args.get("atol").and_then(|v| v.as_f64()).unwrap_or(1e-8);
        let data: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                if (x - y).abs() <= atol + rtol * y.abs() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let out = ArrayD::from_shape_vec(IxDyn(a.shape()), data).context("isclose shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Sample Pearson correlation between two 1-D arrays.
#[no_mangle]
pub extern "C" fn polars__arr_corrcoef(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape() != b.shape() || a.shape().len() != 1 {
            bail!("corrcoef: both must be 1-D same length");
        }
        let n = a.len() as f64;
        if n < 2.0 {
            bail!("corrcoef: need ≥ 2 points");
        }
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut sa = 0.0;
        let mut sb = 0.0;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let dx = x - mean_a;
            let dy = y - mean_b;
            num += dx * dy;
            sa += dx * dx;
            sb += dy * dy;
        }
        let denom = (sa * sb).sqrt();
        let r = if denom == 0.0 { f64::NAN } else { num / denom };
        Ok(json!({"scalar": scalar_to_value(r)}))
    })
}

/// Sample covariance between two 1-D arrays (ddof=1).
#[no_mangle]
pub extern "C" fn polars__arr_cov(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape() != b.shape() || a.shape().len() != 1 {
            bail!("cov: both must be 1-D same length");
        }
        let n = a.len() as f64;
        if n < 2.0 {
            bail!("cov: need ≥ 2 points");
        }
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;
        let sum: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - mean_a) * (y - mean_b))
            .sum();
        let c = sum / (n - 1.0);
        Ok(json!({"scalar": scalar_to_value(c)}))
    })
}

/// Histogram. Returns `{counts, edges}` with `n_bins+1` edges.
#[no_mangle]
pub extern "C" fn polars__arr_histogram(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("histogram: only 1-D arrays supported");
        }
        let n_bins = args
            .get("n_bins")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n_bins`"))? as usize;
        if n_bins == 0 {
            bail!("`n_bins` must be ≥ 1");
        }
        let data: Vec<f64> = arr.iter().copied().filter(|x| !x.is_nan()).collect();
        if data.is_empty() {
            bail!("histogram of empty array");
        }
        let lo = data.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
        let hi = data
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let span = if hi > lo { hi - lo } else { 1.0 };
        let width = span / n_bins as f64;
        let mut counts = vec![0u64; n_bins];
        for &v in &data {
            let idx = (((v - lo) / width) as usize).min(n_bins - 1);
            counts[idx] += 1;
        }
        let edges: Vec<Value> = (0..=n_bins)
            .map(|i| scalar_to_value(lo + width * i as f64))
            .collect();
        let counts_arr: Vec<Value> = counts.iter().map(|&c| json!(c)).collect();
        Ok(json!({"counts": counts_arr, "edges": edges}))
    })
}

/// `np.searchsorted(a, v)` — index where v would maintain ascending order in a.
#[no_mangle]
pub extern "C" fn polars__arr_searchsorted(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 1 {
            bail!("searchsorted: only 1-D arrays supported");
        }
        let v = args
            .get("v")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| anyhow!("missing argument `v` (scalar)"))?;
        let data: Vec<f64> = a.iter().copied().collect();
        // Binary search returning insertion point.
        let mut lo = 0usize;
        let mut hi = data.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if data[mid] < v {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        Ok(json!({"index": lo}))
    })
}

/// `np.random.chisquare(df, n)` — n chi-square samples.
#[no_mangle]
pub extern "C" fn polars__rand_chisquare(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let df = args
            .get("df")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `df` (degrees of freedom)"))?;
        if df <= 0.0 {
            bail!("`df` must be > 0");
        }
        let dist = rand_distr::ChiSquared::new(df).context("ChiSquared::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("chisquare shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.standard_t(df, n)`.
#[no_mangle]
pub extern "C" fn polars__rand_t(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let df = args
            .get("df")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `df`"))?;
        if df <= 0.0 {
            bail!("`df` must be > 0");
        }
        let dist = rand_distr::StudentT::new(df).context("StudentT::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("t shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.f(dfn, dfd, n)` — F-distribution.
#[no_mangle]
pub extern "C" fn polars__rand_f(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let dfn = args
            .get("dfn")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `dfn`"))?;
        let dfd = args
            .get("dfd")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `dfd`"))?;
        if dfn <= 0.0 || dfd <= 0.0 {
            bail!("`dfn` and `dfd` must be > 0");
        }
        let dist = rand_distr::FisherF::new(dfn, dfd).context("FisherF::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("f shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.correlate(a, v)` — discrete cross-correlation (`mode='full'`).
#[no_mangle]
pub extern "C" fn polars__arr_correlate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let v = get_array(&args, "v")?;
        if a.shape().len() != 1 || v.shape().len() != 1 {
            bail!("correlate: both inputs must be 1-D");
        }
        let a_data: Vec<f64> = a.iter().copied().collect();
        // numpy's correlate(a, v) is equivalent to convolve(a, reverse(v)).
        let mut v_data: Vec<f64> = v.iter().copied().collect();
        v_data.reverse();
        if a_data.is_empty() || v_data.is_empty() {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let n = a_data.len() + v_data.len() - 1;
        let mut out = vec![0.0; n];
        for (i, &ai) in a_data.iter().enumerate() {
            for (j, &vj) in v_data.iter().enumerate() {
                out[i + j] += ai * vj;
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("correlate shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.interp(x, xp, fp)` — 1-D linear interpolation.
///
/// Args:   `{x: <array>, xp: <array>, fp: <array>}`
/// Requirements: xp must be monotonically increasing; xp and fp same length.
#[no_mangle]
pub extern "C" fn polars__arr_interp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_array(&args, "x")?;
        let xp = get_array(&args, "xp")?;
        let fp = get_array(&args, "fp")?;
        if xp.shape().len() != 1 || fp.shape().len() != 1 {
            bail!("interp: xp and fp must be 1-D");
        }
        if xp.len() != fp.len() {
            bail!("interp: xp and fp must have same length");
        }
        let xp_data: Vec<f64> = xp.iter().copied().collect();
        let fp_data: Vec<f64> = fp.iter().copied().collect();
        let m = xp_data.len();
        if m == 0 {
            bail!("interp: empty xp/fp");
        }
        let interp_one = |xi: f64| -> f64 {
            if xi <= xp_data[0] {
                return fp_data[0];
            }
            if xi >= xp_data[m - 1] {
                return fp_data[m - 1];
            }
            // Binary search for upper bound.
            let mut lo = 0usize;
            let mut hi = m;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if xp_data[mid] <= xi {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let i = lo - 1;
            let x0 = xp_data[i];
            let x1 = xp_data[i + 1];
            let f0 = fp_data[i];
            let f1 = fp_data[i + 1];
            let frac = (xi - x0) / (x1 - x0);
            f0 + frac * (f1 - f0)
        };
        let data: Vec<f64> = x.iter().map(|&xi| interp_one(xi)).collect();
        let out = ArrayD::from_shape_vec(IxDyn(x.shape()), data).context("interp shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.bincount(x)` — count occurrences. Values truncated to non-negative ints.
#[no_mangle]
pub extern "C" fn polars__arr_bincount(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("bincount: only 1-D arrays supported");
        }
        let mut max_idx: i64 = -1;
        let ints: Vec<i64> = arr
            .iter()
            .map(|&x| {
                let i = x as i64;
                if i > max_idx {
                    max_idx = i;
                }
                i
            })
            .collect();
        if max_idx < 0 {
            return Ok(json!({"array": array_to_value(&arr.mapv(|_| 0.0))}));
        }
        let n_bins = (max_idx + 1) as usize;
        let mut counts = vec![0u64; n_bins];
        for &i in &ints {
            if i >= 0 {
                counts[i as usize] += 1;
            }
        }
        let data: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        let out = ArrayD::from_shape_vec(IxDyn(&[n_bins]), data).context("bincount shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Swap two axes of an N-D array.
///
/// Args:   `{array, axis1: u64, axis2: u64}`
#[no_mangle]
pub extern "C" fn polars__arr_swapaxes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let a1 = args
            .get("axis1")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis1`"))? as usize;
        let a2 = args
            .get("axis2")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis2`"))? as usize;
        if a1 >= arr.shape().len() || a2 >= arr.shape().len() {
            bail!("swapaxes: axis out of range");
        }
        let mut swapped = arr.clone();
        swapped.swap_axes(a1, a2);
        let owned: ArrayD<f64> = swapped.to_owned();
        Ok(json!({"array": array_to_value(&owned)}))
    })
}

/// `np.modf(x)` — split into fractional and integer parts.
/// Returns `{frac, int}` (two arrays of input shape).
#[no_mangle]
pub extern "C" fn polars__np_modf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let frac = arr.mapv(|x| x.fract());
        let int = arr.mapv(|x| x.trunc());
        Ok(json!({"frac": array_to_value(&frac), "int": array_to_value(&int)}))
    })
}

/// Ensure at least 1-D — wraps scalar input into a 1-element array.
#[no_mangle]
pub extern "C" fn polars__arr_atleast_1d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().is_empty() {
            let s = arr.iter().next().copied().unwrap_or(0.0);
            let out = ArrayD::from_shape_vec(IxDyn(&[1]), vec![s]).context("atleast_1d")?;
            return Ok(json!({"array": array_to_value(&out)}));
        }
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Ensure at least 2-D — wraps 1-D into `(1, n)`.
#[no_mangle]
pub extern "C" fn polars__arr_atleast_2d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        match arr.shape().len() {
            0 => {
                let s = arr.iter().next().copied().unwrap_or(0.0);
                let out = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![s]).context("atleast_2d")?;
                Ok(json!({"array": array_to_value(&out)}))
            }
            1 => {
                let n = arr.len();
                let data: Vec<f64> = arr.iter().copied().collect();
                let out =
                    ArrayD::from_shape_vec(IxDyn(&[1, n]), data).context("atleast_2d shape")?;
                Ok(json!({"array": array_to_value(&out)}))
            }
            _ => Ok(json!({"array": array_to_value(&arr)})),
        }
    })
}

/// `np.array_equal(a, b)` — true if same shape AND all elements equal.
#[no_mangle]
pub extern "C" fn polars__arr_array_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape() != b.shape() {
            return Ok(json!({"bool": false}));
        }
        let eq = a
            .iter()
            .zip(b.iter())
            .all(|(&x, &y)| x == y || (x.is_nan() && y.is_nan()));
        Ok(json!({"bool": eq}))
    })
}

/// `np.allclose(a, b, rtol, atol)` — all elementwise close.
#[no_mangle]
pub extern "C" fn polars__arr_allclose(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape() != b.shape() {
            return Ok(json!({"bool": false}));
        }
        let rtol = args.get("rtol").and_then(|v| v.as_f64()).unwrap_or(1e-5);
        let atol = args.get("atol").and_then(|v| v.as_f64()).unwrap_or(1e-8);
        let all = a
            .iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x - y).abs() <= atol + rtol * y.abs());
        Ok(json!({"bool": all}))
    })
}

/// `np.logaddexp(a, b)` — numerically stable log(exp(a) + exp(b)).
#[no_mangle]
pub extern "C" fn polars__np_logaddexp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |a, b| {
            let m = a.max(b);
            if !m.is_finite() {
                m
            } else {
                m + (-(a - b).abs()).exp().ln_1p()
            }
        })
    })
}

/// `np.logaddexp2(a, b)` — base-2 numerically stable log2(2^a + 2^b).
#[no_mangle]
pub extern "C" fn polars__np_logaddexp2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |a, b| {
            let m = a.max(b);
            if !m.is_finite() {
                m
            } else {
                m + (1.0 + 2.0_f64.powf(-(a - b).abs())).log2()
            }
        })
    })
}

/// Compress: keep elements of `array` where `mask` is non-zero.
#[no_mangle]
pub extern "C" fn polars__arr_compress(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let mask = get_array(&args, "mask")?;
        if arr.shape() != mask.shape() {
            bail!("compress: array and mask must share shape");
        }
        let data: Vec<f64> = arr
            .iter()
            .zip(mask.iter())
            .filter_map(|(&v, &m)| if m != 0.0 { Some(v) } else { None })
            .collect();
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("compress shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.size(a)` — total element count as scalar.
#[no_mangle]
pub extern "C" fn polars__arr_size(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        Ok(json!({"size": arr.len()}))
    })
}

/// `np.ndim(a)` — number of dimensions.
#[no_mangle]
pub extern "C" fn polars__arr_ndim(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        Ok(json!({"ndim": arr.shape().len()}))
    })
}

/// `np.shape(a)` — shape as 1-D array.
#[no_mangle]
pub extern "C" fn polars__arr_shape(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let shape: Vec<u64> = arr.shape().iter().map(|&d| d as u64).collect();
        Ok(json!({"shape": shape}))
    })
}

/// `np.fill_diagonal` — set the diagonal of a 2-D array to `value`.
/// Returns the modified array.
#[no_mangle]
pub extern "C" fn polars__arr_fill_diagonal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 2 {
            bail!("fill_diagonal: 2-D required");
        }
        let value = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `value`"))?;
        let rows = arr.shape()[0];
        let cols = arr.shape()[1];
        let n = rows.min(cols);
        let mut data: Vec<f64> = arr.iter().copied().collect();
        for i in 0..n {
            data[i * cols + i] = value;
        }
        let out =
            ArrayD::from_shape_vec(IxDyn(arr.shape()), data).context("fill_diagonal shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.take(a, indices)` — gather values by 1-D indices (flat order).
#[no_mangle]
pub extern "C" fn polars__arr_take(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let idx_arr = args
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `indices`"))?;
        let flat: Vec<f64> = arr.iter().copied().collect();
        let n = flat.len();
        let mut out = Vec::with_capacity(idx_arr.len());
        for v in idx_arr {
            let i = v.as_u64().ok_or_else(|| anyhow!("non-int index"))? as usize;
            if i >= n {
                bail!("take: index {i} out of range ({n})");
            }
            out.push(flat[i]);
        }
        let m = out.len();
        let result = ArrayD::from_shape_vec(IxDyn(&[m]), out).context("take shape")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// `np.put(a, indices, values)` — scatter values into a (returns modified copy).
#[no_mangle]
pub extern "C" fn polars__arr_put(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let idx_arr = args
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `indices`"))?;
        let val_arr = args
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `values`"))?;
        if idx_arr.len() != val_arr.len() {
            bail!("put: indices/values length mismatch");
        }
        let mut data: Vec<f64> = arr.iter().copied().collect();
        let n = data.len();
        for (iv, vv) in idx_arr.iter().zip(val_arr.iter()) {
            let i = iv.as_u64().ok_or_else(|| anyhow!("non-int index"))? as usize;
            let v = vv.as_f64().ok_or_else(|| anyhow!("non-numeric value"))?;
            if i >= n {
                bail!("put: index {i} out of range ({n})");
            }
            data[i] = v;
        }
        let out = ArrayD::from_shape_vec(IxDyn(arr.shape()), data).context("put shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Variance along an axis (population, ddof=0).
#[no_mangle]
pub extern "C" fn polars__arr_var_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis out of range");
        }
        let out = arr.map_axis(Axis(axis), |row| {
            let n = row.len() as f64;
            if n == 0.0 {
                return f64::NAN;
            }
            let mean = row.iter().sum::<f64>() / n;
            row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
        });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Std along an axis (population, ddof=0).
#[no_mangle]
pub extern "C" fn polars__arr_std_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis out of range");
        }
        let out = arr.map_axis(Axis(axis), |row| {
            let n = row.len() as f64;
            if n == 0.0 {
                return f64::NAN;
            }
            let mean = row.iter().sum::<f64>() / n;
            (row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
        });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.prod_axis(a, axis)` — product along an axis.
#[no_mangle]
pub extern "C" fn polars__arr_prod_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis out of range");
        }
        let out = arr.map_axis(Axis(axis), |row| row.iter().product::<f64>());
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.unique_with_counts(a)` — sorted unique values + per-value counts.
/// 1-D only.
#[no_mangle]
pub extern "C" fn polars__arr_unique_counts(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("unique_counts: 1-D required");
        }
        let mut data: Vec<f64> = arr.iter().copied().collect();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut values: Vec<f64> = Vec::new();
        let mut counts: Vec<u64> = Vec::new();
        for v in data {
            if let Some(&last) = values.last() {
                if (v - last).abs() < 1e-15 {
                    *counts.last_mut().unwrap() += 1;
                    continue;
                }
            }
            values.push(v);
            counts.push(1);
        }
        let n = values.len();
        let val_arr = ArrayD::from_shape_vec(IxDyn(&[n]), values).context("unique values")?;
        let count_vals: Vec<Value> = counts.iter().map(|&c| json!(c)).collect();
        Ok(json!({"values": array_to_value(&val_arr), "counts": count_vals}))
    })
}

/// `np.argmax(a, axis)` along an axis.
#[no_mangle]
pub extern "C" fn polars__arr_argmax_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis out of range");
        }
        let out =
            arr.map_axis(Axis(axis), |row| {
                let (i, _) = row.iter().enumerate().fold(
                    (0usize, f64::NEG_INFINITY),
                    |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                );
                i as f64
            });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.argmin(a, axis)` along an axis.
#[no_mangle]
pub extern "C" fn polars__arr_argmin_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis out of range");
        }
        let out = arr.map_axis(Axis(axis), |row| {
            let (i, _) =
                row.iter()
                    .enumerate()
                    .fold((0usize, f64::INFINITY), |(bi, bv), (i, &v)| {
                        if v < bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    });
            i as f64
        });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.select(conditions, choices, default)` — first true condition's choice.
///
/// Args:   `{conditions: [<arr>, ...], choices: [<arr>, ...], default: f64}`
#[no_mangle]
pub extern "C" fn polars__np_select(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let conds = args
            .get("conditions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `conditions`"))?;
        let choices = args
            .get("choices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `choices`"))?;
        if conds.len() != choices.len() {
            bail!("select: conditions/choices length mismatch");
        }
        if conds.is_empty() {
            bail!("select: need at least one condition");
        }
        let default = args.get("default").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let cond_arrs: Vec<ArrayD<f64>> = conds.iter().map(parse_array).collect::<Result<_>>()?;
        let choice_arrs: Vec<ArrayD<f64>> =
            choices.iter().map(parse_array).collect::<Result<_>>()?;
        let shape = cond_arrs[0].shape().to_vec();
        for c in &cond_arrs {
            if c.shape() != shape.as_slice() {
                bail!("select: condition shapes mismatch");
            }
        }
        for c in &choice_arrs {
            if c.shape() != shape.as_slice() {
                bail!("select: choice shapes mismatch");
            }
        }
        let total: usize = shape.iter().product();
        let mut data = vec![default; total];
        for (i, slot) in data.iter_mut().enumerate().take(total) {
            for k in 0..cond_arrs.len() {
                if cond_arrs[k].as_slice().unwrap()[i] != 0.0 {
                    *slot = choice_arrs[k].as_slice().unwrap()[i];
                    break;
                }
            }
        }
        let out = ArrayD::from_shape_vec(IxDyn(&shape), data).context("select shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Element-set equality after sort + dedup. 1-D inputs.
#[no_mangle]
pub extern "C" fn polars__arr_setdiff1d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 1 || b.shape().len() != 1 {
            bail!("setdiff1d: 1-D inputs required");
        }
        let mut b_set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for &v in b.iter() {
            b_set.insert(v.to_bits() as i64);
        }
        let mut result: Vec<f64> = Vec::new();
        let mut seen: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for &v in a.iter() {
            let bits = v.to_bits() as i64;
            if !b_set.contains(&bits) && seen.insert(bits) {
                result.push(v);
            }
        }
        result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = result.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), result).context("setdiff1d shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Sorted intersection of two 1-D arrays (unique elements only).
#[no_mangle]
pub extern "C" fn polars__arr_intersect1d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 1 || b.shape().len() != 1 {
            bail!("intersect1d: 1-D inputs required");
        }
        let mut b_set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for &v in b.iter() {
            b_set.insert(v.to_bits() as i64);
        }
        let mut result: Vec<f64> = Vec::new();
        let mut seen: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for &v in a.iter() {
            let bits = v.to_bits() as i64;
            if b_set.contains(&bits) && seen.insert(bits) {
                result.push(v);
            }
        }
        result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = result.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), result).context("intersect1d shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Sorted union of two 1-D arrays (unique elements).
#[no_mangle]
pub extern "C" fn polars__arr_union1d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 1 || b.shape().len() != 1 {
            bail!("union1d: 1-D inputs required");
        }
        let mut seen: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        let mut result: Vec<f64> = Vec::new();
        for &v in a.iter().chain(b.iter()) {
            let bits = v.to_bits() as i64;
            if seen.insert(bits) {
                result.push(v);
            }
        }
        result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = result.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), result).context("union1d shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.in1d(a, b)` — boolean (1.0/0.0) array: is each element of a in b?
#[no_mangle]
pub extern "C" fn polars__arr_in1d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 1 || b.shape().len() != 1 {
            bail!("in1d: 1-D inputs required");
        }
        let mut b_set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for &v in b.iter() {
            b_set.insert(v.to_bits() as i64);
        }
        let data: Vec<f64> = a
            .iter()
            .map(|&v| {
                if b_set.contains(&(v.to_bits() as i64)) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("in1d shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.cumtrapz(y, dx)` — cumulative trapezoidal integration. 1-D, output length n-1.
#[no_mangle]
pub extern "C" fn polars__arr_cumtrapz(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("cumtrapz: 1-D required");
        }
        let dx = args.get("dx").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let data: Vec<f64> = arr.iter().copied().collect();
        if data.len() < 2 {
            return Ok(json!({
                "array": array_to_value(&ArrayD::from_shape_vec(IxDyn(&[0]), Vec::<f64>::new()).unwrap())
            }));
        }
        let mut acc = 0.0;
        let mut out = Vec::with_capacity(data.len() - 1);
        for i in 0..data.len() - 1 {
            acc += 0.5 * (data[i] + data[i + 1]) * dx;
            out.push(acc);
        }
        let n = out.len();
        let result = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("cumtrapz shape")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// `np.pad(a, pad_width, constant_value)` — constant-pad 1-D array.
#[no_mangle]
pub extern "C" fn polars__arr_pad(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("pad: 1-D required");
        }
        let before = args.get("before").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let after = args.get("after").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let value = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mut data: Vec<f64> = Vec::with_capacity(before + arr.len() + after);
        for _ in 0..before {
            data.push(value);
        }
        for &v in arr.iter() {
            data.push(v);
        }
        for _ in 0..after {
            data.push(value);
        }
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("pad shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.fix(x)` — round toward zero (alias for trunc).
#[no_mangle]
pub extern "C" fn polars__np_fix(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, f64::trunc))
}

/// `np.around(x, decimals)` — round to nearest at given precision.
#[no_mangle]
pub extern "C" fn polars__np_around(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let decimals = args.get("decimals").and_then(|v| v.as_i64()).unwrap_or(0);
        let factor = 10f64.powi(decimals as i32);
        let result = arr.mapv(|x| (x * factor).round() / factor);
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// `np.inner(a, b)` — 1-D inner product (scalar). Same length required.
#[no_mangle]
pub extern "C" fn polars__arr_inner(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 1 || b.shape().len() != 1 {
            bail!("inner: 1-D inputs required");
        }
        if a.len() != b.len() {
            bail!("inner: length mismatch");
        }
        let s: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        Ok(json!({"scalar": scalar_to_value(s)}))
    })
}

/// `np.linalg.norm(a, axis)` — axis-wise norm.
#[no_mangle]
pub extern "C" fn polars__arr_norm_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis out of range");
        }
        let out = arr.map_axis(Axis(axis), |row| {
            row.iter().map(|x| x * x).sum::<f64>().sqrt()
        });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.linalg.lstsq(A, b)` — least-squares solution via SVD pseudo-inverse.
#[no_mangle]
pub extern "C" fn polars__linalg_lstsq(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing `matrix`"))?,
        )?;
        let b_arr = parse_array(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        if b_arr.shape().len() != 1 {
            bail!("lstsq: b must be 1-D");
        }
        if b_arr.len() != a.nrows() {
            bail!("lstsq: shape mismatch");
        }
        let svd = a.svd(true, true);
        let pinv = svd
            .pseudo_inverse(1e-10)
            .map_err(|e| anyhow!("pinv: {e}"))?;
        let bv = nalgebra::DVector::from_iterator(b_arr.len(), b_arr.iter().copied());
        let x = pinv * bv;
        let n = x.len();
        let result = ArrayD::from_shape_vec(IxDyn(&[n]), x.iter().copied().collect())
            .context("lstsq shape")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// `np.diff_n(a, n)` — n-th order discrete difference.
#[no_mangle]
pub extern "C" fn polars__arr_diff_n(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("diff_n: 1-D required");
        }
        let n_diff = args.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mut data: Vec<f64> = arr.iter().copied().collect();
        for _ in 0..n_diff {
            if data.len() < 2 {
                data = Vec::new();
                break;
            }
            let next: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
            data = next;
        }
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("diff_n shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.column_stack(a, b)` — stack two 1-D arrays as columns of a 2-D array.
#[no_mangle]
pub extern "C" fn polars__arr_column_stack(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 1 || b.shape().len() != 1 {
            bail!("column_stack: 1-D inputs required");
        }
        if a.len() != b.len() {
            bail!("column_stack: length mismatch");
        }
        let n = a.len();
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.push(a[[i]]);
            data.push(b[[i]]);
        }
        let out = ArrayD::from_shape_vec(IxDyn(&[n, 2]), data).context("column_stack shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.row_stack(a, b)` — stack two 1-D arrays as rows of a 2-D array.
#[no_mangle]
pub extern "C" fn polars__arr_row_stack(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 1 || b.shape().len() != 1 {
            bail!("row_stack: 1-D inputs required");
        }
        if a.len() != b.len() {
            bail!("row_stack: length mismatch");
        }
        let n = a.len();
        let mut data = Vec::with_capacity(2 * n);
        for &v in a.iter() {
            data.push(v);
        }
        for &v in b.iter() {
            data.push(v);
        }
        let out = ArrayD::from_shape_vec(IxDyn(&[2, n]), data).context("row_stack shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.real(a)` — identity since this layer is real-only.
#[no_mangle]
pub extern "C" fn polars__arr_real(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.imag(a)` — zeros_like since this layer is real-only.
#[no_mangle]
pub extern "C" fn polars__arr_imag(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let zeros = ArrayD::<f64>::zeros(IxDyn(arr.shape()));
        Ok(json!({"array": array_to_value(&zeros)}))
    })
}

/// `np.positive(a)` — identity. Mirrors `np.negative`.
#[no_mangle]
pub extern "C" fn polars__np_positive(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| unary_op(&args, |x| x))
}

/// `np.iscomplex(a)` — false for every element (this layer is real-only).
#[no_mangle]
pub extern "C" fn polars__arr_iscomplex(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let zeros = ArrayD::<f64>::zeros(IxDyn(arr.shape()));
        Ok(json!({"array": array_to_value(&zeros)}))
    })
}

/// `np.isreal(a)` — true for every element.
#[no_mangle]
pub extern "C" fn polars__arr_isreal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let ones = ArrayD::<f64>::ones(IxDyn(arr.shape()));
        Ok(json!({"array": array_to_value(&ones)}))
    })
}

/// `np.flip(a)` — reverse along an axis (1-D only this slice).
#[no_mangle]
pub extern "C" fn polars__arr_flip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("flip: 1-D only in this slice");
        }
        let mut data: Vec<f64> = arr.iter().copied().collect();
        data.reverse();
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("flip shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.ptp(a)` — peak-to-peak (max − min).
#[no_mangle]
pub extern "C" fn polars__arr_ptp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("ptp: empty array");
        }
        let lo = arr.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
        let hi = arr.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        Ok(json!({"scalar": scalar_to_value(hi - lo)}))
    })
}

/// `np.count_nonzero(a, axis)` along axis.
#[no_mangle]
pub extern "C" fn polars__arr_count_nonzero_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis out of range");
        }
        let out = arr.map_axis(Axis(axis), |row| {
            row.iter().filter(|&&x| x != 0.0).count() as f64
        });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.flip` along an axis (2-D supported).
#[no_mangle]
pub extern "C" fn polars__arr_flip_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis out of range");
        }
        let mut flipped = arr.clone();
        flipped.invert_axis(Axis(axis));
        let owned: ArrayD<f64> = flipped.to_owned();
        Ok(json!({"array": array_to_value(&owned)}))
    })
}

/// `np.broadcast_to(a, shape)` — for 1-D inputs duplicated to fit a new shape.
/// Restricted: input length must be 1 or equal to last dim of new shape.
#[no_mangle]
pub extern "C" fn polars__arr_broadcast_to(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let new_shape = shape_arg(&args)?;
        let target: usize = new_shape.iter().product();
        let src_len = arr.len();
        if src_len == 1 {
            let v = arr.iter().next().copied().unwrap_or(0.0);
            let data = vec![v; target];
            let out = ArrayD::from_shape_vec(IxDyn(&new_shape), data).context("broadcast shape")?;
            return Ok(json!({"array": array_to_value(&out)}));
        }
        if !target.is_multiple_of(src_len) {
            bail!("broadcast_to: target size {target} not divisible by source len {src_len}");
        }
        let reps = target / src_len;
        let mut data = Vec::with_capacity(target);
        for _ in 0..reps {
            for &v in arr.iter() {
                data.push(v);
            }
        }
        let out = ArrayD::from_shape_vec(IxDyn(&new_shape), data).context("broadcast shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

// ── P4ab: scalar-broadcast ufuncs ──────────────────────────────────────────

fn scalar_op<F: Fn(f64, f64) -> f64>(args: &Value, f: F) -> Result<Value> {
    let arr = get_array(args, "array")?;
    let scalar = args
        .get("scalar")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow!("missing argument `scalar`"))?;
    let result = arr.mapv(|x| f(x, scalar));
    Ok(json!({"array": array_to_value(&result)}))
}

/// Add a scalar to every element of array.
#[no_mangle]
pub extern "C" fn polars__np_add_scalar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| scalar_op(&args, |x, s| x + s))
}

/// Subtract a scalar from every element.
#[no_mangle]
pub extern "C" fn polars__np_subtract_scalar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| scalar_op(&args, |x, s| x - s))
}

/// Multiply every element by a scalar.
#[no_mangle]
pub extern "C" fn polars__np_multiply_scalar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| scalar_op(&args, |x, s| x * s))
}

/// Divide every element by a scalar.
#[no_mangle]
pub extern "C" fn polars__np_divide_scalar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| scalar_op(&args, |x, s| x / s))
}

/// Raise every element to a scalar power.
#[no_mangle]
pub extern "C" fn polars__np_power_scalar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| scalar_op(&args, f64::powf))
}

/// Mean-center: subtract the array's mean from every element.
#[no_mangle]
pub extern "C" fn polars__arr_mean_centered(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("mean_centered: empty array");
        }
        let mean = arr.iter().sum::<f64>() / arr.len() as f64;
        let result = arr.mapv(|x| x - mean);
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// `np.linalg.normalize_l2(a)` — divide every element by the 2-norm.
#[no_mangle]
pub extern "C" fn polars__arr_normalize_l2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let norm = arr.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            bail!("normalize_l2: zero-norm array");
        }
        let result = arr.mapv(|x| x / norm);
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// `np.geomspace(start, stop, n)` — geometric (multiplicative) progression.
#[no_mangle]
pub extern "C" fn polars__arr_geomspace(args: *const c_char) -> *mut c_char {
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
        if start <= 0.0 || stop <= 0.0 {
            bail!("geomspace: start and stop must be positive");
        }
        if n == 0 {
            bail!("`n` must be ≥ 1");
        }
        let ratio = if n == 1 {
            1.0
        } else {
            (stop / start).powf(1.0 / (n as f64 - 1.0))
        };
        let data: Vec<f64> = (0..n).map(|i| start * ratio.powi(i as i32)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("geomspace shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.partition(a, kth)` — partial-sort: `kth` element in its final position.
/// 1-D only. Returns the rearranged array.
#[no_mangle]
pub extern "C" fn polars__arr_partition(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("partition: 1-D required");
        }
        let kth = args
            .get("kth")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `kth`"))? as usize;
        let mut data: Vec<f64> = arr.iter().copied().collect();
        if kth >= data.len() {
            bail!("partition: kth out of range");
        }
        data.select_nth_unstable_by(kth, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("partition shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.kth_smallest(a, k)` — k-th smallest element (0-indexed). 1-D.
#[no_mangle]
pub extern "C" fn polars__arr_kth_smallest(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("kth_smallest: 1-D required");
        }
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `k`"))? as usize;
        let mut data: Vec<f64> = arr.iter().copied().collect();
        if k >= data.len() {
            bail!("kth_smallest: k out of range");
        }
        let (_, &mut v, _) = data.select_nth_unstable_by(k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(json!({"scalar": scalar_to_value(v)}))
    })
}

/// `np.linspace_endpoint(start, stop, n, endpoint=False)` — exclusive variant.
#[no_mangle]
pub extern "C" fn polars__arr_linspace_no_endpoint(args: *const c_char) -> *mut c_char {
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
        let step = (stop - start) / n as f64;
        let data: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
        let arr =
            ArrayD::from_shape_vec(IxDyn(&[n]), data).context("linspace_no_endpoint shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Min and max in one pass. Returns `{min, max}`.
#[no_mangle]
pub extern "C" fn polars__arr_minmax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("minmax: empty array");
        }
        let (lo, hi) = arr
            .iter()
            .copied()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            });
        Ok(json!({"min": scalar_to_value(lo), "max": scalar_to_value(hi)}))
    })
}

/// Sum of squares (Σ x²).
#[no_mangle]
pub extern "C" fn polars__arr_sum_sq(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let s: f64 = arr.iter().map(|x| x * x).sum();
        Ok(json!({"scalar": scalar_to_value(s)}))
    })
}

/// Root mean square: sqrt(mean(x²)).
#[no_mangle]
pub extern "C" fn polars__arr_rms(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let n = arr.len() as f64;
        if n == 0.0 {
            bail!("rms: empty array");
        }
        let s: f64 = arr.iter().map(|x| x * x).sum();
        Ok(json!({"scalar": scalar_to_value((s / n).sqrt())}))
    })
}

/// Numerically stable softmax (1-D).
#[no_mangle]
pub extern "C" fn polars__arr_softmax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("softmax: empty array");
        }
        let m = arr.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let exps: Vec<f64> = arr.iter().map(|&x| (x - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        let data: Vec<f64> = exps.iter().map(|&e| e / s).collect();
        let out = ArrayD::from_shape_vec(IxDyn(arr.shape()), data).context("softmax shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Geometric mean (positives only).
#[no_mangle]
pub extern "C" fn polars__arr_geometric_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("geometric_mean: empty array");
        }
        if arr.iter().any(|&x| x <= 0.0) {
            bail!("geometric_mean: all elements must be > 0");
        }
        let n = arr.len() as f64;
        let log_sum: f64 = arr.iter().map(|x| x.ln()).sum();
        Ok(json!({"scalar": scalar_to_value((log_sum / n).exp())}))
    })
}

/// Harmonic mean (positives only).
#[no_mangle]
pub extern "C" fn polars__arr_harmonic_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("harmonic_mean: empty array");
        }
        if arr.iter().any(|&x| x <= 0.0) {
            bail!("harmonic_mean: all elements must be > 0");
        }
        let n = arr.len() as f64;
        let recip_sum: f64 = arr.iter().map(|x| 1.0 / x).sum();
        Ok(json!({"scalar": scalar_to_value(n / recip_sum)}))
    })
}

// ── P4ae: stats moments / scaling / info-theoretic ─────────────────────────

/// Sample skewness (g₁).
#[no_mangle]
pub extern "C" fn polars__arr_skewness(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let n = arr.len() as f64;
        if n < 2.0 {
            bail!("skewness: need ≥ 2 points");
        }
        let mean = arr.iter().sum::<f64>() / n;
        let m2: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3: f64 = arr.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        let s = if m2 == 0.0 { 0.0 } else { m3 / m2.powf(1.5) };
        Ok(json!({"scalar": scalar_to_value(s)}))
    })
}

/// Sample kurtosis (g₂, excess).
#[no_mangle]
pub extern "C" fn polars__arr_kurtosis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let n = arr.len() as f64;
        if n < 2.0 {
            bail!("kurtosis: need ≥ 2 points");
        }
        let mean = arr.iter().sum::<f64>() / n;
        let m2: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4: f64 = arr.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
        let k = if m2 == 0.0 { 0.0 } else { m4 / (m2 * m2) - 3.0 };
        Ok(json!({"scalar": scalar_to_value(k)}))
    })
}

/// z-score standardization: `(x − mean) / std` (population std, ddof=0).
#[no_mangle]
pub extern "C" fn polars__arr_zscore(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let n = arr.len() as f64;
        if n == 0.0 {
            bail!("zscore: empty array");
        }
        let mean = arr.iter().sum::<f64>() / n;
        let var: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        if std == 0.0 {
            bail!("zscore: zero std (constant array)");
        }
        let result = arr.mapv(|x| (x - mean) / std);
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// Min-max scale to `[0, 1]`.
#[no_mangle]
pub extern "C" fn polars__arr_minmax_scale(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("minmax_scale: empty array");
        }
        let lo = arr.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
        let hi = arr.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let span = hi - lo;
        if span == 0.0 {
            let zeros = ArrayD::<f64>::zeros(IxDyn(arr.shape()));
            return Ok(json!({"array": array_to_value(&zeros)}));
        }
        let result = arr.mapv(|x| (x - lo) / span);
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// Numerically stable `log(Σ exp(x))`.
#[no_mangle]
pub extern "C" fn polars__arr_logsumexp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("logsumexp: empty array");
        }
        let m = arr.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        if !m.is_finite() {
            return Ok(json!({"scalar": scalar_to_value(m)}));
        }
        let s: f64 = arr.iter().map(|&x| (x - m).exp()).sum();
        Ok(json!({"scalar": scalar_to_value(m + s.ln())}))
    })
}

/// Shannon entropy `-Σ p log p` (natural log). p_i must sum to 1; zeros allowed.
#[no_mangle]
pub extern "C" fn polars__arr_entropy(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.is_empty() {
            bail!("entropy: empty array");
        }
        let s: f64 = arr
            .iter()
            .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
            .sum();
        Ok(json!({"scalar": scalar_to_value(s)}))
    })
}

/// KL divergence `D(p ‖ q) = Σ p log(p/q)`. Both same length; entries > 0 for q.
#[no_mangle]
pub extern "C" fn polars__arr_kl_divergence(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let p = get_array(&args, "p")?;
        let q = get_array(&args, "q")?;
        if p.shape() != q.shape() {
            bail!("kl_divergence: shape mismatch");
        }
        let mut acc = 0.0;
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi <= 0.0 {
                continue;
            }
            if qi <= 0.0 {
                bail!("kl_divergence: q has non-positive at index with positive p");
            }
            acc += pi * (pi / qi).ln();
        }
        Ok(json!({"scalar": scalar_to_value(acc)}))
    })
}

/// `np.clip(a, lo, hi)` — scalar bounds applied elementwise.
#[no_mangle]
pub extern "C" fn polars__np_clip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let lo = args.get("lower").and_then(|v| v.as_f64());
        let hi = args.get("upper").and_then(|v| v.as_f64());
        if lo.is_none() && hi.is_none() {
            bail!("must provide at least one of `lower`, `upper`");
        }
        let out = arr.mapv(|x| {
            let mut v = x;
            if let Some(l) = lo {
                if v < l {
                    v = l;
                }
            }
            if let Some(h) = hi {
                if v > h {
                    v = h;
                }
            }
            v
        });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.where(cond, a, b)` — elementwise; cond non-zero ⇒ a, else b.
#[no_mangle]
pub extern "C" fn polars__np_where(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let cond = get_array(&args, "cond")?;
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if cond.shape() != a.shape() || cond.shape() != b.shape() {
            bail!("where: shape mismatch");
        }
        let data: Vec<f64> = cond
            .iter()
            .zip(a.iter())
            .zip(b.iter())
            .map(|((&c, &x), &y)| if c != 0.0 { x } else { y })
            .collect();
        let out = ArrayD::from_shape_vec(IxDyn(cond.shape()), data).context("where shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// `np.fft.rfftfreq(n, d)` — sample frequencies for `rfft` output (length `n/2+1`).
#[no_mangle]
pub extern "C" fn polars__fft_rfftfreq(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let d = args.get("d").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if n == 0 {
            bail!("`n` must be ≥ 1");
        }
        let m = n / 2 + 1;
        let n_f = n as f64;
        let data: Vec<f64> = (0..m).map(|k| k as f64 / (n_f * d)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[m]), data).context("rfftfreq shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Real eigenvalues of a symmetric matrix (descending).
#[no_mangle]
pub extern "C" fn polars__linalg_eigvalsh(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        if !m.is_square() {
            bail!("eigvalsh: matrix must be square");
        }
        let sym = nalgebra::SymmetricEigen::new(m);
        let mut vals: Vec<f64> = sym.eigenvalues.iter().copied().collect();
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), vals).context("eigvalsh shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
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

// ── P4c: more ndarray reductions / shape ops ───────────────────────────────

/// Any element non-zero?
#[no_mangle]
pub extern "C" fn polars__arr_any(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let v = arr.iter().any(|&x| x != 0.0);
        Ok(json!({"bool": v}))
    })
}

/// All elements non-zero?
#[no_mangle]
pub extern "C" fn polars__arr_all(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let v = arr.iter().all(|&x| x != 0.0);
        Ok(json!({"bool": v}))
    })
}

/// Count of non-zero elements.
#[no_mangle]
pub extern "C" fn polars__arr_count_nonzero(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let n = arr.iter().filter(|&&x| x != 0.0).count();
        Ok(json!({"count": n}))
    })
}

/// Cumulative sum along flat order (output: 1-D array of same total length).
#[no_mangle]
pub extern "C" fn polars__arr_cumsum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let mut acc = 0.0;
        let data: Vec<f64> = arr
            .iter()
            .map(|&x| {
                acc += x;
                acc
            })
            .collect();
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("cumsum shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Sort (ascending). 1-D only this slice.
#[no_mangle]
pub extern "C" fn polars__arr_sort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("sort: only 1-D arrays supported in this slice");
        }
        let mut data: Vec<f64> = arr.iter().copied().collect();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("sort shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Indices that would sort the array (1-D only).
#[no_mangle]
pub extern "C" fn polars__arr_argsort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("argsort: only 1-D arrays supported in this slice");
        }
        let data: Vec<f64> = arr.iter().copied().collect();
        let mut idx: Vec<usize> = (0..data.len()).collect();
        idx.sort_by(|&a, &b| {
            data[a]
                .partial_cmp(&data[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let indices: Vec<Value> = idx.iter().map(|&i| json!(i)).collect();
        Ok(json!({"indices": indices}))
    })
}

/// Clip every element to `[lo, hi]`.
#[no_mangle]
pub extern "C" fn polars__arr_clip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let lo = args.get("lower").and_then(|v| v.as_f64());
        let hi = args.get("upper").and_then(|v| v.as_f64());
        if lo.is_none() && hi.is_none() {
            bail!("must provide at least one of `lower`, `upper`");
        }
        let out = arr.mapv(|x| {
            let mut v = x;
            if let Some(l) = lo {
                if v < l {
                    v = l;
                }
            }
            if let Some(h) = hi {
                if v > h {
                    v = h;
                }
            }
            v
        });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// First-order difference (1-D, output is shorter by 1).
#[no_mangle]
pub extern "C" fn polars__arr_diff(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("diff: only 1-D arrays supported in this slice");
        }
        let data: Vec<f64> = arr.iter().copied().collect();
        if data.len() < 2 {
            let empty = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
            return Ok(json!({"array": array_to_value(&empty)}));
        }
        let diffs: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
        let n = diffs.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), diffs).context("diff shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Sum along an axis (output shape: shape without `axis`).
#[no_mangle]
pub extern "C" fn polars__arr_sum_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis {} out of range for shape {:?}", axis, arr.shape());
        }
        let summed = arr.sum_axis(Axis(axis));
        Ok(json!({"array": array_to_value(&summed)}))
    })
}

/// Mean along an axis.
#[no_mangle]
pub extern "C" fn polars__arr_mean_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis {} out of range", axis);
        }
        let meaned = arr
            .mean_axis(Axis(axis))
            .ok_or_else(|| anyhow!("mean_axis returned None"))?;
        Ok(json!({"array": array_to_value(&meaned)}))
    })
}

/// Product of all elements.
#[no_mangle]
pub extern "C" fn polars__arr_prod(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let p = arr.iter().product::<f64>();
        Ok(json!({"scalar": scalar_to_value(p)}))
    })
}

/// Variance (population, ddof=0).
#[no_mangle]
pub extern "C" fn polars__arr_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let n = arr.len() as f64;
        if n == 0.0 {
            bail!("var of empty array");
        }
        let mean = arr.iter().sum::<f64>() / n;
        let v = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Ok(json!({"scalar": scalar_to_value(v)}))
    })
}

/// Standard deviation (population, ddof=0).
#[no_mangle]
pub extern "C" fn polars__arr_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let n = arr.len() as f64;
        if n == 0.0 {
            bail!("std of empty array");
        }
        let mean = arr.iter().sum::<f64>() / n;
        let v = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Ok(json!({"scalar": scalar_to_value(v.sqrt())}))
    })
}

/// Median (1-D only).
#[no_mangle]
pub extern "C" fn polars__arr_median(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("median: only 1-D arrays supported");
        }
        let mut data: Vec<f64> = arr.iter().copied().collect();
        if data.is_empty() {
            bail!("median of empty array");
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = data.len();
        let m = if n % 2 == 1 {
            data[n / 2]
        } else {
            (data[n / 2 - 1] + data[n / 2]) / 2.0
        };
        Ok(json!({"scalar": scalar_to_value(m)}))
    })
}

/// Quantile via linear interpolation. q ∈ [0, 1]. 1-D only.
#[no_mangle]
pub extern "C" fn polars__arr_quantile(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("quantile: only 1-D arrays supported");
        }
        let q = args
            .get("q")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `q`"))?;
        if !(0.0..=1.0).contains(&q) {
            bail!("`q` must be in [0, 1]");
        }
        let mut data: Vec<f64> = arr.iter().copied().collect();
        if data.is_empty() {
            bail!("quantile of empty array");
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = data.len();
        let pos = q * (n as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        let val = if lo == hi {
            data[lo]
        } else {
            data[lo] + frac * (data[hi] - data[lo])
        };
        Ok(json!({"scalar": scalar_to_value(val)}))
    })
}

/// Max along an axis.
#[no_mangle]
pub extern "C" fn polars__arr_max_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis {} out of range", axis);
        }
        let out = arr.map_axis(Axis(axis), |row| {
            row.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b))
        });
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Min along an axis.
#[no_mangle]
pub extern "C" fn polars__arr_min_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let axis = args
            .get("axis")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `axis`"))? as usize;
        if axis >= arr.shape().len() {
            bail!("axis {} out of range", axis);
        }
        let out = arr.map_axis(Axis(axis), |row| {
            row.iter().copied().fold(f64::INFINITY, |a, b| a.min(b))
        });
        Ok(json!({"array": array_to_value(&out)}))
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

/// Matrix multiplication `A @ B`.
///
/// Args:   `{a: <A>, b: <B>}` both 2-D
/// Result: `{matrix: ...}`
#[no_mangle]
pub extern "C" fn polars__linalg_matmul(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_matrix(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_matrix(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        if a.ncols() != b.nrows() {
            bail!("matmul: A.cols ({}) ≠ B.rows ({})", a.ncols(), b.nrows());
        }
        let c = a * b;
        Ok(json!({"matrix": matrix_to_value(&c)}))
    })
}

/// Singular values (1-D array, descending).
#[no_mangle]
pub extern "C" fn polars__linalg_singular_values(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        let svd = m.svd(false, false);
        let sv = svd.singular_values;
        let data: Vec<f64> = sv.iter().copied().collect();
        let n = data.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("svd shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

// ── P5c: more random distributions ─────────────────────────────────────────

/// `np.random.exponential(scale, n)` — n samples from Exp(λ=1/scale).
///
/// Args:   `{n: u64, scale?: f64 (default 1), seed?: u64}`
#[no_mangle]
pub extern "C" fn polars__rand_exponential(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let scale = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if scale <= 0.0 {
            bail!("`scale` must be > 0");
        }
        let dist = rand_distr::Exp::new(1.0 / scale).context("Exp::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("exp shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.beta(alpha, beta, n)`.
#[no_mangle]
pub extern "C" fn polars__rand_beta(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let a = args
            .get("alpha")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `alpha`"))?;
        let b = args
            .get("beta")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `beta`"))?;
        if a <= 0.0 || b <= 0.0 {
            bail!("alpha and beta must be > 0");
        }
        let dist = rand_distr::Beta::new(a, b).context("Beta::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("beta shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.gamma(shape, scale, n)`.
#[no_mangle]
pub extern "C" fn polars__rand_gamma(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let shape = args
            .get("shape")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `shape`"))?;
        let scale = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if shape <= 0.0 || scale <= 0.0 {
            bail!("shape and scale must be > 0");
        }
        let dist = rand_distr::Gamma::new(shape, scale).context("Gamma::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("gamma shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

// ── P4d: more ndarray (stack/tile/repeat/unique) ───────────────────────────

/// Horizontal stack — concatenate along axis 1 (or last axis for 1-D).
#[no_mangle]
pub extern "C" fn polars__arr_hstack(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let axis = if a.shape().len() == 1 { 0 } else { 1 };
        let result = ndarray::concatenate(Axis(axis), &[a.view(), b.view()]).context("hstack")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// Vertical stack — concatenate along axis 0.
#[no_mangle]
pub extern "C" fn polars__arr_vstack(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let result = ndarray::concatenate(Axis(0), &[a.view(), b.view()]).context("vstack")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// Tile (repeat the whole array `n` times along axis 0).
#[no_mangle]
pub extern "C" fn polars__arr_tile(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        if n == 0 {
            bail!("`n` must be ≥ 1");
        }
        let views: Vec<_> = std::iter::repeat_n(arr.view(), n).collect();
        let result = ndarray::concatenate(Axis(0), &views).context("tile")?;
        Ok(json!({"array": array_to_value(&result)}))
    })
}

/// Unique elements (1-D, sorted ascending).
#[no_mangle]
pub extern "C" fn polars__arr_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let mut data: Vec<f64> = arr.iter().copied().collect();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        data.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("unique shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

// ── P5b: more linalg (lu / qr / matrix_power) ──────────────────────────────

/// LU decomposition. Returns `{l: <L>, u: <U>}` (no permutation in this slice).
#[no_mangle]
pub extern "C" fn polars__linalg_lu(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        let lu = m.lu();
        let l = lu.l();
        let u = lu.u();
        Ok(json!({"l": matrix_to_value(&l), "u": matrix_to_value(&u)}))
    })
}

/// QR decomposition. Returns `{q, r}`.
#[no_mangle]
pub extern "C" fn polars__linalg_qr(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        let qr = m.qr();
        let q = qr.q();
        let r = qr.r();
        Ok(json!({"q": matrix_to_value(&q), "r": matrix_to_value(&r)}))
    })
}

/// Matrix power `A^k` (k ≥ 0).
#[no_mangle]
pub extern "C" fn polars__linalg_matrix_power(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        if !m.is_square() {
            bail!("matrix_power: matrix must be square");
        }
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `k`"))? as u32;
        let n = m.nrows();
        let mut result = DMatrix::<f64>::identity(n, n);
        for _ in 0..k {
            result = &result * &m;
        }
        Ok(json!({"matrix": matrix_to_value(&result)}))
    })
}

// ── P5c: more random / fft ─────────────────────────────────────────────────

/// Random permutation of `0..n` (1-D u64 array as f64).
#[no_mangle]
pub extern "C" fn polars__rand_permutation(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        use rand::seq::SliceRandom;
        let mut rng = rng_for(&args);
        let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        data.shuffle(&mut rng);
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("permutation shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.randint(low, high, n)` — n uniform ints in `[low, high)`.
#[no_mangle]
pub extern "C" fn polars__rand_randint(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let low = args
            .get("low")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing argument `low`"))?;
        let high = args
            .get("high")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing argument `high`"))?;
        if high <= low {
            bail!("`high` must be > `low`");
        }
        use rand::Rng;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| rng.gen_range(low..high) as f64).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("randint shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.fft.fftfreq(n, d=1.0)` — DFT sample frequencies for a length-n signal.
#[no_mangle]
pub extern "C" fn polars__fft_fftfreq(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let d = args.get("d").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if n == 0 {
            bail!("`n` must be ≥ 1");
        }
        let n_f = n as f64;
        let half = n.div_ceil(2);
        let mut data = Vec::with_capacity(n);
        for k in 0..half {
            data.push(k as f64 / (n_f * d));
        }
        for k in half..n {
            data.push((k as f64 - n_f) / (n_f * d));
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("fftfreq shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

// ── P5d: polynomial ────────────────────────────────────────────────────────

// ── P5e: more random / linalg ──────────────────────────────────────────────

/// `np.random.poisson(lambda, n)` — n Poisson samples (integer values as f64).
#[no_mangle]
pub extern "C" fn polars__rand_poisson(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let lambda = args.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if lambda <= 0.0 {
            bail!("`lambda` must be > 0");
        }
        let dist = rand_distr::Poisson::new(lambda).context("Poisson::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("poisson shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// `np.random.binomial(n_trials, p, n)` — n Binomial samples.
#[no_mangle]
pub extern "C" fn polars__rand_binomial(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let trials = args
            .get("n_trials")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n_trials`"))?;
        let p = args
            .get("p")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `p`"))?;
        if !(0.0..=1.0).contains(&p) {
            bail!("`p` must be in [0, 1]");
        }
        let dist = rand_distr::Binomial::new(trials, p).context("Binomial::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("binomial shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Random log-normal samples.
#[no_mangle]
pub extern "C" fn polars__rand_lognormal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if sigma <= 0.0 {
            bail!("`sigma` must be > 0");
        }
        let dist = rand_distr::LogNormal::new(mu, sigma).context("LogNormal::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("lognormal shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// In-place shuffle of an existing array (1-D); returns shuffled copy.
#[no_mangle]
pub extern "C" fn polars__rand_shuffle(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        if arr.shape().len() != 1 {
            bail!("shuffle: only 1-D arrays supported");
        }
        let mut data: Vec<f64> = arr.iter().copied().collect();
        use rand::seq::SliceRandom;
        let mut rng = rng_for(&args);
        data.shuffle(&mut rng);
        let n = data.len();
        let out = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("shuffle shape")?;
        Ok(json!({"array": array_to_value(&out)}))
    })
}

/// Diagonal vector of a square matrix.
#[no_mangle]
pub extern "C" fn polars__linalg_diag(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        let n = m.nrows().min(m.ncols());
        let data: Vec<f64> = (0..n).map(|i| m[(i, i)]).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("diag shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Cholesky decomposition (lower-triangular L such that L * L^T = A).
/// Requires symmetric positive-definite A.
#[no_mangle]
pub extern "C" fn polars__linalg_cholesky(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        let ch = m
            .cholesky()
            .ok_or_else(|| anyhow!("Cholesky failed (matrix not symmetric PD)"))?;
        let l = ch.l();
        Ok(json!({"matrix": matrix_to_value(&l)}))
    })
}

/// Pseudo-inverse (Moore-Penrose) via SVD.
#[no_mangle]
pub extern "C" fn polars__linalg_pinv(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = parse_matrix(
            args.get("matrix")
                .ok_or_else(|| anyhow!("missing argument `matrix`"))?,
        )?;
        let svd = m.svd(true, true);
        let pinv = svd
            .pseudo_inverse(1e-10)
            .map_err(|e| anyhow!("pinv: {e}"))?;
        Ok(json!({"matrix": matrix_to_value(&pinv)}))
    })
}

/// Polynomial derivative: returns coefficients of the derivative polynomial.
///
/// Args:   `{coefficients: [c0, c1, ...]}`
/// Result: `{coefficients: [c1, 2*c2, 3*c3, ...]}`
#[no_mangle]
pub extern "C" fn polars__poly_polyder(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let coeffs_arr = args
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `coefficients`"))?;
        let coeffs: Vec<f64> = coeffs_arr
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        let der: Vec<Value> = coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| scalar_to_value(c * i as f64))
            .collect();
        Ok(json!({"coefficients": der}))
    })
}

/// Polynomial integral (returns coefficients; constant of integration defaults to 0).
///
/// Args:   `{coefficients, c0?: f64 (default 0)}`
/// Result: `{coefficients}`
#[no_mangle]
pub extern "C" fn polars__poly_polyint(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let coeffs_arr = args
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `coefficients`"))?;
        let c0 = args.get("c0").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let coeffs: Vec<f64> = coeffs_arr
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        let mut out = Vec::with_capacity(coeffs.len() + 1);
        out.push(c0);
        for (i, &c) in coeffs.iter().enumerate() {
            out.push(c / (i + 1) as f64);
        }
        let result: Vec<Value> = out.iter().map(|&x| scalar_to_value(x)).collect();
        Ok(json!({"coefficients": result}))
    })
}

/// Outer product of two 1-D vectors → 2-D matrix `len(a) × len(b)`.
#[no_mangle]
pub extern "C" fn polars__linalg_outer(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 1 || b.shape().len() != 1 {
            bail!("outer: both inputs must be 1-D");
        }
        let m = a.len();
        let n = b.len();
        let mut data = Vec::with_capacity(m * n);
        for &x in a.iter() {
            for &y in b.iter() {
                data.push(x * y);
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[m, n]), data).context("outer shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Evaluate polynomial: `c0 + c1*x + c2*x^2 + ...` at each point in `x_array`.
///
/// Args:   `{coefficients: [c0, c1, ...], x: <array>}`
#[no_mangle]
pub extern "C" fn polars__poly_polyval(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let coeffs_arr = args
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `coefficients` (array)"))?;
        let coeffs: Vec<f64> = coeffs_arr
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        let x = get_array(&args, "x")?;
        // Horner's method.
        let out: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let mut acc = 0.0;
                for &c in coeffs.iter().rev() {
                    acc = acc * xi + c;
                }
                acc
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(x.shape()), out).context("polyval shape")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Sample without replacement: `k` distinct values from `array`.
///
/// Args:   `{array, k: u64, seed?: u64}`
#[no_mangle]
pub extern "C" fn polars__rand_choice(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_array(&args, "array")?;
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `k`"))? as usize;
        let data: Vec<f64> = arr.iter().copied().collect();
        if k > data.len() {
            bail!("k ({}) must be ≤ array length ({})", k, data.len());
        }
        use rand::seq::SliceRandom;
        let mut rng = rng_for(&args);
        let picked: Vec<f64> = data.choose_multiple(&mut rng, k).copied().collect();
        let arr_out = ArrayD::from_shape_vec(IxDyn(&[k]), picked).context("choice shape")?;
        Ok(json!({"array": array_to_value(&arr_out)}))
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

#[cfg(test)]
mod tests {
    //! Hand-crafted regression tests targeting specific arithmetic bug
    //! classes in the numpy surface — modular roll, last-chunk-leftover
    //! split contract, binary-search insertion-point edge cases, and arange
    //! step-direction guard. Each test is an adversarial case that a naive
    //! reimplementation (off-by-one, wrong shift sign, premature loop exit,
    //! missing zero-step rejection) would fail loudly.
    use serde_json::json;
    use std::f64;

    use crate::ffi_test::call;

    fn flat_data(v: &serde_json::Value) -> Vec<f64> {
        v["array"]["data"]
            .as_array()
            .expect("array.data")
            .iter()
            .map(|x| x.as_f64().expect("f64"))
            .collect()
    }

    fn flat_shape(v: &serde_json::Value) -> Vec<usize> {
        v["array"]["shape"]
            .as_array()
            .expect("array.shape")
            .iter()
            .map(|x| x.as_u64().expect("u64") as usize)
            .collect()
    }

    /// `arr_roll` must implement true modular shift: negative shifts rotate
    /// left, positive shifts rotate right, and |shift| > len must wrap. A
    /// naive `shift as usize` (without first folding into `[0, n)`) panics
    /// or silently corrupts on negative input. A modulo-only impl missing
    /// the `+ n` normalization step on negative `shift % n` returns garbage
    /// for shift=-1. This fixture pins all four edge cases at once.
    #[test]
    fn arr_roll_handles_negative_oversized_and_zero_shifts() {
        let arr = json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0], "shape": [5]});

        // shift = +1 → right-rotate by one
        let v = call(super::polars__arr_roll, json!({"array": arr, "shift": 1}));
        assert!(v["error"].is_null(), "shift=+1 errored: {v}");
        assert_eq!(flat_data(&v), vec![5.0, 1.0, 2.0, 3.0, 4.0]);

        // shift = -1 → left-rotate by one (equivalent to +4 on len-5)
        let v = call(super::polars__arr_roll, json!({"array": arr, "shift": -1}));
        assert!(v["error"].is_null(), "shift=-1 errored: {v}");
        assert_eq!(
            flat_data(&v),
            vec![2.0, 3.0, 4.0, 5.0, 1.0],
            "shift=-1 must equal shift=+4 (modular wrap)"
        );

        // shift = +6 → right-rotate by (6 mod 5) = 1
        let v = call(super::polars__arr_roll, json!({"array": arr, "shift": 6}));
        assert!(v["error"].is_null(), "shift=+6 errored: {v}");
        assert_eq!(
            flat_data(&v),
            vec![5.0, 1.0, 2.0, 3.0, 4.0],
            "shift=+6 on len-5 must equal shift=+1"
        );

        // shift = -10 → left-rotate by (10 mod 5) = 0 → identity
        let v = call(super::polars__arr_roll, json!({"array": arr, "shift": -10}));
        assert!(v["error"].is_null(), "shift=-10 errored: {v}");
        assert_eq!(
            flat_data(&v),
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            "shift=-10 on len-5 must equal identity"
        );

        // shift = 0 → identity
        let v = call(super::polars__arr_roll, json!({"array": arr, "shift": 0}));
        assert!(v["error"].is_null(), "shift=0 errored: {v}");
        assert_eq!(flat_data(&v), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// `arr_split` docstring promises "leftover elements go to the last
    /// chunk" — i.e., the LAST chunk swallows the remainder while earlier
    /// chunks are exactly `total / n_parts`. The competing numpy
    /// `np.array_split` distributes the remainder to the FIRST chunks. A
    /// refactor toward the numpy convention would silently flip
    /// downstream chunk indexing for every caller that relies on the
    /// documented contract. This pins both directions at once with an
    /// adversarial setup that distinguishes them.
    #[test]
    fn arr_split_leftover_goes_to_the_last_chunk_not_the_first() {
        // 11 / 3 = 3 remainder 2 → docstring contract: [3, 3, 5]
        //                       → numpy contract:     [4, 4, 3]
        let v = call(
            super::polars__arr_split,
            json!({
                "array": {"data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], "shape": [11]},
                "n_parts": 3,
            }),
        );
        assert!(v["error"].is_null(), "split errored: {v}");
        let chunks = v["chunks"].as_array().expect("chunks array");
        assert_eq!(chunks.len(), 3, "got {} chunks", chunks.len());
        let chunk_data: Vec<Vec<f64>> = chunks
            .iter()
            .map(|c| {
                c["data"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(chunk_data[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(chunk_data[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(
            chunk_data[2],
            vec![7.0, 8.0, 9.0, 10.0, 11.0],
            "leftover (2 extra elements) must land in the LAST chunk per docstring"
        );

        // total < n_parts: base=0 so every chunk except the last is empty,
        // last gets everything. Pins that no off-by-one walks `start`
        // past `total` mid-loop (would underflow `data[start..end]`).
        let v = call(
            super::polars__arr_split,
            json!({
                "array": {"data": [42.0, 43.0], "shape": [2]},
                "n_parts": 5,
            }),
        );
        assert!(v["error"].is_null(), "split (total<n_parts) errored: {v}");
        let chunks = v["chunks"].as_array().expect("chunks array");
        assert_eq!(chunks.len(), 5);
        for (i, chunk) in chunks.iter().enumerate().take(4) {
            let d: Vec<f64> = chunk["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            assert!(d.is_empty(), "chunk {i} must be empty, got {d:?}");
        }
        let last: Vec<f64> = chunks[4]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(
            last,
            vec![42.0, 43.0],
            "last chunk must absorb every element when total < n_parts"
        );
    }

    /// `arr_searchsorted` must implement numpy's `side='left'` insertion
    /// point: leftmost index `i` such that `a[i] >= v`. The adversarial
    /// cases are (a) empty array → 0, (b) v less than every element → 0,
    /// (c) v greater than every element → len, (d) v equal to a duplicate
    /// run → first matching index (not last, not middle). A binary-search
    /// off-by-one that uses `<=` instead of `<` would push duplicates to
    /// the right (numpy's `side='right'` behavior); a naive linear
    /// `position(|x| x >= v)` would still pass the first three but a
    /// missing branch on duplicates would silently shift indices.
    #[test]
    fn arr_searchsorted_left_insertion_point_pins_duplicate_and_boundary_cases() {
        // Empty array → 0.
        let v = call(
            super::polars__arr_searchsorted,
            json!({"array": {"data": [], "shape": [0]}, "v": 3.0}),
        );
        assert!(v["error"].is_null(), "empty errored: {v}");
        assert_eq!(v["index"].as_u64(), Some(0));

        // v less than every element → 0.
        let v = call(
            super::polars__arr_searchsorted,
            json!({"array": {"data": [10.0, 20.0, 30.0], "shape": [3]}, "v": 5.0}),
        );
        assert_eq!(v["index"].as_u64(), Some(0));

        // v greater than every element → len.
        let v = call(
            super::polars__arr_searchsorted,
            json!({"array": {"data": [10.0, 20.0, 30.0], "shape": [3]}, "v": 99.0}),
        );
        assert_eq!(v["index"].as_u64(), Some(3));

        // v exactly equals a duplicate run → leftmost matching position
        // (numpy `side='left'`). For [1, 2, 2, 2, 3] and v=2, answer is 1.
        let v = call(
            super::polars__arr_searchsorted,
            json!({"array": {"data": [1.0, 2.0, 2.0, 2.0, 3.0], "shape": [5]}, "v": 2.0}),
        );
        assert_eq!(
            v["index"].as_u64(),
            Some(1),
            "side='left' must return the first matching index, not the last"
        );

        // v equals last element exactly → len-1, not len. Catches a
        // `data[mid] <= v ⇒ lo = mid + 1` bug that would push past the
        // tail for equal values.
        let v = call(
            super::polars__arr_searchsorted,
            json!({"array": {"data": [10.0, 20.0, 30.0], "shape": [3]}, "v": 30.0}),
        );
        assert_eq!(
            v["index"].as_u64(),
            Some(2),
            "v equal to last element must give len-1 under side='left'"
        );
    }

    /// `arr_arange` must (a) reject step=0 with a clean error rather than
    /// looping forever, (b) honor the sign of `step` for the loop guard
    /// (positive step uses `<`, negative step uses `>`), and (c) return
    /// an empty array when the direction implies no progress (e.g.
    /// start=5, stop=0, step=+1). A copy-paste bug that uses the same
    /// `<` for both branches would loop forever or fail to produce
    /// reversed sequences. Default-`start` (omitted) must equal 0 so
    /// `arange(stop=3)` returns `[0,1,2]` per numpy convention — a
    /// regression dropping the `.unwrap_or(0.0)` default would surface
    /// here.
    #[test]
    fn arr_arange_step_direction_zero_rejection_and_default_start() {
        // step=0 → clean error envelope.
        let v = call(
            super::polars__arr_arange,
            json!({"start": 0.0, "stop": 10.0, "step": 0.0}),
        );
        assert!(
            v["error"]
                .as_str()
                .map(|s| s.contains("step"))
                .unwrap_or(false),
            "expected step=0 to bail with `step` in error, got {v}"
        );

        // Negative step descending from 5 to 0.
        let v = call(
            super::polars__arr_arange,
            json!({"start": 5.0, "stop": 0.0, "step": -1.0}),
        );
        assert!(v["error"].is_null(), "negative step errored: {v}");
        assert_eq!(
            flat_data(&v),
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            "negative step must descend and exclude stop"
        );
        assert_eq!(flat_shape(&v), vec![5]);

        // Positive step with start > stop → empty (no progress).
        // A bug that flipped the `>` comparator in the descend branch
        // would also flip the ascend branch and infinite-loop here.
        let v = call(
            super::polars__arr_arange,
            json!({"start": 5.0, "stop": 0.0, "step": 1.0}),
        );
        assert!(v["error"].is_null(), "start>stop +step errored: {v}");
        assert_eq!(
            flat_data(&v),
            Vec::<f64>::new(),
            "positive step with start>stop must yield empty array"
        );
        assert_eq!(flat_shape(&v), vec![0]);

        // Default start=0 when `start` is omitted (numpy convention).
        let v = call(super::polars__arr_arange, json!({"stop": 3.0, "step": 1.0}));
        assert!(v["error"].is_null(), "omitted start errored: {v}");
        assert_eq!(
            flat_data(&v),
            vec![0.0, 1.0, 2.0],
            "omitted `start` must default to 0.0"
        );

        // Negative step with start < stop → empty. Symmetric to the
        // positive-step start>stop case above. Catches a single-branch
        // refactor that drops the `if step > 0.0` dispatch.
        let v = call(
            super::polars__arr_arange,
            json!({"start": 0.0, "stop": 5.0, "step": -1.0}),
        );
        assert!(v["error"].is_null(), "neg step start<stop errored: {v}");
        assert_eq!(
            flat_data(&v),
            Vec::<f64>::new(),
            "negative step with start<stop must yield empty array"
        );
    }
}
