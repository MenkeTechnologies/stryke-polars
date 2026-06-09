//! src/more_nd.rs — additional numpy surface, expanding nd.rs with more
//! ufuncs, linalg helpers, random distributions, FFT variants, and
//! polynomial families.
//!
//! Wire format is the same as `nd.rs`: `{array: {data: [...], shape: [...]}}`
//! for arrays, raw scalars for scalars.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{ArrayD, IxDyn};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{
    Beta, Cauchy, ChiSquared, Distribution, Exp, FisherF, Gamma, Pareto, StudentT, Triangular,
    Weibull,
};
use serde_json::{json, Value};

use crate::ffi_call;

// ── helpers (duplicated from nd.rs to keep modules independent) ─────────────

fn parse_array(v: &Value) -> Result<ArrayD<f64>> {
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("array `data` missing"))?;
    let shape = v
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or_else(|| anyhow!("array `shape` missing"))?;
    let flat: Vec<f64> = data
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect();
    let shape_vec: Vec<usize> = shape
        .iter()
        .filter_map(|x| x.as_u64().map(|n| n as usize))
        .collect();
    ArrayD::from_shape_vec(IxDyn(&shape_vec), flat).context("shape")
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

fn get_array(args: &Value, key: &str) -> Result<ArrayD<f64>> {
    let a = args
        .get(key)
        .ok_or_else(|| anyhow!("missing argument `{key}`"))?;
    parse_array(a)
}

fn scalar(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn rng_for(args: &Value) -> ChaCha8Rng {
    match args.get("seed").and_then(|v| v.as_u64()) {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

// ── more ufuncs ────────────────────────────────────────────────────────────

macro_rules! np_unary {
    ($fn_name:ident, $f:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let a = get_array(&args, "array")?;
                let r = a.mapv(|x| ($f)(x));
                Ok(json!({"array": array_to_value(&r)}))
            })
        }
    };
}

np_unary!(polars__np_asinh, f64::asinh);
np_unary!(polars__np_acosh, f64::acosh);
np_unary!(polars__np_atanh, f64::atanh);
np_unary!(polars__np_logical_signal, |x: f64| if x != 0.0 {
    1.0
} else {
    0.0
});
np_unary!(polars__np_clip_neg, |x: f64| x.max(0.0));
np_unary!(polars__np_clip_pos, |x: f64| x.min(0.0));
np_unary!(polars__np_round_half_even, |x: f64| {
    let f = x.floor();
    let frac = x - f;
    if frac < 0.5 {
        f
    } else if frac > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
});
np_unary!(polars__np_gamma, |x: f64| {
    // Stirling's series; good enough for x > 1.
    if x <= 0.0 {
        f64::NAN
    } else {
        ((2.0 * std::f64::consts::PI / x).sqrt()) * (x / std::f64::consts::E).powf(x)
    }
});
np_unary!(polars__np_lgamma, |x: f64| {
    if x <= 0.0 {
        f64::NAN
    } else {
        ((2.0 * std::f64::consts::PI / x).sqrt()).ln() + x * (x.ln() - 1.0)
    }
});
np_unary!(polars__np_erf, |x: f64| {
    // Abramowitz & Stegun 7.1.26 approximation.
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
});
np_unary!(polars__np_erfc, |x: f64| {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    1.0 - sign * y
});
np_unary!(polars__np_sigmoid, |x: f64| 1.0 / (1.0 + (-x).exp()));
np_unary!(polars__np_relu, |x: f64| x.max(0.0));
np_unary!(polars__np_softplus, |x: f64| (1.0 + x.exp()).ln());
np_unary!(polars__np_silu, |x: f64| x / (1.0 + (-x).exp()));
np_unary!(polars__np_gelu, |x: f64| {
    let t = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3));
    0.5 * x * (1.0 + t.tanh())
});
np_unary!(polars__np_elu, |x: f64| if x >= 0.0 {
    x
} else {
    x.exp() - 1.0
});
np_unary!(polars__np_isneg, |x: f64| if x < 0.0 { 1.0 } else { 0.0 });
np_unary!(polars__np_ispos, |x: f64| if x > 0.0 { 1.0 } else { 0.0 });
np_unary!(polars__np_iszero, |x: f64| if x == 0.0 { 1.0 } else { 0.0 });
np_unary!(polars__np_fabs, f64::abs);
np_unary!(polars__np_conj, |x: f64| x);

// ── binary ufuncs ──────────────────────────────────────────────────────────

fn binary_op<F: Fn(f64, f64) -> f64>(args: &Value, f: F) -> Result<Value> {
    let a = get_array(args, "a")?;
    let b = get_array(args, "b")?;
    if a.shape() != b.shape() {
        bail!("shape mismatch");
    }
    let data: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    let r = ArrayD::from_shape_vec(IxDyn(a.shape()), data).context("shape")?;
    Ok(json!({"array": array_to_value(&r)}))
}

#[no_mangle]
pub extern "C" fn polars__np_logaddexp_pair(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| (x.exp() + y.exp()).ln())
    })
}

#[no_mangle]
pub extern "C" fn polars__np_logaddexp2_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| (x.exp2() + y.exp2()).log2())
    })
}

#[no_mangle]
pub extern "C" fn polars__np_nextafter(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| {
            if x == y {
                x
            } else if y > x {
                f64::from_bits(x.to_bits() + 1)
            } else {
                f64::from_bits(x.to_bits() - 1)
            }
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__np_ldexp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x * 2f64.powf(y)))
}

#[no_mangle]
pub extern "C" fn polars__np_divmod_quot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| (x / y).floor()))
}

#[no_mangle]
pub extern "C" fn polars__np_divmod_rem(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| x - y * (x / y).floor())
    })
}

#[no_mangle]
pub extern "C" fn polars__np_true_divide(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x / y))
}

#[no_mangle]
pub extern "C" fn polars__np_subtract_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x - y))
}

#[no_mangle]
pub extern "C" fn polars__np_multiply_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x * y))
}

#[no_mangle]
pub extern "C" fn polars__np_divide_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x / y))
}

#[no_mangle]
pub extern "C" fn polars__np_add_pair(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| binary_op(&args, |x, y| x + y))
}

#[no_mangle]
pub extern "C" fn polars__np_greater(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| if x > y { 1.0 } else { 0.0 })
    })
}

#[no_mangle]
pub extern "C" fn polars__np_greater_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| if x >= y { 1.0 } else { 0.0 })
    })
}

#[no_mangle]
pub extern "C" fn polars__np_less(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| if x < y { 1.0 } else { 0.0 })
    })
}

#[no_mangle]
pub extern "C" fn polars__np_less_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| if x <= y { 1.0 } else { 0.0 })
    })
}

#[no_mangle]
pub extern "C" fn polars__np_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| if x == y { 1.0 } else { 0.0 })
    })
}

#[no_mangle]
pub extern "C" fn polars__np_not_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| if x != y { 1.0 } else { 0.0 })
    })
}

#[no_mangle]
pub extern "C" fn polars__np_bitwise_and(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| ((x as i64) & (y as i64)) as f64)
    })
}

#[no_mangle]
pub extern "C" fn polars__np_bitwise_or(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| ((x as i64) | (y as i64)) as f64)
    })
}

#[no_mangle]
pub extern "C" fn polars__np_bitwise_xor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| ((x as i64) ^ (y as i64)) as f64)
    })
}

#[no_mangle]
pub extern "C" fn polars__np_left_shift(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| ((x as i64) << (y as i64)) as f64)
    })
}

#[no_mangle]
pub extern "C" fn polars__np_right_shift(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| ((x as i64) >> (y as i64)) as f64)
    })
}

#[no_mangle]
pub extern "C" fn polars__np_gcd(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| {
            let mut a = (x.abs() as i64).max(1);
            let mut b = (y.abs() as i64).max(1);
            while b != 0 {
                let t = b;
                b = a % b;
                a = t;
            }
            a as f64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__np_lcm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        binary_op(&args, |x, y| {
            let a = (x.abs() as i64).max(1);
            let b = (y.abs() as i64).max(1);
            let mut g = a;
            let mut bb = b;
            while bb != 0 {
                let t = bb;
                bb = g % bb;
                g = t;
            }
            (a * b / g) as f64
        })
    })
}

// ── more arr_ construction ─────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__arr_full_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let shape: Vec<usize> = args
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `shape`"))?
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect();
        let v = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let arr = ArrayD::<f64>::from_elem(IxDyn(&shape), v);
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_empty(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let shape: Vec<usize> = args
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `shape`"))?
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect();
        let arr = ArrayD::<f64>::zeros(IxDyn(&shape));
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_eye_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let m = args
            .get("m")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or(n);
        let mut data = vec![0.0; n * m];
        for i in 0..n.min(m) {
            data[i * m + i] = 1.0;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, m]), data).context("eye")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_identity(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, n]), data).context("identity")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_zeros_like_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let arr = ArrayD::<f64>::zeros(IxDyn(a.shape()));
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_ones_like_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let arr = ArrayD::<f64>::ones(IxDyn(a.shape()));
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_full_like_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let arr = ArrayD::<f64>::from_elem(IxDyn(a.shape()), v);
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_diagflat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let mut data = vec![0.0; n * n];
        for (i, x) in v.iter().enumerate() {
            data[i * n + i] = *x;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, n]), data).context("diagflat")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_tri(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                data[i * n + j] = 1.0;
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, n]), data).context("tri")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_vander(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let n = a.len();
        let n_cols = args
            .get("n")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or(n);
        let v: Vec<f64> = a.iter().copied().collect();
        let mut data = Vec::with_capacity(n * n_cols);
        for &x in &v {
            for j in 0..n_cols {
                data.push(x.powi((n_cols - 1 - j) as i32));
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, n_cols]), data).context("vander")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_meshgrid_x(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_array(&args, "x")?;
        let y = get_array(&args, "y")?;
        let nx = x.len();
        let ny = y.len();
        let xs: Vec<f64> = x.iter().copied().collect();
        let mut data = Vec::with_capacity(nx * ny);
        for _ in 0..ny {
            data.extend(&xs);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[ny, nx]), data).context("meshgrid_x")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_meshgrid_y(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_array(&args, "x")?;
        let y = get_array(&args, "y")?;
        let nx = x.len();
        let ys: Vec<f64> = y.iter().copied().collect();
        let mut data = Vec::with_capacity(nx * ys.len());
        for yi in &ys {
            for _ in 0..nx {
                data.push(*yi);
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[ys.len(), nx]), data).context("meshgrid_y")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

// ── more linalg ─────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__linalg_vdot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let r: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(json!({"vdot": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__linalg_inner(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let r: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(json!({"inner": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__linalg_trace_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("trace requires 2-D matrix");
        }
        let (n, m) = (a.shape()[0], a.shape()[1]);
        let mut sum = 0.0;
        for i in 0..n.min(m) {
            sum += a[[i, i].as_slice()];
        }
        Ok(json!({"trace": scalar(sum)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__linalg_matrix_rank(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("matrix_rank requires 2-D matrix");
        }
        let m = nalgebra::DMatrix::from_iterator(a.shape()[0], a.shape()[1], a.iter().copied());
        let svd = m.svd(false, false);
        let tol = 1e-10;
        let rank = svd.singular_values.iter().filter(|s| **s > tol).count();
        Ok(json!({"rank": rank}))
    })
}

#[no_mangle]
pub extern "C" fn polars__linalg_lstsq_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 2 {
            bail!("a must be 2-D");
        }
        let am = nalgebra::DMatrix::from_iterator(a.shape()[0], a.shape()[1], a.iter().copied());
        let bm = nalgebra::DVector::from_iterator(b.len(), b.iter().copied());
        let svd = am.svd(true, true);
        let x = svd.solve(&bm, 1e-12).map_err(|e| anyhow!("lstsq: {e}"))?;
        let data: Vec<f64> = x.iter().copied().collect();
        let n = data.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("lstsq")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__linalg_tensordot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 2 || b.shape().len() != 2 {
            bail!("tensordot: 2-D matrices only in this slice");
        }
        if a.shape()[1] != b.shape()[0] {
            bail!("shape mismatch");
        }
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        let av: Vec<f64> = a.iter().copied().collect();
        let bv: Vec<f64> = b.iter().copied().collect();
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += av[i * k + p] * bv[p * n + j];
                }
                out[i * n + j] = s;
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[m, n]), out).context("tensordot")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

// ── more random distributions ──────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__rand_beta_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let a = args
            .get("alpha")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `alpha`"))?;
        let b = args
            .get("beta")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `beta`"))?;
        let dist = Beta::new(a, b).context("Beta::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("beta")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_cauchy_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let scale = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let dist = Cauchy::new(mu, scale).context("Cauchy::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("cauchy")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_chisquared(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let k = args
            .get("k")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        let dist = ChiSquared::new(k).context("ChiSquared::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("chisquared")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_exponential_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let lambda = args.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let dist = Exp::new(lambda).context("Exp::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("exp")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_f_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let d1 = args
            .get("d1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `d1`"))?;
        let d2 = args
            .get("d2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `d2`"))?;
        let dist = FisherF::new(d1, d2).context("FisherF::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("f")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_gamma_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let shape = args
            .get("shape_p")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `shape_p`"))?;
        let scale = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let dist = Gamma::new(shape, scale).context("Gamma::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("gamma")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_pareto_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let scale = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let alpha = args
            .get("alpha")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `alpha`"))?;
        let dist = Pareto::new(scale, alpha).context("Pareto::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("pareto")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_studentt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let nu = args
            .get("nu")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `nu`"))?;
        let dist = StudentT::new(nu).context("StudentT::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("studentt")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_triangular(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let left = args
            .get("left")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `left`"))?;
        let mode = args
            .get("mode")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `mode`"))?;
        let right = args
            .get("right")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `right`"))?;
        let dist = Triangular::new(left, right, mode).context("Triangular::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("triangular")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_weibull_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let lambda = args.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let k = args
            .get("k")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        let dist = Weibull::new(lambda, k).context("Weibull::new")?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("weibull")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_zipf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let s = args
            .get("s")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `s`"))?;
        let max_val = args.get("max").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
        let mut rng = rng_for(&args);
        // Inverse transform (slow, OK for small max).
        let zk_inv: Vec<f64> = (1..=max_val).map(|k| 1.0 / (k as f64).powf(s)).collect();
        let z: f64 = zk_inv.iter().sum();
        let mut cdf = vec![0.0; max_val];
        let mut acc = 0.0;
        for (i, p) in zk_inv.iter().enumerate() {
            acc += p / z;
            cdf[i] = acc;
        }
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let r: f64 = rng.gen();
                (cdf.iter().position(|c| *c >= r).unwrap_or(max_val - 1) + 1) as f64
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("zipf")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_logistic(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let s = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                mu + s * (u / (1.0 - u)).ln()
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("logistic")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_laplace(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen::<f64>() - 0.5;
                mu - b * u.signum() * (1.0 - 2.0 * u.abs()).ln()
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("laplace")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_rayleigh(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                sigma * (-2.0 * (1.0 - u).ln()).sqrt()
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("rayleigh")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_gumbel(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let beta = args.get("beta").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let u: f64 = rng.gen();
                mu - beta * (-u.ln()).ln()
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("gumbel")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_bernoulli(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let p = args
            .get("p")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `p`"))?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n)
            .map(|_| if rng.gen::<f64>() < p { 1.0 } else { 0.0 })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("bernoulli")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_randint_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let lo = args
            .get("low")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `low`"))?;
        let hi = args
            .get("high")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `high`"))?;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| rng.gen_range(lo..hi) as f64).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("randint")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_random(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("random")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_standard_normal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let dist = rand_distr::StandardNormal;
        let mut rng = rng_for(&args);
        let data: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("standard_normal")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__rand_dirichlet_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let alpha = args
            .get("alpha")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `alpha`"))?;
        let alphas: Vec<f64> = alpha.iter().filter_map(|x| x.as_f64()).collect();
        let mut rng = rng_for(&args);
        let mut samples = vec![];
        for a in &alphas {
            let dist = Gamma::new(*a, 1.0).context("Gamma::new")?;
            samples.push(dist.sample(&mut rng));
        }
        let s: f64 = samples.iter().sum();
        let data: Vec<f64> = samples.iter().map(|x| x / s).collect();
        let n = data.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).context("dirichlet")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

// ── more FFT ────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__fft_rfft_pow(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let n = a.len();
        let mut buf: Vec<rustfft::num_complex::Complex<f64>> = a
            .iter()
            .map(|x| rustfft::num_complex::Complex::new(*x, 0.0))
            .collect();
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buf);
        let half = n / 2 + 1;
        let pow: Vec<f64> = buf[..half].iter().map(|c| c.norm_sqr()).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[half]), pow).context("rfft_pow")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__fft_fftshift_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let mid = n / 2;
        let mut out = Vec::with_capacity(n);
        out.extend_from_slice(&v[mid..]);
        out.extend_from_slice(&v[..mid]);
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("fftshift")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__fft_ifftshift_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let mid = n.div_ceil(2);
        let mut out = Vec::with_capacity(n);
        out.extend_from_slice(&v[mid..]);
        out.extend_from_slice(&v[..mid]);
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("ifftshift")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__fft_rfftfreq_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let d = args.get("d").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let half = n / 2 + 1;
        let n_f = n as f64;
        let data: Vec<f64> = (0..half).map(|k| k as f64 / (n_f * d)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[half]), data).context("rfftfreq")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__fft_fft2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("fft2: 2-D array required");
        }
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let v: Vec<f64> = a.iter().copied().collect();
        let mut buf: Vec<rustfft::num_complex::Complex<f64>> = v
            .iter()
            .map(|x| rustfft::num_complex::Complex::new(*x, 0.0))
            .collect();
        // Row-wise then column-wise.
        let mut planner = rustfft::FftPlanner::new();
        let row_fft = planner.plan_fft_forward(cols);
        for r in 0..rows {
            row_fft.process(&mut buf[r * cols..(r + 1) * cols]);
        }
        let mut col_buf = vec![rustfft::num_complex::Complex::new(0.0, 0.0); rows];
        let col_fft = planner.plan_fft_forward(rows);
        for c in 0..cols {
            for r in 0..rows {
                col_buf[r] = buf[r * cols + c];
            }
            col_fft.process(&mut col_buf);
            for r in 0..rows {
                buf[r * cols + c] = col_buf[r];
            }
        }
        let real: Vec<f64> = buf.iter().map(|c| c.re).collect();
        let imag: Vec<f64> = buf.iter().map(|c| c.im).collect();
        Ok(json!({
            "complex": {
                "real": real,
                "imag": imag,
                "shape": [rows, cols]
            }
        }))
    })
}

// ── more polynomial ─────────────────────────────────────────────────────────

fn coeff_arr(args: &Value, key: &str) -> Result<Vec<f64>> {
    let a = args
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `{key}`"))?;
    Ok(a.iter().filter_map(|x| x.as_f64()).collect())
}

#[no_mangle]
pub extern "C" fn polars__poly_polyadd_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = coeff_arr(&args, "a")?;
        let b = coeff_arr(&args, "b")?;
        let n = a.len().max(b.len());
        let mut out = vec![0.0; n];
        for (i, x) in a.iter().enumerate() {
            out[i] += x;
        }
        for (i, x) in b.iter().enumerate() {
            out[i] += x;
        }
        Ok(json!({"coefficients": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_polysub_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = coeff_arr(&args, "a")?;
        let b = coeff_arr(&args, "b")?;
        let n = a.len().max(b.len());
        let mut out = vec![0.0; n];
        for (i, x) in a.iter().enumerate() {
            out[i] += x;
        }
        for (i, x) in b.iter().enumerate() {
            out[i] -= x;
        }
        Ok(json!({"coefficients": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_polymul_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = coeff_arr(&args, "a")?;
        let b = coeff_arr(&args, "b")?;
        let n = a.len() + b.len() - 1;
        let mut out = vec![0.0; n.max(1)];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                out[i + j] += x * y;
            }
        }
        Ok(json!({"coefficients": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_chebyshev_t(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let x = get_array(&args, "x")?;
        let v: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let mut t0 = 1.0_f64;
                let mut t1 = xi;
                if n == 0 {
                    return t0;
                }
                if n == 1 {
                    return t1;
                }
                for _ in 2..=n {
                    let tn = 2.0 * xi * t1 - t0;
                    t0 = t1;
                    t1 = tn;
                }
                t1
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(x.shape()), v).context("chebyshev_t")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_chebyshev_u(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let x = get_array(&args, "x")?;
        let v: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let mut u0 = 1.0_f64;
                let mut u1 = 2.0 * xi;
                if n == 0 {
                    return u0;
                }
                if n == 1 {
                    return u1;
                }
                for _ in 2..=n {
                    let un = 2.0 * xi * u1 - u0;
                    u0 = u1;
                    u1 = un;
                }
                u1
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(x.shape()), v).context("chebyshev_u")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_legendre(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let x = get_array(&args, "x")?;
        let v: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let mut p0 = 1.0_f64;
                let mut p1 = xi;
                if n == 0 {
                    return p0;
                }
                if n == 1 {
                    return p1;
                }
                for k in 2..=n {
                    let kf = k as f64;
                    let pn = ((2.0 * kf - 1.0) * xi * p1 - (kf - 1.0) * p0) / kf;
                    p0 = p1;
                    p1 = pn;
                }
                p1
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(x.shape()), v).context("legendre")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_hermite(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let x = get_array(&args, "x")?;
        let v: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let mut h0 = 1.0_f64;
                let mut h1 = 2.0 * xi;
                if n == 0 {
                    return h0;
                }
                if n == 1 {
                    return h1;
                }
                for k in 2..=n {
                    let kf = k as f64;
                    let hn = 2.0 * xi * h1 - 2.0 * (kf - 1.0) * h0;
                    h0 = h1;
                    h1 = hn;
                }
                h1
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(x.shape()), v).context("hermite")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_laguerre(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let x = get_array(&args, "x")?;
        let v: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let mut l0 = 1.0_f64;
                let mut l1 = 1.0 - xi;
                if n == 0 {
                    return l0;
                }
                if n == 1 {
                    return l1;
                }
                for k in 2..=n {
                    let kf = k as f64;
                    let ln = ((2.0 * kf - 1.0 - xi) * l1 - (kf - 1.0) * l0) / kf;
                    l0 = l1;
                    l1 = ln;
                }
                l1
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(x.shape()), v).context("laguerre")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_polyroots_real(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let c = coeff_arr(&args, "coefficients")?;
        if c.len() < 2 {
            return Ok(json!({"roots": Vec::<f64>::new()}));
        }
        // Companion matrix approach (real eigenvalues).
        let n = c.len() - 1;
        if n == 0 {
            return Ok(json!({"roots": Vec::<f64>::new()}));
        }
        let mut m = vec![0.0; n * n];
        for i in 0..n - 1 {
            m[(i + 1) * n + i] = 1.0;
        }
        let cn = c[n];
        for i in 0..n {
            m[i * n + (n - 1)] = -c[i] / cn;
        }
        let dm = nalgebra::DMatrix::from_row_slice(n, n, &m);
        let eigs = dm.complex_eigenvalues();
        let roots: Vec<f64> = eigs
            .iter()
            .filter(|e| e.im.abs() < 1e-9)
            .map(|e| e.re)
            .collect();
        Ok(json!({"roots": roots}))
    })
}

#[no_mangle]
pub extern "C" fn polars__poly_polyfit_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_array(&args, "x")?;
        let y = get_array(&args, "y")?;
        let deg = args
            .get("deg")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `deg`"))? as usize;
        if x.len() != y.len() {
            bail!("x/y length mismatch");
        }
        let n = x.len();
        let cols = deg + 1;
        let mut a = vec![0.0; n * cols];
        for (i, &xi) in x.iter().enumerate() {
            for j in 0..cols {
                a[i * cols + j] = xi.powi((cols - 1 - j) as i32);
            }
        }
        let am = nalgebra::DMatrix::from_row_slice(n, cols, &a);
        let bm = nalgebra::DVector::from_iterator(n, y.iter().copied());
        let svd = am.svd(true, true);
        let x = svd.solve(&bm, 1e-12).map_err(|e| anyhow!("polyfit: {e}"))?;
        let coeffs: Vec<f64> = x.iter().rev().copied().collect();
        Ok(json!({"coefficients": coeffs}))
    })
}

// ── arr — reductions axis-naive ────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__arr_argmax_flat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let r = a
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64);
        Ok(json!({"argmax": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_argmin_flat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let r = a
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64);
        Ok(json!({"argmin": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_argwhere(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let r: Vec<i64> = a
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if *x != 0.0 { Some(i as i64) } else { None })
            .collect();
        Ok(json!({"indices": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_nonzero(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let r: Vec<i64> = a
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if *x != 0.0 { Some(i as i64) } else { None })
            .collect();
        Ok(json!({"nonzero": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_take_flat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let idx = args
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `indices`"))?;
        let v: Vec<f64> = a.iter().copied().collect();
        let out: Vec<f64> = idx
            .iter()
            .filter_map(|i| i.as_u64())
            .filter_map(|i| v.get(i as usize).copied())
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("take")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_put_flat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let idx = args
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `indices`"))?;
        let vals = args
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `values`"))?;
        let mut v: Vec<f64> = a.iter().copied().collect();
        for (i, val) in idx.iter().zip(vals.iter()) {
            let i = i.as_u64().unwrap_or(0) as usize;
            let val = val.as_f64().unwrap_or(0.0);
            if i < v.len() {
                v[i] = val;
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(a.shape()), v).context("put")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_diagonal_offset(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("diagonal: 2-D array required");
        }
        let k = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let mut out = vec![];
        if k >= 0 {
            let k = k as usize;
            for i in 0..rows {
                if i + k < cols {
                    out.push(a[[i, i + k].as_slice()]);
                }
            }
        } else {
            let k = (-k) as usize;
            for i in 0..cols {
                if i + k < rows {
                    out.push(a[[i + k, i].as_slice()]);
                }
            }
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("diagonal")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_triu(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("triu: 2-D");
        }
        let k = args.get("k").and_then(|v| v.as_i64()).unwrap_or(0);
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                if (j as i64) >= (i as i64) + k {
                    out[i * cols + j] = a[[i, j].as_slice()];
                }
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[rows, cols]), out).context("triu")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_tril(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("tril: 2-D");
        }
        let k = args.get("k").and_then(|v| v.as_i64()).unwrap_or(0);
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                if (j as i64) <= (i as i64) + k {
                    out[i * cols + j] = a[[i, j].as_slice()];
                }
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[rows, cols]), out).context("tril")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_pad_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let before = args.get("before").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let after = args.get("after").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let value = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let v: Vec<f64> = a.iter().copied().collect();
        let mut out = vec![value; before];
        out.extend(&v);
        out.extend(vec![value; after]);
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("pad")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_roll_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let shift = args.get("shift").and_then(|v| v.as_i64()).unwrap_or(0);
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        if n == 0 {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let s = ((shift % n as i64) + n as i64) as usize % n;
        let mut out = Vec::with_capacity(n);
        out.extend_from_slice(&v[n - s..]);
        out.extend_from_slice(&v[..n - s]);
        let arr = ArrayD::from_shape_vec(IxDyn(a.shape()), out).context("roll")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_clip_axis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let lo = args
            .get("min")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let hi = args
            .get("max")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::INFINITY);
        let r = a.mapv(|x| x.clamp(lo, hi));
        Ok(json!({"array": array_to_value(&r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_resize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let shape: Vec<usize> = args
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `shape`"))?
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect();
        let target: usize = shape.iter().product();
        let v: Vec<f64> = a.iter().copied().collect();
        let mut out = vec![];
        for i in 0..target {
            out.push(v[i % v.len()]);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), out).context("resize")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_unique_flat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let mut seen = std::collections::HashSet::new();
        let mut out = vec![];
        for x in a.iter() {
            if seen.insert(x.to_bits()) {
                out.push(*x);
            }
        }
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("unique")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_isclose_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let rtol = args.get("rtol").and_then(|v| v.as_f64()).unwrap_or(1e-5);
        let atol = args.get("atol").and_then(|v| v.as_f64()).unwrap_or(1e-8);
        if a.shape() != b.shape() {
            bail!("shape mismatch");
        }
        let data: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                if (x - y).abs() <= atol + rtol * y.abs() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let r = ArrayD::from_shape_vec(IxDyn(a.shape()), data).context("isclose")?;
        Ok(json!({"array": array_to_value(&r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_allclose_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let rtol = args.get("rtol").and_then(|v| v.as_f64()).unwrap_or(1e-5);
        let atol = args.get("atol").and_then(|v| v.as_f64()).unwrap_or(1e-8);
        if a.shape() != b.shape() {
            return Ok(json!({"allclose": false}));
        }
        let r = a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs());
        Ok(json!({"allclose": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_array_equal_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape() != b.shape() {
            return Ok(json!({"array_equal": false}));
        }
        let eq = a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()));
        Ok(json!({"array_equal": eq}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_searchsorted_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let pos = a.iter().position(|x| *x >= v).unwrap_or(a.len());
        Ok(json!({"position": pos}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_bincount_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let max = a.iter().map(|x| *x as i64).max().unwrap_or(0).max(0) as usize;
        let mut counts = vec![0i64; max + 1];
        for x in a.iter() {
            let i = *x as i64;
            if i >= 0 && (i as usize) <= max {
                counts[i as usize] += 1;
            }
        }
        Ok(json!({"counts": counts}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_digitize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let bins = args
            .get("bins")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `bins`"))?;
        let b: Vec<f64> = bins.iter().filter_map(|x| x.as_f64()).collect();
        let out: Vec<i64> = a
            .iter()
            .map(|x| b.iter().position(|t| t > x).unwrap_or(b.len()) as i64)
            .collect();
        Ok(json!({"digitize": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_select(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let conds = args
            .get("conditions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `conditions`"))?;
        let choices = args
            .get("choices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `choices`"))?;
        let default = args.get("default").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if conds.len() != choices.len() || conds.is_empty() {
            bail!("conditions/choices length mismatch");
        }
        let cv: Vec<Vec<bool>> = conds
            .iter()
            .map(|c| {
                c.as_array()
                    .map(|a| a.iter().map(|x| x.as_bool().unwrap_or(false)).collect())
                    .unwrap_or_default()
            })
            .collect();
        let chv: Vec<Vec<f64>> = choices
            .iter()
            .map(|c| {
                c.as_array()
                    .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default()
            })
            .collect();
        let n = cv[0].len();
        let mut out = vec![default; n];
        for i in 0..n {
            for (j, c) in cv.iter().enumerate() {
                if i < c.len() && c[i] {
                    out[i] = chv[j][i];
                    break;
                }
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("select")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_where(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let cond = args
            .get("condition")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `condition`"))?;
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let out: Vec<f64> = (0..a.len())
            .map(|i| {
                if cond.get(i).and_then(|v| v.as_bool()).unwrap_or(false) {
                    a.iter().nth(i).copied().unwrap_or(0.0)
                } else {
                    b.iter().nth(i).copied().unwrap_or(0.0)
                }
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(a.shape()), out).context("where")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__arr_choose(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let idx = args
            .get("index")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `index`"))?;
        let choices = args
            .get("choices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `choices`"))?;
        let chv: Vec<Vec<f64>> = choices
            .iter()
            .map(|c| {
                c.as_array()
                    .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default()
            })
            .collect();
        let out: Vec<f64> = idx
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let j = x.as_u64().unwrap_or(0) as usize;
                if j >= chv.len() {
                    0.0
                } else {
                    *chv[j].get(i).unwrap_or(&0.0)
                }
            })
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("choose")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}
