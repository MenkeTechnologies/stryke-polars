//! src/ma.rs — numpy masked arrays surface (`polars__ma_*`).
//!
//! Wire format:
//!   `{masked: {data: [...], mask: [bool, ...]}}`
//! `mask[i] == true` means element i is masked (invalid). Output uses
//! the same envelope.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::ffi_call;

// ── helpers ────────────────────────────────────────────────────────────────

fn parse_masked(v: &Value) -> Result<(Vec<f64>, Vec<bool>)> {
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("masked `data` missing"))?;
    let mask = v
        .get("mask")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow!("masked `mask` missing"))?;
    if data.len() != mask.len() {
        bail!(
            "data/mask length mismatch: {} vs {}",
            data.len(),
            mask.len()
        );
    }
    let d: Vec<f64> = data
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect();
    let m: Vec<bool> = mask.iter().map(|x| x.as_bool().unwrap_or(false)).collect();
    Ok((d, m))
}

fn get_masked(args: &Value) -> Result<(Vec<f64>, Vec<bool>)> {
    let v = args
        .get("masked")
        .ok_or_else(|| anyhow!("missing argument `masked`"))?;
    parse_masked(v)
}

fn return_masked(data: Vec<f64>, mask: Vec<bool>) -> Result<Value> {
    let d: Vec<Value> = data
        .iter()
        .map(|x| {
            serde_json::Number::from_f64(*x)
                .map(Value::Number)
                .unwrap_or(Value::Null)
        })
        .collect();
    let m: Vec<Value> = mask.iter().map(|b| Value::Bool(*b)).collect();
    Ok(json!({"masked": {"data": d, "mask": m}}))
}

fn unmasked(data: &[f64], mask: &[bool]) -> Vec<f64> {
    data.iter()
        .zip(mask.iter())
        .filter_map(|(x, m)| if !*m { Some(*x) } else { None })
        .collect()
}

fn scalar(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

// ── construction ────────────────────────────────────────────────────────────

/// Masked array array.
#[no_mangle]
pub extern "C" fn polars__ma_array(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        return_masked(d, m)
    })
}

/// Masked array masked equal.
#[no_mangle]
pub extern "C" fn polars__ma_masked_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `data`"))?;
        let val = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `value`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| *x == val).collect();
        return_masked(d, m)
    })
}

/// Masked array masked not equal.
#[no_mangle]
pub extern "C" fn polars__ma_masked_not_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `data`"))?;
        let val = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `value`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| *x != val).collect();
        return_masked(d, m)
    })
}

/// Masked array masked greater.
#[no_mangle]
pub extern "C" fn polars__ma_masked_greater(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let val = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| *x > val).collect();
        return_masked(d, m)
    })
}

/// Masked array masked greater equal.
#[no_mangle]
pub extern "C" fn polars__ma_masked_greater_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let val = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| *x >= val).collect();
        return_masked(d, m)
    })
}

/// Masked array masked less.
#[no_mangle]
pub extern "C" fn polars__ma_masked_less(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let val = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| *x < val).collect();
        return_masked(d, m)
    })
}

/// Masked array masked less equal.
#[no_mangle]
pub extern "C" fn polars__ma_masked_less_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let val = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| *x <= val).collect();
        return_masked(d, m)
    })
}

/// Masked array masked inside.
#[no_mangle]
pub extern "C" fn polars__ma_masked_inside(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let lo = args
            .get("lo")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lo`"))?;
        let hi = args
            .get("hi")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `hi`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| *x >= lo && *x <= hi).collect();
        return_masked(d, m)
    })
}

/// Masked array masked outside.
#[no_mangle]
pub extern "C" fn polars__ma_masked_outside(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let lo = args
            .get("lo")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lo`"))?;
        let hi = args
            .get("hi")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `hi`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| *x < lo || *x > hi).collect();
        return_masked(d, m)
    })
}

/// Masked array masked invalid.
#[no_mangle]
pub extern "C" fn polars__ma_masked_invalid(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = d.iter().map(|x| !x.is_finite()).collect();
        return_masked(d, m)
    })
}

/// Masked array masked where.
#[no_mangle]
pub extern "C" fn polars__ma_masked_where(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let cond = args
            .get("condition")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `condition`"))?;
        let d: Vec<f64> = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let m: Vec<bool> = cond.iter().map(|x| x.as_bool().unwrap_or(false)).collect();
        return_masked(d, m)
    })
}

// ── modify ──────────────────────────────────────────────────────────────────

/// Masked array filled.
#[no_mangle]
pub extern "C" fn polars__ma_filled(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let val = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let out: Vec<f64> = d
            .iter()
            .zip(m.iter())
            .map(|(x, mk)| if *mk { val } else { *x })
            .collect();
        let data: Vec<Value> = out.iter().map(|x| scalar(*x)).collect();
        Ok(json!({"data": data}))
    })
}

/// Masked array compressed.
#[no_mangle]
pub extern "C" fn polars__ma_compressed(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let out = unmasked(&d, &m);
        let data: Vec<Value> = out.iter().map(|x| scalar(*x)).collect();
        Ok(json!({"data": data}))
    })
}

/// Masked array count.
#[no_mangle]
pub extern "C" fn polars__ma_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, m) = get_masked(&args)?;
        Ok(json!({"count": m.iter().filter(|b| !**b).count()}))
    })
}

/// Masked array count masked.
#[no_mangle]
pub extern "C" fn polars__ma_count_masked(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, m) = get_masked(&args)?;
        Ok(json!({"count_masked": m.iter().filter(|b| **b).count()}))
    })
}

/// Masked array getmask.
#[no_mangle]
pub extern "C" fn polars__ma_getmask(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, m) = get_masked(&args)?;
        let arr: Vec<Value> = m.iter().map(|b| Value::Bool(*b)).collect();
        Ok(json!({"mask": arr}))
    })
}

/// Masked array getdata.
#[no_mangle]
pub extern "C" fn polars__ma_getdata(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, _) = get_masked(&args)?;
        let arr: Vec<Value> = d.iter().map(|x| scalar(*x)).collect();
        Ok(json!({"data": arr}))
    })
}

/// Masked array size.
#[no_mangle]
pub extern "C" fn polars__ma_size(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, _) = get_masked(&args)?;
        Ok(json!({"size": d.len()}))
    })
}

/// Masked array set fill value.
#[no_mangle]
pub extern "C" fn polars__ma_set_fill_value(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let _ = args.get("fill_value").and_then(|v| v.as_f64());
        return_masked(d, m)
    })
}

// ── aggregations (ignore masked) ────────────────────────────────────────────

macro_rules! ma_agg {
    ($fn_name:ident, $key:literal, $reducer:expr, $empty:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let (d, m) = get_masked(&args)?;
                let v = unmasked(&d, &m);
                let r = if v.is_empty() { $empty } else { $reducer(&v) };
                Ok(json!({ $key: scalar(r) }))
            })
        }
    };
}

ma_agg!(
    polars__ma_sum,
    "sum",
    |v: &Vec<f64>| v.iter().sum::<f64>(),
    0.0
);
ma_agg!(
    polars__ma_mean,
    "mean",
    |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64,
    f64::NAN
);
ma_agg!(
    polars__ma_min,
    "min",
    |v: &Vec<f64>| v.iter().cloned().fold(f64::INFINITY, f64::min),
    f64::NAN
);
ma_agg!(
    polars__ma_max,
    "max",
    |v: &Vec<f64>| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    f64::NAN
);
ma_agg!(
    polars__ma_product,
    "product",
    |v: &Vec<f64>| v.iter().product(),
    1.0
);
ma_agg!(
    polars__ma_ptp,
    "ptp",
    |v: &Vec<f64>| {
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        mx - mn
    },
    f64::NAN
);

/// Masked array std.
#[no_mangle]
pub extern "C" fn polars__ma_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let v = unmasked(&d, &m);
        let n = v.len() as f64;
        let mu = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        Ok(json!({"std": scalar(var.sqrt())}))
    })
}

/// Masked array var.
#[no_mangle]
pub extern "C" fn polars__ma_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let v = unmasked(&d, &m);
        let n = v.len() as f64;
        let mu = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        Ok(json!({"var": scalar(var)}))
    })
}

/// Masked array median.
#[no_mangle]
pub extern "C" fn polars__ma_median(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let mut v = unmasked(&d, &m);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let r = if v.is_empty() {
            f64::NAN
        } else if v.len().is_multiple_of(2) {
            (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
        } else {
            v[v.len() / 2]
        };
        Ok(json!({"median": scalar(r)}))
    })
}

/// Masked array argmin.
#[no_mangle]
pub extern "C" fn polars__ma_argmin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let r = d
            .iter()
            .enumerate()
            .filter(|(i, _)| !m[*i])
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64);
        Ok(json!({"argmin": r}))
    })
}

/// Masked array argmax.
#[no_mangle]
pub extern "C" fn polars__ma_argmax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let r = d
            .iter()
            .enumerate()
            .filter(|(i, _)| !m[*i])
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64);
        Ok(json!({"argmax": r}))
    })
}

/// Masked array any.
#[no_mangle]
pub extern "C" fn polars__ma_any(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let v = unmasked(&d, &m);
        Ok(json!({"any": v.iter().any(|x| *x != 0.0)}))
    })
}

/// Masked array all.
#[no_mangle]
pub extern "C" fn polars__ma_all(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let v = unmasked(&d, &m);
        Ok(json!({"all": v.iter().all(|x| *x != 0.0)}))
    })
}

// ── cumulative (ignoring mask) ──────────────────────────────────────────────

/// Masked array cumsum.
#[no_mangle]
pub extern "C" fn polars__ma_cumsum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let mut acc = 0.0;
        let out: Vec<f64> = d
            .iter()
            .zip(m.iter())
            .map(|(x, mk)| {
                if !*mk {
                    acc += x;
                }
                acc
            })
            .collect();
        return_masked(out, m)
    })
}

/// Masked array cumprod.
#[no_mangle]
pub extern "C" fn polars__ma_cumprod(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let mut acc = 1.0;
        let out: Vec<f64> = d
            .iter()
            .zip(m.iter())
            .map(|(x, mk)| {
                if !*mk {
                    acc *= x;
                }
                acc
            })
            .collect();
        return_masked(out, m)
    })
}

// ── arithmetic (masked element ⇒ masked output) ────────────────────────────

macro_rules! ma_unary {
    ($fn_name:ident, $f:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let (d, m) = get_masked(&args)?;
                let out: Vec<f64> = d.iter().map(|x| ($f)(*x)).collect();
                return_masked(out, m)
            })
        }
    };
}

ma_unary!(polars__ma_abs, f64::abs);
ma_unary!(polars__ma_negative, |x: f64| -x);
ma_unary!(polars__ma_sqrt, f64::sqrt);
ma_unary!(polars__ma_exp, f64::exp);
ma_unary!(polars__ma_log, f64::ln);
ma_unary!(polars__ma_log2, f64::log2);
ma_unary!(polars__ma_log10, f64::log10);
ma_unary!(polars__ma_sin, f64::sin);
ma_unary!(polars__ma_cos, f64::cos);
ma_unary!(polars__ma_tan, f64::tan);
ma_unary!(polars__ma_arcsin, f64::asin);
ma_unary!(polars__ma_arccos, f64::acos);
ma_unary!(polars__ma_arctan, f64::atan);
ma_unary!(polars__ma_sinh, f64::sinh);
ma_unary!(polars__ma_cosh, f64::cosh);
ma_unary!(polars__ma_tanh, f64::tanh);
ma_unary!(polars__ma_floor, f64::floor);
ma_unary!(polars__ma_ceil, f64::ceil);
ma_unary!(polars__ma_round, |x: f64| x.round());
ma_unary!(polars__ma_trunc, f64::trunc);
ma_unary!(polars__ma_square, |x: f64| x * x);
ma_unary!(polars__ma_cbrt, f64::cbrt);
ma_unary!(polars__ma_reciprocal, f64::recip);
ma_unary!(polars__ma_sign, f64::signum);

// ── scalar arithmetic ───────────────────────────────────────────────────────

macro_rules! ma_scalar {
    ($fn_name:ident, $op:tt) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let (d, m) = get_masked(&args)?;
                let s = args.get("scalar").and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow!("missing `scalar`"))?;
                let out: Vec<f64> = d.iter().map(|x| x $op s).collect();
                return_masked(out, m)
            })
        }
    };
}

ma_scalar!(polars__ma_add_scalar, +);
ma_scalar!(polars__ma_sub_scalar, -);
ma_scalar!(polars__ma_mul_scalar, *);
ma_scalar!(polars__ma_div_scalar, /);

// ── binary masked-masked ────────────────────────────────────────────────────

/// (data_a, mask_a, data_b, mask_b) parsed pair of masked arrays.
type TwoMasked = (Vec<f64>, Vec<bool>, Vec<f64>, Vec<bool>);

fn get_two_masked(args: &Value) -> Result<TwoMasked> {
    let a = args.get("a").ok_or_else(|| anyhow!("missing `a`"))?;
    let b = args.get("b").ok_or_else(|| anyhow!("missing `b`"))?;
    let (da, ma) = parse_masked(a)?;
    let (db, mb) = parse_masked(b)?;
    if da.len() != db.len() {
        bail!("len mismatch: {} vs {}", da.len(), db.len());
    }
    Ok((da, ma, db, mb))
}

macro_rules! ma_binary {
    ($fn_name:ident, $op:tt) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let (da, ma, db, mb) = get_two_masked(&args)?;
                let out: Vec<f64> = da.iter().zip(db.iter()).map(|(x, y)| x $op y).collect();
                let m: Vec<bool> = ma.iter().zip(mb.iter()).map(|(a, b)| *a || *b).collect();
                return_masked(out, m)
            })
        }
    };
}

ma_binary!(polars__ma_add, +);
ma_binary!(polars__ma_sub, -);
ma_binary!(polars__ma_mul, *);
ma_binary!(polars__ma_div, /);

/// Masked array power.
#[no_mangle]
pub extern "C" fn polars__ma_power(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, db, mb) = get_two_masked(&args)?;
        let out: Vec<f64> = da.iter().zip(db.iter()).map(|(x, y)| x.powf(*y)).collect();
        let m: Vec<bool> = ma.iter().zip(mb.iter()).map(|(a, b)| *a || *b).collect();
        return_masked(out, m)
    })
}

/// Masked array maximum.
#[no_mangle]
pub extern "C" fn polars__ma_maximum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, db, mb) = get_two_masked(&args)?;
        let out: Vec<f64> = da.iter().zip(db.iter()).map(|(x, y)| x.max(*y)).collect();
        let m: Vec<bool> = ma.iter().zip(mb.iter()).map(|(a, b)| *a || *b).collect();
        return_masked(out, m)
    })
}

/// Masked array minimum.
#[no_mangle]
pub extern "C" fn polars__ma_minimum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, db, mb) = get_two_masked(&args)?;
        let out: Vec<f64> = da.iter().zip(db.iter()).map(|(x, y)| x.min(*y)).collect();
        let m: Vec<bool> = ma.iter().zip(mb.iter()).map(|(a, b)| *a || *b).collect();
        return_masked(out, m)
    })
}

// ── mask manipulation ───────────────────────────────────────────────────────

/// Masked array mask or.
#[no_mangle]
pub extern "C" fn polars__ma_mask_or(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, _db, mb) = get_two_masked(&args)?;
        let m: Vec<bool> = ma.iter().zip(mb.iter()).map(|(a, b)| *a || *b).collect();
        return_masked(da, m)
    })
}

/// Masked array mask and.
#[no_mangle]
pub extern "C" fn polars__ma_mask_and(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, _db, mb) = get_two_masked(&args)?;
        let m: Vec<bool> = ma.iter().zip(mb.iter()).map(|(a, b)| *a && *b).collect();
        return_masked(da, m)
    })
}

/// Masked array nomask.
#[no_mangle]
pub extern "C" fn polars__ma_nomask(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, _) = get_masked(&args)?;
        let m = vec![false; d.len()];
        return_masked(d, m)
    })
}

/// Masked array mask all.
#[no_mangle]
pub extern "C" fn polars__ma_mask_all(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, _) = get_masked(&args)?;
        let m = vec![true; d.len()];
        return_masked(d, m)
    })
}

/// Masked array invert mask.
#[no_mangle]
pub extern "C" fn polars__ma_invert_mask(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let inv: Vec<bool> = m.iter().map(|b| !*b).collect();
        return_masked(d, inv)
    })
}

/// Masked array anymasked.
#[no_mangle]
pub extern "C" fn polars__ma_anymasked(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, m) = get_masked(&args)?;
        Ok(json!({"anymasked": m.iter().any(|b| *b)}))
    })
}

/// Masked array allmasked.
#[no_mangle]
pub extern "C" fn polars__ma_allmasked(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, m) = get_masked(&args)?;
        Ok(json!({"allmasked": !m.is_empty() && m.iter().all(|b| *b)}))
    })
}

// ── sort / unique ───────────────────────────────────────────────────────────

/// Masked array sort.
#[no_mangle]
pub extern "C" fn polars__ma_sort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let mut paired: Vec<(f64, bool)> = d.into_iter().zip(m).collect();
        paired.sort_by(|a, b| {
            if a.1 != b.1 {
                if a.1 {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            } else {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        let (d2, m2): (Vec<f64>, Vec<bool>) = paired.into_iter().unzip();
        return_masked(d2, m2)
    })
}

/// Masked array reverse.
#[no_mangle]
pub extern "C" fn polars__ma_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (mut d, mut m) = get_masked(&args)?;
        d.reverse();
        m.reverse();
        return_masked(d, m)
    })
}

/// Masked array unique.
#[no_mangle]
pub extern "C" fn polars__ma_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let v = unmasked(&d, &m);
        let mut seen = std::collections::HashSet::new();
        let out: Vec<f64> = v.into_iter().filter(|x| seen.insert(x.to_bits())).collect();
        let n = out.len();
        let mask = vec![false; n];
        return_masked(out, mask)
    })
}

/// Masked array diff.
#[no_mangle]
pub extern "C" fn polars__ma_diff(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        if d.is_empty() {
            return return_masked(vec![], vec![]);
        }
        let mut out = Vec::with_capacity(d.len() - 1);
        let mut mo = Vec::with_capacity(d.len() - 1);
        for i in 1..d.len() {
            out.push(d[i] - d[i - 1]);
            mo.push(m[i] || m[i - 1]);
        }
        return_masked(out, mo)
    })
}

/// Masked array clip.
#[no_mangle]
pub extern "C" fn polars__ma_clip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let lo = args
            .get("min")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let hi = args
            .get("max")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::INFINITY);
        let out: Vec<f64> = d.iter().map(|x| x.clamp(lo, hi)).collect();
        return_masked(out, m)
    })
}

/// Masked array concat.
#[no_mangle]
pub extern "C" fn polars__ma_concat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma) = parse_masked(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let (db, mb) = parse_masked(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        let mut d = da;
        d.extend(db);
        let mut m = ma;
        m.extend(mb);
        return_masked(d, m)
    })
}

/// Masked array repeat.
#[no_mangle]
pub extern "C" fn polars__ma_repeat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let times = args.get("times").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mut od = Vec::with_capacity(d.len() * times);
        let mut om = Vec::with_capacity(d.len() * times);
        for _ in 0..times {
            od.extend(&d);
            om.extend(&m);
        }
        return_masked(od, om)
    })
}

/// Masked array take.
#[no_mangle]
pub extern "C" fn polars__ma_take(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let idx = args
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `indices`"))?;
        let mut od = Vec::with_capacity(idx.len());
        let mut om = Vec::with_capacity(idx.len());
        for i in idx {
            let i = i.as_u64().unwrap_or(0) as usize;
            if i >= d.len() {
                bail!("take index {} out of bounds", i);
            }
            od.push(d[i]);
            om.push(m[i]);
        }
        return_masked(od, om)
    })
}

/// Masked array zeros.
#[no_mangle]
pub extern "C" fn polars__ma_zeros(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        return_masked(vec![0.0; n], vec![false; n])
    })
}

/// Masked array ones.
#[no_mangle]
pub extern "C" fn polars__ma_ones(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        return_masked(vec![1.0; n], vec![false; n])
    })
}

/// Masked array empty.
#[no_mangle]
pub extern "C" fn polars__ma_empty(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        return_masked(vec![0.0; n], vec![true; n])
    })
}

/// Masked array dot.
#[no_mangle]
pub extern "C" fn polars__ma_dot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, db, mb) = get_two_masked(&args)?;
        let mut s = 0.0;
        for i in 0..da.len() {
            if !ma[i] && !mb[i] {
                s += da[i] * db[i];
            }
        }
        Ok(json!({"dot": scalar(s)}))
    })
}

/// Masked array corrcoef.
#[no_mangle]
pub extern "C" fn polars__ma_corrcoef(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, db, mb) = get_two_masked(&args)?;
        let pairs: Vec<(f64, f64)> = (0..da.len())
            .filter(|i| !ma[*i] && !mb[*i])
            .map(|i| (da[i], db[i]))
            .collect();
        if pairs.is_empty() {
            return Ok(json!({"corrcoef": Value::Null}));
        }
        let n = pairs.len() as f64;
        let mx = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
        let my = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;
        let cov = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / n;
        let sx = (pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum::<f64>() / n).sqrt();
        let sy = (pairs.iter().map(|(_, y)| (y - my).powi(2)).sum::<f64>() / n).sqrt();
        let r = if sx * sy == 0.0 {
            f64::NAN
        } else {
            cov / (sx * sy)
        };
        Ok(json!({"corrcoef": scalar(r)}))
    })
}

/// Masked array cov.
#[no_mangle]
pub extern "C" fn polars__ma_cov(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, db, mb) = get_two_masked(&args)?;
        let pairs: Vec<(f64, f64)> = (0..da.len())
            .filter(|i| !ma[*i] && !mb[*i])
            .map(|i| (da[i], db[i]))
            .collect();
        if pairs.is_empty() {
            return Ok(json!({"cov": Value::Null}));
        }
        let n = pairs.len() as f64;
        let mx = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
        let my = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;
        let cov = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / (n - 1.0).max(1.0);
        Ok(json!({"cov": scalar(cov)}))
    })
}

/// Masked array average.
#[no_mangle]
pub extern "C" fn polars__ma_average(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let weights = args.get("weights").and_then(|v| v.as_array());
        match weights {
            None => {
                let v = unmasked(&d, &m);
                let r = if v.is_empty() {
                    f64::NAN
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                };
                Ok(json!({"average": scalar(r)}))
            }
            Some(w) => {
                let ws: Vec<f64> = w.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
                if ws.len() != d.len() {
                    bail!("weights len mismatch");
                }
                let mut sw = 0.0;
                let mut sv = 0.0;
                for i in 0..d.len() {
                    if !m[i] {
                        sv += d[i] * ws[i];
                        sw += ws[i];
                    }
                }
                Ok(json!({"average": scalar(if sw == 0.0 { f64::NAN } else { sv / sw })}))
            }
        }
    })
}

/// Masked array array equal.
#[no_mangle]
pub extern "C" fn polars__ma_array_equal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (da, ma, db, mb) = get_two_masked(&args)?;
        let mut eq = true;
        for i in 0..da.len() {
            if ma[i] != mb[i] {
                eq = false;
                break;
            }
            if !ma[i] && (da[i] - db[i]).abs() > 1e-12 {
                eq = false;
                break;
            }
        }
        Ok(json!({"array_equal": eq}))
    })
}

/// Masked array ravel.
#[no_mangle]
pub extern "C" fn polars__ma_ravel(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        return_masked(d, m)
    })
}

/// Masked array anomalies.
#[no_mangle]
pub extern "C" fn polars__ma_anomalies(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let v = unmasked(&d, &m);
        let mu = if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        };
        let out: Vec<f64> = d.iter().map(|x| x - mu).collect();
        return_masked(out, m)
    })
}

/// Masked array count unique.
#[no_mangle]
pub extern "C" fn polars__ma_count_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        let v = unmasked(&d, &m);
        let mut set = std::collections::HashSet::new();
        for x in v {
            set.insert(x.to_bits());
        }
        Ok(json!({"count_unique": set.len()}))
    })
}

/// Masked array isMasked.
#[no_mangle]
pub extern "C" fn polars__ma_isMasked(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, m) = get_masked(&args)?;
        Ok(json!({"is_masked": !m.is_empty() && m.iter().any(|b| *b)}))
    })
}

/// Masked array flatten mask.
#[no_mangle]
pub extern "C" fn polars__ma_flatten_mask(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, m) = get_masked(&args)?;
        let arr: Vec<Value> = m.iter().map(|b| Value::Bool(*b)).collect();
        Ok(json!({"mask": arr}))
    })
}

/// Masked array harden mask.
#[no_mangle]
pub extern "C" fn polars__ma_harden_mask(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        return_masked(d, m)
    })
}

/// Masked array soften mask.
#[no_mangle]
pub extern "C" fn polars__ma_soften_mask(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (d, m) = get_masked(&args)?;
        return_masked(d, m)
    })
}
