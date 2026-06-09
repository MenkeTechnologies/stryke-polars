//! src/sr.rs — pandas Series surface (`polars__sr_*`).
//!
//! Wire format: `{series: {name?: str, data: [...]}}`.
//! Backed by polars Series. Most ops route through a single-column
//! DataFrame for lazy-eval compatibility, then extract column 0.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Context, Result};
use polars::prelude::*;
use serde_json::{json, Value};

use crate::ffi_call;

// ── JSON ↔ Series ───────────────────────────────────────────────────────────

fn parse_series(v: &Value) -> Result<Series> {
    let name = v
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("s")
        .to_string();
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("series `data` missing or not an array"))?;
    coerce_json_array(&name, data)
}

fn coerce_json_array(name: &str, arr: &[Value]) -> Result<Series> {
    if arr.is_empty() {
        return Ok(Series::new_empty(name.into(), &DataType::Null));
    }
    let (mut all_bool, mut all_i64, mut all_f64, mut all_str) = (true, true, true, true);
    for v in arr {
        match v {
            Value::Null => {}
            Value::Bool(_) => {
                all_i64 = false;
                all_f64 = false;
                all_str = false;
            }
            Value::Number(n) => {
                all_bool = false;
                all_str = false;
                if n.as_i64().is_none() {
                    all_i64 = false;
                }
                if n.as_f64().is_none() {
                    all_f64 = false;
                }
            }
            Value::String(_) => {
                all_bool = false;
                all_i64 = false;
                all_f64 = false;
            }
            _ => {
                all_bool = false;
                all_i64 = false;
                all_f64 = false;
                all_str = false;
            }
        }
    }
    if all_bool {
        let v: Vec<Option<bool>> = arr.iter().map(|x| x.as_bool()).collect();
        Ok(Series::new(name.into(), v))
    } else if all_i64 {
        let v: Vec<Option<i64>> = arr.iter().map(|x| x.as_i64()).collect();
        Ok(Series::new(name.into(), v))
    } else if all_f64 {
        let v: Vec<Option<f64>> = arr.iter().map(|x| x.as_f64()).collect();
        Ok(Series::new(name.into(), v))
    } else if all_str {
        let v: Vec<Option<String>> = arr.iter().map(|x| x.as_str().map(String::from)).collect();
        Ok(Series::new(name.into(), v))
    } else {
        let v: Vec<Option<String>> = arr
            .iter()
            .map(|x| match x {
                Value::Null => None,
                _ => Some(x.to_string()),
            })
            .collect();
        Ok(Series::new(name.into(), v))
    }
}

fn series_to_value(s: &Series) -> Result<Value> {
    let n = s.len();
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let av = s.get(i).map_err(|e| anyhow!("Series::get({i}): {e}"))?;
        data.push(any_value_to_json(av));
    }
    Ok(json!({"name": s.name().to_string(), "data": data}))
}

fn any_value_to_json(av: AnyValue<'_>) -> Value {
    match av {
        AnyValue::Null => Value::Null,
        AnyValue::Boolean(b) => Value::Bool(b),
        AnyValue::String(s) => Value::String(s.to_string()),
        AnyValue::StringOwned(s) => Value::String(s.to_string()),
        AnyValue::Int8(n) => json!(n),
        AnyValue::Int16(n) => json!(n),
        AnyValue::Int32(n) => json!(n),
        AnyValue::Int64(n) => json!(n),
        AnyValue::UInt8(n) => json!(n),
        AnyValue::UInt16(n) => json!(n),
        AnyValue::UInt32(n) => json!(n),
        AnyValue::UInt64(n) => json!(n),
        AnyValue::Float32(f) => serde_json::Number::from_f64(f as f64)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        AnyValue::Float64(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        other => Value::String(format!("{other}")),
    }
}

fn get_series(args: &Value) -> Result<Series> {
    let s = args
        .get("series")
        .ok_or_else(|| anyhow!("missing argument `series`"))?;
    parse_series(s)
}

fn get_two(args: &Value) -> Result<(Series, Series)> {
    let a = args
        .get("a")
        .ok_or_else(|| anyhow!("missing argument `a`"))?;
    let b = args
        .get("b")
        .ok_or_else(|| anyhow!("missing argument `b`"))?;
    Ok((parse_series(a)?, parse_series(b)?))
}

fn return_series(s: &Series) -> Result<Value> {
    Ok(json!({"series": series_to_value(s)?}))
}

fn scalar_f64(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn as_f64_vec(s: &Series) -> Result<Vec<f64>> {
    let f = s.cast(&DataType::Float64).context("cast f64")?;
    let ca = f.f64().context("not f64")?;
    Ok(ca.into_no_null_iter().collect())
}

fn from_f64_vec(name: &str, data: Vec<f64>) -> Series {
    Series::new(name.into(), data)
}

// ── construction ────────────────────────────────────────────────────────────

/// Build a Series from JSON array data.
#[no_mangle]
pub extern "C" fn polars__sr_new(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({
            "series": series_to_value(&s)?,
            "dtype": format!("{}", s.dtype()),
            "len": s.len(),
        }))
    })
}

/// Series length.
#[no_mangle]
pub extern "C" fn polars__sr_len(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"len": s.len()}))
    })
}

/// Series dtype.
#[no_mangle]
pub extern "C" fn polars__sr_dtype(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"dtype": format!("{}", s.dtype())}))
    })
}

/// Series name.
#[no_mangle]
pub extern "C" fn polars__sr_name(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"name": s.name().to_string()}))
    })
}

/// Rename a series.
#[no_mangle]
pub extern "C" fn polars__sr_rename(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut s = get_series(&args)?;
        let to = args
            .get("to")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `to`"))?;
        s.rename(to.into());
        return_series(&s)
    })
}

/// First `n` values.
#[no_mangle]
pub extern "C" fn polars__sr_head(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        return_series(&s.head(Some(n)))
    })
}

/// Last `n` values.
#[no_mangle]
pub extern "C" fn polars__sr_tail(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        return_series(&s.tail(Some(n)))
    })
}

/// Reverse order.
#[no_mangle]
pub extern "C" fn polars__sr_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.reverse())
    })
}

/// Slice [offset, offset+length).
#[no_mangle]
pub extern "C" fn polars__sr_slice(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let offset = args.get("offset").and_then(|v| v.as_i64()).unwrap_or(0);
        let length = args
            .get("length")
            .and_then(|v| v.as_u64())
            .unwrap_or(s.len() as u64) as usize;
        return_series(&s.slice(offset, length))
    })
}

// ── stats / aggregations ───────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let r: f64 = v.iter().sum();
        Ok(json!({"sum": scalar_f64(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let r = if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        };
        Ok(json!({"mean": scalar_f64(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_median(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let mut v = as_f64_vec(&s)?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let r = if v.is_empty() {
            f64::NAN
        } else if v.len() % 2 == 0 {
            (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
        } else {
            v[v.len() / 2]
        };
        Ok(json!({"median": scalar_f64(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let r = if v.is_empty() {
            f64::NAN
        } else {
            v.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        Ok(json!({"min": scalar_f64(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let r = if v.is_empty() {
            f64::NAN
        } else {
            v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
        Ok(json!({"max": scalar_f64(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let ddof = args.get("ddof").and_then(|v| v.as_u64()).unwrap_or(1) as f64;
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - ddof).max(1.0);
        Ok(json!({"std": scalar_f64(var.sqrt())}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let ddof = args.get("ddof").and_then(|v| v.as_u64()).unwrap_or(1) as f64;
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - ddof).max(1.0);
        Ok(json!({"var": scalar_f64(var)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"count": s.len() - s.null_count()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_null_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"null_count": s.null_count()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_nunique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"nunique": s.n_unique()?}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_quantile(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let q = args
            .get("q")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `q`"))?;
        let r = s
            .quantile_reduce(q, QuantileMethod::Linear)
            .context("quantile")?;
        Ok(json!({"quantile": format!("{}", r.value())}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_argmax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"argmax": s.arg_max()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_argmin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"argmin": s.arg_min()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_product(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let p: f64 = v.iter().product();
        Ok(json!({"product": p}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_range(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        if v.is_empty() {
            return Ok(json!({"range": Value::Null}));
        }
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Ok(json!({"range": mx - mn}))
    })
}

// ── cumulative ──────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_cumsum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut acc = 0.0;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                acc += x;
                acc
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_cumprod(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut acc = 1.0;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                acc *= x;
                acc
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_cummin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut acc = f64::INFINITY;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                acc = acc.min(*x);
                acc
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_cummax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut acc = f64::NEG_INFINITY;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                acc = acc.max(*x);
                acc
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

// ── nulls ──────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_is_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.is_null().into_series())
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_is_not_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.is_not_null().into_series())
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_drop_nulls(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.drop_nulls())
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_fill_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `value`"))?;
        return_series(
            &s.fill_null(FillNullStrategy::Zero)
                .ok()
                .and_then(|x| if v == 0.0 { Some(x) } else { None })
                .unwrap_or_else(|| {
                    let f = s.cast(&DataType::Float64).unwrap();
                    let ca = f.f64().unwrap();
                    let out: Vec<f64> = ca.iter().map(|x| x.unwrap_or(v)).collect();
                    Series::new(s.name().clone(), out)
                }),
        )
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_fill_null_forward(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.fill_null(FillNullStrategy::Forward(None))?)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_fill_null_backward(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.fill_null(FillNullStrategy::Backward(None))?)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_fill_null_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.fill_null(FillNullStrategy::Mean)?)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_fill_null_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.fill_null(FillNullStrategy::Min)?)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_fill_null_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.fill_null(FillNullStrategy::Max)?)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_fill_nan(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let f = s.cast(&DataType::Float64).context("cast f64")?;
        let ca = f.f64().context("not f64")?;
        let out: Vec<f64> = ca
            .iter()
            .map(|x| x.map(|y| if y.is_nan() { v } else { y }).unwrap_or(v))
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

// ── unique / dedup ──────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        return_series(&s.unique()?)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_value_counts(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let df = s
            .value_counts(true, true, "count".into(), false)
            .context("value_counts")?;
        let mut out = serde_json::Map::new();
        for col in df.get_columns() {
            let series = col.as_materialized_series();
            let n = series.len();
            let mut arr = Vec::with_capacity(n);
            for i in 0..n {
                let av = series.get(i).map_err(|e| anyhow!("get: {e}"))?;
                arr.push(any_value_to_json(av));
            }
            out.insert(col.name().to_string(), Value::Array(arr));
        }
        Ok(json!({"value_counts": Value::Object(out)}))
    })
}

// ── sort / rank ─────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_sort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let desc = args.get("desc").and_then(|v| v.as_bool()).unwrap_or(false);
        let opts = SortOptions::default()
            .with_order_descending(desc)
            .with_nulls_last(true);
        return_series(&s.sort(opts)?)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_argsort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let desc = args.get("desc").and_then(|v| v.as_bool()).unwrap_or(false);
        let opts = SortOptions::default().with_order_descending(desc);
        let idx = s.arg_sort(opts);
        return_series(&idx.into_series())
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rank(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let opts = RankOptions {
            method: RankMethod::Average,
            descending: false,
        };
        return_series(&s.rank(opts, None))
    })
}

// ── shift / diff / pct_change ───────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_shift(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
        return_series(&s.shift(n))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_diff(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1) as usize;
        if n >= v.len() {
            return return_series(&from_f64_vec(s.name(), vec![f64::NAN; v.len()]));
        }
        let mut out = vec![f64::NAN; n];
        for i in n..v.len() {
            out.push(v[i] - v[i - n]);
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_pct_change(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1) as usize;
        if n >= v.len() {
            return return_series(&from_f64_vec(s.name(), vec![f64::NAN; v.len()]));
        }
        let mut out = vec![f64::NAN; n];
        for i in n..v.len() {
            let prev = v[i - n];
            out.push(if prev == 0.0 {
                f64::NAN
            } else {
                v[i] / prev - 1.0
            });
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

// ── scalar arithmetic ───────────────────────────────────────────────────────

macro_rules! sr_scalar_op {
    ($fn_name:ident, $op:tt) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let s = get_series(&args)?;
                let v = args
                    .get("scalar")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow!("missing argument `scalar`"))?;
                let xs = as_f64_vec(&s)?;
                let out: Vec<f64> = xs.iter().map(|x| x $op v).collect();
                return_series(&from_f64_vec(s.name(), out))
            })
        }
    };
}

sr_scalar_op!(polars__sr_add_scalar, +);
sr_scalar_op!(polars__sr_sub_scalar, -);
sr_scalar_op!(polars__sr_mul_scalar, *);
sr_scalar_op!(polars__sr_div_scalar, /);

#[no_mangle]
pub extern "C" fn polars__sr_pow_scalar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = args
            .get("scalar")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `scalar`"))?;
        let xs = as_f64_vec(&s)?;
        let out: Vec<f64> = xs.iter().map(|x| x.powf(v)).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_mod_scalar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = args
            .get("scalar")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `scalar`"))?;
        let xs = as_f64_vec(&s)?;
        let out: Vec<f64> = xs.iter().map(|x| x % v).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_floordiv_scalar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = args
            .get("scalar")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `scalar`"))?;
        let xs = as_f64_vec(&s)?;
        let out: Vec<f64> = xs.iter().map(|x| (x / v).floor()).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

// ── series-series arithmetic ────────────────────────────────────────────────

macro_rules! sr_pair_op {
    ($fn_name:ident, $op:tt) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let (a, b) = get_two(&args)?;
                if a.len() != b.len() {
                    bail!("series len mismatch: {} vs {}", a.len(), b.len());
                }
                let va = as_f64_vec(&a)?;
                let vb = as_f64_vec(&b)?;
                let out: Vec<f64> = va.iter().zip(vb.iter()).map(|(x, y)| x $op y).collect();
                return_series(&from_f64_vec(a.name(), out))
            })
        }
    };
}

sr_pair_op!(polars__sr_add, +);
sr_pair_op!(polars__sr_sub, -);
sr_pair_op!(polars__sr_mul, *);
sr_pair_op!(polars__sr_div, /);

#[no_mangle]
pub extern "C" fn polars__sr_pow(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (a, b) = get_two(&args)?;
        let va = as_f64_vec(&a)?;
        let vb = as_f64_vec(&b)?;
        let out: Vec<f64> = va.iter().zip(vb.iter()).map(|(x, y)| x.powf(*y)).collect();
        return_series(&from_f64_vec(a.name(), out))
    })
}

// ── comparisons (return bool series) ────────────────────────────────────────

macro_rules! sr_cmp_scalar {
    ($fn_name:ident, $op:tt) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let s = get_series(&args)?;
                let v = args.get("scalar").and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow!("missing argument `scalar`"))?;
                let xs = as_f64_vec(&s)?;
                let out: Vec<bool> = xs.iter().map(|x| x $op &v).collect();
                let s_out = Series::new(s.name().clone(), out);
                return_series(&s_out)
            })
        }
    };
}

sr_cmp_scalar!(polars__sr_eq_scalar, ==);
sr_cmp_scalar!(polars__sr_ne_scalar, !=);
sr_cmp_scalar!(polars__sr_gt_scalar, >);
sr_cmp_scalar!(polars__sr_ge_scalar, >=);
sr_cmp_scalar!(polars__sr_lt_scalar, <);
sr_cmp_scalar!(polars__sr_le_scalar, <=);

macro_rules! sr_cmp_pair {
    ($fn_name:ident, $op:tt) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let (a, b) = get_two(&args)?;
                let va = as_f64_vec(&a)?;
                let vb = as_f64_vec(&b)?;
                let out: Vec<bool> = va.iter().zip(vb.iter()).map(|(x, y)| x $op y).collect();
                let s_out = Series::new(a.name().clone(), out);
                return_series(&s_out)
            })
        }
    };
}

sr_cmp_pair!(polars__sr_eq, ==);
sr_cmp_pair!(polars__sr_ne, !=);
sr_cmp_pair!(polars__sr_gt, >);
sr_cmp_pair!(polars__sr_ge, >=);
sr_cmp_pair!(polars__sr_lt, <);
sr_cmp_pair!(polars__sr_le, <=);

// ── unary numeric ───────────────────────────────────────────────────────────

macro_rules! sr_unary {
    ($fn_name:ident, $f:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let s = get_series(&args)?;
                let v = as_f64_vec(&s)?;
                let out: Vec<f64> = v.iter().map(|x| ($f)(*x)).collect();
                return_series(&from_f64_vec(s.name(), out))
            })
        }
    };
}

sr_unary!(polars__sr_abs, f64::abs);
sr_unary!(polars__sr_neg, |x: f64| -x);
sr_unary!(polars__sr_sqrt, f64::sqrt);
sr_unary!(polars__sr_cbrt, f64::cbrt);
sr_unary!(polars__sr_square, |x: f64| x * x);
sr_unary!(polars__sr_exp, f64::exp);
sr_unary!(polars__sr_exp2, f64::exp2);
sr_unary!(polars__sr_expm1, f64::exp_m1);
sr_unary!(polars__sr_log, f64::ln);
sr_unary!(polars__sr_log2, f64::log2);
sr_unary!(polars__sr_log10, f64::log10);
sr_unary!(polars__sr_log1p, f64::ln_1p);
sr_unary!(polars__sr_sin, f64::sin);
sr_unary!(polars__sr_cos, f64::cos);
sr_unary!(polars__sr_tan, f64::tan);
sr_unary!(polars__sr_asin, f64::asin);
sr_unary!(polars__sr_acos, f64::acos);
sr_unary!(polars__sr_atan, f64::atan);
sr_unary!(polars__sr_sinh, f64::sinh);
sr_unary!(polars__sr_cosh, f64::cosh);
sr_unary!(polars__sr_tanh, f64::tanh);
sr_unary!(polars__sr_asinh, f64::asinh);
sr_unary!(polars__sr_acosh, f64::acosh);
sr_unary!(polars__sr_atanh, f64::atanh);
sr_unary!(polars__sr_floor, f64::floor);
sr_unary!(polars__sr_ceil, f64::ceil);
sr_unary!(polars__sr_trunc, f64::trunc);
sr_unary!(polars__sr_sign, f64::signum);
sr_unary!(polars__sr_reciprocal, f64::recip);
sr_unary!(polars__sr_radians, f64::to_radians);
sr_unary!(polars__sr_degrees, f64::to_degrees);

#[no_mangle]
pub extern "C" fn polars__sr_round(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let dp = args.get("dp").and_then(|v| v.as_i64()).unwrap_or(0);
        let mul = 10f64.powi(dp as i32);
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = v.iter().map(|x| (x * mul).round() / mul).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_clip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let lo = args
            .get("min")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let hi = args
            .get("max")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::INFINITY);
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = v.iter().map(|x| x.clamp(lo, hi)).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_between(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let lo = args
            .get("min")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `min`"))?;
        let hi = args
            .get("max")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `max`"))?;
        let v = as_f64_vec(&s)?;
        let out: Vec<bool> = v.iter().map(|x| *x >= lo && *x <= hi).collect();
        let s_out = Series::new(s.name().clone(), out);
        return_series(&s_out)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_isin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let vs = args
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `values`"))?;
        let valset: Vec<f64> = vs.iter().filter_map(|x| x.as_f64()).collect();
        let v = as_f64_vec(&s)?;
        let out: Vec<bool> = v.iter().map(|x| valset.iter().any(|y| y == x)).collect();
        let s_out = Series::new(s.name().clone(), out);
        return_series(&s_out)
    })
}

// ── rolling ─────────────────────────────────────────────────────────────────

fn rolling_window(args: &Value) -> Result<usize> {
    args.get("window")
        .and_then(|v| v.as_u64())
        .map(|w| w as usize)
        .ok_or_else(|| anyhow!("missing argument `window`"))
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_mean(&v, w);
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_apply(&v, w, |w| w.iter().sum::<f64>());
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_apply(&v, w, |w| w.iter().cloned().fold(f64::INFINITY, f64::min));
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_apply(&v, w, |w| {
            w.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        });
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_apply(&v, w, |w| {
            let n = w.len() as f64;
            let m = w.iter().sum::<f64>() / n;
            (w.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt()
        });
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_apply(&v, w, |w| {
            let n = w.len() as f64;
            let m = w.iter().sum::<f64>() / n;
            w.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)
        });
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_median(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_apply(&v, w, |w| {
            let mut sorted: Vec<f64> = w.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        });
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_apply(&v, w, |w| w.len() as f64);
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_rolling_skew(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let w = rolling_window(&args)?;
        let v = as_f64_vec(&s)?;
        let out = rolling_apply(&v, w, |w| {
            let n = w.len() as f64;
            let m = w.iter().sum::<f64>() / n;
            let var = w.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
            let sd = var.sqrt();
            if sd == 0.0 {
                0.0
            } else {
                w.iter().map(|x| ((x - m) / sd).powi(3)).sum::<f64>() / n
            }
        });
        return_series(&from_f64_vec(s.name(), out))
    })
}

fn rolling_apply<F: Fn(&[f64]) -> f64>(v: &[f64], w: usize, f: F) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![f64::NAN; n.min(w.saturating_sub(1))];
    if w == 0 || w > n {
        return vec![f64::NAN; n];
    }
    for i in (w - 1)..n {
        out.push(f(&v[i + 1 - w..=i]));
    }
    out
}

fn rolling_mean(v: &[f64], w: usize) -> Vec<f64> {
    rolling_apply(v, w, |w| w.iter().sum::<f64>() / w.len() as f64)
}

// ── expanding ───────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_expanding_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut sum = 0.0;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                sum += x;
                sum
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_expanding_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut sum = 0.0;
        let out: Vec<f64> = v
            .iter()
            .enumerate()
            .map(|(i, x)| {
                sum += x;
                sum / (i + 1) as f64
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_expanding_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut m = f64::INFINITY;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                m = m.min(*x);
                m
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_expanding_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut m = f64::NEG_INFINITY;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                m = m.max(*x);
                m
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_expanding_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = v.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let win = &v[..=i];
            let m = win.iter().sum::<f64>() / win.len() as f64;
            let var = win.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                / (win.len() as f64 - 1.0).max(1.0);
            out.push(var.sqrt());
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_expanding_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = v.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let win = &v[..=i];
            let m = win.iter().sum::<f64>() / win.len() as f64;
            let var = win.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                / (win.len() as f64 - 1.0).max(1.0);
            out.push(var);
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

// ── ewm ─────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_ewm_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let alpha = args
            .get("alpha")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `alpha`"))?;
        if !(0.0..=1.0).contains(&alpha) {
            bail!("alpha must be in [0, 1]");
        }
        let v = as_f64_vec(&s)?;
        let mut out = Vec::with_capacity(v.len());
        let mut m = 0.0;
        for (i, x) in v.iter().enumerate() {
            if i == 0 {
                m = *x;
            } else {
                m = alpha * x + (1.0 - alpha) * m;
            }
            out.push(m);
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_ewm_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let alpha = args
            .get("alpha")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `alpha`"))?;
        let v = as_f64_vec(&s)?;
        let mut out = Vec::with_capacity(v.len());
        let mut m = 0.0;
        let mut s2 = 0.0;
        for (i, x) in v.iter().enumerate() {
            if i == 0 {
                m = *x;
                s2 = 0.0;
            } else {
                let prev_m = m;
                m = alpha * x + (1.0 - alpha) * m;
                s2 = (1.0 - alpha) * (s2 + alpha * (x - prev_m).powi(2));
            }
            out.push(s2.sqrt());
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_ewm_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let alpha = args
            .get("alpha")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `alpha`"))?;
        let v = as_f64_vec(&s)?;
        let mut out = Vec::with_capacity(v.len());
        let mut m = 0.0;
        let mut s2 = 0.0;
        for (i, x) in v.iter().enumerate() {
            if i == 0 {
                m = *x;
                s2 = 0.0;
            } else {
                let prev_m = m;
                m = alpha * x + (1.0 - alpha) * m;
                s2 = (1.0 - alpha) * (s2 + alpha * (x - prev_m).powi(2));
            }
            out.push(s2);
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

// ── stats: skew / kurt / corr / cov ─────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_skew(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = v.len() as f64;
        if n < 2.0 {
            return Ok(json!({"skew": Value::Null}));
        }
        let m = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let sd = var.sqrt();
        let sk = if sd == 0.0 {
            0.0
        } else {
            v.iter().map(|x| ((x - m) / sd).powi(3)).sum::<f64>() / n
        };
        Ok(json!({"skew": scalar_f64(sk)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_kurt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = v.len() as f64;
        if n < 2.0 {
            return Ok(json!({"kurt": Value::Null}));
        }
        let m = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let sd = var.sqrt();
        let k = if sd == 0.0 {
            0.0
        } else {
            v.iter().map(|x| ((x - m) / sd).powi(4)).sum::<f64>() / n - 3.0
        };
        Ok(json!({"kurt": scalar_f64(k)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_corr(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (a, b) = get_two(&args)?;
        let va = as_f64_vec(&a)?;
        let vb = as_f64_vec(&b)?;
        let n = va.len() as f64;
        let ma = va.iter().sum::<f64>() / n;
        let mb = vb.iter().sum::<f64>() / n;
        let cov = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / n;
        let sda = (va.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / n).sqrt();
        let sdb = (vb.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / n).sqrt();
        let r = if sda * sdb == 0.0 {
            f64::NAN
        } else {
            cov / (sda * sdb)
        };
        Ok(json!({"corr": scalar_f64(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_cov(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (a, b) = get_two(&args)?;
        let va = as_f64_vec(&a)?;
        let vb = as_f64_vec(&b)?;
        let n = va.len() as f64;
        let ma = va.iter().sum::<f64>() / n;
        let mb = vb.iter().sum::<f64>() / n;
        let cov = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        Ok(json!({"cov": scalar_f64(cov)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (a, b) = get_two(&args)?;
        let va = as_f64_vec(&a)?;
        let vb = as_f64_vec(&b)?;
        let p: f64 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
        Ok(json!({"dot": scalar_f64(p)}))
    })
}

// ── concat / repeat / sample ────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_concat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (mut a, b) = get_two(&args)?;
        a.append(&b).context("concat")?;
        return_series(&a)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_repeat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let times = args
            .get("times")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `times`"))? as usize;
        let v = as_f64_vec(&s)?;
        let mut out = Vec::with_capacity(v.len() * times);
        for _ in 0..times {
            out.extend_from_slice(&v);
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_sample(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let with_replacement = args
            .get("with_replacement")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let seed = args.get("seed").and_then(|v| v.as_u64());
        return_series(
            &s.sample_n(n, with_replacement, false, seed)
                .context("sample_n")?,
        )
    })
}

// ── describe / first / last / mode ──────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_first(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"first": any_value_to_json(s.get(0).map_err(|e| anyhow!("get: {e}"))?)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_last(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        if s.is_empty() {
            return Ok(json!({"last": Value::Null}));
        }
        Ok(json!({"last": any_value_to_json(s.get(s.len() - 1).map_err(|e| anyhow!("get: {e}"))?)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_get(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let i = args
            .get("i")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `i`"))? as usize;
        let av = s.get(i).map_err(|e| anyhow!("get: {e}"))?;
        Ok(json!({"value": any_value_to_json(av)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_mode(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for x in &v {
            *counts.entry(x.to_bits()).or_insert(0) += 1;
        }
        let m = counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| f64::from_bits(*k));
        Ok(json!({"mode": m.map(scalar_f64).unwrap_or(Value::Null)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_describe(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let sd = var.sqrt();
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sorted = v.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |p: f64| -> f64 {
            if sorted.is_empty() {
                return f64::NAN;
            }
            let idx = (p * (sorted.len() as f64 - 1.0)).round() as usize;
            sorted[idx]
        };
        Ok(json!({
            "describe": {
                "count": v.len(),
                "mean": scalar_f64(m),
                "std": scalar_f64(sd),
                "min": scalar_f64(mn),
                "25%": scalar_f64(q(0.25)),
                "50%": scalar_f64(q(0.5)),
                "75%": scalar_f64(q(0.75)),
                "max": scalar_f64(mx),
            }
        }))
    })
}

// ── any/all/has ─────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_any(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        Ok(json!({"any": v.iter().any(|x| *x != 0.0)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_all(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        Ok(json!({"all": v.iter().all(|x| *x != 0.0)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_is_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for x in &v {
            *counts.entry(x.to_bits()).or_insert(0) += 1;
        }
        let out: Vec<bool> = v.iter().map(|x| counts[&x.to_bits()] == 1).collect();
        let s_out = Series::new(s.name().clone(), out);
        return_series(&s_out)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_is_duplicated(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for x in &v {
            *counts.entry(x.to_bits()).or_insert(0) += 1;
        }
        let out: Vec<bool> = v.iter().map(|x| counts[&x.to_bits()] > 1).collect();
        let s_out = Series::new(s.name().clone(), out);
        return_series(&s_out)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_has_nulls(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"has_nulls": s.null_count() > 0}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_empty(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        Ok(json!({"empty": s.is_empty()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_to_list(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = s.len();
        let mut arr = Vec::with_capacity(n);
        for i in 0..n {
            let av = s.get(i).map_err(|e| anyhow!("get: {e}"))?;
            arr.push(any_value_to_json(av));
        }
        Ok(json!({"list": arr}))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_to_frame(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = s.len();
        let mut arr = Vec::with_capacity(n);
        for i in 0..n {
            let av = s.get(i).map_err(|e| anyhow!("get: {e}"))?;
            arr.push(any_value_to_json(av));
        }
        let mut m = serde_json::Map::new();
        m.insert(s.name().to_string(), Value::Array(arr));
        Ok(json!({"frame": Value::Object(m)}))
    })
}

// ── set / replace / where / mask / combine ──────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__sr_replace_value(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let from = args
            .get("from")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `from`"))?;
        let to = args
            .get("to")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `to`"))?;
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = v.iter().map(|x| if *x == from { to } else { *x }).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_where(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let mask = args
            .get("mask")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `mask`"))?;
        let other = args
            .get("other")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NAN);
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = v
            .iter()
            .zip(mask.iter())
            .map(|(x, m)| {
                if m.as_bool().unwrap_or(false) {
                    *x
                } else {
                    other
                }
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_mask(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let mask = args
            .get("mask")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `mask`"))?;
        let other = args
            .get("other")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NAN);
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = v
            .iter()
            .zip(mask.iter())
            .map(|(x, m)| {
                if m.as_bool().unwrap_or(false) {
                    other
                } else {
                    *x
                }
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_combine_first(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (a, b) = get_two(&args)?;
        let va = as_f64_vec(&a)?;
        let vb = as_f64_vec(&b)?;
        let out: Vec<f64> = va
            .iter()
            .zip(vb.iter())
            .map(|(x, y)| if x.is_nan() { *y } else { *x })
            .collect();
        return_series(&from_f64_vec(a.name(), out))
    })
}

// ── str accessor (sr_str_*) ────────────────────────────────────────────────

fn as_str_vec(s: &Series) -> Result<Vec<Option<String>>> {
    let cast = s.cast(&DataType::String).context("cast str")?;
    let ca = cast.str().context("not str")?;
    Ok(ca.into_iter().map(|x| x.map(String::from)).collect())
}

fn from_str_vec(name: &str, data: Vec<Option<String>>) -> Series {
    Series::new(name.into(), data)
}

#[no_mangle]
pub extern "C" fn polars__sr_str_lower(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v.into_iter().map(|x| x.map(|s| s.to_lowercase())).collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_upper(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v.into_iter().map(|x| x.map(|s| s.to_uppercase())).collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_len(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<i64>> = v
            .into_iter()
            .map(|x| x.map(|s| s.chars().count() as i64))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_strip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| x.map(|s| s.trim().to_string()))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_lstrip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| x.map(|s| s.trim_start().to_string()))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_rstrip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| x.map(|s| s.trim_end().to_string()))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_contains(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let pat = args
            .get("pat")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `pat`"))?
            .to_string();
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v.into_iter().map(|x| x.map(|s| s.contains(&pat))).collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_startswith(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let pat = args
            .get("pat")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `pat`"))?
            .to_string();
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| x.map(|s| s.starts_with(&pat)))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_endswith(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let pat = args
            .get("pat")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `pat`"))?
            .to_string();
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| x.map(|s| s.ends_with(&pat)))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_replace(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let from = args
            .get("from")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `from`"))?
            .to_string();
        let to = args
            .get("to")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `to`"))?
            .to_string();
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| x.map(|s| s.replacen(&from, &to, 1)))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_replace_all(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let from = args
            .get("from")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `from`"))?
            .to_string();
        let to = args
            .get("to")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `to`"))?
            .to_string();
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| x.map(|s| s.replace(&from, &to)))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_slice(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let start = args.get("start").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
        let length = args
            .get("length")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize);
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    let chars: Vec<char> = s.chars().collect();
                    let end = match length {
                        Some(l) => (start + l).min(chars.len()),
                        None => chars.len(),
                    };
                    if start >= chars.len() {
                        String::new()
                    } else {
                        chars[start..end].iter().collect()
                    }
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| x.map(|s| s.chars().rev().collect::<String>()))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let pat = args
            .get("pat")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `pat`"))?
            .to_string();
        let v = as_str_vec(&s)?;
        let out: Vec<Option<i64>> = v
            .into_iter()
            .map(|x| x.map(|s| s.matches(&pat).count() as i64))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_title(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    s.split_whitespace()
                        .map(|w| {
                            let mut c = w.chars();
                            match c.next() {
                                None => String::new(),
                                Some(f) => {
                                    f.to_uppercase().collect::<String>()
                                        + &c.as_str().to_lowercase()
                                }
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_capitalize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    let mut c = s.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => {
                            f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase()
                        }
                    }
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_swapcase(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    s.chars()
                        .map(|c| {
                            if c.is_uppercase() {
                                c.to_lowercase().next().unwrap_or(c)
                            } else {
                                c.to_uppercase().next().unwrap_or(c)
                            }
                        })
                        .collect()
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_pad(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let width = args
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `width`"))? as usize;
        let fill = args
            .get("fill")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    let n = s.chars().count();
                    if n >= width {
                        s
                    } else {
                        let pad: String = std::iter::repeat_n(fill, width - n).collect();
                        format!("{pad}{s}")
                    }
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_zfill(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let width = args
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `width`"))? as usize;
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    let n = s.chars().count();
                    if n >= width {
                        s
                    } else {
                        let pad: String = std::iter::repeat_n('0', width - n).collect();
                        format!("{pad}{s}")
                    }
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_find(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let pat = args
            .get("pat")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `pat`"))?
            .to_string();
        let v = as_str_vec(&s)?;
        let out: Vec<Option<i64>> = v
            .into_iter()
            .map(|x| x.map(|s| s.find(&pat).map(|i| i as i64).unwrap_or(-1)))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_concat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let suffix = args
            .get("suffix")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let v = as_str_vec(&s)?;
        let out = v
            .into_iter()
            .map(|x| x.map(|s| format!("{s}{suffix}")))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_isalpha(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| x.map(|s| !s.is_empty() && s.chars().all(|c| c.is_alphabetic())))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_isdigit(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| x.map(|s| !s.is_empty() && s.chars().all(|c| c.is_numeric())))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_isalnum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| x.map(|s| !s.is_empty() && s.chars().all(|c| c.is_alphanumeric())))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_isspace(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| x.map(|s| !s.is_empty() && s.chars().all(|c| c.is_whitespace())))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_isupper(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    !s.is_empty() && s.chars().all(|c| c.is_uppercase() || !c.is_alphabetic())
                })
            })
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_islower(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    !s.is_empty() && s.chars().all(|c| c.is_lowercase() || !c.is_alphabetic())
                })
            })
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_repeat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing argument `n`"))? as usize;
        let v = as_str_vec(&s)?;
        let out = v.into_iter().map(|x| x.map(|s| s.repeat(n))).collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_to_int(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<i64>> = v
            .into_iter()
            .map(|x| x.and_then(|s| s.trim().parse().ok()))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_str_to_float(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<f64>> = v
            .into_iter()
            .map(|x| x.and_then(|s| s.trim().parse().ok()))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

// ── dt accessor (sr_dt_*) ───────────────────────────────────────────────────
//
// These take a series of "YYYY-MM-DD[ HH:MM:SS]" strings and return integer
// component series. Lightweight: parse on the fly with chrono.

fn dt_extract<F: Fn(chrono::NaiveDateTime) -> i64>(args: &Value, f: F) -> Result<Value> {
    let s = get_series(args)?;
    let v = as_str_vec(&s)?;
    let out: Vec<Option<i64>> = v
        .into_iter()
        .map(|x| {
            x.and_then(|st| {
                chrono::NaiveDateTime::parse_from_str(&st, "%Y-%m-%d %H:%M:%S")
                    .ok()
                    .or_else(|| {
                        chrono::NaiveDate::parse_from_str(&st, "%Y-%m-%d")
                            .ok()
                            .map(|d| d.and_hms_opt(0, 0, 0).unwrap_or_default())
                    })
                    .map(&f)
            })
        })
        .collect();
    let series = Series::new(s.name().clone(), out);
    Ok(json!({"series": series_to_value(&series)?}))
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_year(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Datelike;
            d.year() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_month(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Datelike;
            d.month() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_day(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Datelike;
            d.day() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_hour(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Timelike;
            d.hour() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_minute(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Timelike;
            d.minute() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_second(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Timelike;
            d.second() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_weekday(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Datelike;
            d.weekday().num_days_from_monday() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_dayofyear(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Datelike;
            d.ordinal() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_quarter(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Datelike;
            ((d.month() - 1) / 3 + 1) as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_week(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Datelike;
            d.iso_week().week() as i64
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_is_leap(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<bool>> = v
            .into_iter()
            .map(|x| {
                x.and_then(|st| {
                    chrono::NaiveDate::parse_from_str(&st, "%Y-%m-%d")
                        .ok()
                        .map(|d| {
                            use chrono::Datelike;
                            let y = d.year();
                            (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
                        })
                })
            })
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_days_in_month(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        dt_extract(&args, |d| {
            use chrono::Datelike;
            let (y, m) = (d.year(), d.month());
            let next = if m == 12 {
                chrono::NaiveDate::from_ymd_opt(y + 1, 1, 1).unwrap()
            } else {
                chrono::NaiveDate::from_ymd_opt(y, m + 1, 1).unwrap()
            };
            let this = chrono::NaiveDate::from_ymd_opt(y, m, 1).unwrap();
            next.signed_duration_since(this).num_days()
        })
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_timestamp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| dt_extract(&args, |d| d.and_utc().timestamp()))
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_date(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<String>> = v
            .into_iter()
            .map(|x| {
                x.and_then(|st| {
                    chrono::NaiveDateTime::parse_from_str(&st, "%Y-%m-%d %H:%M:%S")
                        .ok()
                        .map(|d| d.date().to_string())
                        .or_else(|| {
                            chrono::NaiveDate::parse_from_str(&st, "%Y-%m-%d")
                                .ok()
                                .map(|d| d.to_string())
                        })
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

#[no_mangle]
pub extern "C" fn polars__sr_dt_time(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<String>> = v
            .into_iter()
            .map(|x| {
                x.and_then(|st| {
                    chrono::NaiveDateTime::parse_from_str(&st, "%Y-%m-%d %H:%M:%S")
                        .ok()
                        .map(|d| d.time().to_string())
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}
