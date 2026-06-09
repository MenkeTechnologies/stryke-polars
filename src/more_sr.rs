//! src/more_sr.rs — additional Series ops (`polars__sr_*`) to round out
//! the surface beyond the 1k-fn mark.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Context, Result};
use polars::prelude::*;
use serde_json::{json, Value};

use crate::ffi_call;

fn parse_series(v: &Value) -> Result<Series> {
    let name = v
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("s")
        .to_string();
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("series `data` missing"))?;
    if data.is_empty() {
        return Ok(Series::new_empty(name.into(), &DataType::Null));
    }
    let (mut all_bool, mut all_i64, mut all_f64, mut all_str) = (true, true, true, true);
    for v in data {
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
        let v: Vec<Option<bool>> = data.iter().map(|x| x.as_bool()).collect();
        Ok(Series::new(name.into(), v))
    } else if all_i64 {
        let v: Vec<Option<i64>> = data.iter().map(|x| x.as_i64()).collect();
        Ok(Series::new(name.into(), v))
    } else if all_f64 {
        let v: Vec<Option<f64>> = data.iter().map(|x| x.as_f64()).collect();
        Ok(Series::new(name.into(), v))
    } else if all_str {
        let v: Vec<Option<String>> = data.iter().map(|x| x.as_str().map(String::from)).collect();
        Ok(Series::new(name.into(), v))
    } else {
        let v: Vec<Option<String>> = data
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

fn return_series(s: &Series) -> Result<Value> {
    Ok(json!({"series": series_to_value(s)?}))
}

fn as_f64_vec(s: &Series) -> Result<Vec<f64>> {
    let f = s.cast(&DataType::Float64).context("cast f64")?;
    let ca = f.f64().context("not f64")?;
    Ok(ca.into_no_null_iter().collect())
}

fn from_f64_vec(name: &str, data: Vec<f64>) -> Series {
    Series::new(name.into(), data)
}

fn scalar(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn as_str_vec(s: &Series) -> Result<Vec<Option<String>>> {
    let cast = s.cast(&DataType::String).context("cast str")?;
    let ca = cast.str().context("not str")?;
    Ok(ca.into_iter().map(|x| x.map(String::from)).collect())
}

fn from_str_vec(name: &str, data: Vec<Option<String>>) -> Series {
    Series::new(name.into(), data)
}

// ── more series ops ────────────────────────────────────────────────────────

/// Series argunique.
#[no_mangle]
pub extern "C" fn polars__sr_argunique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut seen = std::collections::HashSet::new();
        let out: Vec<i64> = v
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if seen.insert(x.to_bits()) {
                    Some(i as i64)
                } else {
                    None
                }
            })
            .collect();
        Ok(json!({"indices": out}))
    })
}

/// Series drop duplicates.
#[no_mangle]
pub extern "C" fn polars__sr_drop_duplicates(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut seen = std::collections::HashSet::new();
        let out: Vec<f64> = v.into_iter().filter(|x| seen.insert(x.to_bits())).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series argmax n.
#[no_mangle]
pub extern "C" fn polars__sr_argmax_n(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mut pairs: Vec<(usize, f64)> = v.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<i64> = pairs.into_iter().take(n).map(|(i, _)| i as i64).collect();
        Ok(json!({"argmax_n": out}))
    })
}

/// Series argmin n.
#[no_mangle]
pub extern "C" fn polars__sr_argmin_n(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mut pairs: Vec<(usize, f64)> = v.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<i64> = pairs.into_iter().take(n).map(|(i, _)| i as i64).collect();
        Ok(json!({"argmin_n": out}))
    })
}

/// Series nlargest.
#[no_mangle]
pub extern "C" fn polars__sr_nlargest(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let mut v = as_f64_vec(&s)?;
        v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<f64> = v.into_iter().take(n).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series nsmallest.
#[no_mangle]
pub extern "C" fn polars__sr_nsmallest(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let mut v = as_f64_vec(&s)?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<f64> = v.into_iter().take(n).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series value range.
#[no_mangle]
pub extern "C" fn polars__sr_value_range(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Ok(json!({"range": [scalar(mn), scalar(mx)]}))
    })
}

/// Series first valid index.
#[no_mangle]
pub extern "C" fn polars__sr_first_valid_index(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let r = (0..s.len())
            .find(|i| {
                s.get(*i)
                    .ok()
                    .map(|av| !matches!(av, AnyValue::Null))
                    .unwrap_or(false)
            })
            .map(|x| x as i64);
        Ok(json!({"first_valid_index": r}))
    })
}

/// Series last valid index.
#[no_mangle]
pub extern "C" fn polars__sr_last_valid_index(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = s.len();
        let r = (0..n)
            .rev()
            .find(|i| {
                s.get(*i)
                    .ok()
                    .map(|av| !matches!(av, AnyValue::Null))
                    .unwrap_or(false)
            })
            .map(|x| x as i64);
        Ok(json!({"last_valid_index": r}))
    })
}

/// Series to dict.
#[no_mangle]
pub extern "C" fn polars__sr_to_dict(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let mut m = serde_json::Map::new();
        for i in 0..s.len() {
            let av = s.get(i).map_err(|e| anyhow!("get: {e}"))?;
            m.insert(i.to_string(), any_value_to_json(av));
        }
        Ok(json!({"dict": Value::Object(m)}))
    })
}

/// Series apply lambda.
#[no_mangle]
pub extern "C" fn polars__sr_apply_lambda(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let op = args
            .get("op")
            .and_then(|v| v.as_str())
            .unwrap_or("identity");
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = match op {
            "double" => v.iter().map(|x| x * 2.0).collect(),
            "half" => v.iter().map(|x| x / 2.0).collect(),
            "negate" => v.iter().map(|x| -x).collect(),
            "square" => v.iter().map(|x| x * x).collect(),
            "increment" => v.iter().map(|x| x + 1.0).collect(),
            _ => v,
        };
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series interpolate.
#[no_mangle]
pub extern "C" fn polars__sr_interpolate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut out = v.clone();
        let n = out.len();
        let mut last_valid: Option<(usize, f64)> = None;
        for i in 0..n {
            if !out[i].is_nan() {
                if let Some((j, last)) = last_valid {
                    if j + 1 < i {
                        let step = (out[i] - last) / (i - j) as f64;
                        for (offset, val) in out[j + 1..i].iter_mut().enumerate() {
                            *val = last + step * (offset + 1) as f64;
                        }
                    }
                }
                last_valid = Some((i, out[i]));
            }
        }
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series pad value.
#[no_mangle]
pub extern "C" fn polars__sr_pad_value(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let length = args
            .get("length")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `length`"))? as usize;
        let value = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mut v = as_f64_vec(&s)?;
        while v.len() < length {
            v.push(value);
        }
        return_series(&from_f64_vec(s.name(), v))
    })
}

/// Series truncate.
#[no_mangle]
pub extern "C" fn polars__sr_truncate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let length = args
            .get("length")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `length`"))? as usize;
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = v.into_iter().take(length).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series normalize.
#[no_mangle]
pub extern "C" fn polars__sr_normalize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let r = mx - mn;
        let out: Vec<f64> = if r == 0.0 {
            vec![0.0; v.len()]
        } else {
            v.iter().map(|x| (x - mn) / r).collect()
        };
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series standardize.
#[no_mangle]
pub extern "C" fn polars__sr_standardize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n.max(1.0);
        let sd = var.sqrt();
        let out: Vec<f64> = if sd == 0.0 {
            vec![0.0; v.len()]
        } else {
            v.iter().map(|x| (x - m) / sd).collect()
        };
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series softmax.
#[no_mangle]
pub extern "C" fn polars__sr_softmax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = v.iter().map(|x| (x - mx).exp()).collect();
        let s_sum: f64 = exps.iter().sum();
        let out: Vec<f64> = exps.iter().map(|x| x / s_sum).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series logsumexp.
#[no_mangle]
pub extern "C" fn polars__sr_logsumexp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let r = mx + v.iter().map(|x| (x - mx).exp()).sum::<f64>().ln();
        Ok(json!({"logsumexp": scalar(r)}))
    })
}

/// Series normalize l2.
#[no_mangle]
pub extern "C" fn polars__sr_normalize_l2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let out: Vec<f64> = if norm == 0.0 {
            vec![0.0; v.len()]
        } else {
            v.iter().map(|x| x / norm).collect()
        };
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series normalize l1.
#[no_mangle]
pub extern "C" fn polars__sr_normalize_l1(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let norm: f64 = v.iter().map(|x| x.abs()).sum();
        let out: Vec<f64> = if norm == 0.0 {
            vec![0.0; v.len()]
        } else {
            v.iter().map(|x| x / norm).collect()
        };
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series dropna first.
#[no_mangle]
pub extern "C" fn polars__sr_dropna_first(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = s.len();
        let mut start = 0;
        for i in 0..n {
            if matches!(s.get(i).unwrap_or(AnyValue::Null), AnyValue::Null) {
                start = i + 1;
            } else {
                break;
            }
        }
        return_series(&s.slice(start as i64, n - start))
    })
}

/// Series dropna last.
#[no_mangle]
pub extern "C" fn polars__sr_dropna_last(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = s.len();
        let mut end = n;
        for i in (0..n).rev() {
            if matches!(s.get(i).unwrap_or(AnyValue::Null), AnyValue::Null) {
                end = i;
            } else {
                break;
            }
        }
        return_series(&s.slice(0, end))
    })
}

/// Series argbottom.
#[no_mangle]
pub extern "C" fn polars__sr_argbottom(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let mut pairs: Vec<(usize, f64)> = v.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<i64> = pairs.into_iter().take(n).map(|(i, _)| i as i64).collect();
        Ok(json!({"argbottom": out}))
    })
}

/// Series argtop.
#[no_mangle]
pub extern "C" fn polars__sr_argtop(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let mut pairs: Vec<(usize, f64)> = v.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<i64> = pairs.into_iter().take(n).map(|(i, _)| i as i64).collect();
        Ok(json!({"argtop": out}))
    })
}

/// Series top k.
#[no_mangle]
pub extern "C" fn polars__sr_top_k(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let mut v = as_f64_vec(&s)?;
        v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<f64> = v.into_iter().take(n).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series bottom k.
#[no_mangle]
pub extern "C" fn polars__sr_bottom_k(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let mut v = as_f64_vec(&s)?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<f64> = v.into_iter().take(n).collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series to numpy.
#[no_mangle]
pub extern "C" fn polars__sr_to_numpy(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let n = v.len();
        Ok(json!({"array": {"data": v, "shape": [n]}}))
    })
}

/// Series from numpy.
#[no_mangle]
pub extern "C" fn polars__sr_from_numpy(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("array")
            .ok_or_else(|| anyhow!("missing `array`"))?;
        let data = a
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| anyhow!("missing array.data"))?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("s")
            .to_string();
        let v: Vec<f64> = data
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN))
            .collect();
        return_series(&from_f64_vec(&name, v))
    })
}

/// Series str at.
#[no_mangle]
pub extern "C" fn polars__sr_str_at(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let i = args
            .get("i")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `i`"))? as usize;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<String>> = v
            .into_iter()
            .map(|x| x.map(|s| s.chars().nth(i).map(|c| c.to_string()).unwrap_or_default()))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

/// Series str first.
#[no_mangle]
pub extern "C" fn polars__sr_str_first(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<String>> = v
            .into_iter()
            .map(|x| x.map(|s| s.chars().next().map(|c| c.to_string()).unwrap_or_default()))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

/// Series str last.
#[no_mangle]
pub extern "C" fn polars__sr_str_last(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_str_vec(&s)?;
        let out: Vec<Option<String>> = v
            .into_iter()
            .map(|x| x.map(|s| s.chars().last().map(|c| c.to_string()).unwrap_or_default()))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

/// Series str split n.
#[no_mangle]
pub extern "C" fn polars__sr_str_split_n(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let sep = args
            .get("sep")
            .and_then(|v| v.as_str())
            .unwrap_or(" ")
            .to_string();
        let v = as_str_vec(&s)?;
        let out: Vec<Option<i64>> = v
            .into_iter()
            .map(|x| x.map(|s| s.split(&sep as &str).count() as i64))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

/// Series str join.
#[no_mangle]
pub extern "C" fn polars__sr_str_join(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let sep = args
            .get("sep")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let v = as_str_vec(&s)?;
        let parts: Vec<String> = v.into_iter().flatten().collect();
        Ok(json!({"joined": parts.join(&sep)}))
    })
}

/// Series str strip chars.
#[no_mangle]
pub extern "C" fn polars__sr_str_strip_chars(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let chars = args
            .get("chars")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let v = as_str_vec(&s)?;
        let out: Vec<Option<String>> = v
            .into_iter()
            .map(|x| x.map(|s| s.trim_matches(|c| chars.contains(c)).to_string()))
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

/// Series str pad right.
#[no_mangle]
pub extern "C" fn polars__sr_str_pad_right(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let width = args
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `width`"))? as usize;
        let fill = args
            .get("fill")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');
        let v = as_str_vec(&s)?;
        let out: Vec<Option<String>> = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    let n = s.chars().count();
                    if n >= width {
                        s
                    } else {
                        let pad: String = std::iter::repeat_n(fill, width - n).collect();
                        format!("{s}{pad}")
                    }
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

/// Series str center.
#[no_mangle]
pub extern "C" fn polars__sr_str_center(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let width = args
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `width`"))? as usize;
        let fill = args
            .get("fill")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');
        let v = as_str_vec(&s)?;
        let out: Vec<Option<String>> = v
            .into_iter()
            .map(|x| {
                x.map(|s| {
                    let n = s.chars().count();
                    if n >= width {
                        s
                    } else {
                        let total = width - n;
                        let left = total / 2;
                        let right = total - left;
                        let lp: String = std::iter::repeat_n(fill, left).collect();
                        let rp: String = std::iter::repeat_n(fill, right).collect();
                        format!("{lp}{s}{rp}")
                    }
                })
            })
            .collect();
        return_series(&from_str_vec(s.name(), out))
    })
}

/// Series quantile method.
#[no_mangle]
pub extern "C" fn polars__sr_quantile_method(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let q = args
            .get("q")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `q`"))?;
        let method = args
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("linear");
        let m = match method {
            "lower" => QuantileMethod::Lower,
            "higher" => QuantileMethod::Higher,
            "nearest" => QuantileMethod::Nearest,
            "midpoint" => QuantileMethod::Midpoint,
            _ => QuantileMethod::Linear,
        };
        let r = s.quantile_reduce(q, m).context("quantile")?;
        Ok(json!({"quantile": format!("{}", r.value())}))
    })
}

/// Series cut.
#[no_mangle]
pub extern "C" fn polars__sr_cut(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let bins = args
            .get("bins")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `bins`"))?;
        let b: Vec<f64> = bins.iter().filter_map(|x| x.as_f64()).collect();
        let v = as_f64_vec(&s)?;
        let out: Vec<i64> = v
            .iter()
            .map(|x| b.iter().position(|t| t > x).unwrap_or(b.len()) as i64)
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

/// Series qcut.
#[no_mangle]
pub extern "C" fn polars__sr_qcut(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let q = args
            .get("q")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `q`"))? as usize;
        let mut sorted = as_f64_vec(&s)?;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if sorted.is_empty() {
            return return_series(&from_f64_vec(s.name(), vec![]));
        }
        let breaks: Vec<f64> = (1..q)
            .map(|i| sorted[(i * sorted.len() / q).min(sorted.len() - 1)])
            .collect();
        let v = as_f64_vec(&s)?;
        let out: Vec<i64> = v
            .iter()
            .map(|x| breaks.iter().position(|t| t > x).unwrap_or(breaks.len()) as i64)
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

/// Series pivot count.
#[no_mangle]
pub extern "C" fn polars__sr_pivot_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for x in &v {
            *counts.entry(x.to_bits()).or_insert(0) += 1;
        }
        let mut out: Vec<(f64, u64)> = counts
            .into_iter()
            .map(|(k, v)| (f64::from_bits(k), v))
            .collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let arr: Vec<Value> = out
            .into_iter()
            .map(|(k, v)| json!({"value": scalar(k), "count": v}))
            .collect();
        Ok(json!({"pivot": arr}))
    })
}

/// Series div safe.
#[no_mangle]
pub extern "C" fn polars__sr_div_safe(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let scalar_v = args
            .get("scalar")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `scalar`"))?;
        let default = args.get("default").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                if scalar_v == 0.0 {
                    default
                } else {
                    x / scalar_v
                }
            })
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series swap.
#[no_mangle]
pub extern "C" fn polars__sr_swap(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let i = args
            .get("i")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `i`"))? as usize;
        let j = args
            .get("j")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `j`"))? as usize;
        let mut v = as_f64_vec(&s)?;
        if i < v.len() && j < v.len() {
            v.swap(i, j);
        }
        return_series(&from_f64_vec(s.name(), v))
    })
}

/// Series take.
#[no_mangle]
pub extern "C" fn polars__sr_take(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let idx = args
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `indices`"))?;
        let v = as_f64_vec(&s)?;
        let out: Vec<f64> = idx
            .iter()
            .filter_map(|i| i.as_u64())
            .filter_map(|i| v.get(i as usize).copied())
            .collect();
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series set.
#[no_mangle]
pub extern "C" fn polars__sr_set(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let i = args
            .get("i")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `i`"))? as usize;
        let value = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut v = as_f64_vec(&s)?;
        if i < v.len() {
            v[i] = value;
        }
        return_series(&from_f64_vec(s.name(), v))
    })
}

/// Series append.
#[no_mangle]
pub extern "C" fn polars__sr_append(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let value = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut v = as_f64_vec(&s)?;
        v.push(value);
        return_series(&from_f64_vec(s.name(), v))
    })
}

/// Series prepend.
#[no_mangle]
pub extern "C" fn polars__sr_prepend(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let value = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let v = as_f64_vec(&s)?;
        let mut out = vec![value];
        out.extend(v);
        return_series(&from_f64_vec(s.name(), out))
    })
}

/// Series insert.
#[no_mangle]
pub extern "C" fn polars__sr_insert(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let i = args
            .get("i")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `i`"))? as usize;
        let value = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut v = as_f64_vec(&s)?;
        if i > v.len() {
            bail!("index out of bounds");
        }
        v.insert(i, value);
        return_series(&from_f64_vec(s.name(), v))
    })
}

/// Series remove.
#[no_mangle]
pub extern "C" fn polars__sr_remove(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let i = args
            .get("i")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `i`"))? as usize;
        let mut v = as_f64_vec(&s)?;
        if i < v.len() {
            v.remove(i);
        }
        return_series(&from_f64_vec(s.name(), v))
    })
}

/// Series count value.
#[no_mangle]
pub extern "C" fn polars__sr_count_value(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let value = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let v = as_f64_vec(&s)?;
        let c = v.iter().filter(|x| **x == value).count();
        Ok(json!({"count": c}))
    })
}

/// Series count between.
#[no_mangle]
pub extern "C" fn polars__sr_count_between(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let lo = args
            .get("min")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `min`"))?;
        let hi = args
            .get("max")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `max`"))?;
        let v = as_f64_vec(&s)?;
        let c = v.iter().filter(|x| **x >= lo && **x <= hi).count();
        Ok(json!({"count": c}))
    })
}

/// Series count greater.
#[no_mangle]
pub extern "C" fn polars__sr_count_greater(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v_t = args
            .get("threshold")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `threshold`"))?;
        let v = as_f64_vec(&s)?;
        let c = v.iter().filter(|x| **x > v_t).count();
        Ok(json!({"count": c}))
    })
}

/// Series count less.
#[no_mangle]
pub extern "C" fn polars__sr_count_less(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v_t = args
            .get("threshold")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `threshold`"))?;
        let v = as_f64_vec(&s)?;
        let c = v.iter().filter(|x| **x < v_t).count();
        Ok(json!({"count": c}))
    })
}

/// Series count nan.
#[no_mangle]
pub extern "C" fn polars__sr_count_nan(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let c = v.iter().filter(|x| x.is_nan()).count();
        Ok(json!({"count_nan": c}))
    })
}

/// Series count finite.
#[no_mangle]
pub extern "C" fn polars__sr_count_finite(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let c = v.iter().filter(|x| x.is_finite()).count();
        Ok(json!({"count_finite": c}))
    })
}

/// Series count inf.
#[no_mangle]
pub extern "C" fn polars__sr_count_inf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let c = v.iter().filter(|x| x.is_infinite()).count();
        Ok(json!({"count_inf": c}))
    })
}

/// Series is sorted.
#[no_mangle]
pub extern "C" fn polars__sr_is_sorted(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let r = v.windows(2).all(|w| w[0] <= w[1]);
        Ok(json!({"is_sorted": r}))
    })
}

/// Series is sorted desc.
#[no_mangle]
pub extern "C" fn polars__sr_is_sorted_desc(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let r = v.windows(2).all(|w| w[0] >= w[1]);
        Ok(json!({"is_sorted_desc": r}))
    })
}

/// Series partition.
#[no_mangle]
pub extern "C" fn polars__sr_partition(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `k`"))? as usize;
        let mut v = as_f64_vec(&s)?;
        if k >= v.len() {
            return return_series(&from_f64_vec(s.name(), v));
        }
        v.select_nth_unstable_by(k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        return_series(&from_f64_vec(s.name(), v))
    })
}

/// Series to int.
#[no_mangle]
pub extern "C" fn polars__sr_to_int(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let out: Vec<i64> = v.iter().map(|x| *x as i64).collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

/// Series to bool.
#[no_mangle]
pub extern "C" fn polars__sr_to_bool(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let out: Vec<bool> = v.iter().map(|x| *x != 0.0).collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

/// Series to str.
#[no_mangle]
pub extern "C" fn polars__sr_to_str(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let out: Vec<String> = v.iter().map(|x| x.to_string()).collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

/// Series cumcount.
#[no_mangle]
pub extern "C" fn polars__sr_cumcount(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let n = s.len();
        let out: Vec<i64> = (0..n as i64).collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

/// Series rank dense.
#[no_mangle]
pub extern "C" fn polars__sr_rank_dense(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let v = as_f64_vec(&s)?;
        let mut sorted_unique: Vec<f64> = v.clone();
        sorted_unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_unique.dedup();
        let ranks: std::collections::HashMap<u64, i64> = sorted_unique
            .iter()
            .enumerate()
            .map(|(i, v)| (v.to_bits(), (i + 1) as i64))
            .collect();
        let out: Vec<i64> = v
            .iter()
            .map(|x| *ranks.get(&x.to_bits()).unwrap_or(&0))
            .collect();
        let series = Series::new(s.name().clone(), out);
        return_series(&series)
    })
}

/// Series to index.
#[no_mangle]
pub extern "C" fn polars__sr_to_index(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_series(&args)?;
        let mut arr = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let av = s.get(i).map_err(|e| anyhow!("get: {e}"))?;
            arr.push(any_value_to_json(av));
        }
        Ok(json!({"index": {"data": arr}}))
    })
}
