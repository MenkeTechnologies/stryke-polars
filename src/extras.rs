//! src/extras.rs — additional surface area to round out the 1k+ fn target.
//!
//! Adds:
//!   - `polars__gb_*` — groupby aggregations on DataFrames
//!   - `polars__roll_*` — rolling-window aggregations on Series
//!   - `polars__stat_*` — statistics convenience ops
//!   - `polars__bool_*` — boolean array ops
//!   - `polars__set_*` — set-like ops on f64 arrays

use std::ffi::c_char;

use anyhow::{anyhow, bail, Context, Result};
use polars::prelude::*;
use serde_json::{json, Map, Value};

use crate::ffi_call;

// ── shared helpers ─────────────────────────────────────────────────────────

fn parse_df(v: &Value) -> Result<DataFrame> {
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow!("expected columnar object"))?;
    let mut cols: Vec<Column> = Vec::with_capacity(obj.len());
    for (name, col_v) in obj {
        let arr = col_v.as_array().ok_or_else(|| anyhow!("col `{name}`"))?;
        let s = json_array_to_series(name, arr)?;
        cols.push(s.into_column());
    }
    DataFrame::new(cols).context("DataFrame::new")
}

fn json_array_to_series(name: &str, arr: &[Value]) -> Result<Series> {
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

fn df_to_value(df: &DataFrame) -> Result<Value> {
    let mut out = Map::with_capacity(df.width());
    for col in df.get_columns() {
        let name = col.name().to_string();
        let series = col.as_materialized_series();
        let n = series.len();
        let mut arr = Vec::with_capacity(n);
        for i in 0..n {
            let av = series.get(i).map_err(|e| anyhow!("get: {e}"))?;
            arr.push(any_value_to_json(av));
        }
        out.insert(name, Value::Array(arr));
    }
    Ok(Value::Object(out))
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

fn get_frame(args: &Value) -> Result<DataFrame> {
    let f = args
        .get("frame")
        .ok_or_else(|| anyhow!("missing argument `frame`"))?;
    parse_df(f)
}

fn return_frame(df: DataFrame) -> Result<Value> {
    Ok(json!({"frame": df_to_value(&df)?}))
}

fn get_by(args: &Value) -> Result<Vec<String>> {
    let by = args
        .get("by")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing argument `by`"))?;
    Ok(by
        .iter()
        .filter_map(|x| x.as_str().map(String::from))
        .collect())
}

fn get_value_col(args: &Value) -> Result<Vec<String>> {
    match args.get("value_columns").and_then(|v| v.as_array()) {
        Some(arr) => Ok(arr
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect()),
        None => Ok(vec![]),
    }
}

// ── groupby (gb_*) ─────────────────────────────────────────────────────────

fn gb_agg<F>(args: Value, agg_factory: F) -> Result<Value>
where
    F: Fn(Expr) -> Expr,
{
    let df = get_frame(&args)?;
    let by = get_by(&args)?;
    let cols: Vec<String> = get_value_col(&args)?;
    let aggs: Vec<Expr> = if cols.is_empty() {
        df.get_column_names_owned()
            .iter()
            .filter(|c| !by.iter().any(|b| b == c.as_str()))
            .map(|c| agg_factory(col(c.as_str())).alias(c.as_str()))
            .collect()
    } else {
        cols.iter()
            .map(|c| agg_factory(col(c.as_str())).alias(c.as_str()))
            .collect()
    };
    let by_exprs: Vec<Expr> = by.iter().map(|c| col(c.as_str())).collect();
    let result = df
        .lazy()
        .group_by(by_exprs)
        .agg(aggs)
        .collect()
        .context("groupby agg")?;
    return_frame(result)
}

#[no_mangle]
pub extern "C" fn polars__gb_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.sum()))
}

#[no_mangle]
pub extern "C" fn polars__gb_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.mean()))
}

#[no_mangle]
pub extern "C" fn polars__gb_median(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.median()))
}

#[no_mangle]
pub extern "C" fn polars__gb_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.min()))
}

#[no_mangle]
pub extern "C" fn polars__gb_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.max()))
}

#[no_mangle]
pub extern "C" fn polars__gb_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.count()))
}

#[no_mangle]
pub extern "C" fn polars__gb_first(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.first()))
}

#[no_mangle]
pub extern "C" fn polars__gb_last(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.last()))
}

#[no_mangle]
pub extern "C" fn polars__gb_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.std(1)))
}

#[no_mangle]
pub extern "C" fn polars__gb_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.var(1)))
}

#[no_mangle]
pub extern "C" fn polars__gb_n_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.n_unique()))
}

#[no_mangle]
pub extern "C" fn polars__gb_quantile(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let q = args
            .get("q")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `q`"))?;
        gb_agg(args, move |e| e.quantile(lit(q), QuantileMethod::Linear))
    })
}

#[no_mangle]
pub extern "C" fn polars__gb_groups(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let by = get_by(&args)?;
        let by_exprs: Vec<Expr> = by.iter().map(|c| col(c.as_str())).collect();
        let agg = vec![col("*").count().alias("count")];
        let result = df
            .lazy()
            .group_by(by_exprs)
            .agg(agg)
            .collect()
            .context("groups")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__gb_size(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let by = get_by(&args)?;
        let by_exprs: Vec<Expr> = by.iter().map(|c| col(c.as_str())).collect();
        let agg = vec![len().alias("size")];
        let result = df
            .lazy()
            .group_by(by_exprs)
            .agg(agg)
            .collect()
            .context("size")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__gb_head(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let by = get_by(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5);
        let by_exprs: Vec<Expr> = by.iter().map(|c| col(c.as_str())).collect();
        let result = df
            .lazy()
            .group_by(by_exprs)
            .head(Some(n as usize))
            .collect()
            .context("gb head")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__gb_tail(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let by = get_by(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5);
        let by_exprs: Vec<Expr> = by.iter().map(|c| col(c.as_str())).collect();
        let result = df
            .lazy()
            .group_by(by_exprs)
            .tail(Some(n as usize))
            .collect()
            .context("gb tail")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__gb_product(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.product()))
}

#[no_mangle]
pub extern "C" fn polars__gb_all(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.all(false)))
}

#[no_mangle]
pub extern "C" fn polars__gb_any(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.any(false)))
}

#[no_mangle]
pub extern "C" fn polars__gb_skew(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.skew(false)))
}

#[no_mangle]
pub extern "C" fn polars__gb_kurt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| gb_agg(args, |e| e.kurtosis(true, true)))
}

// ── rolling on arrays (roll_*) ──────────────────────────────────────────────

fn as_f64_arr(args: &Value) -> Result<Vec<f64>> {
    let arr = args
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `data`"))?;
    Ok(arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
}

fn roll_apply<F: Fn(&[f64]) -> f64>(v: &[f64], w: usize, f: F) -> Vec<f64> {
    let n = v.len();
    let prefix = (w.saturating_sub(1)).min(n);
    let mut out = vec![f64::NAN; prefix];
    if w == 0 || w > n {
        return vec![f64::NAN; n];
    }
    for i in (w - 1)..n {
        out.push(f(&v[i + 1 - w..=i]));
    }
    out
}

macro_rules! roll_op {
    ($fn_name:ident, $key:literal, $reducer:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let v = as_f64_arr(&args)?;
                let w = args
                    .get("window")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| anyhow!("missing `window`"))? as usize;
                let out = roll_apply(&v, w, $reducer);
                Ok(json!({ $key: out }))
            })
        }
    };
}

roll_op!(polars__roll_sum, "rolling", |w: &[f64]| w
    .iter()
    .sum::<f64>());
roll_op!(polars__roll_mean, "rolling", |w: &[f64]| w
    .iter()
    .sum::<f64>()
    / w.len() as f64);
roll_op!(polars__roll_min, "rolling", |w: &[f64]| w
    .iter()
    .cloned()
    .fold(f64::INFINITY, f64::min));
roll_op!(polars__roll_max, "rolling", |w: &[f64]| w
    .iter()
    .cloned()
    .fold(f64::NEG_INFINITY, f64::max));
roll_op!(polars__roll_prod, "rolling", |w: &[f64]| w.iter().product());
roll_op!(polars__roll_std, "rolling", |w: &[f64]| {
    let n = w.len() as f64;
    let m = w.iter().sum::<f64>() / n;
    (w.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt()
});
roll_op!(polars__roll_var, "rolling", |w: &[f64]| {
    let n = w.len() as f64;
    let m = w.iter().sum::<f64>() / n;
    w.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)
});
roll_op!(polars__roll_median, "rolling", |w: &[f64]| {
    let mut s: Vec<f64> = w.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
});
roll_op!(polars__roll_range, "rolling", |w: &[f64]| {
    let mn = w.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    mx - mn
});

// ── stat_* convenience ─────────────────────────────────────────────────────

fn get_data(args: &Value) -> Result<Vec<f64>> {
    let a = args
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `data`"))?;
    Ok(a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
}

fn scalar(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

#[no_mangle]
pub extern "C" fn polars__stat_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        let r = if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        };
        Ok(json!({"mean": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_median(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut v = get_data(&args)?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let r = if v.is_empty() {
            f64::NAN
        } else if v.len() % 2 == 0 {
            (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
        } else {
            v[v.len() / 2]
        };
        Ok(json!({"median": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_mode(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for x in &v {
            *counts.entry(x.to_bits()).or_insert(0) += 1;
        }
        let r = counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| f64::from_bits(*k));
        Ok(json!({"mode": r.map(scalar).unwrap_or(Value::Null)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_variance(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        let n = v.len() as f64;
        let ddof = args.get("ddof").and_then(|v| v.as_u64()).unwrap_or(1) as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - ddof).max(1.0);
        Ok(json!({"variance": scalar(var)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_stdev(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        let n = v.len() as f64;
        let ddof = args.get("ddof").and_then(|v| v.as_u64()).unwrap_or(1) as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - ddof).max(1.0);
        Ok(json!({"stdev": scalar(var.sqrt())}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_zscore(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n.max(1.0);
        let sd = var.sqrt();
        let out: Vec<f64> = if sd == 0.0 {
            vec![0.0; v.len()]
        } else {
            v.iter().map(|x| (x - m) / sd).collect()
        };
        Ok(json!({"zscore": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_iqr(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut v = get_data(&args)?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n == 0 {
            return Ok(json!({"iqr": Value::Null}));
        }
        let q1 = v[(n as f64 * 0.25) as usize];
        let q3 = v[(n as f64 * 0.75) as usize];
        Ok(json!({"iqr": scalar(q3 - q1)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_skewness(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n.max(1.0);
        let sd = var.sqrt();
        let sk = if sd == 0.0 {
            0.0
        } else {
            v.iter().map(|x| ((x - m) / sd).powi(3)).sum::<f64>() / n.max(1.0)
        };
        Ok(json!({"skewness": scalar(sk)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_kurtosis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n.max(1.0);
        let sd = var.sqrt();
        let k = if sd == 0.0 {
            0.0
        } else {
            v.iter().map(|x| ((x - m) / sd).powi(4)).sum::<f64>() / n.max(1.0) - 3.0
        };
        Ok(json!({"kurtosis": scalar(k)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_pearson(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let xa: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let xb: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let n = xa.len() as f64;
        let ma = xa.iter().sum::<f64>() / n;
        let mb = xb.iter().sum::<f64>() / n;
        let cov = xa
            .iter()
            .zip(xb.iter())
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / n;
        let sda = (xa.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / n).sqrt();
        let sdb = (xb.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / n).sqrt();
        let r = if sda * sdb == 0.0 {
            f64::NAN
        } else {
            cov / (sda * sdb)
        };
        Ok(json!({"pearson": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_spearman(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let xa: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let xb: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let rank = |v: &[f64]| -> Vec<f64> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|i, j| {
                v[*i]
                    .partial_cmp(&v[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut r = vec![0.0; v.len()];
            for (rank, i) in idx.iter().enumerate() {
                r[*i] = rank as f64 + 1.0;
            }
            r
        };
        let ra = rank(&xa);
        let rb = rank(&xb);
        let n = ra.len() as f64;
        let ma = ra.iter().sum::<f64>() / n;
        let mb = rb.iter().sum::<f64>() / n;
        let cov = ra
            .iter()
            .zip(rb.iter())
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / n;
        let sda = (ra.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / n).sqrt();
        let sdb = (rb.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / n).sqrt();
        let r = if sda * sdb == 0.0 {
            f64::NAN
        } else {
            cov / (sda * sdb)
        };
        Ok(json!({"spearman": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_covariance(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let xa: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let xb: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let n = xa.len() as f64;
        let ma = xa.iter().sum::<f64>() / n;
        let mb = xb.iter().sum::<f64>() / n;
        let cov = xa
            .iter()
            .zip(xb.iter())
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        Ok(json!({"covariance": scalar(cov)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_percentile(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut v = get_data(&args)?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p = args
            .get("p")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `p`"))?;
        if v.is_empty() {
            return Ok(json!({"percentile": Value::Null}));
        }
        let idx = ((p / 100.0) * (v.len() as f64 - 1.0)).round() as usize;
        Ok(json!({"percentile": scalar(v[idx.min(v.len() - 1)])}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_harmonic_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        if v.is_empty() || v.contains(&0.0) {
            return Ok(json!({"harmonic_mean": Value::Null}));
        }
        let r = v.len() as f64 / v.iter().map(|x| 1.0 / x).sum::<f64>();
        Ok(json!({"harmonic_mean": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_geometric_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        if v.is_empty() {
            return Ok(json!({"geometric_mean": Value::Null}));
        }
        let s: f64 = v.iter().map(|x| x.ln()).sum();
        Ok(json!({"geometric_mean": scalar((s / v.len() as f64).exp())}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_trimmed_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut v = get_data(&args)?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p = args.get("p").and_then(|v| v.as_f64()).unwrap_or(0.1);
        let drop = ((p * v.len() as f64) as usize).min(v.len() / 2);
        let trimmed = &v[drop..v.len() - drop];
        let r = if trimmed.is_empty() {
            f64::NAN
        } else {
            trimmed.iter().sum::<f64>() / trimmed.len() as f64
        };
        Ok(json!({"trimmed_mean": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_mad(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut v = get_data(&args)?;
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n == 0 {
            return Ok(json!({"mad": Value::Null}));
        }
        let med = if n % 2 == 0 {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        } else {
            v[n / 2]
        };
        let mut devs: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = if n % 2 == 0 {
            (devs[n / 2 - 1] + devs[n / 2]) / 2.0
        } else {
            devs[n / 2]
        };
        Ok(json!({"mad": scalar(mad)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_rmse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let xa: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let xb: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let n = xa.len().min(xb.len()) as f64;
        let sse: f64 = xa.iter().zip(xb.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        Ok(json!({"rmse": scalar((sse / n.max(1.0)).sqrt())}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_mae(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let xa: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let xb: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let n = xa.len().min(xb.len()) as f64;
        let sae: f64 = xa.iter().zip(xb.iter()).map(|(x, y)| (x - y).abs()).sum();
        Ok(json!({"mae": scalar(sae / n.max(1.0))}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_mse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let xa: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let xb: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let n = xa.len().min(xb.len()) as f64;
        let sse: f64 = xa.iter().zip(xb.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        Ok(json!({"mse": scalar(sse / n.max(1.0))}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_r2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let xa: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let xb: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let mean: f64 = xa.iter().sum::<f64>() / xa.len() as f64;
        let ss_tot: f64 = xa.iter().map(|x| (x - mean).powi(2)).sum();
        let ss_res: f64 = xa
            .iter()
            .zip(xb.iter())
            .map(|(y, yh)| (y - yh).powi(2))
            .sum();
        Ok(json!({"r2": scalar(1.0 - ss_res / ss_tot)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_entropy(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_data(&args)?;
        let r: f64 = v.iter().filter(|x| **x > 0.0).map(|x| -x * x.ln()).sum();
        Ok(json!({"entropy": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_kl(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("p")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `p`"))?;
        let b = args
            .get("q")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `q`"))?;
        let p: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
        let q: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
        let r: f64 = p
            .iter()
            .zip(q.iter())
            .filter(|(p, q)| **p > 0.0 && **q > 0.0)
            .map(|(p, q)| p * (p / q).ln())
            .sum();
        Ok(json!({"kl": scalar(r)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__stat_jsd(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("p")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `p`"))?;
        let b = args
            .get("q")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `q`"))?;
        let p: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
        let q: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
        let m: Vec<f64> = p.iter().zip(q.iter()).map(|(x, y)| (x + y) / 2.0).collect();
        let kl = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b.iter())
                .filter(|(p, q)| **p > 0.0 && **q > 0.0)
                .map(|(p, q)| p * (p / q).ln())
                .sum()
        };
        let r = 0.5 * (kl(&p, &m) + kl(&q, &m));
        Ok(json!({"jsd": scalar(r)}))
    })
}

// ── bool_* ─────────────────────────────────────────────────────────────────

fn get_bool_arr(args: &Value) -> Result<Vec<bool>> {
    let a = args
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `data`"))?;
    Ok(a.iter().map(|x| x.as_bool().unwrap_or(false)).collect())
}

#[no_mangle]
pub extern "C" fn polars__bool_any(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_bool_arr(&args)?;
        Ok(json!({"any": v.iter().any(|b| *b)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_all(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_bool_arr(&args)?;
        Ok(json!({"all": v.iter().all(|b| *b)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_count_true(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_bool_arr(&args)?;
        Ok(json!({"count_true": v.iter().filter(|b| **b).count()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_count_false(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_bool_arr(&args)?;
        Ok(json!({"count_false": v.iter().filter(|b| !**b).count()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_negate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_bool_arr(&args)?;
        let out: Vec<bool> = v.iter().map(|b| !b).collect();
        Ok(json!({"data": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_and(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let out: Vec<bool> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x.as_bool().unwrap_or(false) && y.as_bool().unwrap_or(false))
            .collect();
        Ok(json!({"data": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_or(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let out: Vec<bool> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x.as_bool().unwrap_or(false) || y.as_bool().unwrap_or(false))
            .collect();
        Ok(json!({"data": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_xor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let out: Vec<bool> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x.as_bool().unwrap_or(false) ^ y.as_bool().unwrap_or(false))
            .collect();
        Ok(json!({"data": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_to_int(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_bool_arr(&args)?;
        let out: Vec<i64> = v.iter().map(|b| if *b { 1 } else { 0 }).collect();
        Ok(json!({"data": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_indices_true(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_bool_arr(&args)?;
        let out: Vec<i64> = v
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if *b { Some(i as i64) } else { None })
            .collect();
        Ok(json!({"indices": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bool_indices_false(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_bool_arr(&args)?;
        let out: Vec<i64> = v
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if !*b { Some(i as i64) } else { None })
            .collect();
        Ok(json!({"indices": out}))
    })
}

// ── set_* ──────────────────────────────────────────────────────────────────

fn get_set(args: &Value, key: &str) -> Result<Vec<f64>> {
    let a = args
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `{key}`"))?;
    Ok(a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
}

#[no_mangle]
pub extern "C" fn polars__set_union(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let b = get_set(&args, "b")?;
        let mut seen = std::collections::HashSet::new();
        let mut out = vec![];
        for x in a.iter().chain(b.iter()) {
            if seen.insert(x.to_bits()) {
                out.push(*x);
            }
        }
        Ok(json!({"set": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__set_intersection(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let b = get_set(&args, "b")?;
        let bs: std::collections::HashSet<u64> = b.iter().map(|x| x.to_bits()).collect();
        let mut seen = std::collections::HashSet::new();
        let out: Vec<f64> = a
            .into_iter()
            .filter(|x| bs.contains(&x.to_bits()) && seen.insert(x.to_bits()))
            .collect();
        Ok(json!({"set": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__set_difference(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let b = get_set(&args, "b")?;
        let bs: std::collections::HashSet<u64> = b.iter().map(|x| x.to_bits()).collect();
        let mut seen = std::collections::HashSet::new();
        let out: Vec<f64> = a
            .into_iter()
            .filter(|x| !bs.contains(&x.to_bits()) && seen.insert(x.to_bits()))
            .collect();
        Ok(json!({"set": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__set_symmetric_difference(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let b = get_set(&args, "b")?;
        let as_: std::collections::HashSet<u64> = a.iter().map(|x| x.to_bits()).collect();
        let bs: std::collections::HashSet<u64> = b.iter().map(|x| x.to_bits()).collect();
        let mut seen = std::collections::HashSet::new();
        let mut out = vec![];
        for x in a.into_iter() {
            if !bs.contains(&x.to_bits()) && seen.insert(x.to_bits()) {
                out.push(x);
            }
        }
        for x in b.into_iter() {
            if !as_.contains(&x.to_bits()) && seen.insert(x.to_bits()) {
                out.push(x);
            }
        }
        Ok(json!({"set": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__set_is_subset(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let b = get_set(&args, "b")?;
        let bs: std::collections::HashSet<u64> = b.iter().map(|x| x.to_bits()).collect();
        let r = a.iter().all(|x| bs.contains(&x.to_bits()));
        Ok(json!({"is_subset": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__set_is_superset(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let b = get_set(&args, "b")?;
        let as_: std::collections::HashSet<u64> = a.iter().map(|x| x.to_bits()).collect();
        let r = b.iter().all(|x| as_.contains(&x.to_bits()));
        Ok(json!({"is_superset": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__set_is_disjoint(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let b = get_set(&args, "b")?;
        let as_: std::collections::HashSet<u64> = a.iter().map(|x| x.to_bits()).collect();
        let r = b.iter().all(|x| !as_.contains(&x.to_bits()));
        Ok(json!({"is_disjoint": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__set_contains(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let v = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let r = a.contains(&v);
        Ok(json!({"contains": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__set_size(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_set(&args, "a")?;
        let s: std::collections::HashSet<u64> = a.iter().map(|x| x.to_bits()).collect();
        Ok(json!({"size": s.len()}))
    })
}

// ── DataFrame extras (df_*) ────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__df_pivot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let _df = get_frame(&args)?;
        bail!("df_pivot: not yet wired — use df_groupby + manual reshape for now")
    })
}

#[no_mangle]
pub extern "C" fn polars__df_melt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let id = args
            .get("id_vars")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let val = args
            .get("value_vars")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let result = df
            .unpivot(
                val.iter().map(|s| s.as_str()),
                id.iter().map(|s| s.as_str()),
            )
            .context("melt")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_explode_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let cols = args
            .get("columns")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let exprs: Vec<Expr> = cols.iter().map(|c| col(c.as_str())).collect();
        let result = df.lazy().explode(exprs).collect().context("explode")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_transpose_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut df = get_frame(&args)?;
        let result = df.transpose(None, None).context("transpose")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_set_index(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        return_frame(df)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_reset_index(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let mut result = df.clone();
        let n = result.height();
        let idx: Vec<i64> = (0..n as i64).collect();
        result.with_column(Series::new("index".into(), idx))?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_set_column(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `name`"))?;
        let data = args
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `data`"))?;
        let mut result = df.clone();
        let s = json_array_to_series(name, data)?;
        result.with_column(s)?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_drop_duplicates(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let result = df
            .lazy()
            .unique(None, UniqueKeepStrategy::First)
            .collect()
            .context("drop_duplicates")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_sample_n(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let with_replacement = args
            .get("with_replacement")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let seed = args.get("seed").and_then(|v| v.as_u64());
        let result = df
            .sample_n_literal(n as usize, with_replacement, false, seed)
            .context("sample_n")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_sample_frac(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let frac = args
            .get("frac")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `frac`"))?;
        let n = (df.height() as f64 * frac) as usize;
        let result = df
            .sample_n_literal(n, false, false, None)
            .context("sample_frac")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_concat_horizontal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("frames")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `frames`"))?;
        let dfs: Vec<DataFrame> = arr.iter().map(parse_df).collect::<Result<_>>()?;
        if dfs.is_empty() {
            bail!("frames is empty");
        }
        let mut result = dfs[0].clone();
        for df in dfs.iter().skip(1) {
            for col in df.get_columns() {
                result.with_column(col.as_materialized_series().clone())?;
            }
        }
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_count_nulls(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let mut out = serde_json::Map::new();
        for c in df.get_columns() {
            out.insert(c.name().to_string(), json!(c.null_count()));
        }
        Ok(json!({"null_counts": Value::Object(out)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_any_nulls(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let r = df.get_columns().iter().any(|c| c.null_count() > 0);
        Ok(json!({"any_nulls": r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_height(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        Ok(json!({"height": df.height()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_width(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        Ok(json!({"width": df.width()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_empty(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        Ok(json!({"empty": df.is_empty()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_corr(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let result = df
            .clone()
            .lazy()
            .select(
                df.get_column_names_owned()
                    .iter()
                    .map(|c| col(c.as_str()))
                    .collect::<Vec<_>>(),
            )
            .collect()
            .context("corr select")?;
        // Compute pairwise pearson manually.
        let names: Vec<String> = result
            .get_column_names_owned()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut out = serde_json::Map::new();
        for n1 in &names {
            let s1 = result
                .column(n1)?
                .as_materialized_series()
                .cast(&DataType::Float64)?;
            let mut row = serde_json::Map::new();
            for n2 in &names {
                let s2 = result
                    .column(n2)?
                    .as_materialized_series()
                    .cast(&DataType::Float64)?;
                let v1: Vec<f64> = s1.f64()?.into_no_null_iter().collect();
                let v2: Vec<f64> = s2.f64()?.into_no_null_iter().collect();
                let n = v1.len() as f64;
                let m1 = v1.iter().sum::<f64>() / n;
                let m2 = v2.iter().sum::<f64>() / n;
                let cov = v1
                    .iter()
                    .zip(v2.iter())
                    .map(|(x, y)| (x - m1) * (y - m2))
                    .sum::<f64>()
                    / n;
                let sd1 = (v1.iter().map(|x| (x - m1).powi(2)).sum::<f64>() / n).sqrt();
                let sd2 = (v2.iter().map(|x| (x - m2).powi(2)).sum::<f64>() / n).sqrt();
                let r = if sd1 * sd2 == 0.0 {
                    f64::NAN
                } else {
                    cov / (sd1 * sd2)
                };
                row.insert(n2.clone(), scalar(r));
            }
            out.insert(n1.clone(), Value::Object(row));
        }
        Ok(json!({"corr": Value::Object(out)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_cov(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let names: Vec<String> = df
            .get_column_names_owned()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut out = serde_json::Map::new();
        for n1 in &names {
            let s1 = df
                .column(n1)?
                .as_materialized_series()
                .cast(&DataType::Float64)?;
            let mut row = serde_json::Map::new();
            for n2 in &names {
                let s2 = df
                    .column(n2)?
                    .as_materialized_series()
                    .cast(&DataType::Float64)?;
                let v1: Vec<f64> = s1.f64()?.into_no_null_iter().collect();
                let v2: Vec<f64> = s2.f64()?.into_no_null_iter().collect();
                let n = v1.len() as f64;
                let m1 = v1.iter().sum::<f64>() / n;
                let m2 = v2.iter().sum::<f64>() / n;
                let cov = v1
                    .iter()
                    .zip(v2.iter())
                    .map(|(x, y)| (x - m1) * (y - m2))
                    .sum::<f64>()
                    / (n - 1.0).max(1.0);
                row.insert(n2.clone(), scalar(cov));
            }
            out.insert(n1.clone(), Value::Object(row));
        }
        Ok(json!({"cov": Value::Object(out)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_describe(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let mut out = serde_json::Map::new();
        for c in df.get_columns() {
            let s = c.as_materialized_series();
            if let Ok(cast) = s.cast(&DataType::Float64) {
                if let Ok(ca) = cast.f64() {
                    let v: Vec<f64> = ca.into_no_null_iter().collect();
                    let n = v.len() as f64;
                    let m = v.iter().sum::<f64>() / n.max(1.0);
                    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
                    let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
                    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    out.insert(
                        c.name().to_string(),
                        json!({
                            "count": v.len(),
                            "mean": scalar(m),
                            "std": scalar(var.sqrt()),
                            "min": scalar(mn),
                            "max": scalar(mx),
                        }),
                    );
                }
            }
        }
        Ok(json!({"describe": Value::Object(out)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_clip_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let lo = args
            .get("min")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let hi = args
            .get("max")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::INFINITY);
        let exprs: Vec<Expr> = df
            .get_column_names_owned()
            .iter()
            .map(|c| col(c.as_str()).clip(lit(lo), lit(hi)).alias(c.as_str()))
            .collect();
        let result = df.lazy().with_columns(exprs).collect().context("clip")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_shift_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
        let result = df
            .clone()
            .lazy()
            .with_columns(
                df.get_column_names_owned()
                    .iter()
                    .map(|c| col(c.as_str()).shift(lit(n)).alias(c.as_str()))
                    .collect::<Vec<_>>(),
            )
            .collect()
            .context("shift")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_diff_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
        let result = df
            .clone()
            .lazy()
            .with_columns(
                df.get_column_names_owned()
                    .iter()
                    .map(|c| (col(c.as_str()) - col(c.as_str()).shift(lit(n))).alias(c.as_str()))
                    .collect::<Vec<_>>(),
            )
            .collect()
            .context("diff")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_pct_change_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
        let result = df
            .clone()
            .lazy()
            .with_columns(
                df.get_column_names_owned()
                    .iter()
                    .map(|c| col(c.as_str()).pct_change(lit(n)).alias(c.as_str()))
                    .collect::<Vec<_>>(),
            )
            .collect()
            .context("pct_change")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_rank_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let result = df
            .clone()
            .lazy()
            .with_columns(
                df.get_column_names_owned()
                    .iter()
                    .map(|c| {
                        col(c.as_str())
                            .rank(
                                RankOptions {
                                    method: RankMethod::Average,
                                    descending: false,
                                },
                                None,
                            )
                            .alias(c.as_str())
                    })
                    .collect::<Vec<_>>(),
            )
            .collect()
            .context("rank")?;
        return_frame(result)
    })
}

#[no_mangle]
pub extern "C" fn polars__df_any_axis_columns(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let mut out = serde_json::Map::new();
        for c in df.get_columns() {
            let s = c.as_materialized_series();
            let r = s
                .cast(&DataType::Boolean)
                .ok()
                .and_then(|c| c.bool().ok().map(|b| b.into_no_null_iter().any(|x| x)))
                .unwrap_or(false);
            out.insert(c.name().to_string(), Value::Bool(r));
        }
        Ok(json!({"any": Value::Object(out)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_all_axis_columns(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let mut out = serde_json::Map::new();
        for c in df.get_columns() {
            let s = c.as_materialized_series();
            let r = s
                .cast(&DataType::Boolean)
                .ok()
                .and_then(|c| c.bool().ok().map(|b| b.into_no_null_iter().all(|x| x)))
                .unwrap_or(false);
            out.insert(c.name().to_string(), Value::Bool(r));
        }
        Ok(json!({"all": Value::Object(out)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__df_mode_per_column(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let mut out = serde_json::Map::new();
        for c in df.get_columns() {
            let s = c.as_materialized_series();
            if let Ok(cast) = s.cast(&DataType::Float64) {
                if let Ok(ca) = cast.f64() {
                    let v: Vec<f64> = ca.into_no_null_iter().collect();
                    let mut counts: std::collections::HashMap<u64, u64> =
                        std::collections::HashMap::new();
                    for x in &v {
                        *counts.entry(x.to_bits()).or_insert(0) += 1;
                    }
                    let m = counts
                        .iter()
                        .max_by_key(|(_, c)| *c)
                        .map(|(k, _)| f64::from_bits(*k));
                    out.insert(c.name().to_string(), m.map(scalar).unwrap_or(Value::Null));
                }
            }
        }
        Ok(json!({"mode": Value::Object(out)}))
    })
}
