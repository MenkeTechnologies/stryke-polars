//! src/df.rs — pandas DataFrame surface (`polars__df_*`), P1.1 slice.
//!
//! Backed by `polars` 0.45. JSON-in/JSON-out wrappers around polars'
//! `DataFrame` and `Series`.
//!
//! Wire format:
//!   - Input frame:  columnar JSON `{col1: [v1, v2, ...], col2: [...], ...}`
//!   - Output frame: columnar JSON `{col1: [...], col2: [...], ...}`
//!
//! Dtype inference: when a column is parsed from JSON, the parser picks
//! the narrowest dtype that fits every value (bool → i64 → f64 → string).
//! Mixed columns fall back to string.

use std::ffi::c_char;

use anyhow::{anyhow, Context, Result};
use polars::prelude::*;
use serde_json::{json, Map, Value};

use crate::ffi_call;

// ── JSON ↔ DataFrame conversion ────────────────────────────────────────────

/// Build a DataFrame from a columnar JSON value `{col: [vals]}`.
fn parse_df(v: &Value) -> Result<DataFrame> {
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow!("expected columnar object `{{col: [...], ...}}`"))?;
    let mut cols: Vec<Column> = Vec::with_capacity(obj.len());
    for (name, col_v) in obj {
        let arr = col_v
            .as_array()
            .ok_or_else(|| anyhow!("column `{name}` must be a JSON array"))?;
        let s = json_array_to_series(name, arr)?;
        cols.push(s.into_column());
    }
    DataFrame::new(cols).context("DataFrame::new")
}

/// Coerce a JSON array to a Polars Series, picking the narrowest dtype.
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
        // Mixed / structured — coerce to string.
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

/// Serialize a DataFrame to columnar JSON `{col: [vals], ...}`.
fn df_to_value(df: &DataFrame) -> Result<Value> {
    let mut out = Map::with_capacity(df.width());
    for col in df.get_columns() {
        let name = col.name().to_string();
        let series = col.as_materialized_series();
        out.insert(name, series_to_value(series)?);
    }
    Ok(Value::Object(out))
}

fn series_to_value(s: &Series) -> Result<Value> {
    let len = s.len();
    let mut arr = Vec::with_capacity(len);
    for i in 0..len {
        let av = s.get(i).map_err(|e| anyhow!("Series::get({i}): {e}"))?;
        arr.push(any_value_to_json(av));
    }
    Ok(Value::Array(arr))
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

// ── per-export helpers ─────────────────────────────────────────────────────

fn get_frame(args: &Value) -> Result<DataFrame> {
    let f = args
        .get("frame")
        .ok_or_else(|| anyhow!("missing argument `frame`"))?;
    parse_df(f)
}

fn return_frame(df: DataFrame) -> Result<Value> {
    Ok(json!({"frame": df_to_value(&df)?}))
}

// ── exports ────────────────────────────────────────────────────────────────

/// Build a DataFrame from columnar JSON. Round-trips through polars so the
/// returned `frame` is canonicalized (dtype inferred per column).
///
/// Args:   `{frame: {col: [...], ...}}`
/// Result: `{frame: {col: [...], ...}, shape: [nrows, ncols], dtypes: [...]}`
#[no_mangle]
pub extern "C" fn polars__df_new(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let dtypes: Vec<String> = df.dtypes().iter().map(|d| format!("{d}")).collect();
        Ok(json!({
            "frame": df_to_value(&df)?,
            "shape": [df.height(), df.width()],
            "dtypes": dtypes,
        }))
    })
}

/// Return the first `n` rows (default 5).
///
/// Args:   `{frame, n?}`
/// Result: `{frame}`
#[no_mangle]
pub extern "C" fn polars__df_head(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        return_frame(df.head(Some(n)))
    })
}

/// Return the last `n` rows (default 5).
///
/// Args:   `{frame, n?}`
/// Result: `{frame}`
#[no_mangle]
pub extern "C" fn polars__df_tail(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        return_frame(df.tail(Some(n)))
    })
}

/// Return `{shape: [nrows, ncols]}`.
#[no_mangle]
pub extern "C" fn polars__df_shape(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        Ok(json!({"shape": [df.height(), df.width()]}))
    })
}

/// Return `{columns: [name, ...]}`.
#[no_mangle]
pub extern "C" fn polars__df_columns(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let cols: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        Ok(json!({"columns": cols}))
    })
}

/// Return `{dtypes: ["i64", "f64", "str", ...]}` (polars dtype Display form).
#[no_mangle]
pub extern "C" fn polars__df_dtypes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let dtypes: Vec<String> = df.dtypes().iter().map(|d| format!("{d}")).collect();
        Ok(json!({"dtypes": dtypes}))
    })
}

/// Select a subset of columns by name (order preserved).
///
/// Args:   `{frame, columns: [name, ...]}`
/// Result: `{frame}`
#[no_mangle]
pub extern "C" fn polars__df_select(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let cols = args
            .get("columns")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `columns`"))?;
        let names: Vec<String> = cols
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        let sel = df.select(&names).context("DataFrame::select")?;
        return_frame(sel)
    })
}

/// Round-trip a DataFrame through the canonical columnar JSON form. Useful
/// for testing the parse → serialize pipeline; equivalent to `df_new` minus
/// the shape/dtypes envelope.
#[no_mangle]
pub extern "C" fn polars__df_to_json(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        return_frame(df)
    })
}

// ── P1.1b: drop / rename / sort / filter ───────────────────────────────────
//
// `polars__df_describe` was attempted but polars 0.45 doesn't expose
// `describe` as a `DataFrame` method (despite the `describe` feature being
// enabled — the method lives in a different module path or on `LazyFrame`).
// Deferred to a later slice where it can be hand-rolled from per-stat aggs.

/// Drop one or more columns by name.
///
/// Args:   `{frame, columns: [name, ...]}`
/// Result: `{frame}` (without dropped columns)
#[no_mangle]
pub extern "C" fn polars__df_drop(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut df = get_frame(&args)?;
        let cols = args
            .get("columns")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `columns`"))?;
        for v in cols {
            let name = v
                .as_str()
                .ok_or_else(|| anyhow!("`columns` entries must be strings"))?;
            df = df.drop(name).context(format!("drop({name})"))?;
        }
        return_frame(df)
    })
}

/// Rename one or more columns.
///
/// Args:   `{frame, mapping: {old: new, ...}}`
/// Result: `{frame}` (with renamed columns)
#[no_mangle]
pub extern "C" fn polars__df_rename(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut df = get_frame(&args)?;
        let map = args
            .get("mapping")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("missing argument `mapping`"))?;
        for (old, new_v) in map {
            let new = new_v
                .as_str()
                .ok_or_else(|| anyhow!("mapping[{old}] must be a string"))?;
            df.rename(old, new.into())
                .context(format!("rename({old} → {new})"))?;
        }
        return_frame(df)
    })
}

/// Sort by columns.
///
/// Args:   `{frame, by: [name, ...], descending?: [bool, ...] | bool}`
/// Result: `{frame}` (sorted)
#[no_mangle]
pub extern "C" fn polars__df_sort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let by_arr = args
            .get("by")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `by`"))?;
        let by: Vec<PlSmallStr> = by_arr
            .iter()
            .filter_map(|v| v.as_str().map(PlSmallStr::from))
            .collect();
        if by.is_empty() {
            return Err(anyhow!("`by` must list at least one column name"));
        }
        let descending = parse_bool_vec(args.get("descending"), by.len());
        let opts = SortMultipleOptions::default().with_order_descending_multi(descending);
        let sorted = df.sort(by, opts).context("DataFrame::sort")?;
        return_frame(sorted)
    })
}

fn parse_bool_vec(v: Option<&Value>, n: usize) -> Vec<bool> {
    match v {
        Some(Value::Bool(b)) => vec![*b; n],
        Some(Value::Array(arr)) => {
            let mut out: Vec<bool> = arr.iter().map(|x| x.as_bool().unwrap_or(false)).collect();
            out.resize(n, false);
            out
        }
        _ => vec![false; n],
    }
}

/// Filter rows where `column > value` (numeric / string / bool).
#[no_mangle]
pub extern "C" fn polars__df_filter_gt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| filter_cmp(&args, FilterOp::Gt))
}

/// Filter rows where `column < value`.
#[no_mangle]
pub extern "C" fn polars__df_filter_lt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| filter_cmp(&args, FilterOp::Lt))
}

/// Filter rows where `column == value`.
#[no_mangle]
pub extern "C" fn polars__df_filter_eq(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| filter_cmp(&args, FilterOp::Eq))
}

#[derive(Clone, Copy)]
enum FilterOp {
    Gt,
    Lt,
    Eq,
}

fn filter_cmp(args: &Value, op: FilterOp) -> Result<Value> {
    let df = get_frame(args)?;
    let col_name = args
        .get("column")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing argument `column`"))?;
    let val = args
        .get("value")
        .ok_or_else(|| anyhow!("missing argument `value`"))?;

    let lit_expr = json_to_lit(val)?;
    let col_expr = col(col_name);
    let expr = match op {
        FilterOp::Gt => col_expr.gt(lit_expr),
        FilterOp::Lt => col_expr.lt(lit_expr),
        FilterOp::Eq => col_expr.eq(lit_expr),
    };

    let result = df.lazy().filter(expr).collect().context("filter")?;
    return_frame(result)
}

fn json_to_lit(v: &Value) -> Result<Expr> {
    match v {
        Value::Null => Ok(lit(NULL)),
        Value::Bool(b) => Ok(lit(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(lit(i))
            } else if let Some(f) = n.as_f64() {
                Ok(lit(f))
            } else {
                Err(anyhow!("unsupported numeric literal: {n}"))
            }
        }
        Value::String(s) => Ok(lit(s.clone())),
        _ => Err(anyhow!(
            "filter `value` must be null/bool/number/string (got: {v:?})"
        )),
    }
}
