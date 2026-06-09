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

// ── P1.2: per-column aggregations ──────────────────────────────────────────

fn agg_all<F>(args: &Value, f: F) -> Result<Value>
where
    F: FnOnce(LazyFrame) -> LazyFrame,
{
    let df = get_frame(args)?;
    let agged = f(df.lazy()).collect().context("agg")?;
    return_frame(agged)
}

/// Sum of every column (string/bool error per polars rules). Result: 1-row frame.
#[no_mangle]
pub extern "C" fn polars__df_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| agg_all(&args, |lf| lf.sum()))
}

/// Mean of every column. Result: 1-row frame.
#[no_mangle]
pub extern "C" fn polars__df_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| agg_all(&args, |lf| lf.mean()))
}

/// Median of every column.
#[no_mangle]
pub extern "C" fn polars__df_median(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| agg_all(&args, |lf| lf.median()))
}

/// Min of every column.
#[no_mangle]
pub extern "C" fn polars__df_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| agg_all(&args, |lf| lf.min()))
}

/// Max of every column.
#[no_mangle]
pub extern "C" fn polars__df_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| agg_all(&args, |lf| lf.max()))
}

/// Standard deviation (ddof=1 by default).
///
/// Args:   `{frame, ddof?: u8 (default 1)}`
#[no_mangle]
pub extern "C" fn polars__df_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let ddof = args.get("ddof").and_then(|v| v.as_u64()).unwrap_or(1) as u8;
        agg_all(&args, |lf| lf.std(ddof))
    })
}

/// Variance (ddof=1 by default).
#[no_mangle]
pub extern "C" fn polars__df_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let ddof = args.get("ddof").and_then(|v| v.as_u64()).unwrap_or(1) as u8;
        agg_all(&args, |lf| lf.var(ddof))
    })
}

/// Non-null row count per column.
#[no_mangle]
pub extern "C" fn polars__df_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| agg_all(&args, |lf| lf.count()))
}

/// Quantile per column. `q` in [0,1]. Default linear interpolation.
///
/// Args:   `{frame, q: f64}`
#[no_mangle]
pub extern "C" fn polars__df_quantile(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let q = args
            .get("q")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing argument `q` (f64 in [0,1])"))?;
        agg_all(&args, |lf| lf.quantile(lit(q), QuantileMethod::Linear))
    })
}

/// Per-column distinct-value count.
///
/// Args:   `{frame}`
/// Result: `{counts: {col: n, ...}}`
#[no_mangle]
pub extern "C" fn polars__df_nunique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let mut out = Map::new();
        for name in df.get_column_names_owned() {
            let col = df.column(name.as_str())?;
            let n = col.as_materialized_series().n_unique()?;
            out.insert(name.to_string(), json!(n));
        }
        Ok(json!({"counts": Value::Object(out)}))
    })
}

// ── P1.2b: cumulative ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum CumOp {
    Sum,
    Prod,
    Min,
    Max,
}

fn cum_op(args: &Value, op: CumOp) -> Result<Value> {
    let df = get_frame(args)?;
    let names: Vec<String> = match args.get("columns").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        None => df
            .get_column_names_owned()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    let exprs: Vec<Expr> = names
        .iter()
        .map(|name| {
            let c = col(name.as_str());
            let e = match op {
                CumOp::Sum => c.cum_sum(false),
                CumOp::Prod => c.cum_prod(false),
                CumOp::Min => c.cum_min(false),
                CumOp::Max => c.cum_max(false),
            };
            e.alias(name.as_str())
        })
        .collect();
    let result = df
        .lazy()
        .with_columns(exprs)
        .collect()
        .context("cumulative")?;
    return_frame(result)
}

/// Cumulative sum per column. `columns` optional (default all).
#[no_mangle]
pub extern "C" fn polars__df_cumsum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| cum_op(&args, CumOp::Sum))
}

/// Cumulative product per column.
#[no_mangle]
pub extern "C" fn polars__df_cumprod(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| cum_op(&args, CumOp::Prod))
}

/// Cumulative min per column.
#[no_mangle]
pub extern "C" fn polars__df_cummin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| cum_op(&args, CumOp::Min))
}

/// Cumulative max per column.
#[no_mangle]
pub extern "C" fn polars__df_cummax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| cum_op(&args, CumOp::Max))
}

// ── P1.2c: elementwise math ────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum MathOp {
    Abs,
    Sqrt,
    Exp,
    Log,
    Floor,
    Ceil,
    Round,
}

fn math_op(args: &Value, op: MathOp) -> Result<Value> {
    let df = get_frame(args)?;
    let names: Vec<String> = match args.get("columns").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        None => df
            .get_column_names_owned()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    let decimals = args.get("decimals").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let exprs: Vec<Expr> = names
        .iter()
        .map(|name| {
            let c = col(name.as_str());
            let e = match op {
                MathOp::Abs => c.abs(),
                MathOp::Sqrt => c.sqrt(),
                MathOp::Exp => c.exp(),
                MathOp::Log => c.log(std::f64::consts::E),
                MathOp::Floor => c.floor(),
                MathOp::Ceil => c.ceil(),
                MathOp::Round => c.round(decimals),
            };
            e.alias(name.as_str())
        })
        .collect();
    let result = df.lazy().with_columns(exprs).collect().context("math")?;
    return_frame(result)
}

/// Elementwise abs.
#[no_mangle]
pub extern "C" fn polars__df_abs(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| math_op(&args, MathOp::Abs))
}

/// Elementwise sqrt.
#[no_mangle]
pub extern "C" fn polars__df_sqrt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| math_op(&args, MathOp::Sqrt))
}

/// Elementwise exp.
#[no_mangle]
pub extern "C" fn polars__df_exp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| math_op(&args, MathOp::Exp))
}

/// Elementwise natural log.
#[no_mangle]
pub extern "C" fn polars__df_log(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| math_op(&args, MathOp::Log))
}

/// Elementwise floor.
#[no_mangle]
pub extern "C" fn polars__df_floor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| math_op(&args, MathOp::Floor))
}

/// Elementwise ceil.
#[no_mangle]
pub extern "C" fn polars__df_ceil(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| math_op(&args, MathOp::Ceil))
}

/// Round to `decimals` places (default 0).
#[no_mangle]
pub extern "C" fn polars__df_round(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| math_op(&args, MathOp::Round))
}

// ── P1.4: missing / null handling ──────────────────────────────────────────

/// Fill nulls with `value` (numeric / string / bool).
///
/// Args:   `{frame, value}`
#[no_mangle]
pub extern "C" fn polars__df_fill_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing argument `value`"))?;
        let lit_v = json_to_lit(v)?;
        let result = df.lazy().fill_null(lit_v).collect().context("fill_null")?;
        return_frame(result)
    })
}

/// Drop rows where ANY column is null. Optional `subset` restricts the check.
///
/// Args:   `{frame, subset?: [name, ...]}`
#[no_mangle]
pub extern "C" fn polars__df_drop_nulls(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let subset: Option<Vec<Expr>> = args
            .get("subset")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_str().map(col)).collect());
        let result = df
            .lazy()
            .drop_nulls(subset)
            .collect()
            .context("drop_nulls")?;
        return_frame(result)
    })
}

/// Per-column null-mask DataFrame (every cell becomes true/false).
#[no_mangle]
pub extern "C" fn polars__df_is_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let exprs: Vec<Expr> = df
            .get_column_names_owned()
            .iter()
            .map(|n| col(n.as_str()).is_null().alias(n.as_str()))
            .collect();
        let result = df.lazy().select(exprs).collect().context("is_null")?;
        return_frame(result)
    })
}

/// Per-column not-null mask.
#[no_mangle]
pub extern "C" fn polars__df_is_not_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let exprs: Vec<Expr> = df
            .get_column_names_owned()
            .iter()
            .map(|n| col(n.as_str()).is_not_null().alias(n.as_str()))
            .collect();
        let result = df.lazy().select(exprs).collect().context("is_not_null")?;
        return_frame(result)
    })
}

// ── P1.4b: unique / counts ─────────────────────────────────────────────────

/// Distinct rows. Optional `subset` restricts the uniqueness check.
///
/// Args:   `{frame, subset?: [name, ...]}`
#[no_mangle]
pub extern "C" fn polars__df_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let subset: Option<Vec<String>> =
            args.get("subset").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            });
        // Use LazyFrame::unique — eager DataFrame::unique requires generic
        // type annotation polars 0.45 won't infer.
        let result = df
            .lazy()
            .unique(subset, UniqueKeepStrategy::First)
            .collect()
            .context("unique")?;
        return_frame(result)
    })
}

/// pandas `Series.value_counts` for a single column.
///
/// Args:   `{frame, column}`
/// Result: `{frame}` with `column` + `count` columns
#[no_mangle]
pub extern "C" fn polars__df_value_counts(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let col_name = args
            .get("column")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing argument `column`"))?;
        let s = df.column(col_name)?;
        let counts = s
            .as_materialized_series()
            .value_counts(true, true, "count".into(), false)
            .context("value_counts")?;
        return_frame(counts)
    })
}

// ── P1.3: joins ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum JoinKind {
    Inner,
    Left,
    Full,
    Cross,
    Semi,
    Anti,
}

fn join_op(args: &Value, kind: JoinKind) -> Result<Value> {
    let left = get_frame(args)?;
    let right_v = args
        .get("right")
        .ok_or_else(|| anyhow!("missing argument `right` (the right-hand frame)"))?;
    let right = parse_df(right_v)?;

    let join_type = match kind {
        JoinKind::Inner => JoinType::Inner,
        JoinKind::Left => JoinType::Left,
        JoinKind::Full => JoinType::Full,
        JoinKind::Cross => JoinType::Cross,
        JoinKind::Semi => JoinType::Semi,
        JoinKind::Anti => JoinType::Anti,
    };

    let lf_left = left.lazy();
    let lf_right = right.lazy();

    let result = if matches!(kind, JoinKind::Cross) {
        lf_left
            .join(lf_right, [], [], JoinArgs::new(join_type))
            .collect()
            .context("join")?
    } else {
        let left_on = args
            .get("left_on")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `left_on`"))?;
        let right_on = args
            .get("right_on")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing argument `right_on`"))?;
        let lo: Vec<Expr> = left_on.iter().filter_map(|v| v.as_str().map(col)).collect();
        let ro: Vec<Expr> = right_on
            .iter()
            .filter_map(|v| v.as_str().map(col))
            .collect();
        lf_left
            .join(lf_right, lo, ro, JoinArgs::new(join_type))
            .collect()
            .context("join")?
    };
    return_frame(result)
}

/// Inner join. `right` is the RHS frame; `left_on`/`right_on` are key lists.
#[no_mangle]
pub extern "C" fn polars__df_inner_join(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| join_op(&args, JoinKind::Inner))
}

/// Left join.
#[no_mangle]
pub extern "C" fn polars__df_left_join(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| join_op(&args, JoinKind::Left))
}

/// Full (outer) join.
#[no_mangle]
pub extern "C" fn polars__df_full_join(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| join_op(&args, JoinKind::Full))
}

/// Cross (cartesian) join. `left_on`/`right_on` ignored.
#[no_mangle]
pub extern "C" fn polars__df_cross_join(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| join_op(&args, JoinKind::Cross))
}

/// Semi join (rows of `left` that have a match on `right`).
#[no_mangle]
pub extern "C" fn polars__df_semi_join(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| join_op(&args, JoinKind::Semi))
}

/// Anti join (rows of `left` that do NOT have a match on `right`).
#[no_mangle]
pub extern "C" fn polars__df_anti_join(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| join_op(&args, JoinKind::Anti))
}

/// Vertically stack `left` ⨃ `right` (must share schema).
///
/// Args:   `{frame, right}`
#[no_mangle]
pub extern "C" fn polars__df_concat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let left = get_frame(&args)?;
        let right_v = args
            .get("right")
            .ok_or_else(|| anyhow!("missing argument `right`"))?;
        let right = parse_df(right_v)?;
        let result = concat([left.lazy(), right.lazy()], UnionArgs::default())
            .context("concat")?
            .collect()
            .context("concat collect")?;
        return_frame(result)
    })
}

// ── P1.5a: window / shift ──────────────────────────────────────────────────

fn shift_like<F>(args: &Value, build: F) -> Result<Value>
where
    F: Fn(&str) -> Expr,
{
    let df = get_frame(args)?;
    let names: Vec<String> = match args.get("columns").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        None => df
            .get_column_names_owned()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    let exprs: Vec<Expr> = names
        .iter()
        .map(|n| build(n.as_str()).alias(n.as_str()))
        .collect();
    let result = df.lazy().with_columns(exprs).collect().context("window")?;
    return_frame(result)
}

/// Shift each column by `n` rows (default 1, can be negative).
#[no_mangle]
pub extern "C" fn polars__df_shift(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
        shift_like(&args, |c| col(c).shift(lit(n)))
    })
}

/// First-order difference per column (current − previous).
#[no_mangle]
pub extern "C" fn polars__df_diff(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
        // Manual diff = current − previous (avoids NullBehavior import issue
        // since 0.45 doesn't re-export NullBehavior in `polars::prelude`).
        shift_like(&args, |c| col(c) - col(c).shift(lit(n)))
    })
}

/// Per-column percent change (current/previous − 1).
#[no_mangle]
pub extern "C" fn polars__df_pct_change(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
        shift_like(&args, |c| col(c).pct_change(lit(n)))
    })
}

/// Per-column rank (default: average, ascending).
#[no_mangle]
pub extern "C" fn polars__df_rank(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        shift_like(&args, |c| {
            col(c).rank(
                RankOptions {
                    method: RankMethod::Average,
                    descending: false,
                },
                None,
            )
        })
    })
}

/// Clip each column into `[lower, upper]`. Both bounds optional but at least
/// one must be provided.
///
/// Args:   `{frame, lower?, upper?, columns?}`
#[no_mangle]
pub extern "C" fn polars__df_clip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let lower = args.get("lower").cloned();
        let upper = args.get("upper").cloned();
        if lower.is_none() && upper.is_none() {
            return Err(anyhow!("must provide at least one of `lower`, `upper`"));
        }
        shift_like(&args, |c| {
            let mut e = col(c);
            if let Some(ref lv) = lower {
                if let Ok(lit_lv) = json_to_lit(lv) {
                    e = e.clip_min(lit_lv);
                }
            }
            if let Some(ref uv) = upper {
                if let Ok(lit_uv) = json_to_lit(uv) {
                    e = e.clip_max(lit_uv);
                }
            }
            e
        })
    })
}
