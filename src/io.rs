//! src/io.rs — pandas IO surface (`polars__pd_*`).
//!
//! Backed by polars file readers/writers. CSV / JSON via polars; Parquet /
//! Feather / Arrow are stubbed to surface clear errors (route via
//! stryke-arrow). All paths are filesystem paths.

use std::ffi::c_char;
use std::fs;
use std::io::Cursor;

use anyhow::{anyhow, bail, Context, Result};
use polars::prelude::*;
use serde_json::{json, Map, Value};

use crate::ffi_call;

fn get_path(args: &Value) -> Result<String> {
    args.get("path")
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| anyhow!("missing argument `path`"))
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

fn parse_df(v: &Value) -> Result<DataFrame> {
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow!("expected columnar object"))?;
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

fn get_frame(args: &Value) -> Result<DataFrame> {
    let f = args
        .get("frame")
        .ok_or_else(|| anyhow!("missing argument `frame`"))?;
    parse_df(f)
}

// ── read_csv / write_csv ────────────────────────────────────────────────────

/// pandas IO read csv.
#[no_mangle]
pub extern "C" fn polars__pd_read_csv(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let path = get_path(&args)?;
        let has_header = args.get("header").and_then(|v| v.as_bool()).unwrap_or(true);
        let sep = args
            .get("sep")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(',');
        let df = CsvReadOptions::default()
            .with_has_header(has_header)
            .with_parse_options(CsvParseOptions::default().with_separator(sep as u8))
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()
            .context("read_csv")?;
        Ok(json!({"frame": df_to_value(&df)?}))
    })
}

/// pandas IO to csv.
#[no_mangle]
pub extern "C" fn polars__pd_to_csv(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut df = get_frame(&args)?;
        let path = get_path(&args)?;
        let mut file = fs::File::create(&path).context("create csv")?;
        CsvWriter::new(&mut file)
            .finish(&mut df)
            .context("write csv")?;
        Ok(json!({"path": path}))
    })
}

/// pandas IO read csv string.
#[no_mangle]
pub extern "C" fn polars__pd_read_csv_string(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("data")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `data`"))?
            .to_string();
        let cursor = Cursor::new(s.into_bytes());
        let df = CsvReader::new(cursor).finish().context("read_csv_string")?;
        Ok(json!({"frame": df_to_value(&df)?}))
    })
}

/// pandas IO to csv string.
#[no_mangle]
pub extern "C" fn polars__pd_to_csv_string(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut df = get_frame(&args)?;
        let mut buf = vec![];
        CsvWriter::new(&mut buf)
            .finish(&mut df)
            .context("write csv string")?;
        let s = String::from_utf8(buf).context("utf8")?;
        Ok(json!({"data": s}))
    })
}

// ── read_json / to_json ────────────────────────────────────────────────────

/// pandas IO read json.
#[no_mangle]
pub extern "C" fn polars__pd_read_json(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let path = get_path(&args)?;
        let s = fs::read_to_string(&path).context("read json file")?;
        let v: Value = serde_json::from_str(&s).context("parse json")?;
        let df = parse_df(&v)?;
        Ok(json!({"frame": df_to_value(&df)?}))
    })
}

/// pandas IO to json.
#[no_mangle]
pub extern "C" fn polars__pd_to_json(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let path = get_path(&args)?;
        let v = df_to_value(&df)?;
        fs::write(&path, serde_json::to_string(&v)?).context("write json")?;
        Ok(json!({"path": path}))
    })
}

/// pandas IO read jsonl.
#[no_mangle]
pub extern "C" fn polars__pd_read_jsonl(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let path = get_path(&args)?;
        let s = fs::read_to_string(&path).context("read jsonl")?;
        let mut cols: std::collections::HashMap<String, Vec<Value>> =
            std::collections::HashMap::new();
        for (i, line) in s.lines().enumerate() {
            let v: Value = serde_json::from_str(line).context(format!("parse line {i}"))?;
            if let Some(obj) = v.as_object() {
                for (k, val) in obj {
                    cols.entry(k.clone()).or_default().push(val.clone());
                }
            }
        }
        let mut m = Map::new();
        for (k, v) in cols {
            m.insert(k, Value::Array(v));
        }
        Ok(json!({"frame": Value::Object(m)}))
    })
}

/// pandas IO to jsonl.
#[no_mangle]
pub extern "C" fn polars__pd_to_jsonl(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let path = get_path(&args)?;
        let mut lines = vec![];
        let h = df.height();
        let cols: Vec<(String, Vec<Value>)> = df
            .get_columns()
            .iter()
            .map(|c| {
                let s = c.as_materialized_series();
                let n = s.len();
                let mut arr = Vec::with_capacity(n);
                for i in 0..n {
                    arr.push(any_value_to_json(s.get(i).unwrap_or(AnyValue::Null)));
                }
                (c.name().to_string(), arr)
            })
            .collect();
        for i in 0..h {
            let mut m = Map::new();
            for (n, v) in &cols {
                m.insert(n.clone(), v.get(i).cloned().unwrap_or(Value::Null));
            }
            lines.push(serde_json::to_string(&Value::Object(m))?);
        }
        fs::write(&path, lines.join("\n")).context("write jsonl")?;
        Ok(json!({"path": path}))
    })
}

/// pandas IO to json string.
#[no_mangle]
pub extern "C" fn polars__pd_to_json_string(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let v = df_to_value(&df)?;
        Ok(json!({"json": serde_json::to_string(&v)?}))
    })
}

// ── read_parquet / to_parquet — stubs, route via stryke-arrow ──────────────

/// pandas IO read parquet.
#[no_mangle]
pub extern "C" fn polars__pd_read_parquet(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_read_parquet: route via `stryke-arrow` package — this cdylib does not link arrow-rs")
    })
}

/// pandas IO to parquet.
#[no_mangle]
pub extern "C" fn polars__pd_to_parquet(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!(
            "pd_to_parquet: route via `stryke-arrow` package — this cdylib does not link arrow-rs"
        )
    })
}

/// pandas IO read feather.
#[no_mangle]
pub extern "C" fn polars__pd_read_feather(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_read_feather: route via `stryke-arrow` package")
    })
}

/// pandas IO to feather.
#[no_mangle]
pub extern "C" fn polars__pd_to_feather(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_to_feather: route via `stryke-arrow` package")
    })
}

/// pandas IO read arrow.
#[no_mangle]
pub extern "C" fn polars__pd_read_arrow(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_read_arrow: route via `stryke-arrow` package")
    })
}

/// pandas IO to arrow.
#[no_mangle]
pub extern "C" fn polars__pd_to_arrow(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_to_arrow: route via `stryke-arrow` package")
    })
}

// ── misc ────────────────────────────────────────────────────────────────────

/// pandas IO read tsv.
#[no_mangle]
pub extern "C" fn polars__pd_read_tsv(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let path = get_path(&args)?;
        let has_header = args.get("header").and_then(|v| v.as_bool()).unwrap_or(true);
        let df = CsvReadOptions::default()
            .with_has_header(has_header)
            .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()
            .context("read_tsv")?;
        Ok(json!({"frame": df_to_value(&df)?}))
    })
}

/// pandas IO to tsv.
#[no_mangle]
pub extern "C" fn polars__pd_to_tsv(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut df = get_frame(&args)?;
        let path = get_path(&args)?;
        let mut file = fs::File::create(&path).context("create tsv")?;
        CsvWriter::new(&mut file)
            .with_separator(b'\t')
            .finish(&mut df)
            .context("write tsv")?;
        Ok(json!({"path": path}))
    })
}

/// pandas IO read excel.
#[no_mangle]
pub extern "C" fn polars__pd_read_excel(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_read_excel: not supported in this cdylib (use pd_read_csv or pd_read_json)")
    })
}

/// pandas IO to excel.
#[no_mangle]
pub extern "C" fn polars__pd_to_excel(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| bail!("pd_to_excel: not supported in this cdylib"))
}

/// pandas IO read sql.
#[no_mangle]
pub extern "C" fn polars__pd_read_sql(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_read_sql: route via `stryke-duckdb` package")
    })
}

/// pandas IO to sql.
#[no_mangle]
pub extern "C" fn polars__pd_to_sql(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_to_sql: route via `stryke-duckdb` package")
    })
}

/// pandas IO read html.
#[no_mangle]
pub extern "C" fn polars__pd_read_html(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        bail!("pd_read_html: not supported (no HTML parser in this cdylib)")
    })
}

/// pandas IO to html.
#[no_mangle]
pub extern "C" fn polars__pd_to_html(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let mut s = String::from("<table>\n  <thead><tr>");
        for c in df.get_columns() {
            s.push_str(&format!("<th>{}</th>", c.name()));
        }
        s.push_str("</tr></thead>\n  <tbody>\n");
        let h = df.height();
        for i in 0..h {
            s.push_str("    <tr>");
            for c in df.get_columns() {
                let av = c
                    .as_materialized_series()
                    .get(i)
                    .map_err(|e| anyhow!("get: {e}"))?;
                s.push_str(&format!("<td>{av}</td>"));
            }
            s.push_str("</tr>\n");
        }
        s.push_str("  </tbody>\n</table>\n");
        Ok(json!({"html": s}))
    })
}

/// pandas IO to markdown.
#[no_mangle]
pub extern "C" fn polars__pd_to_markdown(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let cols: Vec<&str> = df.get_columns().iter().map(|c| c.name().as_str()).collect();
        let mut s = String::new();
        s.push_str("| ");
        s.push_str(&cols.join(" | "));
        s.push_str(" |\n|");
        for _ in &cols {
            s.push_str("---|");
        }
        s.push('\n');
        let h = df.height();
        for i in 0..h {
            s.push_str("| ");
            for c in df.get_columns() {
                let av = c
                    .as_materialized_series()
                    .get(i)
                    .map_err(|e| anyhow!("get: {e}"))?;
                s.push_str(&format!("{av} | "));
            }
            s.push('\n');
        }
        Ok(json!({"markdown": s}))
    })
}

/// pandas IO to string.
#[no_mangle]
pub extern "C" fn polars__pd_to_string(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        Ok(json!({"string": format!("{df}")}))
    })
}

/// pandas IO to records.
#[no_mangle]
pub extern "C" fn polars__pd_to_records(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let df = get_frame(&args)?;
        let h = df.height();
        let cols: Vec<(String, Vec<Value>)> = df
            .get_columns()
            .iter()
            .map(|c| {
                let s = c.as_materialized_series();
                let n = s.len();
                let mut arr = Vec::with_capacity(n);
                for i in 0..n {
                    arr.push(any_value_to_json(s.get(i).unwrap_or(AnyValue::Null)));
                }
                (c.name().to_string(), arr)
            })
            .collect();
        let mut records = vec![];
        for i in 0..h {
            let mut m = Map::new();
            for (n, v) in &cols {
                m.insert(n.clone(), v.get(i).cloned().unwrap_or(Value::Null));
            }
            records.push(Value::Object(m));
        }
        Ok(json!({"records": records}))
    })
}

/// pandas IO from records.
#[no_mangle]
pub extern "C" fn polars__pd_from_records(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("records")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `records`"))?;
        let mut cols: std::collections::HashMap<String, Vec<Value>> =
            std::collections::HashMap::new();
        for rec in arr {
            if let Some(obj) = rec.as_object() {
                for (k, v) in obj {
                    cols.entry(k.clone()).or_default().push(v.clone());
                }
            }
        }
        let mut m = Map::new();
        for (k, v) in cols {
            m.insert(k, Value::Array(v));
        }
        let df = parse_df(&Value::Object(m))?;
        Ok(json!({"frame": df_to_value(&df)?}))
    })
}

/// pandas IO concat.
#[no_mangle]
pub extern "C" fn polars__pd_concat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("frames")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `frames`"))?;
        let mut dfs: Vec<LazyFrame> = vec![];
        for f in arr {
            let df = parse_df(f)?;
            dfs.push(df.lazy());
        }
        if dfs.is_empty() {
            bail!("frames is empty");
        }
        let result = concat(dfs, UnionArgs::default())
            .context("concat")?
            .collect()
            .context("collect concat")?;
        Ok(json!({"frame": df_to_value(&result)?}))
    })
}

/// pandas IO merge.
#[no_mangle]
pub extern "C" fn polars__pd_merge(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let left = parse_df(args.get("left").ok_or_else(|| anyhow!("missing `left`"))?)?;
        let right = parse_df(
            args.get("right")
                .ok_or_else(|| anyhow!("missing `right`"))?,
        )?;
        let on = args
            .get("on")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `on`"))?;
        let result = left
            .lazy()
            .join(
                right.lazy(),
                [col(on)],
                [col(on)],
                JoinArgs::new(JoinType::Inner),
            )
            .collect()
            .context("merge")?;
        Ok(json!({"frame": df_to_value(&result)?}))
    })
}
