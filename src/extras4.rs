//! src/extras4.rs — bit manipulation, JSON helpers, format/conversion ops,
//! more array reshape/manipulation.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Value};

use crate::ffi_call;

fn scalar(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn get_i64(args: &Value, key: &str) -> Result<i64> {
    args.get(key)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow!("missing `{key}`"))
}

fn get_u64(args: &Value, key: &str) -> Result<u64> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing `{key}`"))
}

fn get_str(args: &Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| anyhow!("missing `{key}`"))
}

fn parse_array(v: &Value) -> Result<ArrayD<f64>> {
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("array `data`"))?;
    let shape = v
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or_else(|| anyhow!("array `shape`"))?;
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
    let a = args.get(key).ok_or_else(|| anyhow!("missing `{key}`"))?;
    parse_array(a)
}

// ── bit manipulation (bit_*) ──────────────────────────────────────────────

/// Bit popcount.
#[no_mangle]
pub extern "C" fn polars__bit_popcount(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"popcount": x.count_ones()}))
    })
}

/// Bit count zeros.
#[no_mangle]
pub extern "C" fn polars__bit_count_zeros(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"count_zeros": x.count_zeros()}))
    })
}

/// Bit leading zeros.
#[no_mangle]
pub extern "C" fn polars__bit_leading_zeros(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"leading_zeros": x.leading_zeros()}))
    })
}

/// Bit trailing zeros.
#[no_mangle]
pub extern "C" fn polars__bit_trailing_zeros(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"trailing_zeros": x.trailing_zeros()}))
    })
}

/// Bit leading ones.
#[no_mangle]
pub extern "C" fn polars__bit_leading_ones(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"leading_ones": x.leading_ones()}))
    })
}

/// Bit trailing ones.
#[no_mangle]
pub extern "C" fn polars__bit_trailing_ones(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"trailing_ones": x.trailing_ones()}))
    })
}

/// Bit reverse.
#[no_mangle]
pub extern "C" fn polars__bit_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"reverse": x.reverse_bits()}))
    })
}

/// Bit swap bytes.
#[no_mangle]
pub extern "C" fn polars__bit_swap_bytes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"swap_bytes": x.swap_bytes()}))
    })
}

/// Bit rotate left.
#[no_mangle]
pub extern "C" fn polars__bit_rotate_left(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        let n = get_u64(&args, "n")? as u32;
        Ok(json!({"rotate_left": x.rotate_left(n)}))
    })
}

/// Bit rotate right.
#[no_mangle]
pub extern "C" fn polars__bit_rotate_right(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        let n = get_u64(&args, "n")? as u32;
        Ok(json!({"rotate_right": x.rotate_right(n)}))
    })
}

/// Bit and.
#[no_mangle]
pub extern "C" fn polars__bit_and(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_i64(&args, "a")?;
        let b = get_i64(&args, "b")?;
        Ok(json!({"and": a & b}))
    })
}

/// Bit or.
#[no_mangle]
pub extern "C" fn polars__bit_or(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_i64(&args, "a")?;
        let b = get_i64(&args, "b")?;
        Ok(json!({"or": a | b}))
    })
}

/// Bit xor.
#[no_mangle]
pub extern "C" fn polars__bit_xor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_i64(&args, "a")?;
        let b = get_i64(&args, "b")?;
        Ok(json!({"xor": a ^ b}))
    })
}

/// Bit not.
#[no_mangle]
pub extern "C" fn polars__bit_not(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"not": !x}))
    })
}

/// Bit shl.
#[no_mangle]
pub extern "C" fn polars__bit_shl(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"shl": x << n}))
    })
}

/// Bit shr.
#[no_mangle]
pub extern "C" fn polars__bit_shr(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"shr": x >> n}))
    })
}

/// Bit test.
#[no_mangle]
pub extern "C" fn polars__bit_test(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"test": (x >> n) & 1 == 1}))
    })
}

/// Bit set.
#[no_mangle]
pub extern "C" fn polars__bit_set(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"set": x | (1 << n)}))
    })
}

/// Bit clear.
#[no_mangle]
pub extern "C" fn polars__bit_clear(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"clear": x & !(1 << n)}))
    })
}

/// Bit toggle.
#[no_mangle]
pub extern "C" fn polars__bit_toggle(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"toggle": x ^ (1 << n)}))
    })
}

/// Bit msb.
#[no_mangle]
pub extern "C" fn polars__bit_msb(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        if x == 0 {
            return Ok(json!({"msb": -1}));
        }
        Ok(json!({"msb": 63 - x.leading_zeros() as i64}))
    })
}

/// Bit lsb.
#[no_mangle]
pub extern "C" fn polars__bit_lsb(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        if x == 0 {
            return Ok(json!({"lsb": -1}));
        }
        Ok(json!({"lsb": x.trailing_zeros() as i64}))
    })
}

/// Bit is power of two.
#[no_mangle]
pub extern "C" fn polars__bit_is_power_of_two(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        Ok(json!({"is_power_of_two": x.is_power_of_two()}))
    })
}

/// Bit next power of two.
#[no_mangle]
pub extern "C" fn polars__bit_next_power_of_two(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        Ok(json!({"next_power_of_two": x.checked_next_power_of_two()}))
    })
}

/// Bit previous power of two: the largest power of two `<= x`. The downward
/// companion to `next_power_of_two`. 0 has no power-of-two floor and returns
/// null; otherwise it is `1 << floor(log2(x))`.
#[no_mangle]
pub extern "C" fn polars__bit_prev_power_of_two(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        let prev = if x == 0 {
            None
        } else {
            Some(1u64 << x.ilog2())
        };
        Ok(json!({ "prev_power_of_two": prev }))
    })
}

/// Bit to bin.
#[no_mangle]
pub extern "C" fn polars__bit_to_bin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"binary": format!("{:b}", x)}))
    })
}

/// Bit to hex.
#[no_mangle]
pub extern "C" fn polars__bit_to_hex(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"hex": format!("{:x}", x)}))
    })
}

/// Bit to oct.
#[no_mangle]
pub extern "C" fn polars__bit_to_oct(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"octal": format!("{:o}", x)}))
    })
}

/// Bit from bin.
#[no_mangle]
pub extern "C" fn polars__bit_from_bin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let x = i64::from_str_radix(&s, 2).context("from_bin")?;
        Ok(json!({"value": x}))
    })
}

/// Bit from hex.
#[no_mangle]
pub extern "C" fn polars__bit_from_hex(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let x = i64::from_str_radix(&s, 16).context("from_hex")?;
        Ok(json!({"value": x}))
    })
}

/// Bit from oct.
#[no_mangle]
pub extern "C" fn polars__bit_from_oct(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let x = i64::from_str_radix(&s, 8).context("from_oct")?;
        Ok(json!({"value": x}))
    })
}

// ── JSON helpers (json_*) ─────────────────────────────────────────────────

/// JSON parse.
#[no_mangle]
pub extern "C" fn polars__json_parse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let v: Value = serde_json::from_str(&s).context("parse json")?;
        Ok(json!({"value": v}))
    })
}

/// JSON stringify.
#[no_mangle]
pub extern "C" fn polars__json_stringify(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        Ok(json!({"text": serde_json::to_string(v)?}))
    })
}

/// JSON pretty.
#[no_mangle]
pub extern "C" fn polars__json_pretty(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        Ok(json!({"text": serde_json::to_string_pretty(v)?}))
    })
}

/// JSON keys.
#[no_mangle]
pub extern "C" fn polars__json_keys(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let keys: Vec<String> = v
            .as_object()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        Ok(json!({"keys": keys}))
    })
}

/// JSON values.
#[no_mangle]
pub extern "C" fn polars__json_values(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let vals: Vec<Value> = v
            .as_object()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default();
        Ok(json!({"values": vals}))
    })
}

/// JSON get.
#[no_mangle]
pub extern "C" fn polars__json_get(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let key = get_str(&args, "key")?;
        let r = v.get(&key).cloned().unwrap_or(Value::Null);
        Ok(json!({"result": r}))
    })
}

/// JSON has.
#[no_mangle]
pub extern "C" fn polars__json_has(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let key = get_str(&args, "key")?;
        Ok(json!({"has": v.get(&key).is_some()}))
    })
}

/// JSON type.
#[no_mangle]
pub extern "C" fn polars__json_type(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let t = match v {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        };
        Ok(json!({"type": t}))
    })
}

/// JSON is null.
#[no_mangle]
pub extern "C" fn polars__json_is_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        Ok(json!({"is_null": v.is_null()}))
    })
}

/// JSON merge.
#[no_mangle]
pub extern "C" fn polars__json_merge(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args.get("a").ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args.get("b").ok_or_else(|| anyhow!("missing `b`"))?;
        let mut out = a.as_object().cloned().unwrap_or_default();
        if let Some(obj) = b.as_object() {
            for (k, v) in obj {
                out.insert(k.clone(), v.clone());
            }
        }
        Ok(json!({"value": Value::Object(out)}))
    })
}

/// JSON len.
#[no_mangle]
pub extern "C" fn polars__json_len(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let n = match v {
            Value::Array(a) => a.len(),
            Value::Object(o) => o.len(),
            Value::String(s) => s.len(),
            _ => 0,
        };
        Ok(json!({"len": n}))
    })
}

/// JSON pluck.
#[no_mangle]
pub extern "C" fn polars__json_pluck(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("array")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `array`"))?;
        let key = get_str(&args, "key")?;
        let out: Vec<Value> = arr
            .iter()
            .map(|v| v.get(&key).cloned().unwrap_or(Value::Null))
            .collect();
        Ok(json!({"values": out}))
    })
}

/// JSON flatten.
#[no_mangle]
pub extern "C" fn polars__json_flatten(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("array")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `array`"))?;
        let mut out = vec![];
        for v in arr {
            if let Some(a) = v.as_array() {
                out.extend(a.iter().cloned());
            } else {
                out.push(v.clone());
            }
        }
        Ok(json!({"values": out}))
    })
}

/// JSON path.
#[no_mangle]
pub extern "C" fn polars__json_path(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let path = get_str(&args, "path")?;
        let mut cur = v;
        for part in path.split('.') {
            if part.is_empty() {
                continue;
            }
            cur = match cur.get(part) {
                Some(x) => x,
                None => return Ok(json!({"result": Value::Null})),
            };
        }
        Ok(json!({"result": cur.clone()}))
    })
}

// ── format/conversion (fmt_*) ─────────────────────────────────────────────

/// Format round.
#[no_mangle]
pub extern "C" fn polars__fmt_round(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let dp = args.get("dp").and_then(|v| v.as_i64()).unwrap_or(0);
        let mul = 10f64.powi(dp as i32);
        Ok(json!({"rounded": scalar((x * mul).round() / mul)}))
    })
}

/// Format floor.
#[no_mangle]
pub extern "C" fn polars__fmt_floor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        Ok(json!({"floor": scalar(x.floor())}))
    })
}

/// Format ceil.
#[no_mangle]
pub extern "C" fn polars__fmt_ceil(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        Ok(json!({"ceil": scalar(x.ceil())}))
    })
}

/// Format percent.
#[no_mangle]
pub extern "C" fn polars__fmt_percent(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let dp = args.get("dp").and_then(|v| v.as_i64()).unwrap_or(2) as usize;
        Ok(json!({"percent": format!("{:.*}%", dp, x * 100.0)}))
    })
}

/// Format currency.
#[no_mangle]
pub extern "C" fn polars__fmt_currency(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let sym = args.get("symbol").and_then(|v| v.as_str()).unwrap_or("$");
        Ok(json!({"currency": format!("{sym}{:.2}", x)}))
    })
}

/// Format scientific.
#[no_mangle]
pub extern "C" fn polars__fmt_scientific(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let dp = args.get("dp").and_then(|v| v.as_i64()).unwrap_or(2) as usize;
        Ok(json!({"scientific": format!("{:.*e}", dp, x)}))
    })
}

/// Format with commas.
#[no_mangle]
pub extern "C" fn polars__fmt_with_commas(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let s = x.abs().to_string();
        let mut out = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                out.push(',');
            }
            out.push(c);
        }
        let result: String = out.chars().rev().collect();
        let final_s = if x < 0 { format!("-{result}") } else { result };
        Ok(json!({"formatted": final_s}))
    })
}

/// Format human bytes.
#[no_mangle]
pub extern "C" fn polars__fmt_human_bytes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let bytes = args
            .get("bytes")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `bytes`"))? as f64;
        const SUFFIXES: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
        let mut size = bytes;
        let mut i = 0;
        while size >= 1024.0 && i < SUFFIXES.len() - 1 {
            size /= 1024.0;
            i += 1;
        }
        Ok(json!({"formatted": format!("{:.2} {}", size, SUFFIXES[i])}))
    })
}

/// Format a plain count compactly with SI magnitude suffixes — base-1000, so
/// `K`/`M`/`B`/`T` mean thousand/million/billion/trillion (distinct from
/// `human_bytes`, which is base-1024 byte sizes). 1500 → `1.5K`, 2_000_000 → `2M`,
/// −2500 → `-2.5K`, 999 → `999`. Trailing zeros and the point are trimmed
/// (`1000` → `1K`, not `1.0K`); rounding that reaches 1000 rolls up to the next
/// suffix (`999_999` → `1M`). opts: `n` (or `value`), `precision` (decimals, default
/// 1). Returns `{formatted}`.
#[no_mangle]
pub extern "C" fn polars__fmt_human_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .or_else(|| args.get("value"))
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let prec = args.get("precision").and_then(|v| v.as_u64()).unwrap_or(1) as i32;
        const SUF: &[&str] = &["", "K", "M", "B", "T"];
        let neg = n.is_sign_negative() && n != 0.0;
        let mut m = n.abs();
        let mut idx = 0;
        while m >= 1000.0 && idx < SUF.len() - 1 {
            m /= 1000.0;
            idx += 1;
        }
        let p = 10f64.powi(prec);
        let mut r = (m * p).round() / p;
        // Rounding can push the mantissa to 1000 (e.g. 999_999 → 1000K); roll up.
        if r >= 1000.0 && idx < SUF.len() - 1 {
            r /= 1000.0;
            idx += 1;
            r = (r * p).round() / p;
        }
        let mut digits = format!("{r:.*}", prec as usize);
        if digits.contains('.') {
            digits = digits
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string();
        }
        let formatted = format!("{}{}{}", if neg { "-" } else { "" }, digits, SUF[idx]);
        Ok(json!({ "formatted": formatted }))
    })
}

/// Parse a human byte size back to a count — the inverse of `fmt_human_bytes`.
/// Splits the leading number from a unit suffix and multiplies by the matching
/// power of 1024 (B=1, KB=1024, MB=1024², … PB), matching `human_bytes`'s
/// base-1024 scaling. The suffix is case-insensitive and the bare (`K`/`M`/…)
/// and `KiB`/`MiB` spellings are also accepted; a missing suffix means bytes.
/// `parse_bytes(human_bytes(n)) ≈ n` (within the two-decimal rounding
/// `human_bytes` applies). opts: `value` (or `formatted`). Returns `{bytes}`.
#[no_mangle]
pub extern "C" fn polars__fmt_parse_bytes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let raw = args
            .get("value")
            .or_else(|| args.get("formatted"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let s = raw.trim();
        let split = s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len());
        let (num_part, suffix_part) = s.split_at(split);
        let num: f64 = num_part
            .trim()
            .parse()
            .map_err(|_| anyhow!("invalid number in `{raw}`"))?;
        let power = match suffix_part.trim().to_ascii_uppercase().as_str() {
            "" | "B" => 0i32,
            "K" | "KB" | "KIB" => 1,
            "M" | "MB" | "MIB" => 2,
            "G" | "GB" | "GIB" => 3,
            "T" | "TB" | "TIB" => 4,
            "P" | "PB" | "PIB" => 5,
            other => return Err(anyhow!("unknown byte suffix `{other}`")),
        };
        let bytes = (num * 1024f64.powi(power)).round() as i64;
        Ok(json!({ "bytes": bytes }))
    })
}

/// Format human duration.
#[no_mangle]
pub extern "C" fn polars__fmt_human_duration(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let secs = args
            .get("seconds")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `seconds`"))?;
        let h = (secs / 3600.0).floor();
        let m = ((secs - h * 3600.0) / 60.0).floor();
        let s = secs - h * 3600.0 - m * 60.0;
        Ok(json!({"formatted": format!("{}h {}m {:.0}s", h as i64, m as i64, s)}))
    })
}

/// Parse a human duration string back to seconds — the inverse of
/// `human_duration`. A duration is a run of `<number><unit>` components (optional
/// whitespace between them); the units are `d`/`day(s)` (86400), `h`/`hr`/`hour(s)`
/// (3600), `m`/`min`/`minute(s)` (60), `s`/`sec`/`second(s)` (1) and `ms` (0.001),
/// summed. A bare trailing number with no unit is taken as seconds. So
/// `parse_duration("1h 30m 0s")` is `5400`, round-tripping `human_duration`.
#[no_mangle]
pub extern "C" fn polars__fmt_parse_duration(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let raw = args
            .get("value")
            .or_else(|| args.get("duration"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut total = 0.0f64;
        let mut num = String::new();
        let mut any = false;
        let mut chars = raw.chars().peekable();
        while let Some(c) = chars.next() {
            if c.is_ascii_whitespace() {
                continue;
            }
            if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' {
                num.push(c);
                continue;
            }
            let mut unit = String::new();
            unit.push(c.to_ascii_lowercase());
            while let Some(&n) = chars.peek() {
                if n.is_ascii_alphabetic() {
                    unit.push(n.to_ascii_lowercase());
                    chars.next();
                } else {
                    break;
                }
            }
            if num.is_empty() {
                return Err(anyhow!("unit `{unit}` without a number in: {raw}"));
            }
            let val: f64 = num
                .parse()
                .map_err(|_| anyhow!("bad number `{num}` in: {raw}"))?;
            num.clear();
            let mult = match unit.as_str() {
                "d" | "day" | "days" => 86400.0,
                "h" | "hr" | "hrs" | "hour" | "hours" => 3600.0,
                "m" | "min" | "mins" | "minute" | "minutes" => 60.0,
                "s" | "sec" | "secs" | "second" | "seconds" => 1.0,
                "ms" => 0.001,
                _ => return Err(anyhow!("unknown duration unit `{unit}` in: {raw}")),
            };
            total += val * mult;
            any = true;
        }
        if !num.is_empty() {
            total += num
                .parse::<f64>()
                .map_err(|_| anyhow!("bad number `{num}` in: {raw}"))?;
            any = true;
        }
        if !any {
            return Err(anyhow!("no duration components in: {raw}"));
        }
        Ok(json!({ "seconds": total }))
    })
}

/// Format ordinal.
#[no_mangle]
pub extern "C" fn polars__fmt_ordinal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let suffix = match (n % 100, n % 10) {
            (11..=13, _) => "th",
            (_, 1) => "st",
            (_, 2) => "nd",
            (_, 3) => "rd",
            _ => "th",
        };
        Ok(json!({"ordinal": format!("{n}{suffix}")}))
    })
}

/// Encode a number as a Roman numeral (subtractive notation). Shared by
/// `fmt_roman` and `fmt_from_roman` (which uses it to verify canonical form).
fn roman_encode(mut n: u64) -> String {
    let vals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut out = String::new();
    for (v, s) in &vals {
        while n >= *v {
            out.push_str(s);
            n -= v;
        }
    }
    out
}

/// Format roman.
#[no_mangle]
pub extern "C" fn polars__fmt_roman(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        Ok(json!({ "roman": roman_encode(n) }))
    })
}

/// Parse a Roman numeral back to a number — the inverse of `fmt_roman`. Accepts a
/// canonical subtractive numeral (case-insensitive); rejects unknown characters
/// and non-canonical forms (e.g. `IIII`, `VV`) by re-encoding and comparing, so
/// `from_roman(roman(n)) == n`. opts: `roman` (or `value`). Returns `{n}`.
#[no_mangle]
pub extern "C" fn polars__fmt_from_roman(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let raw = args
            .get("roman")
            .or_else(|| args.get("value"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `roman`"))?;
        let s = raw.trim().to_ascii_uppercase();
        if s.is_empty() {
            return Err(anyhow!("empty roman numeral"));
        }
        let val = |c: char| match c {
            'M' => 1000i64,
            'D' => 500,
            'C' => 100,
            'L' => 50,
            'X' => 10,
            'V' => 5,
            'I' => 1,
            _ => 0,
        };
        let chars: Vec<char> = s.chars().collect();
        let mut total: i64 = 0;
        for i in 0..chars.len() {
            let v = val(chars[i]);
            if v == 0 {
                return Err(anyhow!("invalid roman numeral character `{}`", chars[i]));
            }
            let next = chars.get(i + 1).copied().map(val).unwrap_or(0);
            if v < next {
                total -= v;
            } else {
                total += v;
            }
        }
        let n = u64::try_from(total).map_err(|_| anyhow!("invalid roman numeral `{raw}`"))?;
        if roman_encode(n) != s {
            return Err(anyhow!("non-canonical roman numeral `{raw}`"));
        }
        Ok(json!({ "n": n }))
    })
}

/// Format binary str.
#[no_mangle]
pub extern "C" fn polars__fmt_binary_str(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let width = args
            .get("width")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or(8);
        Ok(json!({"binary": format!("{:0>width$b}", n, width = width)}))
    })
}

/// Format pad left.
#[no_mangle]
pub extern "C" fn polars__fmt_pad_left(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let width = get_u64(&args, "width")? as usize;
        let pad = args
            .get("pad")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');
        let n = s.chars().count();
        let p: String = std::iter::repeat_n(pad, width.saturating_sub(n)).collect();
        Ok(json!({"value": format!("{p}{s}")}))
    })
}

/// Format pad right.
#[no_mangle]
pub extern "C" fn polars__fmt_pad_right(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let width = get_u64(&args, "width")? as usize;
        let pad = args
            .get("pad")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');
        let n = s.chars().count();
        let p: String = std::iter::repeat_n(pad, width.saturating_sub(n)).collect();
        Ok(json!({"value": format!("{s}{p}")}))
    })
}

// ── array reshape extras (arr_*) ──────────────────────────────────────────

/// ndarray flatten v2.
#[no_mangle]
pub extern "C" fn polars__arr_flatten_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), v).context("flatten")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray ravel.
#[no_mangle]
pub extern "C" fn polars__arr_ravel(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), v).context("ravel")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray split n.
#[no_mangle]
pub extern "C" fn polars__arr_split_n(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        if n == 0 || !v.len().is_multiple_of(n) {
            bail!("cannot split {} into {} equal parts", v.len(), n);
        }
        let chunk = v.len() / n;
        let chunks: Vec<Value> = v
            .chunks(chunk)
            .map(|c| {
                let arr = ArrayD::from_shape_vec(IxDyn(&[chunk]), c.to_vec()).unwrap();
                array_to_value(&arr)
            })
            .collect();
        Ok(json!({"parts": chunks}))
    })
}

/// ndarray concatenate v2.
#[no_mangle]
pub extern "C" fn polars__arr_concatenate_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arrs = args
            .get("arrays")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `arrays`"))?;
        let mut out = vec![];
        for a in arrs {
            let arr = parse_array(a)?;
            out.extend(arr.iter().copied());
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("concat")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray stack.
#[no_mangle]
pub extern "C" fn polars__arr_stack(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arrs = args
            .get("arrays")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `arrays`"))?;
        let parsed: Vec<ArrayD<f64>> = arrs.iter().map(parse_array).collect::<Result<_>>()?;
        if parsed.is_empty() {
            bail!("arrays is empty");
        }
        let cols = parsed[0].len();
        if !parsed.iter().all(|a| a.len() == cols) {
            bail!("all arrays must have same length");
        }
        let rows = parsed.len();
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[r * cols + c] = parsed[r].iter().nth(c).copied().unwrap_or(0.0);
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[rows, cols]), out).context("stack")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray vstack v2.
#[no_mangle]
pub extern "C" fn polars__arr_vstack_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arrs = args
            .get("arrays")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `arrays`"))?;
        let parsed: Vec<ArrayD<f64>> = arrs.iter().map(parse_array).collect::<Result<_>>()?;
        if parsed.is_empty() {
            bail!("empty");
        }
        let cols = parsed[0].shape().last().copied().unwrap_or(parsed[0].len());
        let mut all_rows = 0;
        let mut out = vec![];
        for a in &parsed {
            let rows = if a.shape().len() == 1 {
                1
            } else {
                a.shape()[0]
            };
            all_rows += rows;
            out.extend(a.iter().copied());
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[all_rows, cols]), out).context("vstack")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray hstack v2.
#[no_mangle]
pub extern "C" fn polars__arr_hstack_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arrs = args
            .get("arrays")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `arrays`"))?;
        let parsed: Vec<ArrayD<f64>> = arrs.iter().map(parse_array).collect::<Result<_>>()?;
        if parsed.is_empty() {
            bail!("empty");
        }
        if parsed[0].shape().len() == 1 {
            let mut out = vec![];
            for a in &parsed {
                out.extend(a.iter().copied());
            }
            let n = out.len();
            let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("hstack")?;
            return Ok(json!({"array": array_to_value(&arr)}));
        }
        let rows = parsed[0].shape()[0];
        let mut out_rows: Vec<Vec<f64>> = vec![vec![]; rows];
        for a in &parsed {
            let cols = a.shape()[1];
            for r in 0..rows {
                for c in 0..cols {
                    out_rows[r].push(a[[r, c].as_slice()]);
                }
            }
        }
        let total_cols = out_rows[0].len();
        let mut flat = vec![];
        for row in out_rows {
            flat.extend(row);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[rows, total_cols]), flat).context("hstack")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray dstack.
#[no_mangle]
pub extern "C" fn polars__arr_dstack(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arrs = args
            .get("arrays")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `arrays`"))?;
        let parsed: Vec<ArrayD<f64>> = arrs.iter().map(parse_array).collect::<Result<_>>()?;
        if parsed.is_empty() {
            bail!("empty");
        }
        // Stack along last axis.
        let first_shape = parsed[0].shape().to_vec();
        let depth = parsed.len();
        let mut new_shape = first_shape.clone();
        new_shape.push(depth);
        let mut out = vec![0.0; first_shape.iter().product::<usize>() * depth];
        let n_per = first_shape.iter().product::<usize>();
        for (d, a) in parsed.iter().enumerate() {
            for (i, v) in a.iter().enumerate() {
                out[i * depth + d] = *v;
            }
        }
        let _ = n_per;
        let arr = ArrayD::from_shape_vec(IxDyn(&new_shape), out).context("dstack")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray tile v2.
#[no_mangle]
pub extern "C" fn polars__arr_tile_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let reps = args.get("reps").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let mut out = Vec::with_capacity(v.len() * reps);
        for _ in 0..reps {
            out.extend(&v);
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("tile")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray repeat along.
#[no_mangle]
pub extern "C" fn polars__arr_repeat_along(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let repeats = args.get("repeats").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let mut out = vec![];
        for x in &v {
            for _ in 0..repeats {
                out.push(*x);
            }
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("repeat_along")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray atleast 1d v2.
#[no_mangle]
pub extern "C" fn polars__arr_atleast_1d_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        Ok(json!({"array": array_to_value(&a)}))
    })
}

/// ndarray atleast 2d v2.
#[no_mangle]
pub extern "C" fn polars__arr_atleast_2d_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() >= 2 {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let n = a.len();
        let v: Vec<f64> = a.iter().copied().collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, n]), v).context("atleast_2d")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray squeeze v2.
#[no_mangle]
pub extern "C" fn polars__arr_squeeze_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let new_shape: Vec<usize> = a.shape().iter().copied().filter(|d| *d > 1).collect();
        let new_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };
        let v: Vec<f64> = a.iter().copied().collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&new_shape), v).context("squeeze")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray expand dims.
#[no_mangle]
pub extern "C" fn polars__arr_expand_dims(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let axis = args.get("axis").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let mut new_shape: Vec<usize> = a.shape().to_vec();
        if axis > new_shape.len() {
            bail!("axis out of range");
        }
        new_shape.insert(axis, 1);
        let v: Vec<f64> = a.iter().copied().collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&new_shape), v).context("expand_dims")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray swap axes.
#[no_mangle]
pub extern "C" fn polars__arr_swap_axes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("swap_axes: 2-D only in this slice");
        }
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = a[[r, c].as_slice()];
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[cols, rows]), out).context("swap_axes")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray moveaxis.
#[no_mangle]
pub extern "C" fn polars__arr_moveaxis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        // Restricted to 2-D for simplicity.
        if a.shape().len() != 2 {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = a[[r, c].as_slice()];
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[cols, rows]), out).context("moveaxis")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// ndarray argmax axis v2.
#[no_mangle]
pub extern "C" fn polars__arr_argmax_axis_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("2-D only");
        }
        let axis = args.get("axis").and_then(|v| v.as_u64()).unwrap_or(0);
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let out: Vec<i64> = if axis == 0 {
            (0..cols)
                .map(|c| {
                    (0..rows)
                        .map(|r| (r, a[[r, c].as_slice()]))
                        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as i64)
                        .unwrap_or(0)
                })
                .collect()
        } else {
            (0..rows)
                .map(|r| {
                    (0..cols)
                        .map(|c| (c, a[[r, c].as_slice()]))
                        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as i64)
                        .unwrap_or(0)
                })
                .collect()
        };
        Ok(json!({"argmax": out}))
    })
}

/// ndarray argmin axis v2.
#[no_mangle]
pub extern "C" fn polars__arr_argmin_axis_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        if a.shape().len() != 2 {
            bail!("2-D only");
        }
        let axis = args.get("axis").and_then(|v| v.as_u64()).unwrap_or(0);
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let out: Vec<i64> = if axis == 0 {
            (0..cols)
                .map(|c| {
                    (0..rows)
                        .map(|r| (r, a[[r, c].as_slice()]))
                        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as i64)
                        .unwrap_or(0)
                })
                .collect()
        } else {
            (0..rows)
                .map(|r| {
                    (0..cols)
                        .map(|c| (c, a[[r, c].as_slice()]))
                        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as i64)
                        .unwrap_or(0)
                })
                .collect()
        };
        Ok(json!({"argmin": out}))
    })
}

// ── checksums (sum_*) ─────────────────────────────────────────────────────

/// Checksum xor.
#[no_mangle]
pub extern "C" fn polars__sum_xor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let mut x: u8 = 0;
        for b in s.bytes() {
            x ^= b;
        }
        Ok(json!({"checksum": x}))
    })
}

/// Checksum simple.
#[no_mangle]
pub extern "C" fn polars__sum_simple(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let total: u64 = s.bytes().map(|b| b as u64).sum();
        Ok(json!({"checksum": total}))
    })
}

/// Checksum adler32.
#[no_mangle]
pub extern "C" fn polars__sum_adler32(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for byte in s.bytes() {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        Ok(json!({"checksum": (b << 16) | a}))
    })
}

/// Checksum bsd16.
#[no_mangle]
pub extern "C" fn polars__sum_bsd16(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let mut c: u16 = 0;
        for b in s.bytes() {
            c = ((c >> 1) | ((c & 1) << 15)).wrapping_add(b as u16);
        }
        Ok(json!({"checksum": c}))
    })
}

/// Checksum fletcher16 — Fletcher's 16-bit checksum (RFC 1146): two running
/// 8-bit sums mod 255, combined as `(sum2 << 8) | sum1`. Distinct from adler32
/// (mod 65521) and bsd16 (rotate-and-add).
#[no_mangle]
pub extern "C" fn polars__sum_fletcher16(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let mut sum1: u16 = 0;
        let mut sum2: u16 = 0;
        for b in s.bytes() {
            sum1 = (sum1 + b as u16) % 255;
            sum2 = (sum2 + sum1) % 255;
        }
        Ok(json!({"checksum": (sum2 << 8) | sum1}))
    })
}

/// Checksum fletcher32 — Fletcher's 32-bit checksum: two running 16-bit sums
/// mod 65535 over little-endian 16-bit words (an odd final byte is zero-padded),
/// combined as `(sum2 << 16) | sum1`. The 32-bit companion to `fletcher16` and
/// distinct from adler32 (mod 65521).
#[no_mangle]
pub extern "C" fn polars__sum_fletcher32(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let bytes = s.as_bytes();
        let mut sum1: u32 = 0;
        let mut sum2: u32 = 0;
        let mut i = 0;
        while i < bytes.len() {
            let lo = bytes[i] as u32;
            let hi = if i + 1 < bytes.len() {
                bytes[i + 1] as u32
            } else {
                0
            };
            let word = lo | (hi << 8);
            sum1 = (sum1 + word) % 65535;
            sum2 = (sum2 + sum1) % 65535;
            i += 2;
        }
        Ok(json!({"checksum": (sum2 << 16) | sum1}))
    })
}

/// Checksum internet.
#[no_mangle]
pub extern "C" fn polars__sum_internet(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let bytes = s.as_bytes();
        let mut sum: u32 = 0;
        let mut i = 0;
        while i + 1 < bytes.len() {
            sum += ((bytes[i] as u32) << 8) | bytes[i + 1] as u32;
            i += 2;
        }
        if i < bytes.len() {
            sum += (bytes[i] as u32) << 8;
        }
        while sum > 0xffff {
            sum = (sum >> 16) + (sum & 0xffff);
        }
        let checksum = !sum as u16;
        Ok(json!({"checksum": checksum}))
    })
}

// ── more (misc_*) ─────────────────────────────────────────────────────────

/// Math clamp.
#[no_mangle]
pub extern "C" fn polars__misc_clamp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let lo = args
            .get("min")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let hi = args
            .get("max")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::INFINITY);
        Ok(json!({"clamped": scalar(x.clamp(lo, hi))}))
    })
}

/// Math lerp.
#[no_mangle]
pub extern "C" fn polars__misc_lerp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let t = args
            .get("t")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `t`"))?;
        Ok(json!({"value": scalar(a + (b - a) * t)}))
    })
}

/// Math smoothstep.
#[no_mangle]
pub extern "C" fn polars__misc_smoothstep(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let edge0 = args.get("edge0").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let edge1 = args.get("edge1").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        Ok(json!({"value": scalar(t * t * (3.0 - 2.0 * t))}))
    })
}

/// Math map.
#[no_mangle]
pub extern "C" fn polars__misc_map(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let in_lo = args
            .get("in_min")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `in_min`"))?;
        let in_hi = args
            .get("in_max")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `in_max`"))?;
        let out_lo = args
            .get("out_min")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `out_min`"))?;
        let out_hi = args
            .get("out_max")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `out_max`"))?;
        let r = (x - in_lo) / (in_hi - in_lo) * (out_hi - out_lo) + out_lo;
        Ok(json!({"value": scalar(r)}))
    })
}

/// Math sign.
#[no_mangle]
pub extern "C" fn polars__misc_sign(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        Ok(json!({"sign": scalar(x.signum())}))
    })
}

/// Math step.
#[no_mangle]
pub extern "C" fn polars__misc_step(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let edge = args.get("edge").and_then(|v| v.as_f64()).unwrap_or(0.0);
        Ok(json!({"value": scalar(if x < edge { 0.0 } else { 1.0 })}))
    })
}

/// Math safe div.
#[no_mangle]
pub extern "C" fn polars__misc_safe_div(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let default = args.get("default").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let r = if b == 0.0 { default } else { a / b };
        Ok(json!({"value": scalar(r)}))
    })
}

/// Math safe log.
#[no_mangle]
pub extern "C" fn polars__misc_safe_log(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        let eps = args.get("eps").and_then(|v| v.as_f64()).unwrap_or(1e-12);
        Ok(json!({"value": scalar((x + eps).ln())}))
    })
}

/// Math safe sqrt.
#[no_mangle]
pub extern "C" fn polars__misc_safe_sqrt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = args
            .get("x")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x`"))?;
        Ok(json!({"value": scalar(x.max(0.0).sqrt())}))
    })
}

/// Math factorial.
#[no_mangle]
pub extern "C" fn polars__misc_factorial(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let mut r: u64 = 1;
        for i in 2..=n {
            r = r.saturating_mul(i);
        }
        Ok(json!({"factorial": r}))
    })
}

/// Math choose.
#[no_mangle]
pub extern "C" fn polars__misc_choose(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        if k > n {
            return Ok(json!({"choose": 0}));
        }
        let k = k.min(n - k);
        let mut r: u64 = 1;
        for i in 0..k {
            r = r.saturating_mul(n - i).saturating_div((i + 1).max(1));
        }
        Ok(json!({"choose": r}))
    })
}

/// Math permute.
#[no_mangle]
pub extern "C" fn polars__misc_permute(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        if k > n {
            return Ok(json!({"permute": 0}));
        }
        let mut r: u64 = 1;
        for i in 0..k {
            r = r.saturating_mul(n - i);
        }
        Ok(json!({"permute": r}))
    })
}

/// Math fibonacci.
#[no_mangle]
pub extern "C" fn polars__misc_fibonacci(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        if n == 0 {
            return Ok(json!({"value": 0}));
        }
        if n == 1 {
            return Ok(json!({"value": 1}));
        }
        let mut a = 0_u64;
        let mut b = 1_u64;
        for _ in 1..n {
            let c = a.saturating_add(b);
            a = b;
            b = c;
        }
        Ok(json!({"value": b}))
    })
}

/// Math is prime.
#[no_mangle]
pub extern "C" fn polars__misc_is_prime(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        if n < 2 {
            return Ok(json!({"is_prime": false}));
        }
        if n < 4 {
            return Ok(json!({"is_prime": true}));
        }
        if n % 2 == 0 {
            return Ok(json!({"is_prime": false}));
        }
        let limit = (n as f64).sqrt() as u64;
        let mut i = 3;
        while i <= limit {
            if n % i == 0 {
                return Ok(json!({"is_prime": false}));
            }
            i += 2;
        }
        Ok(json!({"is_prime": true}))
    })
}

/// Math gcd.
#[no_mangle]
pub extern "C" fn polars__misc_gcd(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut a = args
            .get("a")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `a`"))?
            .abs();
        let mut b = args
            .get("b")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `b`"))?
            .abs();
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        Ok(json!({"gcd": a}))
    })
}

/// Math lcm.
#[no_mangle]
pub extern "C" fn polars__misc_lcm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args
            .get("a")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `a`"))?
            .abs();
        let b = args
            .get("b")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `b`"))?
            .abs();
        if a == 0 || b == 0 {
            return Ok(json!({"lcm": 0}));
        }
        let mut x = a;
        let mut y = b;
        while y != 0 {
            let t = y;
            y = x % y;
            x = t;
        }
        Ok(json!({"lcm": (a * b) / x}))
    })
}

/// Math modpow.
#[no_mangle]
pub extern "C" fn polars__misc_modpow(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut base = args
            .get("base")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `base`"))?;
        let mut exp = args
            .get("exp")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `exp`"))?;
        let m = args
            .get("m")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `m`"))?;
        if m == 1 {
            return Ok(json!({"value": 0}));
        }
        let mut result: u128 = 1;
        base %= m;
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base as u128) % m as u128;
            }
            exp /= 2;
            base = ((base as u128 * base as u128) % m as u128) as u64;
        }
        Ok(json!({"value": result as u64}))
    })
}

/// Math digits.
#[no_mangle]
pub extern "C" fn polars__misc_digits(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let base = args.get("base").and_then(|v| v.as_u64()).unwrap_or(10);
        let mut x = n;
        let mut digits = vec![];
        if x == 0 {
            digits.push(0);
        }
        while x > 0 {
            digits.push((x % base) as i64);
            x /= base;
        }
        digits.reverse();
        Ok(json!({"digits": digits}))
    })
}

/// Math digit sum.
#[no_mangle]
pub extern "C" fn polars__misc_digit_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let mut s = 0;
        while n > 0 {
            s += (n % 10) as i64;
            n /= 10;
        }
        Ok(json!({"sum": s}))
    })
}

/// Math reverse number.
#[no_mangle]
pub extern "C" fn polars__misc_reverse_number(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
        let mut r: u64 = 0;
        while n > 0 {
            r = r * 10 + (n % 10);
            n /= 10;
        }
        Ok(json!({"value": r}))
    })
}

/// Math is palindrome.
#[no_mangle]
pub extern "C" fn polars__misc_is_palindrome(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let r: String = s.chars().rev().collect();
        Ok(json!({"is_palindrome": s == r}))
    })
}

/// Math unique chars.
#[no_mangle]
pub extern "C" fn polars__misc_unique_chars(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let set: std::collections::HashSet<char> = s.chars().collect();
        Ok(json!({"count": set.len()}))
    })
}

/// Math char frequency.
#[no_mangle]
pub extern "C" fn polars__misc_char_frequency(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let mut counts: std::collections::HashMap<char, u64> = std::collections::HashMap::new();
        for c in s.chars() {
            *counts.entry(c).or_insert(0) += 1;
        }
        let arr: Vec<Value> = counts
            .into_iter()
            .map(|(k, v)| json!({"char": k.to_string(), "count": v}))
            .collect();
        Ok(json!({"frequency": arr}))
    })
}

/// Canonicalize and categorize a Polars dtype name. Pure — no DataFrame.
///
/// Args: `{"dtype": "i64"}`. Returns `{dtype, canonical, numeric, integer,
/// float, signed, temporal}`. Accepts the common aliases (`i64`/`int64`/`int`,
/// `f64`/`float64`/`float`, `str`/`string`/`utf8`, …). Errors on an unknown name.
#[no_mangle]
pub extern "C" fn polars__parse_dtype(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let name = get_str(&args, "dtype")?;
        let canonical = match name.to_ascii_lowercase().as_str() {
            "i8" | "int8" => "Int8",
            "i16" | "int16" => "Int16",
            "i32" | "int32" => "Int32",
            "i64" | "int64" | "int" => "Int64",
            "u8" | "uint8" => "UInt8",
            "u16" | "uint16" => "UInt16",
            "u32" | "uint32" => "UInt32",
            "u64" | "uint64" => "UInt64",
            "f32" | "float32" => "Float32",
            "f64" | "float64" | "float" => "Float64",
            "bool" | "boolean" => "Boolean",
            "str" | "string" | "utf8" => "String",
            "binary" => "Binary",
            "date" => "Date",
            "datetime" => "Datetime",
            "time" => "Time",
            "duration" => "Duration",
            "cat" | "categorical" => "Categorical",
            "null" => "Null",
            other => return Err(anyhow!("unknown dtype `{other}`")),
        };
        let integer = matches!(
            canonical,
            "Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32" | "UInt64"
        );
        let float = matches!(canonical, "Float32" | "Float64");
        let signed = matches!(
            canonical,
            "Int8" | "Int16" | "Int32" | "Int64" | "Float32" | "Float64"
        );
        let temporal = matches!(canonical, "Date" | "Datetime" | "Time" | "Duration");
        Ok(json!({
            "dtype": name,
            "canonical": canonical,
            "numeric": integer || float,
            "integer": integer,
            "float": float,
            "signed": signed,
            "temporal": temporal,
        }))
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::ffi_test::call;

    use super::*;

    #[test]
    fn factorial_matches_table() {
        // 20! is the largest value that still fits in u64
        // (2_432_902_008_176_640_000) — the saturation boundary.
        let cases = [(0u64, 1u64), (1, 1), (5, 120), (10, 3_628_800)];
        for (n, expect) in cases {
            let v = call(polars__misc_factorial, json!({"n": n}));
            assert_eq!(v["factorial"].as_u64().unwrap(), expect, "{n}!");
        }
        let v = call(polars__misc_factorial, json!({"n": 20u64}));
        assert_eq!(v["factorial"].as_u64().unwrap(), 2_432_902_008_176_640_000);
    }

    #[test]
    fn choose_matches_pascal_row() {
        // C(n,k) == C(n,n-k); standard Pascal-row values.
        let v = call(polars__misc_choose, json!({"n": 5u64, "k": 2u64}));
        assert_eq!(v["choose"].as_u64().unwrap(), 10);
        let v = call(polars__misc_choose, json!({"n": 5u64, "k": 3u64}));
        assert_eq!(v["choose"].as_u64().unwrap(), 10);
        let v = call(polars__misc_choose, json!({"n": 10u64, "k": 5u64}));
        assert_eq!(v["choose"].as_u64().unwrap(), 252);
        let v = call(polars__misc_choose, json!({"n": 5u64, "k": 10u64}));
        assert_eq!(v["choose"].as_u64().unwrap(), 0, "k > n returns 0");
    }

    #[test]
    fn fibonacci_matches_table() {
        let cases = [
            (0u64, 0u64),
            (1, 1),
            (2, 1),
            (10, 55),
            (20, 6765),
            (30, 832_040),
        ];
        for (n, expect) in cases {
            let v = call(polars__misc_fibonacci, json!({"n": n}));
            assert_eq!(v["value"].as_u64().unwrap(), expect, "F({n})");
        }
    }

    #[test]
    fn is_prime_edge_cases() {
        // 0/1 are not prime; 2 is the only even prime; 25 and 49 sit on the
        // sqrt boundary that a naive loop bound can miss.
        let primes = [2u64, 3, 5, 7, 11, 13, 17, 97, 1009];
        let composites = [0u64, 1, 4, 6, 25, 49, 100, 1000];
        for n in primes {
            let v = call(polars__misc_is_prime, json!({"n": n}));
            assert_eq!(v["is_prime"], true, "{n} prime");
        }
        for n in composites {
            let v = call(polars__misc_is_prime, json!({"n": n}));
            assert_eq!(v["is_prime"], false, "{n} composite");
        }
    }

    #[test]
    fn gcd_lcm_relationship_holds() {
        // gcd(a, b) * lcm(a, b) == a * b for any non-zero pair.
        for (a, b) in [(12i64, 18), (7, 13), (100, 75), (1, 5)] {
            let g = call(polars__misc_gcd, json!({"a": a, "b": b}));
            let l = call(polars__misc_lcm, json!({"a": a, "b": b}));
            let gv = g["gcd"].as_i64().unwrap();
            let lv = l["lcm"].as_i64().unwrap();
            assert_eq!(gv * lv, a * b, "gcd*lcm == a*b for ({a},{b})");
        }
    }

    #[test]
    fn modpow_matches_known_values() {
        let v = call(
            polars__misc_modpow,
            json!({"base": 2u64, "exp": 10u64, "m": 1000u64}),
        );
        assert_eq!(v["value"].as_u64().unwrap(), 24);
        // 7^256 mod 13: by Fermat 7^12 ≡ 1, so 7^256 ≡ 7^4 = 2401 ≡ 9.
        let v = call(
            polars__misc_modpow,
            json!({"base": 7u64, "exp": 256u64, "m": 13u64}),
        );
        assert_eq!(v["value"].as_u64().unwrap(), 9);
    }

    #[test]
    fn palindrome_recognizes_known_strings() {
        for s in ["racecar", "level", "a", ""] {
            let v = call(polars__misc_is_palindrome, json!({"value": s}));
            assert_eq!(v["is_palindrome"], true, "{s}");
        }
        for s in ["hello", "world", "ab"] {
            let v = call(polars__misc_is_palindrome, json!({"value": s}));
            assert_eq!(v["is_palindrome"], false, "{s}");
        }
    }

    #[test]
    fn digit_sum_is_a_divisibility_witness_for_9() {
        // The classic divisibility-by-9 invariant: digit_sum(n) ≡ n (mod 9).
        for n in [0u64, 7, 99, 1234, 99_999, 1_000_000] {
            let v = call(polars__misc_digit_sum, json!({"n": n}));
            let s = v["sum"].as_i64().unwrap() as u64;
            assert_eq!(s % 9, n % 9, "digit_sum({n}) ≢ {n} (mod 9)");
        }
    }

    #[test]
    fn bit_popcount_matches_table() {
        let cases = [(0i64, 0u32), (1, 1), (7, 3), (255, 8), (-1, 64)];
        for (x, expect) in cases {
            let v = call(polars__bit_popcount, json!({"x": x}));
            assert_eq!(
                v["popcount"].as_u64().unwrap(),
                expect as u64,
                "popcount({x})"
            );
        }
    }

    #[test]
    fn bit_prev_power_of_two_floors_to_a_power_of_two() {
        // The largest power of two <= x.
        let cases = [
            (1u64, Some(1u64)),
            (2, Some(2)),
            (3, Some(2)),
            (4, Some(4)),
            (5, Some(4)),
            (7, Some(4)),
            (8, Some(8)),
            (1000, Some(512)),
            (u64::MAX, Some(1u64 << 63)),
        ];
        for (x, expect) in cases {
            let v = call(polars__bit_prev_power_of_two, json!({ "x": x }));
            assert_eq!(
                v["prev_power_of_two"].as_u64(),
                expect,
                "prev_power_of_two({x})"
            );
        }
        // 0 has no power-of-two floor → null.
        let z = call(polars__bit_prev_power_of_two, json!({ "x": 0u64 }));
        assert!(
            z["prev_power_of_two"].is_null(),
            "prev_power_of_two(0) is null"
        );
        // Pairs with next_power_of_two: for an exact power, both agree.
        for &p in &[1u64, 2, 16, 1024] {
            let prev = call(polars__bit_prev_power_of_two, json!({ "x": p }));
            let next = call(polars__bit_next_power_of_two, json!({ "x": p }));
            assert_eq!(
                prev["prev_power_of_two"], next["next_power_of_two"],
                "exact power {p}"
            );
        }
    }

    #[test]
    fn bit_toggle_is_self_inverse() {
        // Setting bit 3 of 0 = 8; clearing it = 0; toggling twice = identity.
        let v = call(polars__bit_set, json!({"x": 0i64, "n": 3u64}));
        assert_eq!(v["set"].as_i64().unwrap(), 8);
        let v = call(polars__bit_clear, json!({"x": 8i64, "n": 3u64}));
        assert_eq!(v["clear"].as_i64().unwrap(), 0);
        let v = call(polars__bit_toggle, json!({"x": 0i64, "n": 3u64}));
        let toggled = v["toggle"].as_i64().unwrap();
        let v = call(polars__bit_toggle, json!({"x": toggled, "n": 3u64}));
        assert_eq!(v["toggle"].as_i64().unwrap(), 0);
    }

    #[test]
    fn bit_radix_round_trips() {
        for &x in &[0i64, 1, 255, 1023, 65_535] {
            let bin = call(polars__bit_to_bin, json!({"x": x}));
            let back = call(
                polars__bit_from_bin,
                json!({"value": bin["binary"].as_str().unwrap()}),
            );
            assert_eq!(back["value"].as_i64().unwrap(), x, "binary {x}");

            let hex = call(polars__bit_to_hex, json!({"x": x}));
            let back = call(
                polars__bit_from_hex,
                json!({"value": hex["hex"].as_str().unwrap()}),
            );
            assert_eq!(back["value"].as_i64().unwrap(), x, "hex {x}");

            let oct = call(polars__bit_to_oct, json!({"x": x}));
            let back = call(
                polars__bit_from_oct,
                json!({"value": oct["octal"].as_str().unwrap()}),
            );
            assert_eq!(back["value"].as_i64().unwrap(), x, "oct {x}");
        }
    }

    #[test]
    fn fmt_roman_handles_subtractive_pairs() {
        // The subtractive cases (CM, CD, XC, XL, IX, IV) are the only places
        // a greedy converter can go wrong.
        let cases = [
            (1u64, "I"),
            (4, "IV"),
            (9, "IX"),
            (40, "XL"),
            (49, "XLIX"),
            (90, "XC"),
            (400, "CD"),
            (900, "CM"),
            (1994, "MCMXCIV"),
            (2024, "MMXXIV"),
            (3999, "MMMCMXCIX"),
        ];
        for (n, expect) in cases {
            let v = call(polars__fmt_roman, json!({"n": n}));
            assert_eq!(v["roman"].as_str().unwrap(), expect, "roman({n})");
        }
    }

    #[test]
    fn fmt_from_roman_inverts_fmt_roman() {
        // Round-trips every subtractive case and the full 1..=3999 range.
        let cases = [
            (1u64, "I"),
            (4, "IV"),
            (49, "XLIX"),
            (1994, "MCMXCIV"),
            (3999, "MMMCMXCIX"),
        ];
        for (n, roman) in cases {
            let v = call(polars__fmt_from_roman, json!({ "roman": roman }));
            assert_eq!(v["n"].as_u64().unwrap(), n, "from_roman({roman})");
        }
        for n in 1u64..=3999 {
            let roman = call(polars__fmt_roman, json!({ "n": n }));
            let back = call(
                polars__fmt_from_roman,
                json!({ "roman": roman["roman"].as_str().unwrap() }),
            );
            assert_eq!(back["n"].as_u64().unwrap(), n, "round-trip {n}");
        }
        // Case-insensitive.
        assert_eq!(
            call(polars__fmt_from_roman, json!({ "roman": "xiv" }))["n"]
                .as_u64()
                .unwrap(),
            14
        );
        // Unknown characters, non-canonical forms, and empty input are rejected.
        for bad in ["ABC", "IIII", "VV", ""] {
            let v = call(polars__fmt_from_roman, json!({ "roman": bad }));
            assert!(v.get("error").is_some(), "from_roman({bad:?}) should error");
        }
    }

    #[test]
    fn fmt_ordinal_handles_teens() {
        // 11/12/13 use "th", not the digit-based st/nd/rd — the standard bug magnet.
        let cases = [
            (1i64, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (21, "21st"),
            (22, "22nd"),
            (23, "23rd"),
            (101, "101st"),
            (111, "111th"),
            (112, "112th"),
        ];
        for (n, expect) in cases {
            let v = call(polars__fmt_ordinal, json!({"n": n}));
            assert_eq!(v["ordinal"].as_str().unwrap(), expect, "ordinal({n})");
        }
    }

    #[test]
    fn fmt_parse_bytes_inverts_human_bytes() {
        let b = |s: &str| {
            call(polars__fmt_parse_bytes, json!({ "value": s }))["bytes"]
                .as_i64()
                .unwrap()
        };
        // Base-1024 scaling, matching human_bytes.
        assert_eq!(b("1.5 MB"), 1_572_864);
        assert_eq!(b("1 KB"), 1024);
        assert_eq!(b("2 GB"), 2 * 1024 * 1024 * 1024);
        // A bare number is bytes; the suffix is case-insensitive and space-optional.
        assert_eq!(b("512"), 512);
        assert_eq!(b("512b"), 512);
        assert_eq!(b("3mb"), 3 * 1024 * 1024);
        // Bare and -iB spellings map to the same power.
        assert_eq!(b("4M"), b("4 MB"));
        assert_eq!(b("4 MiB"), b("4 MB"));
        // Round-trips human_bytes within its 2-decimal rounding for clean values.
        for n in [0i64, 1024, 1_572_864, 5 * 1024 * 1024 * 1024] {
            let formatted = call(polars__fmt_human_bytes, json!({ "bytes": n }))["formatted"]
                .as_str()
                .unwrap()
                .to_string();
            assert_eq!(b(&formatted), n, "round-trip {n}");
        }
        // An unknown suffix and a non-numeric prefix are rejected.
        assert!(call(polars__fmt_parse_bytes, json!({ "value": "5 ZB" }))
            .get("error")
            .is_some());
        assert!(call(polars__fmt_parse_bytes, json!({ "value": "MB" }))
            .get("error")
            .is_some());
        assert!(call(polars__fmt_parse_bytes, json!({}))
            .get("error")
            .is_some());
    }

    #[test]
    fn fmt_human_count_compacts_with_si_suffixes() {
        let c = |n: f64| {
            call(polars__fmt_human_count, json!({ "n": n }))["formatted"]
                .as_str()
                .unwrap()
                .to_string()
        };
        // Base-1000 SI magnitudes (distinct from human_bytes' base-1024).
        assert_eq!(c(1500.0), "1.5K");
        assert_eq!(c(1000.0), "1K");
        assert_eq!(c(999.0), "999");
        assert_eq!(c(2_000_000.0), "2M");
        assert_eq!(c(1_234_567.0), "1.2M");
        assert_eq!(c(1_500_000_000.0), "1.5B");
        assert_eq!(c(1_000_000_000_000.0), "1T");
        // Negatives keep the sign; zero is bare.
        assert_eq!(c(-2500.0), "-2.5K");
        assert_eq!(c(0.0), "0");
        // Rounding that reaches 1000 rolls up to the next suffix.
        assert_eq!(c(999_999.0), "1M");
        // Sub-1000 fractional values keep their decimals (trimmed).
        assert_eq!(c(12.5), "12.5");
        // precision controls the mantissa decimals.
        assert_eq!(
            call(
                polars__fmt_human_count,
                json!({ "n": 1234, "precision": 2 })
            )["formatted"],
            "1.23K"
        );
        assert!(call(polars__fmt_human_count, json!({}))
            .get("error")
            .is_some());
    }

    #[test]
    fn fmt_parse_duration_inverts_human_duration() {
        let d = |s: &str| {
            call(polars__fmt_parse_duration, json!({ "value": s }))["seconds"]
                .as_f64()
                .unwrap()
        };
        // The human_duration output form round-trips.
        assert_eq!(d("1h 30m 0s"), 5400.0);
        assert_eq!(d("2h 5m 30s"), 7530.0);
        // Whitespace optional; multiple unit spellings; days and ms.
        assert_eq!(d("90m"), 5400.0);
        assert_eq!(d("1h30m"), 5400.0);
        assert_eq!(d("1d"), 86400.0);
        assert_eq!(d("500ms"), 0.5);
        assert_eq!(d("2 hours 15 minutes"), 8100.0);
        // A bare number is seconds.
        assert_eq!(d("45"), 45.0);
        // Round-trips human_duration for whole-second inputs.
        for secs in [0i64, 59, 60, 3661, 7530] {
            let formatted = call(polars__fmt_human_duration, json!({ "seconds": secs }))
                ["formatted"]
                .as_str()
                .unwrap()
                .to_string();
            assert_eq!(d(&formatted), secs as f64, "round-trip {secs}");
        }
        // An unknown unit, a unit with no number, and a missing arg are rejected.
        assert!(call(polars__fmt_parse_duration, json!({ "value": "5x" }))
            .get("error")
            .is_some());
        assert!(call(polars__fmt_parse_duration, json!({ "value": "h" }))
            .get("error")
            .is_some());
        assert!(call(polars__fmt_parse_duration, json!({}))
            .get("error")
            .is_some());
    }

    #[test]
    fn fmt_with_commas_handles_negative_and_zero() {
        for (n, expect) in [
            (0i64, "0"),
            (1_000, "1,000"),
            (1_000_000, "1,000,000"),
            (-1_234_567, "-1,234,567"),
        ] {
            let v = call(polars__fmt_with_commas, json!({"x": n}));
            assert_eq!(v["formatted"].as_str().unwrap(), expect);
        }
    }

    #[test]
    fn json_path_resolves_nested_keys() {
        let v = call(
            polars__json_path,
            json!({
                "value": {"a": {"b": {"c": 42}}},
                "path": "a.b.c",
            }),
        );
        assert_eq!(v["result"], 42);
        let v = call(polars__json_path, json!({"value": {"a": 1}, "path": "x.y"}));
        assert!(v["result"].is_null());
    }

    #[test]
    fn arr_concatenate_v2_flattens_in_input_order() {
        // The variadic version takes an array-of-arrays and produces one
        // flat 1-D vector in declaration order. Order preservation across
        // ragged inputs is the load-bearing invariant.
        let v = call(
            polars__arr_concatenate_v2,
            json!({
                "arrays": [
                    {"data": [1.0, 2.0], "shape": [2]},
                    {"data": [3.0], "shape": [1]},
                    {"data": [4.0, 5.0, 6.0], "shape": [3]},
                ],
            }),
        );
        let data = v["array"]["data"].as_array().unwrap();
        let nums: Vec<f64> = data.iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(nums, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn sum_adler32_known_vector() {
        // Adler32("Wikipedia") = 0x11E60398 — RFC 1950 test vector.
        let v = call(polars__sum_adler32, json!({"value": "Wikipedia"}));
        assert_eq!(v["checksum"].as_u64().unwrap(), 0x11E60398);
    }

    #[test]
    fn sum_fletcher16_known_vectors() {
        // Fletcher-16("abcde") = 0xC8F0 — the Wikipedia worked example.
        assert_eq!(
            call(polars__sum_fletcher16, json!({"value": "abcde"}))["checksum"]
                .as_u64()
                .unwrap(),
            0xC8F0
        );
        assert_eq!(
            call(polars__sum_fletcher16, json!({"value": "abcdef"}))["checksum"]
                .as_u64()
                .unwrap(),
            0x2057
        );
        // Empty input is 0.
        assert_eq!(
            call(polars__sum_fletcher16, json!({"value": ""}))["checksum"]
                .as_u64()
                .unwrap(),
            0
        );
    }

    #[test]
    fn sum_fletcher32_known_vectors() {
        // Fletcher-32 Wikipedia test vectors (little-endian 16-bit words).
        let f32 = |s: &str| {
            call(polars__sum_fletcher32, json!({ "value": s }))["checksum"]
                .as_u64()
                .unwrap()
        };
        assert_eq!(f32("abcde"), 0xF04F_C729);
        assert_eq!(f32("abcdef"), 0x5650_2D2A);
        assert_eq!(f32("abcdefgh"), 0xEBE1_9591);
        assert_eq!(f32(""), 0);
    }

    #[test]
    fn parse_dtype_canonicalizes_and_categorizes() {
        let i = call(polars__parse_dtype, json!({"dtype": "int"}));
        assert_eq!(i["canonical"], "Int64");
        assert_eq!(i["numeric"], true);
        assert_eq!(i["integer"], true);
        assert_eq!(i["float"], false);
        assert_eq!(i["signed"], true);

        let u = call(polars__parse_dtype, json!({"dtype": "u32"}));
        assert_eq!(u["canonical"], "UInt32");
        assert_eq!(u["integer"], true);
        assert_eq!(u["signed"], false, "unsigned");

        let f = call(polars__parse_dtype, json!({"dtype": "Float64"}));
        assert_eq!(f["canonical"], "Float64");
        assert_eq!(f["float"], true);
        assert_eq!(f["numeric"], true);

        let s = call(polars__parse_dtype, json!({"dtype": "utf8"}));
        assert_eq!(s["canonical"], "String", "utf8 alias");
        assert_eq!(s["numeric"], false);

        let dt = call(polars__parse_dtype, json!({"dtype": "datetime"}));
        assert_eq!(dt["canonical"], "Datetime");
        assert_eq!(dt["temporal"], true);
        assert_eq!(dt["numeric"], false);

        let bad = call(polars__parse_dtype, json!({"dtype": "complex128"}));
        assert!(bad["error"].as_str().unwrap().contains("unknown dtype"));
    }
}
