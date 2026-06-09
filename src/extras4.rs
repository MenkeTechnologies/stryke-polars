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

#[no_mangle]
pub extern "C" fn polars__bit_popcount(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"popcount": x.count_ones()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_count_zeros(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"count_zeros": x.count_zeros()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_leading_zeros(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"leading_zeros": x.leading_zeros()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_trailing_zeros(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"trailing_zeros": x.trailing_zeros()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_leading_ones(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"leading_ones": x.leading_ones()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_trailing_ones(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"trailing_ones": x.trailing_ones()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"reverse": x.reverse_bits()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_swap_bytes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"swap_bytes": x.swap_bytes()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_rotate_left(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        let n = get_u64(&args, "n")? as u32;
        Ok(json!({"rotate_left": x.rotate_left(n)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_rotate_right(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        let n = get_u64(&args, "n")? as u32;
        Ok(json!({"rotate_right": x.rotate_right(n)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_and(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_i64(&args, "a")?;
        let b = get_i64(&args, "b")?;
        Ok(json!({"and": a & b}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_or(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_i64(&args, "a")?;
        let b = get_i64(&args, "b")?;
        Ok(json!({"or": a | b}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_xor(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_i64(&args, "a")?;
        let b = get_i64(&args, "b")?;
        Ok(json!({"xor": a ^ b}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_not(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"not": !x}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_shl(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"shl": x << n}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_shr(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"shr": x >> n}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_test(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"test": (x >> n) & 1 == 1}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_set(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"set": x | (1 << n)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_clear(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"clear": x & !(1 << n)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_toggle(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        let n = get_u64(&args, "n")?;
        Ok(json!({"toggle": x ^ (1 << n)}))
    })
}

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

#[no_mangle]
pub extern "C" fn polars__bit_is_power_of_two(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        Ok(json!({"is_power_of_two": x.is_power_of_two()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_next_power_of_two(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_u64(&args, "x")?;
        Ok(json!({"next_power_of_two": x.checked_next_power_of_two()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_to_bin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"binary": format!("{:b}", x)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_to_hex(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"hex": format!("{:x}", x)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_to_oct(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_i64(&args, "x")?;
        Ok(json!({"octal": format!("{:o}", x)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_from_bin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let x = i64::from_str_radix(&s, 2).context("from_bin")?;
        Ok(json!({"value": x}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_from_hex(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let x = i64::from_str_radix(&s, 16).context("from_hex")?;
        Ok(json!({"value": x}))
    })
}

#[no_mangle]
pub extern "C" fn polars__bit_from_oct(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let x = i64::from_str_radix(&s, 8).context("from_oct")?;
        Ok(json!({"value": x}))
    })
}

// ── JSON helpers (json_*) ─────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__json_parse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let v: Value = serde_json::from_str(&s).context("parse json")?;
        Ok(json!({"value": v}))
    })
}

#[no_mangle]
pub extern "C" fn polars__json_stringify(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        Ok(json!({"text": serde_json::to_string(v)?}))
    })
}

#[no_mangle]
pub extern "C" fn polars__json_pretty(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        Ok(json!({"text": serde_json::to_string_pretty(v)?}))
    })
}

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

#[no_mangle]
pub extern "C" fn polars__json_is_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("value")
            .ok_or_else(|| anyhow!("missing `value`"))?;
        Ok(json!({"is_null": v.is_null()}))
    })
}

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

#[no_mangle]
pub extern "C" fn polars__fmt_roman(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))?;
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
        Ok(json!({"roman": out}))
    })
}

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

#[no_mangle]
pub extern "C" fn polars__arr_atleast_1d_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        Ok(json!({"array": array_to_value(&a)}))
    })
}

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

#[no_mangle]
pub extern "C" fn polars__sum_simple(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let total: u64 = s.bytes().map(|b| b as u64).sum();
        Ok(json!({"checksum": total}))
    })
}

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

#[no_mangle]
pub extern "C" fn polars__misc_is_palindrome(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let r: String = s.chars().rev().collect();
        Ok(json!({"is_palindrome": s == r}))
    })
}

#[no_mangle]
pub extern "C" fn polars__misc_unique_chars(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let set: std::collections::HashSet<char> = s.chars().collect();
        Ok(json!({"count": set.len()}))
    })
}

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
