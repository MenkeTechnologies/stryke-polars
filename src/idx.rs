//! src/idx.rs — pandas Index / MultiIndex / DatetimeIndex / RangeIndex
//! surface (`polars__idx_*`).
//!
//! Wire format: `{index: {data: [...], name?: "..."}}`.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::ffi_call;

fn parse_index(v: &Value) -> Result<Vec<Value>> {
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("index `data` missing"))?;
    Ok(data.clone())
}

fn get_index(args: &Value) -> Result<Vec<Value>> {
    let v = args
        .get("index")
        .ok_or_else(|| anyhow!("missing argument `index`"))?;
    parse_index(v)
}

fn return_index(data: Vec<Value>) -> Result<Value> {
    Ok(json!({"index": {"data": data}}))
}

fn as_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn as_i64(v: &Value) -> i64 {
    v.as_i64().unwrap_or(0)
}

// ── construction ────────────────────────────────────────────────────────────

/// Index new.
#[no_mangle]
pub extern "C" fn polars__idx_new(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let n = d.len();
        Ok(json!({"index": {"data": d}, "len": n}))
    })
}

/// Index range.
#[no_mangle]
pub extern "C" fn polars__idx_range(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let start = args.get("start").and_then(|v| v.as_i64()).unwrap_or(0);
        let stop = args
            .get("stop")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `stop`"))?;
        let step = args.get("step").and_then(|v| v.as_i64()).unwrap_or(1);
        if step == 0 {
            bail!("step must be non-zero");
        }
        let mut data = vec![];
        let mut x = start;
        if step > 0 {
            while x < stop {
                data.push(json!(x));
                x += step;
            }
        } else {
            while x > stop {
                data.push(json!(x));
                x += step;
            }
        }
        return_index(data)
    })
}

/// Index from list.
#[no_mangle]
pub extern "C" fn polars__idx_from_list(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        return_index(d)
    })
}

/// Index empty.
#[no_mangle]
pub extern "C" fn polars__idx_empty(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| return_index(vec![]))
}

// ── basic ops ───────────────────────────────────────────────────────────────

/// Index len.
#[no_mangle]
pub extern "C" fn polars__idx_len(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        Ok(json!({"len": d.len()}))
    })
}

/// Index is empty.
#[no_mangle]
pub extern "C" fn polars__idx_is_empty(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        Ok(json!({"is_empty": d.is_empty()}))
    })
}

/// Index size.
#[no_mangle]
pub extern "C" fn polars__idx_size(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        Ok(json!({"size": d.len()}))
    })
}

/// Index head.
#[no_mangle]
pub extern "C" fn polars__idx_head(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let out: Vec<Value> = d.into_iter().take(n).collect();
        return_index(out)
    })
}

/// Index tail.
#[no_mangle]
pub extern "C" fn polars__idx_tail(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let start = d.len().saturating_sub(n);
        let out: Vec<Value> = d.into_iter().skip(start).collect();
        return_index(out)
    })
}

/// Index reverse.
#[no_mangle]
pub extern "C" fn polars__idx_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut d = get_index(&args)?;
        d.reverse();
        return_index(d)
    })
}

/// Index slice.
#[no_mangle]
pub extern "C" fn polars__idx_slice(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let start = args.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let stop = args
            .get("stop")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or(d.len());
        if start >= d.len() {
            return return_index(vec![]);
        }
        let end = stop.min(d.len());
        return_index(d[start..end].to_vec())
    })
}

/// Index get.
#[no_mangle]
pub extern "C" fn polars__idx_get(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let i = args
            .get("i")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("missing `i`"))?;
        let i = if i < 0 {
            (d.len() as i64 + i) as usize
        } else {
            i as usize
        };
        if i >= d.len() {
            return Ok(json!({"value": Value::Null}));
        }
        Ok(json!({"value": d[i].clone()}))
    })
}

// ── unique / nunique ───────────────────────────────────────────────────────

/// Index unique.
#[no_mangle]
pub extern "C" fn polars__idx_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let mut seen = std::collections::HashSet::new();
        let out: Vec<Value> = d
            .into_iter()
            .filter(|v| seen.insert(v.to_string()))
            .collect();
        return_index(out)
    })
}

/// Index nunique.
#[no_mangle]
pub extern "C" fn polars__idx_nunique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let set: std::collections::HashSet<String> = d.iter().map(|v| v.to_string()).collect();
        Ok(json!({"nunique": set.len()}))
    })
}

/// Index is unique.
#[no_mangle]
pub extern "C" fn polars__idx_is_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let set: std::collections::HashSet<String> = d.iter().map(|v| v.to_string()).collect();
        Ok(json!({"is_unique": set.len() == d.len()}))
    })
}

/// Index has duplicates.
#[no_mangle]
pub extern "C" fn polars__idx_has_duplicates(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let set: std::collections::HashSet<String> = d.iter().map(|v| v.to_string()).collect();
        Ok(json!({"has_duplicates": set.len() != d.len()}))
    })
}

/// Index duplicated.
#[no_mangle]
pub extern "C" fn polars__idx_duplicated(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
        for v in &d {
            *counts.entry(v.to_string()).or_insert(0) += 1;
        }
        let out: Vec<bool> = d.iter().map(|v| counts[&v.to_string()] > 1).collect();
        Ok(json!({"duplicated": out}))
    })
}

/// Index drop duplicates.
#[no_mangle]
pub extern "C" fn polars__idx_drop_duplicates(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let mut seen = std::collections::HashSet::new();
        let out: Vec<Value> = d
            .into_iter()
            .filter(|v| seen.insert(v.to_string()))
            .collect();
        return_index(out)
    })
}

/// Index value counts.
#[no_mangle]
pub extern "C" fn polars__idx_value_counts(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
        for v in &d {
            *counts.entry(v.to_string()).or_insert(0) += 1;
        }
        let arr: Vec<Value> = counts
            .into_iter()
            .map(|(k, v)| json!({"value": k, "count": v}))
            .collect();
        Ok(json!({"value_counts": arr}))
    })
}

// ── sort / argsort ─────────────────────────────────────────────────────────

/// Index sort.
#[no_mangle]
pub extern "C" fn polars__idx_sort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut d = get_index(&args)?;
        let desc = args.get("desc").and_then(|v| v.as_bool()).unwrap_or(false);
        d.sort_by(|a, b| {
            let ord = if a.is_number() && b.is_number() {
                as_f64(a)
                    .partial_cmp(&as_f64(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.to_string().cmp(&b.to_string())
            };
            if desc {
                ord.reverse()
            } else {
                ord
            }
        });
        return_index(d)
    })
}

/// Index argsort.
#[no_mangle]
pub extern "C" fn polars__idx_argsort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let mut pairs: Vec<(usize, Value)> = d.into_iter().enumerate().collect();
        let desc = args.get("desc").and_then(|v| v.as_bool()).unwrap_or(false);
        pairs.sort_by(|a, b| {
            let ord = if a.1.is_number() && b.1.is_number() {
                as_f64(&a.1)
                    .partial_cmp(&as_f64(&b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.to_string().cmp(&b.1.to_string())
            };
            if desc {
                ord.reverse()
            } else {
                ord
            }
        });
        let out: Vec<i64> = pairs.into_iter().map(|(i, _)| i as i64).collect();
        Ok(json!({"argsort": out}))
    })
}

/// Index is monotonic increasing.
#[no_mangle]
pub extern "C" fn polars__idx_is_monotonic_increasing(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let inc = d.windows(2).all(|w| {
            let a = as_f64(&w[0]);
            let b = as_f64(&w[1]);
            a <= b
        });
        Ok(json!({"is_monotonic_increasing": inc}))
    })
}

/// Index is monotonic decreasing.
#[no_mangle]
pub extern "C" fn polars__idx_is_monotonic_decreasing(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let dec = d.windows(2).all(|w| {
            let a = as_f64(&w[0]);
            let b = as_f64(&w[1]);
            a >= b
        });
        Ok(json!({"is_monotonic_decreasing": dec}))
    })
}

// ── set ops ────────────────────────────────────────────────────────────────

/// Index union.
#[no_mangle]
pub extern "C" fn polars__idx_union(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_index(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_index(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<Value> = vec![];
        for v in a.into_iter().chain(b) {
            if seen.insert(v.to_string()) {
                out.push(v);
            }
        }
        return_index(out)
    })
}

/// Index intersection.
#[no_mangle]
pub extern "C" fn polars__idx_intersection(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_index(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_index(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        let bset: std::collections::HashSet<String> = b.iter().map(|v| v.to_string()).collect();
        let mut seen = std::collections::HashSet::new();
        let out: Vec<Value> = a
            .into_iter()
            .filter(|v| bset.contains(&v.to_string()) && seen.insert(v.to_string()))
            .collect();
        return_index(out)
    })
}

/// Index difference.
#[no_mangle]
pub extern "C" fn polars__idx_difference(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_index(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_index(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        let bset: std::collections::HashSet<String> = b.iter().map(|v| v.to_string()).collect();
        let mut seen = std::collections::HashSet::new();
        let out: Vec<Value> = a
            .into_iter()
            .filter(|v| !bset.contains(&v.to_string()) && seen.insert(v.to_string()))
            .collect();
        return_index(out)
    })
}

/// Index symmetric difference.
#[no_mangle]
pub extern "C" fn polars__idx_symmetric_difference(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_index(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_index(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        let aset: std::collections::HashSet<String> = a.iter().map(|v| v.to_string()).collect();
        let bset: std::collections::HashSet<String> = b.iter().map(|v| v.to_string()).collect();
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<Value> = vec![];
        for v in a.into_iter() {
            if !bset.contains(&v.to_string()) && seen.insert(v.to_string()) {
                out.push(v);
            }
        }
        for v in b.into_iter() {
            if !aset.contains(&v.to_string()) && seen.insert(v.to_string()) {
                out.push(v);
            }
        }
        return_index(out)
    })
}

/// Index append.
#[no_mangle]
pub extern "C" fn polars__idx_append(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut a = parse_index(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_index(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        a.extend(b);
        return_index(a)
    })
}

/// Index repeat.
#[no_mangle]
pub extern "C" fn polars__idx_repeat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let times = args
            .get("times")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `times`"))? as usize;
        let mut out: Vec<Value> = Vec::with_capacity(d.len() * times);
        for _ in 0..times {
            out.extend(d.iter().cloned());
        }
        return_index(out)
    })
}

/// Index take.
#[no_mangle]
pub extern "C" fn polars__idx_take(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let idx = args
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `indices`"))?;
        let out: Vec<Value> = idx
            .iter()
            .filter_map(|i| i.as_u64())
            .filter_map(|i| d.get(i as usize).cloned())
            .collect();
        return_index(out)
    })
}

/// Index delete.
#[no_mangle]
pub extern "C" fn polars__idx_delete(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let idx = args
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `indices`"))?;
        let drop: std::collections::HashSet<u64> = idx.iter().filter_map(|i| i.as_u64()).collect();
        let out: Vec<Value> = d
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !drop.contains(&(*i as u64)))
            .map(|(_, v)| v)
            .collect();
        return_index(out)
    })
}

/// Index insert.
#[no_mangle]
pub extern "C" fn polars__idx_insert(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut d = get_index(&args)?;
        let pos = args
            .get("position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `position`"))? as usize;
        let val = args.get("value").cloned().unwrap_or(Value::Null);
        if pos > d.len() {
            d.push(val);
        } else {
            d.insert(pos, val);
        }
        return_index(d)
    })
}

/// Index drop.
#[no_mangle]
pub extern "C" fn polars__idx_drop(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let drop = args
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `values`"))?;
        let dropset: std::collections::HashSet<String> =
            drop.iter().map(|v| v.to_string()).collect();
        let out: Vec<Value> = d
            .into_iter()
            .filter(|v| !dropset.contains(&v.to_string()))
            .collect();
        return_index(out)
    })
}

// ── membership ─────────────────────────────────────────────────────────────

/// Index contains.
#[no_mangle]
pub extern "C" fn polars__idx_contains(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let val = args.get("value").cloned().unwrap_or(Value::Null);
        let vs = val.to_string();
        let r = d.iter().any(|v| *v == vs);
        Ok(json!({"contains": r}))
    })
}

/// Index isin.
#[no_mangle]
pub extern "C" fn polars__idx_isin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let vals = args
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `values`"))?;
        let set: std::collections::HashSet<String> = vals.iter().map(|v| v.to_string()).collect();
        let out: Vec<bool> = d.iter().map(|v| set.contains(&v.to_string())).collect();
        Ok(json!({"isin": out}))
    })
}

/// Index get loc.
#[no_mangle]
pub extern "C" fn polars__idx_get_loc(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let val = args.get("value").cloned().unwrap_or(Value::Null);
        // Compare by JSON equality first; fall back to string-form equality
        // so that callers can ask `value: "30"` against numeric `[..., 30, ...]`
        // (and vice versa) without dtype-juggling.
        let vs = val.to_string();
        let r = d.iter().position(|v| *v == val || *v.to_string() == *vs);
        Ok(json!({"loc": r.map(|x| x as i64).unwrap_or(-1)}))
    })
}

/// Index get indexer.
#[no_mangle]
pub extern "C" fn polars__idx_get_indexer(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let vals = args
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `values`"))?;
        let out: Vec<i64> = vals
            .iter()
            .map(|x| {
                let xs = x.to_string();
                d.iter()
                    .position(|v| *v == xs)
                    .map(|i| i as i64)
                    .unwrap_or(-1)
            })
            .collect();
        Ok(json!({"indexer": out}))
    })
}

// ── numeric aggregations ───────────────────────────────────────────────────

/// Index min.
#[no_mangle]
pub extern "C" fn polars__idx_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let r = d
            .iter()
            .filter_map(|v| v.as_f64())
            .fold(f64::INFINITY, f64::min);
        Ok(
            json!({"min": serde_json::Number::from_f64(r).map(Value::Number).unwrap_or(Value::Null)}),
        )
    })
}

/// Index max.
#[no_mangle]
pub extern "C" fn polars__idx_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let r = d
            .iter()
            .filter_map(|v| v.as_f64())
            .fold(f64::NEG_INFINITY, f64::max);
        Ok(
            json!({"max": serde_json::Number::from_f64(r).map(Value::Number).unwrap_or(Value::Null)}),
        )
    })
}

/// Index sum.
#[no_mangle]
pub extern "C" fn polars__idx_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let r: f64 = d.iter().filter_map(|v| v.as_f64()).sum();
        Ok(
            json!({"sum": serde_json::Number::from_f64(r).map(Value::Number).unwrap_or(Value::Null)}),
        )
    })
}

/// Index mean.
#[no_mangle]
pub extern "C" fn polars__idx_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let v: Vec<f64> = d.iter().filter_map(|x| x.as_f64()).collect();
        let r = if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        };
        Ok(
            json!({"mean": serde_json::Number::from_f64(r).map(Value::Number).unwrap_or(Value::Null)}),
        )
    })
}

/// Index argmin.
#[no_mangle]
pub extern "C" fn polars__idx_argmin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let r = d
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_f64().map(|x| (i, x)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64);
        Ok(json!({"argmin": r}))
    })
}

/// Index argmax.
#[no_mangle]
pub extern "C" fn polars__idx_argmax(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let r = d
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_f64().map(|x| (i, x)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64);
        Ok(json!({"argmax": r}))
    })
}

// ── shift / rename / map ───────────────────────────────────────────────────

/// Index shift.
#[no_mangle]
pub extern "C" fn polars__idx_shift(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        let n_abs = n.unsigned_abs() as usize;
        let len = d.len();
        let mut out: Vec<Value> = vec![Value::Null; len];
        if n_abs >= len {
            return return_index(out);
        }
        if n >= 0 {
            out[n_abs..len].clone_from_slice(&d[..len - n_abs]);
        } else {
            out[..len - n_abs].clone_from_slice(&d[n_abs..len]);
        }
        return_index(out)
    })
}

/// Index to list.
#[no_mangle]
pub extern "C" fn polars__idx_to_list(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        Ok(json!({"list": d}))
    })
}

/// Index to array.
#[no_mangle]
pub extern "C" fn polars__idx_to_array(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let data: Vec<Value> = d
            .iter()
            .map(|v| {
                v.as_f64()
                    .and_then(|x| serde_json::Number::from_f64(x).map(Value::Number))
                    .unwrap_or(v.clone())
            })
            .collect();
        let n = data.len();
        Ok(json!({"array": {"data": data, "shape": [n]}}))
    })
}

/// Index equals.
#[no_mangle]
pub extern "C" fn polars__idx_equals(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_index(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_index(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        if a.len() != b.len() {
            return Ok(json!({"equals": false}));
        }
        let eq = a.iter().zip(b.iter()).all(|(x, y)| x == y);
        Ok(json!({"equals": eq}))
    })
}

/// Index identical.
#[no_mangle]
pub extern "C" fn polars__idx_identical(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = parse_index(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let b = parse_index(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        Ok(json!({"identical": a == b}))
    })
}

/// Index factorize.
#[no_mangle]
pub extern "C" fn polars__idx_factorize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let mut map: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut uniques: Vec<Value> = vec![];
        let mut codes: Vec<i64> = vec![];
        for v in &d {
            let k = v.to_string();
            if let Some(c) = map.get(&k) {
                codes.push(*c);
            } else {
                let c = uniques.len() as i64;
                map.insert(k, c);
                uniques.push(v.clone());
                codes.push(c);
            }
        }
        Ok(json!({"codes": codes, "uniques": uniques}))
    })
}

/// Index searchsorted.
#[no_mangle]
pub extern "C" fn polars__idx_searchsorted(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let val = args
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let n: Vec<f64> = d.iter().filter_map(|v| v.as_f64()).collect();
        let pos = n.iter().position(|x| *x >= val).unwrap_or(n.len());
        Ok(json!({"position": pos}))
    })
}

/// Index first.
#[no_mangle]
pub extern "C" fn polars__idx_first(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        Ok(json!({"first": d.first().cloned().unwrap_or(Value::Null)}))
    })
}

/// Index last.
#[no_mangle]
pub extern "C" fn polars__idx_last(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        Ok(json!({"last": d.last().cloned().unwrap_or(Value::Null)}))
    })
}

/// Index isnull.
#[no_mangle]
pub extern "C" fn polars__idx_isnull(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<bool> = d.iter().map(|v| v.is_null()).collect();
        Ok(json!({"isnull": out}))
    })
}

/// Index notnull.
#[no_mangle]
pub extern "C" fn polars__idx_notnull(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<bool> = d.iter().map(|v| !v.is_null()).collect();
        Ok(json!({"notnull": out}))
    })
}

/// Index dropna.
#[no_mangle]
pub extern "C" fn polars__idx_dropna(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<Value> = d.into_iter().filter(|v| !v.is_null()).collect();
        return_index(out)
    })
}

/// Index fillna.
#[no_mangle]
pub extern "C" fn polars__idx_fillna(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let val = args.get("value").cloned().unwrap_or(Value::Null);
        let out: Vec<Value> = d
            .into_iter()
            .map(|v| if v.is_null() { val.clone() } else { v })
            .collect();
        return_index(out)
    })
}

/// Index astype int.
#[no_mangle]
pub extern "C" fn polars__idx_astype_int(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<Value> = d.iter().map(|v| json!(as_i64(v))).collect();
        return_index(out)
    })
}

/// Index astype float.
#[no_mangle]
pub extern "C" fn polars__idx_astype_float(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<Value> = d
            .iter()
            .map(|v| {
                let f = as_f64(v);
                serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            })
            .collect();
        return_index(out)
    })
}

/// Index astype str.
#[no_mangle]
pub extern "C" fn polars__idx_astype_str(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<Value> = d
            .iter()
            .map(|v| {
                if let Some(s) = v.as_str() {
                    Value::String(s.to_string())
                } else {
                    Value::String(v.to_string())
                }
            })
            .collect();
        return_index(out)
    })
}

/// Index str upper.
#[no_mangle]
pub extern "C" fn polars__idx_str_upper(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<Value> = d
            .iter()
            .map(|v| Value::String(v.as_str().unwrap_or("").to_uppercase()))
            .collect();
        return_index(out)
    })
}

/// Index str lower.
#[no_mangle]
pub extern "C" fn polars__idx_str_lower(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<Value> = d
            .iter()
            .map(|v| Value::String(v.as_str().unwrap_or("").to_lowercase()))
            .collect();
        return_index(out)
    })
}

/// Index str len.
#[no_mangle]
pub extern "C" fn polars__idx_str_len(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<i64> = d
            .iter()
            .map(|v| v.as_str().map(|s| s.chars().count() as i64).unwrap_or(0))
            .collect();
        Ok(json!({"len": out}))
    })
}

/// Index str strip.
#[no_mangle]
pub extern "C" fn polars__idx_str_strip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let out: Vec<Value> = d
            .iter()
            .map(|v| Value::String(v.as_str().unwrap_or("").trim().to_string()))
            .collect();
        return_index(out)
    })
}

// ── MultiIndex (lightweight: tuples as nested arrays) ──────────────────────

/// Index from tuples.
#[no_mangle]
pub extern "C" fn polars__idx_from_tuples(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("tuples")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `tuples`"))?;
        let data: Vec<Value> = arr.to_vec();
        return_index(data)
    })
}

/// Index from product.
#[no_mangle]
pub extern "C" fn polars__idx_from_product(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let levels = args
            .get("levels")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `levels`"))?;
        let lvls: Vec<Vec<Value>> = levels
            .iter()
            .filter_map(|l| l.as_array().map(|x| x.to_vec()))
            .collect();
        let mut acc: Vec<Vec<Value>> = vec![vec![]];
        for lvl in &lvls {
            let mut next = vec![];
            for prefix in &acc {
                for v in lvl {
                    let mut p = prefix.clone();
                    p.push(v.clone());
                    next.push(p);
                }
            }
            acc = next;
        }
        let data: Vec<Value> = acc.into_iter().map(Value::Array).collect();
        return_index(data)
    })
}

/// Index get level values.
#[no_mangle]
pub extern "C" fn polars__idx_get_level_values(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let level = args.get("level").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let out: Vec<Value> = d
            .iter()
            .filter_map(|v| v.as_array())
            .filter_map(|a| a.get(level).cloned())
            .collect();
        return_index(out)
    })
}

/// Index swap levels.
#[no_mangle]
pub extern "C" fn polars__idx_swap_levels(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let i = args.get("i").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let j = args.get("j").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let out: Vec<Value> = d
            .into_iter()
            .map(|v| match v {
                Value::Array(mut a) => {
                    if i < a.len() && j < a.len() {
                        a.swap(i, j);
                    }
                    Value::Array(a)
                }
                other => other,
            })
            .collect();
        return_index(out)
    })
}

/// Index nlevels.
#[no_mangle]
pub extern "C" fn polars__idx_nlevels(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let n = d
            .first()
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(1);
        Ok(json!({"nlevels": n}))
    })
}

/// Index to frame.
#[no_mangle]
pub extern "C" fn polars__idx_to_frame(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let d = get_index(&args)?;
        let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("index");
        let mut m = serde_json::Map::new();
        m.insert(name.to_string(), Value::Array(d));
        Ok(json!({"frame": Value::Object(m)}))
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::ffi_test::call;

    use super::*;

    #[test]
    fn idx_union_preserves_a_then_appends_b_unique() {
        // a=[1,2,3], b=[3,4,5] → union=[1,2,3,4,5].
        let v = call(
            polars__idx_union,
            json!({"a": {"data": [1, 2, 3]}, "b": {"data": [3, 4, 5]}}),
        );
        let d: Vec<i64> = v["index"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(d, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn idx_intersection_keeps_overlap_only() {
        // [1,2,3,4] ∩ [3,4,5,6] = [3,4].
        let v = call(
            polars__idx_intersection,
            json!({"a": {"data": [1, 2, 3, 4]}, "b": {"data": [3, 4, 5, 6]}}),
        );
        let d: Vec<i64> = v["index"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(d, vec![3, 4]);
    }

    #[test]
    fn idx_difference_drops_b_from_a() {
        let v = call(
            polars__idx_difference,
            json!({"a": {"data": [1, 2, 3, 4]}, "b": {"data": [2, 4]}}),
        );
        let d: Vec<i64> = v["index"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(d, vec![1, 3]);
    }

    #[test]
    fn idx_is_monotonic_increasing_detects_strictly_sorted() {
        let v = call(
            polars__idx_is_monotonic_increasing,
            json!({"index": {"data": [1, 2, 3, 4]}}),
        );
        assert_eq!(v["is_monotonic_increasing"], true);
        let v = call(
            polars__idx_is_monotonic_increasing,
            json!({"index": {"data": [1, 3, 2]}}),
        );
        assert_eq!(v["is_monotonic_increasing"], false);
    }

    #[test]
    fn idx_get_loc_returns_match_index_or_minus_one() {
        let v = call(
            polars__idx_get_loc,
            json!({"index": {"data": [10, 20, 30, 40]}, "value": 30}),
        );
        assert_eq!(v["loc"].as_i64().unwrap(), 2);
        let v = call(
            polars__idx_get_loc,
            json!({"index": {"data": [10, 20, 30]}, "value": 999}),
        );
        assert_eq!(v["loc"].as_i64().unwrap(), -1);
    }
}
