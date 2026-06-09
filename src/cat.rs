//! src/cat.rs — pandas Categorical surface (`polars__cat_*`).
//!
//! Wire format: `{categorical: {codes: [i64], categories: [str], ordered?: bool}}`.
//! Pure-Rust impl; no polars Categorical dependency (which has complex builder
//! state). Codes are 0-based indices into `categories`; -1 means missing.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::ffi_call;

fn parse_cat(v: &Value) -> Result<(Vec<i64>, Vec<String>, bool)> {
    let codes = v
        .get("codes")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("categorical `codes` missing"))?;
    let cats = v
        .get("categories")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("categorical `categories` missing"))?;
    let ordered = v.get("ordered").and_then(|b| b.as_bool()).unwrap_or(false);
    let c: Vec<i64> = codes.iter().map(|x| x.as_i64().unwrap_or(-1)).collect();
    let s: Vec<String> = cats
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
    Ok((c, s, ordered))
}

fn get_cat(args: &Value) -> Result<(Vec<i64>, Vec<String>, bool)> {
    let v = args
        .get("categorical")
        .ok_or_else(|| anyhow!("missing argument `categorical`"))?;
    parse_cat(v)
}

fn return_cat(codes: Vec<i64>, cats: Vec<String>, ordered: bool) -> Result<Value> {
    Ok(json!({"categorical": {
        "codes": codes,
        "categories": cats,
        "ordered": ordered,
    }}))
}

// ── construction ────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__cat_new(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let values = args
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `values`"))?;
        let cats_explicit = args.get("categories").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .map(|x| x.as_str().unwrap_or("").to_string())
                .collect::<Vec<_>>()
        });
        let ordered = args
            .get("ordered")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let mut cats: Vec<String> = cats_explicit.unwrap_or_default();
        let mut cat_index: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();
        for (i, c) in cats.iter().enumerate() {
            cat_index.insert(c.clone(), i as i64);
        }
        let mut codes: Vec<i64> = Vec::with_capacity(values.len());
        for v in values {
            if v.is_null() {
                codes.push(-1);
                continue;
            }
            let s = v
                .as_str()
                .map(String::from)
                .unwrap_or_else(|| v.to_string());
            if let Some(c) = cat_index.get(&s) {
                codes.push(*c);
            } else {
                let c = cats.len() as i64;
                cat_index.insert(s.clone(), c);
                cats.push(s);
                codes.push(c);
            }
        }
        return_cat(codes, cats, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_from_codes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let codes = args
            .get("codes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `codes`"))?;
        let cats = args
            .get("categories")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `categories`"))?;
        let ordered = args
            .get("ordered")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let c: Vec<i64> = codes.iter().map(|x| x.as_i64().unwrap_or(-1)).collect();
        let s: Vec<String> = cats
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();
        return_cat(c, s, ordered)
    })
}

// ── accessors ───────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__cat_codes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        Ok(json!({"codes": codes}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_categories(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, cats, _) = get_cat(&args)?;
        Ok(json!({"categories": cats}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_ordered(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, _, o) = get_cat(&args)?;
        Ok(json!({"ordered": o}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_len(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        Ok(json!({"len": codes.len()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_nunique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        let set: std::collections::HashSet<i64> = codes.into_iter().filter(|c| *c >= 0).collect();
        Ok(json!({"nunique": set.len()}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_to_strings(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let out: Vec<Value> = codes
            .iter()
            .map(|c| {
                if *c < 0 || (*c as usize) >= cats.len() {
                    Value::Null
                } else {
                    Value::String(cats[*c as usize].clone())
                }
            })
            .collect();
        Ok(json!({"strings": out}))
    })
}

// ── modify categories ───────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__cat_add_categories(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, mut cats, ordered) = get_cat(&args)?;
        let news = args
            .get("new_categories")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `new_categories`"))?;
        for n in news {
            if let Some(s) = n.as_str() {
                if !cats.contains(&s.to_string()) {
                    cats.push(s.to_string());
                }
            }
        }
        return_cat(codes, cats, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_remove_categories(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let rem = args
            .get("remove")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `remove`"))?;
        let remset: std::collections::HashSet<String> = rem
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        // Build new categories preserving order, drop matching ones.
        let mut new_cats = vec![];
        let mut remap = vec![-1i64; cats.len()];
        for (i, c) in cats.iter().enumerate() {
            if !remset.contains(c) {
                remap[i] = new_cats.len() as i64;
                new_cats.push(c.clone());
            }
        }
        let new_codes: Vec<i64> = codes
            .iter()
            .map(|c| {
                if *c < 0 || (*c as usize) >= remap.len() {
                    -1
                } else {
                    remap[*c as usize]
                }
            })
            .collect();
        return_cat(new_codes, new_cats, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_rename_categories(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let new = args
            .get("new_categories")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `new_categories`"))?;
        if new.len() != cats.len() {
            bail!("new_categories length must match");
        }
        let n: Vec<String> = new
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();
        return_cat(codes, n, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_reorder_categories(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let new = args
            .get("new_categories")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `new_categories`"))?;
        let n: Vec<String> = new
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();
        let mut remap = vec![-1i64; cats.len()];
        for (i, c) in cats.iter().enumerate() {
            if let Some(j) = n.iter().position(|x| x == c) {
                remap[i] = j as i64;
            }
        }
        let new_codes: Vec<i64> = codes
            .iter()
            .map(|c| {
                if *c < 0 || (*c as usize) >= remap.len() {
                    -1
                } else {
                    remap[*c as usize]
                }
            })
            .collect();
        return_cat(new_codes, n, true)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_set_categories(args: *const c_char) -> *mut c_char {
    ffi_call(args, reorder_categories_impl)
}

fn reorder_categories_impl(args: Value) -> Result<Value> {
    let (codes, cats, _) = parse_cat(
        args.get("categorical")
            .ok_or_else(|| anyhow!("missing `categorical`"))?,
    )?;
    let new = args
        .get("new_categories")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `new_categories`"))?;
    let n: Vec<String> = new
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
    let mut remap = vec![-1i64; cats.len()];
    for (i, c) in cats.iter().enumerate() {
        if let Some(j) = n.iter().position(|x| x == c) {
            remap[i] = j as i64;
        }
    }
    let new_codes: Vec<i64> = codes
        .iter()
        .map(|c| {
            if *c < 0 || (*c as usize) >= remap.len() {
                -1
            } else {
                remap[*c as usize]
            }
        })
        .collect();
    return_cat(new_codes, n, true)
}

#[no_mangle]
pub extern "C" fn polars__cat_as_ordered(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        return_cat(codes, cats, true)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_as_unordered(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        return_cat(codes, cats, false)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_remove_unused(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let used: std::collections::HashSet<i64> = codes.iter().copied().collect();
        let mut new_cats = vec![];
        let mut remap = vec![-1i64; cats.len()];
        for (i, c) in cats.iter().enumerate() {
            if used.contains(&(i as i64)) {
                remap[i] = new_cats.len() as i64;
                new_cats.push(c.clone());
            }
        }
        let new_codes: Vec<i64> = codes
            .iter()
            .map(|c| {
                if *c < 0 || (*c as usize) >= remap.len() {
                    -1
                } else {
                    remap[*c as usize]
                }
            })
            .collect();
        return_cat(new_codes, new_cats, ordered)
    })
}

// ── value_counts / mode / first / last ─────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__cat_value_counts(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let mut counts = vec![0u64; cats.len()];
        for c in codes {
            if c >= 0 && (c as usize) < counts.len() {
                counts[c as usize] += 1;
            }
        }
        let arr: Vec<Value> = cats
            .iter()
            .zip(counts.iter())
            .map(|(c, n)| json!({"category": c, "count": n}))
            .collect();
        Ok(json!({"value_counts": arr}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_mode(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let mut counts = vec![0u64; cats.len()];
        for c in codes {
            if c >= 0 && (c as usize) < counts.len() {
                counts[c as usize] += 1;
            }
        }
        let r = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, n)| *n)
            .map(|(i, _)| cats[i].clone());
        Ok(json!({"mode": r.map(Value::String).unwrap_or(Value::Null)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_first(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let v = codes.first().and_then(|c| {
            if *c < 0 {
                None
            } else {
                cats.get(*c as usize).cloned()
            }
        });
        Ok(json!({"first": v.map(Value::String).unwrap_or(Value::Null)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_last(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let v = codes.last().and_then(|c| {
            if *c < 0 {
                None
            } else {
                cats.get(*c as usize).cloned()
            }
        });
        Ok(json!({"last": v.map(Value::String).unwrap_or(Value::Null)}))
    })
}

// ── filter / mask / drop_null ──────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__cat_drop_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let new: Vec<i64> = codes.into_iter().filter(|c| *c >= 0).collect();
        return_cat(new, cats, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_is_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        let out: Vec<bool> = codes.iter().map(|c| *c < 0).collect();
        Ok(json!({"is_null": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_isin(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let vals = args
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `values`"))?;
        let valset: std::collections::HashSet<String> = vals
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        let out: Vec<bool> = codes
            .iter()
            .map(|c| {
                if *c < 0 || (*c as usize) >= cats.len() {
                    false
                } else {
                    valset.contains(&cats[*c as usize])
                }
            })
            .collect();
        Ok(json!({"isin": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_sort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let desc = args.get("desc").and_then(|v| v.as_bool()).unwrap_or(false);
        let mut c = codes;
        c.sort_by(|a, b| if desc { b.cmp(a) } else { a.cmp(b) });
        return_cat(c, cats, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_argsort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        let mut pairs: Vec<(usize, i64)> = codes.into_iter().enumerate().collect();
        pairs.sort_by_key(|p| p.1);
        let out: Vec<i64> = pairs.into_iter().map(|(i, _)| i as i64).collect();
        Ok(json!({"argsort": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let set: std::collections::HashSet<i64> = codes.into_iter().filter(|c| *c >= 0).collect();
        let mut sorted: Vec<i64> = set.into_iter().collect();
        sorted.sort();
        let out: Vec<String> = sorted
            .into_iter()
            .filter_map(|c| cats.get(c as usize).cloned())
            .collect();
        Ok(json!({"unique": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_concat(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes_a, cats_a, _) = parse_cat(args.get("a").ok_or_else(|| anyhow!("missing `a`"))?)?;
        let (codes_b, cats_b, _) = parse_cat(args.get("b").ok_or_else(|| anyhow!("missing `b`"))?)?;
        let mut new_cats = cats_a.clone();
        let mut cat_idx: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        for (i, c) in new_cats.iter().enumerate() {
            cat_idx.insert(c.clone(), i as i64);
        }
        let mut remap_b = vec![];
        for c in &cats_b {
            if let Some(i) = cat_idx.get(c) {
                remap_b.push(*i);
            } else {
                let i = new_cats.len() as i64;
                cat_idx.insert(c.clone(), i);
                new_cats.push(c.clone());
                remap_b.push(i);
            }
        }
        let mut out = codes_a;
        for c in codes_b {
            if c < 0 {
                out.push(-1);
            } else {
                out.push(remap_b.get(c as usize).copied().unwrap_or(-1));
            }
        }
        return_cat(out, new_cats, false)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_head(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        return_cat(codes.into_iter().take(n).collect(), cats, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_tail(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let start = codes.len().saturating_sub(n);
        return_cat(codes.into_iter().skip(start).collect(), cats, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (mut codes, cats, ordered) = get_cat(&args)?;
        codes.reverse();
        return_cat(codes, cats, ordered)
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let m = codes.iter().filter(|c| **c >= 0).min();
        let r = m.and_then(|c| cats.get(*c as usize).cloned());
        Ok(json!({"min": r.map(Value::String).unwrap_or(Value::Null)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let m = codes.iter().filter(|c| **c >= 0).max();
        let r = m.and_then(|c| cats.get(*c as usize).cloned());
        Ok(json!({"max": r.map(Value::String).unwrap_or(Value::Null)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_describe(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let n = codes.len();
        let n_null = codes.iter().filter(|c| **c < 0).count();
        let mut counts = vec![0u64; cats.len()];
        for c in &codes {
            if *c >= 0 && (*c as usize) < counts.len() {
                counts[*c as usize] += 1;
            }
        }
        let top = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, n)| *n)
            .map(|(i, n)| (cats[i].clone(), *n));
        Ok(json!({
            "describe": {
                "count": n - n_null,
                "unique": cats.len(),
                "top": top.as_ref().map(|(t, _)| t.clone()),
                "freq": top.map(|(_, n)| n),
            }
        }))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_codes_to_strings(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let out: Vec<Value> = codes
            .iter()
            .map(|c| {
                if *c < 0 || (*c as usize) >= cats.len() {
                    Value::Null
                } else {
                    Value::String(cats[*c as usize].clone())
                }
            })
            .collect();
        Ok(json!({"strings": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__cat_map(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let mapping = args
            .get("mapping")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("missing `mapping`"))?;
        let new_cats: Vec<String> = cats
            .iter()
            .map(|c| {
                mapping
                    .get(c)
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| c.clone())
            })
            .collect();
        return_cat(codes, new_cats, ordered)
    })
}
