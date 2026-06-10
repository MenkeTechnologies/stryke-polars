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

/// Categorical new.
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

/// Categorical from codes.
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

/// Categorical codes.
#[no_mangle]
pub extern "C" fn polars__cat_codes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        Ok(json!({"codes": codes}))
    })
}

/// Categorical categories.
#[no_mangle]
pub extern "C" fn polars__cat_categories(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, cats, _) = get_cat(&args)?;
        Ok(json!({"categories": cats}))
    })
}

/// Categorical ordered.
#[no_mangle]
pub extern "C" fn polars__cat_ordered(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (_, _, o) = get_cat(&args)?;
        Ok(json!({"ordered": o}))
    })
}

/// Categorical len.
#[no_mangle]
pub extern "C" fn polars__cat_len(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        Ok(json!({"len": codes.len()}))
    })
}

/// Categorical nunique.
#[no_mangle]
pub extern "C" fn polars__cat_nunique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        let set: std::collections::HashSet<i64> = codes.into_iter().filter(|c| *c >= 0).collect();
        Ok(json!({"nunique": set.len()}))
    })
}

/// Categorical to strings.
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

/// Categorical add categories.
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

/// Categorical remove categories.
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

/// Categorical rename categories.
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

/// Categorical reorder categories.
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

/// Categorical set categories.
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

/// Categorical as ordered.
#[no_mangle]
pub extern "C" fn polars__cat_as_ordered(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        return_cat(codes, cats, true)
    })
}

/// Categorical as unordered.
#[no_mangle]
pub extern "C" fn polars__cat_as_unordered(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        return_cat(codes, cats, false)
    })
}

/// Categorical remove unused.
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

/// Categorical value counts.
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

/// Categorical mode.
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

/// Categorical first.
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

/// Categorical last.
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

/// Categorical drop null.
#[no_mangle]
pub extern "C" fn polars__cat_drop_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let new: Vec<i64> = codes.into_iter().filter(|c| *c >= 0).collect();
        return_cat(new, cats, ordered)
    })
}

/// Categorical is null.
#[no_mangle]
pub extern "C" fn polars__cat_is_null(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, _, _) = get_cat(&args)?;
        let out: Vec<bool> = codes.iter().map(|c| *c < 0).collect();
        Ok(json!({"is_null": out}))
    })
}

/// Categorical isin.
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

/// Categorical sort.
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

/// Categorical argsort.
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

/// Categorical unique.
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

/// Categorical concat.
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

/// Categorical head.
#[no_mangle]
pub extern "C" fn polars__cat_head(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        return_cat(codes.into_iter().take(n).collect(), cats, ordered)
    })
}

/// Categorical tail.
#[no_mangle]
pub extern "C" fn polars__cat_tail(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, ordered) = get_cat(&args)?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let start = codes.len().saturating_sub(n);
        return_cat(codes.into_iter().skip(start).collect(), cats, ordered)
    })
}

/// Categorical reverse.
#[no_mangle]
pub extern "C" fn polars__cat_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (mut codes, cats, ordered) = get_cat(&args)?;
        codes.reverse();
        return_cat(codes, cats, ordered)
    })
}

/// Categorical min.
#[no_mangle]
pub extern "C" fn polars__cat_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let m = codes.iter().filter(|c| **c >= 0).min();
        let r = m.and_then(|c| cats.get(*c as usize).cloned());
        Ok(json!({"min": r.map(Value::String).unwrap_or(Value::Null)}))
    })
}

/// Categorical max.
#[no_mangle]
pub extern "C" fn polars__cat_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (codes, cats, _) = get_cat(&args)?;
        let m = codes.iter().filter(|c| **c >= 0).max();
        let r = m.and_then(|c| cats.get(*c as usize).cloned());
        Ok(json!({"max": r.map(Value::String).unwrap_or(Value::Null)}))
    })
}

/// Categorical describe.
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

/// Categorical codes to strings.
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

/// Categorical map.
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::ffi_test::call;

    /// `cat_concat` must dedupe categories that exist in both `a` and `b`, and
    /// rewrite codes from `b` through the shared category index. A naive
    /// implementation that appends `b`'s categories blindly (or that
    /// concatenates codes without remapping) silently corrupts the join. This
    /// also pins that null code (-1) in `b` survives unchanged.
    #[test]
    fn cat_concat_dedupes_overlapping_categories_and_remaps_codes() {
        // a: codes=[0,1] cats=["x","y"]   → ["x","y"]
        // b: codes=[0,1,-1] cats=["y","z"] → ["y","z", null]
        // expected new_cats = ["x","y","z"] (dedupe y)
        // expected new_codes = [0,1, /*b[0]="y" remap=1*/ 1, /*b[1]="z" remap=2*/ 2, -1]
        let v = call(
            super::polars__cat_concat,
            json!({
                "a": {"codes": [0, 1], "categories": ["x", "y"], "ordered": false},
                "b": {"codes": [0, 1, -1], "categories": ["y", "z"], "ordered": false},
            }),
        );
        let cats: Vec<String> = v["categorical"]["categories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        let codes: Vec<i64> = v["categorical"]["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(cats, vec!["x", "y", "z"], "y must dedupe across a and b");
        assert_eq!(codes, vec![0, 1, 1, 2, -1], "b's y(0) must remap to 1");
    }

    /// `cat_remove_categories` must rewrite any codes pointing to a removed
    /// category to -1 (null), not leave them as dangling indices that would
    /// later out-of-bounds against the shrunken `categories` vec. Concretely:
    /// remove "b" from cats=["a","b","c"] codes=[0,1,2,1] must yield
    /// cats=["a","c"] codes=[0,-1,1,-1]. An off-by-one in the remap building
    /// (or skipping the rewrite of in-use codes) would corrupt downstream
    /// indexing.
    #[test]
    fn cat_remove_categories_rewrites_used_codes_to_minus_one() {
        let v = call(
            super::polars__cat_remove_categories,
            json!({
                "categorical": {
                    "codes": [0, 1, 2, 1],
                    "categories": ["a", "b", "c"],
                    "ordered": false,
                },
                "remove": ["b"],
            }),
        );
        let cats: Vec<String> = v["categorical"]["categories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        let codes: Vec<i64> = v["categorical"]["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(cats, vec!["a", "c"]);
        assert_eq!(
            codes,
            vec![0, -1, 1, -1],
            "removed category's codes must rewrite to -1 and survivors must compact"
        );
    }

    /// `cat_remove_unused` must drop categories with zero usage AND compact
    /// the remaining codes through a remap. Concrete adversarial setup:
    /// cats=["a","b","c","d"] codes=[2,0] — neither b nor d is used, so they
    /// must drop, leaving cats=["a","c"] and remapped codes=[1,0] (c was at
    /// index 2 → now at index 1; a was at 0 → still 0). A bug that forgets to
    /// remap (or off-by-ones the new-index counter) would output
    /// codes=[2,0] against a 2-element `cats` and crash other ops downstream.
    #[test]
    fn cat_remove_unused_drops_unused_and_remaps_survivors() {
        let v = call(
            super::polars__cat_remove_unused,
            json!({
                "categorical": {
                    "codes": [2, 0],
                    "categories": ["a", "b", "c", "d"],
                    "ordered": false,
                },
            }),
        );
        let cats: Vec<String> = v["categorical"]["categories"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        let codes: Vec<i64> = v["categorical"]["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(cats, vec!["a", "c"], "unused b and d must drop");
        assert_eq!(
            codes,
            vec![1, 0],
            "c's code must shift from 2 to 1; a stays at 0"
        );
    }

    /// `cat_argsort` sorts by raw `i64` code value. Because null is encoded
    /// as -1, ascending argsort places nulls FIRST. This pins that behavior
    /// so a future refactor that swaps to an `Ord`-impl that treats -1 as
    /// "missing/last" (matching pandas `na_position='last'`) won't silently
    /// flip the wire contract without a test catching it. Adversarial input:
    /// codes=[1,-1,0] — naive impl returns [1,2,0]; an na-aware impl would
    /// return [2,0,1].
    #[test]
    fn cat_argsort_places_minus_one_nulls_at_the_front() {
        let v = call(
            super::polars__cat_argsort,
            json!({
                "categorical": {
                    "codes": [1, -1, 0],
                    "categories": ["a", "b"],
                    "ordered": false,
                },
            }),
        );
        let out: Vec<i64> = v["argsort"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        // -1 (index 1) sorts first, then 0 (index 2), then 1 (index 0).
        assert_eq!(out, vec![1, 2, 0]);
    }
}
