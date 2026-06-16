//! src/extras2.rs — distance metrics, clustering helpers, encoding, hashing,
//! geometric ops, optimization, time-series helpers, sparse-matrix surface.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Value};

use crate::ffi_call;

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

fn get_vec(args: &Value, key: &str) -> Result<Vec<f64>> {
    let a = args
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `{key}`"))?;
    Ok(a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
}

fn scalar(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

// ── distance metrics (dist_*) ─────────────────────────────────────────────

/// Distance euclidean.
#[no_mangle]
pub extern "C" fn polars__dist_euclidean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let d = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Distance manhattan.
#[no_mangle]
pub extern "C" fn polars__dist_manhattan(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let d = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f64>();
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Distance chebyshev.
#[no_mangle]
pub extern "C" fn polars__dist_chebyshev(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let d = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Distance minkowski.
#[no_mangle]
pub extern "C" fn polars__dist_minkowski(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        let p = args.get("p").and_then(|v| v.as_f64()).unwrap_or(2.0);
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let d = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p);
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Distance cosine.
#[no_mangle]
pub extern "C" fn polars__dist_cosine(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let d = if na * nb == 0.0 {
            f64::NAN
        } else {
            1.0 - dot / (na * nb)
        };
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Distance cosine_similarity.
#[no_mangle]
pub extern "C" fn polars__dist_cosine_similarity(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let s = if na * nb == 0.0 {
            f64::NAN
        } else {
            dot / (na * nb)
        };
        Ok(json!({"similarity": scalar(s)}))
    })
}

/// Distance canberra.
#[no_mangle]
pub extern "C" fn polars__dist_canberra(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let d = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let denom = x.abs() + y.abs();
                if denom == 0.0 {
                    0.0
                } else {
                    (x - y).abs() / denom
                }
            })
            .sum::<f64>();
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Distance braycurtis.
#[no_mangle]
pub extern "C" fn polars__dist_braycurtis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        let num: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        let den: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x + y).abs()).sum();
        let d = if den == 0.0 { f64::NAN } else { num / den };
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Distance jaccard.
#[no_mangle]
pub extern "C" fn polars__dist_jaccard(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as f64;
        let nonzero = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| **x != 0.0 || **y != 0.0)
            .count() as f64;
        let d = if nonzero == 0.0 { 0.0 } else { diff / nonzero };
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Distance hamming.
#[no_mangle]
pub extern "C" fn polars__dist_hamming(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as f64;
        Ok(json!({"distance": scalar(diff / a.len() as f64)}))
    })
}

/// Distance mahalanobis.
#[no_mangle]
pub extern "C" fn polars__dist_mahalanobis(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_vec(&args, "a")?;
        let b = get_vec(&args, "b")?;
        let cov_inv = get_array(&args, "cov_inv")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let n = a.len();
        if cov_inv.shape() != [n, n] {
            bail!("cov_inv must be {n}x{n}");
        }
        let diff: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let mut d = 0.0;
        for i in 0..n {
            for j in 0..n {
                d += diff[i] * cov_inv[[i, j].as_slice()] * diff[j];
            }
        }
        Ok(json!({"distance": scalar(d.sqrt())}))
    })
}

/// Distance pdist.
#[no_mangle]
pub extern "C" fn polars__dist_pdist(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = get_array(&args, "matrix")?;
        if m.shape().len() != 2 {
            bail!("matrix must be 2-D");
        }
        let (rows, cols) = (m.shape()[0], m.shape()[1]);
        let mut out = vec![];
        for i in 0..rows {
            for j in i + 1..rows {
                let mut s = 0.0;
                for k in 0..cols {
                    s += (m[[i, k].as_slice()] - m[[j, k].as_slice()]).powi(2);
                }
                out.push(s.sqrt());
            }
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("pdist")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Distance cdist.
#[no_mangle]
pub extern "C" fn polars__dist_cdist(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        if a.shape().len() != 2 || b.shape().len() != 2 {
            bail!("both must be 2-D");
        }
        let (n1, cols) = (a.shape()[0], a.shape()[1]);
        let n2 = b.shape()[0];
        if b.shape()[1] != cols {
            bail!("cols mismatch");
        }
        let mut out = Vec::with_capacity(n1 * n2);
        for i in 0..n1 {
            for j in 0..n2 {
                let mut s = 0.0;
                for k in 0..cols {
                    s += (a[[i, k].as_slice()] - b[[j, k].as_slice()]).powi(2);
                }
                out.push(s.sqrt());
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n1, n2]), out).context("cdist")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

// ── clustering (cluster_*) ─────────────────────────────────────────────────

/// Clustering kmeans.
#[no_mangle]
pub extern "C" fn polars__cluster_kmeans(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let pts = get_array(&args, "points")?;
        if pts.shape().len() != 2 {
            bail!("points must be 2-D");
        }
        let (n, d) = (pts.shape()[0], pts.shape()[1]);
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `k`"))? as usize;
        let max_iter = args.get("max_iter").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
        // Initialize centroids as first k points.
        let mut centroids: Vec<Vec<f64>> = (0..k)
            .map(|i| (0..d).map(|j| pts[[i % n, j].as_slice()]).collect())
            .collect();
        let mut assignments = vec![0_usize; n];
        for _ in 0..max_iter {
            for i in 0..n {
                let mut best = (0, f64::INFINITY);
                for (ci, c) in centroids.iter().enumerate() {
                    let dist: f64 = (0..d)
                        .map(|j| (pts[[i, j].as_slice()] - c[j]).powi(2))
                        .sum();
                    if dist < best.1 {
                        best = (ci, dist);
                    }
                }
                assignments[i] = best.0;
            }
            let mut new_c = vec![vec![0.0; d]; k];
            let mut counts = vec![0; k];
            for i in 0..n {
                for j in 0..d {
                    new_c[assignments[i]][j] += pts[[i, j].as_slice()];
                }
                counts[assignments[i]] += 1;
            }
            for ci in 0..k {
                if counts[ci] > 0 {
                    let denom = counts[ci] as f64;
                    for val in new_c[ci].iter_mut() {
                        *val /= denom;
                    }
                }
            }
            centroids = new_c;
        }
        let flat: Vec<f64> = centroids.iter().flat_map(|c| c.iter().copied()).collect();
        let c_arr = ArrayD::from_shape_vec(IxDyn(&[k, d]), flat).context("centroids")?;
        let a: Vec<i64> = assignments.iter().map(|x| *x as i64).collect();
        Ok(json!({
            "centroids": array_to_value(&c_arr),
            "assignments": a,
        }))
    })
}

/// Clustering assign.
#[no_mangle]
pub extern "C" fn polars__cluster_assign(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let pts = get_array(&args, "points")?;
        let centroids = get_array(&args, "centroids")?;
        if pts.shape().len() != 2 || centroids.shape().len() != 2 {
            bail!("both must be 2-D");
        }
        let (n, d) = (pts.shape()[0], pts.shape()[1]);
        let k = centroids.shape()[0];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut best = (0, f64::INFINITY);
            for ci in 0..k {
                let dist: f64 = (0..d)
                    .map(|j| (pts[[i, j].as_slice()] - centroids[[ci, j].as_slice()]).powi(2))
                    .sum();
                if dist < best.1 {
                    best = (ci, dist);
                }
            }
            out.push(best.0 as i64);
        }
        Ok(json!({"assignments": out}))
    })
}

/// Clustering inertia.
#[no_mangle]
pub extern "C" fn polars__cluster_inertia(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let pts = get_array(&args, "points")?;
        let centroids = get_array(&args, "centroids")?;
        let assignments: Vec<usize> = args
            .get("assignments")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `assignments`"))?
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect();
        let (n, d) = (pts.shape()[0], pts.shape()[1]);
        let mut s = 0.0;
        for i in 0..n {
            let ci = assignments[i];
            for j in 0..d {
                s += (pts[[i, j].as_slice()] - centroids[[ci, j].as_slice()]).powi(2);
            }
        }
        Ok(json!({"inertia": scalar(s)}))
    })
}

/// Clustering silhouette.
#[no_mangle]
pub extern "C" fn polars__cluster_silhouette(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let pts = get_array(&args, "points")?;
        let labels: Vec<usize> = args
            .get("labels")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `labels`"))?
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect();
        let (n, d) = (pts.shape()[0], pts.shape()[1]);
        if labels.len() != n {
            bail!("labels length mismatch");
        }
        let dist = |i: usize, j: usize| -> f64 {
            (0..d)
                .map(|k| (pts[[i, k].as_slice()] - pts[[j, k].as_slice()]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mut sil = vec![0.0_f64; n];
        for i in 0..n {
            let mut a_sum = 0.0;
            let mut a_count = 0;
            let mut group_dists: std::collections::HashMap<usize, (f64, usize)> =
                std::collections::HashMap::new();
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dij = dist(i, j);
                if labels[j] == labels[i] {
                    a_sum += dij;
                    a_count += 1;
                } else {
                    let e = group_dists.entry(labels[j]).or_insert((0.0, 0));
                    e.0 += dij;
                    e.1 += 1;
                }
            }
            let a = if a_count == 0 {
                0.0
            } else {
                a_sum / a_count as f64
            };
            let b = group_dists
                .values()
                .map(|(s, c)| s / *c as f64)
                .fold(f64::INFINITY, f64::min);
            sil[i] = if a == 0.0 && b == 0.0 {
                0.0
            } else {
                (b - a) / a.max(b)
            };
        }
        let avg = sil.iter().sum::<f64>() / n as f64;
        Ok(json!({"silhouette": scalar(avg), "scores": sil}))
    })
}

/// Clustering dbscan neighbors.
#[no_mangle]
pub extern "C" fn polars__cluster_dbscan_neighbors(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let pts = get_array(&args, "points")?;
        let eps = args
            .get("eps")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `eps`"))?;
        let (n, d) = (pts.shape()[0], pts.shape()[1]);
        let mut neighbors: Vec<Vec<i64>> = vec![vec![]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist: f64 = (0..d)
                    .map(|k| (pts[[i, k].as_slice()] - pts[[j, k].as_slice()]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist <= eps {
                    neighbors[i].push(j as i64);
                }
            }
        }
        let neighbors_json: Vec<Value> = neighbors.into_iter().map(|n| json!(n)).collect();
        Ok(json!({"neighbors": neighbors_json}))
    })
}

// ── encoding / hashing ─────────────────────────────────────────────────────

/// Encoding one hot.
#[no_mangle]
pub extern "C" fn polars__enc_one_hot(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("labels")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `labels`"))?;
        let mut categories: Vec<String> = vec![];
        let mut idx = std::collections::HashMap::new();
        for s in v {
            let s = s.as_str().unwrap_or("").to_string();
            if !idx.contains_key(&s) {
                idx.insert(s.clone(), categories.len());
                categories.push(s);
            }
        }
        let n = v.len();
        let k = categories.len();
        let mut out = vec![0.0; n * k];
        for (i, s) in v.iter().enumerate() {
            let s = s.as_str().unwrap_or("").to_string();
            if let Some(j) = idx.get(&s) {
                out[i * k + j] = 1.0;
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, k]), out).context("one_hot")?;
        Ok(json!({"array": array_to_value(&arr), "categories": categories}))
    })
}

/// Encoding label.
#[no_mangle]
pub extern "C" fn polars__enc_label(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("labels")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `labels`"))?;
        let mut categories: Vec<String> = vec![];
        let mut idx = std::collections::HashMap::new();
        for s in v {
            let s = s.as_str().unwrap_or("").to_string();
            if !idx.contains_key(&s) {
                idx.insert(s.clone(), categories.len());
                categories.push(s);
            }
        }
        let out: Vec<i64> = v
            .iter()
            .map(|s| {
                let key = s.as_str().unwrap_or("").to_string();
                *idx.get(&key).unwrap_or(&0) as i64
            })
            .collect();
        Ok(json!({"labels": out, "categories": categories}))
    })
}

/// Encoding frequency: map each categorical label to the relative frequency of
/// its category (count / total) — the count-based encoder, distinct from `label`
/// (arbitrary integer ids) and `one_hot` (indicator columns). Returns
/// `{frequencies, counts}` where `frequencies` is the per-element proportion (in
/// input order) and `counts` is the per-category occurrence count.
#[no_mangle]
pub extern "C" fn polars__enc_frequency(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = args
            .get("labels")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `labels`"))?;
        let n = v.len();
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for s in v {
            *counts
                .entry(s.as_str().unwrap_or("").to_string())
                .or_insert(0) += 1;
        }
        let out: Vec<f64> = v
            .iter()
            .map(|s| {
                let key = s.as_str().unwrap_or("").to_string();
                if n == 0 {
                    0.0
                } else {
                    *counts.get(&key).unwrap_or(&0) as f64 / n as f64
                }
            })
            .collect();
        let counts_obj: serde_json::Map<String, Value> =
            counts.iter().map(|(k, c)| (k.clone(), json!(c))).collect();
        Ok(json!({"frequencies": out, "counts": Value::Object(counts_obj)}))
    })
}

/// Encoding binary.
#[no_mangle]
pub extern "C" fn polars__enc_binary(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let threshold = args
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let out: Vec<f64> = v
            .iter()
            .map(|x| if *x > threshold { 1.0 } else { 0.0 })
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("binary")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Encoding target.
#[no_mangle]
pub extern "C" fn polars__enc_target(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let cats = args
            .get("categories")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `categories`"))?;
        let target = get_vec(&args, "target")?;
        let mut group_sums: std::collections::HashMap<String, (f64, u64)> =
            std::collections::HashMap::new();
        for (i, c) in cats.iter().enumerate() {
            let key = c.as_str().unwrap_or("").to_string();
            let entry = group_sums.entry(key).or_insert((0.0, 0));
            if i < target.len() {
                entry.0 += target[i];
                entry.1 += 1;
            }
        }
        let out: Vec<f64> = cats
            .iter()
            .map(|c| {
                let key = c.as_str().unwrap_or("").to_string();
                let (s, n) = group_sums.get(&key).copied().unwrap_or((0.0, 1));
                s / n.max(1) as f64
            })
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("target_enc")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Hash djb2.
#[no_mangle]
pub extern "C" fn polars__hash_djb2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut h: u64 = 5381;
        for b in s.bytes() {
            h = h.wrapping_mul(33).wrapping_add(b as u64);
        }
        Ok(json!({"hash": h}))
    })
}

/// Hash sdbm — the classic companion of djb2 (from the canonical public-domain
/// string-hash set). `h = c + (h<<6) + (h<<16) - h`, i.e. `h*65599 + c`, over the
/// bytes with u64 wrapping.
#[no_mangle]
pub extern "C" fn polars__hash_sdbm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut h: u64 = 0;
        for b in s.bytes() {
            h = h.wrapping_mul(65599).wrapping_add(b as u64);
        }
        Ok(json!({"hash": h}))
    })
}

/// Hash fnv1a.
#[no_mangle]
pub extern "C" fn polars__hash_fnv1a(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Ok(json!({"hash": h}))
    })
}

/// Hash jenkins.
#[no_mangle]
pub extern "C" fn polars__hash_jenkins(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut h: u32 = 0;
        for b in s.bytes() {
            h = h.wrapping_add(b as u32);
            h = h.wrapping_add(h << 10);
            h ^= h >> 6;
        }
        h = h.wrapping_add(h << 3);
        h ^= h >> 11;
        h = h.wrapping_add(h << 15);
        Ok(json!({"hash": h as u64}))
    })
}

/// Hash crc32.
#[no_mangle]
pub extern "C" fn polars__hash_crc32(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let mut crc: u32 = 0xffffffff;
        for b in s.bytes() {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xedb88320 & ((crc & 1).wrapping_neg()));
            }
        }
        crc = !crc;
        Ok(json!({"hash": crc as u64}))
    })
}

/// Hash murmur32.
#[no_mangle]
pub extern "C" fn polars__hash_murmur32(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let bytes = s.as_bytes();
        let len = bytes.len();
        let mut h: u32 = args.get("seed").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let nblocks = len / 4;
        let c1: u32 = 0xcc9e2d51;
        let c2: u32 = 0x1b873593;
        for i in 0..nblocks {
            let mut k = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
            k = k.wrapping_mul(c1);
            k = k.rotate_left(15);
            k = k.wrapping_mul(c2);
            h ^= k;
            h = h.rotate_left(13);
            h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
        }
        let mut k1: u32 = 0;
        let tail_start = nblocks * 4;
        let tail = &bytes[tail_start..];
        if tail.len() >= 3 {
            k1 ^= (tail[2] as u32) << 16;
        }
        if tail.len() >= 2 {
            k1 ^= (tail[1] as u32) << 8;
        }
        if !tail.is_empty() {
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(c1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(c2);
            h ^= k1;
        }
        h ^= len as u32;
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        Ok(json!({"hash": h as u64}))
    })
}

// ── encoding: base64 / hex ─────────────────────────────────────────────────

/// Encoding base64.
#[no_mangle]
pub extern "C" fn polars__enc_base64(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let bytes = s.as_bytes();
        let mut out = String::new();
        let mut i = 0;
        while i < bytes.len() {
            let b1 = bytes[i];
            let b2 = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
            let b3 = if i + 2 < bytes.len() { bytes[i + 2] } else { 0 };
            out.push(ALPHA[(b1 >> 2) as usize] as char);
            out.push(ALPHA[(((b1 & 0x03) << 4) | (b2 >> 4)) as usize] as char);
            if i + 1 < bytes.len() {
                out.push(ALPHA[(((b2 & 0x0f) << 2) | (b3 >> 6)) as usize] as char);
            } else {
                out.push('=');
            }
            if i + 2 < bytes.len() {
                out.push(ALPHA[(b3 & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
            i += 3;
        }
        Ok(json!({"encoded": out}))
    })
}

/// Encoding hex.
#[no_mangle]
pub extern "C" fn polars__enc_hex(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let out: String = s.bytes().map(|b| format!("{:02x}", b)).collect();
        Ok(json!({"encoded": out}))
    })
}

/// Encoding url.
#[no_mangle]
pub extern "C" fn polars__enc_url(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing `value`"))?;
        let out: String = s
            .chars()
            .flat_map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '~' {
                    vec![c]
                } else {
                    format!("%{:02X}", c as u32).chars().collect()
                }
            })
            .collect();
        Ok(json!({"encoded": out}))
    })
}

// ── geometric (geo_*) ──────────────────────────────────────────────────────

/// Geometric distance 2d.
#[no_mangle]
pub extern "C" fn polars__geo_distance_2d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x1 = args
            .get("x1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x1`"))?;
        let y1 = args
            .get("y1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `y1`"))?;
        let x2 = args
            .get("x2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x2`"))?;
        let y2 = args
            .get("y2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `y2`"))?;
        let d = ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt();
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Geometric distance 3d.
#[no_mangle]
pub extern "C" fn polars__geo_distance_3d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x1 = args
            .get("x1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x1`"))?;
        let y1 = args
            .get("y1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `y1`"))?;
        let z1 = args
            .get("z1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `z1`"))?;
        let x2 = args
            .get("x2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `x2`"))?;
        let y2 = args
            .get("y2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `y2`"))?;
        let z2 = args
            .get("z2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `z2`"))?;
        let d = ((x1 - x2).powi(2) + (y1 - y2).powi(2) + (z1 - z2).powi(2)).sqrt();
        Ok(json!({"distance": scalar(d)}))
    })
}

/// Geometric haversine.
#[no_mangle]
pub extern "C" fn polars__geo_haversine(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let lat1 = args
            .get("lat1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lat1`"))?;
        let lon1 = args
            .get("lon1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lon1`"))?;
        let lat2 = args
            .get("lat2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lat2`"))?;
        let lon2 = args
            .get("lon2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lon2`"))?;
        let r = args
            .get("radius")
            .and_then(|v| v.as_f64())
            .unwrap_or(6371.0);
        let rl1 = lat1.to_radians();
        let rl2 = lat2.to_radians();
        let dlat = (lat2 - lat1).to_radians();
        let dlon = (lon2 - lon1).to_radians();
        let a = (dlat / 2.0).sin().powi(2) + rl1.cos() * rl2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        Ok(json!({"distance": scalar(r * c)}))
    })
}

/// Geometric bearing.
#[no_mangle]
pub extern "C" fn polars__geo_bearing(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let lat1 = args
            .get("lat1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lat1`"))?
            .to_radians();
        let lon1 = args
            .get("lon1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lon1`"))?
            .to_radians();
        let lat2 = args
            .get("lat2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lat2`"))?
            .to_radians();
        let lon2 = args
            .get("lon2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lon2`"))?
            .to_radians();
        let dlon = lon2 - lon1;
        let y = dlon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
        let bearing = y.atan2(x).to_degrees();
        Ok(json!({"bearing": scalar((bearing + 360.0) % 360.0)}))
    })
}

/// Geometric polygon area.
#[no_mangle]
pub extern "C" fn polars__geo_polygon_area(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let xs = get_vec(&args, "x")?;
        let ys = get_vec(&args, "y")?;
        if xs.len() != ys.len() || xs.len() < 3 {
            bail!("need at least 3 points");
        }
        let n = xs.len();
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += xs[i] * ys[j];
            area -= xs[j] * ys[i];
        }
        Ok(json!({"area": scalar(area.abs() / 2.0)}))
    })
}

/// Geometric polygon perimeter.
#[no_mangle]
pub extern "C" fn polars__geo_polygon_perimeter(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let xs = get_vec(&args, "x")?;
        let ys = get_vec(&args, "y")?;
        if xs.len() != ys.len() || xs.len() < 2 {
            bail!("need at least 2 points");
        }
        let n = xs.len();
        let mut p = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            p += ((xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2)).sqrt();
        }
        Ok(json!({"perimeter": scalar(p)}))
    })
}

/// Geometric centroid.
#[no_mangle]
pub extern "C" fn polars__geo_centroid(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let xs = get_vec(&args, "x")?;
        let ys = get_vec(&args, "y")?;
        let n = xs.len() as f64;
        let cx = xs.iter().sum::<f64>() / n;
        let cy = ys.iter().sum::<f64>() / n;
        Ok(json!({"x": scalar(cx), "y": scalar(cy)}))
    })
}

/// Geometric bbox.
#[no_mangle]
pub extern "C" fn polars__geo_bbox(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let xs = get_vec(&args, "x")?;
        let ys = get_vec(&args, "y")?;
        let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Ok(json!({
            "x_min": scalar(x_min), "x_max": scalar(x_max),
            "y_min": scalar(y_min), "y_max": scalar(y_max),
        }))
    })
}

/// Geometric point in polygon.
#[no_mangle]
pub extern "C" fn polars__geo_point_in_polygon(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let px = args
            .get("px")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `px`"))?;
        let py = args
            .get("py")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `py`"))?;
        let xs = get_vec(&args, "x")?;
        let ys = get_vec(&args, "y")?;
        let n = xs.len();
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            if ((ys[i] > py) != (ys[j] > py))
                && (px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i]) + xs[i])
            {
                inside = !inside;
            }
            j = i;
        }
        Ok(json!({"inside": inside}))
    })
}

/// Geometric rotate.
#[no_mangle]
pub extern "C" fn polars__geo_rotate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let xs = get_vec(&args, "x")?;
        let ys = get_vec(&args, "y")?;
        let angle = args
            .get("angle")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `angle`"))?;
        let cx = args.get("cx").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let cy = args.get("cy").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let theta = angle.to_radians();
        let cos = theta.cos();
        let sin = theta.sin();
        let xo: Vec<f64> = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| cos * (x - cx) - sin * (y - cy) + cx)
            .collect();
        let yo: Vec<f64> = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| sin * (x - cx) + cos * (y - cy) + cy)
            .collect();
        Ok(json!({"x": xo, "y": yo}))
    })
}

/// Geometric scale.
#[no_mangle]
pub extern "C" fn polars__geo_scale(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let xs = get_vec(&args, "x")?;
        let ys = get_vec(&args, "y")?;
        let sx = args.get("sx").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let sy = args.get("sy").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let xo: Vec<f64> = xs.iter().map(|x| x * sx).collect();
        let yo: Vec<f64> = ys.iter().map(|y| y * sy).collect();
        Ok(json!({"x": xo, "y": yo}))
    })
}

/// Geometric translate.
#[no_mangle]
pub extern "C" fn polars__geo_translate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let xs = get_vec(&args, "x")?;
        let ys = get_vec(&args, "y")?;
        let tx = args.get("tx").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ty = args.get("ty").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let xo: Vec<f64> = xs.iter().map(|x| x + tx).collect();
        let yo: Vec<f64> = ys.iter().map(|y| y + ty).collect();
        Ok(json!({"x": xo, "y": yo}))
    })
}

// ── optimization (opt_*) ───────────────────────────────────────────────────

/// Optimization minimize bisection.
#[no_mangle]
pub extern "C" fn polars__opt_minimize_bisection(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let coeffs = args
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `coefficients`"))?;
        let c: Vec<f64> = coeffs.iter().filter_map(|x| x.as_f64()).collect();
        let lo = args
            .get("lo")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lo`"))?;
        let hi = args
            .get("hi")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `hi`"))?;
        let tol = args.get("tol").and_then(|v| v.as_f64()).unwrap_or(1e-6);
        let max_iter = args.get("max_iter").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
        let f = |x: f64| -> f64 {
            // Polynomial evaluation via Horner.
            let mut acc = 0.0;
            for c in c.iter().rev() {
                acc = acc * x + c;
            }
            acc
        };
        let mut a = lo;
        let mut b = hi;
        if f(a) * f(b) > 0.0 {
            bail!("f(lo) and f(hi) must have opposite signs");
        }
        for _ in 0..max_iter {
            let m = (a + b) / 2.0;
            let fm = f(m);
            if (b - a) / 2.0 < tol {
                return Ok(json!({"x": scalar(m), "f": scalar(fm)}));
            }
            if f(a) * fm < 0.0 {
                b = m;
            } else {
                a = m;
            }
        }
        Ok(json!({"x": scalar((a + b) / 2.0), "f": scalar(f((a + b) / 2.0))}))
    })
}

/// Optimization minimize golden.
#[no_mangle]
pub extern "C" fn polars__opt_minimize_golden(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let coeffs = args
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `coefficients`"))?;
        let c: Vec<f64> = coeffs.iter().filter_map(|x| x.as_f64()).collect();
        let mut a = args
            .get("lo")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lo`"))?;
        let mut b = args
            .get("hi")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `hi`"))?;
        let tol = args.get("tol").and_then(|v| v.as_f64()).unwrap_or(1e-6);
        let f = |x: f64| -> f64 {
            let mut acc = 0.0;
            for c in c.iter().rev() {
                acc = acc * x + c;
            }
            acc
        };
        let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
        let mut x1 = b - gr * (b - a);
        let mut x2 = a + gr * (b - a);
        for _ in 0..200 {
            if (b - a).abs() < tol {
                break;
            }
            if f(x1) < f(x2) {
                b = x2;
                x2 = x1;
                x1 = b - gr * (b - a);
            } else {
                a = x1;
                x1 = x2;
                x2 = a + gr * (b - a);
            }
        }
        let x = (a + b) / 2.0;
        Ok(json!({"x": scalar(x), "f": scalar(f(x))}))
    })
}

/// Optimization newton raphson.
#[no_mangle]
pub extern "C" fn polars__opt_newton_raphson(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let coeffs = args
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `coefficients`"))?;
        let c: Vec<f64> = coeffs.iter().filter_map(|x| x.as_f64()).collect();
        let mut x = args.get("x0").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let tol = args.get("tol").and_then(|v| v.as_f64()).unwrap_or(1e-8);
        let max_iter = args.get("max_iter").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
        let f = |x: f64| -> f64 {
            let mut acc = 0.0;
            for c in c.iter().rev() {
                acc = acc * x + c;
            }
            acc
        };
        let fp = |x: f64| -> f64 {
            let mut acc = 0.0;
            for (i, c) in c.iter().enumerate().rev().skip(1) {
                acc = acc * x + c * (c.signum() * 0.0 + (i as f64) + 1.0);
            }
            // derivative: sum k*c_k*x^(k-1); compute via shifted coeffs.
            let mut d_coeffs = vec![];
            for (i, ci) in c.iter().enumerate().skip(1) {
                d_coeffs.push(ci * i as f64);
            }
            let mut acc = 0.0;
            for c in d_coeffs.iter().rev() {
                acc = acc * x + c;
            }
            acc
        };
        for _ in 0..max_iter {
            let fx = f(x);
            let dx = fp(x);
            if dx.abs() < tol {
                break;
            }
            let new_x = x - fx / dx;
            if (new_x - x).abs() < tol {
                x = new_x;
                break;
            }
            x = new_x;
        }
        Ok(json!({"x": scalar(x), "f": scalar(f(x))}))
    })
}

/// Optimization secant.
#[no_mangle]
pub extern "C" fn polars__opt_secant(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let coeffs = args
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `coefficients`"))?;
        let c: Vec<f64> = coeffs.iter().filter_map(|x| x.as_f64()).collect();
        let mut x0 = args.get("x0").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mut x1 = args.get("x1").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let tol = args.get("tol").and_then(|v| v.as_f64()).unwrap_or(1e-8);
        let f = |x: f64| -> f64 {
            let mut acc = 0.0;
            for c in c.iter().rev() {
                acc = acc * x + c;
            }
            acc
        };
        for _ in 0..100 {
            let f0 = f(x0);
            let f1 = f(x1);
            if (f1 - f0).abs() < tol {
                break;
            }
            let new_x = x1 - f1 * (x1 - x0) / (f1 - f0);
            x0 = x1;
            x1 = new_x;
            if (x1 - x0).abs() < tol {
                break;
            }
        }
        Ok(json!({"x": scalar(x1), "f": scalar(f(x1))}))
    })
}

/// Optimization gradient descent.
#[no_mangle]
pub extern "C" fn polars__opt_gradient_descent(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let coeffs = args
            .get("coefficients")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `coefficients`"))?;
        let c: Vec<f64> = coeffs.iter().filter_map(|x| x.as_f64()).collect();
        let mut x = args.get("x0").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let lr = args.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.01);
        let max_iter = args
            .get("max_iter")
            .and_then(|v| v.as_u64())
            .unwrap_or(1000) as usize;
        let tol = args.get("tol").and_then(|v| v.as_f64()).unwrap_or(1e-8);
        let f = |x: f64| -> f64 {
            let mut acc = 0.0;
            for c in c.iter().rev() {
                acc = acc * x + c;
            }
            acc
        };
        for _ in 0..max_iter {
            let d_coeffs: Vec<f64> = c
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, ci)| ci * i as f64)
                .collect();
            let mut grad = 0.0;
            for c in d_coeffs.iter().rev() {
                grad = grad * x + c;
            }
            let new_x = x - lr * grad;
            if (new_x - x).abs() < tol {
                x = new_x;
                break;
            }
            x = new_x;
        }
        Ok(json!({"x": scalar(x), "f": scalar(f(x))}))
    })
}

// ── time series helpers (ts_*) ─────────────────────────────────────────────

/// Time series lag.
#[no_mangle]
pub extern "C" fn polars__ts_lag(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let lag = args.get("lag").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mut out = vec![f64::NAN; lag.min(v.len())];
        if lag < v.len() {
            out.extend_from_slice(&v[..v.len() - lag]);
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("lag")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Time series lead.
#[no_mangle]
pub extern "C" fn polars__ts_lead(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let lead = args.get("lead").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let n = v.len();
        let mut out: Vec<f64> = if lead < n { v[lead..].to_vec() } else { vec![] };
        while out.len() < n {
            out.push(f64::NAN);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("lead")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Time series diff.
#[no_mangle]
pub extern "C" fn polars__ts_diff(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let mut out = vec![f64::NAN; n.min(v.len())];
        for i in n..v.len() {
            out.push(v[i] - v[i - n]);
        }
        let n_out = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n_out]), out).context("diff")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Time series log returns.
#[no_mangle]
pub extern "C" fn polars__ts_log_returns(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let mut out = vec![f64::NAN; 1.min(v.len())];
        for i in 1..v.len() {
            out.push((v[i] / v[i - 1]).ln());
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("log_returns")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Time series simple returns.
#[no_mangle]
pub extern "C" fn polars__ts_simple_returns(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let mut out = vec![f64::NAN; 1.min(v.len())];
        for i in 1..v.len() {
            out.push(v[i] / v[i - 1] - 1.0);
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("simple_returns")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Time series cumulative returns.
#[no_mangle]
pub extern "C" fn polars__ts_cumulative_returns(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let mut acc = 1.0;
        let out: Vec<f64> = v
            .iter()
            .map(|r| {
                acc *= 1.0 + r;
                acc - 1.0
            })
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("cumulative_returns")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Time series drawdown.
#[no_mangle]
pub extern "C" fn polars__ts_drawdown(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let mut peak = v.first().copied().unwrap_or(0.0);
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                peak = peak.max(*x);
                if peak == 0.0 {
                    0.0
                } else {
                    (x - peak) / peak
                }
            })
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("drawdown")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Time series max drawdown.
#[no_mangle]
pub extern "C" fn polars__ts_max_drawdown(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let mut peak = v.first().copied().unwrap_or(0.0);
        let mut mdd = 0.0_f64;
        for x in &v {
            peak = peak.max(*x);
            if peak != 0.0 {
                mdd = mdd.min((x - peak) / peak);
            }
        }
        Ok(json!({"max_drawdown": scalar(mdd)}))
    })
}

/// Time series volatility.
#[no_mangle]
pub extern "C" fn polars__ts_volatility(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let annualize = args
            .get("annualize")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        Ok(json!({"volatility": scalar(var.sqrt() * annualize.sqrt())}))
    })
}

/// Time series sharpe.
#[no_mangle]
pub extern "C" fn polars__ts_sharpe(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let rf = args
            .get("risk_free")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let sd = var.sqrt();
        let sharpe = if sd == 0.0 { f64::NAN } else { (m - rf) / sd };
        Ok(json!({"sharpe": scalar(sharpe)}))
    })
}

/// Time series sortino.
#[no_mangle]
pub extern "C" fn polars__ts_sortino(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let v = get_vec(&args, "data")?;
        let rf = args
            .get("risk_free")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n.max(1.0);
        let down_var = v
            .iter()
            .filter(|x| **x < rf)
            .map(|x| (x - rf).powi(2))
            .sum::<f64>()
            / n.max(1.0);
        let sd = down_var.sqrt();
        let sortino = if sd == 0.0 { f64::NAN } else { (m - rf) / sd };
        Ok(json!({"sortino": scalar(sortino)}))
    })
}

/// Time series var.
#[no_mangle]
pub extern "C" fn polars__ts_var(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut v = get_vec(&args, "data")?;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.05);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (alpha * v.len() as f64) as usize;
        let var_val = v.get(idx).copied().unwrap_or(0.0);
        Ok(json!({"var": scalar(var_val)}))
    })
}

/// Time series cvar.
#[no_mangle]
pub extern "C" fn polars__ts_cvar(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut v = get_vec(&args, "data")?;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.05);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff = (alpha * v.len() as f64) as usize;
        if cutoff == 0 {
            return Ok(json!({"cvar": Value::Null}));
        }
        let tail: f64 = v[..cutoff].iter().sum::<f64>() / cutoff as f64;
        Ok(json!({"cvar": scalar(tail)}))
    })
}

// ── sparse matrix (sparse_*) ───────────────────────────────────────────────

/// (data, row indices, col indices, n_rows, n_cols).
type SparseCoo = (Vec<f64>, Vec<usize>, Vec<usize>, usize, usize);

fn parse_sparse(v: &Value) -> Result<SparseCoo> {
    let data: Vec<f64> = v
        .get("data")
        .and_then(|d| d.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
        .unwrap_or_default();
    let rows: Vec<usize> = v
        .get("rows")
        .and_then(|d| d.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();
    let cols: Vec<usize> = v
        .get("cols")
        .and_then(|d| d.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();
    let n_rows = v.get("n_rows").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let n_cols = v.get("n_cols").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    Ok((data, rows, cols, n_rows, n_cols))
}

/// Sparse matrix from dense.
#[no_mangle]
pub extern "C" fn polars__sparse_from_dense(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = get_array(&args, "matrix")?;
        if m.shape().len() != 2 {
            bail!("matrix must be 2-D");
        }
        let (rows, cols) = (m.shape()[0], m.shape()[1]);
        let mut data = vec![];
        let mut row_idx = vec![];
        let mut col_idx = vec![];
        for r in 0..rows {
            for c in 0..cols {
                let v = m[[r, c].as_slice()];
                if v != 0.0 {
                    data.push(v);
                    row_idx.push(r);
                    col_idx.push(c);
                }
            }
        }
        Ok(json!({"sparse": {
            "data": data,
            "rows": row_idx,
            "cols": col_idx,
            "n_rows": rows,
            "n_cols": cols,
        }}))
    })
}

/// Sparse matrix to dense.
#[no_mangle]
pub extern "C" fn polars__sparse_to_dense(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, rows, cols, n_rows, n_cols) = parse_sparse(s)?;
        let mut out = vec![0.0; n_rows * n_cols];
        for i in 0..data.len() {
            out[rows[i] * n_cols + cols[i]] = data[i];
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n_rows, n_cols]), out).context("to_dense")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Sparse matrix nnz.
#[no_mangle]
pub extern "C" fn polars__sparse_nnz(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, _, _, _, _) = parse_sparse(s)?;
        Ok(json!({"nnz": data.len()}))
    })
}

/// Sparse matrix density.
#[no_mangle]
pub extern "C" fn polars__sparse_density(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, _, _, n_rows, n_cols) = parse_sparse(s)?;
        let total = n_rows * n_cols;
        let d = if total == 0 {
            0.0
        } else {
            data.len() as f64 / total as f64
        };
        Ok(json!({"density": scalar(d)}))
    })
}

/// Sparse matrix transpose.
#[no_mangle]
pub extern "C" fn polars__sparse_transpose(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, rows, cols, n_rows, n_cols) = parse_sparse(s)?;
        Ok(json!({"sparse": {
            "data": data,
            "rows": cols,
            "cols": rows,
            "n_rows": n_cols,
            "n_cols": n_rows,
        }}))
    })
}

/// Sparse matrix add.
#[no_mangle]
pub extern "C" fn polars__sparse_add(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args.get("a").ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args.get("b").ok_or_else(|| anyhow!("missing `b`"))?;
        let (da, ra, ca, nra, nca) = parse_sparse(a)?;
        let (db, rb, cb, nrb, ncb) = parse_sparse(b)?;
        if (nra, nca) != (nrb, ncb) {
            bail!("shape mismatch");
        }
        let mut map: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        for i in 0..da.len() {
            *map.entry((ra[i], ca[i])).or_insert(0.0) += da[i];
        }
        for i in 0..db.len() {
            *map.entry((rb[i], cb[i])).or_insert(0.0) += db[i];
        }
        let mut data = vec![];
        let mut rows = vec![];
        let mut cols = vec![];
        for ((r, c), v) in map {
            if v != 0.0 {
                data.push(v);
                rows.push(r);
                cols.push(c);
            }
        }
        Ok(json!({"sparse": {
            "data": data, "rows": rows, "cols": cols,
            "n_rows": nra, "n_cols": nca,
        }}))
    })
}

/// Sparse element-wise (Hadamard) product `a ∘ b`. A result entry is non-zero
/// only where BOTH inputs have a stored value at that position (the product is
/// zero everywhere else), so the output is at most as dense as the sparser
/// input — distinct from `mat_mul`, which is the matrix product. Same shape
/// required.
#[no_mangle]
pub extern "C" fn polars__sparse_hadamard(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args.get("a").ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args.get("b").ok_or_else(|| anyhow!("missing `b`"))?;
        let (da, ra, ca, nra, nca) = parse_sparse(a)?;
        let (db, rb, cb, nrb, ncb) = parse_sparse(b)?;
        if (nra, nca) != (nrb, ncb) {
            bail!("shape mismatch");
        }
        let amap: std::collections::HashMap<(usize, usize), f64> =
            (0..da.len()).map(|i| ((ra[i], ca[i]), da[i])).collect();
        let mut data = vec![];
        let mut rows = vec![];
        let mut cols = vec![];
        for i in 0..db.len() {
            if let Some(av) = amap.get(&(rb[i], cb[i])) {
                let v = av * db[i];
                if v != 0.0 {
                    data.push(v);
                    rows.push(rb[i]);
                    cols.push(cb[i]);
                }
            }
        }
        Ok(json!({"sparse": {
            "data": data, "rows": rows, "cols": cols,
            "n_rows": nra, "n_cols": nca,
        }}))
    })
}

/// Sparse matrix subtract (`a - b`). Same-shape required; explicit zeros are
/// dropped from the result, mirroring `sparse_add`.
#[no_mangle]
pub extern "C" fn polars__sparse_sub(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args.get("a").ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args.get("b").ok_or_else(|| anyhow!("missing `b`"))?;
        let (da, ra, ca, nra, nca) = parse_sparse(a)?;
        let (db, rb, cb, nrb, ncb) = parse_sparse(b)?;
        if (nra, nca) != (nrb, ncb) {
            bail!("shape mismatch");
        }
        let mut map: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        for i in 0..da.len() {
            *map.entry((ra[i], ca[i])).or_insert(0.0) += da[i];
        }
        for i in 0..db.len() {
            *map.entry((rb[i], cb[i])).or_insert(0.0) -= db[i];
        }
        let mut data = vec![];
        let mut rows = vec![];
        let mut cols = vec![];
        for ((r, c), v) in map {
            if v != 0.0 {
                data.push(v);
                rows.push(r);
                cols.push(c);
            }
        }
        Ok(json!({"sparse": {
            "data": data, "rows": rows, "cols": cols,
            "n_rows": nra, "n_cols": nca,
        }}))
    })
}

/// Sparse matrix–matrix multiply (`a · b`). Requires `a.n_cols == b.n_rows`;
/// the result is `a.n_rows × b.n_cols`. Computed coordinate-wise (no dense
/// intermediate) and explicit zeros are dropped, mirroring `sparse_add`.
#[no_mangle]
pub extern "C" fn polars__sparse_mat_mul(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = args.get("a").ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args.get("b").ok_or_else(|| anyhow!("missing `b`"))?;
        let (da, ra, ca, nra, nca) = parse_sparse(a)?;
        let (db, rb, cb, nrb, ncb) = parse_sparse(b)?;
        if nca != nrb {
            bail!("inner dimension mismatch: a has {nca} cols, b has {nrb} rows");
        }
        // Index b by its row so each a-entry only visits the matching b-row.
        let mut b_rows: std::collections::HashMap<usize, Vec<(usize, f64)>> =
            std::collections::HashMap::new();
        for i in 0..db.len() {
            b_rows.entry(rb[i]).or_default().push((cb[i], db[i]));
        }
        let mut map: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        for i in 0..da.len() {
            if let Some(row) = b_rows.get(&ca[i]) {
                for &(j, bv) in row {
                    *map.entry((ra[i], j)).or_insert(0.0) += da[i] * bv;
                }
            }
        }
        let mut data = vec![];
        let mut rows = vec![];
        let mut cols = vec![];
        for ((r, c), v) in map {
            if v != 0.0 {
                data.push(v);
                rows.push(r);
                cols.push(c);
            }
        }
        Ok(json!({"sparse": {
            "data": data, "rows": rows, "cols": cols,
            "n_rows": nra, "n_cols": ncb,
        }}))
    })
}

/// Sparse matrix mul vec.
#[no_mangle]
pub extern "C" fn polars__sparse_mul_vec(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, rows, cols, n_rows, _) = parse_sparse(s)?;
        let v = get_vec(&args, "vector")?;
        let mut out = vec![0.0; n_rows];
        for i in 0..data.len() {
            out[rows[i]] += data[i] * v[cols[i]];
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("sparse_mul_vec")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Sparse matrix scale.
#[no_mangle]
pub extern "C" fn polars__sparse_scale(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, rows, cols, n_rows, n_cols) = parse_sparse(s)?;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = data.iter().map(|x| x * alpha).collect();
        Ok(json!({"sparse": {
            "data": out, "rows": rows, "cols": cols,
            "n_rows": n_rows, "n_cols": n_cols,
        }}))
    })
}

/// Sparse matrix diagonal.
#[no_mangle]
pub extern "C" fn polars__sparse_diagonal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, rows, cols, n_rows, n_cols) = parse_sparse(s)?;
        let n = n_rows.min(n_cols);
        let mut diag = vec![0.0; n];
        for i in 0..data.len() {
            if rows[i] == cols[i] && rows[i] < n {
                diag[rows[i]] = data[i];
            }
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), diag).context("diagonal")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Sparse matrix trace.
#[no_mangle]
pub extern "C" fn polars__sparse_trace(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, rows, cols, _, _) = parse_sparse(s)?;
        let mut tr = 0.0;
        for i in 0..data.len() {
            if rows[i] == cols[i] {
                tr += data[i];
            }
        }
        Ok(json!({"trace": scalar(tr)}))
    })
}

/// Sparse matrix Frobenius norm: the square root of the sum of squared entries.
/// Only stored (nonzero) entries contribute, so it is exact for the sparse
/// representation. `eye(n)` has norm `sqrt(n)`.
#[no_mangle]
pub extern "C" fn polars__sparse_frobenius(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = args
            .get("sparse")
            .ok_or_else(|| anyhow!("missing `sparse`"))?;
        let (data, _, _, _, _) = parse_sparse(s)?;
        let sumsq: f64 = data.iter().map(|x| x * x).sum();
        Ok(json!({"norm": scalar(sumsq.sqrt())}))
    })
}

/// Sparse matrix eye.
#[no_mangle]
pub extern "C" fn polars__sparse_eye(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let data = vec![1.0; n];
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        Ok(json!({"sparse": {
            "data": data, "rows": rows, "cols": cols,
            "n_rows": n, "n_cols": n,
        }}))
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::ffi_test::call;

    use super::*;

    #[test]
    fn dist_euclidean_3_4_5_pythagorean() {
        // The canonical right triangle.
        let v = call(
            polars__dist_euclidean,
            json!({"a": [0.0, 0.0], "b": [3.0, 4.0]}),
        );
        assert!((v["distance"].as_f64().unwrap() - 5.0).abs() < 1e-12);
        // Zero vectors → 0.
        let v = call(
            polars__dist_euclidean,
            json!({"a": [0.0, 0.0], "b": [0.0, 0.0]}),
        );
        assert_eq!(v["distance"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn dist_manhattan_sums_axis_diffs() {
        // L1 distance: 1+2+3 = 6.
        let v = call(
            polars__dist_manhattan,
            json!({"a": [0.0, 0.0, 0.0], "b": [1.0, 2.0, 3.0]}),
        );
        assert_eq!(v["distance"].as_f64().unwrap(), 6.0);
    }

    #[test]
    fn dist_chebyshev_picks_max_axis() {
        // L∞ distance: max(|x-y|) = max(1, 5, 2) = 5.
        let v = call(
            polars__dist_chebyshev,
            json!({"a": [0.0, 0.0, 0.0], "b": [1.0, 5.0, 2.0]}),
        );
        assert_eq!(v["distance"].as_f64().unwrap(), 5.0);
    }

    #[test]
    fn dist_minkowski_collapses_to_l2_at_p_2() {
        let v = call(
            polars__dist_minkowski,
            json!({"a": [0.0, 0.0], "b": [3.0, 4.0], "p": 2.0}),
        );
        assert!((v["distance"].as_f64().unwrap() - 5.0).abs() < 1e-12);
        // At p=1 it collapses to Manhattan.
        let v = call(
            polars__dist_minkowski,
            json!({"a": [0.0, 0.0], "b": [3.0, 4.0], "p": 1.0}),
        );
        assert!((v["distance"].as_f64().unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn dist_cosine_orthogonal_vs_parallel() {
        // Orthogonal unit vectors → distance 1 (similarity 0).
        let v = call(
            polars__dist_cosine,
            json!({"a": [1.0, 0.0], "b": [0.0, 1.0]}),
        );
        assert!((v["distance"].as_f64().unwrap() - 1.0).abs() < 1e-12);
        // Same direction → distance 0 (similarity 1).
        let v = call(
            polars__dist_cosine,
            json!({"a": [1.0, 2.0], "b": [2.0, 4.0]}),
        );
        assert!(v["distance"].as_f64().unwrap().abs() < 1e-12);
    }

    #[test]
    fn geo_haversine_zero_for_same_point() {
        let v = call(
            polars__geo_haversine,
            json!({"lat1": 40.7128, "lon1": -74.0060, "lat2": 40.7128, "lon2": -74.0060}),
        );
        assert!(v["distance"].as_f64().unwrap().abs() < 1e-9);
    }

    #[test]
    fn geo_haversine_nyc_to_la() {
        // NYC (40.7128, -74.006) → LA (34.0522, -118.2437) ≈ 3936 km (great-circle).
        let v = call(
            polars__geo_haversine,
            json!({
                "lat1": 40.7128, "lon1": -74.0060,
                "lat2": 34.0522, "lon2": -118.2437,
            }),
        );
        let d = v["distance"].as_f64().unwrap();
        assert!((d - 3936.0).abs() < 20.0, "got {d} km, expected ~3936");
    }

    #[test]
    fn geo_polygon_area_unit_square() {
        // Shoelace on the unit square: area = 1.
        let v = call(
            polars__geo_polygon_area,
            json!({"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0]}),
        );
        assert!((v["area"].as_f64().unwrap() - 1.0).abs() < 1e-12);
        // 3-4-5 right triangle: area = (3*4)/2 = 6.
        let v = call(
            polars__geo_polygon_area,
            json!({"x": [0.0, 4.0, 0.0], "y": [0.0, 0.0, 3.0]}),
        );
        assert!((v["area"].as_f64().unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn geo_polygon_perimeter_unit_square() {
        let v = call(
            polars__geo_polygon_perimeter,
            json!({"x": [0.0, 1.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0, 1.0]}),
        );
        assert!((v["perimeter"].as_f64().unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn geo_point_in_polygon_unit_square() {
        // Inside, outside, and edge cases. The well-known ray-casting
        // implementation has corner-case issues on vertical edges; we test
        // away from those.
        let xs = json!([0.0, 4.0, 4.0, 0.0]);
        let ys = json!([0.0, 0.0, 4.0, 4.0]);
        let v = call(
            polars__geo_point_in_polygon,
            json!({"px": 2.0, "py": 2.0, "x": xs.clone(), "y": ys.clone()}),
        );
        assert_eq!(v["inside"], true, "center inside");
        let v = call(
            polars__geo_point_in_polygon,
            json!({"px": 5.0, "py": 5.0, "x": xs, "y": ys}),
        );
        assert_eq!(v["inside"], false, "far point outside");
    }

    #[test]
    fn geo_bearing_due_north_is_zero() {
        // 0° lon, going from equator north → bearing 0°.
        let v = call(
            polars__geo_bearing,
            json!({"lat1": 0.0, "lon1": 0.0, "lat2": 1.0, "lon2": 0.0}),
        );
        assert!(v["bearing"].as_f64().unwrap().abs() < 1e-9);
    }

    #[test]
    fn hash_djb2_known_vectors() {
        // hash("") = 5381 (the initial seed); the empty string is a strong
        // sanity check that the loop is actually entered conditionally.
        let v = call(polars__hash_djb2, json!({"value": ""}));
        assert_eq!(v["hash"].as_u64().unwrap(), 5381);
    }

    #[test]
    fn hash_sdbm_known_vectors() {
        // sdbm seeds at 0, so the empty string hashes to 0.
        assert_eq!(
            call(polars__hash_sdbm, json!({"value": ""}))["hash"]
                .as_u64()
                .unwrap(),
            0
        );
        // A single byte is just its value (0*65599 + c).
        assert_eq!(
            call(polars__hash_sdbm, json!({"value": "a"}))["hash"]
                .as_u64()
                .unwrap(),
            97
        );
        // Two bytes follow the recurrence h*65599 + c, computed by hand.
        let expected = 97u64.wrapping_mul(65599).wrapping_add(b'b' as u64);
        assert_eq!(
            call(polars__hash_sdbm, json!({"value": "ab"}))["hash"]
                .as_u64()
                .unwrap(),
            expected
        );
        // Deterministic.
        let a = call(polars__hash_sdbm, json!({"value": "stryke"}))["hash"].clone();
        let b = call(polars__hash_sdbm, json!({"value": "stryke"}))["hash"].clone();
        assert_eq!(a, b);
    }

    #[test]
    fn hash_fnv1a_known_vector() {
        // FNV-1a 64-bit initial offset = 0xcbf29ce484222325; "a" gives a
        // deterministic value we can hardcode.
        // hash = ((0xcbf29ce484222325 ^ 'a') * 0x100000001b3) wrapping
        let v = call(polars__hash_fnv1a, json!({"value": ""}));
        assert_eq!(v["hash"].as_u64().unwrap(), 0xcbf29ce484222325);
        // hash("a") computed by hand:
        let expected = (0xcbf29ce484222325_u64 ^ b'a' as u64).wrapping_mul(0x100000001b3);
        let v = call(polars__hash_fnv1a, json!({"value": "a"}));
        assert_eq!(v["hash"].as_u64().unwrap(), expected);
    }

    #[test]
    fn hash_crc32_known_vector() {
        // CRC32("123456789") = 0xCBF43926 — universal test vector for CRC-32/ISO-HDLC.
        let v = call(polars__hash_crc32, json!({"value": "123456789"}));
        assert_eq!(v["hash"].as_u64().unwrap(), 0xCBF43926);
    }

    #[test]
    fn cluster_kmeans_converges_on_separated_clusters() {
        // Two tight clusters around (0,0) and (10,10) should yield 2 clean groups.
        let v = call(
            polars__cluster_kmeans,
            json!({
                "points": {"data": [0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0], "shape": [4, 2]},
                "k": 2,
                "max_iter": 50,
            }),
        );
        let assign: Vec<i64> = v["assignments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        // Points 0/1 share a cluster; 2/3 share the other.
        assert_eq!(assign[0], assign[1]);
        assert_eq!(assign[2], assign[3]);
        assert_ne!(assign[0], assign[2]);
    }

    #[test]
    fn sparse_dense_round_trip() {
        // dense → sparse → dense reproduces the original matrix.
        let dense = json!({"data": [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0], "shape": [3, 3]});
        let sp = call(polars__sparse_from_dense, json!({"matrix": dense.clone()}));
        // 4 nonzeros: 1, 2, 3, 4 at fixed positions.
        let nnz = sp["sparse"]["data"].as_array().unwrap().len();
        assert_eq!(nnz, 4);
        let back = call(polars__sparse_to_dense, json!({"sparse": sp["sparse"]}));
        let expected: Vec<f64> = dense["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let actual: Vec<f64> = back["array"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn sparse_sub_cancels_equal_entries_and_drops_zeros() {
        // A - B where the (1,1) entries are equal → that cell cancels to 0
        // and must NOT appear in the result (explicit zeros dropped).
        let a = call(
            polars__sparse_from_dense,
            json!({"matrix": {"data": [5.0, 0.0, 0.0, 3.0], "shape": [2, 2]}}),
        );
        let b = call(
            polars__sparse_from_dense,
            json!({"matrix": {"data": [1.0, 0.0, 0.0, 3.0], "shape": [2, 2]}}),
        );
        let diff = call(
            polars__sparse_sub,
            json!({"a": a["sparse"], "b": b["sparse"]}),
        );
        // Only (0,0) = 4 survives; (1,1) cancelled.
        assert_eq!(diff["sparse"]["data"].as_array().unwrap().len(), 1);
        let dense = call(polars__sparse_to_dense, json!({"sparse": diff["sparse"]}));
        let got: Vec<f64> = dense["array"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(got, vec![4.0, 0.0, 0.0, 0.0]);
        // Shape mismatch is rejected.
        let wide = call(
            polars__sparse_from_dense,
            json!({"matrix": {"data": [1.0, 2.0, 3.0], "shape": [1, 3]}}),
        );
        let err = call(
            polars__sparse_sub,
            json!({"a": a["sparse"], "b": wide["sparse"]}),
        );
        assert!(err["error"]
            .as_str()
            .unwrap_or("")
            .contains("shape mismatch"));
    }

    #[test]
    fn sparse_mat_mul_matches_dense_product() {
        // A = [[1,2],[0,3]], B = [[4,0],[1,5]]  →  A·B = [[6,10],[3,15]].
        let a = call(
            polars__sparse_from_dense,
            json!({"matrix": {"data": [1.0, 2.0, 0.0, 3.0], "shape": [2, 2]}}),
        );
        let b = call(
            polars__sparse_from_dense,
            json!({"matrix": {"data": [4.0, 0.0, 1.0, 5.0], "shape": [2, 2]}}),
        );
        let prod = call(
            polars__sparse_mat_mul,
            json!({"a": a["sparse"], "b": b["sparse"]}),
        );
        let dense = call(polars__sparse_to_dense, json!({"sparse": prod["sparse"]}));
        let got: Vec<f64> = dense["array"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(got, vec![6.0, 10.0, 3.0, 15.0]);
        // Non-square inner dim: (2×3)·(3×2) = (2×2).
        let m = call(
            polars__sparse_from_dense,
            json!({"matrix": {"data": [1.0, 0.0, 2.0, 0.0, 3.0, 0.0], "shape": [2, 3]}}),
        );
        let n = call(
            polars__sparse_from_dense,
            json!({"matrix": {"data": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0], "shape": [3, 2]}}),
        );
        let mn = call(
            polars__sparse_mat_mul,
            json!({"a": m["sparse"], "b": n["sparse"]}),
        );
        assert_eq!(mn["sparse"]["n_rows"], 2);
        assert_eq!(mn["sparse"]["n_cols"], 2);
        // Inner-dimension mismatch is rejected.
        // Inner-dimension mismatch is rejected: a is 2×2, n is 3×2 (2 ≠ 3).
        let bad = call(
            polars__sparse_mat_mul,
            json!({"a": a["sparse"], "b": n["sparse"]}),
        );
        assert!(bad["error"]
            .as_str()
            .unwrap_or("")
            .contains("inner dimension mismatch"));
    }

    #[test]
    fn opt_minimize_bisection_finds_quadratic_root() {
        // f(x) = x^2 - 4. Roots at ±2. Bracket [0, 3] catches the +2 root.
        // Polynomial-form coefficients = [-4, 0, 1] (lowest-degree first).
        let v = call(
            polars__opt_minimize_bisection,
            json!({"coefficients": [-4.0, 0.0, 1.0], "lo": 0.0, "hi": 3.0, "tol": 1e-9}),
        );
        let x = v["x"].as_f64().unwrap();
        assert!((x - 2.0).abs() < 1e-6, "bisection on x^2-4 found {x}");
    }

    #[test]
    fn ts_max_drawdown_matches_manual_calc() {
        // Series [100, 110, 90, 105]. Peak 110, trough 90 → DD = -20/110.
        let v = call(
            polars__ts_max_drawdown,
            json!({"data": [100.0, 110.0, 90.0, 105.0]}),
        );
        let mdd = v["max_drawdown"].as_f64().unwrap();
        assert!((mdd - (-20.0 / 110.0)).abs() < 1e-12, "mdd = {mdd}");
    }

    #[test]
    fn ts_log_returns_match_manual_calc() {
        // log_return[i] = ln(p[i] / p[i-1]); first entry is NaN.
        let v = call(polars__ts_log_returns, json!({"data": [1.0, 2.0, 4.0]}));
        let d: Vec<f64> = v["array"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN))
            .collect();
        assert!(d[0].is_nan());
        assert!((d[1] - 2.0_f64.ln()).abs() < 1e-12);
        assert!((d[2] - 2.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn enc_one_hot_dimensions_match_categories() {
        let v = call(
            polars__enc_one_hot,
            json!({"labels": ["red", "blue", "red", "green"]}),
        );
        let cats = v["categories"].as_array().unwrap();
        let shape = v["array"]["shape"].as_array().unwrap();
        assert_eq!(cats.len(), 3, "3 unique categories");
        assert_eq!(shape[0].as_u64().unwrap(), 4, "n samples");
        assert_eq!(shape[1].as_u64().unwrap(), 3, "n categories");
        // Each row sums to 1 (exactly one 1).
        let data: Vec<f64> = v["array"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        for chunk in data.chunks(3) {
            let s: f64 = chunk.iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn enc_frequency_maps_each_label_to_its_proportion() {
        // red×2, blue×1, green×1 of 4 → 0.5 / 0.25 / 0.25, in input order.
        let v = call(
            polars__enc_frequency,
            json!({"labels": ["red", "blue", "red", "green"]}),
        );
        let freqs: Vec<f64> = v["frequencies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(freqs, vec![0.5, 0.25, 0.5, 0.25]);
        // Per-category counts.
        assert_eq!(v["counts"]["red"].as_u64().unwrap(), 2);
        assert_eq!(v["counts"]["blue"].as_u64().unwrap(), 1);
        // Every frequency is in (0, 1] and the distinct values sum to 1.
        assert!(freqs.iter().all(|&f| f > 0.0 && f <= 1.0));
        // A single uniform category → all 1.0.
        let u = call(polars__enc_frequency, json!({"labels": ["x", "x", "x"]}));
        assert_eq!(
            u["frequencies"].as_array().unwrap(),
            &vec![json!(1.0), json!(1.0), json!(1.0)]
        );
        // Missing labels errors.
        assert!(call(polars__enc_frequency, json!({}))
            .get("error")
            .is_some());
    }
}
