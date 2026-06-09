//! src/extras3.rs — ML metrics, text/NLP ops, graph helpers, more linalg.

use std::ffi::c_char;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Value};

use crate::ffi_call;

fn get_vec(args: &Value, key: &str) -> Result<Vec<f64>> {
    let a = args
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `{key}`"))?;
    Ok(a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
}

fn get_str(args: &Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| anyhow!("missing `{key}`"))
}

fn scalar(x: f64) -> Value {
    serde_json::Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
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

// ── ML metrics (metric_*) ─────────────────────────────────────────────────

/// ML metric accuracy.
#[no_mangle]
pub extern "C" fn polars__metric_accuracy(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        if y.len() != p.len() {
            bail!("length mismatch");
        }
        let correct = y
            .iter()
            .zip(p.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-9)
            .count() as f64;
        Ok(json!({"accuracy": scalar(correct / y.len() as f64)}))
    })
}

/// ML metric precision.
#[no_mangle]
pub extern "C" fn polars__metric_precision(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let tp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 1.0)
            .count() as f64;
        let fp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 0.0 && **p == 1.0)
            .count() as f64;
        let r = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
        Ok(json!({"precision": scalar(r)}))
    })
}

/// ML metric recall.
#[no_mangle]
pub extern "C" fn polars__metric_recall(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let tp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 1.0)
            .count() as f64;
        let fn_ = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 0.0)
            .count() as f64;
        let r = if tp + fn_ == 0.0 {
            0.0
        } else {
            tp / (tp + fn_)
        };
        Ok(json!({"recall": scalar(r)}))
    })
}

/// ML metric f1.
#[no_mangle]
pub extern "C" fn polars__metric_f1(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let tp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 1.0)
            .count() as f64;
        let fp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 0.0 && **p == 1.0)
            .count() as f64;
        let fn_ = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 0.0)
            .count() as f64;
        let prec = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
        let rec = if tp + fn_ == 0.0 {
            0.0
        } else {
            tp / (tp + fn_)
        };
        let f1 = if prec + rec == 0.0 {
            0.0
        } else {
            2.0 * prec * rec / (prec + rec)
        };
        Ok(json!({"f1": scalar(f1)}))
    })
}

/// ML metric fbeta.
#[no_mangle]
pub extern "C" fn polars__metric_fbeta(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let beta = args.get("beta").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let tp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 1.0)
            .count() as f64;
        let fp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 0.0 && **p == 1.0)
            .count() as f64;
        let fn_ = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 0.0)
            .count() as f64;
        let prec = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
        let rec = if tp + fn_ == 0.0 {
            0.0
        } else {
            tp / (tp + fn_)
        };
        let b2 = beta * beta;
        let r = if prec + rec == 0.0 {
            0.0
        } else {
            (1.0 + b2) * prec * rec / (b2 * prec + rec)
        };
        Ok(json!({"fbeta": scalar(r)}))
    })
}

/// ML metric confusion matrix.
#[no_mangle]
pub extern "C" fn polars__metric_confusion_matrix(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let n_classes = args.get("n_classes").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let mut cm = vec![vec![0i64; n_classes]; n_classes];
        for (t, pr) in y.iter().zip(p.iter()) {
            let ti = *t as usize;
            let pi = *pr as usize;
            if ti < n_classes && pi < n_classes {
                cm[ti][pi] += 1;
            }
        }
        let flat: Vec<i64> = cm.iter().flatten().copied().collect();
        Ok(json!({"confusion_matrix": flat, "shape": [n_classes, n_classes]}))
    })
}

/// ML metric roc auc.
#[no_mangle]
pub extern "C" fn polars__metric_roc_auc(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_score")?;
        let mut pairs: Vec<(f64, f64)> = y.into_iter().zip(p).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let pos: f64 = pairs.iter().map(|(t, _)| t).sum();
        let neg = pairs.len() as f64 - pos;
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut auc = 0.0;
        let mut prev_tpr = 0.0;
        let mut prev_fpr = 0.0;
        for (t, _) in pairs {
            if t == 1.0 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            let tpr = if pos == 0.0 { 0.0 } else { tp / pos };
            let fpr = if neg == 0.0 { 0.0 } else { fp / neg };
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_tpr = tpr;
            prev_fpr = fpr;
        }
        Ok(json!({"auc": scalar(auc)}))
    })
}

/// ML metric log loss.
#[no_mangle]
pub extern "C" fn polars__metric_log_loss(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_prob")?;
        let eps = 1e-15;
        let loss: f64 = y
            .iter()
            .zip(p.iter())
            .map(|(t, p)| {
                let p = p.clamp(eps, 1.0 - eps);
                -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
            })
            .sum::<f64>()
            / y.len() as f64;
        Ok(json!({"log_loss": scalar(loss)}))
    })
}

/// ML metric hinge loss.
#[no_mangle]
pub extern "C" fn polars__metric_hinge_loss(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_score")?;
        let loss: f64 = y
            .iter()
            .zip(p.iter())
            .map(|(t, p)| (1.0 - t * p).max(0.0))
            .sum::<f64>()
            / y.len() as f64;
        Ok(json!({"hinge_loss": scalar(loss)}))
    })
}

/// ML metric matthews corrcoef.
#[no_mangle]
pub extern "C" fn polars__metric_matthews_corrcoef(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let tp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 1.0)
            .count() as f64;
        let tn = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 0.0 && **p == 0.0)
            .count() as f64;
        let fp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 0.0 && **p == 1.0)
            .count() as f64;
        let fn_ = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 0.0)
            .count() as f64;
        let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        let r = if denom == 0.0 {
            0.0
        } else {
            (tp * tn - fp * fn_) / denom
        };
        Ok(json!({"mcc": scalar(r)}))
    })
}

/// ML metric cohen kappa.
#[no_mangle]
pub extern "C" fn polars__metric_cohen_kappa(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let n = y.len() as f64;
        let po = y.iter().zip(p.iter()).filter(|(t, p)| **t == **p).count() as f64 / n;
        let p_y_1: f64 = y.iter().filter(|x| **x == 1.0).count() as f64 / n;
        let p_p_1: f64 = p.iter().filter(|x| **x == 1.0).count() as f64 / n;
        let pe = p_y_1 * p_p_1 + (1.0 - p_y_1) * (1.0 - p_p_1);
        let kappa = if pe == 1.0 {
            0.0
        } else {
            (po - pe) / (1.0 - pe)
        };
        Ok(json!({"kappa": scalar(kappa)}))
    })
}

/// ML metric balanced accuracy.
#[no_mangle]
pub extern "C" fn polars__metric_balanced_accuracy(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let tp = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 1.0)
            .count() as f64;
        let tn = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 0.0 && **p == 0.0)
            .count() as f64;
        let pos = y.iter().filter(|x| **x == 1.0).count() as f64;
        let neg = y.iter().filter(|x| **x == 0.0).count() as f64;
        let sens = if pos == 0.0 { 0.0 } else { tp / pos };
        let spec = if neg == 0.0 { 0.0 } else { tn / neg };
        Ok(json!({"balanced_accuracy": scalar((sens + spec) / 2.0)}))
    })
}

/// ML metric mae.
#[no_mangle]
pub extern "C" fn polars__metric_mae(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let mae: f64 = y
            .iter()
            .zip(p.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / y.len() as f64;
        Ok(json!({"mae": scalar(mae)}))
    })
}

/// ML metric mse.
#[no_mangle]
pub extern "C" fn polars__metric_mse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let mse: f64 = y
            .iter()
            .zip(p.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        Ok(json!({"mse": scalar(mse)}))
    })
}

/// ML metric rmse.
#[no_mangle]
pub extern "C" fn polars__metric_rmse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let mse: f64 = y
            .iter()
            .zip(p.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        Ok(json!({"rmse": scalar(mse.sqrt())}))
    })
}

/// ML metric mape.
#[no_mangle]
pub extern "C" fn polars__metric_mape(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let n = y.len() as f64;
        let mape: f64 = y
            .iter()
            .zip(p.iter())
            .filter(|(t, _)| **t != 0.0)
            .map(|(t, pr)| ((t - pr) / t).abs())
            .sum::<f64>()
            / n;
        Ok(json!({"mape": scalar(mape)}))
    })
}

/// ML metric smape.
#[no_mangle]
pub extern "C" fn polars__metric_smape(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let n = y.len() as f64;
        let smape: f64 = y
            .iter()
            .zip(p.iter())
            .map(|(t, pr)| {
                let d = t.abs() + pr.abs();
                if d == 0.0 {
                    0.0
                } else {
                    2.0 * (t - pr).abs() / d
                }
            })
            .sum::<f64>()
            / n;
        Ok(json!({"smape": scalar(smape)}))
    })
}

/// ML metric r2.
#[no_mangle]
pub extern "C" fn polars__metric_r2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let m = y.iter().sum::<f64>() / y.len() as f64;
        let ss_tot: f64 = y.iter().map(|x| (x - m).powi(2)).sum();
        let ss_res: f64 = y.iter().zip(p.iter()).map(|(t, pr)| (t - pr).powi(2)).sum();
        let r2 = if ss_tot == 0.0 {
            f64::NAN
        } else {
            1.0 - ss_res / ss_tot
        };
        Ok(json!({"r2": scalar(r2)}))
    })
}

/// ML metric explained variance.
#[no_mangle]
pub extern "C" fn polars__metric_explained_variance(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let m_y = y.iter().sum::<f64>() / y.len() as f64;
        let var_y: f64 = y.iter().map(|x| (x - m_y).powi(2)).sum::<f64>() / y.len() as f64;
        let diff: Vec<f64> = y.iter().zip(p.iter()).map(|(t, pr)| t - pr).collect();
        let m_d = diff.iter().sum::<f64>() / diff.len() as f64;
        let var_diff: f64 = diff.iter().map(|x| (x - m_d).powi(2)).sum::<f64>() / diff.len() as f64;
        let ev = if var_y == 0.0 {
            f64::NAN
        } else {
            1.0 - var_diff / var_y
        };
        Ok(json!({"explained_variance": scalar(ev)}))
    })
}

/// ML metric brier score.
#[no_mangle]
pub extern "C" fn polars__metric_brier_score(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_prob")?;
        let b: f64 = y
            .iter()
            .zip(p.iter())
            .map(|(t, pr)| (t - pr).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        Ok(json!({"brier_score": scalar(b)}))
    })
}

/// ML metric top k accuracy.
#[no_mangle]
pub extern "C" fn polars__metric_top_k_accuracy(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let scores = args
            .get("y_score")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `y_score`"))?;
        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let mut correct = 0;
        let n = y.len();
        for i in 0..n {
            let row = scores[i]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect::<Vec<_>>())
                .unwrap_or_default();
            let mut indexed: Vec<(usize, f64)> = row.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
            if top_k.contains(&(y[i] as usize)) {
                correct += 1;
            }
        }
        Ok(json!({"top_k_accuracy": scalar(correct as f64 / n as f64)}))
    })
}

/// ML metric jaccard score.
#[no_mangle]
pub extern "C" fn polars__metric_jaccard_score(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let y = get_vec(&args, "y_true")?;
        let p = get_vec(&args, "y_pred")?;
        let inter = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 && **p == 1.0)
            .count() as f64;
        let union = y
            .iter()
            .zip(p.iter())
            .filter(|(t, p)| **t == 1.0 || **p == 1.0)
            .count() as f64;
        let r = if union == 0.0 { 0.0 } else { inter / union };
        Ok(json!({"jaccard": scalar(r)}))
    })
}

// ── text/NLP (text_*) ──────────────────────────────────────────────────────

/// Text tokenize.
#[no_mangle]
pub extern "C" fn polars__text_tokenize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let tokens: Vec<String> = s.split_whitespace().map(String::from).collect();
        Ok(json!({"tokens": tokens}))
    })
}

/// Text word count.
#[no_mangle]
pub extern "C" fn polars__text_word_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let n = s.split_whitespace().count();
        Ok(json!({"word_count": n}))
    })
}

/// Text char count.
#[no_mangle]
pub extern "C" fn polars__text_char_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        Ok(json!({"char_count": s.chars().count()}))
    })
}

/// Text byte count.
#[no_mangle]
pub extern "C" fn polars__text_byte_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        Ok(json!({"byte_count": s.len()}))
    })
}

/// Text line count.
#[no_mangle]
pub extern "C" fn polars__text_line_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        Ok(json!({"line_count": s.lines().count()}))
    })
}

/// Text ngrams.
#[no_mangle]
pub extern "C" fn polars__text_ngrams(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let tokens: Vec<&str> = s.split_whitespace().collect();
        if tokens.len() < n {
            return Ok(json!({"ngrams": Vec::<String>::new()}));
        }
        let ngrams: Vec<String> = tokens.windows(n).map(|w| w.join(" ")).collect();
        Ok(json!({"ngrams": ngrams}))
    })
}

/// Text char ngrams.
#[no_mangle]
pub extern "C" fn polars__text_char_ngrams(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let chars: Vec<char> = s.chars().collect();
        if chars.len() < n {
            return Ok(json!({"ngrams": Vec::<String>::new()}));
        }
        let ngrams: Vec<String> = chars.windows(n).map(|w| w.iter().collect()).collect();
        Ok(json!({"ngrams": ngrams}))
    })
}

/// Text word frequency.
#[no_mangle]
pub extern "C" fn polars__text_word_frequency(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
        for w in s.split_whitespace() {
            *counts.entry(w.to_string()).or_insert(0) += 1;
        }
        let mut out: Vec<Value> = counts
            .into_iter()
            .map(|(k, v)| json!({"word": k, "count": v}))
            .collect();
        out.sort_by(|a, b| {
            b["count"]
                .as_u64()
                .unwrap_or(0)
                .cmp(&a["count"].as_u64().unwrap_or(0))
        });
        Ok(json!({"frequency": out}))
    })
}

/// Text unique words.
#[no_mangle]
pub extern "C" fn polars__text_unique_words(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let set: std::collections::HashSet<&str> = s.split_whitespace().collect();
        Ok(json!({"unique_words": set.len()}))
    })
}

/// Text lower.
#[no_mangle]
pub extern "C" fn polars__text_lower(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        Ok(json!({"text": s.to_lowercase()}))
    })
}

/// Text upper.
#[no_mangle]
pub extern "C" fn polars__text_upper(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        Ok(json!({"text": s.to_uppercase()}))
    })
}

/// Text reverse.
#[no_mangle]
pub extern "C" fn polars__text_reverse(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        Ok(json!({"text": s.chars().rev().collect::<String>()}))
    })
}

/// Text strip.
#[no_mangle]
pub extern "C" fn polars__text_strip(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        Ok(json!({"text": s.trim().to_string()}))
    })
}

/// Text levenshtein.
#[no_mangle]
pub extern "C" fn polars__text_levenshtein(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_str(&args, "a")?;
        let b = get_str(&args, "b")?;
        let av: Vec<char> = a.chars().collect();
        let bv: Vec<char> = b.chars().collect();
        let m = av.len();
        let n = bv.len();
        let mut dp = vec![vec![0_usize; n + 1]; m + 1];
        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, val) in dp[0].iter_mut().enumerate() {
            *val = j;
        }
        for i in 1..=m {
            for j in 1..=n {
                let cost = if av[i - 1] == bv[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1).min((dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost));
            }
        }
        Ok(json!({"levenshtein": dp[m][n]}))
    })
}

/// Text jaccard.
#[no_mangle]
pub extern "C" fn polars__text_jaccard(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_str(&args, "a")?;
        let b = get_str(&args, "b")?;
        let sa: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let sb: std::collections::HashSet<&str> = b.split_whitespace().collect();
        let inter = sa.intersection(&sb).count() as f64;
        let union = sa.union(&sb).count() as f64;
        let j = if union == 0.0 { 0.0 } else { inter / union };
        Ok(json!({"jaccard": scalar(j)}))
    })
}

/// Text cosine word.
#[no_mangle]
pub extern "C" fn polars__text_cosine_word(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_str(&args, "a")?;
        let b = get_str(&args, "b")?;
        let mut ca: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        let mut cb: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        for w in a.split_whitespace() {
            *ca.entry(w).or_insert(0.0) += 1.0;
        }
        for w in b.split_whitespace() {
            *cb.entry(w).or_insert(0.0) += 1.0;
        }
        let words: std::collections::HashSet<&str> = ca.keys().chain(cb.keys()).copied().collect();
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for w in &words {
            let va = ca.get(w).copied().unwrap_or(0.0);
            let vb = cb.get(w).copied().unwrap_or(0.0);
            dot += va * vb;
            na += va * va;
            nb += vb * vb;
        }
        let c = if na * nb == 0.0 {
            0.0
        } else {
            dot / (na.sqrt() * nb.sqrt())
        };
        Ok(json!({"cosine": scalar(c)}))
    })
}

/// Text sentence split.
#[no_mangle]
pub extern "C" fn polars__text_sentence_split(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let sentences: Vec<String> = s
            .split(['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Ok(json!({"sentences": sentences}))
    })
}

/// Text stopwords remove.
#[no_mangle]
pub extern "C" fn polars__text_stopwords_remove(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let stops = args
            .get("stopwords")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `stopwords`"))?;
        let stopset: std::collections::HashSet<String> = stops
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        let out: String = s
            .split_whitespace()
            .filter(|w| !stopset.contains(*w))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(json!({"text": out}))
    })
}

/// Text replace regex.
#[no_mangle]
pub extern "C" fn polars__text_replace_regex(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let pat = get_str(&args, "pattern")?;
        let to = get_str(&args, "replacement")?;
        // Simple non-regex replace; full regex would need the regex crate.
        Ok(json!({"text": s.replace(&pat, &to)}))
    })
}

/// Text extract numbers.
#[no_mangle]
pub extern "C" fn polars__text_extract_numbers(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let nums: Vec<f64> = s
            .split(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
            .filter_map(|x| x.parse::<f64>().ok())
            .collect();
        Ok(json!({"numbers": nums}))
    })
}

/// Text lcs.
#[no_mangle]
pub extern "C" fn polars__text_lcs(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_str(&args, "a")?;
        let b = get_str(&args, "b")?;
        let av: Vec<char> = a.chars().collect();
        let bv: Vec<char> = b.chars().collect();
        let m = av.len();
        let n = bv.len();
        let mut dp = vec![vec![0_usize; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                if av[i - 1] == bv[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        Ok(json!({"length": dp[m][n]}))
    })
}

/// Text edit distance ratio.
#[no_mangle]
pub extern "C" fn polars__text_edit_distance_ratio(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_str(&args, "a")?;
        let b = get_str(&args, "b")?;
        let av: Vec<char> = a.chars().collect();
        let bv: Vec<char> = b.chars().collect();
        let m = av.len();
        let n = bv.len();
        let mut dp = vec![vec![0_usize; n + 1]; m + 1];
        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, val) in dp[0].iter_mut().enumerate() {
            *val = j;
        }
        for i in 1..=m {
            for j in 1..=n {
                let cost = if av[i - 1] == bv[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1).min((dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost));
            }
        }
        let max_len = m.max(n) as f64;
        let ratio = if max_len == 0.0 {
            1.0
        } else {
            1.0 - dp[m][n] as f64 / max_len
        };
        Ok(json!({"ratio": scalar(ratio)}))
    })
}

/// Text camel to snake.
#[no_mangle]
pub extern "C" fn polars__text_camel_to_snake(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let mut out = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                out.push('_');
                out.push(c.to_ascii_lowercase());
            } else {
                out.push(c.to_ascii_lowercase());
            }
        }
        Ok(json!({"text": out}))
    })
}

/// Text snake to camel.
#[no_mangle]
pub extern "C" fn polars__text_snake_to_camel(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let mut out = String::new();
        let mut next_upper = false;
        for c in s.chars() {
            if c == '_' {
                next_upper = true;
            } else if next_upper {
                out.push(c.to_ascii_uppercase());
                next_upper = false;
            } else {
                out.push(c);
            }
        }
        Ok(json!({"text": out}))
    })
}

/// Text pad to.
#[no_mangle]
pub extern "C" fn polars__text_pad_to(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let width = args.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let pad = args
            .get("pad")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or(' ');
        let n = s.chars().count();
        if n >= width {
            return Ok(json!({"text": s}));
        }
        let padding: String = std::iter::repeat_n(pad, width - n).collect();
        Ok(json!({"text": format!("{padding}{s}")}))
    })
}

/// Text truncate.
#[no_mangle]
pub extern "C" fn polars__text_truncate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let width = args.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let truncated: String = s.chars().take(width).collect();
        Ok(json!({"text": truncated}))
    })
}

/// Text titlecase.
#[no_mangle]
pub extern "C" fn polars__text_titlecase(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "text")?;
        let out: String = s
            .split_whitespace()
            .map(|w| {
                let mut c = w.chars();
                match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        Ok(json!({"text": out}))
    })
}

// ── graph (graph_*) ────────────────────────────────────────────────────────

/// (u, v, w) edge tuple.
type Edge = (usize, usize, f64);

fn parse_graph(args: &Value) -> Result<(usize, Vec<Edge>)> {
    let n_nodes = args
        .get("n_nodes")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing `n_nodes`"))? as usize;
    let edges = args
        .get("edges")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `edges`"))?;
    let mut out = vec![];
    for e in edges {
        let arr = e.as_array().ok_or_else(|| anyhow!("edge must be array"))?;
        let u = arr.first().and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let v = arr.get(1).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let w = arr.get(2).and_then(|x| x.as_f64()).unwrap_or(1.0);
        out.push((u, v, w));
    }
    Ok((n_nodes, out))
}

/// Graph degree.
#[no_mangle]
pub extern "C" fn polars__graph_degree(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let mut degree = vec![0i64; n];
        for (u, v, _) in edges {
            degree[u] += 1;
            degree[v] += 1;
        }
        Ok(json!({"degree": degree}))
    })
}

/// Graph in degree.
#[no_mangle]
pub extern "C" fn polars__graph_in_degree(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let mut d = vec![0i64; n];
        for (_, v, _) in edges {
            d[v] += 1;
        }
        Ok(json!({"in_degree": d}))
    })
}

/// Graph out degree.
#[no_mangle]
pub extern "C" fn polars__graph_out_degree(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let mut d = vec![0i64; n];
        for (u, _, _) in edges {
            d[u] += 1;
        }
        Ok(json!({"out_degree": d}))
    })
}

/// Graph bfs.
#[no_mangle]
pub extern "C" fn polars__graph_bfs(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let start = args.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let mut adj = vec![vec![]; n];
        for (u, v, _) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut visited = vec![false; n];
        let mut order = vec![];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(node) = queue.pop_front() {
            order.push(node as i64);
            for &next in &adj[node] {
                if !visited[next] {
                    visited[next] = true;
                    queue.push_back(next);
                }
            }
        }
        Ok(json!({"order": order}))
    })
}

/// Graph dfs.
#[no_mangle]
pub extern "C" fn polars__graph_dfs(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let start = args.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let mut adj = vec![vec![]; n];
        for (u, v, _) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut visited = vec![false; n];
        let mut order = vec![];
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            order.push(node as i64);
            for &next in adj[node].iter().rev() {
                if !visited[next] {
                    stack.push(next);
                }
            }
        }
        Ok(json!({"order": order}))
    })
}

/// Graph shortest path.
#[no_mangle]
pub extern "C" fn polars__graph_shortest_path(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let start = args.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for (u, v, w) in edges {
            adj[u].push((v, w));
            adj[v].push((u, w));
        }
        // Dijkstra.
        let mut dist = vec![f64::INFINITY; n];
        dist[start] = 0.0;
        let mut visited = vec![false; n];
        for _ in 0..n {
            let mut best = (usize::MAX, f64::INFINITY);
            for i in 0..n {
                if !visited[i] && dist[i] < best.1 {
                    best = (i, dist[i]);
                }
            }
            if best.0 == usize::MAX {
                break;
            }
            visited[best.0] = true;
            for &(v, w) in &adj[best.0] {
                if dist[best.0] + w < dist[v] {
                    dist[v] = dist[best.0] + w;
                }
            }
        }
        Ok(json!({"distances": dist}))
    })
}

/// Graph connected components.
#[no_mangle]
pub extern "C" fn polars__graph_connected_components(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let mut adj = vec![vec![]; n];
        for (u, v, _) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut comp = vec![-1i64; n];
        let mut cid = 0i64;
        for s in 0..n {
            if comp[s] != -1 {
                continue;
            }
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(s);
            comp[s] = cid;
            while let Some(node) = queue.pop_front() {
                for &next in &adj[node] {
                    if comp[next] == -1 {
                        comp[next] = cid;
                        queue.push_back(next);
                    }
                }
            }
            cid += 1;
        }
        Ok(json!({"components": comp, "n_components": cid}))
    })
}

/// Graph adjacency matrix.
#[no_mangle]
pub extern "C" fn polars__graph_adjacency_matrix(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let mut m = vec![0.0; n * n];
        for (u, v, w) in edges {
            m[u * n + v] = w;
            m[v * n + u] = w;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, n]), m).context("adj")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Graph density.
#[no_mangle]
pub extern "C" fn polars__graph_density(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let m = edges.len() as f64;
        let nf = n as f64;
        let max_edges = nf * (nf - 1.0) / 2.0;
        let d = if max_edges == 0.0 { 0.0 } else { m / max_edges };
        Ok(json!({"density": scalar(d)}))
    })
}

/// Graph average degree.
#[no_mangle]
pub extern "C" fn polars__graph_average_degree(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let total: f64 = edges.len() as f64 * 2.0;
        Ok(json!({"average_degree": scalar(total / n as f64)}))
    })
}

/// Graph floyd warshall.
#[no_mangle]
pub extern "C" fn polars__graph_floyd_warshall(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let mut d = vec![vec![f64::INFINITY; n]; n];
        for (i, row) in d.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        for (u, v, w) in edges {
            d[u][v] = d[u][v].min(w);
            d[v][u] = d[v][u].min(w);
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if d[i][k] + d[k][j] < d[i][j] {
                        d[i][j] = d[i][k] + d[k][j];
                    }
                }
            }
        }
        let flat: Vec<f64> = d.into_iter().flatten().collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n, n]), flat).context("floyd")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Graph pagerank.
#[no_mangle]
pub extern "C" fn polars__graph_pagerank(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let damping = args.get("damping").and_then(|v| v.as_f64()).unwrap_or(0.85);
        let iters = args.get("iters").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let mut adj_out = vec![vec![]; n];
        for (u, v, _) in &edges {
            adj_out[*u].push(*v);
        }
        let mut rank = vec![1.0 / n as f64; n];
        for _ in 0..iters {
            let mut new_rank = vec![(1.0 - damping) / n as f64; n];
            for u in 0..n {
                if adj_out[u].is_empty() {
                    continue;
                }
                let share = damping * rank[u] / adj_out[u].len() as f64;
                for &v in &adj_out[u] {
                    new_rank[v] += share;
                }
            }
            rank = new_rank;
        }
        Ok(json!({"pagerank": rank}))
    })
}

/// Graph clustering coef.
#[no_mangle]
pub extern "C" fn polars__graph_clustering_coef(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (n, edges) = parse_graph(&args)?;
        let mut adj: Vec<std::collections::HashSet<usize>> =
            vec![std::collections::HashSet::new(); n];
        for (u, v, _) in edges {
            adj[u].insert(v);
            adj[v].insert(u);
        }
        let mut cc = vec![0.0; n];
        for i in 0..n {
            let neighbors: Vec<&usize> = adj[i].iter().collect();
            let k = neighbors.len();
            if k < 2 {
                continue;
            }
            let mut links = 0;
            for j in 0..k {
                for l in j + 1..k {
                    if adj[*neighbors[j]].contains(neighbors[l]) {
                        links += 1;
                    }
                }
            }
            cc[i] = 2.0 * links as f64 / (k * (k - 1)) as f64;
        }
        Ok(json!({"clustering": cc}))
    })
}

// ── linalg extensions (linalg_*) ──────────────────────────────────────────

/// Linear algebra matrix power v2.
#[no_mangle]
pub extern "C" fn polars__linalg_matrix_power_v2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        let n = arr.shape()[0];
        if arr.shape().len() != 2 || arr.shape()[1] != n {
            bail!("matrix_power: square matrix required");
        }
        let p = args.get("p").and_then(|v| v.as_i64()).unwrap_or(1);
        if p < 0 {
            bail!("p must be >= 0");
        }
        let dm = nalgebra::DMatrix::from_iterator(n, n, arr.iter().copied());
        let mut acc = nalgebra::DMatrix::<f64>::identity(n, n);
        for _ in 0..p {
            acc *= &dm;
        }
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = acc[(i, j)];
            }
        }
        let arr_out = ArrayD::from_shape_vec(IxDyn(&[n, n]), out).context("matrix_power")?;
        Ok(json!({"matrix": array_to_value(&arr_out)}))
    })
}

fn parse_array_local(v: &Value) -> Result<ArrayD<f64>> {
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

/// Linear algebra expm.
#[no_mangle]
pub extern "C" fn polars__linalg_expm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        let n = arr.shape()[0];
        if arr.shape().len() != 2 || arr.shape()[1] != n {
            bail!("expm: square matrix required");
        }
        let dm = nalgebra::DMatrix::from_iterator(n, n, arr.iter().copied());
        // Pade-like via series (truncated). Not numerically robust but no extra deps.
        let mut acc = nalgebra::DMatrix::<f64>::identity(n, n);
        let mut term = nalgebra::DMatrix::<f64>::identity(n, n);
        for k in 1..20 {
            term = &term * &dm / k as f64;
            acc += &term;
        }
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = acc[(i, j)];
            }
        }
        let arr_out = ArrayD::from_shape_vec(IxDyn(&[n, n]), out).context("expm")?;
        Ok(json!({"matrix": array_to_value(&arr_out)}))
    })
}

/// Linear algebra frobenius norm.
#[no_mangle]
pub extern "C" fn polars__linalg_frobenius_norm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        let s: f64 = arr.iter().map(|x| x * x).sum();
        Ok(json!({"frobenius": scalar(s.sqrt())}))
    })
}

/// Linear algebra l1 norm.
#[no_mangle]
pub extern "C" fn polars__linalg_l1_norm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        let s: f64 = arr.iter().map(|x| x.abs()).sum();
        Ok(json!({"l1": scalar(s)}))
    })
}

/// Linear algebra l2 norm.
#[no_mangle]
pub extern "C" fn polars__linalg_l2_norm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        let s: f64 = arr.iter().map(|x| x * x).sum();
        Ok(json!({"l2": scalar(s.sqrt())}))
    })
}

/// Linear algebra inf norm.
#[no_mangle]
pub extern "C" fn polars__linalg_inf_norm(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        let r = arr.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        Ok(json!({"inf_norm": scalar(r)}))
    })
}

/// Linear algebra condition number.
#[no_mangle]
pub extern "C" fn polars__linalg_condition_number(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        if arr.shape().len() != 2 {
            bail!("matrix required");
        }
        let dm =
            nalgebra::DMatrix::from_iterator(arr.shape()[0], arr.shape()[1], arr.iter().copied());
        let svd = dm.svd(false, false);
        let sv: Vec<f64> = svd.singular_values.iter().copied().collect();
        if sv.is_empty() {
            return Ok(json!({"condition": Value::Null}));
        }
        let mx = sv.iter().cloned().fold(0.0_f64, f64::max);
        let mn = sv.iter().cloned().fold(f64::INFINITY, f64::min);
        let cond = if mn == 0.0 { f64::INFINITY } else { mx / mn };
        Ok(json!({"condition": scalar(cond)}))
    })
}

/// Linear algebra is symmetric.
#[no_mangle]
pub extern "C" fn polars__linalg_is_symmetric(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        if arr.shape().len() != 2 || arr.shape()[0] != arr.shape()[1] {
            return Ok(json!({"is_symmetric": false}));
        }
        let n = arr.shape()[0];
        for i in 0..n {
            for j in i + 1..n {
                if (arr[[i, j].as_slice()] - arr[[j, i].as_slice()]).abs() > 1e-12 {
                    return Ok(json!({"is_symmetric": false}));
                }
            }
        }
        Ok(json!({"is_symmetric": true}))
    })
}

/// Linear algebra is orthogonal.
#[no_mangle]
pub extern "C" fn polars__linalg_is_orthogonal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        if arr.shape().len() != 2 || arr.shape()[0] != arr.shape()[1] {
            return Ok(json!({"is_orthogonal": false}));
        }
        let n = arr.shape()[0];
        let dm = nalgebra::DMatrix::from_iterator(n, n, arr.iter().copied());
        let prod = &dm * dm.transpose();
        let mut ok = true;
        for i in 0..n {
            for j in 0..n {
                let target = if i == j { 1.0 } else { 0.0 };
                if (prod[(i, j)] - target).abs() > 1e-9 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
        }
        Ok(json!({"is_orthogonal": ok}))
    })
}

/// Linear algebra normalize rows.
#[no_mangle]
pub extern "C" fn polars__linalg_normalize_rows(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        if arr.shape().len() != 2 {
            bail!("matrix required");
        }
        let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            let row_norm: f64 = (0..cols)
                .map(|c| arr[[r, c].as_slice()].powi(2))
                .sum::<f64>()
                .sqrt();
            for c in 0..cols {
                out[r * cols + c] = if row_norm == 0.0 {
                    0.0
                } else {
                    arr[[r, c].as_slice()] / row_norm
                };
            }
        }
        let arr_out =
            ArrayD::from_shape_vec(IxDyn(&[rows, cols]), out).context("normalize_rows")?;
        Ok(json!({"matrix": array_to_value(&arr_out)}))
    })
}

/// Linear algebra normalize cols.
#[no_mangle]
pub extern "C" fn polars__linalg_normalize_cols(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        if arr.shape().len() != 2 {
            bail!("matrix required");
        }
        let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
        let mut out = vec![0.0; rows * cols];
        for c in 0..cols {
            let col_norm: f64 = (0..rows)
                .map(|r| arr[[r, c].as_slice()].powi(2))
                .sum::<f64>()
                .sqrt();
            for r in 0..rows {
                out[r * cols + c] = if col_norm == 0.0 {
                    0.0
                } else {
                    arr[[r, c].as_slice()] / col_norm
                };
            }
        }
        let arr_out =
            ArrayD::from_shape_vec(IxDyn(&[rows, cols]), out).context("normalize_cols")?;
        Ok(json!({"matrix": array_to_value(&arr_out)}))
    })
}

/// Linear algebra gram schmidt.
#[no_mangle]
pub extern "C" fn polars__linalg_gram_schmidt(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let m = args
            .get("matrix")
            .ok_or_else(|| anyhow!("missing `matrix`"))?;
        let arr = parse_array_local(m)?;
        if arr.shape().len() != 2 {
            bail!("matrix required");
        }
        let (rows, cols) = (arr.shape()[0], arr.shape()[1]);
        // Treat columns as vectors.
        let cols_vec: Vec<Vec<f64>> = (0..cols)
            .map(|c| (0..rows).map(|r| arr[[r, c].as_slice()]).collect())
            .collect();
        let mut out_cols: Vec<Vec<f64>> = vec![];
        for v in &cols_vec.clone() {
            let mut u = v.clone();
            for w in &out_cols {
                let dot: f64 = u.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                for (ui, wi) in u.iter_mut().zip(w.iter()) {
                    *ui -= dot * wi;
                }
            }
            let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for ui in u.iter_mut() {
                    *ui /= norm;
                }
                out_cols.push(u);
            }
        }
        let _ = &cols_vec;
        let nc = out_cols.len();
        let mut flat = vec![0.0; rows * nc];
        for c in 0..nc {
            for r in 0..rows {
                flat[r * nc + c] = out_cols[c][r];
            }
        }
        let arr_out = ArrayD::from_shape_vec(IxDyn(&[rows, nc]), flat).context("gram_schmidt")?;
        Ok(json!({"matrix": array_to_value(&arr_out)}))
    })
}
