//! src/signal.rs — signal processing surface (`polars__sig_*`, `polars__win_*`).
//!
//! convolve / correlate / fft helpers / windows (hann, hamming, blackman,
//! kaiser, tukey, etc.) / detrend / smoothing / peak finding.
//! Wire format same as nd.rs: `{array: {data, shape}}`.

use std::f64::consts::PI;
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
    let a = args
        .get(key)
        .ok_or_else(|| anyhow!("missing argument `{key}`"))?;
    parse_array(a)
}

fn get_n(args: &Value) -> Result<usize> {
    args.get("n")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .ok_or_else(|| anyhow!("missing `n`"))
}

// ── window functions ────────────────────────────────────────────────────────

fn window_to_arr(name: &str, n: usize, w: Vec<f64>) -> Result<Value> {
    let _ = name;
    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), w).context("window")?;
    Ok(json!({"array": array_to_value(&arr)}))
}

/// Window function hann.
#[no_mangle]
pub extern "C" fn polars__win_hann(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let w: Vec<f64> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1).max(1) as f64).cos())
            .collect();
        window_to_arr("hann", n, w)
    })
}

/// Window function hamming.
#[no_mangle]
pub extern "C" fn polars__win_hamming(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let w: Vec<f64> = (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1).max(1) as f64).cos())
            .collect();
        window_to_arr("hamming", n, w)
    })
}

/// Window function blackman.
#[no_mangle]
pub extern "C" fn polars__win_blackman(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let m = (n - 1).max(1) as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / m;
                0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos()
            })
            .collect();
        window_to_arr("blackman", n, w)
    })
}

/// Window function bartlett.
#[no_mangle]
pub extern "C" fn polars__win_bartlett(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let m = (n - 1).max(1) as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / m;
                if x <= 0.5 {
                    2.0 * x
                } else {
                    2.0 - 2.0 * x
                }
            })
            .collect();
        window_to_arr("bartlett", n, w)
    })
}

/// Window function triangular.
#[no_mangle]
pub extern "C" fn polars__win_triangular(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let m = n as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| 1.0 - (2.0 * i as f64 / m - 1.0).abs())
            .collect();
        window_to_arr("triangular", n, w)
    })
}

/// Window function nuttall.
#[no_mangle]
pub extern "C" fn polars__win_nuttall(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let m = (n - 1).max(1) as f64;
        let (a0, a1, a2, a3) = (0.355768, 0.487396, 0.144232, 0.012604);
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / m;
                a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
                    - a3 * (6.0 * PI * x).cos()
            })
            .collect();
        window_to_arr("nuttall", n, w)
    })
}

/// Window function flattop.
#[no_mangle]
pub extern "C" fn polars__win_flattop(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let m = (n - 1).max(1) as f64;
        let (a0, a1, a2, a3, a4) = (
            0.21557895,
            0.41663158,
            0.277263158,
            0.083578947,
            0.006947368,
        );
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / m;
                a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
                    - a3 * (6.0 * PI * x).cos()
                    + a4 * (8.0 * PI * x).cos()
            })
            .collect();
        window_to_arr("flattop", n, w)
    })
}

/// Window function blackman harris.
#[no_mangle]
pub extern "C" fn polars__win_blackman_harris(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let m = (n - 1).max(1) as f64;
        let (a0, a1, a2, a3) = (0.35875, 0.48829, 0.14128, 0.01168);
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / m;
                a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
                    - a3 * (6.0 * PI * x).cos()
            })
            .collect();
        window_to_arr("blackman_harris", n, w)
    })
}

/// Window function rectangular.
#[no_mangle]
pub extern "C" fn polars__win_rectangular(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        window_to_arr("rectangular", n, vec![1.0; n])
    })
}

/// Window function kaiser.
#[no_mangle]
pub extern "C" fn polars__win_kaiser(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let beta = args.get("beta").and_then(|v| v.as_f64()).unwrap_or(8.6);
        // Bessel I_0 series approximation.
        let i0 = |x: f64| -> f64 {
            let mut sum = 1.0;
            let mut term = 1.0;
            for k in 1..30 {
                term *= (x / (2.0 * k as f64)).powi(2);
                sum += term;
            }
            sum
        };
        let denom = i0(beta);
        let m = (n - 1).max(1) as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let arg = 2.0 * i as f64 / m - 1.0;
                i0(beta * (1.0 - arg * arg).sqrt()) / denom
            })
            .collect();
        window_to_arr("kaiser", n, w)
    })
}

/// Window function tukey.
#[no_mangle]
pub extern "C" fn polars__win_tukey(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let m = (n - 1).max(1) as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / m;
                if x < alpha / 2.0 {
                    0.5 * (1.0 + (PI * (2.0 * x / alpha - 1.0)).cos())
                } else if x <= 1.0 - alpha / 2.0 {
                    1.0
                } else {
                    0.5 * (1.0 + (PI * (2.0 * x / alpha - 2.0 / alpha + 1.0)).cos())
                }
            })
            .collect();
        window_to_arr("tukey", n, w)
    })
}

/// Window function gaussian.
#[no_mangle]
pub extern "C" fn polars__win_gaussian(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mid = (n - 1) as f64 / 2.0;
        let w: Vec<f64> = (0..n)
            .map(|i| (-0.5 * ((i as f64 - mid) / sigma).powi(2)).exp())
            .collect();
        window_to_arr("gaussian", n, w)
    })
}

/// Window function exponential.
#[no_mangle]
pub extern "C" fn polars__win_exponential(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let tau = args
            .get("tau")
            .and_then(|v| v.as_f64())
            .unwrap_or((n - 1) as f64 / 2.0 / 8.69);
        let mid = (n - 1) as f64 / 2.0;
        let w: Vec<f64> = (0..n)
            .map(|i| (-(i as f64 - mid).abs() / tau).exp())
            .collect();
        window_to_arr("exponential", n, w)
    })
}

/// Window function cosine.
#[no_mangle]
pub extern "C" fn polars__win_cosine(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let w: Vec<f64> = (0..n)
            .map(|i| (PI * i as f64 / (n - 1).max(1) as f64).sin())
            .collect();
        window_to_arr("cosine", n, w)
    })
}

/// Window function lanczos.
#[no_mangle]
pub extern "C" fn polars__win_lanczos(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let m = (n - 1).max(1) as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / m - PI;
                if x == 0.0 {
                    1.0
                } else {
                    x.sin() / x
                }
            })
            .collect();
        window_to_arr("lanczos", n, w)
    })
}

/// Window function parzen.
#[no_mangle]
pub extern "C" fn polars__win_parzen(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let mid = n as f64 / 2.0;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = (i as f64 - mid + 0.5).abs() / (n as f64 / 2.0);
                if x <= 0.5 {
                    1.0 - 6.0 * x * x + 6.0 * x.powi(3)
                } else if x <= 1.0 {
                    2.0 * (1.0 - x).powi(3)
                } else {
                    0.0
                }
            })
            .collect();
        window_to_arr("parzen", n, w)
    })
}

/// Window function bohman.
#[no_mangle]
pub extern "C" fn polars__win_bohman(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let n = get_n(&args)?;
        let m = (n - 1).max(1) as f64;
        let w: Vec<f64> = (0..n)
            .map(|i| {
                let x = (2.0 * i as f64 / m - 1.0).abs();
                (1.0 - x) * (PI * x).cos() + (1.0 / PI) * (PI * x).sin()
            })
            .collect();
        window_to_arr("bohman", n, w)
    })
}

// ── convolution / correlation ──────────────────────────────────────────────

fn conv(a: &[f64], b: &[f64], mode: &str) -> Vec<f64> {
    let la = a.len();
    let lb = b.len();
    if la == 0 || lb == 0 {
        return vec![];
    }
    let nfull = la + lb - 1;
    let mut full = vec![0.0; nfull];
    for i in 0..la {
        for j in 0..lb {
            full[i + j] += a[i] * b[j];
        }
    }
    match mode {
        "same" => {
            let start = (lb - 1) / 2;
            full[start..start + la].to_vec()
        }
        "valid" => {
            if la >= lb {
                full[lb - 1..la].to_vec()
            } else {
                full[la - 1..lb].to_vec()
            }
        }
        _ => full,
    }
}

/// Signal convolve.
#[no_mangle]
pub extern "C" fn polars__sig_convolve(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("full")
            .to_string();
        let av: Vec<f64> = a.iter().copied().collect();
        let bv: Vec<f64> = b.iter().copied().collect();
        let out = conv(&av, &bv, &mode);
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("convolve")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal correlate.
#[no_mangle]
pub extern "C" fn polars__sig_correlate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "a")?;
        let b = get_array(&args, "b")?;
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("full")
            .to_string();
        let av: Vec<f64> = a.iter().copied().collect();
        let mut bv: Vec<f64> = b.iter().copied().collect();
        bv.reverse();
        let out = conv(&av, &bv, &mode);
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("correlate")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal autocorrelate.
#[no_mangle]
pub extern "C" fn polars__sig_autocorrelate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let mut out = vec![0.0; v.len()];
        for lag in 0..v.len() {
            for i in 0..(v.len() - lag) {
                out[lag] += v[i] * v[i + lag];
            }
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("autocorrelate")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal detrend.
#[no_mangle]
pub extern "C" fn polars__sig_detrend(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        if n < 2 {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        // Linear least-squares fit y = mx + b
        let nf = n as f64;
        let sx: f64 = (0..n).map(|i| i as f64).sum();
        let sy: f64 = v.iter().sum();
        let sxx: f64 = (0..n).map(|i| (i as f64).powi(2)).sum();
        let sxy: f64 = v.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let denom = nf * sxx - sx * sx;
        let m = if denom == 0.0 {
            0.0
        } else {
            (nf * sxy - sx * sy) / denom
        };
        let b = (sy - m * sx) / nf;
        let out: Vec<f64> = v
            .iter()
            .enumerate()
            .map(|(i, y)| y - (m * i as f64 + b))
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("detrend")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal detrend mean.
#[no_mangle]
pub extern "C" fn polars__sig_detrend_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let out: Vec<f64> = v.iter().map(|x| x - m).collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("detrend_mean")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal smooth box.
#[no_mangle]
pub extern "C" fn polars__sig_smooth_box(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let half = w / 2;
        let n = v.len();
        let out: Vec<f64> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(n);
                let slice = &v[lo..hi];
                slice.iter().sum::<f64>() / slice.len() as f64
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("smooth_box")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal smooth triangle.
#[no_mangle]
pub extern "C" fn polars__sig_smooth_triangle(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let half = w / 2;
        let n = v.len();
        let out: Vec<f64> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(n);
                let mut wsum = 0.0;
                let mut vsum = 0.0;
                for (offset, &x) in v[lo..hi].iter().enumerate() {
                    let k = lo + offset;
                    let weight = 1.0 - (k as f64 - i as f64).abs() / (half as f64 + 1.0);
                    wsum += weight;
                    vsum += weight * x;
                }
                vsum / wsum
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("smooth_triangle")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal smooth gaussian.
#[no_mangle]
pub extern "C" fn polars__sig_smooth_gaussian(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let v: Vec<f64> = a.iter().copied().collect();
        let radius = (3.0 * sigma).ceil() as usize;
        let kernel: Vec<f64> = (0..=2 * radius)
            .map(|i| {
                let x = i as f64 - radius as f64;
                (-0.5 * (x / sigma).powi(2)).exp()
            })
            .collect();
        let ksum: f64 = kernel.iter().sum();
        let n = v.len();
        let out: Vec<f64> = (0..n)
            .map(|i| {
                let mut acc = 0.0;
                for (k, &kv) in kernel.iter().enumerate() {
                    let idx = i as i64 + k as i64 - radius as i64;
                    if idx >= 0 && (idx as usize) < n {
                        acc += kv * v[idx as usize];
                    }
                }
                acc / ksum
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("smooth_gaussian")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal diff.
#[no_mangle]
pub extern "C" fn polars__sig_diff(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        if v.len() < 2 {
            let arr = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).context("diff")?;
            return Ok(json!({"array": array_to_value(&arr)}));
        }
        let out: Vec<f64> = (1..v.len()).map(|i| v[i] - v[i - 1]).collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("diff")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal gradient.
#[no_mangle]
pub extern "C" fn polars__sig_gradient(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        if n < 2 {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let mut out = vec![0.0; n];
        out[0] = v[1] - v[0];
        out[n - 1] = v[n - 1] - v[n - 2];
        for i in 1..n - 1 {
            out[i] = (v[i + 1] - v[i - 1]) / 2.0;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("gradient")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal integral.
#[no_mangle]
pub extern "C" fn polars__sig_integral(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let mut acc = 0.0;
        let out: Vec<f64> = v
            .iter()
            .map(|x| {
                acc += x;
                acc
            })
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("integral")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal trapz.
#[no_mangle]
pub extern "C" fn polars__sig_trapz(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        if v.len() < 2 {
            return Ok(json!({"trapz": 0.0}));
        }
        let dx = args.get("dx").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let s: f64 = v.windows(2).map(|w| 0.5 * (w[0] + w[1])).sum::<f64>() * dx;
        Ok(json!({"trapz": s}))
    })
}

/// Signal simpson.
#[no_mangle]
pub extern "C" fn polars__sig_simpson(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let dx = args.get("dx").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if v.len() < 3 {
            return Ok(json!({"simpson": 0.0}));
        }
        let n = v.len();
        let mut s = v[0] + v[n - 1];
        for (offset, &x) in v[1..n - 1].iter().enumerate() {
            let i = offset + 1;
            s += if i % 2 == 0 { 2.0 * x } else { 4.0 * x };
        }
        Ok(json!({"simpson": s * dx / 3.0}))
    })
}

/// Signal rms.
#[no_mangle]
pub extern "C" fn polars__sig_rms(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let r = (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        Ok(json!({"rms": r}))
    })
}

/// Signal peak to peak.
#[no_mangle]
pub extern "C" fn polars__sig_peak_to_peak(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Ok(json!({"peak_to_peak": mx - mn}))
    })
}

/// Signal zero crossings.
#[no_mangle]
pub extern "C" fn polars__sig_zero_crossings(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let r = v
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        Ok(json!({"zero_crossings": r}))
    })
}

/// Signal find peaks.
#[no_mangle]
pub extern "C" fn polars__sig_find_peaks(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let height = args
            .get("height")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let mut peaks = vec![];
        for i in 1..v.len().saturating_sub(1) {
            if v[i] > v[i - 1] && v[i] > v[i + 1] && v[i] >= height {
                peaks.push(i as i64);
            }
        }
        Ok(json!({"peaks": peaks}))
    })
}

/// Signal find valleys.
#[no_mangle]
pub extern "C" fn polars__sig_find_valleys(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let mut valleys = vec![];
        for i in 1..v.len().saturating_sub(1) {
            if v[i] < v[i - 1] && v[i] < v[i + 1] {
                valleys.push(i as i64);
            }
        }
        Ok(json!({"valleys": valleys}))
    })
}

/// Signal envelope upper.
#[no_mangle]
pub extern "C" fn polars__sig_envelope_upper(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let half = w / 2;
        let n = v.len();
        let out: Vec<f64> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(n);
                v[lo..hi].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("envelope_upper")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal envelope lower.
#[no_mangle]
pub extern "C" fn polars__sig_envelope_lower(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let half = w / 2;
        let n = v.len();
        let out: Vec<f64> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(n);
                v[lo..hi].iter().cloned().fold(f64::INFINITY, f64::min)
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("envelope_lower")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal median filter.
#[no_mangle]
pub extern "C" fn polars__sig_median_filter(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let half = w / 2;
        let n = v.len();
        let out: Vec<f64> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(n);
                let mut s: Vec<f64> = v[lo..hi].to_vec();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                s[s.len() / 2]
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("median_filter")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal ema.
#[no_mangle]
pub extern "C" fn polars__sig_ema(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let v: Vec<f64> = a.iter().copied().collect();
        let mut acc = v.first().copied().unwrap_or(0.0);
        let out: Vec<f64> = v
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if i == 0 {
                    acc = *x;
                } else {
                    acc = alpha * x + (1.0 - alpha) * acc;
                }
                acc
            })
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("ema")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal dema.
#[no_mangle]
pub extern "C" fn polars__sig_dema(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let v: Vec<f64> = a.iter().copied().collect();
        let mut ema1 = v.first().copied().unwrap_or(0.0);
        let mut ema2 = ema1;
        let out: Vec<f64> = v
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if i == 0 {
                    ema1 = *x;
                    ema2 = *x;
                } else {
                    ema1 = alpha * x + (1.0 - alpha) * ema1;
                    ema2 = alpha * ema1 + (1.0 - alpha) * ema2;
                }
                2.0 * ema1 - ema2
            })
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("dema")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal sma.
#[no_mangle]
pub extern "C" fn polars__sig_sma(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let mut out = vec![f64::NAN; n];
        if w > n {
            let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("sma")?;
            return Ok(json!({"array": array_to_value(&arr)}));
        }
        for i in (w - 1)..n {
            out[i] = v[i + 1 - w..=i].iter().sum::<f64>() / w as f64;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("sma")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal wma.
#[no_mangle]
pub extern "C" fn polars__sig_wma(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let weights: Vec<f64> = (1..=w).map(|i| i as f64).collect();
        let wsum: f64 = weights.iter().sum();
        let mut out = vec![f64::NAN; n];
        if w > n {
            let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("wma")?;
            return Ok(json!({"array": array_to_value(&arr)}));
        }
        for i in (w - 1)..n {
            let slice = &v[i + 1 - w..=i];
            out[i] = slice
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f64>()
                / wsum;
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("wma")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal macd.
#[no_mangle]
pub extern "C" fn polars__sig_macd(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let fast = args.get("fast").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
        let slow = args.get("slow").and_then(|v| v.as_u64()).unwrap_or(26) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let ema = |vs: &[f64], w: usize| -> Vec<f64> {
            let alpha = 2.0 / (w as f64 + 1.0);
            let mut e = vs[0];
            vs.iter()
                .map(|x| {
                    e = alpha * x + (1.0 - alpha) * e;
                    e
                })
                .collect()
        };
        let ef = ema(&v, fast);
        let es = ema(&v, slow);
        let out: Vec<f64> = ef.iter().zip(es.iter()).map(|(f, s)| f - s).collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("macd")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal rsi.
#[no_mangle]
pub extern "C" fn polars__sig_rsi(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(14) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        if v.len() < w + 1 {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let mut gains = vec![];
        let mut losses = vec![];
        for i in 1..v.len() {
            let d = v[i] - v[i - 1];
            gains.push(d.max(0.0));
            losses.push((-d).max(0.0));
        }
        let mut out = vec![f64::NAN; v.len()];
        for i in w..v.len() {
            let avg_gain: f64 = gains[i - w..i].iter().sum::<f64>() / w as f64;
            let avg_loss: f64 = losses[i - w..i].iter().sum::<f64>() / w as f64;
            let rs = if avg_loss == 0.0 {
                f64::INFINITY
            } else {
                avg_gain / avg_loss
            };
            out[i] = 100.0 - 100.0 / (1.0 + rs);
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("rsi")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal bollinger upper.
#[no_mangle]
pub extern "C" fn polars__sig_bollinger_upper(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let k = args.get("k").and_then(|v| v.as_f64()).unwrap_or(2.0);
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let mut out = vec![f64::NAN; n];
        if w > n {
            let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("bollinger")?;
            return Ok(json!({"array": array_to_value(&arr)}));
        }
        for i in (w - 1)..n {
            let slice = &v[i + 1 - w..=i];
            let m = slice.iter().sum::<f64>() / w as f64;
            let var = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / w as f64;
            out[i] = m + k * var.sqrt();
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("bollinger_upper")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal bollinger lower.
#[no_mangle]
pub extern "C" fn polars__sig_bollinger_lower(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let k = args.get("k").and_then(|v| v.as_f64()).unwrap_or(2.0);
        let v: Vec<f64> = a.iter().copied().collect();
        let n = v.len();
        let mut out = vec![f64::NAN; n];
        if w > n {
            let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("bollinger")?;
            return Ok(json!({"array": array_to_value(&arr)}));
        }
        for i in (w - 1)..n {
            let slice = &v[i + 1 - w..=i];
            let m = slice.iter().sum::<f64>() / w as f64;
            let var = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / w as f64;
            out[i] = m - k * var.sqrt();
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("bollinger_lower")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal chunk mean.
#[no_mangle]
pub extern "C" fn polars__sig_chunk_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let chunk = args.get("chunk").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        if chunk == 0 {
            bail!("chunk must be > 0");
        }
        let out: Vec<f64> = v
            .chunks(chunk)
            .map(|c| c.iter().sum::<f64>() / c.len() as f64)
            .collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("chunk_mean")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal resample.
#[no_mangle]
pub extern "C" fn polars__sig_resample(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let n_out = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let n_in = v.len();
        if n_in == 0 {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let out: Vec<f64> = (0..n_out)
            .map(|i| {
                let x = i as f64 * (n_in - 1) as f64 / (n_out - 1).max(1) as f64;
                let lo = x.floor() as usize;
                let hi = (lo + 1).min(n_in - 1);
                let frac = x - lo as f64;
                v[lo] * (1.0 - frac) + v[hi] * frac
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n_out]), out).context("resample")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal decimate.
#[no_mangle]
pub extern "C" fn polars__sig_decimate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let q = args
            .get("q")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `q`"))? as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let out: Vec<f64> = v.iter().step_by(q).copied().collect();
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("decimate")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal upsample.
#[no_mangle]
pub extern "C" fn polars__sig_upsample(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let factor = args.get("factor").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let v: Vec<f64> = a.iter().copied().collect();
        let mut out: Vec<f64> = Vec::with_capacity(v.len() * factor);
        let zeros = vec![0.0; factor.saturating_sub(1)];
        for x in &v {
            out.push(*x);
            out.extend_from_slice(&zeros);
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("upsample")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal normalize.
#[no_mangle]
pub extern "C" fn polars__sig_normalize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        let mx = v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let out: Vec<f64> = if mx == 0.0 {
            v
        } else {
            v.iter().map(|x| x / mx).collect()
        };
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("normalize")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

/// Signal unwrap.
#[no_mangle]
pub extern "C" fn polars__sig_unwrap(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_array(&args, "array")?;
        let v: Vec<f64> = a.iter().copied().collect();
        if v.is_empty() {
            return Ok(json!({"array": array_to_value(&a)}));
        }
        let mut out = vec![v[0]];
        for i in 1..v.len() {
            let d = v[i] - out[i - 1];
            let dadj = d - (d / (2.0 * PI)).round() * 2.0 * PI;
            out.push(out[i - 1] + dadj);
        }
        let n = out.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out).context("unwrap")?;
        Ok(json!({"array": array_to_value(&arr)}))
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::ffi_test::call;

    use super::*;

    fn data(v: serde_json::Value) -> Vec<f64> {
        v["array"]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    #[test]
    fn win_hann_symmetric_and_zero_at_endpoints() {
        // Hann window: 0 at i=0 and i=n-1, peak in the middle.
        let v = call(polars__win_hann, json!({"n": 9}));
        let w = data(v);
        assert!(w[0].abs() < 1e-12, "left endpoint zero");
        assert!(w[8].abs() < 1e-12, "right endpoint zero");
        for i in 0..4 {
            assert!((w[i] - w[8 - i]).abs() < 1e-12, "symmetric at {i}");
        }
    }

    #[test]
    fn win_hamming_endpoints_match_alpha() {
        // Hamming endpoints: w[0] = w[n-1] = a0 - a1 = 0.54 - 0.46 = 0.08.
        let v = call(polars__win_hamming, json!({"n": 11}));
        let w = data(v);
        assert!((w[0] - 0.08).abs() < 1e-12);
        assert!((w[10] - 0.08).abs() < 1e-12);
    }

    #[test]
    fn win_rectangular_is_all_ones() {
        let v = call(polars__win_rectangular, json!({"n": 7}));
        for x in data(v) {
            assert_eq!(x, 1.0);
        }
    }

    #[test]
    fn win_bartlett_peaks_at_middle() {
        // Triangular: max at center = 1.
        let v = call(polars__win_bartlett, json!({"n": 9}));
        let w = data(v);
        assert_eq!(w[4], 1.0, "center is 1");
        assert!(w[0].abs() < 1e-12, "endpoint 0");
    }

    #[test]
    fn sig_convolve_full_size_matches_formula() {
        // For arrays of length M, N, full convolution has length M + N - 1.
        let v = call(
            polars__sig_convolve,
            json!({
                "a": {"data": [1.0, 2.0, 3.0], "shape": [3]},
                "b": {"data": [1.0, 1.0], "shape": [2]},
                "mode": "full",
            }),
        );
        let out = data(v);
        assert_eq!(out.len(), 4, "M + N - 1 = 4");
        // [1*1, 1*1+2*1, 2*1+3*1, 3*1] = [1, 3, 5, 3].
        assert_eq!(out, vec![1.0, 3.0, 5.0, 3.0]);
    }

    #[test]
    fn sig_convolve_identity_kernel() {
        // Convolving with [1] returns the input unchanged.
        let v = call(
            polars__sig_convolve,
            json!({
                "a": {"data": [1.0, 2.0, 3.0, 4.0], "shape": [4]},
                "b": {"data": [1.0], "shape": [1]},
                "mode": "same",
            }),
        );
        assert_eq!(data(v), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sig_trapz_matches_known_integral() {
        // ∫_0^1 x dx = 0.5; sampled at [0, 1] with dx=1 gives trapz = 0.5.
        let v = call(
            polars__sig_trapz,
            json!({"array": {"data": [0.0, 1.0], "shape": [2]}, "dx": 1.0}),
        );
        assert!((v["trapz"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        // [0, 1, 2, 3] with dx=1: triangle area = ((0+1) + (1+2) + (2+3))/2 = 4.5.
        let v = call(
            polars__sig_trapz,
            json!({"array": {"data": [0.0, 1.0, 2.0, 3.0], "shape": [4]}, "dx": 1.0}),
        );
        assert!((v["trapz"].as_f64().unwrap() - 4.5).abs() < 1e-12);
    }

    #[test]
    fn sig_rms_matches_formula() {
        // RMS([3, 4]) = sqrt((9 + 16) / 2) = sqrt(12.5).
        let v = call(
            polars__sig_rms,
            json!({"array": {"data": [3.0, 4.0], "shape": [2]}}),
        );
        let rms = v["rms"].as_f64().unwrap();
        assert!((rms - 12.5_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn sig_zero_crossings_counts_sign_flips() {
        // [1, -1, 1, -1] has 3 zero crossings.
        let v = call(
            polars__sig_zero_crossings,
            json!({"array": {"data": [1.0, -1.0, 1.0, -1.0], "shape": [4]}}),
        );
        assert_eq!(v["zero_crossings"].as_u64().unwrap(), 3);
    }

    #[test]
    fn sig_find_peaks_classic_triangle() {
        // Local max at index 2 (value 3).
        let v = call(
            polars__sig_find_peaks,
            json!({"array": {"data": [1.0, 2.0, 3.0, 2.0, 1.0], "shape": [5]}}),
        );
        let p: Vec<i64> = v["peaks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(p, vec![2]);
    }
}
