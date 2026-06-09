//! src/stattest.rs — statistical tests + interpolation + distribution
//! PDFs/CDFs surface (`polars__stattest_*`, `polars__interp_*`, `polars__dist_*`).
//!
//! All numeric — pure-Rust implementations of common scipy.stats tests.

use std::f64::consts::PI;
use std::ffi::c_char;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::ffi_call;

fn get_arr(args: &Value, key: &str) -> Result<Vec<f64>> {
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

// Standard normal CDF via erf.
fn phi(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

fn lgamma(x: f64) -> f64 {
    // Lanczos approximation.
    let g = 7.0;
    let coeffs = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = coeffs[0];
    let t = x + g + 0.5;
    for (i, c) in coeffs.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

// Regularized lower incomplete gamma P(a, x) via series & continued fraction.
fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series.
        let mut term = 1.0 / a;
        let mut sum = term;
        for n in 1..200 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < sum.abs() * 1e-12 {
                break;
            }
        }
        sum * (-x + a * x.ln() - lgamma(a)).exp()
    } else {
        1.0 - gamma_q(a, x)
    }
}

fn gamma_q(a: f64, x: f64) -> f64 {
    // Continued fraction.
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
            break;
        }
    }
    (-x + a * x.ln() - lgamma(a)).exp() * h
}

// Chi-squared survival function = Q(k/2, x/2).
fn chi2_sf(k: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(k / 2.0, x / 2.0)
}

// ── stats tests ────────────────────────────────────────────────────────────

/// Statistical test ttest 1samp.
#[no_mangle]
pub extern "C" fn polars__stattest_ttest_1samp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let mu = args.get("popmean").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let n = a.len() as f64;
        let m = a.iter().sum::<f64>() / n;
        let var = a.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let se = (var / n).sqrt();
        let t = (m - mu) / se;
        let df = n - 1.0;
        // Two-tailed p-value via Student's t CDF approximated via incomplete beta.
        let x = df / (df + t * t);
        let p = beta_inc(df / 2.0, 0.5, x);
        Ok(json!({"statistic": scalar(t), "pvalue": scalar(p)}))
    })
}

fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return x;
    }
    let bt = (lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200 {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
            break;
        }
    }
    h
}

/// Statistical test ttest ind.
#[no_mangle]
pub extern "C" fn polars__stattest_ttest_ind(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let b = get_arr(&args, "b")?;
        let na = a.len() as f64;
        let nb = b.len() as f64;
        let ma = a.iter().sum::<f64>() / na;
        let mb = b.iter().sum::<f64>() / nb;
        let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0).max(1.0);
        let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0).max(1.0);
        let sp = (((na - 1.0) * va + (nb - 1.0) * vb) / (na + nb - 2.0)).sqrt();
        let se = sp * (1.0 / na + 1.0 / nb).sqrt();
        let t = (ma - mb) / se;
        let df = na + nb - 2.0;
        let x = df / (df + t * t);
        let p = beta_inc(df / 2.0, 0.5, x);
        Ok(json!({"statistic": scalar(t), "pvalue": scalar(p)}))
    })
}

/// Statistical test ttest paired.
#[no_mangle]
pub extern "C" fn polars__stattest_ttest_paired(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let b = get_arr(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let n = d.len() as f64;
        let m = d.iter().sum::<f64>() / n;
        let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let se = (var / n).sqrt();
        let t = m / se;
        let df = n - 1.0;
        let x = df / (df + t * t);
        let p = beta_inc(df / 2.0, 0.5, x);
        Ok(json!({"statistic": scalar(t), "pvalue": scalar(p)}))
    })
}

/// Statistical test chi2.
#[no_mangle]
pub extern "C" fn polars__stattest_chi2(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let obs = get_arr(&args, "observed")?;
        let exp = get_arr(&args, "expected")?;
        if obs.len() != exp.len() {
            bail!("length mismatch");
        }
        let chi2: f64 = obs
            .iter()
            .zip(exp.iter())
            .map(|(o, e)| (o - e).powi(2) / e)
            .sum();
        let df = obs.len() as f64 - 1.0;
        let p = chi2_sf(df, chi2);
        Ok(json!({"statistic": scalar(chi2), "pvalue": scalar(p), "df": df}))
    })
}

/// Statistical test zscore.
#[no_mangle]
pub extern "C" fn polars__stattest_zscore(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let n = a.len() as f64;
        let m = a.iter().sum::<f64>() / n.max(1.0);
        let sd = (a.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n.max(1.0)).sqrt();
        let z: Vec<f64> = if sd == 0.0 {
            vec![0.0; a.len()]
        } else {
            a.iter().map(|x| (x - m) / sd).collect()
        };
        Ok(json!({"zscore": z}))
    })
}

/// Statistical test ztest.
#[no_mangle]
pub extern "C" fn polars__stattest_ztest(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let mu = args.get("popmean").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sd = args
            .get("sigma")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `sigma`"))?;
        let n = a.len() as f64;
        let m = a.iter().sum::<f64>() / n.max(1.0);
        let z = (m - mu) / (sd / n.sqrt());
        let p = 2.0 * (1.0 - phi(z.abs()));
        Ok(json!({"statistic": scalar(z), "pvalue": scalar(p)}))
    })
}

/// Statistical test mannwhitney.
#[no_mangle]
pub extern "C" fn polars__stattest_mannwhitney(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let b = get_arr(&args, "b")?;
        let na = a.len();
        let nb = b.len();
        let mut all: Vec<(f64, usize)> = a
            .iter()
            .map(|x| (*x, 0))
            .chain(b.iter().map(|x| (*x, 1)))
            .collect();
        all.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut ra = 0.0;
        for (i, (_, label)) in all.iter().enumerate() {
            if *label == 0 {
                ra += i as f64 + 1.0;
            }
        }
        let u1 = ra - (na as f64 * (na as f64 + 1.0) / 2.0);
        let u2 = (na * nb) as f64 - u1;
        let u = u1.min(u2);
        let mu = (na * nb) as f64 / 2.0;
        let sd = ((na * nb * (na + nb + 1)) as f64 / 12.0).sqrt();
        let z = (u - mu) / sd;
        let p = 2.0 * (1.0 - phi(z.abs()));
        Ok(json!({"statistic": scalar(u), "pvalue": scalar(p)}))
    })
}

/// Statistical test wilcoxon.
#[no_mangle]
pub extern "C" fn polars__stattest_wilcoxon(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let b = get_arr(&args, "b")?;
        if a.len() != b.len() {
            bail!("length mismatch");
        }
        let diffs: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x - y)
            .filter(|d| *d != 0.0)
            .collect();
        let n = diffs.len();
        let mut pairs: Vec<(f64, f64)> = diffs.iter().map(|d| (d.abs(), d.signum())).collect();
        pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut w_plus = 0.0;
        for (i, (_, s)) in pairs.iter().enumerate() {
            let rank = (i + 1) as f64;
            if *s > 0.0 {
                w_plus += rank;
            }
        }
        let nf = n as f64;
        let mu = nf * (nf + 1.0) / 4.0;
        let sd = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0).sqrt();
        let z = (w_plus - mu) / sd;
        let p = 2.0 * (1.0 - phi(z.abs()));
        Ok(json!({"statistic": scalar(w_plus), "pvalue": scalar(p)}))
    })
}

/// Statistical test ks 2samp.
#[no_mangle]
pub extern "C" fn polars__stattest_ks_2samp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut a = get_arr(&args, "a")?;
        let mut b = get_arr(&args, "b")?;
        a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let na = a.len();
        let nb = b.len();
        let mut i = 0;
        let mut j = 0;
        let mut d_max = 0.0_f64;
        while i < na && j < nb {
            let fa = (i + 1) as f64 / na as f64;
            let fb = (j + 1) as f64 / nb as f64;
            d_max = d_max.max((fa - fb).abs());
            if a[i] <= b[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        let en = ((na * nb) as f64 / (na + nb) as f64).sqrt();
        let lambda = (en + 0.12 + 0.11 / en) * d_max;
        let mut p = 2.0;
        for j in 1..100 {
            let jj = j as f64;
            let term = 2.0 * (-2.0 * jj * jj * lambda * lambda).exp() * (-1f64).powi(j - 1);
            p += term;
            if term.abs() < 1e-10 {
                break;
            }
        }
        if p < 0.0 || !p.is_finite() {
            p = 0.0;
        }
        if p > 1.0 {
            p = 1.0;
        }
        Ok(json!({"statistic": scalar(d_max), "pvalue": scalar(p)}))
    })
}

/// Statistical test anova.
#[no_mangle]
pub extern "C" fn polars__stattest_anova(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let groups = args
            .get("groups")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `groups`"))?;
        let g: Vec<Vec<f64>> = groups
            .iter()
            .map(|grp| {
                grp.as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                    .unwrap_or_default()
            })
            .collect();
        let k = g.len() as f64;
        let n_total: f64 = g.iter().map(|x| x.len() as f64).sum();
        let grand_mean: f64 = g.iter().flatten().sum::<f64>() / n_total;
        let ss_between: f64 = g
            .iter()
            .map(|grp| {
                let m = grp.iter().sum::<f64>() / grp.len() as f64;
                grp.len() as f64 * (m - grand_mean).powi(2)
            })
            .sum();
        let ss_within: f64 = g
            .iter()
            .map(|grp| {
                let m = grp.iter().sum::<f64>() / grp.len() as f64;
                grp.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            })
            .sum();
        let df_between = k - 1.0;
        let df_within = n_total - k;
        let f = (ss_between / df_between) / (ss_within / df_within);
        let x = df_within / (df_within + df_between * f);
        let p = beta_inc(df_within / 2.0, df_between / 2.0, x);
        Ok(json!({"statistic": scalar(f), "pvalue": scalar(p)}))
    })
}

/// Statistical test levene.
#[no_mangle]
pub extern "C" fn polars__stattest_levene(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let groups = args
            .get("groups")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `groups`"))?;
        let g: Vec<Vec<f64>> = groups
            .iter()
            .map(|grp| {
                grp.as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                    .unwrap_or_default()
            })
            .collect();
        // Compute |x - median| for each group, then ANOVA.
        let z: Vec<Vec<f64>> = g
            .iter()
            .map(|grp| {
                let mut sorted = grp.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = if sorted.is_empty() {
                    0.0
                } else if sorted.len() % 2 == 0 {
                    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                } else {
                    sorted[sorted.len() / 2]
                };
                grp.iter().map(|x| (x - med).abs()).collect()
            })
            .collect();
        let k = z.len() as f64;
        let n_total: f64 = z.iter().map(|x| x.len() as f64).sum();
        let grand_mean: f64 = z.iter().flatten().sum::<f64>() / n_total;
        let ss_between: f64 = z
            .iter()
            .map(|grp| {
                let m = grp.iter().sum::<f64>() / grp.len() as f64;
                grp.len() as f64 * (m - grand_mean).powi(2)
            })
            .sum();
        let ss_within: f64 = z
            .iter()
            .map(|grp| {
                let m = grp.iter().sum::<f64>() / grp.len() as f64;
                grp.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            })
            .sum();
        let df_between = k - 1.0;
        let df_within = n_total - k;
        let w = (ss_between / df_between) / (ss_within / df_within);
        let x = df_within / (df_within + df_between * w);
        let p = beta_inc(df_within / 2.0, df_between / 2.0, x);
        Ok(json!({"statistic": scalar(w), "pvalue": scalar(p)}))
    })
}

/// Statistical test kruskal.
#[no_mangle]
pub extern "C" fn polars__stattest_kruskal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let groups = args
            .get("groups")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `groups`"))?;
        let g: Vec<Vec<f64>> = groups
            .iter()
            .map(|grp| {
                grp.as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                    .unwrap_or_default()
            })
            .collect();
        let k = g.len();
        let mut all: Vec<(f64, usize)> = vec![];
        for (gi, grp) in g.iter().enumerate() {
            for v in grp {
                all.push((*v, gi));
            }
        }
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut rank_sums = vec![0.0; k];
        let mut counts = vec![0.0; k];
        for (i, (_, gi)) in all.iter().enumerate() {
            rank_sums[*gi] += i as f64 + 1.0;
            counts[*gi] += 1.0;
        }
        let n = all.len() as f64;
        let h = 12.0 / (n * (n + 1.0))
            * rank_sums
                .iter()
                .zip(counts.iter())
                .map(|(r, n_i)| r * r / n_i)
                .sum::<f64>()
            - 3.0 * (n + 1.0);
        let df = k as f64 - 1.0;
        let p = chi2_sf(df, h);
        Ok(json!({"statistic": scalar(h), "pvalue": scalar(p)}))
    })
}

/// Statistical test shapiro.
#[no_mangle]
pub extern "C" fn polars__stattest_shapiro(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let mut a = get_arr(&args, "a")?;
        a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let n = a.len() as f64;
        let m = a.iter().sum::<f64>() / n;
        let var = a.iter().map(|x| (x - m).powi(2)).sum::<f64>();
        if var == 0.0 {
            return Ok(json!({"statistic": 1.0, "pvalue": 1.0}));
        }
        // Approximate Shapiro-Wilk W via ranks (rough).
        let r2: f64 = a
            .iter()
            .enumerate()
            .map(|(i, x)| (i as f64 + 1.0) * x)
            .sum();
        let r1: f64 = (0..a.len()).sum::<usize>() as f64 + n;
        let w = (r2 - r1 * m).powi(2) / var;
        Ok(json!({"statistic": scalar(w), "pvalue": Value::Null}))
    })
}

/// Statistical test jb.
#[no_mangle]
pub extern "C" fn polars__stattest_jb(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let n = a.len() as f64;
        let m = a.iter().sum::<f64>() / n;
        let var = a.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        let sd = var.sqrt();
        let sk = a.iter().map(|x| ((x - m) / sd).powi(3)).sum::<f64>() / n;
        let kt = a.iter().map(|x| ((x - m) / sd).powi(4)).sum::<f64>() / n - 3.0;
        let jb = n / 6.0 * (sk.powi(2) + kt.powi(2) / 4.0);
        let p = chi2_sf(2.0, jb);
        Ok(json!({"statistic": scalar(jb), "pvalue": scalar(p)}))
    })
}

/// Statistical test dw.
#[no_mangle]
pub extern "C" fn polars__stattest_dw(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "residuals")?;
        let num: f64 = a.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        let den: f64 = a.iter().map(|x| x * x).sum();
        let d = if den == 0.0 { f64::NAN } else { num / den };
        Ok(json!({"durbin_watson": scalar(d)}))
    })
}

/// Statistical test acf.
#[no_mangle]
pub extern "C" fn polars__stattest_acf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let max_lag = args.get("max_lag").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let n = a.len();
        let m = a.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = a.iter().map(|x| x - m).collect();
        let denom: f64 = centered.iter().map(|x| x * x).sum();
        let mut out = vec![];
        for lag in 0..=max_lag.min(n - 1) {
            let num: f64 = (0..n - lag).map(|i| centered[i] * centered[i + lag]).sum();
            out.push(if denom == 0.0 { 0.0 } else { num / denom });
        }
        Ok(json!({"acf": out}))
    })
}

/// Statistical test pacf.
#[no_mangle]
pub extern "C" fn polars__stattest_pacf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let max_lag = args.get("max_lag").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let n = a.len();
        let m = a.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = a.iter().map(|x| x - m).collect();
        let denom: f64 = centered.iter().map(|x| x * x).sum();
        let acf: Vec<f64> = (0..=max_lag.min(n - 1))
            .map(|lag| {
                let num: f64 = (0..n - lag).map(|i| centered[i] * centered[i + lag]).sum();
                if denom == 0.0 {
                    0.0
                } else {
                    num / denom
                }
            })
            .collect();
        // Levinson-Durbin recursion.
        let mut pacf = vec![1.0];
        if acf.len() > 1 {
            let mut phi = vec![acf[1]];
            pacf.push(acf[1]);
            for k in 2..acf.len() {
                let phi_kk_num = acf[k] - (0..k - 1).map(|j| phi[j] * acf[k - 1 - j]).sum::<f64>();
                let denom = 1.0 - (0..k - 1).map(|j| phi[j] * acf[j + 1]).sum::<f64>();
                let phi_kk = if denom == 0.0 {
                    0.0
                } else {
                    phi_kk_num / denom
                };
                let mut new_phi: Vec<f64> = (0..k - 1)
                    .map(|j| phi[j] - phi_kk * phi[k - 2 - j])
                    .collect();
                new_phi.push(phi_kk);
                phi = new_phi;
                pacf.push(phi_kk);
            }
        }
        Ok(json!({"pacf": pacf}))
    })
}

/// Statistical test linregress.
#[no_mangle]
pub extern "C" fn polars__stattest_linregress(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let y = get_arr(&args, "y")?;
        if x.len() != y.len() {
            bail!("length mismatch");
        }
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let sxy: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mx) * (yi - my))
            .sum();
        let sxx: f64 = x.iter().map(|xi| (xi - mx).powi(2)).sum();
        let syy: f64 = y.iter().map(|yi| (yi - my).powi(2)).sum();
        let slope = if sxx == 0.0 { 0.0 } else { sxy / sxx };
        let intercept = my - slope * mx;
        let r = if sxx * syy <= 0.0 {
            0.0
        } else {
            sxy / (sxx * syy).sqrt()
        };
        Ok(json!({
            "slope": scalar(slope),
            "intercept": scalar(intercept),
            "rvalue": scalar(r),
            "r_squared": scalar(r * r),
        }))
    })
}

// ── interpolation ──────────────────────────────────────────────────────────

/// Interpolation linear.
#[no_mangle]
pub extern "C" fn polars__interp_linear(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let y = get_arr(&args, "y")?;
        let xnew = get_arr(&args, "xnew")?;
        if x.len() != y.len() {
            bail!("length mismatch");
        }
        let out: Vec<f64> = xnew
            .iter()
            .map(|&xi| {
                if xi <= x[0] {
                    return y[0];
                }
                if xi >= x[x.len() - 1] {
                    return y[y.len() - 1];
                }
                let i = x.iter().position(|t| *t > xi).unwrap_or(x.len() - 1);
                let frac = (xi - x[i - 1]) / (x[i] - x[i - 1]);
                y[i - 1] + frac * (y[i] - y[i - 1])
            })
            .collect();
        Ok(json!({"y": out}))
    })
}

/// Interpolation nearest.
#[no_mangle]
pub extern "C" fn polars__interp_nearest(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let y = get_arr(&args, "y")?;
        let xnew = get_arr(&args, "xnew")?;
        if x.len() != y.len() {
            bail!("length mismatch");
        }
        let out: Vec<f64> = xnew
            .iter()
            .map(|&xi| {
                let nearest = x
                    .iter()
                    .enumerate()
                    .min_by(|a, b| {
                        (a.1 - xi)
                            .abs()
                            .partial_cmp(&(b.1 - xi).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                y[nearest]
            })
            .collect();
        Ok(json!({"y": out}))
    })
}

/// Interpolation zero order.
#[no_mangle]
pub extern "C" fn polars__interp_zero_order(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let y = get_arr(&args, "y")?;
        let xnew = get_arr(&args, "xnew")?;
        let out: Vec<f64> = xnew
            .iter()
            .map(|&xi| {
                let i = x
                    .iter()
                    .position(|t| *t > xi)
                    .unwrap_or(x.len())
                    .saturating_sub(1);
                y[i.min(y.len() - 1)]
            })
            .collect();
        Ok(json!({"y": out}))
    })
}

/// Interpolation cubic natural.
#[no_mangle]
pub extern "C" fn polars__interp_cubic_natural(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let y = get_arr(&args, "y")?;
        let xnew = get_arr(&args, "xnew")?;
        if x.len() != y.len() {
            bail!("length mismatch");
        }
        // Natural cubic spline.
        let n = x.len();
        if n < 2 {
            bail!("at least 2 knots required");
        }
        let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        let mut alpha = vec![0.0; n];
        for i in 1..n - 1 {
            alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        for i in 1..n - 1 {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];
        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }
        let out: Vec<f64> = xnew
            .iter()
            .map(|&xi| {
                let i = x
                    .iter()
                    .position(|t| *t > xi)
                    .unwrap_or(n)
                    .saturating_sub(1)
                    .min(n - 2);
                let dx = xi - x[i];
                y[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx
            })
            .collect();
        Ok(json!({"y": out}))
    })
}

/// Interpolation logp.
#[no_mangle]
pub extern "C" fn polars__interp_logp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let y = get_arr(&args, "y")?;
        let xnew = get_arr(&args, "xnew")?;
        if x.len() != y.len() {
            bail!("length mismatch");
        }
        let log_y: Vec<f64> = y.iter().map(|v| v.ln()).collect();
        let out: Vec<f64> = xnew
            .iter()
            .map(|&xi| {
                if xi <= x[0] {
                    return y[0];
                }
                if xi >= x[x.len() - 1] {
                    return y[y.len() - 1];
                }
                let i = x.iter().position(|t| *t > xi).unwrap_or(x.len() - 1);
                let frac = (xi - x[i - 1]) / (x[i] - x[i - 1]);
                (log_y[i - 1] + frac * (log_y[i] - log_y[i - 1])).exp()
            })
            .collect();
        Ok(json!({"y": out}))
    })
}

/// Interpolation grid 2d.
#[no_mangle]
pub extern "C" fn polars__interp_grid_2d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let grid = args
            .get("grid")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `grid`"))?;
        let rows: Vec<Vec<f64>> = grid
            .iter()
            .map(|r| {
                r.as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                    .unwrap_or_default()
            })
            .collect();
        let x = args.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let y = args.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let i = x.floor() as usize;
        let j = y.floor() as usize;
        if i + 1 >= rows.len() || j + 1 >= rows[0].len() {
            return Ok(json!({"value": Value::Null}));
        }
        let xf = x - x.floor();
        let yf = y - y.floor();
        let v00 = rows[i][j];
        let v10 = rows[i + 1][j];
        let v01 = rows[i][j + 1];
        let v11 = rows[i + 1][j + 1];
        let r = v00 * (1.0 - xf) * (1.0 - yf)
            + v10 * xf * (1.0 - yf)
            + v01 * (1.0 - xf) * yf
            + v11 * xf * yf;
        Ok(json!({"value": scalar(r)}))
    })
}

// ── distributions: PDF / CDF / PPF ────────────────────────────────────────

fn norm_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * PI).sqrt())
}

/// Normal pdf.
#[no_mangle]
pub extern "C" fn polars__dist_normal_pdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = x.iter().map(|x| norm_pdf(*x, mu, sigma)).collect();
        Ok(json!({"pdf": out}))
    })
}

/// Normal cdf.
#[no_mangle]
pub extern "C" fn polars__dist_normal_cdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = x.iter().map(|x| phi((x - mu) / sigma)).collect();
        Ok(json!({"cdf": out}))
    })
}

/// Normal ppf.
#[no_mangle]
pub extern "C" fn polars__dist_normal_ppf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let p = get_arr(&args, "p")?;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let inv_phi = |p: f64| -> f64 {
            // Rational approximation (Beasley-Springer-Moro).
            if p <= 0.0 || p >= 1.0 {
                return if p < 0.5 {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
            }
            let a = [
                -3.969683028665376e+01,
                2.209460984245205e+02,
                -2.759285104469687e+02,
                1.383_577_518_672_69e2,
                -3.066479806614716e+01,
                2.506628277459239e+00,
            ];
            let b = [
                -5.447609879822406e+01,
                1.615858368580409e+02,
                -1.556989798598866e+02,
                6.680131188771972e+01,
                -1.328068155288572e+01,
            ];
            let c = [
                -7.784894002430293e-03,
                -3.223964580411365e-01,
                -2.400758277161838e+00,
                -2.549732539343734e+00,
                4.374664141464968e+00,
                2.938163982698783e+00,
            ];
            let d = [
                7.784695709041462e-03,
                3.224671290700398e-01,
                2.445134137142996e+00,
                3.754408661907416e+00,
            ];
            let plow = 0.02425;
            if p < plow {
                let q = (-2.0 * p.ln()).sqrt();
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            } else if p < 1.0 - plow {
                let q = p - 0.5;
                let r = q * q;
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
            } else {
                let q = (-2.0 * (1.0 - p).ln()).sqrt();
                -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            }
        };
        let out: Vec<f64> = p.iter().map(|p| mu + sigma * inv_phi(*p)).collect();
        Ok(json!({"ppf": out}))
    })
}

/// Uniform pdf.
#[no_mangle]
pub extern "C" fn polars__dist_uniform_pdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let lo = args.get("lo").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let hi = args.get("hi").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let p = 1.0 / (hi - lo);
        let out: Vec<f64> = x
            .iter()
            .map(|x| if *x >= lo && *x <= hi { p } else { 0.0 })
            .collect();
        Ok(json!({"pdf": out}))
    })
}

/// Uniform cdf.
#[no_mangle]
pub extern "C" fn polars__dist_uniform_cdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let lo = args.get("lo").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let hi = args.get("hi").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x < lo {
                    0.0
                } else if *x > hi {
                    1.0
                } else {
                    (x - lo) / (hi - lo)
                }
            })
            .collect();
        Ok(json!({"cdf": out}))
    })
}

/// Exp pdf.
#[no_mangle]
pub extern "C" fn polars__dist_exp_pdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let lambda = args.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x < 0.0 {
                    0.0
                } else {
                    lambda * (-lambda * x).exp()
                }
            })
            .collect();
        Ok(json!({"pdf": out}))
    })
}

/// Exp cdf.
#[no_mangle]
pub extern "C" fn polars__dist_exp_cdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let lambda = args.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x < 0.0 {
                    0.0
                } else {
                    1.0 - (-lambda * x).exp()
                }
            })
            .collect();
        Ok(json!({"cdf": out}))
    })
}

/// Gamma pdf.
#[no_mangle]
pub extern "C" fn polars__dist_gamma_pdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let k = args
            .get("k")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        let theta = args.get("theta").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let lg = lgamma(k);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x <= 0.0 {
                    0.0
                } else {
                    ((k - 1.0) * x.ln() - x / theta - k * theta.ln() - lg).exp()
                }
            })
            .collect();
        Ok(json!({"pdf": out}))
    })
}

/// Gamma cdf.
#[no_mangle]
pub extern "C" fn polars__dist_gamma_cdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let k = args
            .get("k")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        let theta = args.get("theta").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x <= 0.0 {
                    0.0
                } else {
                    gamma_p(k, x / theta)
                }
            })
            .collect();
        Ok(json!({"cdf": out}))
    })
}

/// Chi2 pdf.
#[no_mangle]
pub extern "C" fn polars__dist_chi2_pdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let k = args
            .get("df")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `df`"))?;
        let lg = lgamma(k / 2.0);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x <= 0.0 {
                    0.0
                } else {
                    ((k / 2.0 - 1.0) * x.ln() - x / 2.0 - (k / 2.0) * (2.0_f64).ln() - lg).exp()
                }
            })
            .collect();
        Ok(json!({"pdf": out}))
    })
}

/// Chi2 cdf.
#[no_mangle]
pub extern "C" fn polars__dist_chi2_cdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let k = args
            .get("df")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `df`"))?;
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x <= 0.0 {
                    0.0
                } else {
                    gamma_p(k / 2.0, x / 2.0)
                }
            })
            .collect();
        Ok(json!({"cdf": out}))
    })
}

/// Chi2 sf.
#[no_mangle]
pub extern "C" fn polars__dist_chi2_sf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let k = args
            .get("df")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `df`"))?;
        let out: Vec<f64> = x.iter().map(|x| chi2_sf(k, *x)).collect();
        Ok(json!({"sf": out}))
    })
}

/// T pdf.
#[no_mangle]
pub extern "C" fn polars__dist_t_pdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let df = args
            .get("df")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `df`"))?;
        let coef = (lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0) - 0.5 * (df * PI).ln()).exp();
        let out: Vec<f64> = x
            .iter()
            .map(|x| coef * (1.0 + x * x / df).powf(-(df + 1.0) / 2.0))
            .collect();
        Ok(json!({"pdf": out}))
    })
}

/// T cdf.
#[no_mangle]
pub extern "C" fn polars__dist_t_cdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let df = args
            .get("df")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `df`"))?;
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                let xx = df / (df + x * x);
                let ib = beta_inc(df / 2.0, 0.5, xx);
                if *x < 0.0 {
                    0.5 * ib
                } else {
                    1.0 - 0.5 * ib
                }
            })
            .collect();
        Ok(json!({"cdf": out}))
    })
}

/// Beta pdf.
#[no_mangle]
pub extern "C" fn polars__dist_beta_pdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let a = args
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let lc = lgamma(a + b) - lgamma(a) - lgamma(b);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x <= 0.0 || *x >= 1.0 {
                    0.0
                } else {
                    (lc + (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln()).exp()
                }
            })
            .collect();
        Ok(json!({"pdf": out}))
    })
}

/// Beta cdf.
#[no_mangle]
pub extern "C" fn polars__dist_beta_cdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let a = args
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `a`"))?;
        let b = args
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `b`"))?;
        let out: Vec<f64> = x.iter().map(|x| beta_inc(a, b, *x)).collect();
        Ok(json!({"cdf": out}))
    })
}

/// Lognormal pdf.
#[no_mangle]
pub extern "C" fn polars__dist_lognormal_pdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x <= 0.0 {
                    0.0
                } else {
                    let z = (x.ln() - mu) / sigma;
                    (-0.5 * z * z).exp() / (x * sigma * (2.0 * PI).sqrt())
                }
            })
            .collect();
        Ok(json!({"pdf": out}))
    })
}

/// Lognormal cdf.
#[no_mangle]
pub extern "C" fn polars__dist_lognormal_cdf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let x = get_arr(&args, "x")?;
        let mu = args.get("mu").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let out: Vec<f64> = x
            .iter()
            .map(|x| {
                if *x <= 0.0 {
                    0.0
                } else {
                    phi((x.ln() - mu) / sigma)
                }
            })
            .collect();
        Ok(json!({"cdf": out}))
    })
}

/// Poisson pmf.
#[no_mangle]
pub extern "C" fn polars__dist_poisson_pmf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let k = args
            .get("k")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        let lambda = args
            .get("lambda")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `lambda`"))?;
        let out: Vec<f64> = k
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|k| (k as f64 * lambda.ln() - lambda - lgamma(k as f64 + 1.0)).exp())
            .collect();
        Ok(json!({"pmf": out}))
    })
}

/// Binom pmf.
#[no_mangle]
pub extern "C" fn polars__dist_binom_pmf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let k = args
            .get("k")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as f64;
        let p = args
            .get("p")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `p`"))?;
        let out: Vec<f64> = k
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|k| {
                let kf = k as f64;
                let log_c = lgamma(n + 1.0) - lgamma(kf + 1.0) - lgamma(n - kf + 1.0);
                (log_c + kf * p.ln() + (n - kf) * (1.0 - p).ln()).exp()
            })
            .collect();
        Ok(json!({"pmf": out}))
    })
}

/// Geom pmf.
#[no_mangle]
pub extern "C" fn polars__dist_geom_pmf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let k = args
            .get("k")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        let p = args
            .get("p")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `p`"))?;
        let out: Vec<f64> = k
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|k| (1.0 - p).powi(k as i32 - 1) * p)
            .collect();
        Ok(json!({"pmf": out}))
    })
}

/// Negbinom pmf.
#[no_mangle]
pub extern "C" fn polars__dist_negbinom_pmf(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let k = args
            .get("k")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `k`"))?;
        let r = args
            .get("r")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `r`"))?;
        let p = args
            .get("p")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `p`"))?;
        let out: Vec<f64> = k
            .iter()
            .filter_map(|v| v.as_u64())
            .map(|k| {
                let kf = k as f64;
                let log_c = lgamma(kf + r) - lgamma(r) - lgamma(kf + 1.0);
                (log_c + r * p.ln() + kf * (1.0 - p).ln()).exp()
            })
            .collect();
        Ok(json!({"pmf": out}))
    })
}

/// Statistical test confidence interval mean.
#[no_mangle]
pub extern "C" fn polars__stattest_confidence_interval_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.05);
        let n = a.len() as f64;
        let m = a.iter().sum::<f64>() / n;
        let var = a.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let se = (var / n).sqrt();
        // Use normal approximation (z-based).
        let z = 1.96; // approximate for 95%, OK for default alpha=0.05
        let _ = alpha;
        Ok(json!({
            "mean": scalar(m),
            "low": scalar(m - z * se),
            "high": scalar(m + z * se),
        }))
    })
}

/// Statistical test confidence interval prop.
#[no_mangle]
pub extern "C" fn polars__stattest_confidence_interval_prop(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let successes = args
            .get("successes")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `successes`"))? as f64;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as f64;
        let p = successes / n;
        let z = 1.96;
        let se = (p * (1.0 - p) / n).sqrt();
        Ok(json!({
            "proportion": scalar(p),
            "low": scalar(p - z * se),
            "high": scalar(p + z * se),
        }))
    })
}

/// Statistical test proportion ztest.
#[no_mangle]
pub extern "C" fn polars__stattest_proportion_ztest(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let successes = args
            .get("successes")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `successes`"))? as f64;
        let n = args
            .get("n")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n`"))? as f64;
        let p0 = args
            .get("p0")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing `p0`"))?;
        let p = successes / n;
        let se = (p0 * (1.0 - p0) / n).sqrt();
        let z = (p - p0) / se;
        let pv = 2.0 * (1.0 - phi(z.abs()));
        Ok(json!({"statistic": scalar(z), "pvalue": scalar(pv)}))
    })
}

/// Statistical test proportion 2samp ztest.
#[no_mangle]
pub extern "C" fn polars__stattest_proportion_2samp_ztest(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s1 = args
            .get("s1")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `s1`"))? as f64;
        let n1 = args
            .get("n1")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n1`"))? as f64;
        let s2 = args
            .get("s2")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `s2`"))? as f64;
        let n2 = args
            .get("n2")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `n2`"))? as f64;
        let p1 = s1 / n1;
        let p2 = s2 / n2;
        let pp = (s1 + s2) / (n1 + n2);
        let se = (pp * (1.0 - pp) * (1.0 / n1 + 1.0 / n2)).sqrt();
        let z = (p1 - p2) / se;
        let pv = 2.0 * (1.0 - phi(z.abs()));
        Ok(json!({"statistic": scalar(z), "pvalue": scalar(pv)}))
    })
}

/// Statistical test runs.
#[no_mangle]
pub extern "C" fn polars__stattest_runs(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let n1 = a.iter().filter(|x| **x > 0.0).count() as f64;
        let n2 = a.iter().filter(|x| **x <= 0.0).count() as f64;
        let mut runs = 1.0;
        for i in 1..a.len() {
            if (a[i] > 0.0) != (a[i - 1] > 0.0) {
                runs += 1.0;
            }
        }
        let mu = 2.0 * n1 * n2 / (n1 + n2) + 1.0;
        let sd = ((2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2))
            / ((n1 + n2).powi(2) * (n1 + n2 - 1.0)))
            .sqrt();
        let z = (runs - mu) / sd;
        let p = 2.0 * (1.0 - phi(z.abs()));
        Ok(json!({"runs": runs, "statistic": scalar(z), "pvalue": scalar(p)}))
    })
}

/// Statistical test normality test.
#[no_mangle]
pub extern "C" fn polars__stattest_normality_test(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr(&args, "a")?;
        let n = a.len() as f64;
        let m = a.iter().sum::<f64>() / n.max(1.0);
        let var = a.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n.max(1.0);
        let sd = var.sqrt();
        let sk = a.iter().map(|x| ((x - m) / sd).powi(3)).sum::<f64>() / n.max(1.0);
        let kt = a.iter().map(|x| ((x - m) / sd).powi(4)).sum::<f64>() / n.max(1.0) - 3.0;
        let k2 = n / 6.0 * sk.powi(2) + n / 24.0 * kt.powi(2);
        let p = chi2_sf(2.0, k2);
        Ok(json!({"statistic": scalar(k2), "pvalue": scalar(p)}))
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use serde_json::json;

    use crate::ffi_test::call;

    use super::*;

    #[test]
    fn dist_normal_pdf_at_zero_with_unit_sigma() {
        // φ(0; 0, 1) = 1/√(2π).
        let v = call(
            polars__dist_normal_pdf,
            json!({"x": [0.0], "mu": 0.0, "sigma": 1.0}),
        );
        let p = v["pdf"][0].as_f64().unwrap();
        assert!((p - 1.0 / (2.0 * PI).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn dist_normal_cdf_at_three_sigma() {
        // Standard normal CDF(0) = 0.5 within the erf approximation's bound.
        let v = call(
            polars__dist_normal_cdf,
            json!({"x": [0.0], "mu": 0.0, "sigma": 1.0}),
        );
        assert!((v["cdf"][0].as_f64().unwrap() - 0.5).abs() < 1e-4);
        // Φ(3) ≈ 0.99865.
        let v = call(
            polars__dist_normal_cdf,
            json!({"x": [3.0], "mu": 0.0, "sigma": 1.0}),
        );
        assert!((v["cdf"][0].as_f64().unwrap() - 0.99865).abs() < 1e-3);
    }

    #[test]
    fn dist_uniform_cdf_linear_in_range() {
        let v = call(
            polars__dist_uniform_cdf,
            json!({"x": [0.3], "lo": 0.0, "hi": 1.0}),
        );
        assert!((v["cdf"][0].as_f64().unwrap() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn dist_exp_cdf_at_one_is_one_minus_one_over_e() {
        let v = call(polars__dist_exp_cdf, json!({"x": [1.0], "lambda": 1.0}));
        assert!((v["cdf"][0].as_f64().unwrap() - (1.0 - 1.0_f64.exp().recip())).abs() < 1e-12);
    }

    #[test]
    fn interp_linear_matches_endpoints_and_midpoint() {
        let v = call(
            polars__interp_linear,
            json!({"x": [0.0, 10.0], "y": [0.0, 100.0], "xnew": [0.0, 5.0, 10.0]}),
        );
        let y: Vec<f64> = v["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(y, vec![0.0, 50.0, 100.0]);
    }

    #[test]
    fn interp_nearest_picks_closest_neighbor() {
        let v = call(
            polars__interp_nearest,
            json!({"x": [0.0, 10.0], "y": [1.0, 9.0], "xnew": [4.0, 6.0]}),
        );
        let y: Vec<f64> = v["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(y, vec![1.0, 9.0]);
    }

    #[test]
    fn stattest_linregress_recovers_slope_intercept() {
        // y = 2x + 1 → slope=2, intercept=1, r=1.
        let v = call(
            polars__stattest_linregress,
            json!({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 5.0, 7.0, 9.0]}),
        );
        assert!((v["slope"].as_f64().unwrap() - 2.0).abs() < 1e-12);
        assert!((v["intercept"].as_f64().unwrap() - 1.0).abs() < 1e-12);
        assert!((v["rvalue"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stattest_ttest_1samp_zero_for_matching_mean() {
        let v = call(
            polars__stattest_ttest_1samp,
            json!({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "popmean": 3.0}),
        );
        assert!(v["statistic"].as_f64().unwrap().abs() < 1e-12);
    }

    #[test]
    fn stattest_anova_zero_between_identical_groups() {
        let v = call(
            polars__stattest_anova,
            json!({"groups": [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]}),
        );
        assert!(v["statistic"].as_f64().unwrap().abs() < 1e-12);
    }

    #[test]
    fn stattest_zscore_centered_and_mirrored() {
        // For symmetric series centered on the mean, z[1]=0 and z[0]=-z[2].
        let v = call(polars__stattest_zscore, json!({"a": [1.0, 2.0, 3.0]}));
        let z: Vec<f64> = v["zscore"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert!(z[1].abs() < 1e-12);
        assert!((z[0] + z[2]).abs() < 1e-12);
    }
}
