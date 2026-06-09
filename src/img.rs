//! src/img.rs — 2-D image / matrix processing (`polars__img_*`).
//!
//! 2-D filters: blur, sharpen, edge detection (Sobel, Prewitt, Laplacian),
//! morphology, transforms, basic histogram. All take a 2-D ndarray.

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

fn get_2d(args: &Value) -> Result<(Vec<Vec<f64>>, usize, usize)> {
    let a = get_array(args, "image")?;
    if a.shape().len() != 2 {
        bail!("image must be 2-D");
    }
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let mut grid = vec![vec![0.0; cols]; rows];
    for (r, row) in grid.iter_mut().enumerate() {
        for (c, val) in row.iter_mut().enumerate() {
            *val = a[[r, c].as_slice()];
        }
    }
    Ok((grid, rows, cols))
}

fn grid_to_value(grid: &[Vec<f64>], rows: usize, cols: usize) -> Result<Value> {
    let mut data = Vec::with_capacity(rows * cols);
    for row in grid {
        data.extend(row.iter().copied());
    }
    let arr = ArrayD::from_shape_vec(IxDyn(&[rows, cols]), data).context("grid")?;
    Ok(json!({"image": array_to_value(&arr)}))
}

fn conv2d(grid: &[Vec<f64>], rows: usize, cols: usize, kernel: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let kh = kernel.len();
    let kw = kernel[0].len();
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let mut out = vec![vec![0.0; cols]; rows];
    for (r, out_row) in out.iter_mut().enumerate() {
        for (c, val) in out_row.iter_mut().enumerate() {
            let mut s = 0.0;
            for (ki, krow) in kernel.iter().enumerate() {
                for (kj, &kv) in krow.iter().enumerate() {
                    let rr = r as i64 + ki as i64 - pad_h as i64;
                    let cc = c as i64 + kj as i64 - pad_w as i64;
                    if rr >= 0 && rr < rows as i64 && cc >= 0 && cc < cols as i64 {
                        s += grid[rr as usize][cc as usize] * kv;
                    }
                }
            }
            *val = s;
        }
    }
    out
}

/// Apply a per-window neighborhood reducer over each (i, j) pixel.
fn windowed<F: Fn(f64, f64) -> f64>(
    grid: &[Vec<f64>],
    rows: usize,
    cols: usize,
    r: i64,
    init: f64,
    op: F,
) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; cols]; rows];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (j, val) in out_row.iter_mut().enumerate() {
            let mut m = init;
            for di in -r..=r {
                for dj in -r..=r {
                    let ni = i as i64 + di;
                    let nj = j as i64 + dj;
                    if ni >= 0 && ni < rows as i64 && nj >= 0 && nj < cols as i64 {
                        m = op(m, grid[ni as usize][nj as usize]);
                    }
                }
            }
            *val = m;
        }
    }
    out
}

fn dilate_grid(grid: &[Vec<f64>], rows: usize, cols: usize, r: i64) -> Vec<Vec<f64>> {
    windowed(grid, rows, cols, r, f64::NEG_INFINITY, f64::max)
}

fn erode_grid(grid: &[Vec<f64>], rows: usize, cols: usize, r: i64) -> Vec<Vec<f64>> {
    windowed(grid, rows, cols, r, f64::INFINITY, f64::min)
}

// ── kernels ────────────────────────────────────────────────────────────────

/// Image box blur.
#[no_mangle]
pub extern "C" fn polars__img_box_blur(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let w = args.get("window").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let k: Vec<Vec<f64>> = (0..w).map(|_| vec![1.0 / (w * w) as f64; w]).collect();
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

/// Image gaussian blur.
#[no_mangle]
pub extern "C" fn polars__img_gaussian_blur(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let sigma = args.get("sigma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let radius = (3.0 * sigma).ceil() as usize;
        let size = 2 * radius + 1;
        let mut k = vec![vec![0.0; size]; size];
        let mut sum = 0.0;
        for (i, krow) in k.iter_mut().enumerate() {
            for (j, kval) in krow.iter_mut().enumerate() {
                let x = i as f64 - radius as f64;
                let y = j as f64 - radius as f64;
                let v = (-0.5 * (x * x + y * y) / (sigma * sigma)).exp();
                *kval = v;
                sum += v;
            }
        }
        for row in k.iter_mut() {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

/// Image sharpen.
#[no_mangle]
pub extern "C" fn polars__img_sharpen(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let k = vec![
            vec![0.0, -1.0, 0.0],
            vec![-1.0, 5.0, -1.0],
            vec![0.0, -1.0, 0.0],
        ];
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

/// Image sobel x.
#[no_mangle]
pub extern "C" fn polars__img_sobel_x(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let k = vec![
            vec![-1.0, 0.0, 1.0],
            vec![-2.0, 0.0, 2.0],
            vec![-1.0, 0.0, 1.0],
        ];
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

/// Image sobel y.
#[no_mangle]
pub extern "C" fn polars__img_sobel_y(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let k = vec![
            vec![-1.0, -2.0, -1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 1.0],
        ];
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

/// Image sobel magnitude.
#[no_mangle]
pub extern "C" fn polars__img_sobel_magnitude(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let kx = vec![
            vec![-1.0, 0.0, 1.0],
            vec![-2.0, 0.0, 2.0],
            vec![-1.0, 0.0, 1.0],
        ];
        let ky = vec![
            vec![-1.0, -2.0, -1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 1.0],
        ];
        let gx = conv2d(&g, rows, cols, &kx);
        let gy = conv2d(&g, rows, cols, &ky);
        let mut out = vec![vec![0.0; cols]; rows];
        for (r, out_row) in out.iter_mut().enumerate() {
            for (c, val) in out_row.iter_mut().enumerate() {
                *val = (gx[r][c].powi(2) + gy[r][c].powi(2)).sqrt();
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image prewitt x.
#[no_mangle]
pub extern "C" fn polars__img_prewitt_x(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let k = vec![
            vec![-1.0, 0.0, 1.0],
            vec![-1.0, 0.0, 1.0],
            vec![-1.0, 0.0, 1.0],
        ];
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

/// Image prewitt y.
#[no_mangle]
pub extern "C" fn polars__img_prewitt_y(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let k = vec![
            vec![-1.0, -1.0, -1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

/// Image laplacian.
#[no_mangle]
pub extern "C" fn polars__img_laplacian(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let k = vec![
            vec![0.0, -1.0, 0.0],
            vec![-1.0, 4.0, -1.0],
            vec![0.0, -1.0, 0.0],
        ];
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

/// Image emboss.
#[no_mangle]
pub extern "C" fn polars__img_emboss(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let k = vec![
            vec![-2.0, -1.0, 0.0],
            vec![-1.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let out = conv2d(&g, rows, cols, &k);
        grid_to_value(&out, rows, cols)
    })
}

// ── morphology ──────────────────────────────────────────────────────────────

/// Image dilate.
#[no_mangle]
pub extern "C" fn polars__img_dilate(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let r = args.get("radius").and_then(|v| v.as_u64()).unwrap_or(1) as i64;
        let out = dilate_grid(&g, rows, cols, r);
        grid_to_value(&out, rows, cols)
    })
}

/// Image erode.
#[no_mangle]
pub extern "C" fn polars__img_erode(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let r = args.get("radius").and_then(|v| v.as_u64()).unwrap_or(1) as i64;
        let out = erode_grid(&g, rows, cols, r);
        grid_to_value(&out, rows, cols)
    })
}

/// Image open.
#[no_mangle]
pub extern "C" fn polars__img_open(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let r = args.get("radius").and_then(|v| v.as_u64()).unwrap_or(1) as i64;
        let e = erode_grid(&g, rows, cols, r);
        let d = dilate_grid(&e, rows, cols, r);
        grid_to_value(&d, rows, cols)
    })
}

/// Image close.
#[no_mangle]
pub extern "C" fn polars__img_close(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let r = args.get("radius").and_then(|v| v.as_u64()).unwrap_or(1) as i64;
        let d = dilate_grid(&g, rows, cols, r);
        let e = erode_grid(&d, rows, cols, r);
        grid_to_value(&e, rows, cols)
    })
}

/// Image median filter.
#[no_mangle]
pub extern "C" fn polars__img_median_filter(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let r = args.get("radius").and_then(|v| v.as_u64()).unwrap_or(1) as i64;
        let mut out = vec![vec![0.0; cols]; rows];
        for (i, out_row) in out.iter_mut().enumerate() {
            for (j, val) in out_row.iter_mut().enumerate() {
                let mut v = vec![];
                for di in -r..=r {
                    for dj in -r..=r {
                        let ni = i as i64 + di;
                        let nj = j as i64 + dj;
                        if ni >= 0 && ni < rows as i64 && nj >= 0 && nj < cols as i64 {
                            v.push(g[ni as usize][nj as usize]);
                        }
                    }
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                *val = v[v.len() / 2];
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image threshold.
#[no_mangle]
pub extern "C" fn polars__img_threshold(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let t = args
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let max = args.get("max").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r][c] = if g[r][c] >= t { max } else { 0.0 };
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image invert.
#[no_mangle]
pub extern "C" fn polars__img_invert(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let max = args.get("max").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r][c] = max - g[r][c];
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image rotate90.
#[no_mangle]
pub extern "C" fn polars__img_rotate90(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let mut out = vec![vec![0.0; rows]; cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c][rows - 1 - r] = g[r][c];
            }
        }
        grid_to_value(&out, cols, rows)
    })
}

/// Image rotate180.
#[no_mangle]
pub extern "C" fn polars__img_rotate180(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[rows - 1 - r][cols - 1 - c] = g[r][c];
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image rotate270.
#[no_mangle]
pub extern "C" fn polars__img_rotate270(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let mut out = vec![vec![0.0; rows]; cols];
        for r in 0..rows {
            for c in 0..cols {
                out[cols - 1 - c][r] = g[r][c];
            }
        }
        grid_to_value(&out, cols, rows)
    })
}

/// Image flip horizontal.
#[no_mangle]
pub extern "C" fn polars__img_flip_horizontal(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r][cols - 1 - c] = g[r][c];
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image flip vertical.
#[no_mangle]
pub extern "C" fn polars__img_flip_vertical(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[rows - 1 - r][c] = g[r][c];
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image crop.
#[no_mangle]
pub extern "C" fn polars__img_crop(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _rows, _cols) = get_2d(&args)?;
        let r0 = args.get("r0").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let c0 = args.get("c0").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let r1 = args
            .get("r1")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `r1`"))? as usize;
        let c1 = args
            .get("c1")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `c1`"))? as usize;
        let h = r1 - r0;
        let w = c1 - c0;
        let mut out = vec![vec![0.0; w]; h];
        for r in 0..h {
            for c in 0..w {
                out[r][c] = g[r0 + r][c0 + c];
            }
        }
        grid_to_value(&out, h, w)
    })
}

/// Image resize nearest.
#[no_mangle]
pub extern "C" fn polars__img_resize_nearest(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let new_rows = args
            .get("rows")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `rows`"))? as usize;
        let new_cols = args
            .get("cols")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `cols`"))? as usize;
        let mut out = vec![vec![0.0; new_cols]; new_rows];
        for (r, out_row) in out.iter_mut().enumerate() {
            for (c, val) in out_row.iter_mut().enumerate() {
                let sr = (r * rows / new_rows).min(rows - 1);
                let sc = (c * cols / new_cols).min(cols - 1);
                *val = g[sr][sc];
            }
        }
        grid_to_value(&out, new_rows, new_cols)
    })
}

/// Image resize bilinear.
#[no_mangle]
pub extern "C" fn polars__img_resize_bilinear(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let new_rows = args
            .get("rows")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `rows`"))? as usize;
        let new_cols = args
            .get("cols")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing `cols`"))? as usize;
        let mut out = vec![vec![0.0; new_cols]; new_rows];
        for (r, out_row) in out.iter_mut().enumerate() {
            for (c, val) in out_row.iter_mut().enumerate() {
                let fr = r as f64 * (rows - 1) as f64 / (new_rows - 1).max(1) as f64;
                let fc = c as f64 * (cols - 1) as f64 / (new_cols - 1).max(1) as f64;
                let r0 = fr.floor() as usize;
                let c0 = fc.floor() as usize;
                let r1 = (r0 + 1).min(rows - 1);
                let c1 = (c0 + 1).min(cols - 1);
                let dr = fr - r0 as f64;
                let dc = fc - c0 as f64;
                *val = g[r0][c0] * (1.0 - dr) * (1.0 - dc)
                    + g[r1][c0] * dr * (1.0 - dc)
                    + g[r0][c1] * (1.0 - dr) * dc
                    + g[r1][c1] * dr * dc;
            }
        }
        grid_to_value(&out, new_rows, new_cols)
    })
}

/// Image histogram.
#[no_mangle]
pub extern "C" fn polars__img_histogram(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _rows, _cols) = get_2d(&args)?;
        let bins = args.get("bins").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let mn = g.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let mx = g
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let width = (mx - mn) / bins as f64;
        let mut hist = vec![0i64; bins];
        for row in &g {
            for v in row {
                let idx = (((v - mn) / width) as usize).min(bins - 1);
                hist[idx] += 1;
            }
        }
        Ok(json!({"histogram": hist, "min": mn, "max": mx, "width": width}))
    })
}

/// Image equalize.
#[no_mangle]
pub extern "C" fn polars__img_equalize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let levels = args.get("levels").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
        let mn = g.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let mx = g
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let bin_w = (mx - mn) / levels as f64;
        let mut hist = vec![0u64; levels];
        for row in &g {
            for v in row {
                let idx = (((v - mn) / bin_w) as usize).min(levels - 1);
                hist[idx] += 1;
            }
        }
        let mut cdf = vec![0u64; levels];
        let mut acc = 0u64;
        for i in 0..levels {
            acc += hist[i];
            cdf[i] = acc;
        }
        let total = (rows * cols) as f64;
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                let idx = (((g[r][c] - mn) / bin_w) as usize).min(levels - 1);
                out[r][c] = cdf[idx] as f64 / total;
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image contrast.
#[no_mangle]
pub extern "C" fn polars__img_contrast(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let factor = args.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mn = g.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let mx = g
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mid = (mn + mx) / 2.0;
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r][c] = mid + (g[r][c] - mid) * factor;
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image brightness.
#[no_mangle]
pub extern "C" fn polars__img_brightness(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let delta = args.get("delta").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r][c] = g[r][c] + delta;
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image gamma.
#[no_mangle]
pub extern "C" fn polars__img_gamma(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let gamma = args.get("gamma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r][c] = g[r][c].max(0.0).powf(gamma);
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image normalize.
#[no_mangle]
pub extern "C" fn polars__img_normalize(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let mn = g.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let mx = g
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = mx - mn;
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r][c] = if range == 0.0 {
                    0.0
                } else {
                    (g[r][c] - mn) / range
                };
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image mean.
#[no_mangle]
pub extern "C" fn polars__img_mean(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _rows, _cols) = get_2d(&args)?;
        let n: usize = g.iter().map(|r| r.len()).sum();
        let s: f64 = g.iter().flatten().sum();
        Ok(json!({"mean": s / n as f64}))
    })
}

/// Image std.
#[no_mangle]
pub extern "C" fn polars__img_std(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _rows, _cols) = get_2d(&args)?;
        let n: usize = g.iter().map(|r| r.len()).sum();
        let s: f64 = g.iter().flatten().sum();
        let m = s / n as f64;
        let var: f64 = g.iter().flatten().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        Ok(json!({"std": var.sqrt()}))
    })
}

/// Image min.
#[no_mangle]
pub extern "C" fn polars__img_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _rows, _cols) = get_2d(&args)?;
        let r = g.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        Ok(json!({"min": r}))
    })
}

/// Image max.
#[no_mangle]
pub extern "C" fn polars__img_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _rows, _cols) = get_2d(&args)?;
        let r = g
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        Ok(json!({"max": r}))
    })
}

/// Image sum.
#[no_mangle]
pub extern "C" fn polars__img_sum(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _rows, _cols) = get_2d(&args)?;
        let r: f64 = g.iter().flatten().sum();
        Ok(json!({"sum": r}))
    })
}

/// Image transpose.
#[no_mangle]
pub extern "C" fn polars__img_transpose(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let mut out = vec![vec![0.0; rows]; cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c][r] = g[r][c];
            }
        }
        grid_to_value(&out, cols, rows)
    })
}

/// Image integral image.
#[no_mangle]
pub extern "C" fn polars__img_integral_image(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let mut out = vec![vec![0.0; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                let mut v = g[r][c];
                if r > 0 {
                    v += out[r - 1][c];
                }
                if c > 0 {
                    v += out[r][c - 1];
                }
                if r > 0 && c > 0 {
                    v -= out[r - 1][c - 1];
                }
                out[r][c] = v;
            }
        }
        grid_to_value(&out, rows, cols)
    })
}

/// Image pad.
#[no_mangle]
pub extern "C" fn polars__img_pad(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, rows, cols) = get_2d(&args)?;
        let pad = args.get("pad").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let value = args.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let new_rows = rows + 2 * pad;
        let new_cols = cols + 2 * pad;
        let mut out = vec![vec![value; new_cols]; new_rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r + pad][c + pad] = g[r][c];
            }
        }
        grid_to_value(&out, new_rows, new_cols)
    })
}

/// Image to grayscale.
#[no_mangle]
pub extern "C" fn polars__img_to_grayscale(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let r = get_array(&args, "r")?;
        let g = get_array(&args, "g")?;
        let b = get_array(&args, "b")?;
        if r.shape() != g.shape() || g.shape() != b.shape() {
            bail!("RGB shapes must match");
        }
        let out: Vec<f64> = r
            .iter()
            .zip(g.iter())
            .zip(b.iter())
            .map(|((rv, gv), bv)| 0.299 * rv + 0.587 * gv + 0.114 * bv)
            .collect();
        let arr = ndarray::ArrayD::from_shape_vec(IxDyn(r.shape()), out).context("grayscale")?;
        Ok(json!({"image": array_to_value(&arr)}))
    })
}

/// Image argmax 2d.
#[no_mangle]
pub extern "C" fn polars__img_argmax_2d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _, _) = get_2d(&args)?;
        let mut best = (0usize, 0usize, f64::NEG_INFINITY);
        for (r, row) in g.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v > best.2 {
                    best = (r, c, v);
                }
            }
        }
        Ok(json!({"row": best.0, "col": best.1, "value": best.2}))
    })
}

/// Image argmin 2d.
#[no_mangle]
pub extern "C" fn polars__img_argmin_2d(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let (g, _, _) = get_2d(&args)?;
        let mut best = (0usize, 0usize, f64::INFINITY);
        for (r, row) in g.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v < best.2 {
                    best = (r, c, v);
                }
            }
        }
        Ok(json!({"row": best.0, "col": best.1, "value": best.2}))
    })
}
