//! stryke-polars — Polars-backed pandas + numpy surface cdylib loaded
//! in-process by stryke via dlopen.
//!
//! Each `#[no_mangle] extern "C" fn polars__*` is a JSON-string-in /
//! JSON-string-out wrapper around `polars`, `ndarray`, `ndarray-linalg`,
//! `nalgebra`, `rustfft`, `rand_distr`, and `statrs`-class math.
//!
//! stryke's FFI bridge (`rust_ffi.rs::load_cdylib`) resolves these symbols
//! at first `use Polars`, registers each one as a stryke-callable function,
//! and on each call passes a JSON-encoded args dict and copies the returned
//! JSON into a stryke string. The cdylib's `stryke_free_cstring` export
//! plugs the returned-allocation leak the inline-FFI v1 had.
//!
//! Surfaces (registered as exports are added in phases):
//!   - `polars__df_*`     — pandas DataFrame (head/tail/loc/iloc/at/iat/...)
//!   - `polars__sr_*`     — pandas Series
//!   - `polars__pd_*`     — IO + module-level (read_csv/read_parquet/...)
//!   - `polars__idx_*`    — Index / MultiIndex / DatetimeIndex
//!   - `polars__arr_*`    — numpy ndarray (zeros/ones/arange/reshape/...)
//!   - `polars__np_*`     — numpy ufuncs (add/sub/sin/exp/log/...)
//!   - `polars__linalg_*` — np.linalg (svd/qr/lu/cholesky/eig/solve/...)
//!   - `polars__rand_*`   — np.random distributions
//!   - `polars__fft_*`    — np.fft
//!   - `polars__poly_*`   — np.polynomial (Chebyshev/Hermite/Legendre/...)
//!   - `polars__ma_*`     — masked arrays
//!   - `polars__dt64_*`   — datetime64 / timedelta64

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use serde_json::{json, Value};

const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

// ── FFI plumbing ────────────────────────────────────────────────────────────

/// Allocate a CString from a JSON value, transfer ownership to caller.
/// Caller MUST release via `stryke_free_cstring`.
fn json_to_cstring(v: &Value) -> *mut c_char {
    let s = v.to_string();
    CString::new(s)
        .map(|c| c.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

/// Decode a JSON arg buffer or fall back to `{}`.
fn decode_args(ptr: *const c_char) -> Value {
    if ptr.is_null() {
        return json!({});
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or_else(|| json!({}))
}

/// Run a JSON-in/JSON-out body inside `catch_unwind`; convert any panic or
/// returned `Err` into `{"error": "..."}` so the stryke side gets a
/// well-formed JSON envelope every time.
fn ffi_call<F>(args_ptr: *const c_char, body: F) -> *mut c_char
where
    F: FnOnce(Value) -> anyhow::Result<Value>,
{
    let args = decode_args(args_ptr);
    let result = catch_unwind(AssertUnwindSafe(|| body(args)));
    let v = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => json!({"error": format!("{e:#}")}),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&'static str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic in cdylib".to_string());
            json!({"error": format!("panic: {msg}")})
        }
    };
    json_to_cstring(&v)
}

// ── exports ────────────────────────────────────────────────────────────────

/// Free a CString previously returned by any `polars__*` export.
///
/// # Safety
/// `ptr` must be a pointer returned by this cdylib and not freed before.
#[no_mangle]
pub unsafe extern "C" fn stryke_free_cstring(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    drop(CString::from_raw(ptr));
}

/// Return package version + backing crate versions.
///
/// Args: `{}` (none required).
/// Returns: `{"version": "...", "polars": "...", "ndarray": "..."}`.
#[no_mangle]
pub extern "C" fn polars__version(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        Ok(json!({
            "version": PKG_VERSION,
            "polars": polars_version(),
            "ndarray": ndarray_version(),
        }))
    })
}

fn polars_version() -> &'static str {
    // Reported by the polars crate at compile time.
    env!("CARGO_PKG_VERSION")
}

fn ndarray_version() -> &'static str {
    // Bumped manually when ndarray major version changes.
    "0.16"
}
