//! src/dt64.rs — numpy datetime64 / timedelta64 surface (`polars__dt64_*`).
//!
//! Wire format: dates as ISO-8601 strings `YYYY-MM-DD[ HH:MM:SS]`;
//! timedeltas as f64 in unit (seconds default).

use std::ffi::c_char;

use anyhow::{anyhow, bail, Result};
use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime, Timelike};
use serde_json::{json, Value};

use crate::ffi_call;

fn parse_dt(s: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
        .or_else(|_| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap_or_default())
        })
        .map_err(|e| anyhow!("parse datetime: {e}"))
}

fn fmt_dt(d: NaiveDateTime) -> String {
    d.format("%Y-%m-%d %H:%M:%S").to_string()
}

fn get_str(args: &Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| anyhow!("missing argument `{key}`"))
}

fn get_arr_str(args: &Value, key: &str) -> Result<Vec<String>> {
    let a = args
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing argument `{key}`"))?;
    Ok(a.iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect())
}

// ── parse / format / construct ─────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__dt64_datetime64(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let d = parse_dt(&s)?;
        Ok(json!({"datetime": fmt_dt(d)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_from_iso(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<String> = arr
            .iter()
            .map(|s| parse_dt(s).map(fmt_dt).unwrap_or_default())
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_to_iso(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(fmt_dt)
            .collect();
        Ok(json!({"iso": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_to_timestamp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<i64> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| d.and_utc().timestamp())
            .collect();
        Ok(json!({"timestamp": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_from_timestamp(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = args
            .get("timestamp")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing `timestamp`"))?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|x| x.as_i64())
            .filter_map(|t| chrono::DateTime::from_timestamp(t, 0).map(|d| d.naive_utc()))
            .map(fmt_dt)
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_now(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        // chrono::Utc::now reads the system clock — accept that determinism is
        // not relevant here.
        let n = chrono::Utc::now().naive_utc();
        Ok(json!({"now": fmt_dt(n)}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_today(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        let n = chrono::Utc::now().naive_utc().date();
        Ok(json!({"today": n.to_string()}))
    })
}

// ── components ──────────────────────────────────────────────────────────────

macro_rules! dt64_component {
    ($fn_name:ident, $key:literal, $f:expr) => {
        #[no_mangle]
        pub extern "C" fn $fn_name(args: *const c_char) -> *mut c_char {
            ffi_call(args, |args| {
                let arr = get_arr_str(&args, "values")?;
                let out: Vec<i64> = arr.iter().filter_map(|s| parse_dt(s).ok()).map(|d| ($f)(d)).collect();
                Ok(json!({ $key: out }))
            })
        }
    };
}

dt64_component!(polars__dt64_year, "year", |d: NaiveDateTime| d.year()
    as i64);
dt64_component!(polars__dt64_month, "month", |d: NaiveDateTime| d.month()
    as i64);
dt64_component!(polars__dt64_day, "day", |d: NaiveDateTime| d.day() as i64);
dt64_component!(polars__dt64_hour, "hour", |d: NaiveDateTime| d.hour()
    as i64);
dt64_component!(polars__dt64_minute, "minute", |d: NaiveDateTime| d.minute()
    as i64);
dt64_component!(polars__dt64_second, "second", |d: NaiveDateTime| d.second()
    as i64);
dt64_component!(polars__dt64_weekday, "weekday", |d: NaiveDateTime| d
    .weekday()
    .num_days_from_monday()
    as i64);
dt64_component!(
    polars__dt64_dayofyear,
    "dayofyear",
    |d: NaiveDateTime| d.ordinal() as i64
);
dt64_component!(
    polars__dt64_quarter,
    "quarter",
    |d: NaiveDateTime| ((d.month() - 1) / 3 + 1) as i64
);
dt64_component!(
    polars__dt64_week,
    "week",
    |d: NaiveDateTime| d.iso_week().week() as i64
);

#[no_mangle]
pub extern "C" fn polars__dt64_is_leap_year(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<bool> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| {
                let y = d.year();
                (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
            })
            .collect();
        Ok(json!({"is_leap_year": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_days_in_month(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<i64> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| {
                let (y, m) = (d.year(), d.month());
                let next = if m == 12 {
                    NaiveDate::from_ymd_opt(y + 1, 1, 1).unwrap()
                } else {
                    NaiveDate::from_ymd_opt(y, m + 1, 1).unwrap()
                };
                let this = NaiveDate::from_ymd_opt(y, m, 1).unwrap();
                next.signed_duration_since(this).num_days()
            })
            .collect();
        Ok(json!({"days_in_month": out}))
    })
}

// ── arithmetic ──────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__dt64_add_days(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| fmt_dt(d + Duration::days(n)))
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_add_hours(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| fmt_dt(d + Duration::hours(n)))
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_add_minutes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| fmt_dt(d + Duration::minutes(n)))
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_add_seconds(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| fmt_dt(d + Duration::seconds(n)))
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_add_weeks(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| fmt_dt(d + Duration::weeks(n)))
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_diff_days(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr_str(&args, "a")?;
        let b = get_arr_str(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let out: Vec<i64> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let da = parse_dt(x).ok();
                let db = parse_dt(y).ok();
                match (da, db) {
                    (Some(p), Some(q)) => p.signed_duration_since(q).num_days(),
                    _ => 0,
                }
            })
            .collect();
        Ok(json!({"days": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_diff_seconds(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr_str(&args, "a")?;
        let b = get_arr_str(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let out: Vec<i64> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let da = parse_dt(x).ok();
                let db = parse_dt(y).ok();
                match (da, db) {
                    (Some(p), Some(q)) => p.signed_duration_since(q).num_seconds(),
                    _ => 0,
                }
            })
            .collect();
        Ok(json!({"seconds": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_diff_hours(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr_str(&args, "a")?;
        let b = get_arr_str(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let out: Vec<i64> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let da = parse_dt(x).ok();
                let db = parse_dt(y).ok();
                match (da, db) {
                    (Some(p), Some(q)) => p.signed_duration_since(q).num_hours(),
                    _ => 0,
                }
            })
            .collect();
        Ok(json!({"hours": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_diff_minutes(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let a = get_arr_str(&args, "a")?;
        let b = get_arr_str(&args, "b")?;
        if a.len() != b.len() {
            bail!("len mismatch");
        }
        let out: Vec<i64> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let da = parse_dt(x).ok();
                let db = parse_dt(y).ok();
                match (da, db) {
                    (Some(p), Some(q)) => p.signed_duration_since(q).num_minutes(),
                    _ => 0,
                }
            })
            .collect();
        Ok(json!({"minutes": out}))
    })
}

// ── ranges ──────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__dt64_date_range(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let start = get_str(&args, "start")?;
        let end = get_str(&args, "end")?;
        let s = parse_dt(&start)?;
        let e = parse_dt(&end)?;
        let mut out = vec![];
        let mut cur = s;
        while cur <= e {
            out.push(fmt_dt(cur));
            cur += Duration::days(1);
        }
        Ok(json!({"dates": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_arange(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let start = get_str(&args, "start")?;
        let end = get_str(&args, "end")?;
        let step_days = args.get("step_days").and_then(|v| v.as_i64()).unwrap_or(1);
        let s = parse_dt(&start)?;
        let e = parse_dt(&end)?;
        let mut out = vec![];
        let mut cur = s;
        while cur < e {
            out.push(fmt_dt(cur));
            cur += Duration::days(step_days);
        }
        Ok(json!({"dates": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_business_day_count(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let start = get_str(&args, "start")?;
        let end = get_str(&args, "end")?;
        let s = parse_dt(&start)?;
        let e = parse_dt(&end)?;
        let mut count = 0;
        let mut cur = s.date();
        while cur < e.date() {
            let wd = cur.weekday().num_days_from_monday();
            if wd < 5 {
                count += 1;
            }
            cur += Duration::days(1);
        }
        Ok(json!({"business_days": count}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_is_busday(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<bool> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| d.weekday().num_days_from_monday() < 5)
            .collect();
        Ok(json!({"is_busday": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_busday_offset(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let n = args.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| {
                let mut cur = d;
                let mut moved: i64 = 0;
                let step: i64 = if n >= 0 { 1 } else { -1 };
                while moved.abs() < n.abs() {
                    cur += Duration::days(step);
                    if cur.weekday().num_days_from_monday() < 5 {
                        moved += step;
                    }
                }
                fmt_dt(cur)
            })
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

// ── max/min / sort ─────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn polars__dt64_min(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .min()
            .map(fmt_dt);
        Ok(json!({"min": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_max(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .max()
            .map(fmt_dt);
        Ok(json!({"max": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_sort(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let mut parsed: Vec<NaiveDateTime> = arr.iter().filter_map(|s| parse_dt(s).ok()).collect();
        parsed.sort();
        let out: Vec<String> = parsed.into_iter().map(fmt_dt).collect();
        Ok(json!({"sorted": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_unique(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let mut seen = std::collections::HashSet::new();
        let mut out = vec![];
        for s in arr {
            if let Ok(d) = parse_dt(&s) {
                let f = fmt_dt(d);
                if seen.insert(f.clone()) {
                    out.push(f);
                }
            }
        }
        Ok(json!({"unique": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_truncate_day(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| {
                let day = d.date().and_hms_opt(0, 0, 0).unwrap_or(d);
                fmt_dt(day)
            })
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_truncate_hour(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| {
                let h = d.date().and_hms_opt(d.hour(), 0, 0).unwrap_or(d);
                fmt_dt(h)
            })
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_truncate_month(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .filter_map(|d| {
                NaiveDate::from_ymd_opt(d.year(), d.month(), 1)
                    .map(|nd| fmt_dt(nd.and_hms_opt(0, 0, 0).unwrap_or_default()))
            })
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_truncate_year(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .filter_map(|d| {
                NaiveDate::from_ymd_opt(d.year(), 1, 1)
                    .map(|nd| fmt_dt(nd.and_hms_opt(0, 0, 0).unwrap_or_default()))
            })
            .collect();
        Ok(json!({"datetimes": out}))
    })
}

#[no_mangle]
pub extern "C" fn polars__dt64_format(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let arr = get_arr_str(&args, "values")?;
        let fmt = get_str(&args, "format")?;
        let out: Vec<String> = arr
            .iter()
            .filter_map(|s| parse_dt(s).ok())
            .map(|d| d.format(&fmt).to_string())
            .collect();
        Ok(json!({"formatted": out}))
    })
}
