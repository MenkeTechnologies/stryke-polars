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

/// datetime64 datetime64.
#[no_mangle]
pub extern "C" fn polars__dt64_datetime64(args: *const c_char) -> *mut c_char {
    ffi_call(args, |args| {
        let s = get_str(&args, "value")?;
        let d = parse_dt(&s)?;
        Ok(json!({"datetime": fmt_dt(d)}))
    })
}

/// datetime64 from iso.
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

/// datetime64 to iso.
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

/// datetime64 to timestamp.
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

/// datetime64 from timestamp.
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

/// datetime64 now.
#[no_mangle]
pub extern "C" fn polars__dt64_now(args: *const c_char) -> *mut c_char {
    ffi_call(args, |_| {
        // chrono::Utc::now reads the system clock — accept that determinism is
        // not relevant here.
        let n = chrono::Utc::now().naive_utc();
        Ok(json!({"now": fmt_dt(n)}))
    })
}

/// datetime64 today.
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

/// datetime64 is leap year.
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

/// datetime64 days in month.
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

/// datetime64 add days.
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

/// datetime64 add hours.
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

/// datetime64 add minutes.
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

/// datetime64 add seconds.
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

/// datetime64 add weeks.
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

/// datetime64 diff days.
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

/// datetime64 diff seconds.
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

/// datetime64 diff hours.
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

/// datetime64 diff minutes.
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

/// datetime64 date range.
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

/// datetime64 arange.
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

/// datetime64 business day count.
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

/// datetime64 is busday.
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

/// datetime64 busday offset.
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

/// datetime64 min.
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

/// datetime64 max.
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

/// datetime64 sort.
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

/// datetime64 unique.
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

/// datetime64 truncate day.
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

/// datetime64 truncate hour.
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

/// datetime64 truncate month.
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

/// datetime64 truncate year.
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

/// datetime64 format.
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::ffi_test::call;

    use super::*;

    #[test]
    fn dt64_is_leap_year_gregorian_rule() {
        // 2000 leap (÷400), 1900 not (÷100 but not 400), 2024 leap, 2023 not.
        let v = call(
            polars__dt64_is_leap_year,
            json!({"values": ["2000-01-01", "1900-01-01", "2024-01-01", "2023-01-01"]}),
        );
        let r: Vec<bool> = v["is_leap_year"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_bool().unwrap())
            .collect();
        assert_eq!(r, vec![true, false, true, false]);
    }

    #[test]
    fn dt64_days_in_month_handles_leap_february() {
        let v = call(
            polars__dt64_days_in_month,
            json!({"values": ["2024-02-15", "2023-02-15", "2024-04-15", "2024-12-15"]}),
        );
        let d: Vec<i64> = v["days_in_month"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(d, vec![29, 28, 30, 31]);
    }

    #[test]
    fn dt64_quarter_partition() {
        let v = call(
            polars__dt64_quarter,
            json!({"values": ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15", "2024-12-31"]}),
        );
        let q: Vec<i64> = v["quarter"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(q, vec![1, 2, 3, 4, 4]);
    }

    #[test]
    fn dt64_add_days_round_trip_via_diff() {
        let added = call(
            polars__dt64_add_days,
            json!({"values": ["2024-01-01 00:00:00"], "n": 30}),
        );
        let added_str = added["datetimes"][0].as_str().unwrap().to_string();
        let diff = call(
            polars__dt64_diff_days,
            json!({"a": [added_str], "b": ["2024-01-01 00:00:00"]}),
        );
        assert_eq!(diff["days"][0].as_i64().unwrap(), 30);
    }

    #[test]
    fn dt64_business_day_count_january_2024() {
        // Jan 1 (Mon) → Feb 1 (Thu), excluding end: 23 business days.
        let v = call(
            polars__dt64_business_day_count,
            json!({"start": "2024-01-01", "end": "2024-02-01"}),
        );
        assert_eq!(v["business_days"].as_u64().unwrap(), 23);
    }

    #[test]
    fn dt64_date_range_inclusive_count() {
        let v = call(
            polars__dt64_date_range,
            json!({"start": "2024-01-01", "end": "2024-01-05"}),
        );
        let n = v["dates"].as_array().unwrap().len();
        assert_eq!(n, 5);
    }
}
