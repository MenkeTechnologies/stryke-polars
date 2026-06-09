```
 ███████╗████████╗██████╗ ██╗   ██╗██╗  ██╗███████╗
 ██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝██║ ██╔╝██╔════╝
 ███████╗   ██║   ██████╔╝ ╚████╔╝ █████╔╝ █████╗
 ╚════██║   ██║   ██╔══██╗  ╚██╔╝  ██╔═██╗ ██╔══╝
 ███████║   ██║   ██║  ██║   ██║   ██║  ██╗███████╗
 ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
                  [ p o l a r s ]
```

[![CI](https://github.com/MenkeTechnologies/stryke-polars/actions/workflows/ci.yml/badge.svg)](https://github.com/MenkeTechnologies/stryke-polars/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![stryke](https://img.shields.io/badge/stryke-package-cyan.svg)](https://github.com/MenkeTechnologies/strykelang)

### `[POLARS + NDARRAY + LINALG + FFT + RANDOM // STRYKE PACKAGE]`

> *"The full pandas + numpy surface in one cdylib. No core bloat."*

pandas DataFrame + Series + Index + IO, numpy ndarray + ufuncs + linalg + random + fft + polynomial + masked arrays + datetime64 — all in a single cdylib, dlopened in-process by stryke via `use Polars`. Opt-in package, kept out of the stryke core binary so the daily-driver install stays slim. Created by MenkeTechnologies.

### [`strykelang`](https://github.com/MenkeTechnologies/strykelang) &middot; [`MenkeTechnologiesMeta`](https://github.com/MenkeTechnologies/MenkeTechnologiesMeta) · [`stryke-arrow`](https://github.com/MenkeTechnologies/stryke-arrow) · [`stryke-duckdb`](https://github.com/MenkeTechnologies/stryke-duckdb) · [`stryke-parquet`](https://github.com/MenkeTechnologies/stryke-parquet)

---

## Table of Contents

- [\[0x00\] Why a Package, Not a Builtin](#0x00-why-a-package-not-a-builtin)
- [\[0x01\] Install](#0x01-install)
- [\[0x02\] Quick Start](#0x02-quick-start)
- [\[0x03\] Surface](#0x03-surface)
- [\[0x04\] API Reference](#0x04-api-reference)
- [\[0x05\] FFI Layer](#0x05-ffi-layer)
- [\[0x06\] Backing Crates](#0x06-backing-crates)
- [\[0x07\] Naming Convention](#0x07-naming-convention)
- [\[0x08\] Phases](#0x08-phases)
- [\[0x09\] Tests](#0x09-tests)
- [\[0x0A\] Dev Workflow](#0x0a-dev-workflow)
- [\[0x0B\] Layout](#0x0b-layout)
- [\[0xFF\] License](#0xff-license)

---

## [0x00] Why a Package, Not a Builtin

stryke's core stays small on purpose — most one-liner / awk-replacement work doesn't need 200 transitive crates of pandas + numpy machinery. The full DataFrame + ndarray + linalg + FFT surface hits a different scale:

| Tier | Properties | This package |
|---|---|---|
| Core builtins (~40 MB stryke) | small deps, used everywhere | string, math, regex, parallel ops, scipy-class math |
| Package tier (opt-in) | heavy deps, narrow use cases | parquet, arrow, big-ML, cloud SDKs, **full pandas + numpy** |

`stryke-polars` ships as a local stryke package + a Rust cdylib (`libstryke_polars.{dylib,so}`) loaded on demand. The stryke side is a thin JSON-pipe wrapper; the heavy `polars`/`ndarray`/`ndarray-linalg`/`nalgebra`/`rustfft` code lives in the cdylib and is loaded on first `use Polars`. Core stryke is never linked against any of them.

## [0x01] Install

From source (development):

```sh
git clone https://github.com/MenkeTechnologies/stryke-polars
cd stryke-polars
make install        # cargo build --release && s pkg install -g .
```

From a published GitHub release:

```sh
s pkg install -g github:MenkeTechnologies/stryke-polars
```

## [0x02] Quick Start

```perl
use Polars

my $v = Polars::version()
say "stryke-polars $v->{version} (polars $v->{polars}, ndarray $v->{ndarray})"
```

Real DataFrame / ndarray / linalg / random / FFT examples land per phase — see [\[0x08\]](#0x08-phases).

## [0x03] Surface

| Family | Stryke prefix | cdylib prefix | Target fns |
|---|---|---|---|
| pandas DataFrame | `df_*` | `polars__df_*` | ~280 |
| pandas Series | `sr_*` | `polars__sr_*` | ~250 |
| pandas Index family | `idx_*` | `polars__idx_*` | ~80 |
| pandas IO | `pd_read_*` / `pd_to_*` | `polars__pd_*` | ~30 |
| pandas groupby / rolling / resample / EWM | `df_*` (verb forms) | `polars__df_*` | ~150 |
| pandas `.str.*` accessor | `sr_str_*` | `polars__sr_str_*` | ~50 |
| pandas `.dt.*` accessor | `sr_dt_*` | `polars__sr_dt_*` | ~50 |
| pandas Categorical | `cat_*` | `polars__cat_*` | ~30 |
| numpy ndarray | `arr_*` | `polars__arr_*` | ~180 |
| numpy ufuncs | `np_*` | `polars__np_*` | ~70 |
| numpy linalg | `linalg_*` | `polars__linalg_*` | ~50 |
| numpy random | `rand_*` | `polars__rand_*` | ~80 |
| numpy fft | `fft_*` | `polars__fft_*` | ~20 |
| numpy polynomial | `poly_*` | `polars__poly_*` | ~40 |
| numpy masked | `ma_*` | `polars__ma_*` | ~80 |
| numpy datetime64 / timedelta64 | `dt64_*` | `polars__dt64_*` | ~30 |
| **Total** | | | **~1470** |

## [0x04] API Reference

Per-family `.stk` wrappers live in `lib/`:

| File | Wraps |
|---|---|
| `lib/Polars.stk` | Root: `Polars::version`, `Polars::_decode` |
| `lib/DataFrame.stk` | `df_*` |
| `lib/Series.stk` | `sr_*` |
| `lib/Index.stk` | `idx_*` |
| `lib/IO.stk` | `pd_read_*` / `pd_to_*` |
| `lib/NdArray.stk` | `arr_*` |
| `lib/Ufunc.stk` | `np_*` |
| `lib/Linalg.stk` | `linalg_*` |
| `lib/Random.stk` | `rand_*` |
| `lib/FFT.stk` | `fft_*` |
| `lib/Polynomial.stk` | `poly_*` |
| `lib/Masked.stk` | `ma_*` |
| `lib/DateTime64.stk` | `dt64_*` |

Per-fn docs land alongside each export as it's added per phase.

## [0x05] FFI Layer

Each `polars__*` export takes a single `*const c_char` (NUL-terminated JSON args) and returns a `*mut c_char` (NUL-terminated JSON result). The cdylib owns the returned allocation; the stryke side **must** release it via the cdylib-exported `stryke_free_cstring`. stryke's `rust_ffi::load_cdylib` wires this automatically.

JSON envelope on success is the per-fn shape (see [\[0x04\]](#0x04-api-reference)). JSON envelope on error is `{"error": "<message>"}`. Panics inside the cdylib are caught and surfaced as errors.

## [0x06] Backing Crates

| Subsystem | Backing crate(s) |
|---|---|
| DataFrame / Series / Index / pandas IO | `polars` (full feature set) |
| ndarray + ufuncs | `ndarray` + `rayon` |
| linalg | `ndarray-linalg` (OpenBLAS), `nalgebra` |
| random | `rand` + `rand_distr` + `ndarray-rand` + `rand_chacha` |
| fft | `rustfft` + `realfft` |
| polynomial | hand-rolled on `ndarray` (recurrence formulas) |
| masked arrays | `ndarray` (mask vec parallel to data) |
| datetime64 / timedelta64 | `chrono` + `chrono-tz` |
| Decimal dtype | `rust_decimal` |

Parquet / Arrow IO routes through `stryke-arrow` to share a single arrow-rs link in-process — `stryke-polars` does not link `arrow-rs` directly to avoid `dlsym` conflicts.

## [0x07] Naming Convention

Stryke-side builtins follow the existing flat prefix pattern set by `strykelang/builtins_*.rs`:

- `df_<verb>` for DataFrame ops (matches the 57 existing `df_*` already in core).
- `sr_<verb>` for Series ops.
- `arr_<verb>` for ndarray ops.
- `np_<verb>` for ufuncs (matches Python `np.<verb>` muscle memory).
- `linalg_<verb>` / `rand_<verb>` / `fft_<verb>` / `poly_<verb>` / `ma_<verb>` / `dt64_<verb>` for namespaced families (matches Python `np.linalg.<verb>` etc.).
- `pd_read_<fmt>` / `pd_to_<fmt>` for IO (matches Python `pd.read_<fmt>`).

cdylib-side FFI symbols prefix every name with `polars__` (double-underscore namespace).

## [0x08] Phases

The full ~1470-fn surface lands in numbered phases. Each phase is one git commit / one CI green / one release tag.

| Phase | Scope | Files |
|---|---|---|
| **P0** | Scaffold (this commit): Cargo.toml, stryke.toml, src/lib.rs (FFI plumbing + `polars__version`), lib/*.stk wrappers, README, LICENSE, Makefile, CI, tests | this commit |
| **P1** | DataFrame full surface (`df_*` ~280) | `src/df.rs`, `lib/DataFrame.stk` |
| **P2** | Series + Index + IO | `src/sr.rs`, `src/idx.rs`, `src/io.rs` |
| **P3** | groupby + rolling + resample + EWM + str + dt + Categorical | `src/groupby.rs`, `src/accessors.rs`, `src/cat.rs` |
| **P4** | ndarray + ufuncs | `src/nd.rs`, `src/ufunc.rs` |
| **P5** | linalg + random + fft + polynomial + masked + datetime64 | `src/linalg.rs`, `src/random.rs`, `src/fft.rs`, `src/poly.rs`, `src/ma.rs`, `src/dt64.rs` |

## [0x09] Tests

- `cargo test` — Rust-side unit tests per `src/*.rs` (each phase adds its own).
- `s test t/` — stryke-side integration tests against the installed cdylib.
- `tests/*.sh` — contract gates (final newline, badges, https links, h2 sections, shell-shebang, etc.) wired into CI.

Per-fn correctness is gated by reference checks against pandas/numpy where possible (numerical tolerance for floats, exact match for ints/bools/strings).

## [0x0A] Dev Workflow

```sh
make debug          # cargo build (fast iter)
make test           # cargo test + stryke t/
make release        # cargo build --release
make install        # release + s pkg install -g .
cargo fmt --all     # required before every push (CI gate)
cargo clippy --all-targets --locked -- -D warnings
```

## [0x0B] Layout

```
stryke-polars/
├── Cargo.toml              # crate-type=cdylib, deps
├── stryke.toml             # package meta + FFI exports + scripts
├── src/
│   └── lib.rs              # FFI plumbing + per-phase fn impls
├── lib/                    # stryke-side .stk wrappers
│   ├── Polars.stk
│   ├── DataFrame.stk
│   ├── Series.stk
│   ├── Index.stk
│   ├── IO.stk
│   ├── NdArray.stk
│   ├── Ufunc.stk
│   ├── Linalg.stk
│   ├── Random.stk
│   ├── FFT.stk
│   ├── Polynomial.stk
│   ├── Masked.stk
│   └── DateTime64.stk
├── tests/                  # contract gates (shell)
├── t/                      # stryke integration tests
├── examples/
├── docs/                   # GitHub Pages content
├── .github/workflows/
│   ├── ci.yml
│   └── release.yml
├── Makefile
├── LICENSE
└── README.md
```

## [0xFF] License

MIT. See [LICENSE](LICENSE).
