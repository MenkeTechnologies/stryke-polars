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

`stryke-polars` ships as a local stryke package + a Rust cdylib (`libstryke_polars.{dylib,so}`) loaded on demand. The stryke side is a thin JSON-pipe wrapper; the heavy `polars`/`ndarray`/`nalgebra`/`rustfft` code lives in the cdylib and is loaded on first `use Polars`. Core stryke is never linked against any of them.

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

Working DataFrame / groupby examples live in `examples/` (`discover.stk`,
`groupby.stk`); the full per-family surface is exercised by the suites in
`t/`.

## [0x03] Surface

46 wrapper modules in `lib/`, 1,479 stryke-side fns total
(`grep -c '^fn ' lib/*.stk`), each calling a `polars__*` cdylib export:

| Module | Package | Fns |
|---|---|---|
| `Series.stk` | `Polars::Series` | 206 |
| `NdArray.stk` | `Polars::NdArray` | 124 |
| `DataFrame.stk` | `Polars::DataFrame` | 116 |
| `Ufunc.stk` | `Polars::Ufunc` | 70 |
| `Index.stk` | `Polars::Index` | 69 |
| `NdArrayExt.stk` | `Polars::NdArrayExt` | 68 |
| `UfuncExt.stk` | `Polars::UfuncExt` | 66 |
| `Masked.stk` | `Polars::Masked` | 61 |
| `DataFrameExt.stk` | `Polars::DataFrameExt` | 47 |
| `DateTime64.stk` | `Polars::DateTime64` | 42 |
| `Image.stk` | `Polars::Image` | 42 |
| `Categorical.stk` | `Polars::Categorical` | 35 |
| `Dist.stk` | `Polars::Dist` | 35 |
| `Signal.stk` | `Polars::Signal` | 35 |
| `Bit.stk` | `Polars::Bit` | 31 |
| `IO.stk` | `Polars::IO` | 29 |
| `Text.stk` | `Polars::Text` | 27 |
| `Stat.stk` | `Polars::Stat` | 24 |
| `Stattest.stk` | `Polars::Stattest` | 24 |
| `Misc.stk` | `Polars::Misc` | 23 |
| `Metric.stk` | `Polars::Metric` | 22 |
| `GroupBy.stk` | `Polars::GroupBy` | 21 |
| `Random.stk` | `Polars::Random` | 20 |
| `RandomExt.stk` | `Polars::RandomExt` | 20 |
| `Linalg.stk` | `Polars::Linalg` | 19 |
| `LinalgExt.stk` | `Polars::LinalgExt` | 18 |
| `Window.stk` | `Polars::Window` | 17 |
| `Fmt.stk` | `Polars::Fmt` | 14 |
| `Json.stk` | `Polars::Json` | 14 |
| `Graph.stk` | `Polars::Graph` | 13 |
| `TS.stk` | `Polars::TS` | 13 |
| `Geo.stk` | `Polars::Geo` | 12 |
| `Bool.stk` | `Polars::Bool` | 11 |
| `Sparse.stk` | `Polars::Sparse` | 15 |
| `PolynomialExt.stk` | `Polars::PolynomialExt` | 10 |
| `Set.stk` | `Polars::Set` | 9 |
| `FFT.stk` | `Polars::FFT` | 8 |
| `Encoding.stk` | `Polars::Encoding` | 7 |
| `Polynomial.stk` | `Polars::Polynomial` | 7 |
| `Interp.stk` | `Polars::Interp` | 6 |
| `Checksum.stk` | `Polars::Checksum` | 7 |
| `Cluster.stk` | `Polars::Cluster` | 5 |
| `FFTExt.stk` | `Polars::FFTExt` | 5 |
| `Hash.stk` | `Polars::Hash` | 5 |
| `Opt.stk` | `Polars::Opt` | 5 |
| `Polars.stk` | `Polars` (root: `version`, `_decode`) | 2 |

## [0x04] API Reference

Per-family `.stk` wrappers live in `lib/` — one module per family, listed
with fn counts in [\[0x03\]](#0x03-surface). Per-fn docs live inline as
`##` doc comments above each wrapper fn.

## [0x05] FFI Layer

Each `polars__*` export takes a single `*const c_char` (NUL-terminated JSON args) and returns a `*mut c_char` (NUL-terminated JSON result). The cdylib owns the returned allocation; the stryke side **must** release it via the cdylib-exported `stryke_free_cstring`. stryke's `rust_ffi::load_cdylib` wires this automatically.

JSON envelope on success is the per-fn shape (see [\[0x04\]](#0x04-api-reference)). JSON envelope on error is `{"error": "<message>"}`. Panics inside the cdylib are caught and surfaced as errors.

## [0x06] Backing Crates

| Subsystem | Backing crate(s) |
|---|---|
| DataFrame / Series / Index / pandas IO | `polars` (full feature set) |
| ndarray + ufuncs | `ndarray` + `rayon` |
| linalg | `nalgebra` |
| random | `rand` + `rand_distr` + `ndarray-rand` + `rand_chacha` |
| fft | `rustfft` + `realfft` |
| polynomial | hand-rolled on `ndarray` (recurrence formulas) |
| masked arrays | `ndarray` (mask vec parallel to data) |
| datetime64 / timedelta64 | `chrono` + `chrono-tz` |
| Decimal dtype | `rust_decimal` |

Parquet / Arrow IO routes through `stryke-arrow` to share a single arrow-rs link in-process — `stryke-polars` does not link `arrow-rs` directly to avoid `dlsym` conflicts.

## [0x07] Naming Convention

Stryke-side wrappers are namespaced packages — `use Polars::DataFrame`
gives `Polars::DataFrame::head`, `use Polars::Linalg` gives
`Polars::Linalg::*`, etc. (one package per `lib/*.stk` module).

cdylib-side FFI symbols are flat, prefixed `polars__` (double-underscore
namespace) plus a per-family verb prefix: `polars__df_<verb>` for
DataFrame, `polars__sr_<verb>` for Series, `polars__arr_<verb>` for
ndarray, `polars__np_<verb>` for ufuncs, `polars__linalg_<verb>` /
`polars__rand_<verb>` / `polars__fft_<verb>` / `polars__poly_<verb>` /
`polars__ma_<verb>` / `polars__dt64_<verb>` for the namespaced families,
`polars__pd_read_<fmt>` / `polars__pd_to_<fmt>` for IO.

## [0x08] Phases

The surface landed in numbered phases (each phase one git commit / one CI
green / one release tag). The original P0–P5 plan — scaffold, DataFrame,
Series + Index + IO, groupby / accessors / Categorical, ndarray + ufuncs,
then linalg / random / fft / polynomial / masked / datetime64 — has
shipped, and the surface has since expanded well past it (image, signal,
distributions, stat tests, text, graph, geo, sparse, and more — see the
module table in [\[0x03\]](#0x03-surface)).

## [0x09] Tests

- `cargo test` — Rust-side unit tests per `src/*.rs` (each phase adds its own).
- `s test t/` — stryke-side integration tests against the installed cdylib.
- `tests/*.sh` — contract gates (final newline, badges, https links, h2 sections, shell-shebang, etc.) wired into CI.

Per-fn correctness is gated by reference checks against pandas/numpy where possible (numerical tolerance for floats, exact match for ints/bools/strings).

## [0x0A] Dev Workflow

```sh
make release        # cargo build --release (default target)
make test           # cargo test + stryke t/
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
│   ├── lib.rs              # FFI plumbing + version export
│   ├── df.rs               # DataFrame + .str/.dt accessors
│   ├── sr.rs / more_sr.rs  # Series
│   ├── idx.rs              # Index
│   ├── io.rs               # pandas IO
│   ├── cat.rs              # Categorical
│   ├── nd.rs / more_nd.rs  # ndarray, ufuncs, linalg, random, fft, polynomial
│   ├── ma.rs               # masked arrays
│   ├── dt64.rs             # datetime64 / timedelta64
│   ├── img.rs              # image ops
│   ├── signal.rs           # signal processing + windows
│   ├── stattest.rs         # stat tests, distributions, interpolation
│   └── extras{,2,3,4}.rs   # groupby, stat, set, bool, cluster, geo, graph, text, json, … expansion families
├── lib/                    # 46 stryke-side .stk wrapper modules (see [0x03])
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
