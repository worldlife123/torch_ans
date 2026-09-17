# Benchmark Status

Local measurements of the rANS operators on one machine, comparing **v0.2.1** and **v0.3.0** (measured 2026-09-11).

> These numbers are produced **locally and committed by hand**; the CI job that used to regenerate this file on every full-matrix run has been removed. CI runners are virtualised CPUs whose throughput varies by tens of percent between runs, which made the table useless for comparing releases (it could only ever show virtualisation noise, not a regression).

## Environment

| item | value |
|---|---|
| CPU | Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz (6C/12T) |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| torch | 2.11.0+cu130 |
| Python | 3.11.5 |
| torch intra-op threads | 6 |
| workload | 8 distributions, alphabet 255 + bypass tail, `freq_precision=15`, uniform-ish random PMF, in-range symbols |
| batch | 48 streams x 8192 symbols (CPU), 2048 streams x 8192 symbols (CUDA) |
| method | best of 3 passes x 10 stream clones, stream clones and table builds outside the timed region, every configuration verified to round-trip before it is timed |

All throughput numbers are **Msymbol/s** (larger is better); `gain` is `v0.3.0 / v0.2.1`. `n/a` = the operator does not exist in that release, `broken` = it exists but decoded to the wrong symbols (the benchmark refuses to time those).

## Highlights

- **CPU, unchanged configuration** (1-way, binary search): decode 1.27×–1.44×, encode 1.03×–1.24×.
- **CPU, 4-way interleaved**: decode 2.40×–2.75×, encode 1.45×–1.83×. Interleaving only pays off in v0.3.0: 4-way/1-way decode is 0.88×–0.91× in v0.2.1 (a slowdown) and 1.68×–1.76× in v0.3.0.
- **CUDA, 1-way**: essentially unchanged — decode 1.00×–1.01×, encode 1.00×–1.01×.
- **CUDA, 1-way dense inverse CDF** is the one regression: 0.90×–0.92× (v0.3.0 stages the table differently); the sparse table (`inverse_cdf_precision="auto"`) is the replacement and lands back at ~1.0× of the v0.2.1 dense number while using 1/256 of the memory.
- **CUDA, 32-way warp kernels (new)**: 27845 Msym/s encode, 25697 Msym/s decode — an order of magnitude above the 1-way path.
- **CPU `pmf_to_cdf`**: 0.080 ms → 0.007 ms (10.9×).
- **CPU `invcdf_dense`**: 54.643 ms → 0.088 ms (617.8×).
- **CUDA `pmf_to_cdf`**: 0.252 ms → 0.042 ms (6.0×).
- **CUDA `invcdf_dense`**: 1.642 ms → 0.027 ms (61.7×).
- **High-level API, best configuration on CPU**: decode 32.6 → 86.7 Msym/s (2.66×), encode 91.3 → 160.1 Msym/s.
- **High-level API, best configuration on CUDA**: decode 1.7 → 8.7 Msym/s (5.27×), encode 1.8 → 6.9 Msym/s.

## Raw operators — CPU

| family | interleave | symbol lookup | v0.2.1 encode | v0.3.0 encode | enc gain | v0.2.1 decode | v0.3.0 decode | dec gain |
|---|---|---|---|---|---|---|---|---|
| `rans64` | 1 | binary search | 368.5 | 458.5 | 1.24× | 192.4 | 251.7 | 1.31× |
| `rans64` | 1 | inverse CDF (dense) | 369.5 | 455.8 | 1.23× | 266.5 | 233.6 | 0.88× |
| `rans64` | 1 | inverse CDF (sparse) | — (n/a) | 463.1 | — | — (n/a) | 464.6 | — |
| `rans64` | 1 | alias sampling | 219.9 | 226.1 | 1.03× | 289.5 | 319.3 | 1.10× |
| `rans64` | 4 | binary search | 383.4 | 701.8 | 1.83× | 175.9 | 422.3 | 2.40× |
| `rans64` | 4 | inverse CDF (dense) | — (n/a) | 831.0 | — | — (n/a) | 455.6 | — |
| `rans64` | 4 | inverse CDF (sparse) | — (n/a) | 732.6 | — | — (n/a) | 559.4 | — |
| `rans64` | 4 | alias sampling | — (n/a) | 461.8 | — | — (n/a) | 364.0 | — |
| `rans32` | 1 | binary search | 541.5 | 557.3 | 1.03× | 202.6 | 257.1 | 1.27× |
| `rans32` | 1 | inverse CDF (dense) | 542.3 | 561.8 | 1.04× | 265.5 | 235.6 | 0.89× |
| `rans32` | 1 | inverse CDF (sparse) | — (n/a) | 545.6 | — | — (n/a) | 467.6 | — |
| `rans32` | 1 | alias sampling | 245.7 | 260.0 | 1.06× | 281.3 | 309.1 | 1.10× |
| `rans32` | 4 | binary search | 577.5 | 909.6 | 1.58× | 179.1 | 452.4 | 2.53× |
| `rans32` | 4 | inverse CDF (dense) | 580.7 | 911.4 | 1.57× | 334.1 | 461.5 | 1.38× |
| `rans32` | 4 | inverse CDF (sparse) | — (n/a) | 896.5 | — | — (n/a) | 606.7 | — |
| `rans32` | 4 | alias sampling | — (n/a) | 556.9 | — | — (n/a) | 390.6 | — |
| `rans32_16` | 1 | binary search | 504.3 | 539.6 | 1.07× | 182.4 | 263.1 | 1.44× |
| `rans32_16` | 1 | inverse CDF (dense) | 491.6 | 575.9 | 1.17× | 276.9 | 241.4 | 0.87× |
| `rans32_16` | 1 | inverse CDF (sparse) | — (n/a) | 575.6 | — | — (n/a) | 484.1 | — |
| `rans32_16` | 1 | alias sampling | 238.2 | 247.5 | 1.04× | 295.6 | 318.3 | 1.08× |
| `rans32_16` | 4 | binary search | 612.0 | 885.2 | 1.45× | 165.7 | 456.0 | 2.75× |
| `rans32_16` | 4 | inverse CDF (dense) | 606.9 | 884.6 | 1.46× | 369.6 | 467.4 | 1.26× |
| `rans32_16` | 4 | inverse CDF (sparse) | — (n/a) | 878.1 | — | — (n/a) | 550.1 | — |
| `rans32_16` | 4 | alias sampling | — (n/a) | 544.6 | — | — (n/a) | 375.8 | — |
| `rans32_16` | 32 | binary search | — (n/a) | 478.9 | — | — (n/a) | 212.6 | — |
| `rans32_16` | 32 | inverse CDF (dense) | — (n/a) | 480.2 | — | — (n/a) | 238.1 | — |
| `rans32_16` | 32 | inverse CDF (sparse) | — (n/a) | 477.4 | — | — (n/a) | 247.6 | — |

## Raw operators — CUDA

| family | interleave | symbol lookup | v0.2.1 encode | v0.3.0 encode | enc gain | v0.2.1 decode | v0.3.0 decode | dec gain |
|---|---|---|---|---|---|---|---|---|
| `rans64` | 1 | binary search | 1403.4 | 1407.9 | 1.00× | 1321.7 | 1332.4 | 1.01× |
| `rans64` | 1 | inverse CDF (dense) | 1403.3 | 1396.5 | 1.00× | 2119.0 | 1913.3 | 0.90× |
| `rans64` | 1 | inverse CDF (sparse) | — (n/a) | 1390.3 | — | — (n/a) | 2096.0 | — |
| `rans64` | 1 | alias sampling | 1197.6 | 1166.1 | 0.97× | 1614.5 | 1627.5 | 1.01× |
| `rans32` | 1 | binary search | 1695.6 | 1704.9 | 1.01× | 1261.8 | 1261.7 | 1.00× |
| `rans32` | 1 | inverse CDF (dense) | 1695.3 | 1705.6 | 1.01× | 2010.2 | 1856.8 | 0.92× |
| `rans32` | 1 | inverse CDF (sparse) | — (n/a) | 1705.2 | — | — (n/a) | 2004.7 | — |
| `rans32` | 1 | alias sampling | 1386.1 | 1389.3 | 1.00× | 1530.8 | 1531.3 | 1.00× |
| `rans32_16` | 1 | binary search | 1994.6 | 1992.2 | 1.00× | 1323.8 | 1334.5 | 1.01× |
| `rans32_16` | 1 | inverse CDF (dense) | 1994.6 | 1974.2 | 0.99× | 2156.9 | 1956.5 | 0.91× |
| `rans32_16` | 1 | inverse CDF (sparse) | — (n/a) | 1974.1 | — | — (n/a) | 2125.9 | — |
| `rans32_16` | 1 | alias sampling | 1562.3 | 1545.0 | 0.99× | 1623.5 | 1623.8 | 1.00× |
| `rans32_16` | 4 | binary search | — (broken) | 9374.1 | — | — (broken) | 7076.1 | — |
| `rans32_16` | 4 | inverse CDF (dense) | — (broken) | 9401.9 | — | — (broken) | 7242.4 | — |
| `rans32_16` | 4 | inverse CDF (sparse) | — (n/a) | 9380.9 | — | — (n/a) | 8494.1 | — |
| `rans32_16` | 4 | alias sampling | — (n/a) | 6800.7 | — | — (n/a) | 6053.8 | — |
| `rans32_16` | 32 | binary search | — (n/a) | 27845.3 | — | — (n/a) | 25697.1 | — |
| `rans32_16` | 32 | inverse CDF (dense) | — (n/a) | 27901.9 | — | — (n/a) | 22966.4 | — |
| `rans32_16` | 32 | inverse CDF (sparse) | — (n/a) | 27834.1 | — | — (n/a) | 26366.5 | — |

## Parameter build (`init_params` hot path)

Latency in **ms** (smaller is better); `gain` is `v0.2.1 / v0.3.0`.

### CPU

| step | table | v0.2.1 | v0.3.0 | gain |
|---|---|---|---|---|
| `pmf_to_cdf` |  | 0.080 | 0.007 | 10.88× |
| `invcdf_dense` | q=15 | 54.643 | 0.088 | 617.81× |
| `invcdf_sparse` | q=7 | n/a | 0.005 | — |

### CUDA

| step | table | v0.2.1 | v0.3.0 | gain |
|---|---|---|---|---|
| `pmf_to_cdf` |  | 0.252 | 0.042 | 6.02× |
| `invcdf_dense` | q=15 | 1.642 | 0.027 | 61.69× |
| `invcdf_sparse` | q=7 | n/a | 0.025 | — |

## High-level API, end to end

`TorchANSInterface.encode_with_indexes` / `decode_with_indexes` including the stream <-> bytes conversion, i.e. what an unchanged user program measures. 48 x 8192 symbols (CPU) / 256 x 8192 (CUDA).

### CPU

| configuration | v0.2.1 encode | v0.3.0 encode | enc gain | v0.2.1 decode | v0.3.0 decode | dec gain |
|---|---|---|---|---|---|---|
| rans64 default | 71.2 | 100.2 | 1.41× | 32.2 | 41.2 | 1.28× |
| rans32_16 default | 91.3 | 104.0 | 1.14× | 32.6 | 40.6 | 1.24× |
| rans32_16 i4 + auto invcdf | — | 160.1 | — | — | 86.7 | — |

### CUDA

| configuration | v0.2.1 encode | v0.3.0 encode | enc gain | v0.2.1 decode | v0.3.0 decode | dec gain |
|---|---|---|---|---|---|---|
| rans64 default | 1.8 | 1.9 | 1.01× | 1.7 | 1.7 | 1.00× |
| rans32_16 default | 2.1 | 2.5 | 1.16× | 1.6 | 1.7 | 1.06× |
| rans32_16 i4 + auto invcdf | — | 6.9 | — | — | 8.7 | — |

> Note: in v0.2.1 `impl="rans32_16"` was silently shadowed by the `"rans32"` prefix match, so the v0.2.1 `rans32_16 default` row actually ran the `rans32` operators. The end-to-end numbers are also dominated by the stream <-> bytes conversion on both releases, so they move much less than the raw operators.

## Reproducing

The raw JSON behind the tables above is committed under `benchmarks/`.

```bash
# current tree
pip install . --no-build-isolation            # WITH_CUDA=1 for CUDA
python scripts/bench_version_matrix.py --version-label 0.3.0 --best-effort \
    --json benchmarks/v0.3.0-codec-params.json
python scripts/bench_version_matrix.py --version-label 0.3.0 --best-effort \
    --rows-cuda 256 --sections high_level \
    --json benchmarks/v0.3.0-high_level.json

# v0.2.1 baseline (checked out and built in a worktree)
git worktree add --detach .bench/v0.2.1 v0.2.1
cd .bench/v0.2.1 && WITH_CUDA=1 python setup.py build_ext --inplace && cd -
cp scripts/bench_version_matrix.py .bench/v0.2.1/scripts/
PYTHONPATH=$PWD/.bench/v0.2.1 \
    python .bench/v0.2.1/scripts/bench_version_matrix.py \
        --version-label 0.2.1 --json benchmarks/v0.2.1-codec-params.json

python scripts/generate_benchmark_report.py \
    --baseline benchmarks/v0.2.1-codec-params.json benchmarks/v0.2.1-high_level.json \
    --candidate benchmarks/v0.3.0-codec-params.json benchmarks/v0.3.0-high_level.json \
    --output benchmark_status.md
```

A version that cannot be built any more can be skipped; the report then shows `—` in that column and no gain.

---

# tANS vs rANS (measured 2026-09-18)

A second, separate measurement: the **tANS families against the rANS ones at matched
parameters**, with `scripts/bench_tans_vs_rans.py` (raw results:
`benchmarks/tans_vs_rans_cpu.json`). Same rules as above — local, committed by hand,
every configuration round-trip checked before it is timed.

## Environment

| item | value |
|---|---|
| CPU | Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz (6C/12T) |
| torch | 2.11.0+cu130 (CPU-only extension: this machine's CUDA toolchain does not build) |
| Python | 3.11.5 |
| torch intra-op threads | 1 (isolates the out-of-order effect of interleaving) |
| workload | 1,048,576 symbols, 256 distributions, 8 parallel states, bypass on, symbols drawn **from** the per-distribution pmf |
| method | high-level API (`encode_with_indexes` / `decode_with_indexes`), best of 5 |

Throughput is **MB/s of symbols coded** (1 byte per symbol). `H` is the mean
per-distribution entropy and `over` is `bits/symbol - H`, i.e. what the coder loses
against the distribution it is given — the column that makes "faster" comparable
across families. `table KB` is the per-distribution state the impl carries (for tANS
that *is* the spread table, `3 * 2**freq_precision + 2 * cdf_row` int32 words).

## Selected results (alphabet 64; alphabet 256 behaves the same way)

| impl | ilv | fp | enc MB/s | dec MB/s | bits/sym | H | over | table KB |
|---|---|---|---|---|---|---|---|---|
| tans32 | 1 | 7 | 66.0 | 86.6 | 5.877 | 5.847 | 0.030 | 516 |
| tans32 | 8 | 7 | 83.5 | 108.3 | 5.879 | 5.847 | 0.032 | 516 |
| rans32 | 1 | 7 | 101.5 | 45.1 | 5.863 | 5.847 | 0.017 | 66 |
| rans32 | 8 | 7 | 111.3 | 70.4 | 5.865 | 5.847 | 0.018 | 66 |
| tans32 | 1 | 10 | 46.2 | 50.1 | 5.734 | 5.730 | 0.004 | 3 204 |
| tans32 | 8 | 10 | 75.5 | 91.0 | 5.736 | 5.730 | 0.005 | 3 204 |
| rans32 | 1 | 10 | 96.3 | 44.8 | 5.733 | 5.730 | 0.002 | 66 |
| rans32 | 8 | 10 | 108.2 | 69.6 | 5.734 | 5.730 | 0.004 | 66 |
| tans32 | 1 | 15 | 15.3 | 13.6 | 5.725 | 5.724 | 0.001 | 98 436 |
| tans32 | 8 | 15 | 39.7 | 29.7 | 5.727 | 5.724 | 0.003 | 98 436 |
| rans32 | 1 | 15 | 94.3 | 44.7 | 5.725 | 5.724 | 0.001 | 66 |
| rans32 | 8 | 15 | 108.7 | 69.6 | 5.726 | 5.724 | 0.002 | 66 |

`alphabet=256, fp=15` (the worst case for tANS, 96 MB of tables) is the same picture:
tans32 8-way 38.4 MB/s encode / 29.7 MB/s decode against rans32 8-way 103.9 / 58.3.

## Findings

- **Decode is what tANS is for.** Against the fastest rANS family at the same
  `freq_precision`, tANS decodes **1.5×–1.9× faster at `freq_precision=7`** (one table
  access instead of a search) and ~1.0×–1.25× at 10; from 12 upwards it loses, and at 15
  with 256 distributions it is 3×–4× *slower* — the table no longer fits any cache level
  (98 MB against rANS' 258 KB) and every symbol is a cache miss.
- **Encode is rANS' side**, at every precision: tANS reaches 0.65×–0.75× of the fastest
  rANS at `freq_precision=7`, 0.48×–0.66× at 10 and 0.16×–0.34× at 15. The table-driven
  encode is not cheaper than rANS' division plus renormalization on a CPU.
- **Interleaving helps tANS more than rANS when the table is cache-bound**: up to
  **2.6× encode / 2.2× decode at `freq_precision=15`, 8 lanes** (rANS: 1.3×–1.5×), because
  the lanes overlap exactly the cache misses that otherwise serialize. Where the table
  fits (`freq_precision=7`), the gain shrinks to ~1.2×–1.25× — the same order as rANS.
  The three tANS families track each other within noise at every configuration.
- **Compression**: at `freq_precision=7` tANS loses ~0.030 bit/symbol to the spread
  against rANS' ~0.017; by 12–15 both are within 0.002–0.005 of the entropy. So the
  tANS compression penalty is real but small, and it is a penalty of the *small*
  `freq_precision` — which is also the only regime where tANS is fast.

**Rule of thumb:** `tans8` / `tans16` / `tans32` with `freq_precision <= 10` and a few
dozen to a few hundred distributions, for decode-heavy work; a rANS family with
`num_interleaves=4` (plus `inverse_cdf_precision="auto"`) otherwise. `tans8` (cap 7) is
the pick when there are thousands of distributions, since its table is 1.5 kB each.

Reproduce with:

```bash
pip install . --no-build-isolation   # the scripts import the *installed* torch_ans
python scripts/bench_tans_vs_rans.py --json benchmarks/tans_vs_rans_cpu.json
```

## Supplement: `tans8` at `freq_precision=7`

`tans8` is the small-table tier (its `freq_precision` cap *is* 7, i.e. 128 spread slots
per distribution) and its whole point is many distributions, so the main sweep — which
only had `alphabet=64` at that precision, and which cannot code `alphabet=256` at all
(257 symbols into 128 slots) — is not enough to judge it. Measured separately
(`benchmarks/tans8_fp7_cpu.json`, `benchmarks/tans8_fp7_many_dists_cpu.json`, same
environment as above, `freq_precision=7`, bypass on):

### 1 M symbols, 256 distributions

| alphabet | impl | ilv | enc MB/s | dec MB/s | bits/sym | H | over | table KB |
|---|---|---|---|---|---|---|---|---|
| 16 | `tans8` | 1 | 78.0 | 84.2 | 3.757 | 3.736 | 0.020 | 420 |
| 16 | `tans8` | 8 | **94.7** | **102.8** | 3.758 | 3.736 | 0.022 | 420 |
| 16 | `rans32_16` | 1 | 91.1 | 63.3 | 3.752 | 3.736 | 0.016 | 18 |
| 16 | `rans32_16` | 8 | 112.1 | 81.3 | 3.753 | 3.736 | 0.017 | 18 |
| 32 | `tans8` | 1 | 76.3 | 86.1 | 4.799 | 4.775 | 0.024 | 452 |
| 32 | `tans8` | 8 | 91.1 | 101.5 | 4.800 | 4.775 | 0.026 | 452 |
| 32 | `rans32_16` | 1 | 91.7 | 54.7 | 4.790 | 4.775 | 0.015 | 34 |
| 32 | `rans32_16` | 8 | 107.6 | 73.7 | 4.791 | 4.775 | 0.016 | 34 |
| 64 | `tans8` | 1 | 70.4 | 88.8 | 5.877 | 5.847 | 0.030 | 516 |
| 64 | `tans8` | 8 | 85.1 | 99.1 | 5.879 | 5.847 | 0.032 | 516 |
| 64 | `rans32_16` | 1 | 91.8 | 46.8 | 5.863 | 5.847 | 0.017 | 66 |
| 64 | `rans32_16` | 8 | 102.6 | 68.1 | 5.865 | 5.847 | 0.018 | 66 |
| 127 | `tans8` | 1 | 66.6 | 94.1 | 7.001 | 6.984 | 0.017 | 642 |
| 127 | `tans8` | 8 | 85.6 | **102.8** | 7.003 | 6.984 | 0.018 | 642 |
| 127 | `rans32` | 1 | 116.2 | 44.5 | 7.000 | 6.984 | 0.016 | 129 |
| 127 | `rans32` | 8 | 131.2 | 80.7 | 7.002 | 6.984 | 0.018 | 129 |

### 1 M symbols, 4096 distributions (the regime `tans8` is for)

| alphabet | impl | ilv | enc MB/s | dec MB/s | bits/sym | over | table KB |
|---|---|---|---|---|---|---|---|
| 32 | `tans8` | 1 | 41.4 | 41.9 | 4.798 | 0.024 | 7 232 |
| 32 | `tans8` | 8 | 65.3 | **74.4** | 4.800 | 0.026 | 7 232 |
| 32 | `rans32_16` | 1 | 72.7 | 38.6 | 4.790 | 0.016 | 544 |
| 32 | `rans32_16` | 8 | 83.2 | 56.6 | 4.791 | 0.017 | 544 |
| 64 | `tans8` | 1 | 40.5 | 41.3 | 5.876 | 0.031 | 8 256 |
| 64 | `tans8` | 8 | 62.5 | 71.9 | 5.877 | 0.032 | 8 256 |
| 64 | `rans32_16` | 1 | 70.3 | 28.1 | 5.862 | 0.017 | 1 056 |
| 64 | `rans32_16` | 8 | 78.9 | 49.1 | 5.863 | 0.018 | 1 056 |

What it adds:

- **`num_interleaves=8` on `tans8` (fp=7, 256 distributions): 85–95 MB/s encode,
  99–103 MB/s decode**, against 67–78 / 84–94 at one lane — a 1.19–1.28× encode and
  1.09–1.22× decode gain. Decode is the number that matters here: at one lane `tans8`
  decodes **1.33×–2.11× as fast as the fastest rANS at the same precision** (2.11× at
  `alphabet=127`, where the rANS search is longest), and at eight lanes it still beats
  the 8-lane rANS decode by **1.26×–1.46×**.
- **Encode stays behind** (0.57–0.85× of rANS at every alphabet and lane count), which is
  the same story as the main sweep.
- **The `tans8` cap is not a handicap in its regime**: more distributions cost both
  families throughput (more table/cdf rows to touch), and the tANS interleave gain
  *grows* to **1.55× encode / 1.74× decode at 4096 distributions** — the same
  cache-miss-overlap effect the main sweep showed at `freq_precision=15`, reached here at
  a table size `tans8` can still afford (`8 MB` for 4096 distributions, 1.5 kB each,
  against rANS' `1 MB` — and against ~40 kB per distribution for `tans32`).
- Compression: `tans8` at `fp=7` carries 0.017–0.032 bit/symbol of overhead against rANS'
  0.015–0.018 at the same precision. It is the price of the 128-slot spread: it peaks at
  `alphabet=64` (~0.03) and drops back at 127, where 127 symbols in 128 slots is nearly
  uniform by construction. The lane count does not change it (the coded size moves by
  <0.003 bit/symbol between 1 and 8 lanes).

Reproduce with:

```bash
python scripts/bench_tans_vs_rans.py --precision 7 --alphabet 16 32 64 127 \
    --impl tans8 tans16 rans32 rans32_16 --json benchmarks/tans8_fp7_cpu.json
python scripts/bench_tans_vs_rans.py --dists 4096 --precision 7 --alphabet 32 64 \
    --interleaves 1 8 --impl tans8 rans32_16 \
    --json benchmarks/tans8_fp7_many_dists_cpu.json
```

