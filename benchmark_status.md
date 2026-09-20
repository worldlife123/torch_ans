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

## tANS on CUDA (unreleased)

Measured 2026-09-18 with the CUDA kernels added in `torch_ans/tans_cuda.cu` (see
`CHANGELOG.md`): the three tANS families now code device tensors through the same
operator names and the same stream layout, single-lane (one thread per stream row),
byte-compatible with the CPU kernels. This section is the device counterpart of the
tANS sweep above; it answers three questions: how tANS compares to rANS on the GPU,
what the shared-memory table staging buys, and which launch block size to default to.

### Environment and method

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti (sm_75), driver 580.178.04 |
| torch | 2.7.1+cu118 (`cbench_pth27` environment; nvcc 11.8 + gcc-10, the only local toolkit that matches a torch build) |
| workload | 1,048,576 symbols, 1024 streams × 1024 symbols, `bypass_coding=True`, symbols drawn from the distributions |
| method | high-level API (`TorchANSInterface`), encode + decode, best of 10 reps (20 for the A/B), `torch.cuda.synchronize()` around every timed call, every configuration round-trip checked before it is timed |
| tANS kernels | `TORCH_ANS_TANS_SMEM_BUDGET` (staging budget, default 16 KiB/block), `TORCH_ANS_TANS_BLOCK_THREADS` (launch block size, unset = derived from the batch) |

`scripts/bench_tans_vs_rans.py --device cuda` runs the same matrix as the CPU sweep.
tANS was single-lane on CUDA when this section was measured, so `--interleaves` only
added rANS configurations (the `rans32_16` warp path, which is the fastest rANS
configuration on the device) and the tANS ones were reported as skipped. The interleaved
tANS kernels have since been added - script and measurements in *Interleaved tANS on
CUDA* below - so a rerun of the commands in *Reproduce* would measure them here too.

### tANS vs rANS, 256 distributions (tables in global memory)

The tables here are 420 KB (`fp=7`) to 3 MB (`fp=15`) per configuration, i.e. far
beyond any staging budget: `smem=no` on every row of
`benchmarks/tans_vs_rans_cuda.json`. Throughput in Msymbol/s; `bits/sym` is the coded
size over the 2^20 symbols (all rows are labelled `-` in the `blk` column, i.e. they use
the derived block size).

| alphabet | fp | tANS enc/dec | tANS bits/sym | rANS x1 enc/dec | rANS x1 bits/sym | fastest rANS | its enc/dec | its bits/sym | dec vs x1 | dec vs fastest |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 7 | 488 / 788 | 3.873 | 557 / 733 | 3.806 | `rans32_16` x32 | 705 / 1286 | 4.543 | 1.07× | 0.61× |
| 16 | 10 | 537 / 887 | 3.836 | 545 / 733 | 3.774 | `rans32_16` x32 | 695 / 1280 | 4.525 | 1.21× | 0.69× |
| 16 | 12 | 503 / 727 | 3.834 | 540 / 735 | 3.772 | `rans32_16` x32 | 679 / 1280 | 4.524 | 0.99× | 0.57× |
| 16 | 15 | 463 / 632 | 3.834 | 524 / 733 | 3.776 | `rans32_16` x32 | 676 / 1262 | 4.525 | 0.86× | 0.50× |
| 64 | 7 | 505 / 820 | 5.994 | 501 / 589 | 5.918 | `rans32_16` x32 | 677 / 1103 | 6.589 | 1.39× | 0.74× |
| 64 | 10 | 505 / 820 | 5.851 | 482 / 597 | 5.787 | `rans32_16` x32 | 644 / 1112 | 6.533 | 1.37× | 0.74× |
| 64 | 12 | 464 / 683 | 5.843 | 474 / 594 | 5.780 | `rans32_16` x32 | 593 / 1104 | 6.529 | 1.15× | 0.62× |
| 64 | 15 | 444 / 600 | 5.842 | 475 / 598 | 5.784 | `rans32_16` x32 | 618 / 1114 | 6.531 | 1.00× | 0.54× |
| 256 | 10 | 447 / 692 | 7.900 | 454 / 449 | 7.831 | `rans32_16` x32 | 569 / 820 | 8.551 | 1.54× | 0.84× |
| 256 | 12 | 428 / 589 | 7.848 | 435 / 451 | 7.784 | `rans32_16` x32 | 573 / 829 | 8.531 | 1.31× | 0.71× |
| 256 | 15 | 404 / 529 | 7.841 | 443 / 450 | 7.783 | `rans32_16` x32 | 575 / 875 | 8.531 | 1.18× | 0.60× |

- **Against single-lane rANS at the same `freq_precision`** (the like-for-like comparison
  - same quantization, same alphabet, only the coder differs) tANS decodes **1.0-1.5×
  faster from `alphabet 64` up** (1.39× at `fp=7`, 1.54× at alphabet 256 / `fp=10`) and
  is within ±5% of it on encode (0.85-1.05×) except at `alphabet 16 / fp=15`. At
  `alphabet 16` the table is small enough that rANS' cheaper per-symbol work wins on
  decode once the precision makes the search cheap.
- **Against the fastest rANS configuration on the device** (`rans32_16` with the 32-way
  warp path) tANS is behind in both directions (0.50-0.84× decode, 0.68-0.78× encode at
  the same precision). That path is what to use when throughput matters more than the
  coded size.
- **The warp path's speed costs ~0.6-0.8 bit/symbol on these rows** (4.543 vs 3.873
  bits/sym at `alphabet=16, fp=7`; 8.531 vs 7.841 at `alphabet=256, fp=15`): every one of
  its 32 lanes flushes its own 4-byte state per stream, i.e. 128 B per 1024-symbol row.
  Longer rows amortize it, 1024 symbols do not.
- tANS's own coded size matches single-lane rANS (within 0.06-0.07 bits/symbol, the
  table's quantization at these `freq_precision`s) - so at `alphabet >= 64` it is faster
  *and* as compact as single-lane rANS.

### Table staging A/B and launch block size

`benchmarks/tans_cuda_staging_off.json` and `benchmarks/tans_cuda_staging_on.json`: the
same configuration (8 distributions, `fp=7`, alphabet 16, `parallel_states=1024`), i.e. a
13.1 KB table that fits the 16 KiB budget, measured with the staging disabled
(`--smem-budget 0`, global memory, `smem=no`) and enabled (`--smem-budget 16384`,
`smem=yes`), each at three block sizes. Msymbol/s, `tans32`:

| block threads | blocks | global enc | global dec | staged enc | staged dec |
|---|---|---|---|---|---|
| 1 | 1024 | 544 | 905 | 252 | 356 |
| 8 | 128 | 588 | 972 | 595 | 964 |
| 64 | 16 | 592 | 870 | 567 | 869 |

`tans16`/`tans8` behave the same way (see the JSONs; e.g. `tans8` at 8 threads: 611/1020
global against 589/956 staged).

- **The staging copy is paid per block**, which is what the first row shows: with one
  thread per block the 13.1 KB table is copied 1024 times (13 MB of loads) next to ~1 MB
  of payload, and encode drops to 0.46×. With 8-64 threads per block the copy disappears
  into the noise and the two paths agree within run-to-run variation (588 vs 595 encode,
  972 vs 964 decode at 8 threads).
- **A table smaller than the work per row does gain from staging**: at 4 distributions
  (6.7 KB) with 64 streams × 16K symbols the staged path measured +3-5% on both directions
  at 2-8 threads per block and +23% encode at one thread (same script,
  `--dists 4 --parallel-states 64`; not committed as a JSON because it is the same
  script with two different `--dists` values).
- **A table comparable to the payload loses**: at 24 distributions (39.4 KB, still staged
  under `--smem-budget 49152`) the staged path was 6-20% slower than the global one at
  every block size (8/32/64), because the per-block copy competes with the coding itself.
- **Block size**: the optimum tracks "about 128 blocks", i.e. a couple per SM on this
  device - small batches want one or two threads per block, large ones want wide blocks.
  A fixed 64-thread default (the first version) put 1024 rows into 16 blocks, i.e. on a
  quarter of the SMs, and lost ~18% decode against 8 threads per block (measured sweep in
  the `tans_num_threads_per_block` comment in `tans_cuda.cu`: 256 rows → best 1-8 threads,
  1024 rows → best 2-8, 4096 rows → best 32-64). The kernel now derives the block size as
  `ceil(batch / 128)` clamped to `[1, 128]`, which reproduces those optima.
- Consequence for the default: the staging stays **enabled** (16 KiB budget), because in
  its regime (a small table, few distributions, a modest block count) it is a wash or a few
  percent up and never a large loss, and because it costs nothing when it is not triggered -
  which is the common case, as the 256-distribution table above shows.

### Low distribution counts: the regime the staging can reach

The 256-distribution tables above are 420 KB - 3 MB, so they can never be staged. This
subsection repeats the comparison where the whole table *does* fit the budget - 8 and 4
distributions - with the rANS families present as the baseline, so the staging question
("does `__shared__` pay?") and the coder question ("is tANS faster?") are answered on the
same rows. Raw results: `benchmarks/tans_vs_rans_cuda_dists8_smem_{on,off}.json`,
`benchmarks/tans_vs_rans_cuda_dists4_smem_{on,off}.json` and the long-row pair
`benchmarks/tans_vs_rans_cuda_longrows_smem_{on,off}.json`; all of them 1M symbols,
`reps=20`, table sizes as listed (staging needs the *whole* table inside the budget, so
`--smem-budget` is set high enough to reach the listed cases).

**8 distributions, 1024 rows × 1024 symbols, `--smem-budget 32768` vs `0`** (Msymbol/s;
"staging on/off" is the staged run over the global run, so 1.000 means the two agree - the
rows that are `staged=no` in *both* runs are the noise floor of that ratio):

| alphabet | fp | table KB | staged | tANS enc/dec | rANS x1 enc/dec | fastest rANS (`rans32_16` x32) | dec vs x1 | dec vs fastest | staging on/off enc | staging on/off dec |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 7 | 13.1 | yes | 543 / 856 | 559 / 743 | 719 / 1261 | 1.15× | 0.68× | 0.967 | 0.922 |
| 16 | 10 | 97.1 | no | 570 / 908 | 541 / 741 | 675 / 1266 | 1.22× | 0.72× | 1.007 | 0.989 |
| 16 | 12 | 385.1 | no | 546 / 902 | 521 / 739 | 655 / 1263 | 1.22× | 0.71× | 1.004 | 0.992 |
| 64 | 7 | 16.1 | yes | 503 / 833 | 479 / 631 | 629 / 1115 | 1.32× | 0.75× | 0.971 | 0.917 |
| 64 | 10 | 100.1 | no | 516 / 818 | 477 / 640 | 577 / 1102 | 1.28× | 0.74× | 0.977 | 0.978 |
| 64 | 12 | 388.1 | no | 501 / 827 | 488 / 646 | 595 / 1097 | 1.28× | 0.75× | 0.958 | 0.992 |

**4 distributions, 1024 rows × 1024 symbols, `--smem-budget 49152` vs `0`** (same shape,
reaching tables up to the 48 KiB per-block limit):

| alphabet | fp | table KB | staged | tANS enc/dec | rANS x1 enc/dec | fastest rANS (`rans32_16` x32) | dec vs x1 | dec vs fastest | staging on/off enc | staging on/off dec |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 7 | 6.3 | yes | 591 / 992 | 587 / 818 | 782 / 1500 | 1.21× | 0.66× | 0.997 | 0.983 |
| 8 | 8 | 12.3 | yes | 607 / 1072 | 575 / 825 | 762 / 1489 | 1.30× | 0.72× | 0.956 | 0.961 |
| 8 | 9 | 24.3 | yes | 588 / 991 | 576 / 827 | 738 / 1486 | 1.20× | 0.67× | 0.936 | 0.901 |
| 8 | 10 | 48.3 | no | 613 / 1082 | 562 / 823 | 713 / 1480 | 1.31× | 0.73× | 0.998 | 1.024 |
| 16 | 7 | 6.6 | yes | 584 / 1011 | 547 / 758 | 689 / 1304 | 1.33× | 0.78× | 0.984 | 0.982 |
| 16 | 8 | 12.6 | yes | 566 / 969 | 539 / 760 | 691 / 1306 | 1.28× | 0.74× | 0.972 | 0.955 |
| 16 | 9 | 24.6 | yes | 537 / 906 | 537 / 765 | 691 / 1316 | 1.18× | 0.69× | 0.915 | 0.904 |
| 16 | 10 | 48.6 | no | 589 / 982 | 532 / 766 | 669 / 1307 | 1.28× | 0.75× | 1.027 | 1.026 |

**4 distributions, 64 rows × 16384 symbols, `--smem-budget 16384` vs `0`** (few blocks,
long streams - the regime where a per-block copy is amortized):

| alphabet | fp | table KB | staged | tANS enc/dec | rANS x1 enc/dec | dec vs x1 | staging on/off enc | staging on/off dec |
|---|---|---|---|---|---|---|---|---|
| 8 | 7 | 6.3 | yes | 176 / 295 | 136 / 131 | 2.26× | 1.046 | 1.031 |
| 16 | 7 | 6.6 | yes | 173 / 284 | 136 / 122 | 2.32× | 1.054 | 1.036 |

Findings:

- **tANS keeps its advantage in the small-table regime.** Against single-lane rANS at the
  same `freq_precision` it decodes **1.15-1.33×** faster at 1024-symbol rows and **2.3×**
  faster with long rows (fewer, longer streams make the per-symbol cost dominate), while
  encoding is at parity (0.97-1.06×) and its coded size stays within 0.02-0.04 bit/symbol
  of rANS. The rANS 32-way warp path is still the fastest configuration in both directions
  (tANS reaches 0.66-0.78× of it) and costs ~0.6-0.8 bit/symbol more at these row lengths.
- **With many blocks (1024-symbol rows) the staging is neutral to slightly negative.** The
  staged rows are 0.92-0.97 of the global ones, i.e. inside or barely outside the band the
  two *non-staged* rows span (0.96-1.03). The cost is structural: the copy is per block
  (~128 blocks × 13-16 KB ≈ 1.7-2 MB of extra loads for ~1 MB of payload) and 6-24 KB of
  shared memory per block lowers occupancy, while L1 already serves a table that size. The
  deeper the staged table, the worse it gets (0.94/0.90 at 24 KB).
- **With few blocks and long streams the staging does pay**: 64 streams × 16K symbols
  staging a 6.6 KB table measured **+4.6-5.4% encode and +3.1-3.6% decode** — there the
  copy is amortized over 16K symbols per row and each block reuses the table a lot.
- The feature therefore stays enabled (16 KiB budget) for exactly the configurations its
  size arithmetic admits — a table small enough to fit is a wash or a few percent up — and
  the 256-distribution sweeps never trigger it at all. `--smem-budget 0` turns it off when
  a measurement or a workload prefers the global path.

### Reproduce

```bash
# the comparison table (256 distributions, table in global memory)
PYTHONPATH=$PWD python scripts/bench_tans_vs_rans.py --device cuda \
    --symbols 1048576 --parallel-states 1024 --dists 256 \
    --alphabet 16 64 256 --precision 7 10 12 15 \
    --interleaves 1 2 4 8 32 --reps 10 \
    --json benchmarks/tans_vs_rans_cuda.json

# the staging A/B (13.1 KB table, three block sizes each)
PYTHONPATH=$PWD python scripts/bench_tans_vs_rans.py --device cuda \
    --symbols 1048576 --parallel-states 1024 --dists 8 \
    --alphabet 16 --precision 7 --impl tans32 tans16 tans8 --interleaves 1 \
    --reps 20 --block-threads 1 8 64 --smem-budget 0 \
    --json benchmarks/tans_cuda_staging_off.json
PYTHONPATH=$PWD python scripts/bench_tans_vs_rans.py --device cuda \
    --symbols 1048576 --parallel-states 1024 --dists 8 \
    --alphabet 16 --precision 7 --impl tans32 tans16 tans8 --interleaves 1 \
    --reps 20 --block-threads 1 8 64 --smem-budget 16384 \
    --json benchmarks/tans_cuda_staging_on.json

# the low-distribution sweep, tANS *and* rANS, staging on/off (tables 6-48 KB)
for budget in 0 32768; do
  PYTHONPATH=$PWD python scripts/bench_tans_vs_rans.py --device cuda \
      --symbols 1048576 --parallel-states 1024 --dists 8 \
      --alphabet 16 64 --precision 7 10 12 --interleaves 1 2 4 8 32 \
      --impl tans32 tans16 tans8 rans64 rans32 rans32_16 --reps 20 \
      --smem-budget $budget \
      --json benchmarks/tans_vs_rans_cuda_dists8_smem_$([ $budget = 0 ] && echo off || echo on).json
done
for budget in 0 49152; do
  PYTHONPATH=$PWD python scripts/bench_tans_vs_rans.py --device cuda \
      --symbols 1048576 --parallel-states 1024 --dists 4 \
      --alphabet 8 16 --precision 7 8 9 10 --interleaves 1 2 4 8 32 \
      --impl tans32 tans16 tans8 rans64 rans32 rans32_16 --reps 20 \
      --smem-budget $budget \
      --json benchmarks/tans_vs_rans_cuda_dists4_smem_$([ $budget = 0 ] && echo off || echo on).json
done
for budget in 0 16384; do
  PYTHONPATH=$PWD python scripts/bench_tans_vs_rans.py --device cuda \
      --symbols 1048576 --parallel-states 64 --dists 4 \
      --alphabet 8 16 --precision 7 --interleaves 1 \
      --impl tans32 tans16 tans8 rans32_16 --reps 20 \
      --smem-budget $budget \
      --json benchmarks/tans_vs_rans_cuda_longrows_smem_$([ $budget = 0 ] && echo off || echo on).json
done
```

(`PYTHONPATH=$PWD` is what makes the script measure this working tree rather than an
installed `torch_ans` - see the script's docstring.)

## Interleaved tANS on CUDA (unreleased)

Measured 2026-09-19 with the warp-level kernels in `torch_ans/tans_warp_cuda.cuh` (see
`CHANGELOG.md`): `num_interleaves` 2/4/8/32 on the device, one *lane group* per stream
row, byte-identical to the CPU interleaved kernels (`tests/test_tans.py` compares the
coded payloads at every interleave factor and cross-decodes). This section answers what
interleaving buys over one lane, what it does *not* buy against the rANS warp path, and
what the batch size and the table staging change.

### Environment and method

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti (sm_75) |
| torch | 2.4.1+cu118 (`cbench2` environment; nvcc 11.8 via `CUDA_HOME=/usr/local/cuda-11.8`) |
| workload | 4,194,304 symbols, 64 distributions, `bypass_coding=True`, symbols drawn from the distributions, `parallel_states` ∈ {8, 512, 4096} (i.e. 524288 / 8192 / 1024 symbols per row) |
| method | high-level API (`TorchANSInterface`), encode + decode, best of 5, `torch.cuda.synchronize()` around every timed call, every configuration round-trip checked before it is timed |
| tANS knobs | `TORCH_ANS_TANS_SMEM_BUDGET` (staging budget, 16 KiB default - not reached by the 64-distribution tables), `TORCH_ANS_TANS_WARP_BLOCK_THREADS` (unset = derived from the batch and the lane count) |
| raw results | `benchmarks/tans_interleaved_cuda.json` (this section), `benchmarks/tans_interleaved_cuda_smem_{off,on}.json`, `benchmarks/tans_interleaved_cuda_optimized.json` (the reworked kernels, section below) |

(The older `benchmarks/tans_vs_rans_cuda*.json` files list every `tans* x2/4/8/32` row under
`skipped` with the reason "interleaved tANS is CPU-only for now" — that was true when they were
measured, and they are left as they were produced; the numbers in this section replace those
rows.)

Throughput is Msymbol/s of *symbols* coded (as everywhere in this file); `bits/sym` is in
the JSONs and is what the row length and the header size show up in: a 32-lane tANS row
carries a 136-byte header (one state word per lane plus the shared bit buffer), so at
1024 symbols per row it is ~1.1 bits/symbol of the coded size on its own.

### Interleave gain vs one lane (`tans32`; 64 distributions, table in global memory)

| fp | batch | symbols/row | x1 enc/dec | x2 | x4 | x8 | x32 |
|---|---|---|---|---|---|---|---|
| 7 | 8 | 524288 | 19 / 28 | 22 / 35 | 30 / 41 | 30 / 41 | 18 / 21 |
| 7 | 512 | 8192 | 525 / 855 | 576 / 995 | 677 / 1104 | 705 / 1112 | 634 / 908 |
| 7 | 4096 | 1024 | 752 / 1122 | 773 / 1167 | 794 / 1183 | 789 / 1167 | 650 / 692 |
| 10 | 8 | 524288 | 18 / 25 | 21 / 34 | 29 / 41 | 30 / 41 | 19 / 21 |
| 10 | 512 | 8192 | 511 / 851 | 564 / 999 | 637 / 1092 | 626 / 1121 | 572 / 889 |
| 10 | 4096 | 1024 | 655 / 1033 | 676 / 1102 | 726 / 1175 | 685 / 1163 | 601 / 676 |
| 12 | 8 | 524288 | 18 / 25 | 21 / 33 | 29 / 41 | 30 / 40 | 19 / 20 |
| 12 | 512 | 8192 | 493 / 845 | 538 / 1016 | 610 / 1076 | 628 / 1080 | 553 / 884 |
| 12 | 4096 | 1024 | 682 / 1095 | 705 / 1146 | 692 / 1167 | 678 / 1165 | 573 / 676 |

`tans16` and `tans8` track `tans32` within a few percent at the same `freq_precision`
(their state/stream width changes the header and the number of renorm words, not the
per-symbol table work); at `fp=7` they are, encode / decode (Msymbol/s): `tans16` 19/26,
21/31, 27/37, 27/34, 18/19 at batch 8; 520/843 → 688/1086 (x4) at batch 512; 702/1101 →
755/1165 (x8) at batch 4096, and `tans8` within ~2% of `tans16`.

Findings:

- **The gain is largest exactly where one thread per row cannot fill the device.** At 8
  stream rows, 4 lanes turn 19/28 Msymbol/s into 30/41 (**+54%/+47%**) and 8 lanes into
  30/41 (**+58%/+44%**); the row is 524288 symbols long, i.e. the machine has 8 serial
  dependent chains and nothing else to run. Interleaving turns each row into 4-8
  independent chains within the same warp.
- **It shrinks as the batch grows** (512 rows: +29%/+29% at 4 lanes, +34%/+30% at 8; 4096
  rows: ~+6%/+5% at 4-8 lanes), because the device is already busy: there the win is the
  memory-level parallelism and the coalesced `symbols`/`indexes` loads (consecutive lanes
  read consecutive elements; the one-thread-per-row kernel reads a row's symbols with
  row-sized strides), not the extra warps.
- **32 lanes used to be the wrong choice for tANS - and that was a kernel problem, not a
  law of nature.** The bit layer is a single sequence the group commits in lane order, so
  the 32-lane group performed 32 sequential appends per 32 symbols (the per-symbol serial
  cost of one lane) *and* blew the register budget (229 registers/thread, one 256-thread
  block per SM). Both are fixed now: see
  [the 32-lane regression and what fixed it](#the-32-lane-regression-and-what-fixed-it).
  What stays is the 136-byte header per row at 32 lanes, which is what the `bits/sym`
  column shows (6.96 vs 5.99 bits/symbol - the coded *size* is ~16% bigger, which is why
  32 lanes is still not the answer for short rows even after the rework).
  `tests/test_tans.py` covers 32 lanes at every factor, and it is the only factor whose
  lane order (lane 31 first on the way out, lane 0 on the way in) differs from the
  sub-warp ones.

### The 32-lane regression and what fixed it

Measured 2026-09-20 on the same machine and workload, after reworking
`torch_ans/tans_warp_cuda.cuh` (`benchmarks/tans_interleaved_cuda_optimized.json`;
`benchmarks/tans_interleaved_cuda.json` is the "before"). µs per 4,194,304 symbols,
`tans32`, `fp=7`, `bypass_coding=True`:

| batch | direction | x1 before → after | x4 | x8 | x32 |
|---|---|---|---|---|---|
| 8 | encode | 219855 → 211094 | 141970 → 126738 | 139446 → 118074 | **228356 → 88000 (2.59×)** |
| 8 | decode | 148188 → 150503 | 101189 → 88324 | 103180 → 54119 | **203334 → 17548 (11.6×)** |
| 512 | encode | 7985 → 7763 | 6194 → 5863 | 5952 → 5691 | 6616 → **4963 (1.33×)** |
| 512 | decode | 4905 → 4919 | 3798 → 3350 | 3771 → 2676 | 4619 → **2126 (2.17×)** |
| 4096 | encode | 5581 → 5623 | 5285 → 5293 | 5314 → 5258 | 6456 → 6875 (0.94×) |
| 4096 | decode | 3738 → 3760 | 3547 → 3423 | 3593 → 3407 | 6064 → **3521 (1.72×)** |

Four things were wrong, in the order they mattered:

1. **Registers: the 32-lane pop kernel used 229 registers/thread.**
   `cuobjdump -res-usage` on the built extension: 8 lanes cost 80-86, 32 lanes 229-232 -
   i.e. one 256-thread block per SM, 25% occupancy for a kernel that lives on memory
   latency. The fully unrolled `constexpr_for` serial section was the cause; emitting it
   as a rolled loop (`tans_warp_roll_serial`, ≥16 lanes) brings it to 42-44 registers and
   the occupancy back to 100%. That alone was most of the decode regression.
2. **The shared bit layer was a 64-bit window updated by every lane, N times per group.**
   At rest the window holds < `STREAM_BITS` bits and one digit is ≤ 15 bits, so ≤ 31 bits
   are ever live (`TansWarpBitWriter`/`TansWarpBitReader`): a 64-bit shift is a pair of
   instructions, and it ran N times per symbol group in every lane.
3. **Two broadcasts per lane where one carries everything.** `tans_warp_pack_lane` puts the
   low 15 bits of the state (all an append can use: `nb_bits ≤ 15`), `nb_bits`, the
   validity and the bypass flag in one word, so the serial section costs one `__shfl_sync`
   per lane instead of two (push) - and the decode fast path none.
4. **The decode walk itself.** A lane's bits are a function of the prefix sum of the lanes'
   `nbBits`, so `TansWarpBitWindow` lets every lane *read* its own bits directly (≤2 words
   for `tans32`, ≤3 for `tans16`/`tans8`) instead of replaying the whole group's takes -
   the counterpart of rANS' ballot+popc ordering. Groups with a bypass run still walk
   (their digit count is only discoverable while decoding), and so do 2-lane groups, where
   there is no chain worth breaking (`tans_warp_parallel_read`).

Push keeps its lane walk, and that is deliberate: the bits have to be *written*, and a
stream word mixes the bits of several lanes, so no lane owns a word the way it owns a
read - which is exactly why rANS can get away with ballot+popc (one lane, one whole word).
The 0.94× at 4096 rows is the coded size, not the walk: 32 lanes write 6.96 bits/symbol
against 5.99 at one lane.

Also in the same pass, and included in the numbers above: the decoded symbol leaves the
serial section (one coalescing store per lane instead of N predicated single-lane stores)
and the head/tail symbols (`num_symbols % N`) are looked up by the owning lane only, which
removes N-1 redundant table walks per leftover symbol.

**Caveat on the numbers:** this GPU sits at 300 MHz when idle and boosts to 2100 MHz under
load, so a short run can be ~2.5× slower than a long one. Every table above is one run of
`best of 5` inside a full matrix run; rANS (untouched) reproduces the old file within 1-3%
in it, and every 32-lane delta above was reproduced in at least two separate runs.

### Would a per-lane bit buffer be worth it? (evaluated, not implemented)

The remaining structural idea is to give every lane its own bit buffer (its own residual
and its own payload cursor) instead of one shared one. That is the change which would make
the lanes fully independent - push included, since each lane would then own whole words and
could place them the way rANS does with ballot+popc - and it is the only remaining way to
remove the lane walk entirely.

Cost side, and why it is not being done:

- **It is a format change.** Every interleaved stream's byte string changes, so the
  CPU↔CUDA byte-string tests and any stream written by an earlier build stop decoding. The
  header layout (`tans_state_slot` / `tans_bitbuf_slot` / `tans_nbits_slot` in
  `tans_utils.hpp`) would grow from `4*(N+2)` bytes per row to roughly `4*(2N+2)`, and
  `tans_cpu.cpp`, `tans_cuda.cu`, `tans_warp_cuda.cuh`, `torch_ans/utils.py`
  (`_STREAM_SIZES` / `_stream_budget`) and `tests/test_capacity_contract.py` all move with
  it.
- **It does not compress better.** N residual buffers each hold up to `STREAM_BITS-1` bits
  that never reach the payload: at 32 lanes that is up to 60 bytes per row at the end of a
  stream, i.e. ~0.5 bit/symbol at 1024-symbol rows - on top of the ~1.1 bits/symbol the
  136-byte header already costs. The shared buffer has one residual, not N.
- **The upside is now small.** After the rework the walk is not what binds: 32 lanes is the
  fastest factor at 8 and 512 rows, and at 4096 rows its encode (0.94×) is explained by the
  coded size, not by the serial section. Expect ≤10-20% on large-batch encode for a format
  break. Revisit only if a workload wants more than 32 lanes, or if encode at a
  device-filling batch becomes the target.

### tANS against the rANS warp path (the two device interleaved kernels head to head)

| fp | batch | `tans32` x32 | `rans32_16` x32 | best tANS | its ilv | best tANS / rANS |
|---|---|---|---|---|---|---|
| 7 | 8 | 18 / 21 | 227 / 209 | 30 / 41 | x4 | 0.13 / 0.20 |
| 7 | 512 | 634 / 908 | 1039 / 1986 | 705 / 1112 | x8 | 0.68 / 0.56 |
| 7 | 4096 | 650 / 692 | 774 / 1213 | 794 / 1183 | x4 | 1.03 / 0.97 |
| 10 | 8 | 19 / 21 | 226 / 212 | 30 / 41 | x8 | 0.13 / 0.19 |
| 10 | 512 | 572 / 889 | 925 / 1828 | 626 / 1121 | x8 | 0.68 / 0.61 |
| 10 | 4096 | 601 / 676 | 665 / 1091 | 726 / 1175 | x4 | 1.09 / 1.08 |
| 12 | 8 | 19 / 20 | 221 / 210 | 30 / 40 | x8 | 0.13 / 0.19 |
| 12 | 512 | 553 / 884 | 924 / 1936 | 628 / 1080 | x8 | 0.68 / 0.56 |
| 12 | 4096 | 573 / 676 | 701 / 1218 | 692 / 1167 | x4 | 0.99 / 0.96 |

- **The rANS warp path is 5-12× faster at a tiny batch, and up to 1.8× at a realistic
  one.** The reason is structural, not a tuning miss: an interleaved rANS row owns 32
  independent *words streams* (each lane renormalizes its own words, placed by
  `__ballot_sync`), so 32 lanes give 32 parallel chains per row; a tANS row shares one bit
  buffer, so its lanes still funnel through one serial bit sequence - 4-8 lanes is where
  its table lookups are parallel enough to matter and the serialisation is still cheap.
- **At 4096 rows the two are within a few percent** (0.96-1.09×), i.e. once the device is
  saturated the coders are limited by the same thing: tANS spends its per-symbol budget on
  one table access, rANS on the division/search and its per-lane word flush. On short rows
  rANS pays ~0.6-0.8 bit/symbol more (32 flushes per row of 4 bytes), which the
  `bits/sym` column of the JSONs shows.
- Practical advice: pick `impl="rans32_16", num_interleaves=32` when a rANS-compatible
  `freq_precision <= 15` is acceptable; pick a tANS family when the table is cheap
  (small `freq_precision`, few distributions) or when the coded size on short rows
  matters, and use `num_interleaves=4` or `8` there - it is also the configuration the CPU
  path benefits from, so the same interface setting is good on both devices.

### Table staging (4 distributions, `fp=7`, alphabet 64: an 8.0 KB table, `--smem-budget 0` vs `131072`)

| impl | ilv | batch | global enc/dec | staged enc/dec | staged/global |
|---|---|---|---|---|---|
| `tans32` | 1 | 512 | 455 / 843 | 465 / 850 | 1.02 / 1.01 |
| `tans32` | 1 | 4096 | 739 / 1124 | 761 / 1105 | 1.03 / 0.98 |
| `tans32` | 4 | 512 | 630 / 969 | 637 / 972 | 1.01 / 1.00 |
| `tans32` | 4 | 4096 | 763 / 1192 | 786 / 1202 | 1.03 / 1.01 |
| `tans32` | 32 | 512 | 508 / 739 | 521 / 765 | 1.03 / 1.04 |
| `tans32` | 32 | 4096 | 578 / 589 | 611 / 694 | 1.06 / 1.18 |
| `rans32_16` | 32 | 512 | 972 / 1912 | 1009 / 1985 | 1.04 / 1.04 |
| `rans32_16` | 32 | 4096 | 722 / 1247 | 738 / 1241 | 1.02 / 1.00 |

- **Staging the whole table is a small win here and never a large loss** (+1-6% encode,
  +0-18% decode) - the same conclusion the single-lane sweeps above reached, and the warp
  kernels share the decision (`tans_table_staging` / `tans_stage_table`). The largest gains
  are at the widest group and the longest rows (32 lanes, 1024-symbol rows: +6%/+18%),
  i.e. where every lane of a group looks the table up in the same memory and the copy is
  amortized over the most symbols; the 64-distribution tables of the tables above are far
  beyond the budget and never stage.
- The rANS rows move by the same few percent, which is what makes this a staging
  measurement rather than a coder one: the CDF rows of `rans32_16` are staged by the same
  budget (A9 in `rans_warp_cuda.cuh`).

### Reproduce

```bash
# the interleave x quantization-bucket x batch matrix (the tables above)
PYTHONPATH=$PWD python scripts/bench_tans_vs_rans.py --device cuda \
    --impl tans32 tans16 tans8 rans32_16 --precision 7 10 12 --alphabet 64 \
    --dists 64 --symbols 4194304 --interleaves 1 2 4 8 32 \
    --parallel-states 8 512 4096 --reps 5 \
    --json benchmarks/tans_interleaved_cuda.json

# the staging A/B for the interleaved kernels (4 distributions, fp=7)
for budget in 0 131072; do
  PYTHONPATH=$PWD python scripts/bench_tans_vs_rans.py --device cuda \
      --impl tans32 rans32_16 --precision 7 --alphabet 64 --dists 4 \
      --symbols 4194304 --interleaves 1 4 32 --parallel-states 512 4096 --reps 5 \
      --smem-budget $budget \
      --json benchmarks/tans_interleaved_cuda_smem_$([ $budget = 0 ] && echo off || echo on).json
done
```

The CUDA build these numbers come from is `WITH_CUDA=1 pip install . --no-build-isolation`
with `CUDA_HOME=/usr/local/cuda-11.8` in the `cbench2` environment, i.e. the same kernels
`tests/test_tans.py` exercises (96 interleaved cases: byte equality with the CPU at every
factor, cross-decoding in both directions, bypass, padding indexes, head/tail and partial
blocks, the staging A/B and the high-level API including the 32-lane capacity contract).

