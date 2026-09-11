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

