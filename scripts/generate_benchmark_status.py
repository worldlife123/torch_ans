#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark_status.md from pytest-benchmark JSON artifacts.")
    parser.add_argument("results_dir", type=Path, help="Directory containing benchmark artifact folders.")
    parser.add_argument("output_file", type=Path, help="Output markdown file path.")
    return parser.parse_args()


def format_benchmark_results(results_dir: Path) -> str:
    lines = [
        "## Benchmark Results",
        "",
        "| OS | Platform | Python | CUDA | Group | Benchmark | Params | Median | Unit |",
        "|----|----------|--------|------|-------|-----------|--------|--------|------|",
    ]

    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("benchmark-result-"):
            continue

        env = entry.name[len("benchmark-result-"):]
        parts = env.split("-")
        if len(parts) < 4:
            continue

        os_name = "-".join(parts[:-3])
        platform = parts[-3]
        python_version = parts[-2]
        cuda = parts[-1]
        bench_file = entry / "benchmark.json"
        if not bench_file.exists():
            continue

        data = json.loads(bench_file.read_text())
        for bench in data.get("benchmarks", []):
            group = bench.get("group", "")
            name = bench.get("name", "")
            params = bench.get("params", {})
            params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
            stats = bench.get("stats", {})
            median = stats.get("median")
            unit = stats.get("unit", "")
            median_str = "" if median is None else f"{median:.6f}"
            lines.append(
                f"| {os_name} | {platform} | {python_version} | {cuda} | {group} | {name} | {params_str} | {median_str} | {unit} |"
            )

    lines.append("")
    lines.append(f"Last updated: {datetime.utcnow().isoformat()}Z")
    return "\n".join(lines)


def main():
    args = parse_args()
    args.output_file.write_text(format_benchmark_results(args.results_dir))


if __name__ == "__main__":
    main()
