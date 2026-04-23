#!/usr/bin/env python3
"""
Script to fetch supported Python and CUDA versions from PyTorch's official wheel index.
Outputs JSON for use in CI matrix.
"""

import json
import re
import urllib.request
import urllib.parse
from collections import defaultdict

def fetch_pytorch_versions(cuda_versions):
    """Fetch available PyTorch wheel versions from official indices."""
    # Known CUDA versions from PyTorch
    base_url = "https://download.pytorch.org/whl/{}/torch/"

    # Patterns for different OS/platforms
    wheel_patterns = [
        # Linux x86_64
        (r'href="[^"]*torch-[^"]*linux_x86_64\.whl[^"]*"', 'linux', 'x86_64'),
        # Linux aarch64
        (r'href="[^"]*torch-[^"]*linux_aarch64\.whl[^"]*"', 'linux', 'aarch64'),
        # Linux ppc64le
        (r'href="[^"]*torch-[^"]*linux_ppc64le\.whl[^"]*"', 'linux', 'ppc64le'),
        # Windows x86_64
        (r'href="[^"]*torch-[^"]*win_amd64\.whl[^"]*"', 'windows', 'x86_64'),
        # macOS x86_64
        (r'href="[^"]*torch-[^"]*macosx_10_9_x86_64\.whl[^"]*"', 'macos', 'x86_64'),
        # macOS arm64 (Apple Silicon)
        (r'href="[^"]*torch-[^"]*macosx_11_0_arm64\.whl[^"]*"', 'macos', 'arm64'),
    ]

    variants = set()
    python_versions = set()
    cuda_versions_found = set()

    for cuda in cuda_versions:
        index_url = base_url.format(cuda)
        try:
            with urllib.request.urlopen(index_url) as response:
                html = response.read().decode('utf-8')
        except Exception as e:
            print(f"Error fetching {index_url}: {e}")
            continue

        for wheel_pattern, os_name, arch in wheel_patterns:
            wheels = re.findall(wheel_pattern, html)
            for wheel in wheels:
                wheel = urllib.parse.unquote(wheel)
                # Extract version info from wheel name
                # Format: torch-X.Y.Z+{cpu|cuXXX}-cp{PYTHON}-cp{PYTHON}-PLATFORM.whl
                match = re.search(r'torch-[^+-]+(?:\+([^+-]+))?-cp(\d+)-cp\d+-[^.]+\.whl', wheel)
                if match:
                    cuda_part = match.group(1) or 'cpu'
                    py_tag = match.group(2)
                    if py_tag.startswith('3'):
                        py_version = f"3.{py_tag[1:]}"
                    else:
                        py_version = f"2.{py_tag[1:]}"
                    python_versions.add(py_version)
                    cuda_versions_found.add(cuda_part)
                    variants.add((py_version, cuda_part, os_name, arch))

    return sorted(python_versions), sorted(cuda_versions_found), sorted(variants)

def main():

    # get CUDA versions from command line or use defaults
    import argparse
    parser = argparse.ArgumentParser(description="Fetch supported PyTorch versions for CI matrix.")
    parser.add_argument('--cuda-versions', nargs='*', default=['cpu', 'cu118', 'cu124', 'cu128'], help="CUDA versions to check (e.g., cpu, cu102, cu113, cu116)")
    args = parser.parse_args()

    python_versions, cuda_versions, variants = fetch_pytorch_versions(args.cuda_versions)

    # Filter to reasonable versions (PyTorch typically supports recent Python versions)
    python_versions = [v for v in python_versions if tuple(map(int, v.split('.'))) >= (3, 7)]

    # Create matrix entries with os and arch
    matrix = []
    for py, cuda, os_name, arch in variants:
        if py in python_versions:
            matrix.append({
                "python-version": py,
                "cuda-version": cuda,
                "os": os_name,
                "platform": arch
            })

    # Output as JSON for GitHub Actions
    output = {
        "python_versions": python_versions,
        "cuda_versions": cuda_versions,
        "matrix": matrix
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()