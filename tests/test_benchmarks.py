import pytest
import torch

pytest.importorskip("pytest_benchmark")

from torch_ans.utils import TorchANSInterface


def is_cuda_supported():
    if not torch.cuda.is_available():
        return False
    try:
        # Try a small encode/decode operation on CUDA
        pmf = torch.ones(1, 2, dtype=torch.int32)
        num_freqs = torch.full((1,), 2, dtype=torch.int32)
        offsets = torch.zeros(1, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=12, device="cuda")
        coder.init_params(pmf, num_freqs, offsets)

        symbols = torch.zeros(1, 1, dtype=torch.int32).cuda()
        indexes = torch.zeros(1, 1, dtype=torch.int32).cuda()

        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)
        return True
    except Exception as e:
        return False


devices = ["cpu"]
if is_cuda_supported():
    devices.append("cuda")


@pytest.mark.parametrize("batch_size", [1, 16, 256, 4096])
@pytest.mark.parametrize("device", devices)
def test_rans64_encode_throughput(benchmark, batch_size, device):
    benchmark.group = "encode"
    torch.manual_seed(0)
    num_dists = 8
    num_symbols = 256
    seq_len = 1024 * 1024 // batch_size

    pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
    num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
    offsets = torch.zeros(num_dists, dtype=torch.int32)

    coder = TorchANSInterface(impl="rans64", freq_precision=12, device=device)
    coder.init_params(pmf, num_freqs, offsets)

    symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32).to(device)
    indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32).to(device)

    def encode():
        coder.encode_with_indexes(symbols, indexes)
        if device == "cuda":
            torch.cuda.synchronize()
    benchmark(encode)


@pytest.mark.parametrize("batch_size", [1, 16, 256, 4096])
@pytest.mark.parametrize("device", devices)
def test_rans64_decode_throughput(benchmark, batch_size, device):
    benchmark.group = "decode"
    torch.manual_seed(0)
    num_dists = 8
    num_symbols = 256
    seq_len = 1024 * 1024 // batch_size

    pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
    num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
    offsets = torch.zeros(num_dists, dtype=torch.int32)

    coder = TorchANSInterface(impl="rans64", freq_precision=12, device=device)
    coder.init_params(pmf, num_freqs, offsets)

    symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32).to(device)
    indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32).to(device)

    encoded = coder.encode_with_indexes(symbols, indexes)
    decoder = TorchANSInterface(impl="rans64", freq_precision=12, device=device)
    decoder.set_cdfs(coder.cdfs, coder.cdfs_sizes, coder.offsets)

    def decode():
        decoder.decode_with_indexes(encoded, indexes)
        if device == "cuda":
            torch.cuda.synchronize()

    benchmark(decode)

