"""The coded bytes of an int32 stream, on every torch the matrix covers.

The tests that compare a stream against `rans_stream_to_byte_strings` need the
stream as bytes, and the one-liner for that is `stream.view(torch.uint8)`. It
does not survive the oldest entry of the CI matrix:

    RuntimeError: Viewing a tensor as a new dtype with a different number of
    bytes per element is not supported.

torch < 2.0 has that restriction (`pyproject.toml` still promises
`torch>=1.10`), and with `test_capacity_contract` + `test_tans` both reaching
for it, one job reported 33 failures from this single line.

`int.to_bytes` is what a `.view` does internally, so the portable spelling is
the same bytes - and the streams the tests inspect are a handful of rows, so
building them in Python costs nothing.
"""

import torch


def uint8_view(stream: torch.Tensor) -> torch.Tensor:
    """`stream` as a `(rows, bytes)` uint8 tensor, little-endian per word."""
    data = stream.detach().to(torch.int32).contiguous().cpu()
    words = data.reshape(-1).tolist()
    flat = [byte for word in words
            for byte in int(word & 0xFFFFFFFF).to_bytes(4, "little")]
    return torch.tensor(flat, dtype=torch.uint8).reshape(data.shape[0], -1)
