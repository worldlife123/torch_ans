import torch
# import numpy as np
import struct
import io

# from cbench.ans import Rans64Encoder, Rans64Decoder, TansEncoder, TansDecoder
# TODO: provide a better impl of pmf_to_quantized_cdf in torch_ans
# from cbench.rans import pmf_to_quantized_cdf, pmf_to_quantized_cdf_np

try:
    from torch_ans._C import rans_stream_to_byte_strings, rans_byte_strings_to_stream, rans_pmf_to_quantized_cdf, rans_alias_build_table
    from torch_ans._C import rans64_init_stream, rans64_push, rans64_pop
    from torch_ans._C import rans32_init_stream, rans32_push, rans32_pop
    from torch_ans._C import rans32_16_init_stream, rans32_16_push, rans32_16_pop
except:
    print("Torch ANS is not compiled properly!")

from typing import Dict, Optional, List, Tuple


def _get_bytes_format(num_bytes=4):
    """
    Returns the struct format string for a given number of bytes.

    Args:
        num_bytes (int): Number of bytes (1, 2, 4, or 8).

    Returns:
        str: Struct format string.

    Raises:
        ValueError: If num_bytes is not supported.
    """
    if num_bytes == 1:
        len_format = "B"
    elif num_bytes == 2:
        len_format = "<H"
    elif num_bytes == 4:
        len_format = "<I"
    elif num_bytes == 8:
        len_format = "<L"
    else:
        raise ValueError("")
    return len_format


def merge_bytes(data: List[bytes], num_bytes_length=4, num_segments=None) -> bytes:
    """
    Merges a list of byte strings into a single bytes object, optionally with segment lengths.

    Args:
        data (List[bytes]): List of byte strings to merge.
        num_bytes_length (int): Number of bytes to store each segment length.
        num_segments (int, optional): If provided, only writes lengths for num_segments-1 segments.

    Returns:
        bytes: Merged byte string.
    """
    stream = io.BytesIO(b"")
    len_format = _get_bytes_format(num_bytes_length)
    for i, bs in enumerate(data):
        len_bytes = struct.pack(len_format, len(bs))
        # no need to write length if num_segments is known!
        if num_segments is not None:
            assert(i < num_segments), "Number of segments exceed predefined {}".format(num_segments)
            if i < num_segments - 1:
                stream.write(len_bytes)
        else:
            stream.write(len_bytes)
        stream.write(bs)
    stream.flush()
    return stream.getvalue()

def split_merged_bytes(data: bytes, num_bytes_length=4, num_segments=None) -> List[bytes]:
    """
    Splits a merged bytes object into a list of byte strings.

    Args:
        data (bytes): Merged byte string.
        num_bytes_length (int): Number of bytes used for each segment length.
        num_segments (int, optional): Number of segments to split into.

    Returns:
        List[bytes]: List of split byte strings.
    """
    stream = io.BytesIO(data)
    len_format = _get_bytes_format(num_bytes_length)
    byte_strings = []
    while (stream.tell() < len(data)):
        if num_segments is not None and len(byte_strings) >= num_segments - 1:
            byte_string = stream.read()
        else:
            len_bytes = stream.read(num_bytes_length)
            length_stream = struct.unpack(len_format, len_bytes)[0]
            byte_string = stream.read(length_stream)
        byte_strings.append(byte_string)
    # tmp fix for empty segments
    if num_segments is not None:
        for _ in range(num_segments - len(byte_strings)):
            byte_strings.append(b'')
    return byte_strings

def pmf_to_quantized_cdf_batched(pmf : torch.Tensor, precision=16, add_tail=True, tail_mass=1e-10, normalize=True):
    """
    Converts batched PMF tensors to quantized CDFs. 
    An alternative to rans_pmf_to_quantized_cdf (without stealing trick), but may have slighly worse compression ratio.

    Args:
        pmf (torch.Tensor): Batched PMF tensor of shape (batch, symbols).
        precision (int): Number of bits for quantization.
        add_tail (bool): Whether to add a tail symbol.
        tail_mass (float): Probability mass for the tail symbol.
        normalize (bool): Whether to normalize PMF.

    Returns:
        torch.Tensor: Quantized CDF tensor.
    """
    max_index = 1 << precision
    assert pmf.shape[-1] * 2 < max_index # NOTE: is this needed?
    if add_tail:
        pmf = torch.cat([pmf.clone(), torch.zeros(pmf.shape[0], 1).type_as(pmf) + tail_mass], dim=1)
    # pmf[:, pmf_length] = tail_mass
    # for i, p in enumerate(pmf):
    #     p[(pmf_length[i]+1):] = 0

    # normalize pmf
    if normalize:
        pmf = pmf / pmf.sum(1, keepdim=True)

    # make sure all element in pmf is at least 1
    pmf_norm = pmf * max_index + 1.0
    # for i, p in enumerate(pmf_norm):
    #     p[(pmf_length[i]+1):] = 0
    pmf_sum_step = pmf.shape[-1] // 2 # NOTE: just a good value in practice. Why is that?
    pmf_norm_int = (pmf_norm * max_index / (pmf_norm.sum(1, keepdim=True) + pmf_sum_step)).round()
    # reduce some pdf to limit max cdf
    cdf_max = pmf_norm_int.sum(dim=1, keepdim=True)
    pmf_sum = cdf_max.clone()
    while (cdf_max > max_index).any():
        # further reducing (needed?)
        # idx = cdf_max > max_index
        pmf_sum[cdf_max > max_index] += pmf_sum_step
        pmf_norm_int = (pmf_norm_int * max_index / pmf_sum).round()
        cdf_max = pmf_norm_int.sum(dim=1, keepdim=True)
        # using steal technique (not deterministic for all elements)
        # max_pdf_steals = cdf_max.max().int().item() - max_index
        # steal_pdfs, steal_indices = torch.topk(pmf_norm_int, max_pdf_steals, dim=1, sorted=False)
        # # steal_indices = steal_indices[steal_pdfs > 1]
        # steal_values = ((steal_pdfs - 1) / (steal_pdfs-1).sum(1, keepdim=True) * max_pdf_steals).ceil().int()
        # pmf_norm_int[torch.arange(len(pmf_norm_int)).type_as(steal_indices).unsqueeze(-1), steal_indices] -= steal_values
    
    # Alt: use random sampling to generate integer to keep sum below max_index
    # num_samples = pmf_norm_int.sum(1) - max_index - 1
    # sample_distribution = pmf.cumsum(dim=1)
    # sample_prob = torch.rand(num_samples)

    # convert pmf to cdf
    cdf_float = pmf_norm_int.cumsum(dim=1)
    # cdf_float = pmf_norm.cumsum(dim=1)
    # cdf_float = cdf_float / cdf_float[:, -2:-1] * max_index # renormalize cdf
    cdf_float = torch.cat([
        torch.zeros(pmf.shape[0], 1).type_as(cdf_float), 
        cdf_float[:, :-1], 
        torch.zeros(pmf.shape[0], 1).type_as(cdf_float) + max_index], dim=1)
    cdf = cdf_float.int()
    # cdf[:, pmf_length] = max_index
    return cdf


def inverse_quantized_cdf(quantized_cdf, freq_precision=16):
    """
    Computes the inverse quantized CDF lookup table.

    Args:
        quantized_cdf (torch.Tensor): Quantized CDF tensor.
        freq_precision (int): Frequency precision in bits.

    Returns:
        torch.Tensor: Inverse CDF tensor.
    """
    table_size = (1<<freq_precision)
    freq_range = torch.arange(table_size).unsqueeze(0).type_as(quantized_cdf)
    inversed_cdf = (freq_range.unsqueeze(-1) >= quantized_cdf.unsqueeze(1)).sum(-1, dtype=quantized_cdf.dtype)-1
    return inversed_cdf # .type_as(quantized_cdf)


class TorchEntropyCoderBaseInterface(object):
    """
    Base interface for entropy coders using torch tensors.
    Provides caching, tensor initialization, and encode/decode method stubs.

    Args:
        symbol_precision (int): Precision of symbols in bits.
        freq_precision (int): Precision of frequencies in bits.
        mode (str): "encoder", "decoder", or "encdec".
        dtype (torch.dtype): Tensor dtype for internal buffers.
        device (str or torch.device, optional): Device for tensors.
        **kwargs: Additional arguments for encoder/decoder initialization.
    """
    def __init__(self, 
                 symbol_precision: int = 8, 
                 freq_precision: int = 16, 
                 mode="encdec", 
                 dtype=torch.int32, 
                 device=None, 
                 **kwargs) -> None:
        self.symbol_precision = symbol_precision
        self.freq_precision = freq_precision
        
        self.mode = mode
        if self.mode == "encoder" or self.mode == "encdec":
            self._init_encoder(**kwargs)
        if self.mode == "decoder" or self.mode == "encdec":
            self._init_decoder(**kwargs)
            
        self._set_dtype(dtype)
        self._set_device(device)
        
    def _init_encoder(self, **kwargs):
        """
        Initializes encoder caches for different encoding methods.
        """
        self._encode_queue = []
        # self.cache_encode_with_indexes = dict()
        # self.cache_encode_with_freqs = dict()
        # self.cache_encode_symbols = dict()

    def _init_decoder(self, **kwargs):
        """
        Initializes decoder state. Override in subclasses if needed.
        """
        self._encoded_stream = None
    
    def _init_tensor(self, tensor : torch.Tensor) -> torch.Tensor:
        """
        Moves tensor to the configured dtype and device.
        """
        return tensor.to(dtype=self.dtype, device=self.device)
    
    def _init_stream(self, **kwargs) -> torch.Tensor:
        """
        Initializes the encoding stream tensor.

        Args:
            **kwargs: Additional arguments for stream initialization.

        Returns:
            torch.Tensor: Initialized stream tensor.
        """
        return torch.zeros(1, dtype=self.dtype, device=self.device) # dummy stream tensor, override in subclass if needed
    
    def _set_dtype(self, dtype: torch.dtype) -> None:
        """
        Sets the dtype for internal tensors.

        Args:
            dtype (torch.dtype): Desired tensor dtype.
        """
        self.dtype = dtype  

    def _set_device(self, device: torch.device) -> None:
        """
        Sets the device for internal tensors.

        Args:
            device (torch.device): Desired device.
        """
        self.device = device

    def init_params(self, freqs : torch.Tensor, num_freqs : torch.Tensor, offsets : torch.Tensor) -> None:
        """
        Initializes frequency, number of frequencies, and offset tensors for coding.

        Args:
            freqs (torch.Tensor): Frequency table tensor.
            num_freqs (torch.Tensor): Number of frequencies per distribution.
            offsets (torch.Tensor): Offset tensor for each distribution.
        """
        self.freqs = self._init_tensor(freqs)
        self.num_freqs = self._init_tensor(num_freqs)
        self.offsets = self._init_tensor(offsets)

    # Override by subclass
    def _encode_func(self, symbols: torch.Tensor, stream: Optional[torch.Tensor], **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    # Override by subclass
    def _decode_func(self, stream: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def _cache_concat(self, prev : torch.Tensor, new : torch.Tensor):
        """
        Concatenates two tensors along a new batch dimension for caching.

        Args:
            prev (torch.Tensor): Previous tensor.
            new (torch.Tensor): New tensor to concatenate.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        return torch.stack([prev, new], dim=0)

    def reset_cache(self) -> None:
        """
        Resets all encoding caches.
        """
        self._encode_queue = []
        # self.cache_encode_with_indexes = dict()
        # self.cache_encode_with_freqs = dict()
        # self.cache_encode_symbols = dict()

    # def _check_patterns(self, symbols: torch.Tensor, 
    #            dist_indexes: Optional[torch.Tensor]=None, 
    #            dist_freqs: Optional[torch.Tensor]=None, # aka freqs
    #            dist_num_freqs: Optional[torch.Tensor]=None, # aka num_freqs
    #            dist_min: Optional[torch.Tensor]=None, # aka offsets
    #            cache=False, **kwargs) -> Dict[str, Optional[torch.Tensor]]:
    #     if dist_indexes is not None:
    #         assert symbols.shape == dist_indexes.shape, "For encode_with_indexes pattern, symbols and dist_indexes should have the same shape"
    #     elif dist_freqs is not None:
    #         assert symbols.shape == dist_freqs.shape[:-1], "For encode_with_freqs pattern, symbols.shape should match dist_freqs.shape[:-1]"
    #     else:
    #         raise NotImplementedError("Unsupported encoding pattern. Please provide either dist_indexes or dist_freqs.")

    # NOTE: we might add an optional stream argument for encode function later,
    # in case some entropy coder support FIFO encode (e.g. arithmetic coders are FIFO, but ANS are LIFO)
    def encode(self, symbols: torch.Tensor, 
            #    stream: Optional[torch.Tensor], 
               dist_indexes: Optional[torch.Tensor]=None, 
               dist_freqs: Optional[torch.Tensor]=None, # aka freqs
               dist_num_freqs: Optional[torch.Tensor]=None, # aka num_freqs
               dist_min: Optional[torch.Tensor]=None, # aka offsets
               cache=False, **kwargs) -> Optional[torch.Tensor]:
        """
        Main encode function that check Tensor requirements for specific encoding patterns based on provided arguments.
        1. If only symbols are provided, pass. (encode_symbols pattern)
        2. If dist_indexes is provided, check symbols.shape == dist_indexes.shape. (encode_with_indexes pattern).
        3. If dist_freqs are provided, check symbols.shape == dist_freqs.shape[:-1]. (encode_with_freqs pattern).
        4. Otherwise, raise NotImplementedError.

        If cache=True, store all necessary information into self._encode_queue. Later call self.flush to encode all cached data and return encoded stream.
        Otherwise, directly call the corresponding encode function (here just raise NotImplementedError).

        """
        if dist_indexes is not None:
            assert symbols.shape == dist_indexes.shape, "For encode_with_indexes pattern, symbols and dist_indexes should have the same shape"
            dist_indexes = self._init_tensor(dist_indexes)
        elif dist_freqs is not None:
            assert symbols.shape == dist_freqs.shape[:-1], "For encode_with_freqs pattern, symbols.shape should match dist_freqs.shape[:-1]"
            dist_freqs = self._init_tensor(dist_freqs)
            dist_num_freqs = self._init_tensor(dist_num_freqs if dist_num_freqs is not None else torch.zeros_like(symbols) + dist_freqs.shape[-1])
            dist_min = self._init_tensor(dist_min if dist_min is not None else torch.zeros_like(symbols))
        else:
            raise NotImplementedError("Unsupported encoding pattern. Please provide either dist_indexes or dist_freqs.")

        symbols = self._init_tensor(symbols)
        if cache:
            self._encode_queue.append(dict(
                symbols=symbols,
                dist_indexes=dist_indexes,
                dist_freqs=dist_freqs,
                dist_num_freqs=dist_num_freqs,
                dist_min=dist_min,
                **kwargs
            ))
        else:
            stream = self._init_stream()
            return self._encode_func(symbols, 
                stream=stream,
                dist_indexes=dist_indexes, 
                dist_freqs=dist_freqs, 
                dist_num_freqs=dist_num_freqs, 
                dist_min=dist_min, 
                **kwargs
            )

    def encode_flush(self, **kwargs) -> Optional[torch.Tensor]:
        """
        Flushes cached encoding data and returns encoded bytes.

        Args:
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        stream = None
        if len(self._encode_queue) > 0:
            # NOTE: reverse the queue to encode in LIFO order
            for item in reversed(self._encode_queue): 
                stream = self._encode_func(stream=stream, **item, **kwargs)
        return stream
        # Deprecated calls (not working properly)
        # if len(self.cache_encode_with_indexes) > 0:
        #     return self.encode_with_indexes(**self.cache_encode_with_indexes, cache=False, **kwargs)
        # if len(self.cache_encode_with_freqs) > 0:
        #     return self.encode_with_freqs(**self.cache_encode_with_freqs, cache=False, **kwargs)
        # if len(self.cache_encode_symbols) > 0:
        #     return self.encode_symbols(**self.cache_encode_symbols, cache=False, **kwargs)
        # raise NotImplementedError()

    def decode(self, stream: Optional[torch.Tensor], 
               dist_indexes: Optional[torch.Tensor]=None, 
               dist_freqs: Optional[torch.Tensor]=None, # aka freqs
               dist_num_freqs: Optional[torch.Tensor]=None, # aka num_freqs
               dist_min: Optional[torch.Tensor]=None, # aka offsets
               **kwargs) -> torch.Tensor:
        """
        Main decode function that check Tensor requirements for specific decoding patterns based on provided arguments.
        0. If stream is not provided, use self._encoded_stream.
        1. If only stream is provided, pass. (decode_symbols pattern)
        2. If dist_indexes is provided, check dist_indexes.shape matches expected shape for the encoded data. (decode_with_indexes pattern).
        3. If dist_freqs are provided, check dist_freqs.shape[:-1] matches expected shape for the encoded data. (decode_with_freqs pattern).
        4. Otherwise, raise NotImplementedError.

        Then call the corresponding decode function (here just raise NotImplementedError).

        """
        if stream is None:
            stream = self._encoded_stream

        if dist_indexes is not None:
            dist_indexes = self._init_tensor(dist_indexes)
        elif dist_freqs is not None:
            dist_freqs = self._init_tensor(dist_freqs)
            dist_num_freqs = self._init_tensor(dist_num_freqs if dist_num_freqs is not None else torch.zeros_like(dist_freqs[..., 0]) + dist_freqs.shape[-1])
            dist_min = self._init_tensor(dist_min if dist_min is not None else torch.zeros_like(dist_freqs[..., 0]))
        else:
            raise NotImplementedError("Unsupported decoding pattern. Please provide either dist_indexes or dist_freqs.")
        
        decoded = self._decode_func(stream, 
            dist_indexes=dist_indexes, 
            dist_freqs=dist_freqs, 
            dist_num_freqs=dist_num_freqs, 
            dist_min=dist_min, 
            **kwargs
        )
        return decoded

    ####### CompressAI/FSAR like API functions #########
    # NOTE: these functions should represent streams as bytes instead of torch.Tensor, following CompressAI's API design. 
    def encode_with_indexes(self, symbols: torch.Tensor, indexes: torch.Tensor, cache=False, **kwargs) -> Optional[bytes]:
        """
        Caches or encodes symbols with indexes.

        Args:
            symbols (torch.Tensor): Symbols to encode.
            indexes (torch.Tensor): Distribution indexes for each symbol.
            cache (bool): If True, cache for later flush.
            **kwargs: Additional arguments.

        Returns:
            bytes: Encoded bytes (if cache=False).
        """
        # if cache:
        #     if len(self.cache_encode_with_indexes) == 0:
        #         self.cache_encode_with_indexes = dict(
        #             symbols=self._init_tensor(symbols), 
        #             indexes=self._init_tensor(indexes), 
        #             **kwargs
        #         )
        #     else:
        #         self.cache_encode_with_indexes["symbols"] = self._cache_concat(self.cache_encode_with_indexes["symbols"], self._init_tensor(symbols))
        #         self.cache_encode_with_indexes["indexes"] = self._cache_concat(self.cache_encode_with_indexes["indexes"], self._init_tensor(indexes))
        # else:
        raise NotImplementedError()

    def decode_with_indexes(self, encoded: str, indexes: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes encoded data using indexes.

        Args:
            encoded (str): Encoded stream.
            indexes (torch.Tensor): Distribution indexes for each symbol.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded symbols.
        """
        raise NotImplementedError()

    def flush(self, **kwargs) -> bytes:
        """
        Flushes cached encoding data and returns encoded bytes.

        Args:
            **kwargs: Additional arguments.

        Returns:
            bytes: Encoded bytes.
        """
        # raise NotImplementedError()
        stream = self.encode_flush(**kwargs)
        if stream is None:
            return b''
        else:
            byte_strings = rans_stream_to_byte_strings(stream)
            return merge_bytes(byte_strings, num_bytes_length=self.num_bytes_code_length, num_segments=self.num_parallel_states)

    def set_stream(self, encoded: str) -> None:
        """
        Sets the encoded stream for decoding operations.

        Args:
            encoded (str): Encoded stream (bytes or string).
        """
        self.encoded = encoded

    def decode_stream(self, indexes: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes a stream using provided indexes.

        Args:
            indexes (torch.Tensor): Distribution indexes for each symbol.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded symbols.
        """
        raise NotImplementedError()

    def encode_with_freqs(self, symbols: torch.Tensor, freqs: torch.Tensor, num_freqs: torch.Tensor, offsets: torch.Tensor, cache=False, **kwargs) -> Optional[bytes]:
        """
        Caches or encodes symbols with per-symbol frequencies.

        Args:
            symbols (torch.Tensor): Symbols to encode.
            freqs (torch.Tensor): Frequency table tensor.
            num_freqs (torch.Tensor): Number of frequencies per distribution.
            offsets (torch.Tensor): Offset tensor for each distribution.
            cache (bool): If True, cache for later flush.
            **kwargs: Additional arguments.

        Returns:
            bytes: Encoded bytes (if cache=False).
        """
        # if cache:
        #     if len(self.cache_encode_with_indexes) == 0:
        #         self.cache_encode_with_indexes = dict(
        #             symbols=self._init_tensor(symbols), 
        #             indexes=self._init_tensor(indexes), 
        #             **kwargs
        #         )
        #     else:
        #         self.cache_encode_with_indexes["symbols"] = self._cache_concat(self.cache_encode_with_indexes["symbols"], self._init_tensor(symbols))
        #         self.cache_encode_with_indexes["indexes"] = self._cache_concat(self.cache_encode_with_indexes["indexes"], self._init_tensor(indexes))
        # else:
        raise NotImplementedError()

    def decode_with_freqs(self, encoded: str, freqs: torch.Tensor, num_freqs: torch.Tensor, offsets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes encoded data using per-symbol frequencies.

        Args:
            encoded (str): Encoded stream.
            freqs (torch.Tensor): Frequency table tensor.
            num_freqs (torch.Tensor): Number of frequencies per distribution.
            offsets (torch.Tensor): Offset tensor for each distribution.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded symbols.
        """
        raise NotImplementedError()

    def decode_stream_with_freqs(self, freqs: torch.Tensor, num_freqs: torch.Tensor, offsets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes a stream using provided per-symbol frequencies.

        Args:
            freqs (torch.Tensor): Frequency table tensor.
            num_freqs (torch.Tensor): Number of frequencies per distribution.
            offsets (torch.Tensor): Offset tensor for each distribution.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded symbols.
        """
        raise NotImplementedError()


class TorchANSInterface(TorchEntropyCoderBaseInterface):
    """
    Torch ANS (Asymmetric Numeral Systems) entropy coder interface.
    Implements encoding/decoding using rANS variants and manages coder parameters.

    Args:
        impl (str): rANS implementation variant ("rans64", "rans32", etc.).
        bypass_coding (bool): Enable bypass coding for out-of-range symbols. 
                              This may not be supported in all implementations, or may reduce throughput!
        bypass_precision (int): Precision for bypass coding.
        num_parallel_states (int, optional): Number of parallel ANS states. Or leave it None to use batch size of the input tensor as parallel states.
        num_bytes_code_length (int): Number of bytes allocated for code length in serialization.
        **kwargs: Passed to TorchEntropyCoderBaseInterface.
    """
    def __init__(self, 
                 impl="rans64", 
                 bypass_coding: bool = True, 
                 bypass_precision: int = 4, 
                 num_parallel_states=None, 
                 num_bytes_code_length=4, 
                 **kwargs) -> None:
        self.impl = impl
        self.bypass_coding = bypass_coding
        self.bypass_precision = bypass_precision
        self.num_parallel_states = num_parallel_states
        self.num_bytes_code_length = num_bytes_code_length
        super().__init__(**kwargs)
        
        # TODO: expose them to options
        self.impl_use_inverse_cdf = False
        self.impl_use_alias_table = False
        
        if self.impl.startswith("rans64"):
            self.ans_init_func = rans64_init_stream
            self.ans_encode_func = rans64_push
            self.ans_decode_func = rans64_pop
            self.freq_precision = min(self.freq_precision, 31)
        elif self.impl.startswith("rans32") or self.impl.startswith("rans_byte"):
            self.ans_init_func = rans32_init_stream
            self.ans_encode_func = rans32_push
            self.ans_decode_func = rans32_pop
            self.freq_precision = min(self.freq_precision, 23)
        elif self.impl.startswith("rans32_16"):
            self.ans_init_func = rans32_16_init_stream
            self.ans_encode_func = rans32_16_push
            self.ans_decode_func = rans32_16_pop
            self.freq_precision = min(self.freq_precision, 15)
        else:
            raise NotImplementedError(f"Unknown impl {self.impl}")

    def _init_stream(self, num_parallel_states=None, **kwargs) -> torch.Tensor:
        """
        Initializes the rANS stream tensor with the configured number of parallel states.
        """
        num_parallel_states = num_parallel_states if num_parallel_states is not None else self.num_parallel_states
        num_parallel_states = int(num_parallel_states) if num_parallel_states is not None else 1
        return self._init_tensor(self.ans_init_func(num_parallel_states))

    def init_params(self, freqs, num_freqs, offsets) -> None:
        """
        Initializes coder parameters and quantized CDFs for rANS encoding/decoding.

        Args:
            freqs (torch.Tensor): Frequency table tensor.
            num_freqs (torch.Tensor): Number of frequencies per distribution.
            offsets (torch.Tensor): Offset tensor for each distribution.
        """
        # cdfs = pmf_to_quantized_cdf_batched(freqs, add_tail=self.bypass_coding, freq_precision=self.freq_precision)
        # cdfs = torch.zeros(freqs.shape[0], num_freqs.max() + (2 if self.bypass_coding else 1), dtype=torch.int32)
        # for i, p in enumerate(freqs):
        #     prob = (p[:num_freqs[i]].float() / (1<<self.freq_precision)).tolist() 
        #     if self.bypass_coding: prob.append(1/(1<<self.freq_precision))
        #     cdf = pmf_to_quantized_cdf(prob, self.freq_precision)
        #     cdfs[i, : len(cdf)] = torch.as_tensor(cdf, dtype=torch.int32)
        if self.bypass_coding:
            freqs = torch.cat([freqs.clone(), torch.zeros_like(freqs[..., :1]) + 1e-10], dim=-1)
        pmf = freqs.float() / freqs.sum(-1, keepdim=True)
        cdfs = rans_pmf_to_quantized_cdf(pmf.to(device=self.device), self.freq_precision)
        cdfs_sizes = num_freqs + (2 if self.bypass_coding else 1)
        if self.impl_use_inverse_cdf:
            inversed_cdfs = inverse_quantized_cdf(cdfs, freq_precision=self.freq_precision)
            cdfs = torch.cat([cdfs, inversed_cdfs], dim=-1)
        elif self.impl_use_alias_table:
            cdfs, cdfs_with_alias_table = rans_alias_build_table(
                cdfs, cdfs_sizes, symbol_precision=self.symbol_precision, freq_precision=self.freq_precision
            )
            self.cdfs_with_alias_table = self._init_tensor(cdfs_with_alias_table).contiguous()

        self.cdfs = self._init_tensor(cdfs).contiguous()
        self.cdfs_sizes = self._init_tensor(cdfs_sizes).contiguous()
        self.offsets = self._init_tensor(offsets).contiguous()

        # if self.device is not None:
        #     self.cdfs = self.cdfs.to(device=self.device)
        #     self.cdfs_sizes = self.cdfs_sizes.to(device=self.device)
        #     self.offsets = self.offsets.to(device=self.device)

    def _encode_func(self, symbols, 
                     stream: Optional[torch.Tensor]=None,
                     dist_indexes: Optional[torch.Tensor]=None,
                     dist_freqs: Optional[torch.Tensor]=None, # aka freqs
                     dist_num_freqs: Optional[torch.Tensor]=None, # aka num_freqs
                     dist_min: Optional[torch.Tensor]=None, # aka offsets
                     **kwargs) -> torch.Tensor:
        
        if stream is None:
            num_parallel_states = symbols.size(0) if self.num_parallel_states is None else self.num_parallel_states
            # assert num_parallel_states == symbols.size(0)
            stream = self._init_stream(num_parallel_states=num_parallel_states)
        else:
            num_parallel_states = stream.size(0)

        if dist_indexes is not None:
            self.ans_encode_func(stream, 
                symbols.reshape(num_parallel_states, -1).contiguous(), 
                dist_indexes.reshape(num_parallel_states, -1).contiguous(), 
                self.cdfs, self.cdfs_sizes, self.offsets,
                freq_precision=self.freq_precision, 
                bypass_coding=self.bypass_coding, 
                bypass_precision=self.bypass_precision
            )
        elif dist_freqs is not None:
            self.init_params(
                dist_freqs.reshape(-1, dist_freqs.shape[-1]), 
                dist_num_freqs.reshape(-1), 
                dist_min.reshape(-1)
            )
            dist_indexes = torch.arange(symbols.numel(), dtype=self.dtype, device=stream.device)\
                .reshape_as(symbols)
            self.ans_encode_func(stream, 
                symbols.reshape(num_parallel_states, -1).contiguous(), 
                dist_indexes.reshape(num_parallel_states, -1).contiguous(), 
                self.cdfs, self.cdfs_sizes, self.offsets,
                freq_precision=self.freq_precision, 
                bypass_coding=self.bypass_coding, 
                bypass_precision=self.bypass_precision
            )
        else:
            # TODO: implement encode_symbols pattern if needed
            raise NotImplementedError("Unsupported encoding pattern. Please provide either dist_indexes or dist_freqs.")

        return stream

    def _decode_func(self, stream, 
                     dist_indexes: Optional[torch.Tensor]=None,
                     dist_freqs: Optional[torch.Tensor]=None, # aka freqs
                     dist_num_freqs: Optional[torch.Tensor]=None, # aka num_freqs
                     dist_min: Optional[torch.Tensor]=None, # aka offsets
                     **kwargs) -> torch.Tensor:
        num_parallel_states = stream.size(0) if self.num_parallel_states is None else self.num_parallel_states
        if dist_indexes is not None:
            cdfs = self.cdfs_with_alias_table if self.impl_use_alias_table else self.cdfs

            decoded = self.ans_decode_func(stream, 
                dist_indexes.reshape(num_parallel_states, -1).contiguous(), 
                cdfs, self.cdfs_sizes, self.offsets,
                freq_precision=self.freq_precision, 
                bypass_coding=self.bypass_coding, 
                bypass_precision=self.bypass_precision,
            )
            decoded = decoded.reshape_as(dist_indexes)
        elif dist_freqs is not None:
            self.init_params(
                dist_freqs.reshape(-1, dist_freqs.shape[-1]), 
                dist_num_freqs.reshape(-1), 
                dist_min.reshape(-1)
            )
            dist_indexes = torch.arange(dist_num_freqs.numel(), dtype=self.dtype, device=stream.device)\
                .reshape_as(dist_num_freqs)
            cdfs = self.cdfs_with_alias_table if self.impl_use_alias_table else self.cdfs
            decoded = self.ans_decode_func(stream, 
                dist_indexes.reshape(num_parallel_states, -1).contiguous(), 
                cdfs, self.cdfs_sizes, self.offsets,
                freq_precision=self.freq_precision, 
                bypass_coding=self.bypass_coding, 
                bypass_precision=self.bypass_precision,
            )
            decoded = decoded.reshape_as(dist_num_freqs)
        else:
            # TODO: implement decode_symbols pattern if needed
            raise NotImplementedError("Unsupported decoding pattern. Please provide either dist_indexes or dist_freqs.")

        return decoded.to(device=stream.device)

    ####### CompressAI/FSAR like API functions #########
    # NOTE: these functions should represent streams as bytes instead of torch.Tensor, following CompressAI's API design. 
    def set_cdfs(self, cdfs: torch.Tensor, cdfs_sizes: torch.Tensor, offsets: torch.Tensor) -> None:
        """
        Sets coder CDFs, sizes, and offsets directly.

        Args:
            cdfs (torch.Tensor): CDF tensor.
            cdfs_sizes (torch.Tensor): Sizes of each CDF.
            offsets (torch.Tensor): Offset tensor for each distribution.
        """
        self.cdfs = self._init_tensor(cdfs).contiguous()
        self.cdfs_sizes = self._init_tensor(cdfs_sizes).contiguous()
        self.offsets = self._init_tensor(offsets).contiguous()

        # if self.device is not None:
        #     self.cdfs = self.cdfs.to(device=self.device)
        #     self.cdfs_sizes = self.cdfs_sizes.to(device=self.device)
        #     self.offsets = self.offsets.to(device=self.device)

    def _init_decoder(self, **kwargs):
        """
        Initializes decoder stream for rANS decoding.
        """
        # self._stream = None
        return super()._init_decoder(**kwargs)

    def encode_with_indexes(self, symbols: torch.Tensor, indexes: torch.Tensor, cache=False, **kwargs) -> Optional[bytes]:
        """
        Encodes symbols using indexes with rANS.

        Args:
            symbols (torch.Tensor): Symbols to encode.
            indexes (torch.Tensor): Distribution indexes for each symbol.
            cache (bool): If True, use base class caching.
            **kwargs: Additional arguments.

        Returns:
            bytes: Encoded byte stream.
        """
        stream = self.encode(symbols, dist_indexes=indexes, cache=cache, **kwargs)
        if cache:
            return
        else:
            byte_strings = rans_stream_to_byte_strings(stream)
            return merge_bytes(byte_strings, num_bytes_length=self.num_bytes_code_length, num_segments=self.num_parallel_states)
        # if cache:
        #     return super().encode_with_indexes(symbols, indexes, cache, **kwargs)
        # else:
        #     num_parallel_states = symbols.size(0) if self.num_parallel_states is None else self.num_parallel_states
        #     # assert num_parallel_states == symbols.size(0)
            
        #     stream = self._init_stream(num_parallel_states=num_parallel_states)
        #     # if self.device is not None:
        #     #     stream = stream.to(device=self.device)
        #     #     symbols = symbols.to(device=self.device)
        #     #     indexes = indexes.to(device=self.device)

        #     self.ans_encode_func(stream, 
        #         self._init_tensor(symbols).reshape(num_parallel_states, -1).contiguous(), 
        #         self._init_tensor(indexes).reshape(num_parallel_states, -1).contiguous(), 
        #         self.cdfs, self.cdfs_sizes, self.offsets,
        #         freq_precision=self.freq_precision, 
        #         bypass_coding=self.bypass_coding, 
        #         bypass_precision=self.bypass_precision
        #     )
        #     byte_strings = rans_stream_to_byte_strings(stream)
            
        #     return merge_bytes(byte_strings, num_bytes_length=self.num_bytes_code_length, num_segments=self.num_parallel_states)
        
    def decode_with_indexes(self, encoded: str, indexes: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes encoded data using indexes with rANS.

        Args:
            encoded (str): Encoded byte stream.
            indexes (torch.Tensor): Distribution indexes for each symbol.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded symbols.
        """

        byte_strings = split_merged_bytes(encoded, num_bytes_length=self.num_bytes_code_length, num_segments=self.num_parallel_states)
        stream = self._init_tensor(rans_byte_strings_to_stream(byte_strings))

        return self.decode(stream=stream, dist_indexes=indexes, **kwargs)
        
        # num_parallel_states = indexes.size(0) if self.num_parallel_states is None else self.num_parallel_states
        # # assert num_parallel_states == indexes.size(0)

        # cdfs = self.cdfs_with_alias_table if self.impl_use_alias_table else self.cdfs

        # # if self.device is not None:
        # #     stream = stream.to(device=self.device)
        # #     indexes = indexes.to(device=self.device)

        # decoded = self.ans_decode_func(stream, 
        #     self._init_tensor(indexes).reshape(num_parallel_states, -1).contiguous(), 
        #     cdfs, self.cdfs_sizes, self.offsets,
        #     freq_precision=self.freq_precision, 
        #     bypass_coding=self.bypass_coding, 
        #     bypass_precision=self.bypass_precision,
        # )
        
        # decoded = decoded.reshape_as(indexes).to(device=indexes.device)
         
        # return decoded

    def set_stream(self, encoded: str) -> None:
        """
        Sets the internal stream for rANS decoding from encoded bytes.

        Args:
            encoded (str): Encoded byte stream.
        """
        byte_strings = split_merged_bytes(encoded, num_bytes_length=self.num_bytes_code_length, num_segments=self.num_parallel_states)
        self._encoded_stream = self._init_tensor(rans_byte_strings_to_stream(byte_strings))

    def decode_stream(self, indexes: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes the internal stream using provided indexes with rANS.

        Args:
            indexes (torch.Tensor): Distribution indexes for each symbol.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded symbols.
        """
        return self.decode(stream=self._encoded_stream, dist_indexes=indexes, **kwargs)

        # decoded = self.ans_decode_func(self._encoded_stream, 
        #     self._init_tensor(indexes).contiguous(), 
        #     self.cdfs, self.cdfs_sizes, self.offsets,
        #     freq_precision=self.freq_precision, 
        #     bypass_coding=self.bypass_coding, 
        #     bypass_precision=self.bypass_precision,
        # )
        
        # decoded = decoded.reshape_as(indexes).to(device=indexes.device)

        # return decoded

    def encode_with_freqs(self, symbols: torch.Tensor, freqs: torch.Tensor, num_freqs: torch.Tensor, offsets: torch.Tensor, cache=False, **kwargs) -> Optional[bytes]:
        """
        Encodes symbols using per-symbol frequencies with rANS.

        Args:
            symbols (torch.Tensor): Symbols to encode.
            freqs (torch.Tensor): Frequency table tensor.
            num_freqs (torch.Tensor): Number of frequencies per distribution.
            offsets (torch.Tensor): Offset tensor for each distribution.
            cache (bool): If True, use base class caching.
            **kwargs: Additional arguments.

        Returns:
            bytes: Encoded byte stream.
        """
        stream = self.encode(symbols, dist_freqs=freqs, dist_num_freqs=num_freqs, dist_min=offsets, cache=cache, **kwargs)
        if cache:
            return
        else:
            byte_strings = rans_stream_to_byte_strings(stream)
            return merge_bytes(byte_strings, num_bytes_length=self.num_bytes_code_length, num_segments=self.num_parallel_states)

    def decode_with_freqs(self, encoded: str, freqs: torch.Tensor, num_freqs: torch.Tensor, offsets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes encoded data using per-symbol frequencies with rANS.

        Args:
            encoded (str): Encoded byte stream.
            freqs (torch.Tensor): Frequency table tensor.
            num_freqs (torch.Tensor): Number of frequencies per distribution.
            offsets (torch.Tensor): Offset tensor for each distribution.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded symbols.
        """
        byte_strings = split_merged_bytes(encoded, num_bytes_length=self.num_bytes_code_length, num_segments=self.num_parallel_states)
        stream = self._init_tensor(rans_byte_strings_to_stream(byte_strings))

        return self.decode(stream=stream, dist_freqs=freqs, dist_num_freqs=num_freqs, dist_min=offsets, **kwargs)

    def decode_stream_with_freqs(self, freqs: torch.Tensor, num_freqs: torch.Tensor, offsets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decodes the internal stream using provided per-symbol frequencies with rANS.

        Args:
            freqs (torch.Tensor): Frequency table tensor.
            num_freqs (torch.Tensor): Number of frequencies per distribution.
            offsets (torch.Tensor): Offset tensor for each distribution.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Decoded symbols.
        """
        return self.decode(stream=self._encoded_stream, dist_freqs=freqs, dist_num_freqs=num_freqs, dist_min=offsets, **kwargs)
    


