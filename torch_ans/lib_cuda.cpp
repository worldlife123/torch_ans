// Entry point of the *CUDA* flavour of the prebuilt libtorch-stable-ABI wheel.
//
// The wheel ships a pair of libraries:
//
//   torch_ans/_C_cpu.so   lib.cpp         - every schema + CPU/catch-all impls
//   torch_ans/_C_cuda.so  this file       - CUDA impls only, no schema at all
//
// The split exists because a schema may be declared exactly once per process:
// calling `def()` a second time (even with a byte-identical schema string) is
// rejected by the dispatcher - "Tried to register an operator ... with the same
// name and overload name multiple times" - and the C shim prints the whole c10
// stack to stderr before carrying on. Registering *implementations* for another
// dispatch key is allowed, and the load order of the two .so files does not
// matter (both verified in the POC, poc/stable_abi/tests/test_poc_cuda.py).
//
// There is deliberately no `init_stream` here: it takes no Tensor argument, the
// dispatcher therefore has no backend key to derive, and it is registered for
// the catch-all key (CompositeExplicitAutograd) in lib.cpp, which serves CPU and
// CUDA alike.
//
// NOTE: the whole registration is compiled out unless this is really the stable
// CUDA library. `setup.py`'s WITH_STABLE_ABI branch lists this file explicitly,
// but the other two build paths collect their sources with a `*.cpp` glob
// (setup.py's ATen module and torch_ans/_dynamic_build.py's JIT build); both
// exclude this file by name, and the guard below keeps a stray inclusion
// harmless either way.
#include "rans.hpp"
#include "rans_bindings.hpp"

#if defined(TORCH_ANS_STABLE_ABI) && TA_WITH_CUDA
// TA_WITH_CUDA is set by ta/torch_api.hpp from WITH_CUDA/WITH_HIP; the stable
// CUDA build passes -DWITH_CUDA=1 -DUSE_CUDA=1 (setup.py), where USE_CUDA is
// what makes the stable C shim declare its CUDA entries.
STABLE_TORCH_LIBRARY_IMPL(torch_ans, CUDA, m) {
  torch_ans_register_cuda_impls(m);
}
#endif  // TORCH_ANS_STABLE_ABI && TA_WITH_CUDA
