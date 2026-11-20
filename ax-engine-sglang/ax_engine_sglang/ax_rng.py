"""Sampled decoding loses its distribution when flashinfer and the next generator user draw the same numbers.

flashinfer.sampling.get_seed_and_offset (flashinfer 0.7.2) reserves philox offsets [old, new) on the
CUDA generator and then hands its own kernel the *new* offset, which is where the generator's next user
starts. With DFlash2 selector sampling the next user is the draft walk's in-graph `uniform_()` in the
first decode round after a prefill, so the walk's first uniform equals the coin prefill used to sample
the first output token: given that token the draft is no longer drawn from the q it reports, and the
rejection test over-accepts it (bf16, stock DFlash2: " l" after "The" 0.53 against 0.19 in plain
decoding; 7 of 28 distribution tests fail; with the old offset 0 of 28, docs/decisions section 16).

install() makes get_seed_and_offset return the old offset (the start of the reserved range) in
flashinfer.sampling and in modules that imported it by name. Greedy decoding draws nothing and is
unchanged. AX_FI_OFFSET_FIX=0 leaves flashinfer as it is.

Upstream fixed this in flashinfer PR 5651 (released in 0.7.3; 0.7.4 also serialises reservations
with a lock, PR 9148), and SGLang main pins flashinfer 0.7.4; SGLang 0.5.23 pins 0.7.2. install()
therefore patches only flashinfer versions below 0.7.3 and leaves newer ones alone.
"""
from ax_engine_sglang.compat import patch_attribute
import os
import sys

_MODULES = ("flashinfer.sampling", "flashinfer.cake_sampling")


def _fixed(module):
    import torch

    def get_seed_and_offset(increment, generator=None, device=None):
        if generator is None:
            generator = module.get_default_generators(device)
        state = generator.get_state()
        seed, offset = (int(x) for x in state.view(torch.int64))
        generator.set_state(torch.tensor([seed, offset + (increment + 3) // 4 * 4],
                                         dtype=torch.int64).view(torch.uint8))
        return seed, offset

    get_seed_and_offset._ax_fixed = True
    return get_seed_and_offset


def install():
    if os.environ.get("AX_FI_OFFSET_FIX", "1") != "1":
        return
    try:
        import flashinfer.sampling as fs
    except ImportError:
        return
    if getattr(fs.get_seed_and_offset, "_ax_fixed", False):
        return
    try:
        from importlib.metadata import version
        v = tuple(int(x) for x in version("flashinfer-python").split(".")[:3] if x.isdigit())
    except Exception:
        v = ()
    if v and v >= (0, 7, 3):
        return
    if not hasattr(fs, "get_default_generators"):
        raise RuntimeError("ax-engine: flashinfer.sampling has no get_default_generators; "
                           "re-check get_seed_and_offset (AX_FI_OFFSET_FIX=0 to skip)")
    fix = _fixed(fs)
    for name in _MODULES:
        m = sys.modules.get(name)
        if m is None and name != "flashinfer.sampling":
            try:
                m = __import__(name, fromlist=["_"])
            except ImportError:
                continue
        if m is not None and hasattr(m, "get_seed_and_offset"):
            patch_attribute(m, 'get_seed_and_offset', fix)
