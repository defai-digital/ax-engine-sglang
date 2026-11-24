"""Check native and in-place DFlash replay indices on SGLang 0.5.23."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_dflash_plan import _graph_inputs, install


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("This check requires CUDA")

    from sglang.kernels.ops.attention.kv_indices import create_flashinfer_kv_indices_triton
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    device = torch.device("cuda", torch.cuda.current_device())
    table = torch.arange(1280, device=device, dtype=torch.int32).reshape(20, 64)
    table[0].zero_()

    class Translator:
        def pack_read_stream(self, plan, *, req_pool_indices, seq_lens, indptr, out, **_):
            rows = seq_lens.numel()
            create_flashinfer_kv_indices_triton[(rows, 1)](
                table, req_pool_indices, seq_lens, indptr, None, out, table.stride(0),
                ENTRY_PAGE_SIZE=1, TOKEN_BLOCK_PARALLEL=False,
            )

    native = DFlashVerifyInput.generate_attn_arg_prefill
    install()
    # Real lengths change at the same capture shape. Row zero stands for padding.
    cases = (
        ([8, 5, 9, 1, 11, 3, 0, 0], [9, 13, 31, 21, 22, 18, 1, 1]),
        ([5, 8, 3, 7, 0, 0, 0, 0], [4, 5, 26, 13, 1, 1, 1, 1]),
        ([1, 6, 9, 4, 3, 5, 7, 8], [51, 19, 36, 19, 9, 24, 19, 9]),
    )
    backing = torch.full((1024,), -1, device=device, dtype=torch.int32)
    wrapper = SimpleNamespace(_paged_kv_indices_buf=backing)
    spec = DFlashVerifyInput(draft_token=None, positions=None, draft_token_num=8)
    first_offsets = None
    for requests, lengths in cases:
        tokens = sum(lengths) + len(requests) * spec.draft_token_num
        args = dict(
            req_pool_indices=torch.tensor(requests, device=device, dtype=torch.int32),
            paged_kernel_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
            paged_kernel_lens_sum=sum(lengths), translator=Translator(), plan=None,
        )
        expected = native(spec, **args)
        backing.fill_(-1)
        host, device_offsets, captured_ids = _graph_inputs(wrapper, spec, len(requests), sum(lengths), device)
        first_offsets = device_offsets if first_offsets is None else first_offsets
        assert device_offsets is first_offsets
        actual = spec.generate_attn_arg_prefill(**args, qo_indptr_buf=device_offsets, kv_indices_buf=captured_ids)
        assert actual[0].data_ptr() == backing.data_ptr()
        assert host.tolist() == expected[2].tolist()
        for left, right in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert actual[3] is expected[3] is None
        assert bool((backing[tokens:] == -1).all())

    print("DFlash replay page IDs and offsets match native across changing and padded batches")


if __name__ == "__main__":
    main()
