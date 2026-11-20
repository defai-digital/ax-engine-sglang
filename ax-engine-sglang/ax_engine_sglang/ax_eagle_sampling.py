"""Match FlashInfer AR top-k/top-p filtering in SGLang's EAGLE verifier."""
import ast
import inspect
import sys
from textwrap import dedent, indent

from ax_engine_sglang.compat import patch_attribute, patch_item


_STOCK_FILTER = """
if sampling_info.need_top_k_sampling:
    target_probs = top_k_renorm_prob(
        target_probs,
        torch.repeat_interleave(sampling_info.top_ks, verify_input.draft_token_num, dim=0),
    )
    maybe_detect_nan(target_probs, "v2 verify: target_probs after top_k_renorm")
if sampling_info.need_top_p_sampling:
    target_probs = top_p_renorm_prob(
        target_probs,
        torch.repeat_interleave(sampling_info.top_ps, verify_input.draft_token_num, dim=0),
    )
    maybe_detect_nan(target_probs, "v2 verify: target_probs after top_p_renorm")
"""


def joint_probs(probs, top_ks, top_ps, top_k_renorm, top_p_renorm):
    """Intersect the support of both filters applied to the original distribution."""
    kept = top_k_renorm(probs, top_ks)
    kept.masked_fill_(top_p_renorm(probs, top_ps) <= 0, 0)
    return kept / kept.sum(dim=-1, keepdim=True)


def _patched_source(original):
    source = ast.unparse(ast.parse(dedent(original)))
    stock = indent(ast.unparse(ast.parse(_STOCK_FILTER)), "        ")
    if source.count(stock) != 1:
        raise RuntimeError("ax-engine-sglang: EAGLE sampling filter changed in SGLang")
    joint = dedent("""
        from sglang.srt.runtime_context import get_exec
        if (get_exec().kernel.sampling_backend == 'flashinfer'
                and not use_rejection_sampling and not sampling_info.need_min_p_sampling
                and sampling_info.need_top_k_sampling and sampling_info.need_top_p_sampling):
            target_probs = _ax_joint_probs(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, verify_input.draft_token_num, dim=0),
                torch.repeat_interleave(sampling_info.top_ps, verify_input.draft_token_num, dim=0),
                top_k_renorm_prob, top_p_renorm_prob,
            )
            maybe_detect_nan(target_probs, 'v2 verify: target_probs after joint filtering')
        else:
    """).strip("\n")
    return "from __future__ import annotations\n" + source.replace(
        stock, indent(joint, "        ") + "\n" + indent(stock, "    "), 1)


def install():
    from sglang.srt.speculative import eagle_utils

    original = eagle_utils.eagle_sample
    if getattr(original, "_ax_joint_sampling", False):
        return
    source = _patched_source(inspect.getsource(original))
    compiled = compile(source, "<ax-engine-sglang EAGLE joint sampling>", "exec")
    patch_item(original.__globals__, "_ax_joint_probs", joint_probs)
    local = {}
    exec(compiled, original.__globals__, local)
    patched = local["eagle_sample"]
    patched._ax_joint_sampling = True
    patch_attribute(eagle_utils, "eagle_sample", patched)
    for name in ("eagle_worker_common", "ngram_worker"):
        consumer = sys.modules.get(f"sglang.srt.speculative.{name}")
        if consumer is not None and getattr(consumer, "eagle_sample", None) is original:
            patch_attribute(consumer, "eagle_sample", patched)
