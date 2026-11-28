"""Device-side greedy walk over compact AX Draft candidates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _argmax(values, columns, k: tl.constexpr):
    # PyTorch argmax selects the first NaN, then breaks finite ties to the left.
    first_nan = tl.min(tl.where((columns < k) & (values != values), columns, 8), axis=0)
    return tl.where(first_nan < 8, first_nan, tl.argmax(values, axis=0, tie_break_left=True))


@triton.jit
def _select(scores, candidates, edges, vocabulary, output, depth: tl.constexpr, k: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, 8)
    offset = row * depth * k
    values = tl.load(scores + offset + columns, columns < k, other=-float("inf"))
    selected = _argmax(values, columns, k)
    candidate = tl.load(candidates + offset + selected)
    tl.store(output + row * depth, tl.load(vocabulary + candidate))
    for position in tl.static_range(1, depth):
        unary = tl.load(scores + offset + position * k + columns, columns < k, other=-float("inf"))
        edge = (row * (depth - 1) + position - 1) * k * k + selected * k
        correction = tl.load(edges + edge + columns, columns < k, other=0.0)
        selected = _argmax(unary + correction, columns, k)
        candidate = tl.load(candidates + offset + position * k + selected)
        tl.store(output + row * depth + position, tl.load(vocabulary + candidate))


def select_chain(scores, candidates, edges, vocabulary):
    batch, depth, k = scores.shape
    if (not 1 <= k <= 8 or depth < 1 or candidates.shape != scores.shape
            or edges.shape != (batch, depth - 1, k, k)
            or scores.dtype != torch.float32 or edges.dtype != torch.float32
            or candidates.dtype != torch.int64 or vocabulary.dtype != torch.int64
            or vocabulary.ndim != 1
            or any(t.device != scores.device or not t.is_contiguous()
                   for t in (scores, candidates, edges, vocabulary))):
        raise ValueError("invalid compact AX Draft candidate tensors")
    output = torch.empty((batch, depth), device=scores.device, dtype=vocabulary.dtype)
    if batch:
        with torch.cuda.device(scores.device):
            _select[(batch,)](scores, candidates, edges, vocabulary, output, depth, k, num_warps=1)
    return output
