"""Stable QSA selection without the 128-KiB FilteredTopK shared-memory requirement.

This opt-in correctness fallback uses fixed tensor shapes, including during CUDA
graph replay. It returns relative block indices, pads short rows with -1, and
breaks equal scores in favor of the smaller relative index.
"""


def stable_topk(logits, row_starts, row_ends, topk):
    import torch

    if logits.ndim != 2 or topk <= 0:
        raise ValueError("QSA selection needs a score matrix and positive topk")
    rows, columns = logits.shape
    if row_starts.numel() != rows or row_ends.numel() != rows:
        raise ValueError("QSA row bounds must match the score matrix")
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    if not rows or not columns:
        return output
    starts = row_starts.to(device=logits.device, dtype=torch.long).reshape(rows, 1)
    ends = row_ends.to(device=logits.device, dtype=torch.long).reshape(rows, 1)
    relative = torch.arange(columns, device=logits.device).reshape(1, columns)
    absolute = starts + relative
    valid = (absolute >= 0) & (absolute < columns) & (absolute < ends)
    scores = logits.gather(1, absolute.clamp(0, columns - 1))
    scores = torch.where(valid, scores, float("-inf"))
    # Valid entries precede invalid tail entries in relative order. Stable sort
    # therefore also handles valid -inf values without selecting padded entries.
    selected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(topk, columns)]
    selected_valid = valid.gather(1, selected)
    output[:, :selected.shape[1]] = torch.where(selected_valid, selected, -1).to(torch.int32)
    return output
