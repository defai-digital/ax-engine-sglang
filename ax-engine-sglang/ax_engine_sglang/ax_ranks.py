# SPDX-License-Identifier: Apache-2.0
"""Count the scheduler processes that serve one model in this server."""


def server_ranks():
    """The larger of the torch.distributed world size and SGLang's --dp-size.

    Data-parallel replicas without DP attention each run their own one-rank process group, so the
    world size alone reports 1 for each of them.
    """
    world = 1
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
    except ImportError:
        pass
    dp = 1
    try:
        from sglang.srt.server_args import get_global_server_args
        dp = int(getattr(get_global_server_args(), "dp_size", 1) or 1)
    except Exception:   # no SGLang, or server args not set in this process
        pass
    return max(world, dp)
