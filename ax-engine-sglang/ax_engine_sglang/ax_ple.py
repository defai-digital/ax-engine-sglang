# SPDX-License-Identifier: Apache-2.0
"""Start every boot's file-backed PLE table from a fresh sparse file.

With `--ple-offload-embedding --ple-offload-backend file`, SGLang 0.5.23 keeps Flash Next's PLE
table (51 GB FP8 for the NVFP4 checkpoint) in an mmapped file that is reused across restarts, and
every boot rewrites the whole table through the weight loader. The mapping carries MADV_RANDOM, so
once the file holds data each written page first costs a synchronous 4 KB read with no readahead.
On a Jetson Thor NVMe a 2 GB test wrote 27 MB/s into a filled file and 1549 MB/s into a sparse
one; in the server the 14 PLE shards took 24 of a 29-minute target load.

Before SGLang maps the file, this hook unlinks an existing table file so SGLang creates a new
sparse one. The contents are rewritten in full on every boot either way. Unlinking instead of
truncating in place leaves a server that still maps the old file on its own inode, so it cannot
fault past the end of a shrunk file. AX_PLE_FRESH_FILE=0 leaves SGLang as is.

The file name depends only on shape, dtype and tag, so with more than one rank (torch.distributed
world size or --dp-size above one) the ranks could share a name; one rank unlinking it while another
writes it would split them onto separate inodes. The hook therefore acts only in a single-rank
server. Separate servers must not share a PLE directory (SGLang's own rule: the directory is private
to one checkpoint, by default $SGLANG_CACHE_DIR/ple/<model path>); a second server booting on the
same directory would unlink the file the first one maps.
"""
import os
import sys

from ax_engine_sglang.ax_ranks import server_ranks
from ax_engine_sglang.compat import patch_attribute


def _fresh(allocate, table_module, world_size=server_ranks):
    def allocate_ple_host_table(shape, dtype, backend="pinned", table_dir=None, tag=None):
        ranks = world_size() if backend == "file" else 1
        if ranks > 1:
            print(f"ax-engine-sglang: PLE table file kept with {ranks} ranks", file=sys.stderr)
        elif backend == "file":
            # Resolved exactly as allocate_ple_host_table does.
            directory = os.path.expanduser(
                table_dir or table_module.envs.SGLANG_QWEN4_PLE_FILE_DIR.get())
            path = os.path.join(directory, table_module.ple_table_file_name(shape, dtype, tag))
            try:
                os.unlink(path)
                print(f"ax-engine-sglang: PLE table file {path} replaced by a sparse file "
                      "(AX_PLE_FRESH_FILE=0 keeps it)", file=sys.stderr)
            except FileNotFoundError:
                pass
        return allocate(shape, dtype, backend=backend, table_dir=table_dir, tag=tag)

    allocate_ple_host_table._ax_fresh = True
    return allocate_ple_host_table


def install():
    if os.environ.get("AX_PLE_FRESH_FILE", "1") == "0":
        return
    try:
        import sglang.srt.models.qwen4_exp_ple_table as table_module
    except ImportError:
        return   # an SGLang without the Qwen4-Exp PLE file backend
    allocate = table_module.allocate_ple_host_table
    if getattr(allocate, "_ax_fresh", False):
        return
    fresh = _fresh(allocate, table_module)
    patch_attribute(table_module, "allocate_ple_host_table", fresh)
    # qwen4_exp imports the function by name; once loaded, its binding must be patched too. Importing
    # it here would load the whole model module for every model, so a later import picks up `fresh`.
    model_module = sys.modules.get("sglang.srt.models.qwen4_exp")
    if model_module is not None and getattr(model_module, "allocate_ple_host_table", None) is allocate:
        patch_attribute(model_module, "allocate_ple_host_table", fresh)
