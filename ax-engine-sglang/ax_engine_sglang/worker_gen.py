# SPDX-License-Identifier: Apache-2.0
"""Build the tree worker from the installed SGLang instead of shipping a copy of it.

The DForkTree release carries a 1,357-line copy of SGLang 0.5.22's `dflash_worker_v2.py` with four
small edits. A copy pins the plugin to one SGLang version, so this module makes the worker at
start-up from whatever SGLang is installed: it reads `sglang/srt/speculative/dflash_worker_v2.py`,
normalises it with `ast.unparse`, applies the four edits of the release (and three for several requests, see
ax_chain.py) and loads the result as the module
`fc1_dflash_worker`. Every edit must match exactly once; otherwise start-up fails with the name of
the edit, so an SGLang release that moved the code cannot silently run without the tree branch.

Checked: for SGLang 0.5.22 the generated module has the same AST as the release's file
(`python worker_gen.py --check <release fc1_dflash_worker.py>`).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import linecache
import sys
import types
from pathlib import Path
from ax_engine_sglang.compat import TESTED_SGLANG, patch_item

MODULE = "fc1_dflash_worker"
STOCK = "sglang.srt.speculative.dflash_worker_v2"
TESTED = TESTED_SGLANG

# (name, text in the normalised stock source, replacement)
EDITS = [
    ("verify input with tree links",
     "from sglang.srt.speculative.dflash_info import DFlashVerifyInput\n",
     "from fc1_verify import ChainTreeVerifyInput as DFlashVerifyInput\n"),
    ("drafter block from FC1_DRAFT_BLOCK",
     "            self.block_size = int(get_spec().speculative_num_draft_tokens)\n",
     "            self.block_size = int(os.environ.get('FC1_DRAFT_BLOCK', '24')) if os.environ.get('FC1_TREE64') == '1' "
     "else int(get_spec().speculative_num_draft_tokens)\n"),
    ("tree verify branch",
     "        grammar_tree = GrammarTree.from_linear_chain(draft_tokens) if batch.has_grammar else None\n",
     "        grammar_tree = GrammarTree.from_linear_chain(draft_tokens) if batch.has_grammar else None\n"
     "        if __import__('os').environ.get('FC1_BRANCH_PILOT') == '1':\n"
     "            import ax_chain\n"
     "            draft_features_ready = self._draft_sampler is not None and draft_out.can_run_graph\n"
     "            if ax_chain.choose(self, batch, draft_features_ready=draft_features_ready) in ('tree', 'btree'):\n"
     "                from fc1_branch64 import verify_branch\n"
     "                return verify_branch(self, batch, draft_tokens, prefix_lens, positions, kv_loc_plan, "
     "verify_out_cache_loc_2d, on_publish)\n"
     "            draft_tokens, positions, verify_out_cache_loc, verify_out_cache_loc_2d, kv_loc_plan, block_size = "
     "ax_chain.narrow(self, batch, draft_tokens, positions, verify_out_cache_loc_2d, prefix_lens, kv_loc_plan, block_size)\n"
     "            grammar_tree = GrammarTree.from_linear_chain(draft_tokens) if batch.has_grammar else None\n"),
    ("restore the drafter block at the start of a call (ax_chain)",
     "        self._validate_phase1_sampling_support(batch)\n",
     "        self._validate_phase1_sampling_support(batch)\n        __import__('ax_chain').begin_round(self, batch)\n"),
    ("record the accepted length of a chain round (ax_chain)",
     "        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(",
     "        __import__('ax_chain').record_accept(self, batch, candidates, accept_len)\n"
     "        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits("),
    ("keep the drafter logits for the tree builder",
     "        logits = torch.matmul(hs, self.weight[:self.num_org].T)\n",
     "        logits = torch.matmul(hs, self.weight[:self.num_org].T)\n        self.fc1_logits = logits\n"),
]


def stock_path():
    spec = importlib.util.find_spec(STOCK)
    if spec is None or not spec.origin:
        raise RuntimeError(f"ax-engine-sglang: {STOCK} not found; is SGLang installed?")
    return spec.origin


def generate(path=None):
    """Source of the tree worker for the SGLang file at `path` (default: the installed one)."""
    src = ast.unparse(ast.parse(Path(path or stock_path()).read_text())) + "\n"
    src = src.replace("DFlashWorkerV2", "DFlashTreeWorkerV2")
    for name, old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"ax-engine-sglang: cannot build the tree worker from {path or stock_path()}: the anchor of "
                               f"'{name}' occurs {n} times (expected 1). This SGLang version needs a port.")
        src = src.replace(old, new)
    return src


def install():
    """Load the generated worker as `fc1_dflash_worker` (once)."""
    if MODULE in sys.modules:
        return sys.modules[MODULE]
    import sglang
    path = stock_path()
    src = generate(path)
    sha = hashlib.sha256(src.encode()).hexdigest()[:12]
    version = getattr(sglang, "__version__", "?")
    name = f"<ax-engine-sglang worker from sglang {version} sha256 {sha}>"
    patch_item(linecache.cache, name, (len(src), None, src.splitlines(True), name))
    mod = types.ModuleType(MODULE)
    mod.__file__ = name
    patch_item(sys.modules, MODULE, mod)
    try:
        exec(compile(src, name, "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(MODULE, None)
        raise
    note = "" if version in TESTED else f" (untested SGLang version; tested: {', '.join(TESTED)})"
    print(f"[ax-engine] tree worker built from sglang {version}, sha256 {sha}{note}", flush=True)
    return mod


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", help="dflash_worker_v2.py to build from (default: the installed SGLang)")
    ap.add_argument("--check", help="a reference fc1_dflash_worker.py; exit 1 unless the ASTs are equal")
    ap.add_argument("--out", help="write the generated source here")
    args = ap.parse_args()
    out = generate(args.stock)
    print("generated", len(out.splitlines()), "lines, sha256", hashlib.sha256(out.encode()).hexdigest()[:12])
    if args.out:
        open(args.out, "w").write(out)
    if args.check:
        same = ast.dump(ast.parse(out)) == ast.dump(ast.parse(open(args.check).read()))
        print("AST equal to", args.check, ":", same)
        sys.exit(0 if same else 1)
