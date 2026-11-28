"""AX Draft proposals with the SGLang 0.5.23 chain verifier."""

import inspect
import json
from pathlib import Path
import textwrap

from .compat import check_sglang, patch_attribute, rebuild_function
from .config import get_config
from .defaults import AX_DRAFT_VERIFY_WIDTHS


class ProposalModel:
    """Run the independent AX checkpoint. Each depth scores a compact vocabulary."""

    def __init__(self, directory, target, device):
        import torch
        from safetensors.torch import load_file

        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        self.depth = config["depth"]
        self.hidden_size = config["hidden_size"]
        vocabulary = config["vocabulary"]
        if (config["version"] != 1 or self.depth != 7 or self.hidden_size != target.hidden_size
                or config["target_vocab_size"] != target.vocab_size
                or len(vocabulary) < 4 or len(set(vocabulary)) != len(vocabulary)
                or min(vocabulary) < 0 or max(vocabulary) >= target.vocab_size):
            raise ValueError("invalid AX Draft checkpoint for the target")
        width, size, rank = config["width"], len(vocabulary), config["rank"]
        if min(width, rank) < 1:
            raise ValueError("invalid AX Draft dimensions")
        self.rank_scale = rank ** -0.5
        shapes = {
            "context.weight": (width, self.hidden_size), "anchor.weight": (size + 1, width),
            "mix.weight": (width, width), "mix.bias": (width,), "depth": (self.depth, width),
            "output.weight": (size, width), "output.bias": (size,),
            "parent.weight": (size, rank), "child": (size, rank),
        }
        tensors = load_file(directory / "model.safetensors")
        for name, shape in shapes.items():
            if name not in tensors or tensors[name].shape != shape or tensors[name].dtype != torch.float32:
                raise ValueError(f"invalid AX Draft tensor {name}")
        self.weights = {name: tensors[name].to(device) for name in shapes}
        self.vocabulary = torch.tensor(vocabulary, device=device)
        self.classes = torch.full((target.vocab_size,), size, dtype=torch.long, device=device)
        self.classes[self.vocabulary] = torch.arange(size, device=device)
        self.select_chain = None
        if self.vocabulary.is_cuda and torch.version.hip is None:
            from .ax_draft_kernels import select_chain

            self.select_chain = select_chain

    def __call__(self, hidden, anchor, depth=None):
        import torch
        from torch.nn import functional as F

        w = self.weights
        depth = self.depth if depth is None else depth
        normalized = F.rms_norm(hidden.float(), (self.hidden_size,), eps=1e-5)
        shared = F.linear(normalized, w["context.weight"]) + w["anchor.weight"][self.classes[anchor]]
        shared = shared + F.silu(F.linear(shared, w["mix.weight"], w["mix.bias"]))
        queries = F.silu(shared[:, None, :] + w["depth"][:depth])
        logits = F.linear(queries, w["output.weight"], w["output.bias"])
        if depth == 1:
            return self.vocabulary[logits.argmax(dim=-1)]
        # Select each position conditioned on the actual proposed parent.
        # The parallel logits and all shortlist edges are computed only once.
        scores, candidates = logits.topk(min(8, self.vocabulary.numel()), dim=-1)
        parent = w["parent.weight"][candidates[:, :-1]]
        child = w["child"][candidates[:, 1:]]
        edges = (parent @ child.transpose(-1, -2)) * self.rank_scale
        if self.select_chain is not None:
            return self.select_chain(scores, candidates, edges, self.vocabulary)
        picks = [scores[:, 0].argmax(-1)]
        lanes = torch.arange(anchor.numel(), device=anchor.device)
        for position in range(1, depth):
            picks.append((scores[:, position] + edges[lanes, position - 1, picks[-1]]).argmax(-1))
        selected = torch.gather(candidates, 2, torch.stack(picks, dim=1).unsqueeze(-1)).squeeze(-1)
        return self.vocabulary[selected]


def _forward(native):
    cls = native.DFlashWorkerV2
    name = "forward_batch_generation"
    source = textwrap.dedent(inspect.getsource(getattr(cls, name)))

    def section(start, end, inserted):
        start, end = "\n" + start, "\n" + end
        if source.count(start) != 1 or source.count(end) != 1:
            raise RuntimeError("AX Draft could not identify the SGLang 0.5.23 verifier boundaries")
        begin = source.index(start) + 1
        finish = source.index(end, begin) + 1
        return source[begin:finish], inserted, 1

    edits = (
        section("        # Materialize prompt tokens", "        # Avoid copying large",
                "        self._remember_prefill(batch, logits_output.hidden_states)\n\n"),
        section("    # --- 1) Draft a fixed block", "    block_size = int(self.block_size)", ""),
        section("    if self._full_embed_gpu is not None:", "    positions = positions_2d.reshape", ""),
        section("    seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]",
                "    draft_tokens = self._draft_block_tokens_buf[:bs]",
                "    draft_next = self._ax_model(self._ax_hidden[batch.req_pool_indices], "
                "block_ids[:, 0], block_size - 1)\n\n"),
        section("    # Consume in this step:", "    # Avoid copying large",
                "    self._remember_verify(batch, hidden, commit_lens)\n\n"),
    )
    return rebuild_function(cls, name, edits, feature="AX_DRAFT_MODEL")


def install(directory):
    if check_sglang(get_config()).split("+")[0] != "0.5.23":
        raise RuntimeError("AX Draft requires SGLang 0.5.23")
    import torch
    from transformers import AutoConfig, PretrainedConfig
    from sglang.srt.model_executor.model_runner_components import spec_aux_hidden_state
    from sglang.srt.speculative import dflash_worker_v2 as native

    original_aux = spec_aux_hidden_state._resolve_dflash_aux_hidden_state

    def resolve_aux(**args):
        # With no auxiliary taps the native logits processor exposes final post-norm.
        if not args["spec_algorithm"].is_dflash():
            return original_aux(**args)
        return None

    forward = _forward(native)

    class AxDraftConfig(PretrainedConfig):
        model_type = "ax_draft"

    class AxDraftWorker(native.DFlashWorkerV2):
        forward_batch_generation = forward

        def __init__(self, server_args, gpu_id, nccl_port, target_worker):
            native.BaseSpecWorker.__init__(self)
            self.server_args = server_args
            self.gpu_id = gpu_id
            self.nccl_port = nccl_port
            self._target_worker = target_worker
            self.model_runner = target_worker.model_runner
            self.device = target_worker.device
            self._draft_worker = None
            self._need_mamba_verify_commit = False
            self._warned_sampling_fallback = False
            self._selector_sample = None
            self.selector = self.lilicorr = None
            self._lilicorr_sampling_enabled = False
            self.block_size = self.speculative_num_draft_tokens = int(
                native.get_spec().speculative_num_draft_tokens
            )
            if self.block_size not in AX_DRAFT_VERIFY_WIDTHS:
                raise ValueError("AX Draft requires 2, 4 or 8 verify rows")
            self._ax_model = ProposalModel(directory, self.model_runner.model_config, self.device)
            self._tp_sync = native.SpecTpSync(native.get_dp_tp_group())
            native.check_simulate_acc_token_mode()
            self._block_pos_offsets = native.build_block_pos_offsets(length=self.block_size, device=self.device)
            self._draft_block_ids_buf = None
            self._mask_token_id = 0  # overwritten by the draft before verify
            self._use_triton_prepare_block = self._use_triton_accept_bonus = True
            self._accept_bonus_buffer_cap = self._accept_bonus_buffer_slot = 0
            self._accept_len_buf = None
            self._commit_lens_bufs, self._bonus_id_bufs = [], []
            self._out_tokens_bufs, self._new_seq_lens_bufs = [], []
            self._target_logits_processor = self._defer_target_verify_gather()
            native.logger.info("Using independent AX Draft checkpoint (%s); verifier width %d", directory,
                               self.block_size)

        def weight_update_runners(self):
            return []

        @property
        def spec_v2_attn_backends(self):
            return (self.model_runner.attn_backend,)

        def alloc_memory_pool(self, memory_pool_config=None, req_to_token_pool=None,
                              token_to_kv_pool_allocator=None):
            self._ax_hidden = torch.empty((req_to_token_pool.req_to_token.shape[0], self._ax_model.hidden_size),
                                          dtype=torch.bfloat16, device=self.device)

        def init_attention_backends(self):
            self._need_mamba_verify_commit = native.mambaish_config(
                self.model_runner.model_config
            ) is not None and hasattr(self.model_runner.attn_backend, "update_mamba_state_after_mtp_verify")

        def init_cuda_graphs(self):
            pass  # The target owns its verify graphs. Proposal scoring is eager.

        def _remember_prefill(self, batch, hidden):
            lengths = native.async_h2d(batch.extend_lens, dtype=torch.int64, device=hidden.device)
            last = lengths.cumsum(0) - 1
            self._ax_hidden.index_copy_(0, batch.req_pool_indices, hidden[last].to(self._ax_hidden.dtype))

        def _remember_verify(self, batch, hidden, commit_lens):
            lanes = torch.arange(commit_lens.numel(), device=hidden.device)
            last = hidden[lanes, commit_lens.to(torch.long) - 1]
            self._ax_hidden.index_copy_(0, batch.req_pool_indices, last.to(self._ax_hidden.dtype))

    AutoConfig.register("ax_draft", AxDraftConfig, exist_ok=True)
    patch_attribute(spec_aux_hidden_state, "_resolve_dflash_aux_hidden_state", resolve_aux)
    patch_attribute(native, "DFlashWorkerV2", AxDraftWorker)
