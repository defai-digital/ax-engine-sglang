# SPDX-License-Identifier: Apache-2.0
# See NOTICE for upstream attribution.
# ax-engine-sglang (`SGLANG_PLUGINS=ax-engine`); adapted from DForkTree v0.1.0 fc1_plugin.py.
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from .config import get_config, startup_config
from .compat import check_sglang, installation, patch_attribute
from .runtime import current_runtime
_registered = False
_tree_installed = False

class TreeSpec(CustomSpecAlgo):

    def handle_server_args(self, server_args):
        config = get_config()
        if config.ssm_positions and not server_args.disable_radix_cache:
            raise ValueError("AX_SSM_POSITIONS requires --disable-radix-cache")
        _install_tree()
        SpeculativeAlgorithm.DFLASH.handle_server_args(server_args)
        from sglang.srt.arg_groups.overrides import declare_resolution
        # A chain-only server captures its main graph with the same backend mode it uses at replay.
        declare_resolution(server_args, 'fc1_chain_tree', speculative_eagle_topk=2 if config.tree else 1)
        if config.tree64:
            declare_resolution(server_args, 'fc1_tree64', speculative_num_draft_tokens=config.tree_nodes, speculative_num_steps=config.draft_block - 1)
for name, value in vars(SpeculativeAlgorithm).items():
    if name.startswith(('is_', 'supports_')) or name in ('need_topk', 'has_draft_kv', 'carries_draft_hidden_states', 'create_future_map', 'build_disagg_draft_input', 'get_num_tokens_per_req_for_target_verify'):

        def delegate(self, *args, _name=name, **kwargs):
            return getattr(SpeculativeAlgorithm.DFLASH, _name)(*args, **kwargs)
        setattr(TreeSpec, name, delegate)

def register():
    global _registered
    if _registered:
        return
    # Validate every cross-option constraint before changing upstream objects.
    config, provisional = startup_config()
    check_sglang(config)
    with installation():
        from . import defaults   # ax-engine-sglang: model/GPU-derived launch defaults (AX_AUTO_DEFAULTS=0 disables)
        defaults.install()
        _install_fixes(config)
        # A provisional launcher config still lacks the resolved defaults; TreeSpec.handle_server_args
        # installs the tree after resolution. Scheduler processes inherit the resolved environment.
        if config.tree_requested and not provisional:
            _install_tree()

        @SpeculativeAlgorithm.register('FC1_DFLASH_TREE', supports_overlap=True, spec_class=TreeSpec)
        def factory(server_args):
            _install_tree()
            from fc1_dflash_worker import DFlashTreeWorkerV2
            return DFlashTreeWorkerV2
    _registered = True


def _install_fixes(config):
    import sys
    directory = str(__import__('pathlib').Path(__file__).resolve().parent)
    patch_attribute(sys, 'path', [directory, *(p for p in sys.path if p != directory)])
    import ax_rng   # ax-engine-sglang: flashinfer sampling offset reuse (sampled decoding), AX_FI_OFFSET_FIX
    ax_rng.install()
    if config.gdn_chain_replay:
        from . import ax_gdn_chain_replay
        ax_gdn_chain_replay.install()
    if config.nvfp4_verify_parity:
        from . import ax_nvfp4_verify, ax_verify_attention
        ax_nvfp4_verify.install(config.nvfp4_parity_rows)
        ax_verify_attention.install()
    if config.eagle_joint_sampling:
        from . import ax_eagle_sampling
        ax_eagle_sampling.install()
    if config.gdn_unpacked_decode:
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
        patch_attribute(TritonGDNKernel, "supports_packed_decode", False)
    if config.nvfp4_verify_parity:
        from .ax_gdn_recurrent import install_value_tile
        install_value_tile()
    import ax_qsa   # ax-engine-sglang: AX_QSA_RING widens the QSA pending ring (Qwen4-Exp verify windows)
    ax_qsa.install()


def _install_tree():
    global _tree_installed
    if _tree_installed:
        return
    config = get_config()
    check_sglang(config)
    with installation():
        _install_tree_hooks(config)
        patch_attribute(__import__('sys').modules[__name__], '_tree_installed', True)


def _install_tree_hooks(config):
    if config.dflash_joint_sampling:
        from . import ax_dflash_sampling
        ax_dflash_sampling.install()
    import ax_state_cap   # ax-engine-sglang: AX_SSM_POSITIONS caps the intermediate SSM cache (64 slots)
    ax_state_cap.install()
    import worker_gen   # ax-engine-sglang: the tree worker is built from the installed SGLang
    worker_gen.install()
    if config.fused:
        from fc1_fused import install
        install()
        from fc1_conv import install as install_conv
        install_conv()
    if config.gdn_recurrent_verify:
        from . import ax_gdn_recurrent
        ax_gdn_recurrent.install()
    if config.fused or config.gdn_recurrent_verify:
        from fc1_plan import install as install_plan
        install_plan()
    import ax_lattice   # ax-engine-sglang: wraps the DFlash2 selector sampler when AX_LATTICE=1
    ax_lattice.install()
    import ax_chain   # ax-engine-sglang: several running requests (chain rounds)
    ax_chain.install()
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    for attr in ('retrieve_next_token', 'retrieve_next_sibling', 'retrieve_parent_token'):
        if not hasattr(DFlashVerifyInput, attr):
            patch_attribute(DFlashVerifyInput, attr, None)
    if config.branch_pilot:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        original = DecodeCudaGraphRunner.get_spec_info

        def tree_capture_info(self, num_tokens):
            if (not ax_chain.is_tree_runner(self) or self.model_runner.is_draft_worker
                    or not config.tree or getattr(self, '_ax_chain_width', None) is not None
                    or current_runtime().kind == 'chain'):
                return original(self, num_tokens)   # ax-engine-sglang: chain runners capture the stock input
            return EagleVerifyInput(draft_token=None, custom_mask=self.buffers.custom_mask, positions=None, retrieve_index=None, retrieve_next_token=None, retrieve_next_sibling=None, retrieve_cum_len=None, spec_steps=config.draft_block - 1, topk=2, draft_token_num=self.speculative_num_draft_tokens, capture_hidden_mode=CaptureHiddenMode.FULL, seq_lens_sum=None, seq_lens_cpu=None)
        patch_attribute(DecodeCudaGraphRunner, 'get_spec_info', tree_capture_info)

def _verify_width(self, num_draft_tokens, is_draft_worker):
    config = get_config()
    if config.tree64 and is_draft_worker:
        override = current_runtime().draft_width_override
        if override is not None:   # ax-engine-sglang: a narrower drafter graph (ax_chain._build_draft_runners)
            return int(override)
        return config.draft_block
    return num_draft_tokens
TreeSpec.get_num_tokens_per_req_for_target_verify = _verify_width
