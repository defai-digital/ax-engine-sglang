from array import array
import importlib.util
from types import SimpleNamespace
import unittest

from ax_engine_sglang.ax_dflash_draft_vocab import HISTORY, LEADING_ROWS, TAIL_START, VOCAB_SIZE
from ax_engine_sglang.ax_dflash_draft_vocab import history_in_subset, pack_weight, subset_topk, use_subset
from ax_engine_sglang.config import PluginConfig


class DraftVocabularyTests(unittest.TestCase):
    def test_id_ranges_include_global_special_token_tail(self):
        self.assertTrue(history_in_subset([0, LEADING_ROWS - 1, TAIL_START, VOCAB_SIZE - 1], []))
        for token in (-1, LEADING_ROWS, TAIL_START - 1, VOCAB_SIZE):
            with self.subTest(token=token):
                self.assertFalse(history_in_subset([token], []))
        self.assertFalse(history_in_subset([], []))

    def test_recent_history_combines_prompt_and_committed_tokens(self):
        excluded = LEADING_ROWS
        prompt = array("q", [excluded, 1, 2])
        self.assertFalse(history_in_subset(prompt, array("q", [3] * (HISTORY - 3))))
        self.assertTrue(history_in_subset(prompt, array("q", [3] * (HISTORY - 2))))
        self.assertFalse(history_in_subset(prompt, array("q", [3] * HISTORY + [excluded])))
        self.assertTrue(history_in_subset(prompt, array("q", [excluded] + [3] * HISTORY)))

    def test_batch_requires_greedy_and_all_recent_histories_in_subset(self):
        mode = SimpleNamespace(is_idle=lambda: False, is_extend=lambda: False)
        greedy = SimpleNamespace(is_all_greedy=True)
        req = SimpleNamespace(origin_input_ids=[1], output_ids=[2])
        batch = SimpleNamespace(forward_mode=mode, is_extend_in_batch=False, has_grammar=False,
                                sampling_info=greedy, reqs=[req, req])
        self.assertTrue(use_subset(batch))
        for changes in ({"has_grammar": True}, {"is_extend_in_batch": True}, {"reqs": []},
                        {"sampling_info": SimpleNamespace(is_all_greedy=False)},
                        {"reqs": [req, SimpleNamespace(origin_input_ids=[LEADING_ROWS], output_ids=[])]}):
            with self.subTest(changes=changes):
                self.assertFalse(use_subset(SimpleNamespace(**{**vars(batch), **changes})))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for the tensor check")
    def test_projection_excludes_gap_and_remaps_tail_ids(self):
        import torch

        weight = torch.zeros(VOCAB_SIZE, 1)
        weight[LEADING_ROWS] = 10
        weight[LEADING_ROWS - 1] = 2
        weight[TAIL_START] = 3
        head = SimpleNamespace(quant_method=None, weight=pack_weight(weight))
        ids, values = subset_topk(torch.ones(1, 1), head, 2, torch.topk)
        self.assertEqual(ids.tolist(), [[TAIL_START, LEADING_ROWS - 1]])
        self.assertEqual(values.tolist(), [[3.0, 2.0]])

    def test_requires_native_replay_and_is_opt_in(self):
        self.assertFalse(PluginConfig.from_env({}).dflash_draft_vocab)
        with self.assertRaisesRegex(ValueError, "requires AX_GDN_CHAIN_REPLAY"):
            PluginConfig.from_env({"AX_DFLASH_DRAFT_VOCAB": "1"})
        config = PluginConfig.from_env({"AX_DFLASH_DRAFT_VOCAB": "1", "AX_GDN_CHAIN_REPLAY": "1"})
        self.assertTrue(config.dflash_draft_vocab)
