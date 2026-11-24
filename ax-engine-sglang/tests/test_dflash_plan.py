from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.ax_dflash_plan import _graph_inputs
from ax_engine_sglang.compat import rebuild_function


class DeferredAnnotation:
    def run(self, value: RuntimeOnlyType):
        return value + 1


class DFlashPlanTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0"
        self.backing = MagicMock(device=self.device, spec=["device", "numel", "__getitem__"])
        self.backing.numel.return_value = 1024
        self.wrapper = SimpleNamespace(_paged_kv_indices_buf=self.backing)
        self.spec = SimpleNamespace(draft_token_num=8, custom_mask=None, ragged_verify_layout=None)
        self.host_offsets = Mock()
        self.torch = SimpleNamespace(arange=Mock(return_value=self.host_offsets), int32=object())

    def test_fixed_offsets_reused_with_fresh_live_page_length(self):
        with patch.dict(sys.modules, torch=self.torch):
            host, device, _ = _graph_inputs(self.wrapper, self.spec, 8, 220, self.device)
            again_host, again_device, _ = _graph_inputs(self.wrapper, self.spec, 8, 240, self.device)
        self.assertIs(host, again_host)
        self.assertIs(device, again_device)
        self.torch.arange.assert_called_once_with(0, 72, 8, dtype=self.torch.int32, device="cpu")
        self.host_offsets.to.assert_called_once_with(self.device)
        self.assertEqual(self.backing.__getitem__.call_args_list[0].args, (slice(None, 284),))
        self.assertEqual(self.backing.__getitem__.call_args_list[1].args, (slice(None, 304),))

    def test_falls_back_on_capacity_and_custom_layout(self):
        self.assertIsNone(_graph_inputs(self.wrapper, self.spec, 8, 1000, self.device))
        self.assertIsNone(_graph_inputs(self.wrapper, self.spec, 8, 200, "cuda:1"))
        self.spec.ragged_verify_layout = object()
        self.assertIsNone(_graph_inputs(self.wrapper, self.spec, 8, 200, self.device))
        self.spec.ragged_verify_layout = None
        self.spec.custom_mask = object()
        self.assertIsNone(_graph_inputs(self.wrapper, self.spec, 8, 200, self.device))
        self.assertFalse(hasattr(self.wrapper, "_ax_dflash_qo_offsets"))

    def test_rebuild_preserves_deferred_annotations(self):
        run = rebuild_function(DeferredAnnotation, "run", (("value + 1", "value + 2", 1),),
                               feature="test")
        self.assertEqual(run(DeferredAnnotation(), 3), 5)
        self.assertEqual(run.__annotations__, DeferredAnnotation.run.__annotations__)
