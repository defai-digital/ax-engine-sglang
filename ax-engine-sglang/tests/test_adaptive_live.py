import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ax_engine_sglang import ax_adaptive_live  # noqa: E402


class Slot:
    def __init__(self, initial_steps, cfg):
        self.candidate_steps = sorted(set(cfg["candidate_steps"]))
        self.current_steps = initial_steps if initial_steps in self.candidate_steps else self.candidate_steps[0]
        self.ema_alpha = cfg.get("ema_alpha", 0.2)


class Params:
    candidate_steps = [1, 2, 3, 4, 5, 6, 7]

    def __init__(self):
        self._bs_list = [1, 4]
        self._slots = {1: Slot(3, {"candidate_steps": self.candidate_steps}),
                       4: Slot(3, {"candidate_steps": self.candidate_steps})}
        self._cuda_graph_bs = None   # CUDA graphs off, as SGLang sets it before init_states
        self.calls = []
        self.slots_seen = []

    def _route(self, batch_size):
        return self._slots[max(b for b in self._bs_list if b <= batch_size) if batch_size >= self._bs_list[0]
                           else self._bs_list[0]]

    def on_verify_complete(self, drafts, batch_size):
        self.calls.append((drafts, batch_size))
        self.slots_seen.append(self._slots)
        return None


def module():
    return types.SimpleNamespace(AdaptiveStepSlot=Slot, AdaptiveSpeculativeParams=Params)


class TestAdaptiveLive(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "live.json")
        self.now = [0.0]
        self.state = ax_adaptive_live.LiveState(self.path, 0.5)
        self.verify = ax_adaptive_live._wrap(Params.on_verify_complete, module(), self.state,
                                             clock=lambda: self.now[0], world_size=lambda: 1)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, cfg, mtime):
        with open(self.path, "w") as f:
            json.dump(cfg, f)
        os.utime(self.path, ns=(mtime, mtime))

    def ack(self):
        with open(self.path + ".ack") as f:
            return json.load(f)

    def test_no_file_keeps_the_boot_policy_and_delegates(self):
        params = Params()
        self.verify(params, [2, 3], 2)
        self.assertEqual(params.calls, [([2, 3], 2)])
        self.assertFalse(os.path.exists(self.path + ".ack"))

    def test_reload_replaces_slots_and_reports_previous_steps(self):
        params = Params()
        self.verify(params, [1], 1)
        self.verify(params, [1], 4)
        self.write({"version": 7, "ema_alpha": 0.5, "1": {"candidate_steps": [3, 5, 7]},
                    "2": {"candidate_steps": [2, 4]}}, 10**9)
        self.now[0] = 1.0
        self.verify(params, [1], 1)
        self.assertEqual(params._bs_list, [1, 2])
        self.assertEqual(params._slots[1].candidate_steps, [3, 5, 7])
        self.assertEqual(params._slots[1].current_steps, 3)   # keeps the running step when allowed
        self.assertEqual(params._slots[1].ema_alpha, 0.5)     # shared keys apply to every slot
        # the batch that triggered the reload ran at step 3 under the old policy and counts there
        self.assertEqual(self.ack(), {"version": 7, "applied": True, "step_counts": {"3": 3}})

    def test_polls_at_most_every_interval_and_only_on_mtime_change(self):
        params = Params()
        self.write({"version": 1, "1": {"candidate_steps": [5]}}, 10**9)
        self.verify(params, [1], 1)
        self.assertEqual(params._slots[1].candidate_steps, [5])
        self.write({"version": 2, "1": {"candidate_steps": [7]}}, 2 * 10**9)
        self.now[0] = 0.2
        self.verify(params, [1], 1)
        self.assertEqual(params._slots[1].candidate_steps, [5])
        self.now[0] = 0.6
        self.verify(params, [1], 1)
        self.assertEqual(params._slots[1].candidate_steps, [7])

    def test_uncaptured_steps_are_rejected(self):
        params = Params()
        boot = params._slots
        self.write({"version": 3, "1": {"candidate_steps": [8]}}, 10**9)
        self.verify(params, [1], 1)
        self.assertIs(params._slots, boot)
        ack = self.ack()
        self.assertFalse(ack["applied"])
        self.assertIn("subset of the boot candidates", ack["error"])

    def test_malformed_json_is_rejected_without_raising(self):
        for mtime, cfg in enumerate([[], {"1": [1, 2]}, "x"], start=1):
            params = Params()
            boot = params._slots
            self.write(cfg, mtime * 10**9)
            self.now[0] = mtime
            self.verify(params, [1], 1)   # must not raise on the verify path
            self.assertIs(params._slots, boot)
            self.assertFalse(self.ack()["applied"])

    def test_rejected_ack_names_the_attempted_version(self):
        self.write({"version": 9, "1": {"candidate_steps": [8]}}, 10**9)
        self.verify(Params(), [1], 1)
        self.assertEqual(self.ack()["version"], 9)

    def test_steps_must_have_been_captured_for_each_graph_batch_size(self):
        params = Params()
        params._slots = {1: Slot(3, {"candidate_steps": [1, 2, 3, 4, 5, 6, 7]}),
                         4: Slot(3, {"candidate_steps": [3, 5, 7]})}
        params._cuda_graph_bs = [1, 2, 4, 8]
        boot = params._slots
        # batch sizes 4 and 8 had graphs only for 3, 5 and 7 at boot
        self.write({"version": 1, "1": {"candidate_steps": [1, 2, 3]}}, 10**9)
        self.verify(params, [1], 1)
        self.assertIs(params._slots, boot)
        self.assertIn("CUDA-graph batch size 4", self.ack()["error"])
        self.write({"version": 2, "1": {"candidate_steps": [1, 2, 3]}, "4": {"candidate_steps": [3, 5]}},
                   2 * 10**9)
        self.now[0] = 1.0
        self.verify(params, [1], 1)
        self.assertEqual(params._bs_list, [1, 4])
        self.assertTrue(self.ack()["applied"])

    def test_the_triggering_batch_feeds_the_old_policy(self):
        params = Params()
        boot = params._slots
        self.write({"version": 1, "1": {"candidate_steps": [5, 7]}}, 10**9)
        new_step = self.verify(params, [4], 1)
        self.assertIs(params.slots_seen[0], boot)       # delegated before the swap
        self.assertIsNot(params._slots, boot)
        self.assertEqual(new_step, 5)                   # the new slot's step runs next (3 not allowed)
        # the overlap scheduler launched the next batch before the reload: not fed, not counted
        self.now[0] = 0.1
        self.assertIsNone(self.verify(params, [0], 1))
        self.assertEqual(len(params.calls), 1)
        self.verify(params, [4], 1)
        self.assertEqual(len(params.calls), 2)
        self.assertIs(params.slots_seen[1], params._slots)

    def test_out_of_range_numbers_are_rejected(self):
        bad = [{"update_interval": 0}, {"update_interval": 2.5}, {"warmup_batches": -1},
               {"ema_alpha": 0}, {"ema_alpha": 1.5}, {"ema_alpha": "0.2"}, {"up_hysteresis": float("nan")},
               {"down_hysteresis": True}, {"1": {"candidate_steps": [True]}}, {"ema_alpha": None},
               {"ceiling_coeff": 1e308}, {"up_hysteresis": 10**401}, {"warmup_batches": None}]
        for mtime, extra in enumerate(bad, start=1):
            params = Params()
            boot = params._slots
            self.write({"version": mtime, "1": {"candidate_steps": [3]}, **extra}, mtime * 10**9)
            self.now[0] = mtime
            self.verify(params, [1], 1)
            self.assertIs(params._slots, boot, extra)
            self.assertFalse(self.ack()["applied"], extra)
        params = Params()
        self.write({"version": 99, "ema_alpha": 1, "update_interval": 1, "warmup_batches": 0,
                    "1": {"candidate_steps": [3], "up_hysteresis": -0.5}}, 99 * 10**9)
        self.now[0] = 99
        self.verify(params, [1], 1)
        self.assertTrue(self.ack()["applied"])

    def test_an_unreadable_file_keeps_the_policy(self):
        params = Params()
        boot = params._slots
        with mock.patch.object(ax_adaptive_live.os, "stat", side_effect=PermissionError("denied")):
            self.verify(params, [1], 1)   # must not raise on the verify path
        self.assertIs(params._slots, boot)

    def test_an_sglang_without_graph_batch_sizes_refuses_reloads(self):
        params = Params()
        del params._cuda_graph_bs
        boot = params._slots
        self.write({"version": 1, "1": {"candidate_steps": [3]}}, 10**9)
        self.verify(params, [1], 1)
        self.assertIs(params._slots, boot)
        self.assertIn("_cuda_graph_bs", self.ack()["error"])

    def test_more_than_one_rank_keeps_the_boot_policy(self):
        verify = ax_adaptive_live._wrap(Params.on_verify_complete, module(), self.state,
                                        clock=lambda: self.now[0], world_size=lambda: 2)
        params = Params()
        boot = params._slots
        self.write({"version": 1, "1": {"candidate_steps": [5]}}, 10**9)
        verify(params, [1], 1)
        self.assertIs(params._slots, boot)
        self.assertEqual(params.calls, [([1], 1)])
        self.assertFalse(os.path.exists(self.path + ".ack"))

    def test_install_needs_the_environment(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict(sys.modules, {"sglang.srt.speculative.adaptive_spec_params": None}):
            ax_adaptive_live.install()   # would fail to import if it tried


if __name__ == "__main__":
    unittest.main()
