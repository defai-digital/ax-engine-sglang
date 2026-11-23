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
        self.calls = []

    def _route(self, batch_size):
        return self._slots[max(b for b in self._bs_list if b <= batch_size) if batch_size >= self._bs_list[0]
                           else self._bs_list[0]]

    def on_verify_complete(self, drafts, batch_size):
        self.calls.append((drafts, batch_size))
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
