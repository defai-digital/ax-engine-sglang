"""AX_POLICY_MODE=adaptive: the chain-width controller (ax_engine_sglang/policy.py), offline."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang.config import PluginConfig
from ax_engine_sglang.policy import WidthController, batch_bucket

ADAPTIVE = {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8", "FC1_DRAFT_BLOCK": "16",
            "FC1_FUSED": "1"}


def accept_counts(accepted, width, requests=1):
    counts = [0] * 16
    counts[min(accepted, width - 1)] = requests
    return counts


def drive(controller, rounds, *, accepted, seconds, bucket=0):
    """Serve `rounds` chain rounds whose drafts would be accepted up to `accepted` tokens."""
    chosen = []
    for _ in range(rounds):
        width = controller.choose(bucket)
        controller.observe_cost(bucket, width, seconds[width])
        controller.observe_accepts(bucket, width, accept_counts(accepted, width))
        chosen.append(width)
    return chosen


class BucketTests(unittest.TestCase):
    def test_batch_sizes_map_to_buckets(self):
        self.assertEqual([batch_bucket(n) for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 64)],
                         [0, 1, 2, 2, 3, 3, 4, 4, 5, 5])


class WidthControllerTests(unittest.TestCase):
    def controller(self, **values):
        settings = dict(min_samples=8, cooldown=4, explore=8)
        settings.update(values)
        return WidthController((8, 16), 16, **settings)

    def test_stays_on_the_fallback_until_every_width_is_measured(self):
        controller = self.controller(min_samples=64)
        chosen = drive(controller, 40, accepted=3, seconds={8: 0.010, 16: 0.020})
        self.assertEqual({w for i, w in enumerate(chosen, 1) if i % 4}, {16})
        self.assertEqual(chosen[3::4], [8] * 10)   # every fourth round samples the other width
        self.assertEqual(controller.snapshot()["buckets"][0]["switches"], 0)

    def test_switches_to_the_narrow_width_when_drafts_are_rejected_early(self):
        controller = self.controller()
        chosen = drive(controller, 200, accepted=3, seconds={8: 0.010, 16: 0.014})
        self.assertEqual(controller.snapshot()["buckets"][0]["width"], 8)
        self.assertGreater(chosen[-64:].count(8), 50)
        self.assertEqual(controller.snapshot()["buckets"][0]["switches"], 1)

    def test_keeps_the_wide_width_when_long_drafts_are_accepted(self):
        controller = self.controller()
        drive(controller, 200, accepted=15, seconds={8: 0.010, 16: 0.014})
        self.assertEqual(controller.snapshot()["buckets"][0]["width"], 16)

    def test_returns_to_the_wide_width_when_acceptance_recovers(self):
        controller = self.controller()
        drive(controller, 200, accepted=3, seconds={8: 0.010, 16: 0.014})
        drive(controller, 2000, accepted=15, seconds={8: 0.010, 16: 0.014})
        self.assertEqual(controller.snapshot()["buckets"][0]["width"], 16)

    def test_a_gain_below_the_margin_does_not_switch(self):
        controller = self.controller(margin=0.08)
        # width 8: 4 tokens / 10.0 ms; width 16: 4 tokens / 10.5 ms (5% slower).
        drive(controller, 400, accepted=3, seconds={8: 0.0100, 16: 0.0105})
        self.assertEqual(controller.snapshot()["buckets"][0]["width"], 16)

    def test_cooldown_spaces_the_switches(self):
        def switches(cooldown):
            controller = self.controller(cooldown=cooldown, explore=4, window=8)
            for phase in range(8):
                drive(controller, 60, accepted=15 if phase % 2 else 3, seconds={8: 0.010, 16: 0.014})
            return controller.snapshot()["buckets"][0]["switches"]
        self.assertGreaterEqual(switches(0), 6)       # acceptance flips every 60 rounds
        self.assertLessEqual(switches(200), 480 // 200)

    def test_a_slowdown_of_every_width_does_not_switch(self):
        controller = self.controller(explore=32)
        drive(controller, 600, accepted=15, seconds={8: 0.010, 16: 0.014})
        # Both widths take twice as long (a busier GPU); the narrow one is timed once in 32 rounds.
        chosen = drive(controller, 600, accepted=15, seconds={8: 0.020, 16: 0.028})
        bucket = controller.snapshot()["buckets"][0]
        self.assertEqual((bucket["width"], bucket["switches"]), (16, 0))
        self.assertLessEqual(chosen.count(8), 600 // 32 + 1)   # samples only

    def test_round_time_ratios_follow_a_switch(self):
        seconds = {4: 0.008, 8: 0.010, 16: 0.016}
        controller = WidthController((4, 8, 16), 16, min_samples=8, cooldown=4, explore=4)
        drive(controller, 400, accepted=5, seconds=seconds)     # 6 tokens at width 8 and 16, 4 at width 4
        state = controller.buckets[0]
        self.assertEqual((state.width, state.switches), (8, 1))
        self.assertEqual(state.arms[8].ratio_samples, 0)        # the current width is timed directly
        for width in (4, 16):                                   # the others relative to it
            self.assertAlmostEqual(state.arms[width].ratio, seconds[width] / seconds[8], delta=0.02)
            self.assertAlmostEqual(controller._cost(state, width), seconds[width], delta=0.0003)
        # Re-expressed at the switch, before any new sample: the old current width against the new one.
        fresh = WidthController((4, 8, 16), 16, min_samples=8)
        bucket = fresh._bucket(0)
        for width, ratio in ((4, 0.5), (8, 0.625)):
            bucket.arms[width].ratio, bucket.arms[width].ratio_samples = ratio, 10
        fresh._switch(bucket, 8)
        self.assertEqual((bucket.width, bucket.arms[8].ratio_samples), (8, 0))
        self.assertAlmostEqual(bucket.arms[16].ratio, 1 / 0.625)
        self.assertAlmostEqual(bucket.arms[4].ratio, 0.5 / 0.625)

    def test_a_slowdown_after_a_switch_keeps_the_faster_width(self):
        controller = self.controller(cooldown=0, explore=16, window=32)
        drive(controller, 400, accepted=3, seconds={8: 0.010, 16: 0.014})
        state = controller.buckets[0]
        self.assertEqual((state.width, state.switches), (8, 1))
        self.assertGreater(state.arms[16].ratio_samples, 0)     # kept through the switch
        drive(controller, 400, accepted=3, seconds={8: 0.020, 16: 0.028})
        self.assertEqual((state.width, state.switches), (8, 1))

    def test_ratios_use_round_times_measured_under_the_same_load(self):
        # Width 16 ran at 20 ms for a long time; then both get faster and width 8 is the slower one.
        controller = self.controller(explore=8)
        drive(controller, 600, accepted=7, seconds={8: 0.024, 16: 0.020})
        chosen = drive(controller, 600, accepted=7, seconds={8: 0.012, 16: 0.010})
        state = controller.buckets[0]
        self.assertEqual((state.width, state.switches), (16, 0))
        self.assertAlmostEqual(state.arms[8].ratio, 1.2, delta=0.05)
        # The converse: width 8 becomes much the faster one and is taken.
        drive(controller, 600, accepted=7, seconds={8: 0.007, 16: 0.020})
        self.assertEqual(state.width, 8)

    def test_idle_gaps_do_not_turn_one_stall_into_a_round_time(self):
        controller = self.controller()
        for seconds in [0.010] * 10 + [0.9, 0.9, 0.05]:
            controller.observe_cost(0, 16, seconds)
        arm = controller.buckets[0].arms[16]
        self.assertAlmostEqual(arm.cost, 0.010)
        for seconds in (0.05, 0.9, 0.05, 0.9, 0.05):            # stalls separated by idle gaps
            controller.observe_cost(0, 16, seconds)
        self.assertAlmostEqual(arm.cost, 0.010)

    def test_implausibly_short_round_times_are_dropped(self):
        controller = self.controller()
        controller.observe_cost(0, 16, 1e-9)
        self.assertEqual(controller.buckets[0].arms[16].cost_samples, 0)

    def test_the_other_widths_are_sampled_in_turn(self):
        controller = WidthController((4, 8, 16), 16, min_samples=8, cooldown=4, explore=8)
        state = controller._bucket(0)
        state.arms[4].cost_samples = 1000                       # sampled a lot, long ago
        chosen = drive(controller, 320, accepted=15, seconds={4: 0.009, 8: 0.010, 16: 0.012})
        explored = [w for w in chosen if w != 16]
        self.assertLessEqual(abs(explored.count(4) - explored.count(8)), 1)

    def test_a_width_never_timed_against_the_current_one_is_not_taken(self):
        controller = self.controller()
        state = controller._bucket(0)
        for width, cost in ((16, 0.014), (8, 0.001)):           # width 8: a stale, very low round time
            arm = state.arms[width]
            arm.cost, arm.cost_samples, arm.tokens, arm.token_samples = cost, 100, 8.0, 100
        state.since_switch = 100
        self.assertEqual(controller.choose(0), 16)
        # Not decided yet: the bucket keeps sampling width 8 every fourth round until it is comparable.
        self.assertEqual(controller.snapshot()["reasons"], {"warmup": 1})
        self.assertEqual([controller.choose(0) for _ in range(3)], [16, 16, 8])

    def test_warmup_sampling_backs_off_when_a_width_cannot_be_compared(self):
        controller = self.controller(min_samples=2, explore=16)     # budget: 4 * 2 * 1 = 8 fast samples
        choices = [controller.choose(0) for _ in range(96)]         # no round time is ever reported
        self.assertEqual(choices[:32].count(8), 8)                  # every fourth round
        self.assertEqual(choices[32:].count(8), 4)                  # then every sixteenth
        self.assertEqual(controller.buckets[0].switches, 0)

    def test_a_few_comparisons_do_not_allow_a_switch(self):
        controller = self.controller(min_samples=8)
        state = controller._bucket(0)
        for width, cost in ((16, 0.014), (8, 0.014)):
            arm = state.arms[width]
            arm.cost, arm.cost_samples, arm.tokens, arm.token_samples = cost, 100, 4.0, 100
        state.recent, state.since_switch = 0.014, 100
        state.arms[8].ratio, state.arms[8].ratio_samples = 0.5, 7     # looks twice as fast, on 7 comparisons
        self.assertEqual(controller.choose(0), 16)
        self.assertEqual(controller.buckets[0].switches, 0)
        state.arms[8].ratio_samples = 8
        self.assertEqual(controller.choose(0), 8)

    def test_one_large_readback_does_not_replace_the_average(self):
        controller = self.controller()
        drive(controller, 400, accepted=15, seconds={8: 0.010, 16: 0.014})
        before = controller.buckets[0].arms[16].tokens
        controller.observe_accepts(0, 16, [0, 0, 64] + [0] * 13)   # 16 rounds of 4 requests, 2 accepted
        after = controller.buckets[0].arms[16].tokens
        self.assertAlmostEqual(before, 16.0)
        self.assertGreater(after, 7.0)    # weight 1 - (1 - 2/129)^64, about 0.63
        self.assertLess(after, 9.0)
        self.assertEqual(controller.choose(0), 16)

    def test_a_lasting_longer_round_time_is_learned(self):
        controller = self.controller()
        for _ in range(8):
            controller.observe_cost(5, 16, 0.020)     # 17 requests
        for _ in range(200):
            controller.observe_cost(5, 16, 0.100)     # 64 requests, the same bucket
        arm = controller.buckets[5].arms[16]
        self.assertEqual(arm.cost_samples, 8 + 198)   # the first two long rounds count as stalls
        self.assertGreater(arm.cost, 0.09)

    def test_buckets_decide_separately(self):
        controller = self.controller()
        drive(controller, 200, accepted=3, seconds={8: 0.010, 16: 0.014}, bucket=0)
        drive(controller, 200, accepted=15, seconds={8: 0.020, 16: 0.022}, bucket=3)
        buckets = controller.snapshot()["buckets"]
        self.assertEqual((buckets[0]["width"], buckets[3]["width"]), (8, 16))

    def test_wide_rounds_measure_the_narrow_width(self):
        controller = self.controller()
        controller.observe_accepts(0, 16, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0])
        arms = controller.buckets[0].arms
        self.assertAlmostEqual(arms[16].tokens, (3 + 3 * 13) / 4)
        self.assertAlmostEqual(arms[8].tokens, (3 + 3 * 8) / 4)   # capped at the anchor's 7 drafts + bonus
        self.assertEqual((arms[16].token_samples, arms[8].token_samples), (4, 4))

    def test_narrow_rounds_do_not_measure_the_wide_width(self):
        controller = self.controller()
        controller.observe_accepts(0, 8, accept_counts(7, 8))
        self.assertEqual(controller.buckets[0].arms[16].token_samples, 0)

    def test_a_width_with_its_own_draft_graph_uses_its_own_rounds_only(self):
        controller = self.controller(direct_only=(8,))
        controller.observe_accepts(0, 16, accept_counts(12, 16))
        self.assertEqual(controller.buckets[0].arms[8].token_samples, 0)
        controller.observe_accepts(0, 8, accept_counts(5, 8))
        self.assertAlmostEqual(controller.buckets[0].arms[8].tokens, 6)

    def test_idle_gaps_are_not_round_times(self):
        controller = self.controller()
        for _ in range(10):
            controller.observe_cost(0, 16, 0.010)
        for seconds in (0.9, 0.2, 0.0, -1.0):   # idle server, 20 x the mean, clock noise
            controller.observe_cost(0, 16, seconds)
        arm = controller.buckets[0].arms[16]
        self.assertEqual(arm.cost_samples, 10)
        self.assertAlmostEqual(arm.cost, 0.010)
        self.assertEqual(controller.snapshot()["reasons"]["cost_dropped"], 4)

    def test_without_round_times_the_fallback_stays(self):
        controller = self.controller()
        for _ in range(500):
            width = controller.choose(0)
            controller.observe_accepts(0, width, accept_counts(1, width))
        self.assertEqual(controller.snapshot()["buckets"][0]["width"], 16)

    def test_the_fallback_must_be_a_candidate(self):
        with self.assertRaisesRegex(ValueError, "fallback"):
            WidthController((8, 16), 12)
        with self.assertRaisesRegex(ValueError, "two entries"):
            WidthController((8,), 8)


class AdaptiveConfigTests(unittest.TestCase):
    def test_adaptive_reads_its_settings(self):
        config = PluginConfig.from_env({**ADAPTIVE, "AX_ADAPTIVE_MARGIN": "0.1", "AX_ADAPTIVE_COOLDOWN": "20"})
        self.assertEqual((config.policy_mode, config.adaptive_margin, config.adaptive_cooldown,
                          config.adaptive_min_samples, config.adaptive_explore), ("adaptive", 0.1, 20, 32, 32))

    def test_adaptive_starts_before_the_draft_block_is_resolved(self):
        # The launcher validates once before SGLang resolves --speculative-num-draft-tokens.
        from ax_engine_sglang.config import startup_config
        environ = {"AX_POLICY_MODE": "adaptive", "AX_CHAIN_GRAPH_WIDTHS": "16 8"}
        config, _ = startup_config(environ, hooks_available=True)
        self.assertEqual((config.graph_widths, config.draft_block), ((16, 8), 24))
        with self.assertRaisesRegex(ValueError, "AX_CHAIN_GRAPH_WIDTHS runner"):
            PluginConfig.from_env({**ADAPTIVE, "AX_CHAIN_WIDTH": "12"})

    def test_adaptive_works_with_the_ssm_cap_without_a_pinned_width(self):
        config = PluginConfig.from_env({**ADAPTIVE, "AX_SSM_POSITIONS": "16"})
        self.assertEqual((config.ssm_positions, config.chain_width), (16, None))
        with self.assertRaisesRegex(ValueError, "exceed AX_SSM_POSITIONS"):
            PluginConfig.from_env({**ADAPTIVE, "AX_SSM_POSITIONS": "8"})
        with self.assertRaisesRegex(ValueError, "AX_CHAIN_WIDTH"):
            PluginConfig.from_env({**ADAPTIVE, "AX_POLICY_MODE": "static", "AX_SSM_POSITIONS": "16"})

    def test_static_is_the_default(self):
        self.assertEqual(PluginConfig.from_env({}).policy_mode, "static")

    def test_a_width_table_maps_batch_sizes_to_widths(self):
        from ax_engine_sglang.policy import chain_width
        base = {"AX_CHAIN_GRAPH_WIDTHS": "16 8", "FC1_DRAFT_BLOCK": "16"}
        config = PluginConfig.from_env({**base, "AX_CHAIN_WIDTH_BY_BATCH": "1:16, 5:8"})
        self.assertEqual(config.chain_width_table, ((1, 16), (5, 8)))
        self.assertTrue(config.tree_requested)
        self.assertEqual([chain_width(config, 16, 7, n) for n in (1, 4, 5, 9)], [16, 16, 8, 8])
        self.assertEqual(chain_width(config, 16, 0), 16)            # a caller that gives no batch size
        self.assertEqual(chain_width(config, 16, 0, 0), 16)         # an empty batch uses the first entry
        # The table alone asks for the tree worker, before the other settings are validated.
        self.assertTrue(PluginConfig.from_env({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8"}, validate=False).tree_requested)
        with self.assertRaisesRegex(ValueError, "outside 2..8"):
            chain_width(config, 8, 0, 2)                            # the server's draft block is smaller
        ssm = PluginConfig.from_env({**base, "AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8", "AX_SSM_POSITIONS": "16",
                                     "FC1_FUSED": "1"})
        self.assertEqual(ssm.ssm_positions, 16)
        cases = (
            ({"AX_CHAIN_WIDTH_BY_BATCH": "16,8"}, "requests:width"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": " , "}, "requests:width"),     # not silently no table
            ({"AX_CHAIN_WIDTH_BY_BATCH": "2:16,5:8"}, "starting at 1"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8,5:4"}, "ascending"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:12"}, "AX_CHAIN_GRAPH_WIDTHS runner"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:1"}, "outside 2..16"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8", "AX_CHAIN_WIDTH": "8"}, "unset AX_CHAIN_WIDTH"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8", "AX_CHAIN_SCHEDULE": "16x2,8x2"}, "unset AX_CHAIN_WIDTH"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8", "AX_POLICY_MODE": "adaptive"}, "unset AX_CHAIN_WIDTH"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8", "AX_CHAIN_GRAPH_WIDTHS": ""}, "in AX_CHAIN_GRAPH_WIDTHS"),
            ({"AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8", "AX_NVFP4_VERIFY_PARITY": "1", "AX_GDN_UNPACKED_DECODE": "1",
              "FC1_FUSED": "0", "AX_GDN_RECURRENT_VERIFY": "1"}, "fixed chain width"),
        )
        for extra, message in cases:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                PluginConfig.from_env({**base, **extra})
        self.assertEqual(PluginConfig.from_env({**base, "AX_CHAIN_WIDTH_BY_BATCH": ""}).chain_width_table, ())

    def test_a_width_table_is_checked_against_the_servers_draft_block(self):
        from types import SimpleNamespace
        from ax_engine_sglang.config import server_args_error
        table = PluginConfig.from_env({"AX_CHAIN_GRAPH_WIDTHS": "16 8", "AX_CHAIN_WIDTH_BY_BATCH": "1:16,5:8"})
        self.assertIn("chain width 16 exceeds the draft block 8",
                      server_args_error(table, SimpleNamespace(speculative_num_draft_tokens=8)))
        # Every rank sees the same batch, so several GPUs choose the same width.
        self.assertIsNone(server_args_error(table, SimpleNamespace(speculative_num_draft_tokens=16, tp_size=2)))
        self.assertIn("supports DP1", server_args_error(table, SimpleNamespace(speculative_num_draft_tokens=16,
                                                                               dp_size=2)))

    def test_adaptive_refuses_settings_it_cannot_combine_with(self):
        cases = (
            ({"AX_CHAIN_GRAPH_WIDTHS": "16"}, "at least two widths"),
            ({"AX_CHAIN_SCHEDULE": "16x4,8x4"}, "AX_CHAIN_SCHEDULE"),
            ({"AX_NVFP4_VERIFY_PARITY": "1", "AX_GDN_UNPACKED_DECODE": "1", "FC1_FUSED": "0",
              "AX_GDN_RECURRENT_VERIFY": "1"}, "fixed chain width"),
            ({"AX_ADAPTIVE_EXPLORE": "1"}, "AX_ADAPTIVE_EXPLORE"),
            ({"AX_ADAPTIVE_MARGIN": "nan"}, "AX_ADAPTIVE_MARGIN"),
            ({"AX_ADAPTIVE_MARGIN": "wide"}, "must be a number"),
            ({"AX_POLICY_MODE": "auto"}, "static or adaptive"),
            ({"AX_GDN_CHAIN_REPLAY": "1"}, "stock width-eight DFLASH worker"),
        )
        for extra, message in cases:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                PluginConfig.from_env({**ADAPTIVE, **extra})


if __name__ == "__main__":
    unittest.main()
