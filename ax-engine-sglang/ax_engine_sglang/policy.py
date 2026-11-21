"""Round decisions without backend objects. Tensor operations stay in ax_chain."""
from dataclasses import dataclass

from .config import PluginConfig


@dataclass(frozen=True)
class RoundFeatures:
    requests: int
    grammar: bool = False
    logprobs: bool = False
    custom_processor: bool = False
    greedy: bool = True
    min_p: bool = False


def use_tree(config: PluginConfig, features: RoundFeatures) -> bool:
    if (not config.tree or features.requests != 1 or features.grammar
            or features.logprobs or features.custom_processor):
        return False
    return features.greedy or (config.tree_sampling and not features.min_p)


def chain_width(config: PluginConfig, block: int, round_index: int) -> int:
    if config.chain_schedule:
        width = config.chain_schedule[round_index % len(config.chain_schedule)]
    else:
        width = config.chain_width if config.chain_width is not None else block
    if not 2 <= width <= block:
        raise ValueError(f"ax-engine: chain width {width} outside 2..{block} (drafter block)")
    return width


# Largest batch size of each bucket; larger batches share the last one.
BUCKET_LIMITS = (1, 2, 4, 8, 16)


def batch_bucket(requests: int) -> int:
    for index, limit in enumerate(BUCKET_LIMITS):
        if requests <= limit:
            return index
    return len(BUCKET_LIMITS)


class _Arm:
    __slots__ = ("cost", "cost_samples", "tokens", "token_samples", "ratio", "ratio_samples", "outliers",
                 "sampled")

    def __init__(self):
        self.cost = self.tokens = self.ratio = 0.0
        self.cost_samples = self.token_samples = self.ratio_samples = self.outliers = self.sampled = 0


class _Bucket:
    def __init__(self, widths, width):
        self.width = width
        self.rounds = self.since_switch = self.switches = 0
        self.recent = 0.0   # the current width's round time over its last few rounds
        self.arms = {w: _Arm() for w in widths}


class WidthController:
    """AX_POLICY_MODE=adaptive: choose a chain width from measured tokens per round and round time.

    One instance per worker. It holds numbers only; ax_chain feeds it and applies its choice.

    - A round at width W whose request accepted `a` draft tokens commits a + 1 tokens. Prefix
      acceptance does not depend on later positions, so the same round at a narrower width V would
      have committed min(a, V - 1) + 1: wide rounds also measure the narrower widths. Widths in
      `direct_only` draft with their own graph (other draft tokens) and use their own rounds only.
    - The bucket's current width is timed every round, the others only when sampled. Their round
      time is kept as a ratio to the current width's over its last few rounds, so a slowdown
      common to all widths (more requests, a busier GPU) does not favour the rarely timed ones.
    - Until every width has `min_samples` round times and accept samples, the bucket stays on
      `fallback` and samples the other widths in turn every fourth round.
    - Afterwards it switches when another width scores more than `margin` above the current one,
      at most once per `cooldown` rounds, and samples the other widths in turn every `explore` rounds.
    """

    def __init__(self, widths, fallback, *, min_samples=32, margin=0.08, cooldown=50, explore=32,
                 window=128, max_gap=0.5, min_gap=2e-4, direct_only=()):
        self.widths = tuple(sorted(set(widths)))
        if len(self.widths) < 2 or fallback not in self.widths:
            raise ValueError(f"adaptive chain widths {self.widths} need two entries including the fallback {fallback}")
        self.fallback = fallback
        self.min_samples, self.margin, self.cooldown, self.explore = min_samples, margin, cooldown, explore
        self.alpha = 2.0 / (window + 1)
        self.max_gap, self.min_gap = max_gap, min_gap
        self.direct_only = frozenset(direct_only)
        self.buckets: dict[int, _Bucket] = {}
        self.reasons: dict[str, int] = {}

    def _bucket(self, bucket):
        found = self.buckets.get(bucket)
        if found is None:
            found = self.buckets[bucket] = _Bucket(self.widths, self.fallback)
        return found

    def _ready(self, arm):
        return arm.cost_samples >= self.min_samples and arm.token_samples >= self.min_samples

    def _cost(self, state, width):
        arm, current = state.arms[width], state.arms[state.width]
        if width != state.width and arm.ratio_samples and current.cost > 0:
            return arm.ratio * current.cost
        return arm.cost

    def _comparable(self, state, width):
        """A width can replace the current one once it was timed against it a few times."""
        return width == state.width or state.arms[width].ratio_samples >= min(self.min_samples, 4)

    def _score(self, state, width):
        cost = self._cost(state, width)
        return state.arms[width].tokens / cost if cost > 0 else 0.0

    def choose(self, bucket: int) -> int:
        state = self._bucket(bucket)
        state.rounds += 1
        state.since_switch += 1
        ready = all(self._ready(arm) for arm in state.arms.values())
        if state.rounds % (self.explore if ready else min(4, self.explore)) == 0:
            reason = "explore"
            width = min((w for w in self.widths if w != state.width),
                        key=lambda w: (state.arms[w].sampled, w))
            state.arms[width].sampled = state.rounds
        elif not ready:
            reason, width = "warmup", state.width
        elif state.since_switch <= self.cooldown:
            reason, width = "cooldown", state.width
        else:
            best = max((w for w in self.widths if self._comparable(state, w)),
                       key=lambda w: (self._score(state, w), w == state.width))
            if best != state.width and self._score(state, best) > self._score(state, state.width) * (1 + self.margin):
                reason = "switch"
                self._switch(state, best)
            else:
                reason = "hold"
            width = state.width
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return width

    def _switch(self, state, width):
        # Re-express the round-time ratios against the new current width.
        samples = state.arms[width].ratio_samples
        base = state.arms[width].ratio if samples else 0.0
        for other, arm in state.arms.items():
            if other == width or not base:
                arm.ratio, arm.ratio_samples = 0.0, 0
            elif other == state.width:
                arm.ratio, arm.ratio_samples = 1.0 / base, samples
            elif arm.ratio_samples:
                arm.ratio /= base
        state.recent *= base
        state.width, state.since_switch = width, 0
        state.switches += 1

    def _blend(self, mean, samples, value, count):
        """Moving average after `count` more samples of `value`; a plain mean while it has few."""
        weight = max(count / (samples + count), 1.0 - (1.0 - self.alpha) ** count)
        return mean + (value - mean) * weight

    def observe_cost(self, bucket: int, width: int, seconds: float):
        """Host time from the start of a chain round to the start of the next round."""
        state = self._bucket(bucket)
        arm = state.arms[width]
        # A gap with no running request is not a round time. One long round is dropped as a stall;
        # three in a row are the new round time (a bucket spans several batch sizes).
        idle = not self.min_gap <= seconds <= self.max_gap
        stall = not idle and arm.cost_samples >= 8 and seconds > 4 * arm.cost
        arm.outliers = arm.outliers + 1 if stall else 0
        if idle or (stall and arm.outliers < 3):
            self.reasons["cost_dropped"] = self.reasons.get("cost_dropped", 0) + 1
            return
        arm.cost = self._blend(arm.cost, arm.cost_samples, seconds, 1)
        arm.cost_samples += 1
        if width == state.width:
            state.recent = seconds if not state.recent else state.recent + (seconds - state.recent) * 0.5
        elif state.recent:
            # Against the current width's last few rounds, not its long average: both were
            # measured under the same load.
            arm.ratio = self._blend(arm.ratio, arm.ratio_samples, seconds / state.recent, 1)
            arm.ratio_samples += 1

    def observe_accepts(self, bucket: int, width: int, counts):
        """counts[a]: requests of rounds at `width` that accepted `a` draft tokens."""
        total = sum(counts)
        if not total:
            return
        arms = self._bucket(bucket).arms
        for other, arm in arms.items():
            if other != width and (other > width or other in self.direct_only or width in self.direct_only):
                continue
            tokens = sum(n * (min(a, other - 1) + 1) for a, n in enumerate(counts)) / total
            arm.tokens = self._blend(arm.tokens, arm.token_samples, tokens, total)
            arm.token_samples += total

    def snapshot(self) -> dict:
        return {
            "reasons": dict(self.reasons),
            "buckets": {bucket: {"width": state.width, "switches": state.switches, "rounds": state.rounds,
                                 "tokens_per_second": {w: round(self._score(state, w), 1) for w in state.arms}}
                        for bucket, state in sorted(self.buckets.items())},
        }
