"""Pure round decisions. Backend objects and tensor operations stay in ax_chain."""
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
