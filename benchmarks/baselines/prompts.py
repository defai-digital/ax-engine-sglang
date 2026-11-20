"""Prompt sets for the speed benchmark: code, math and chat."""
from __future__ import annotations

CHAT_PROMPTS = [
    "Explain how a hash map handles collisions, with a short example.",
    "Write a polite email declining a meeting invitation and proposing two alternative times.",
    "Summarize the causes of the 2008 financial crisis in five bullet points.",
    "What are the trade-offs between microservices and a monolith for a 10-person startup?",
    "Describe how photosynthesis works to a 12-year-old.",
    "Give me a 3-day itinerary for Kyoto focused on temples and food.",
    "Compare TCP and UDP and say when you would choose each.",
    "Write a short story (about 200 words) about a lighthouse keeper who finds a robot.",
    "How does public-key cryptography let two strangers communicate securely?",
    "List practical tips to improve sleep quality, with the reasoning behind each.",
    "Explain the difference between supervised, unsupervised and reinforcement learning.",
    "Draft a product description for a waterproof hiking backpack.",
    "What is the CAP theorem? Give a concrete example for each trade-off.",
    "Explain what a mixture-of-experts language model is and why it is efficient.",
    "Write a haiku sequence (three haiku) about autumn in a city.",
    "How should a beginner structure a weekly strength-training plan?",
]


def load_prompt_sets(n_per_set: int = 32) -> dict[str, list[str]]:
    from datasets import load_dataset

    he = load_dataset("openai/openai_humaneval", split="test")
    code = [
        "Complete the following Python function. Reply with the full function in a ```python code block.\n\n"
        + ex["prompt"]
        for ex in he.select(range(min(n_per_set, len(he))))
    ]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    math = [
        "Solve the problem step by step, then give the final answer on the last line as '#### <number>'.\n\n"
        + ex["question"]
        for ex in gsm.select(range(min(n_per_set, len(gsm))))
    ]
    chat = list(CHAT_PROMPTS[:n_per_set])
    if len(chat) < n_per_set:
        # Fill with distinct first-turn prompts so no prompt repeats inside a benchmark level.
        uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        seen = set(chat)
        for ex in uc:
            p = ex["prompt"].strip()
            if 20 <= len(p) <= 600 and p not in seen:
                chat.append(p)
                seen.add(p)
                if len(chat) == n_per_set:
                    break
    return {"code": code, "math": math, "chat": chat}
