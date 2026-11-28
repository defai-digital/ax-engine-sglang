#!/usr/bin/env python3
"""Record target-only greedy continuations and pre-anchor features for AX Draft."""

import argparse
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np


def generate(base_url, prompt, length):
    body = {
        "input_ids": prompt,
        "return_hidden_states": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": length, "ignore_eos": True},
    }
    request = urllib.request.Request(
        f"{base_url}/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=240 + length) as response:
        return json.load(response)


def recording(result, prompt, length, hidden_size, vocab_size):
    tokens = result["output_ids"]
    meta = result["meta_info"]
    if len(tokens) != length or meta["finish_reason"]["type"] != "length":
        raise ValueError("unexpected teacher output length")
    if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
        raise ValueError("invalid target token ID")
    states = meta["hidden_states"]
    if len(states) != length:
        raise ValueError(f"expected {length} teacher features, got {len(states)}")
    prefill = np.asarray(states[0], dtype="<f4")
    decode = np.asarray(states[1:], dtype="<f4")
    if prefill.shape != (len(prompt), hidden_size) or decode.shape != (length - 1, hidden_size):
        raise ValueError(f"expected unchunked prefill and serial decode features, got "
                         f"{prefill.shape} and {decode.shape}")
    features = np.concatenate((prefill[-1:], decode))
    if not np.isfinite(features).all():
        raise ValueError("non-finite teacher feature")
    return tokens, features


def capture(base_url, prompts, output, length, hidden_size, vocab_size):
    if output.exists():
        raise FileExistsError(output)
    if not prompts or len({row["id"] for row in prompts}) != len(prompts):
        raise ValueError("expected nonempty prompts with unique IDs")
    with tempfile.TemporaryDirectory(prefix=".ax-teacher-", dir=output.parent) as directory:
        temporary = Path(directory)
        row_offset = 0
        with (temporary / "states.f32").open("wb") as binary, (temporary / "sequences.jsonl").open("w") as index:
            for row in prompts:
                prompt = row["input_ids"]
                if not prompt or any(type(token) is not int or not 0 <= token < vocab_size for token in prompt):
                    raise ValueError(f"{row['id']}: invalid prompt token IDs")
                result = generate(base_url, prompt, length)
                tokens, features = recording(result, prompt, length, hidden_size, vocab_size)
                features.tofile(binary)
                index.write(json.dumps({"id": row["id"], "row": row_offset, "tokens": tokens}) + "\n")
                row_offset += length
        (temporary / "metadata.json").write_text(json.dumps({
            "version": 1, "hidden_size": hidden_size, "vocab_size": vocab_size, "rows": row_offset,
        }) + "\n")
        temporary.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30517")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()
    if args.length < 2 or args.hidden_size < 1 or args.vocab_size < 1:
        parser.error("expected valid model dimensions and at least two output tokens")
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    capture(args.url.rstrip("/"), prompts, args.output, args.length, args.hidden_size, args.vocab_size)


if __name__ == "__main__":
    main()
