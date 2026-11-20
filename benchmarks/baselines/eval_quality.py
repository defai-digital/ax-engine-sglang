"""Quality evals: HumanEval, MBPP (sanitized), GSM8K against a running vLLM server.

Saves per-sample outputs (used for AR-vs-MTP greedy parity) and pass@1 / accuracy.
WARNING: executes model-generated Python in subprocesses with a timeout. Run on the pod, not on
a machine you care about.

  python benchmarks/baselines/eval_quality.py --mode ar --out results/run1 --limit 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import AsyncOpenAI

from common import DEFAULT_BASE_URL, request_kwargs, write_json

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


OPEN_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*)\Z", re.S)
DONE_MARK = "__AX_CHECK_DONE__"


def extract_code(text: str) -> str:
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len)
    # an answer cut at the token limit can lack the closing fence: take the open block to the end
    m = OPEN_BLOCK.search(text)
    return m.group(1) if m else text


def run_python(program: str, timeout: float = 15.0) -> bool:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"
        # The marker is printed only after every check ran: generated code that calls exit() or
        # raises SystemExit(0) ends the process with status 0 before the tests finish.
        f.write_text(program + f"\nprint({DONE_MARK!r})\n")
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True,
                               timeout=timeout, env={"PATH": os.environ.get("PATH", "")})
            return r.returncode == 0 and r.stdout.decode(errors="replace").rstrip().endswith(DONE_MARK)
        except subprocess.TimeoutExpired:
            return False


# ---- tasks -------------------------------------------------------------------------------

def load_task(name: str, limit: int | None):
    from datasets import load_dataset

    if name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        items = [{
            "id": ex["task_id"],
            "prompt": "Complete the following Python function. Reply with the complete function "
                      "(including the signature and any imports) in a single ```python code block.\n\n"
                      + ex["prompt"],
            "check": lambda out, ex=ex: run_python(
                ex["prompt"] + "\n" + extract_code(out) + "\n\n" + ex["test"]
                + f"\n\ncheck({ex['entry_point']})\n"),
        } for ex in ds]
    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        items = [{
            "id": str(ex["task_id"]),
            "prompt": f"{ex['prompt']}\nYour code should pass this test:\n{ex['test_list'][0]}\n"
                      "Reply with the complete Python code in a single ```python code block.",
            "check": lambda out, ex=ex: run_python(
                "\n".join(ex["test_imports"]) + "\n" + extract_code(out) + "\n\n"
                + "\n".join(ex["test_list"]) + "\n"),
        } for ex in ds]
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")

        def check(out, ex):
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            # Only the requested final-answer line counts. Falling back to the last number in the
            # text would pass unfinished or derailed answers that merely mention the right value.
            nums = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", out or "")
            if not nums:
                return False
            try:
                return abs(float(nums[-1].replace(",", "")) - float(gold)) < 1e-6
            except ValueError:
                return False

        items = [{
            "id": str(i),
            "prompt": "Solve the problem step by step, then give the final answer on the last line "
                      "as '#### <number>'.\n\n" + ex["question"],
            "check": lambda out, ex=ex: check(out, ex),
        } for i, ex in enumerate(ds)]
    else:
        raise ValueError(name)
    return items[:limit] if limit else items


async def generate(client, items, kw, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        async with sem:
            r = await client.chat.completions.create(
                messages=[{"role": "user", "content": it["prompt"]}], **kw)
            return r.choices[0].message.content or "", r.usage.completion_tokens

    return await asyncio.gather(*(one(it) for it in items))


async def main_async(args):
    client = AsyncOpenAI(base_url=f"{args.base_url}/v1", api_key="EMPTY", timeout=7200)
    kw = request_kwargs(args.preset, args.thinking, args.max_tokens)
    scores = {}
    for task in args.tasks:
        items = load_task(task, args.limit)
        outs = await generate(client, items, kw, args.concurrency)
        with ThreadPoolExecutor(max_workers=16) as pool:
            passed = list(pool.map(lambda p: p[0]["check"](p[1][0]), zip(items, outs)))
        acc = sum(passed) / len(passed)
        scores[task] = {"n": len(items), "accuracy": acc,
                        "mean_completion_tokens": sum(o[1] for o in outs) / len(outs)}
        print(f"[{args.mode}] {task}: {acc:.3f} (n={len(items)})", flush=True)
        path = Path(args.out) / args.mode / f"samples_{task}_{args.preset}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for it, (text, ntok), ok in zip(items, outs, passed):
                f.write(json.dumps({"id": it["id"], "output": text, "tokens": ntok,
                                    "passed": ok}, ensure_ascii=False) + "\n")
    write_json(Path(args.out) / args.mode / f"quality_{args.preset}.json",
               {"mode": args.mode, "preset": args.preset, "thinking": args.thinking,
                "scores": scores})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--tasks", nargs="+", default=["humaneval", "mbpp", "gsm8k"],
                    choices=["humaneval", "mbpp", "gsm8k"])
    ap.add_argument("--limit", type=int, default=None, help="max items per task")
    ap.add_argument("--preset", default="greedy", choices=["greedy", "instruct", "thinking"])
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=64)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
