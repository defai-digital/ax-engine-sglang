"""Two-B200 Flash Next BF16 serving A/B: one SGLang server per lane, bench_speed.py at C1/C4/C16.

Runs on the pod layout of results/b200x2_flash_next_prefill_graph: the checkpoint at /root/fn/checkpoint,
the BF16 MTP draft from scripts/prepare_flash_mtp.py at /dev/shm/fn-mtp-bf16, this repository at
/root/ax/source with SGLang in /root/ax/venv, prompts at /root/fn/evidence/prompts.json.

    python flash_next_tp2_ab.py base-mtp3 pcg-mtp3 pcg-mtp7      (lane names from LANES; <lane>-r2 repeats)

`base-` lanes are the cross-engine launch (Triton attention, no FlashInfer autotune, eager prefill);
`pcg-` lanes keep SGLang's prefill CUDA graphs.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

base = Path('/root/fn')
repo = Path('/root/ax/source')
model = base / 'checkpoint'
python = Path('/root/ax/venv/bin/python')
port = 30563
url = f'http://127.0.0.1:{port}'
out = base / 'evidence/tune'
out.mkdir(parents=True, exist_ok=True)
CONC = os.environ.get('CONC', '1 4 16').split()
PASSES = int(os.environ.get('PASSES', '2'))

TRITON = ['--attention-backend', 'triton']
NO_AUTOTUNE = ['--disable-flashinfer-autotune']


def spec(steps):
    return ['--speculative-algorithm', 'EAGLE', '--speculative-num-steps', str(steps), '--speculative-eagle-topk', '1',
            '--speculative-num-draft-tokens', str(steps + 1), '--speculative-draft-model-path', '/dev/shm/fn-mtp-bf16']


# name: (extra args, env)
LANES = {
    'base-mtp3': (TRITON + NO_AUTOTUNE + spec(3), {}),
    'autotune-mtp3': (TRITON + spec(3), {}),
    'default-attn-mtp3': (NO_AUTOTUNE + spec(3), {}),
    'both-mtp3': (spec(3), {}),
    'base-mtp7': (TRITON + NO_AUTOTUNE + spec(7), {'SGLANG_PLUGINS': 'ax-engine', 'AX_QSA_RING': '16'}),
    'both-mtp7': (spec(7), {'SGLANG_PLUGINS': 'ax-engine', 'AX_QSA_RING': '16'}),
    'both-mtp5': (spec(5), {'SGLANG_PLUGINS': 'ax-engine', 'AX_QSA_RING': '16'}),
    'both-ar': ([], {}),
    'pcg-base-mtp3': (TRITON + NO_AUTOTUNE + spec(3), {}),
    'pcg-mtp3': (spec(3), {}),
    'pcg-mtp7': (spec(7), {'SGLANG_PLUGINS': 'ax-engine', 'AX_QSA_RING': '16'}),
    'pcg-ar': ([], {}),
    'pcg-mtp5': (spec(5), {'SGLANG_PLUGINS': 'ax-engine', 'AX_QSA_RING': '16'}),
    'prof-base-mtp3': (TRITON + NO_AUTOTUNE + spec(3), {}),
    'prof-both-mtp3': (spec(3), {}),
}


for _name in list(LANES):   # repeat lanes: <lane>-r2 runs <lane> again into its own directory
    LANES[_name + '-r2'] = LANES[_name]


def say(message):
    print(time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime()), message, flush=True)


def api(path, data=None):
    request = urllib.request.Request(url + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30 if data is None else 600) as response:
        return json.load(response) if 'json' in response.headers.get('Content-Type', '') else response.read()


def command_of(name):
    extra, _ = LANES[name]
    command = [str(python), '-u', '-m', 'sglang.launch_server', '--model-path', str(model), '--served-model-name', 'qwen',
            '--host', '127.0.0.1', '--port', str(port), '--tp-size', '2', '--dtype', 'bfloat16',
            '--linear-attn-prefill-backend', 'flashinfer', '--linear-attn-decode-backend', 'flashinfer',
            '--linear-attn-verify-backend', 'triton', '--mamba-ssm-dtype', 'bfloat16', '--kv-cache-dtype', 'auto',
            '--context-length', '4096', '--max-total-tokens', '16384', '--chunked-prefill-size', '8192',
            '--mem-fraction-static', '0.85', '--max-running-requests', '16', '--cuda-graph-max-bs-decode', '16',
            '--ple-offload-embedding', '--disable-radix-cache', '--disable-prefill-cuda-graph', '--enable-metrics',
            '--decode-log-interval', '1', '--random-seed', '161803', '--qsa-indexer-dtype', 'bfloat16', *extra]
    if name.startswith('pcg-'):   # prefill CUDA graphs on
        command.remove('--disable-prefill-cuda-graph')
    return command


def run_lane(name):
    lane = out / name
    lane.mkdir(exist_ok=True)
    _, extra_env = LANES[name]
    env = dict(os.environ, CUDA_HOME='/usr/local/cuda', HF_HUB_OFFLINE='1', SGLANG_PLUGINS='none', AX_QSA_RING='0',
               PATH=f'{python.parent}:/usr/local/cuda/bin:{os.environ["PATH"]}')
    env.update(extra_env)
    if name.startswith('prof-'):
        env['SGLANG_TORCH_PROFILER_DIR'] = str(lane / 'torch_profile')
    command = command_of(name)
    (lane / 'launch.json').write_text(json.dumps({'command': command, 'env': extra_env}, indent=1) + '\n')
    try:
        api('/server_info')
        raise RuntimeError('port busy')
    except OSError:
        pass
    say(f'{name} launching')
    t = time.monotonic()
    with (lane / 'server.log').open('w') as log:
        server = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
        try:
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'server exited {server.returncode}')
                if time.monotonic() - t > 2400:
                    raise RuntimeError('not ready after 2400 s')
                try:
                    api('/server_info')
                    break
                except (OSError, ValueError):
                    time.sleep(3)
            say(f'{name} ready after {time.monotonic() - t:.0f} s')
            if name.startswith('prof-'):
                profile(name, lane)
                return
            for p in range(1, PASSES + 1):
                cmd = [str(python), '-u', str(repo / 'benchmarks/baselines/bench_speed.py'), '--base-url', url,
                       '--mode', name, '--preset', 'greedy', '--max-tokens', '256', '--concurrency', *CONC,
                       '--n-requests', '24', '--prompt-file', str(base / 'evidence/prompts.json'),
                       '--out', str(lane / f'pass{p}')]
                with (lane / f'pass{p}.log').open('w') as stream:
                    code = subprocess.run(cmd, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                                          env={**os.environ, 'SERVED_MODEL': 'qwen'}).returncode
                say(f'{name} pass{p} exit {code}')
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server.wait()
            time.sleep(15)
            say(f'{name} stopped')


def profile(name, lane):
    """Warm up, then profile four concurrent code requests (C4)."""
    prompts = json.loads((base / 'evidence/prompts.json').read_text())
    code = prompts['code']
    from concurrent.futures import ThreadPoolExecutor

    def ask(p, tokens=256):
        text = p if isinstance(p, str) else (p.get('prompt') or p['messages'][-1]['content'])
        return api('/generate', {'text': text, 'sampling_params': {'max_new_tokens': tokens, 'temperature': 0}})
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(ask, code[:8]))
        api('/start_profile', {'activities': ['CPU', 'GPU']})
        results = list(pool.map(ask, code[8:12]))
        api('/stop_profile', {})
    (lane / 'profile.json').write_text(json.dumps([r.get('meta_info', {}) for r in results], indent=1) + '\n')
    say(f'{name} profile done')


for name in sys.argv[1:]:
    try:
        run_lane(name)
    except Exception as error:
        say(f'{name} FAILED {type(error).__name__}: {error}')
say('TUNE_DONE')
