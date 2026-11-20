"""Launch the SGLang language-model server with the AX plugin enabled."""

import argparse
import os
import sys

from . import __version__


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Run ax-engine-sglang serve MODEL [SGLang server flags]."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("command", nargs="?", choices=["serve"], help="start serving a model")
    parser.add_argument("server_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    return serve(args.server_args)


def serve(argv):
    if argv and not argv[0].startswith("-"):
        argv = ["--model-path", argv[0], *argv[1:]]

    # SGLang loads all discovered plugins when the allowlist is empty.
    selection = os.environ.get("SGLANG_PLUGINS")
    if selection and "ax-engine" not in {name.strip() for name in selection.split(",")}:
        os.environ["SGLANG_PLUGINS"] = f"{selection},ax-engine"

    try:
        from sglang.launch_server import run_server
        from sglang.srt.plugins import load_plugins
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils import kill_process_tree
    except ModuleNotFoundError as error:
        if error.name != "sglang":
            raise
        raise SystemExit(
            "SGLang is missing. Install ax-engine-sglang with its dependencies "
            "or use a CUDA-enabled SGLang environment."
        ) from None

    if argv in (["--help"], ["-h"]):
        prepare_server_args(argv)
        return

    try:
        load_plugins()
        plugin = sys.modules.get("ax_engine_sglang.plugin")
        if plugin is None or not plugin._registered:
            raise SystemExit("The AX Engine SGLang plugin failed to load. Check the SGLang startup logs.")
        run_server(prepare_server_args(argv))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
