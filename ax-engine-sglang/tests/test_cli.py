import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ax_engine_sglang import cli


class ServeTests(unittest.TestCase):
    def setUp(self):
        launcher = ModuleType("sglang.launch_server")
        launcher.run_server = self.run_server = Mock()
        plugins = ModuleType("sglang.srt.plugins")
        plugins.load_plugins = self.load_plugins = Mock()
        server_args = ModuleType("sglang.srt.server_args")
        server_args.prepare_server_args = self.prepare_server_args = Mock()
        utils = ModuleType("sglang.srt.utils")
        utils.kill_process_tree = self.kill_process_tree = Mock()
        self.plugin = ModuleType("ax_engine_sglang.plugin")
        self.plugin._registered = False
        modules = {module.__name__: module for module in (launcher, plugins, server_args, utils, self.plugin)}
        self.enterContext(patch.dict(sys.modules, modules))

    def test_serve_enables_ax_before_loading_and_passes_through_flags(self):
        def load_plugins():
            self.assertEqual(os.environ["SGLANG_PLUGINS"], "other-plugin,ax-engine")
            self.plugin._registered = True

        self.load_plugins.side_effect = load_plugins
        with patch.dict(os.environ, {"SGLANG_PLUGINS": "other-plugin"}):
            with patch.object(sys, "argv", ["ax-engine-sglang", "serve", "model", "--port", "30001"]):
                cli.main()

        self.prepare_server_args.assert_called_once_with(["--model-path", "model", "--port", "30001"])
        self.run_server.assert_called_once_with(self.prepare_server_args.return_value)
        self.kill_process_tree.assert_called_once_with(os.getpid(), include_parent=False)

    def test_serve_stops_when_ax_registration_fails(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "plugin failed to load"):
                cli.serve(["--model-path", "model"])
            self.assertNotIn("SGLANG_PLUGINS", os.environ)

        self.prepare_server_args.assert_not_called()
        self.run_server.assert_not_called()
        self.kill_process_tree.assert_called_once_with(os.getpid(), include_parent=False)

    def test_serve_help_uses_sglang_help_without_loading_plugins(self):
        with patch.object(sys, "argv", ["ax-engine-sglang", "serve", "--help"]):
            cli.main()

        self.prepare_server_args.assert_called_once_with(["--help"])
        self.load_plugins.assert_not_called()
        self.run_server.assert_not_called()
