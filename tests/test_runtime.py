"""Dependency-light regressions; training, downloads and GPU calls are mocked."""

import ast
from contextlib import redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CACHED_SCRIPTS = (
    "fedln/fedln__nnc.py", "fedln/fedln__akd.py",
    "fedln/fedln__nafedavg.py", "baselines/fedavg__cl.py",
)


def load_module(relative, modules=None):
    spec = importlib.util.spec_from_file_location("fedln_test_module", SRC / relative)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules or {}):
        spec.loader.exec_module(module)
    return module


def load_function(relative, name, namespace):
    """Isolate orchestration from unavailable TensorFlow/Flower imports."""
    tree = ast.parse((SRC / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC / relative), "exec"), namespace)
    return namespace[name]


class LauncherTests(unittest.TestCase):
    def test_each_experiment_help_runs_before_optional_imports(self):
        for script in sorted(SRC.glob("baselines/*.py")) + sorted(SRC.glob("fedln/*.py")):
            with self.subTest(script=script.name):
                result = subprocess.run([sys.executable, "-S", str(script), "--help"],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--dataset_name", result.stdout)

    def test_help_needs_no_python_or_training_dependencies(self):
        result = subprocess.run(["/bin/bash", str(SRC / "run.sh"), "--help"],
                                text=True, capture_output=True, env={"PATH": "/nonexistent"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage:", result.stdout)
        self.assertIn("na_fedavg", result.stdout)

    def test_unknown_method_fails(self):
        result = subprocess.run(["/bin/bash", str(SRC / "run.sh"), "invalid"],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_executable_dispatch_preserves_arguments_from_another_directory(self):
        self.assertTrue(os.access(SRC / "run.sh", os.X_OK), "run.sh must be executable")
        methods = {"centralized": "baselines/centralized.py", "fedavg": "baselines/fedavg__standard.py",
                   "nnc": "fedln/fedln__nnc.py", "akd": "fedln/fedln__akd.py",
                   "na_fedavg": "fedln/fedln__nafedavg.py",
                   "confidence_learning": "baselines/fedavg__cl.py",
                   "bi_tempered_loss": "baselines/fedavg__loss.py",
                   "label_smoothing": "baselines/fedavg__smooth.py"}
        with tempfile.TemporaryDirectory(prefix="fedln dispatch ") as directory:
            python = Path(directory) / "python"
            fixture = Path(directory) / "capture arguments.py"
            fixture.write_text("import json,os,sys\nprint(json.dumps([os.getcwd(),sys.argv[1:]]))\n")
            # A shebang cannot quote an interpreter path containing spaces.
            python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " "
                              + shlex.quote(str(fixture)) + ' "$@"\n')
            python.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ.get("PATH", ""))
            for method, expected in methods.items():
                with self.subTest(method=method):
                    result = subprocess.run([str(SRC / "run.sh"), method, "--temp_dir", "with spaces"],
                                            cwd=directory, env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    cwd, argv = json.loads(result.stdout.splitlines()[-1])
                    self.assertEqual(Path(cwd), SRC)
                    self.assertEqual((Path(cwd) / argv[0]).resolve(), SRC / expected)
                    self.assertEqual(argv[1:], ["--temp_dir", "with spaces"])


class DataAndGpuTests(unittest.TestCase):
    def eurosat_module(self):
        numpy = Mock()
        numpy.load.side_effect = lambda filename: open(filename, "rb")
        modules = {"tensorflow": SimpleNamespace(data=SimpleNamespace(experimental=SimpleNamespace(AUTOTUNE=1))),
                   "numpy": numpy, "utils.image_utils": Mock(), "utils.noise": Mock(), "utils.shard": Mock()}
        return load_module("data/eurosat.py", modules)

    def test_eurosat_import_does_not_require_environment_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.eurosat_module()

    def test_missing_eurosat_file_reports_expected_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TFDS_DATA_DIR": directory}):
            module = self.eurosat_module()
            with self.assertRaisesRegex(FileNotFoundError, "REPRODUCIBILITY"):
                module.load_eurosat_test()

    def test_no_gpu_raises_instead_of_sleeping_forever(self):
        module = load_module("utils/gpu_utils.py", {"GPUtil": SimpleNamespace(
            getGPUs=lambda: [], getAvailable=lambda **kwargs: [])})
        with patch.object(module, "sleep", side_effect=AssertionError("No GPU must fail before waiting")):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA"):
                module.get_available_gpu()

    def test_available_gpu_selection_is_unchanged(self):
        module = load_module("utils/gpu_utils.py", {"GPUtil": SimpleNamespace(
            getGPUs=lambda: [object()], getAvailable=lambda **kwargs: [2])})
        self.assertEqual(module.get_available_gpu(), "2")


class CacheTests(unittest.TestCase):
    def test_import_does_not_delete_supplied_directory(self):
        for relative in CACHED_SCRIPTS:
            with self.subTest(script=relative), tempfile.TemporaryDirectory() as directory:
                sentinel = Path(directory) / "user-result.txt"
                sentinel.write_text("keep")
                with patch.object(sys, "argv", [relative, "--temp_dir", directory]):
                    load_module(relative, {"flwr": Mock(), "GPUtil": Mock()})
                self.assertTrue(sentinel.exists(), "Import deleted user-owned files")

    def test_simulations_use_private_cache_and_preserve_parent(self):
        for relative in CACHED_SCRIPTS:
            for fail in (False, True):
                with self.subTest(script=relative, fail=fail), tempfile.TemporaryDirectory() as directory:
                    sentinel = Path(directory) / "user-result.txt"
                    sentinel.write_text("keep")
                    args = SimpleNamespace(temp_dir=directory, num_clients=2, num_rounds=1)
                    observed = []

                    def simulate(**kwargs):
                        observed.append(Path(args.temp_dir))
                        (Path(args.temp_dir) / "score_0.npy").write_text("test cache")
                        if fail:
                            raise RuntimeError("simulation failed")
                        return "history"

                    namespace = {"args": args, "Path": Path, "shutil": shutil,
                                 "create_server": lambda: object(), "create_client": lambda cid: None,
                                 "fl": SimpleNamespace(simulation=SimpleNamespace(start_simulation=simulate),
                                                       server=SimpleNamespace(ServerConfig=lambda **kw: kw))}
                    if (SRC / "utils/runtime.py").exists():
                        namespace["temporary_cache_directory"] = load_module("utils/runtime.py").temporary_cache_directory
                    run = load_function(relative, "run_simulation", namespace)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                            run()
                    else:
                        self.assertEqual(run(), "history")
                    self.assertTrue(sentinel.exists(), "Simulation deleted user-owned files")
                    self.assertEqual(observed[0].parent, Path(directory))
                    self.assertFalse(observed[0].exists(), "Owned cache was not cleaned up")
                    self.assertEqual(args.temp_dir, directory)


class CentralizedTests(unittest.TestCase):
    def test_evaluation_accepts_three_value_loader_contract(self):
        model = Mock()
        model.evaluate.return_value = [0.2, 0.8]
        training_data = SimpleNamespace(element_spec=[SimpleNamespace(shape=(None, 32, 32, 3))])
        args = SimpleNamespace(model_name="cnn", dataset_name="cifar10", batch_size=8,
                               seed=42, noisy_frac=0.8, noise_level=0.4, noise_sparsity=0.7,
                               lr=0.001, epochs=1)
        def datasets(train=True):
            return {"cifar10": (lambda **kw: (training_data, 10, 8, None, None)) if train
                    else (lambda **kw: ("test_dataset", 10, 8))}
        run = load_function("baselines/centralized.py", "execute_experiment", {
            "args": args, "tf": Mock(), "load_available_models": lambda: {"cnn": lambda **kw: model},
            "load_available_datasets": datasets})
        with redirect_stdout(io.StringIO()):
            run()
        model.evaluate.assert_called_once_with("test_dataset")


if __name__ == "__main__":
    unittest.main()
