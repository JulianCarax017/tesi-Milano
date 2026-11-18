"""Offline orchestration tests. No dataset/model imports, downloads or GPU work."""
import ast
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment_options as options
from scripts import reproduce as runner

RUN_ID = "20261118_120000_000001"
OTHER_ID = "20261118_130000_000001"
CONFIG = {"epochs": 4, "merged_mode": "impose", "seeds": [42, 43, 44]}

def temporary_workspace():
    temporary = tempfile.TemporaryDirectory()
    # The managed Windows temp root is long; fixtures include timestamp/seed folders.
    if os.name == "nt":
        temporary.name = "\\\\?\\" + temporary.name
    return temporary


def save_json(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def make_completed(project, run_id=RUN_ID, config=CONFIG):
    group = project / "Secondo_Esperimento/Baselines/RoutedModel/Results" / run_id
    group.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "timestamp": run_id, "status": "completed",
             "seeds": [42, 43, 44], "run_options": config}
    save_json(group / "execution_manifest.json", state)
    for index, seed in enumerate((42, 43, 44), 1):
        folder = group / f"{run_id}_seed_{index}"
        folder.mkdir(exist_ok=True)
        save_json(folder / "comparison_manifest.json", {
            "status": "evaluation_completed", "measurement_status": "completed",
            "experiment_group": {"seed": seed},
            "splits": {f"{task}_{split}": {} for task in ("sst", "mnli")
                       for split in ("validation", "test")},
        })
    return group


class OptionTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(options.option_int("TESI_EPOCHS", 8), 8)
            self.assertFalse(options.option_bool("TESI_RUN_TRAINING", False))
            self.assertTrue(options.option_bool("TESI_MEASURE_BASELINES", True))
            self.assertEqual(options.option_choice("MODE", "impose", ("off", "impose")), "impose")
        for value in ("0", "-1", "4.0", "abc"):
            with patch.dict(os.environ, {"TESI_EPOCHS": value}, clear=True):
                with self.assertRaises(ValueError):
                    options.option_int("TESI_EPOCHS", 8)
        with patch.dict(os.environ, {"FLAG": "maybe"}, clear=True):
            with self.assertRaises(ValueError):
                options.option_bool("FLAG", False)

    def test_invalid_run_ids(self):
        for value in ("../" + RUN_ID, "/tmp/" + RUN_ID, "", "latest"):
            with self.assertRaises(ValueError):
                options.valid_run_id(value)

    def test_new_run_does_not_resume_or_overwrite_incomplete(self):
        with temporary_workspace() as tmp, patch.dict(os.environ, {"TESI_ROUTED_ACTION": "new"}, clear=True):
            root = Path(tmp)
            path, first = options.select_execution(root, (42, 43, 44), CONFIG, save_json)
            before = path.read_bytes()
            other_path, second = options.select_execution(root, (42, 43, 44), CONFIG, save_json)
            self.assertNotEqual(first["timestamp"], second["timestamp"])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(second["run_options"], CONFIG)

    def test_resume_rejects_budget_change_before_skipping_completed_seed(self):
        with temporary_workspace() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            path, state = options.select_execution(root, (42, 43, 44), CONFIG, save_json)
            state["seed_states"]["1"]["status"] = "completed"
            save_json(path, state)
            before = path.read_bytes()
            with patch.dict(os.environ, {"TESI_ROUTED_ACTION": "resume", "TESI_ROUTED_RUN_ID": state["timestamp"]}):
                with self.assertRaisesRegex(ValueError, "different epochs"):
                    options.select_execution(root, (42, 43, 44), dict(CONFIG, epochs=8), save_json)
            self.assertEqual(path.read_bytes(), before)

    def test_explicit_resume_selects_requested_run_not_latest(self):
        with temporary_workspace() as tmp, patch.dict(os.environ, {"TESI_ROUTED_ACTION": "new"}, clear=True):
            root = Path(tmp)
            path, first = options.select_execution(root, (42, 43, 44), CONFIG, save_json)
            options.select_execution(root, (42, 43, 44), CONFIG, save_json)
            with patch.dict(os.environ, {"TESI_ROUTED_ACTION": "resume", "TESI_ROUTED_RUN_ID": first["timestamp"]}):
                resumed_path, resumed = options.select_execution(root, (42, 43, 44), CONFIG, save_json)
            self.assertEqual(path, resumed_path)
            self.assertEqual(first["timestamp"], resumed["timestamp"])

    def test_auto_still_resumes_manual_notebook(self):
        with temporary_workspace() as tmp, patch.dict(os.environ, {}, clear=True):
            path, first = options.select_execution(Path(tmp), (42, 43, 44), CONFIG, save_json)
            same_path, same = options.select_execution(Path(tmp), (42, 43, 44), CONFIG, save_json)
            self.assertEqual(path, same_path)
            self.assertEqual(first["timestamp"], same["timestamp"])


class WorkflowTests(unittest.TestCase):
    def args(self, mode, output, extra=()):
        return runner.parser().parse_args(["--mode", mode, "--epochs", "4",
                                          "--output-root", str(output), "--yes", *extra])

    def test_plan_contains_no_duplicate_seed_loop(self):
        self.assertEqual([s.name for s in runner.stage_plan("full")],
                         ["preprocessing", "train-sst", "train-mnli", "evaluate-sst", "evaluate-mnli",
                          "multitask", "weight-averaging", "routed", "report"])
        self.assertEqual([s.name for s in runner.stage_plan("routed")], ["routed", "report"])
        self.assertEqual([s.name for s in runner.stage_plan("report")], ["report"])

    def test_stage_flags_override_inherited_environment(self):
        args = self.args("full", Path("/unused"))
        with patch.dict(os.environ, {"TESI_RUN_TRAINING": "false", "TESI_ANALYSIS_RUN": OTHER_ID}):
            env = runner.stage_environment(args, runner.STAGES[5], Path("/project"), Path("/session"))
        self.assertEqual(env["TESI_RUN_TRAINING"], "true")
        self.assertEqual(env["TESI_CROSS_TASK_TRAINING"], "false")
        self.assertNotIn("TESI_ANALYSIS_RUN", env)
        self.assertEqual(env["TESI_EPOCHS"], "4")
        self.assertIn(str(runner.REPOSITORY_ROOT / runner.NOTEBOOK_ROOT / "Baselines"), env["PYTHONPATH"])
        env = runner.stage_environment(args, runner.STAGES[3], Path("/project"), Path("/session"))
        self.assertEqual(env["TESI_CROSS_TASK_TRAINING"], "true")

    def test_report_is_explicit_and_cached_by_default(self):
        args = self.args("report", Path("/unused"), ("--run-id", RUN_ID))
        runner.validate_arguments(args)
        env = runner.stage_environment(args, runner.STAGES[-1], Path("/project"), Path("/session"), RUN_ID)
        self.assertEqual(env["TESI_ANALYSIS_RUN"], RUN_ID)
        self.assertEqual(env["TESI_MEASURE_BASELINES"], "false")
        args.run_id = None
        with self.assertRaises(ValueError):
            runner.validate_arguments(args)

    def test_completed_run_requires_all_three_seeds(self):
        with temporary_workspace() as tmp:
            project = Path(tmp)
            group = make_completed(project)
            self.assertEqual(len(runner.completed_run(group)), 3)
            manifest = group / f"{RUN_ID}_seed_3/comparison_manifest.json"
            item = json.loads(manifest.read_text())
            item["measurement_status"] = "failed"
            save_json(manifest, item)
            with self.assertRaises(ValueError):
                runner.completed_run(group)

    def test_full_sequence_hands_exact_run_to_report(self):
        with temporary_workspace() as tmp, patch.object(runner, "check_runtime"):
            project, output = Path(tmp) / "project", Path(tmp) / "logs"
            args = self.args("full", output)
            calls = []
            def fake_stage(stage, env, session):
                calls.append(stage.name)
                if stage.name == "routed":
                    group = make_completed(project)
                    make_completed(project, OTHER_ID)
                    save_json(session / "routed-receipt.json", {
                        "run_id": RUN_ID, "run_options": CONFIG})
                elif stage.name == "report":
                    self.assertEqual(env["TESI_ANALYSIS_RUN"], RUN_ID)
                    report = project / "Secondo_Esperimento/Baselines/RoutedModel/Results" / RUN_ID / "routing_policy_report.html"
                    report.write_text("report")
                    save_json(session / "report-receipt.json", {
                        "run_id": RUN_ID, "report_path": str(report),
                        "measurement_error": None, "comparisons": 2})
            with patch.object(runner, "execute_stage", side_effect=fake_stage):
                session = runner.run(args, project)
            record = json.loads((session / "launcher.json").read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["run_id"], RUN_ID)
            self.assertEqual(calls, [s.name for s in runner.stage_plan("full")])
            kernel = json.loads((session / "jupyter/kernels/tesi-reproduce/kernel.json").read_text())
            self.assertEqual(kernel["argv"][0], sys.executable)

    def test_stage_failure_stops_pipeline_and_preserves_failure_record(self):
        with temporary_workspace() as tmp, patch.object(runner, "check_runtime"):
            args = self.args("full", Path(tmp) / "logs")
            calls = []
            def fail(stage, env, session):
                calls.append(stage.name)
                if stage.name == "train-sst":
                    raise subprocess.CalledProcessError(1, ["synthetic-worker"])
            with patch.object(runner, "execute_stage", side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run(args, Path(tmp) / "project")
            self.assertEqual(calls, ["preprocessing", "train-sst"])
            record = json.loads(next((Path(tmp) / "logs").glob("*/launcher.json")).read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["completed_stages"], ["preprocessing"])

    def test_swallowed_report_measurement_error_is_not_success(self):
        with temporary_workspace() as tmp, patch.object(runner, "check_runtime"):
            args = self.args("report", Path(tmp) / "logs", ("--run-id", RUN_ID))
            def fake(stage, env, session):
                save_json(session / "report-receipt.json", {
                    "run_id": RUN_ID, "measurement_error": "synthetic measurement failure"})
            with patch.object(runner, "execute_stage", side_effect=fake):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    runner.run(args, Path(tmp) / "project")
            record = json.loads(next((Path(tmp) / "logs").glob("*/launcher.json")).read_text())
            self.assertEqual(record["status"], "failed")

    def test_dry_run_creates_no_output_and_imports_no_experiment_packages(self):
        with temporary_workspace() as tmp:
            project, output = Path(tmp) / "new-project", Path(tmp) / "new-logs"
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts/reproduce.py"), "--mode", "full",
                "--epochs", "4", "--project-root", str(project),
                "--output-root", str(output), "--dry-run",
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("preprocessing -> train-sst", completed.stdout)
            self.assertFalse(project.exists())
            self.assertFalse(output.exists())

    def test_worker_preserves_partial_notebook_on_error(self):
        notebook = {"cells": [{"outputs": []}]}
        written = []
        fake_format = types.ModuleType("nbformat")
        fake_format.read = lambda *a, **k: notebook
        fake_format.write = lambda nb, target: written.append((nb, target))
        fake_preprocessors = types.ModuleType("nbconvert.preprocessors")
        class FakeExecutor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def preprocess(self, nb, resources):
                nb["cells"][0]["outputs"].append("partial output")
                raise RuntimeError("synthetic kernel failure")
        fake_preprocessors.ExecutePreprocessor = FakeExecutor
        fake_convert = types.ModuleType("nbconvert")
        fake_convert.preprocessors = fake_preprocessors
        with patch.dict(sys.modules, {"nbformat": fake_format, "nbconvert": fake_convert,
                                      "nbconvert.preprocessors": fake_preprocessors}):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                runner.execute_notebook("input.ipynb", "output.ipynb")
        self.assertEqual(written[0][0]["cells"][0]["outputs"], ["partial output"])
        self.assertEqual(written[0][1], "output.ipynb")

    def test_schedules_keep_the_original_four_and_eight_epoch_endpoints(self):
        path = ROOT / runner.NOTEBOOK_ROOT / "Baselines/RoutedModel/RoutedModel.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = next("".join(c["source"]) for c in nb["cells"]
                    if "def configure_loss(" in "".join(c["source"]))
        outer = ast.parse(code).body[0]
        functions = [node for node in outer.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"schedule_endpoint", "scheduled_weight"}]
        import math
        scope = {"math": math}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<schedule-test>", "exec"), scope)
        endpoint = scope["schedule_endpoint"]
        self.assertEqual([endpoint(n, 5 / 8) for n in (4, 8)], [3, 5])
        self.assertEqual([endpoint(n, 1.0) for n in (4, 8)], [4, 8])
        weight = scope["scheduled_weight"]
        self.assertAlmostEqual(weight(2, 4, .020, .030, 5/8), .030)
        self.assertAlmostEqual(weight(4, 8, .020, .030, 5/8), .030)

    def test_analysis_notebook_selects_requested_older_run_and_rejects_partial_group(self):
        import re
        nb = json.loads((ROOT / runner.NOTEBOOK_ROOT /
                        "Baselines/ExperimentalAnalysis/RouterPolicyAnalysis.ipynb").read_text(encoding="utf-8"))
        code = next("".join(c["source"]) for c in nb["cells"]
                    if "def latest_completed_run(" in "".join(c["source"]))
        functions = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)
                     and node.name in {"read_json", "latest_completed_run"}]
        scope = {"Path": Path, "json": json, "re": re, "_repository_os": os,
                 "option_bool": options.option_bool, "valid_run_id": options.valid_run_id,
                 "TASKS": ("sst", "mnli"), "NOTES": [], "SOURCES": []}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<analysis-selection-test>", "exec"), scope)
        with temporary_workspace() as tmp:
            project = Path(tmp)
            group = make_completed(project)
            make_completed(project, OTHER_ID)
            with patch.dict(os.environ, {"TESI_ANALYSIS_RUN": RUN_ID,
                                         "TESI_REQUIRE_COMPLETE_RUN": "true"}, clear=True):
                selected, completed = scope["latest_completed_run"](group.parent)
                self.assertEqual(selected, group)
                self.assertEqual(len(completed), 3)
                file = group / f"{RUN_ID}_seed_3/comparison_manifest.json"
                item = json.loads(file.read_text())
                item["measurement_status"] = "pending"
                save_json(file, item)
                with self.assertRaises(ValueError):
                    scope["latest_completed_run"](group.parent)

    def test_initial_routed_options_are_effective_and_defaults_stay_eight_impose(self):
        nb = json.loads((ROOT / runner.NOTEBOOK_ROOT /
                        "Baselines/RoutedModel/RoutedModel.ipynb").read_text(encoding="utf-8"))
        code = next("".join(c["source"]) for c in nb["cells"]
                    if "ROUTED_EPOCHS = " in "".join(c["source"]))
        for environment, expected in (({}, 8), ({"TESI_EPOCHS": "4"}, 4)):
            scope = {"option_int": options.option_int, "option_choice": options.option_choice,
                     "TESI_PROJECT_ROOT": ROOT}
            with patch.dict(os.environ, environment, clear=True):
                exec(compile(code, "<routed-options-test>", "exec"), scope)
            self.assertEqual(scope["ROUTED_EPOCHS"], expected)
            self.assertEqual(scope["ROUTED_MERGED_MODE"], "impose")
            self.assertEqual(scope["RUN_SEEDS"], (42, 43, 44))


if __name__ == "__main__":
    unittest.main()
