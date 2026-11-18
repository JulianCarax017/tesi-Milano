"""Run the existing notebooks in fresh kernels; never reimplement experiments."""
import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from experiment_options import valid_run_id

NOTEBOOK_ROOT = Path("DATA/Alessandro Tesi/Tesi-main/Secondo_Esperimento")
DEFAULT_PROJECT = REPOSITORY_ROOT / "DATA/Alessandro Tesi/Tesi-main"
SEEDS = (42, 43, 44)
OPTION_VARIABLES = (
    "TESI_EPOCHS", "TESI_MERGED_MODE", "TESI_RUN_TRAINING", "TESI_CROSS_TASK_TRAINING",
    "TESI_ROUTED_ACTION", "TESI_ROUTED_RUN_ID", "TESI_ROUTED_RECEIPT",
    "TESI_ANALYSIS_RUN", "TESI_REPORT_RECEIPT", "TESI_REQUIRE_COMPLETE_RUN",
    "TESI_MEASURE_BASELINES",
)


@dataclass(frozen=True)
class Stage:
    name: str
    notebook: str
    train_baseline: bool = False
    cross_task: bool = False


STAGES = (
    Stage("preprocessing", "Preprocessing/Preprocessing.ipynb"),
    Stage("train-sst", "Baselines/TrainingOfUncasedModel/TrainingExpertSST.ipynb"),
    Stage("train-mnli", "Baselines/TrainingOfUncasedModel/TrainingExpertMNLI.ipynb"),
    Stage("evaluate-sst", "Baselines/EvaluationExpertSST/EvaluationExpertSST.ipynb", cross_task=True),
    Stage("evaluate-mnli", "Baselines/EvaluationExpertMNLI/EvaluationExpertMNLI.ipynb", cross_task=True),
    Stage("multitask", "Baselines/EvaluationExpertMultitaskBert/EvaluationExpertMultitaskBert.ipynb", train_baseline=True),
    Stage("weight-averaging", "Baselines/EvaluationExpertWeigthAveraging/EvaluationExpertWeigthAveraging.ipynb", train_baseline=True),
    Stage("routed", "Baselines/RoutedModel/RoutedModel.ipynb"),
    Stage("report", "Baselines/ExperimentalAnalysis/RouterPolicyAnalysis.ipynb"),
)


def stage_plan(mode):
    return list(STAGES if mode == "full" else STAGES[-2:] if mode == "routed" else STAGES[-1:])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("full", "routed", "report"))
    p.add_argument("--epochs", type=int, choices=(4, 8), default=8,
                   help="RoutedModel budget only; expert/baseline budgets stay unchanged")
    p.add_argument("--merged-mode", choices=("impose", "off", "auto"), default="impose",
                   help="Advanced ablation option; both thesis budget configurations use impose")
    p.add_argument("--project-root", type=Path,
                   help="Tesi-main directory containing Secondo_Esperimento")
    p.add_argument("--run-id", help="Explicit completed run for report mode")
    p.add_argument("--resume", metavar="RUN_ID", help="Resume an incomplete run in routed mode")
    p.add_argument("--measure-baselines", action="store_true",
                   help="In report mode, also run missing simple-baseline measurements on GPU")
    p.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / ".reproduction",
                   help="Local logs and executed notebooks (excluded from Git)")
    p.add_argument("--dry-run", action="store_true", help="Print/check the plan without running notebooks")
    p.add_argument("--yes", action="store_true", help="Confirm execution without the terminal prompt")
    return p


def completed_run(group):
    """Accept historical runs too, but never silently aggregate fewer than three seeds."""
    valid_run_id(group.name)
    if not group.is_dir():
        raise FileNotFoundError(f"Run not found: {group}")
    manifests = []
    for index, seed in enumerate(SEEDS, 1):
        path = group / f"{group.name}_seed_{index}" / "comparison_manifest.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        required = {f"{task}_{split}" for task in ("sst", "mnli")
                    for split in ("validation", "test")}
        if (item.get("measurement_status") != "completed"
                or item.get("status") != "evaluation_completed"
                or item.get("experiment_group", {}).get("seed") != seed
                or not required.issubset(item.get("splits", {}))):
            raise ValueError(f"Incomplete or mismatched seed {seed}: {path}")
        manifests.append(item)
    return manifests


def interactive(args, project):
    choice = input("Percorso: [1] tutto da zero, [2] solo RoutedModel e analisi, [3] report [2]: ").strip() or "2"
    if choice not in {"1", "2", "3"}:
        raise ValueError("Scegliere 1, 2 o 3")
    args.mode = {"1": "full", "2": "routed", "3": "report"}[choice]
    if args.mode == "report":
        runs = project / "Secondo_Esperimento/Baselines/RoutedModel/Results"
        print("Run disponibili (la selezione viene verificata prima dell'avvio):")
        if runs.is_dir():
            for group in sorted(runs.iterdir(), reverse=True):
                if group.is_dir():
                    try:
                        completed_run(group)
                    except (ValueError, OSError):
                        continue
                    print(" ", group.name)
        args.run_id = input("ID della run completata: ").strip()
        args.measure_baselines = input("Eseguire nuove misure delle baseline semplici sulla GPU? [s/N]: ").strip().lower() == "s"
    else:
        value = input("Epoche RoutedModel: [4] alternativa breve, [8] riferimento [8]: ").strip() or "8"
        if value not in {"4", "8"}:
            raise ValueError("Scegliere 4 o 8 epoche")
        args.epochs = int(value)
    return args


def validate_arguments(args):
    if args.resume and args.mode != "routed":
        raise ValueError("--resume is available only with --mode routed")
    if args.run_id and args.mode != "report":
        raise ValueError("--run-id is available only with --mode report")
    if args.mode == "report" and not args.run_id:
        raise ValueError("--mode report requires --run-id (no automatic latest-run selection)")
    for value in (args.run_id, args.resume):
        if value:
            valid_run_id(value)


def preflight(args, project):
    for stage in stage_plan(args.mode):
        path = REPOSITORY_ROOT / NOTEBOOK_ROOT / stage.notebook
        if not path.is_file():
            raise FileNotFoundError(f"Notebook missing: {path}")
    baselines = project / "Secondo_Esperimento/Baselines"
    runs = baselines / "RoutedModel/Results"
    if args.mode == "report":
        completed_run(runs / args.run_id)
        return
    if args.mode == "routed":
        missing = []
        data = project / "Secondo_Esperimento/Preprocessing/Dataset/ProcessedDB"
        for task in ("SST", "MNLI"):
            for field in ("X", "y"):
                for split in ("train", "val", "test"):
                    path = data / task / f"{field}_{split}.parquet"
                    if not path.is_file():
                        missing.append(path)
        for task in ("sst", "mnli"):
            expert = baselines / "TrainingOfUncasedModel/Models" / f"{task}_bert_base_uncased"
            for path in (expert / "classifier.pt", expert / "bert_backbone/config.json"):
                if not path.is_file():
                    missing.append(path)
            if not (list((expert / "bert_backbone").glob("*.safetensors"))
                    or list((expert / "bert_backbone").glob("pytorch_model*.bin"))):
                missing.append(expert / "bert_backbone/<weights>")
        if missing:
            raise FileNotFoundError("Prepare data/experts with --mode full or select their --project-root:\n"
                                    + "\n".join(str(p) for p in missing))
        if args.resume:
            state = json.loads((runs / args.resume / "execution_manifest.json").read_text(encoding="utf-8"))
            expected = {"epochs": args.epochs, "merged_mode": args.merged_mode, "seeds": list(SEEDS)}
            if state.get("run_options") != expected or state.get("status") == "completed":
                raise ValueError("Resume requires an incomplete run with the same recorded epochs/mode/seeds")


def stage_environment(args, stage, project, session, run_id=None):
    env = os.environ.copy()
    for name in OPTION_VARIABLES:
        env.pop(name, None)
    env.update({
        "TESI_REPOSITORY_ROOT": str(REPOSITORY_ROOT), "TESI_PROJECT_ROOT": str(project),
        "TESI_EPOCHS": str(args.epochs), "TESI_MERGED_MODE": args.merged_mode,
        "TESI_RUN_TRAINING": str(stage.train_baseline).lower(),
        "TESI_CROSS_TASK_TRAINING": str(stage.cross_task).lower(),
        "TESI_ROUTED_ACTION": "resume" if args.resume else "new",
        "TESI_REQUIRE_COMPLETE_RUN": "true",
        "TESI_MEASURE_BASELINES": str(args.mode != "report" or args.measure_baselines).lower(),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / NOTEBOOK_ROOT / "Baselines")]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])),
        "JUPYTER_PATH": str(session / "jupyter") + (os.pathsep + env["JUPYTER_PATH"] if env.get("JUPYTER_PATH") else ""),
    })
    if stage.name == "routed":
        env["TESI_ROUTED_RECEIPT"] = str(session / "routed-receipt.json")
        if args.resume:
            env["TESI_ROUTED_RUN_ID"] = args.resume
    if stage.name == "report":
        if not run_id:
            raise ValueError("The report must be linked to an explicit run")
        env["TESI_ANALYSIS_RUN"] = valid_run_id(run_id)
        env["TESI_REPORT_RECEIPT"] = str(session / "report-receipt.json")
    return env


def execute_stage(stage, env, session):
    output = session / f"{stage.name}.ipynb"
    log = session / f"{stage.name}.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--_execute",
               str(REPOSITORY_ROOT / NOTEBOOK_ROOT / stage.notebook), str(output)]
    print(f"Executing {stage.name}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def execute_notebook(source, output):
    """Private worker: isolate Python state and preserve partial outputs on failure."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    class ProgressExecutor(ExecutePreprocessor):
        def preprocess_cell(self, cell, resources, index):
            print(f"Cell {index + 1}: start", flush=True)
            result = super().preprocess_cell(cell, resources, index)
            print(f"Cell {index + 1}: completed", flush=True)
            return result

    notebook = nbformat.read(source, as_version=4)
    executor = ProgressExecutor(timeout=None, kernel_name="tesi-reproduce", allow_errors=False)
    try:
        executor.preprocess(notebook, {"metadata": {"path": str(REPOSITORY_ROOT)}})
    finally:
        nbformat.write(notebook, output)


def check_runtime(needs_gpu):
    missing = [name for name in ("nbformat", "nbconvert", "ipykernel")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Install requirements-notebooks.txt in this Python environment; missing: "
                           + ", ".join(missing))
    if needs_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This workflow requires a CUDA GPU; no training has been launched")


def run(args, project):
    needs_gpu = args.mode != "report" or args.measure_baselines
    check_runtime(needs_gpu)
    session = args.output_root.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session.mkdir(parents=True, exist_ok=False)
    kernel = session / "jupyter/kernels/tesi-reproduce"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Thesis reproduction", "language": "python",
    }), encoding="utf-8")
    if args.mode == "full":
        (project / "Secondo_Esperimento").mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "mode": args.mode, "epochs": args.epochs,
              "merged_mode": args.merged_mode, "seeds": list(SEEDS), "project_root": str(project),
              "python": sys.executable, "completed_stages": [], "run_id": args.run_id,
              "measure_baselines": needs_gpu, "source": str(REPOSITORY_ROOT)}
    source_paths = [REPOSITORY_ROOT / NOTEBOOK_ROOT / s.notebook for s in stage_plan(args.mode)]
    source_paths += [Path(__file__).resolve(), REPOSITORY_ROOT / "experiment_options.py",
                     REPOSITORY_ROOT / "project_paths.py",
                     REPOSITORY_ROOT / NOTEBOOK_ROOT / "Baselines/classifier.py"]
    record["source_sha256"] = {
        str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths}
    manifest = session / "launcher.json"
    run_id = args.run_id
    try:
        for stage in stage_plan(args.mode):
            record["current_stage"] = stage.name
            manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
            env = stage_environment(args, stage, project, session, run_id)
            execute_stage(stage, env, session)
            if stage.name == "routed":
                receipt = json.loads((session / "routed-receipt.json").read_text(encoding="utf-8"))
                run_id = valid_run_id(receipt["run_id"])
                expected = {"epochs": args.epochs, "merged_mode": args.merged_mode, "seeds": list(SEEDS)}
                if receipt.get("run_options") != expected or (args.resume and run_id != args.resume):
                    raise ValueError("Routed run receipt does not match the requested configuration")
                group = project / "Secondo_Esperimento/Baselines/RoutedModel/Results" / run_id
                state = json.loads((group / "execution_manifest.json").read_text(encoding="utf-8"))
                if state.get("status") != "completed" or state.get("run_options") != expected:
                    raise ValueError("Routed execution is not complete")
                completed_run(group)
                record["run_id"] = run_id
            if stage.name == "report":
                receipt = json.loads((session / "report-receipt.json").read_text(encoding="utf-8"))
                if receipt.get("run_id") != run_id or receipt.get("measurement_error"):
                    raise ValueError(f"Report/measurements incomplete: {receipt}")
                if env["TESI_MEASURE_BASELINES"] == "true" and receipt.get("comparisons") != 2:
                    raise ValueError("Simple-baseline comparisons are incomplete")
                report = Path(receipt["report_path"]).resolve()
                expected_group = project / "Secondo_Esperimento/Baselines/RoutedModel/Results" / run_id
                if report.parent != expected_group.resolve() or not report.is_file():
                    raise ValueError("Report path does not belong to the selected run")
                record["report_path"] = str(report)
            record["completed_stages"].append(stage.name)
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt_path = session / "routed-receipt.json"
        if record.get("run_id") is None and receipt_path.is_file():
            try:
                record["run_id"] = valid_run_id(json.loads(receipt_path.read_text(encoding="utf-8"))["run_id"])
            except (OSError, ValueError, KeyError):
                pass
        manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Execution record: {manifest}", flush=True)
    print(f"Completed. Report: {record['report_path']}")
    return session


def main(argv=None):
    args = parser().parse_args(argv)
    project = (args.project_root or Path(os.environ.get("TESI_PROJECT_ROOT", DEFAULT_PROJECT))).expanduser().resolve()
    if args.mode is None:
        if sys.stdin.isatty() and not args.dry_run:
            args = interactive(args, project)
        else:
            args.mode = "full"
    validate_arguments(args)
    preflight(args, project)
    print(f"Project: {project}\nMode: {args.mode}")
    if args.mode != "report":
        print(f"RoutedModel: {args.epochs} epochs, merged={args.merged_mode}; seeds={SEEDS}")
    print("Sequence: " + " -> ".join(s.name for s in stage_plan(args.mode)))
    if args.mode == "full":
        print("Full mode trains experts and baselines; existing outputs at this project root may be replaced.")
    if args.mode == "report":
        print(f"Selected run: {args.run_id}; new GPU measurements: {args.measure_baselines}")
    if args.dry_run:
        print("Dry run: no notebooks executed, no directories created.")
        return 0
    if not args.yes:
        if not sys.stdin.isatty() or input("Avviare questo percorso? [s/N]: ").strip().lower() != "s":
            print("Cancelled. Use --yes for an unattended invocation.")
            return 0
    run(args, project)
    return 0


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--_execute":
            execute_notebook(sys.argv[2], sys.argv[3])
        else:
            raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
