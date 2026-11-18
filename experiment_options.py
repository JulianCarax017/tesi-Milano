"""Optional notebook settings and run selection; standard library only.

Without TESI_* overrides, notebooks retain their original settings.
"""
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path


def option_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = {"1": True, "true": True, "yes": True,
              "0": False, "false": False, "no": False}
    try:
        return values[raw.strip().lower()]
    except KeyError:
        raise ValueError(f"{name} must be true or false") from None


def option_int(name, default):
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer >= 1") from None
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


def option_choice(name, default, choices):
    value = os.environ.get(name, default)
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return value


def valid_run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?", value):
        raise ValueError("Run ID must be a timestamp directory name, not a path")
    return value


def write_receipt(variable, **record):
    """Only the launcher requests these small local hand-off files."""
    destination = os.environ.get(variable)
    if destination:
        path = Path(destination)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(path)


def select_execution(root, seeds, options, save_json):
    """Preserve automatic notebook resume; add explicit new/resume selection."""
    root = Path(root)
    results = root / "Results"
    results.mkdir(parents=True, exist_ok=True)
    action = option_choice("TESI_ROUTED_ACTION", "auto", ("auto", "new", "resume"))
    requested = os.environ.get("TESI_ROUTED_RUN_ID")
    if requested and action != "resume":
        raise ValueError("TESI_ROUTED_RUN_ID requires TESI_ROUTED_ACTION=resume")
    if action == "resume" and not requested:
        raise ValueError("Explicit resume requires TESI_ROUTED_RUN_ID")
    folders = sorted(p for p in results.iterdir()
                     if p.is_dir() and re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?", p.name))
    parent = None
    state = None
    if action == "resume":
        parent = results / valid_run_id(requested)
        if not parent.is_dir():
            raise FileNotFoundError(f"Run not found: {parent}")
    elif action == "auto" and folders:
        parent = folders[-1]
    if parent is not None:
        manifest = parent / "execution_manifest.json"
        if manifest.is_file():
            state = json.loads(manifest.read_text(encoding="utf-8"))
            if state.get("schema_version") != 1 or state.get("timestamp") != parent.name:
                raise ValueError("Unsupported or inconsistent execution manifest")
        else:
            # Compatibility with original per-seed manifests during manual use.
            entries = {}
            for index, seed in enumerate(seeds, 1):
                path = parent / f"{parent.name}_seed_{index}" / "comparison_manifest.json"
                item = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                complete = (item.get("measurement_status") == "completed"
                            and item.get("status") == "evaluation_completed"
                            and item.get("experiment_group", {}).get("seed") == seed)
                entries[str(index)] = {"seed": seed, "status": "completed" if complete else "pending",
                                       "epochs_completed": 0}
            finished = (all(x["status"] == "completed" for x in entries.values())
                        and (parent / "statistical_summary.csv").is_file())
            state = {"schema_version": 1, "timestamp": parent.name, "seeds": list(seeds),
                     "status": "completed" if finished else "in_progress", "seed_states": entries}
        if state.get("status") == "completed":
            if action == "resume":
                raise ValueError("This run is already completed; select report or start a new run")
            state = None
        else:
            if state.get("seeds") != list(seeds):
                raise ValueError("Incomplete run uses different seeds")
            recorded = state.get("run_options")
            if recorded is not None and recorded != options:
                raise ValueError("Incomplete run uses different epochs or merged mode; start a new run")
            if action == "resume" and recorded is None:
                raise ValueError("Legacy run has no run_options; resume it manually in the notebook")
    if state is None:
        moment = datetime.now().astimezone()
        while True:
            stamp = moment.strftime("%Y%m%d_%H%M%S_%f")
            parent = results / stamp
            try:
                parent.mkdir(exist_ok=False)
            except FileExistsError:
                moment += timedelta(microseconds=1)
            else:
                break
        state = {"schema_version": 1, "timestamp": stamp, "seeds": list(seeds),
                 "run_options": options, "status": "in_progress", "seed_states": {
                     str(i): {"seed": seed, "status": "pending", "epochs_completed": 0}
                     for i, seed in enumerate(seeds, 1)}}
        print("New execution:", stamp)
    else:
        print("Resuming execution:", state["timestamp"])
    path = parent / "execution_manifest.json"
    save_json(path, state)
    write_receipt("TESI_ROUTED_RECEIPT", run_id=state["timestamp"],
                  manifest=str(path), run_options=state.get("run_options"))
    return path, state
