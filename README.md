# tesi-Milano

Research notebooks for adaptive layer routing across frozen BERT experts on binary sentiment classification and three-class natural language inference. The routed model selects expert layers or skips encoder computation. The repository also contains preprocessing, expert training, baseline evaluation, and routing-policy analysis.

## Repository layout

The original directory hierarchy is preserved. `DATA` inside this repository is a **relative folder**, not the filesystem root `/DATA`.

```text
tesi-Milano/
├── README.md
├── requirements.txt
├── requirements-notebooks.txt
├── project_paths.py
├── experiment_options.py
├── scripts/reproduce.py
├── scripts/export_environment.py
├── docs/
└── DATA/Alessandro Tesi/Tesi-main/Secondo_Esperimento/
    ├── Preprocessing/Preprocessing.ipynb
    └── Baselines/
        ├── classifier.py
        ├── TrainingOfUncasedModel/
        │   ├── TrainingExpertSST.ipynb
        │   └── TrainingExpertMNLI.ipynb
        ├── EvaluationExpertSST/EvaluationExpertSST.ipynb
        ├── EvaluationExpertMNLI/EvaluationExpertMNLI.ipynb
        ├── EvaluationExpertMultitaskBert/EvaluationExpertMultitaskBert.ipynb
        ├── EvaluationExpertWeigthAveraging/EvaluationExpertWeigthAveraging.ipynb
        ├── RoutedModel/RoutedModel.ipynb
        └── ExperimentalAnalysis/RouterPolicyAnalysis.ipynb
```

`WeigthAveraging` retains the original directory spelling for compatibility.

## Installation

Use an isolated Python environment on the intended experiment machine. The researcher reports Python **3.13.5** on **Linux x86_64** for the experiment environment; `.python-version` records that Python version. GPU experiments are managed and launched by the researcher; no training or inference was executed during repository preparation.

1. Install Python and create an environment, if a suitable environment is not already available.
2. Install the PyTorch build appropriate for that machine using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Select the CUDA build appropriate for the remote GPU environment.
3. From the repository root, install the remaining runtime and notebook dependencies:

```bash
python -m pip install -r requirements-notebooks.txt
```

For an environment that already provides Jupyter and only needs the experiment libraries, use `requirements.txt` instead. The project does not require torchvision, torchaudio, or TensorFlow.

`requirements.txt` pins runtime packages to the versions reported by the researcher from the experiment environment, including PyTorch **2.11.0** and Transformers **5.8.0**. `requirements-notebooks.txt` adds the recorded notebook tools. `requirements-experiment.txt` preserves the complete selected-package output supplied by the researcher. These files are **not a full transitive lockfile**, and no clean installation has been tested during preparation.

The notebooks use internal PyTorch FLOP-counting APIs. Keep the recorded versions when comparing measurements. The exact CUDA runtime, GPU/driver details, and PyTorch package source remain to be documented; the version string `2.11.0` alone does not identify all of those details.

A metadata-only helper remains available for future environment snapshots, to be run manually by the researcher in the intended environment:

```bash
python scripts/export_environment.py > requirements-experiment.txt
```

It reads installed package metadata without loading models or importing PyTorch. Do not replace the recorded experiment snapshot with output from an unrelated local environment.

## Paths

Each notebook begins with the same lightweight path setup cell. It searches for the repository's `project_paths.py`, starting from the script directory (when available) and current working directory. Start Jupyter from the repository root or any directory beneath it. Keep the helper with the notebooks.

By default, generated data and model files go under the bundled `DATA/Alessandro Tesi/Tesi-main/Secondo_Esperimento` hierarchy. To use an existing directory tree on the experiment machine without copying its artifacts, set the root **before starting the notebook kernel**:

```bash
export TESI_PROJECT_ROOT="/path/to/Tesi-main"
```

The directory must contain `Secondo_Esperimento`. When Jupyter is started outside this repository, also set:

```bash
export TESI_REPOSITORY_ROOT="/path/to/tesi-Milano"
```

These are Linux shell examples. No personal filesystem path is required by the prepared code. Restart the kernel after changing root environment variables; imported path constants are cached by Python.

## Dataset identity

The code uses the historical task and folder name **SST**, but the provided preprocessing notebook downloads `yassiracharki/Amazon_Reviews_Binary_for_Sentiment_Analysis`, not the Stanford Sentiment Treebank. The other source is `ICKD/mnli`. These are the dataset identifiers present in the provided source; the thesis must describe the actual data used.

The preprocessing notebook selects 25,000 examples per task and creates stratified 80/10/10 splits with seed 42. It remaps sentiment labels by subtracting one, and maps MNLI labels `0→1, 1→2, 2→0`. Sources, sampling, mappings, and splits have been preserved, not rerun or independently validated. Dataset revisions are not pinned in the original code, which is an outstanding reproducibility limitation.

## Workflow and prerequisites

Execute notebooks in cell order only in the intended experiment environment. Some notebooks train models; others also perform new inference. This preparation does not launch any of them.

| Stage | Notebook(s) | Required artifacts / behavior |
| --- | --- | --- |
| Preprocessing | `Preprocessing.ipynb` | Downloads the configured datasets and writes parquet splits. |
| Expert training | `TrainingExpertSST.ipynb`, `TrainingExpertMNLI.ipynb` | Requires processed splits; produces the original experts under `TrainingOfUncasedModel/Models`. |
| Expert evaluation | `EvaluationExpertSST.ipynb`, `EvaluationExpertMNLI.ipynb` | Requires experts and processed splits. Cross-task head training is disabled by default (`RUN_CROSS_TASK_TRAINING=False`). |
| Multitask baseline | `EvaluationExpertMultitaskBert.ipynb` | Defaults to `RUN_TRAINING=False`: requires its saved baseline. Set training mode deliberately to generate it from scratch. |
| Weight averaging | `EvaluationExpertWeigthAveraging.ipynb` | Requires original experts. Defaults to `RUN_TRAINING=False`: requires its saved baseline; training mode also trains classification heads. |
| Routed model | `RoutedModel.ipynb` | Requires processed splits and original experts. Launches three training seeds, evaluation, exports, and representation diagnostics. Existing baseline reports are copied when available. |
| Routing analysis | `RouterPolicyAnalysis.ipynb` | Requires completed routed manifests/traces and fixed-expert predictions. New baseline inference is enabled by default; saved checkpoints and a matching GPU environment are needed for uncached measurements. |

## Reproducing the thesis configurations

### Automated notebook launcher

The launcher executes the existing notebooks, one at a time, in fresh kernels. It
does not contain another implementation of preprocessing, models, losses or metrics.
Run it from the Python environment prepared above:

```bash
python scripts/reproduce.py
```

With an interactive terminal it asks which workflow to run and whether to use
four or eight RoutedModel epochs. The main reference uses eight epochs; the shorter
alternative uses four. Both default to `merged=impose` and the existing three
training seeds `(42, 43, 44)`. Expert and baseline training budgets are unchanged.
Before execution it prints the plan and asks for confirmation; `--yes` confirms an
unattended invocation. The launcher does not install packages automatically.

Inspect a plan without executing notebooks, importing ML packages, or creating output folders:

```bash
python scripts/reproduce.py --mode full --epochs 8 --dry-run
```

The three workflows are:

| Mode | What runs |
| --- | --- |
| `full` | Preprocessing; training both experts; expert evaluations including cross-task head training; multitask and weight-averaging training/evaluation; RoutedModel; routing analysis and simple-baseline comparisons. |
| `routed` | RoutedModel and its analysis, using already prepared parquet splits and original expert checkpoints. Existing multitask/weight-averaging reports are reused when available; missing reports remain unavailable in the analysis. |
| `report` | Analysis of the explicitly selected completed run, using saved measurements by default. It does not train models. Missing cached simple-baseline comparisons are reported as unavailable. |

Examples (execution starts only after confirmation, unless `--yes` is supplied):

```bash
# Complete reproduction from source; trains the models.
python scripts/reproduce.py --mode full --epochs 8

# Separate shorter RoutedModel training with the existing data and specialists.
python scripts/reproduce.py --mode routed --epochs 4

# Existing run: replace the timestamp with its actual directory name.
python scripts/reproduce.py --mode report --run-id 20261118_120000_000001

# Also calculate missing simple-baseline measurements for that report on the GPU.
python scripts/reproduce.py --mode report --run-id 20261118_120000_000001 --measure-baselines
```

`full` and `routed` require a CUDA GPU and include new baseline inference during
analysis. Default `report` mode needs the notebook/runtime dependencies but does
not require CUDA. Notebook kernels use the same Python executable as the launcher:
a session-local kernel specification avoids accidentally selecting another installed
Python environment.

Use `--project-root "/path/to/Tesi-main"` to select an artifact tree. Alternatively,
the launcher respects `TESI_PROJECT_ROOT`. Full mode can create a new tree containing
`Secondo_Esperimento`. It can replace existing processed splits, specialist exports
and baseline outputs at the chosen root, so use a separate root for a fresh
reproduction or choose `routed` when reusing the original preparation.

Each new RoutedModel invocation starts a new timestamp group instead of implicitly
resuming the latest unfinished experiment. To resume an incomplete run created with
the new options, select its ID and the same configuration explicitly:

```bash
python scripts/reproduce.py --mode routed --epochs 4 --resume 20261118_120000_000001
```

The stored epochs, merged mode and seeds must match before any completed seeds are
skipped. Existing dataset/expert/checkpoint identity checks remain in the notebook.
Legacy runs without recorded `run_options` are not accepted by launcher resume;
use their original compatible notebook/environment for manual recovery.

The report is linked to the exact run returned by RoutedModel, not whichever run
happens to be newest. All three completed seeds are required. If the notebook
records a simple-baseline measurement failure, the launcher marks the workflow as
failed even if a partial HTML report was written.

Logs, executed notebooks (including partial outputs on cell failure), and an
execution record including source hashes are saved under `.reproduction/<timestamp>/`. Actual model/data
artifacts and reports stay in the configured experiment tree. All are local outputs
excluded from Git. `--output-root` changes the launch-log directory. The log shows
cell start/completion; the executed notebook contains its captured cell output.
After an error the remaining stages are not launched.

The advanced `--merged-mode off` and `--merged-mode auto` options run the other
documented eight-epoch ablations. They are not selected by the simple 4/8 menu.

### Manual notebook execution and parameters

The main thesis reference uses `MERGED_MODE="impose"`, `EPOCHS=8`, training seeds `(42,43,44)`, data seed 42, and classification loss weight 1.15. The published notebook retains this default. The thesis also evaluates the following configurations:

| Experiment | `MERGED_MODE` | `EPOCHS` |
| --- | --- | --- |
| Main reference | `"impose"` | 8 |
| First merged layer disabled | `"off"` | 8 |
| Shorter training | `"impose"` | 4 |
| First merged layer optional | `"auto"` | 8 |

In `impose`, the first merged encoder layer is mandatory. In `off`, it is unavailable. In `auto`, it is available to the router without being mandatory. Shared averaged embeddings and pooler remain in all three modes.

For each experiment:

1. Prepare the same processed splits and original expert checkpoints using the preceding workflow.
2. Set the optional environment variables before starting the kernel, or edit the defaults in the initial RoutedModel options cell. Retain the other reference settings and the three training seeds.
3. Execute from a fresh kernel for a new experiment. Do not resume an existing run after changing its configuration.
4. Run `RouterPolicyAnalysis.ipynb` on the completed run, checking that its selected manifest identifies the intended configuration and seeds.

The published implementation derives the temperature and entropy schedules from `EPOCHS`. Keep `TAU_ANNEALING_EPOCHS=EPOCHS`, `ENTROPY_RAMP_END_FRACTION=1.0`, and `MNLI_COST_RAMP_END_FRACTION=5.0/8.0`. Temperature and entropy complete their progression within the selected training duration; the MNLI cost schedule reaches its endpoint at epoch 5 for eight epochs and epoch 3 for four epochs. The four-epoch experiment is a separate training configuration with adapted schedules, rather than the epoch-four checkpoint of an eight-epoch run.

| Environment variable | Manual default | Purpose |
| --- | --- | --- |
| `TESI_EPOCHS` | `8` | RoutedModel training duration. The launcher offers 4 or 8. |
| `TESI_MERGED_MODE` | `impose` | First-layer encoder choice: `impose`, `off`, `auto`. |
| `TESI_RUN_TRAINING` | `false` | Train the multitask/weight-averaging baseline rather than load its saved checkpoint. |
| `TESI_CROSS_TASK_TRAINING` | `false` | Train the additional cross-task heads in the expert evaluation notebooks. |
| `TESI_MEASURE_BASELINES` | `true` | Allow new simple-baseline inference in routing analysis. |
| `TESI_ANALYSIS_RUN` | unset | Analyze this timestamp group instead of the latest completed group. |
| `TESI_ROUTED_ACTION` | `auto` | `auto` preserves manual notebook selection; `new` starts a new group; `resume` selects `TESI_ROUTED_RUN_ID`. |

For example, before starting Jupyter:

```bash
export TESI_EPOCHS=4
export TESI_MERGED_MODE=impose
jupyter lab
```

The launcher sets these options itself; no notebook editing is required. Without
overrides the manual notebook defaults remain unchanged. Changes to environment
variables require a fresh kernel.

These settings describe how to configure the published implementation. They do not establish that its source is identical to every historical VM execution.

### Baseline latency evaluation

The four evaluation notebooks include inference timing. To evaluate existing baselines, keep `RUN_CROSS_TASK_TRAINING=False` in the SST and MNLI expert notebooks and `RUN_TRAINING=False` in the weight-averaging and multitask notebooks. Their saved model checkpoints and processed test splits must already be available.

The thesis baseline timings use an NVIDIA A2, float32, batch size 32, ten warm-up batches, and three synchronised repetitions. Timing covers the model forward pass, including the classifier, with tokenised inputs on the device; tokenisation, data loading and host-to-device transfer are excluded. Latency is hardware-dependent, so use the same environment and protocol for comparisons. These evaluations do not retrain the baselines.

## Publication contents

Only source notebooks, the shared classifier, path/option helpers, the launcher, offline tests, dependency specifications, and documentation are included. Datasets, weights, model checkpoints, predictions, cached downloads, notebook outputs, and experiment results are excluded.

Notebook outputs and execution counts are cleared. Launcher integration changes only configuration and run/report orchestration; model, loss, schedules, metric and FLOP-counting implementations are retained. The original preparation checks are documented in the [preparation notes](docs/PREPARATION.md).

Offline orchestration tests can be run without loading models or datasets:

```bash
python -m unittest discover -s tests -v
```

They cover option validation, workflow order, exact run selection, incompatible
resume rejection, failure handling, saved partial notebook output and the unchanged
4/8-epoch schedule endpoints. They use synthetic artifacts and mocked notebook
execution; no end-to-end GPU reproduction or numerical equivalence is claimed.

No software license has been selected yet. A license should be chosen by the author before making reuse permissions explicit.

