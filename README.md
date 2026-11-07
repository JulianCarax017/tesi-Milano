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

The supplied remote snapshot has `MERGED_MODE="impose"`, seeds `(42,43,44)`, data seed 42, eight training epochs, and classification loss weight 1.15. The publication copy preserves those settings. The author has selected `impose` for the current publication/ablation setup; changing it remains a later decision. The earlier separate local file configured with `off` is not the reference configuration for this package. Choose and document a configuration before an experiment; a run in progress is subject to resume-contract checks.

## Publication contents

Only source notebooks, the shared classifier, path configuration, dependency specifications, and documentation are included. Datasets, weights, model checkpoints, predictions, cached downloads, notebook outputs, and experiment results are excluded. The original local files remain unchanged.

Notebook outputs and execution counts were cleared; algorithm code and hyperparameters were retained apart from path portability and routing-analysis root configuration. Preparation checks cover syntax, notebook structure, paths, and source-change boundaries; they do not establish runtime correctness or numerical equivalence. See [preparation notes](docs/PREPARATION.md) and [open reproducibility items](docs/REPRODUCIBILITY.md).

No software license has been selected yet. A license should be chosen by the author before making reuse permissions explicit.
