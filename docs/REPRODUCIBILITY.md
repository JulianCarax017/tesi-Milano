# Reproducing an experiment: choosing the right workflow

This guide explains what the launcher does, what it reuses, and what its output
can establish. Installation and manual notebook settings are in the
[main README](../README.md#installation).

The repository contains source code, not the original data splits, model weights
or experiment results. A fresh clone therefore does **not** satisfy the prerequisites
for every mode.

## Choose by the result you need

| Your situation | Mode | Meaning |
| --- | --- | --- |
| You have a fresh clone and want to recreate the experimental procedure. | `full` | Prepare data, train specialists and comparison models, then train/evaluate RoutedModel and build its report. |
| You already have the prepared data and original specialists, and want a new RoutedModel experiment. | `routed` | Reuse that preparation; train a new router and two new heads for each of the three seeds, then evaluate and analyse that run. |
| You have a completed three-seed run and want to rebuild its report. | `report` | Read that run's saved artifacts. By default, use existing measurements without new training or inference. |

**Use `full` for a fresh reproduction from source. Use `routed` to extend
an existing preparation, such as comparing the four- and eight-epoch budgets.
Use `report` to inspect existing results, not to generate a newly trained model.**

A successfully completed command means that its requested workflow finished.
It does not mean that every comparison in the thesis is present, or that newly
trained models must reproduce exactly the historical numbers.

## Starting the launcher

From the repository root, in the environment described in the README:

```bash
python scripts/reproduce.py
```

The interactive menu asks for the workflow and, for new training, four or eight
RoutedModel epochs. For a report it asks for a completed run ID and whether to
allow new GPU measurements of the simple baselines. It then prints the plan and
asks for confirmation.

The same choices are available as command-line arguments. The examples below use
an illustrative absolute artifact directory and run ID: replace them with your
actual paths and ID.

```bash
PROJECT_ROOT="/path/to/Tesi-main"
RUN_ID="20261118_120000_000001"
```

The project root is the directory **containing** `Secondo_Esperimento`, not the
repository root and not the `Baselines` directory. The source notebooks always
come from the checkout. Artifact paths come from `--project-root`, then
`TESI_PROJECT_ROOT` if set, otherwise the bundled
`DATA/Alessandro Tesi/Tesi-main` tree.

To inspect a plan without starting work:

```bash
python scripts/reproduce.py --mode full --epochs 8 --project-root "$PROJECT_ROOT" --dry-run
```

A dry run checks the notebook paths and the prerequisites applicable to the
selected mode. It does not create folders, install dependencies, import ML
packages, check CUDA availability, download data or execute notebooks. A successful
dry run is therefore not a GPU/runtime compatibility test.

## `full`: prepare and train the complete workflow

```bash
python scripts/reproduce.py --mode full --epochs 8 --project-root "$PROJECT_ROOT"
```

Use this when starting from source without the prepared data and specialists.
The sequence is:

1. `Preprocessing.ipynb`: download the configured datasets and create parquet splits.
2. `TrainingExpertSST.ipynb` and `TrainingExpertMNLI.ipynb`: fine-tune the two specialists.
3. `EvaluationExpertSST.ipynb` and `EvaluationExpertMNLI.ipynb`: evaluate the specialists and train/evaluate their additional cross-task heads.
4. `EvaluationExpertMultitaskBert.ipynb`: train and evaluate the multitask baseline.
5. `EvaluationExpertWeigthAveraging.ipynb`: construct the fixed merge and train/evaluate its heads.
6. `RoutedModel.ipynb`: train/evaluate the selected routed configuration across seeds 42, 43 and 44.
7. `RouterPolicyAnalysis.ipynb`: build the report, allowing simple-baseline inference and reuse of compatible cached measurements.

The launcher enables `RUN_TRAINING` for the two trained baselines and
`RUN_CROSS_TASK_TRAINING` for expert evaluation. Their standalone notebook
defaults are false; simply executing every unconfigured notebook would not be the
same workflow.

Full mode requires CUDA and can create a new artifact tree. It can also replace
processed data, specialist exports and baseline outputs already at the selected
root. Choose a separate empty root when you want an independent reproduction.
Re-running full mode starts the preparation sequence again; it is not a resume
command for the whole pipeline.

**`--epochs 4` or `--epochs 8` changes only RoutedModel.** It does not
change the three-epoch specialist or multitask budgets, the weight-averaging head
training budget, or cross-task head training.

## `routed`: a new integration experiment with existing preparation

```bash
python scripts/reproduce.py --mode routed --epochs 4 --project-root "$PROJECT_ROOT"
```

This runs **RoutedModel followed by routing analysis**. It does not mean
"inference only": it trains a new router and new classification heads for each of
the three seeds, keeping the original specialists frozen.

It does not rerun preprocessing, retrain the specialists, train multitask or
weight averaging, or recreate their missing evaluation reports. Starting this
mode on a fresh clone will fail because the prepared data and experts are absent.

### What must already exist

Relative to `$PROJECT_ROOT/Secondo_Esperimento`:

| Artifact | Purpose |
| --- | --- |
| `Preprocessing/Dataset/ProcessedDB/SST/` and `.../MNLI/` | Each needs `X_train.parquet`, `y_train.parquet`, `X_val.parquet`, `y_val.parquet`, `X_test.parquet` and `y_test.parquet`. |
| `Baselines/TrainingOfUncasedModel/Models/sst_bert_base_uncased/` | The selected sentiment expert, including its backbone/config, classifier and saved preparation artifacts. |
| `Baselines/TrainingOfUncasedModel/Models/mnli_bert_base_uncased/` | The selected inference expert in the corresponding format. |

The launcher's preflight checks parquet files, classifier files, backbone configs
and weight files. These existence checks do not establish that arbitrary files
are the original splits or experts. Keep the preparation together and retain the
notebooks' additional input, checkpoint and protocol checks.

### Which comparisons will be available

RoutedModel evaluates its fixed task experts alongside the new routed run.
Analysis can additionally read existing baseline reports. Their usual paths,
relative to `$PROJECT_ROOT/Secondo_Esperimento/Baselines`, are:

| Report | Source |
| --- | --- |
| `EvaluationExpertSST/Results/expert_a_metrics.json` | Expert evaluation notebook. |
| `EvaluationExpertMNLI/Results/expert_b_metrics.json` | Expert evaluation notebook. |
| `EvaluationExpertMultitaskBert/Results/multitask_bert_test_results.json` | Multitask evaluation notebook. |
| `EvaluationExpertWeigthAveraging/Results/weight_averaging_test_results.json` | Weight-averaging evaluation notebook. |

The analysis prefers the matching fixed-expert measurements stored with the routed
run. For other baseline reports it checks the current report directory and then
the run's saved `baseline_reports` copies where available.

**Missing multitask or weight-averaging reports are not generated automatically
in this mode.** Their comparisons can remain unavailable while the routed run and
HTML report are otherwise completed. Reused reports must concern the same data
preparation and an appropriate evaluation protocol; a JSON file being present is
not sufficient evidence that it is the correct comparison.

The analysis also allows new measurements of static layer removal, random routing
and early exit, using the fixed specialists and the current routed validation
budget. These are the **simple baselines**, distinct from the trained multitask
and weight-averaging models.

### Comparing eight and four epochs correctly

Use the same processed splits and fixed specialists for both configurations.
For example, after one completed full eight-epoch workflow:

```bash
python scripts/reproduce.py --mode routed --epochs 4 --project-root "$PROJECT_ROOT"
```

This starts a separate four-epoch training group. It does not evaluate epoch four
of an eight-epoch run or shorten an existing run. Both budgets use
`merged=impose` by default and three training seeds. Temperature and entropy
reach their endpoints within the selected duration; the MNLI cost ramp ends at
epoch 3 for four epochs and epoch 5 for eight.

Running full mode a second time would also regenerate the preparation and
specialists. That is a new complete reproduction, rather than a controlled
training-budget comparison with the same fixed preparation.

## `report`: rebuild analysis from a selected completed run

```bash
python scripts/reproduce.py --mode report --run-id "$RUN_ID" --project-root "$PROJECT_ROOT"
```

The run ID is the timestamp directory under
`Secondo_Esperimento/Baselines/RoutedModel/Results/`. Supply the group ID,
not an individual `..._seed_1` directory. The launcher requires all three
completed seed manifests, the expected seed identities and validation/test split
records. The analysis also needs the run's referenced traces and saved artifacts.

By default, report mode reads saved results and cached simple-baseline
measurements. It does not retrain models or launch new inference and does not
require CUDA. It still requires the Python/notebook dependencies and files read by
the analysis. It rewrites `routing_policy_report.html` in the selected group;
it does not create another trained experiment.

If a compatible simple-baseline comparison has not been saved, a report can
display that comparison as unavailable. To allow its calculation on the GPU:

```bash
python scripts/reproduce.py --mode report --run-id "$RUN_ID" --project-root "$PROJECT_ROOT" --measure-baselines
```

**`--measure-baselines` means static removal, random routing and early exit.
It does not train multitask or weight averaging, recreate their reports, or train
cross-task heads.** It enables missing simple-baseline inference and can reuse
compatible cached measurements. The fixed expert checkpoints used by the selected
run and a matching runtime/protocol must remain available.

In report mode, the recorded run determines the training configuration. Passing
`--epochs` does not turn an existing eight-epoch run into a four-epoch one.

## Run identity, new training and resume

For both full and routed workflows the launcher passes the exact completed
RoutedModel group to analysis. It does not select a different group merely because
another run was created later.

New training starts a new timestamp group. To continue an incomplete routed group
created with the current options:

```bash
python scripts/reproduce.py --mode routed --epochs 4 --project-root "$PROJECT_ROOT" --resume "$RUN_ID"
```

This requires the same recorded epochs, merged mode and seeds. The existing
notebook checkpoint checks also validate configuration, datasets and experts.
Completed seeds are skipped; unfinished work follows the notebook's epoch-checkpoint
recovery. It is not a request to restart the experiment under new hyperparameters.

Resume applies only to RoutedModel. It is rejected for completed groups, for a
different budget/configuration and for legacy groups without recorded
`run_options`. Use the original compatible notebook/environment for legacy
manual recovery.

If RoutedModel completed but report generation failed, do not retrain it. Run
`--mode report --run-id ...`, adding `--measure-baselines` if the missing
work is simple-baseline inference.

## Options and what they change

| Option | Effect |
| --- | --- |
| `--mode full|routed|report` | Select which workflow stages execute. |
| `--epochs 4|8` | Select the budget for new/resumed RoutedModel training only. |
| `--merged-mode impose|off|auto` | Select the first-layer encoder option for RoutedModel; default `impose`. |
| `--project-root PATH` | Select the data/model/result tree used by every stage. |
| `--run-id ID` | Select a completed group for report mode. |
| `--resume ID` | Resume a compatible incomplete group in routed mode. |
| `--measure-baselines` | In report mode, allow new simple-baseline inference on the GPU. Full/routed analysis already enables it. |
| `--output-root PATH` | Change the launch-log/executed-notebook directory; does not relocate model/data artifacts. |
| `--dry-run` | Check/print a plan without executing it. |
| `--yes` | Skip the final execution confirmation, for unattended use. |

For the thesis's other eight-epoch ablations, add `--epochs 8 --merged-mode off`
or `--epochs 8 --merged-mode auto`. In all modes the averaged embeddings
and pooler remain shared; `off` disables the first merged encoder layer, not
all parameter averaging.

When using the launcher, select options through its arguments/menu. It sets its
own notebook environment and clears conflicting inherited launcher options.
The README's `TESI_*` variable table is mainly for direct notebook use; for
example, exporting `TESI_EPOCHS=4` does not replace `--epochs 4` in a
launcher command. `TESI_PROJECT_ROOT` remains supported as the artifact-root
fallback.

## Files to inspect after a launch

There are two independent locations:

| Location | Contents |
| --- | --- |
| `.reproduction/<launch timestamp>/` in the checkout by default | Stage logs, executed notebooks, session-local kernel specification and `launcher.json` with configuration, stage status, source hashes and the routed run ID. |
| The selected project tree | Processed datasets, expert/baseline exports, RoutedModel checkpoints, prediction/routing artifacts and reports. |

A launch timestamp identifies the orchestration session; a RoutedModel run ID
identifies the actual three-seed experiment. They need not be equal.

On an error, inspect the failing stage's `.log` and executed notebook, then
`launcher.json`. Partial notebook output is retained for cell-execution
failures, and subsequent stages are not started. If the analysis caught a
simple-baseline measurement error and still produced partial HTML, the launcher
marks the requested measurement workflow as failed.

Generated artifacts and the default `.reproduction/` directory are excluded
from Git. Keep a custom `--output-root` outside the checkout or add its ignore
rule if needed.

Before interpreting a report, check:

- The run ID, actual epochs, merged mode and all three seeds match the intended experiment.
- The data splits and original specialists are the intended shared preparation.
- The requested baseline comparisons are present, rather than marked unavailable.
- Reused baseline reports belong to the same experimental setting.
- A failed or interrupted launch is not being treated as a completed measurement.

## Scope of reproducibility

The launcher automates the published procedure and removes manual configuration
steps. It does not supply the original historical checkpoints, pin dataset
revisions that the original code did not pin, or guarantee bitwise-identical
training and timing on another environment.

The implementation has offline tests for orchestration, parameter handling,
run/report identity and error paths. Full GPU reproduction has not been executed
as part of launcher development. See [validation](VALIDATION.md) for the checks
actually performed.
