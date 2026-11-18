# Preparation validation

The first section records the initial 7 November preparation. The launcher checks
are recorded separately below. Usage and interpretation of its modes are in the
[reproduction guide](REPRODUCIBILITY.md).

Checked on 7 November 2026:

- Nine notebooks parsed as JSON; all 117 original code cells parsed as Python AST.
- Original and prepared ASTs match after allowing only the documented project-path replacements and routing-analysis root assignment.
- All notebook code outputs are empty and execution counts are null.
- The classifier is byte-for-byte unchanged.
- Original source hashes match the pre-preparation manifest.
- The standard-library path helper and initial setup cell resolve the project from the repository root, a nested notebook directory, and an outside directory with explicit repository/project root overrides.
- No dataset, checkpoint, trace JSONL, or experiment log is present in the package.
- Source scanning found no matches for the checked password/private-key/GitHub-token patterns or the known remote access identifiers. Pattern scanning is not a security audit.

Full validation with the optional `nbformat` library was not available in the preparation runtime. JSON/cell structure and syntax were checked directly instead. No experimental cell, model import, dataset download, dependency installation, training, or inference was executed. Numerical equivalence and runtime compatibility remain untested.

Dependency pins were subsequently compared with all 16 package versions supplied by the researcher. No package installation or dependency-resolution test was performed. Merged mode remains `impose`; no notebook source changed during this environment update.


## Launcher validation — 18 November 2026

The launcher integration was checked separately from the original publication
preparation:

- Eighteen standard-library tests passed, using synthetic artifacts and mocked
  notebook execution. They exercise workflow ordering, option isolation, new/run
  resume selection, rejection of incompatible budgets, exact report-group
  selection, three-seed completeness, failure propagation and preservation of
  partial notebook output.
- Tests also exercise the actual notebook selection/configuration functions and
  verify the original four/eight-epoch schedule endpoints.
- All 125 code cells in the nine current notebooks parsed. A source/AST comparison
  against the pre-launcher version allowed the intended configuration and
  run/report selection changes and confirmed preservation of model, loss,
  optimizer, metric, FLOP-counting and schedule implementations.
- Notebook outputs, execution counts and metadata were preserved. The three
  preprocessing/expert-training notebooks were not changed by launcher integration.
- Dry-run behavior was checked without launching notebooks or creating output trees.

Run the offline suite from the repository root with:

```bash
python -m unittest discover -s tests -v
```

These checks do not constitute an end-to-end GPU reproduction. No experiment
training, new inference, dataset download or clean dependency installation was
performed for launcher validation.

A successful `full` launch, a successful `routed` launch and a successful
cached `report` launch have different scopes. In particular, a completed report
can explicitly omit comparisons for which compatible saved results are missing.
See [workflow meanings](REPRODUCIBILITY.md#choose-by-the-result-you-need) and
[output interpretation](REPRODUCIBILITY.md#files-to-inspect-after-a-launch) before
treating a generated HTML document as a complete set of thesis comparisons.
