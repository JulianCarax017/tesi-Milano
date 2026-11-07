# Publication preparation

Prepared on 7 November 2026 from the ten selected source files supplied by the researcher. Original files are preserved outside the publication folder.

Changes:

- Removed saved outputs, execution counts, widget state, and machine-specific kernel details from nine notebooks.
- Retained the `DATA/Alessandro Tesi/Tesi-main` directory hierarchy.
- Added `project_paths.py` and a shared setup cell to resolve that hierarchy from any checkout location, with optional root overrides.
- Replaced 27 hard-coded project paths with expressions relative to the shared project root.
- Pointed routing-analysis `BASELINES_OVERRIDE` at the configured baseline root; the existing prerequisite checks remain active.
- Copied `classifier.py` without changing its contents.
- Added runtime/notebook dependency lists, documentation, ignore rules, and a metadata-only environment export helper.

Training settings, sampling, label remapping, checkpoint-selection logic, losses, FLOP-counting rules, baseline flags, and merged mode are preserved from the supplied snapshot. `RoutedModel` still uses `impose`; it is not the separate `off` ablation file.

Verification uses JSON parsing, Python AST parsing without importing or executing notebook code, notebook-schema validation when available, source comparison, and path-helper checks without ML imports. All source originals are checked against their pre-preparation SHA256 values. No package installation, dataset download, training, inference, or remote access is performed.

The extra setup cell shifts original notebook cell indices by one. No experimental outputs are included in this source-only package. A local preparation manifest records source hashes outside the upload folder.

## Environment and configuration update

On 7 November 2026, the researcher supplied Python 3.13.5, Linux x86_64, and versions of 16 installed packages. Dependency files were updated to exact reported pins, the selected environment snapshot was retained, and `.python-version` was added. The earlier inferred Transformers `<5` constraint was removed because it excluded the reported Transformers 5.8.0 environment. No installation compatibility claim is made without a fresh environment check. The author confirmed keeping merged mode `impose` for the current setup.
