# Preparation validation

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
