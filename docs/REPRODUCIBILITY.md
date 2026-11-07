# Open reproducibility items

These items must not be presented as established experimental facts.

| Item | Current evidence | Next step |
| --- | --- | --- |
| Package versions | Researcher supplied Python 3.13.5, Linux x86_64 and 16 package versions; requirements now use these pins. | Recover the full transitive environment snapshot and PyTorch package source if exact installation reproduction is required. |
| Data identity | Historical `SST` variable names refer to an Amazon Reviews source in preprocessing. | Use the actual dataset identity in the thesis and verify it against original experiment manifests. |
| Dataset revisions | Hugging Face IDs are provided without revision pins. | Recover dataset versions/fingerprints and pin revisions where possible. |
| Configuration provenance | Supplied remote notebook uses merged `impose`; author confirmed keeping `impose` for the current setup. | Match each reported result to its manifest, code/configuration, and seed; revisit mode only by explicit decision. |
| Baseline checkpoints | Multitask and weight-averaging notebooks default to loading saved checkpoints. | Generate or supply the required checkpoints on the experiment machine; they are not part of GitHub. |
| Measurements | Routing analysis defaults to permitting new baseline inference. | Researcher chooses whether to use cache-only reporting or new measurements. |
| Hardware and numerical reproducibility | No new runtime measurements made during preparation. | Record GPU, driver, CUDA, numerical precision, library versions, and FLOP coverage from original runs. |
| Data/model distribution | Only source files are included. | Document any permitted artifact retrieval separately, respecting source terms. |

The absence of a runtime test is intentional: repository preparation never executes the experimental pipeline.
