# Provenance

This repository is an independent Nousphera copy of [FedLN](https://github.com/FederatedML/FedLN). The original FederatedML repository remains in place.

- Source commit: [`01e4afe26fba92b8ed0fd7e2b16b044fba0d2588`](https://github.com/FederatedML/FedLN/commit/01e4afe26fba92b8ed0fd7e2b16b044fba0d2588)
- Source branch: `main`
- Baseline tag in this repository: `federatedml-baseline`
- Original Git history and author attribution are retained.

Maintenance changes after this baseline improve documentation, setup, and validation. They do not constitute a new reproduction of the published experiments. For the original code and configuration, inspect the baseline tag.

GitHub issues, pull requests, stars, releases, and Pages configuration are not part of Git history and have not been transferred. Existing discussions remain at the source repository.

## Maintained execution release (2026-10-15)

The new `run.py` uses the original clients/models/strategies, CPU execution and
stock Flower serialization. Numerical bug fixes, AKD projection ownership and
remaining paper/source discrepancies are itemized in `docs/REPRODUCIBILITY.md`.
The original requirements remain in `src/requirements.txt`; maintained pins are
in the root requirements file. No private history, dataset or trained result was
added. Synthetic features are generated fixtures, not recovered CLIP artifacts.
