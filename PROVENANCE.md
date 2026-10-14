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

Recovered SpeechCommands loader: author HomeLab archive `FedLN/data/spcm.py`,
SHA-256 `73bfa53c0bba349a9596c4bf0695bf75cbea12c833abb83eccab8a9b8a3b6804`. Restored and repaired interface is
documented in `docs/RECOVERED_DATA.md`; no dataset or model weights copied.

## Dataset and teacher completion

`prepare_features.py`, `extract_features.py`, `prepare_audio.py`, and
`prepare_human_labels.py` are maintained implementations, not recovered historical
extractors. They use public model/dataset APIs and record hashes rather than
redistributing weights or datasets. Public source revisions and explicit data
recipes are documented in `docs/PAPER_DATA.md`. The global-logit AKD option is
a maintained implementation of the final paper's printed Eq. 9.
