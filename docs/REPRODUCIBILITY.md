# Reproducing FedLN experiments

## Baseline and validation boundary

The original implementation is available at the `federatedml-baseline` tag; see [provenance](../PROVENANCE.md) for its source and history. The maintenance changes repair entry points, defer EuroSAT configuration, isolate disposable cache directories, and fail promptly when GPU discovery finds no NVIDIA hardware. They do not change model architectures, optimization, random seeds, partitions, noise generation, KNN decisions, or aggregation formulas.

Dependency-light tests exercise launch dispatch/help, missing data/GPU behavior, cache ownership, centralized evaluation plumbing, and NumPy data invariants. TensorFlow training, Flower/Ray workers, GPU execution, and the paper's quantitative results have not been validated in this maintenance pass. Tests that mock training do not establish distributed compatibility.

## Historical environment

`src/requirements.txt` retains the original pins, including TensorFlow 2.10.1, Keras 2.10.0, TensorFlow Addons 0.19.0, Flower 1.4.0, Ray 2.2.0, NumPy 1.24.1, and scikit-learn 1.2.0. It is an archival environment specification, not a currently verified install lock.

Published [TensorFlow 2.10.1](https://pypi.org/project/tensorflow/2.10.1/#files) and [Ray 2.2.0](https://pypi.org/project/ray/2.2.0/#files) wheels include CPython 3.10. TensorFlow 2.10.1 has no native macOS arm64 wheel under that package name. Python 3.10 on Linux with NVIDIA hardware is therefore a candidate starting point, not a tested environment guarantee. Do not interpret successful Python 3.14 syntax or utility tests as support for training with that interpreter.

There is a retained dependency discrepancy: [Flower 1.4.0 package metadata](https://pypi.org/pypi/flwr/1.4.0/json) declares `ray[default]>=2.3.0,<3.0.0` for its simulation extra, while this repository directly pins `ray==2.2.0`. A bare `flwr` installation and `pip check` may not flag the optional-extra constraint. Resolve and record this in a separate environment-validation effort before claiming that simulations reproduce the paper; no dependency version has been silently changed here.

Most experiment scripts select an NVIDIA GPU through GPUtil and wait while detected GPUs are busy. An empty GPU inventory now raises an actionable error. The historical standard FedAvg and centralized scripts still select GPU IDs `0` and `1`, respectively. CPU and Apple GPU training have not been qualified. The scripts do not provide portable distributed resource scheduling.

Before a scientific run, record the Git revision, command, Python/package versions, GPU/driver/runtime versions, dataset/embedding hashes, and output logs. Keep any environment adaptations distinct from the original pins.

## Data contracts

CIFAR-10 images and labels come from `tf.keras.datasets.cifar10.load_data()`. The shipped `features/cifar10.npy` has shape `(50000, 512)` and float16 values; rows are indexed using training-set shard indices.

EuroSAT requires an NPZ archive with `images` and `labels` arrays at `${TFDS_DATA_DIR}/eurosat/eurosat.npz`, defaulting to `src/data/eurosat/eurosat.npz`. The implementation takes rows `[:21600]` as training data and `[21600:]` as test data. Its shipped embedding array has shape `(27000, 512)`, float16. The image archive, original archive ordering procedure, and embedding-generation script are not present. A newly downloaded or shuffled EuroSAT dataset cannot be assumed to align with these embeddings. Preserve and verify the original order before running NNC or AKD; do not substitute a newly invented split and label it a reproduction.

Both embedding arrays were read successfully and checked for finite values during maintenance. Their values, rows, and file contents were not changed. No trained final model checkpoint or complete audio experiment pipeline is shipped. Audio helper functions do not make the paper's audio experiments runnable from the provided entry points.

## Research-sensitive behaviors retained

The following are confirmed source observations, not proposed method changes. Line references describe this maintenance revision; the named function is the stable locator.

| Location | Behavior and implication |
| --- | --- |
| [`src/utils/flwr_client.py:11`](../src/utils/flwr_client.py#L11), `_Client.__init__` | Passes `seed=kwargs['batch_size']` to the loader. The configured CLI seed is therefore not used on this path. |
| [`src/fedln/fedln__akd.py:91`](../src/fedln/fedln__akd.py#L91) and line 103, `AKDClient.fit` | Reloads data without forwarding CLI noise, seed, or batch settings. Lines 106–109 also retain 512 embedding dimensions and batch size 128 on the distillation path. |
| [`src/fedln/fedln__nnc.py:75`](../src/fedln/fedln__nnc.py#L75), `NNCClient.__init__` | Repeated loader calls omit the CLI seed, retaining loader defaults. |
| [`src/utils/knn_relabel.py:9`](../src/utils/knn_relabel.py#L9) and line 21 | Replaces fitted `knn.classes_` with all class IDs. A one-neighbour fixture with labels `[1,1,3,3]` and separated features returned `[0,0,1,1]` in the audit environment (NumPy 2.4.1, scikit-learn 1.8.0); this was not a test of the pinned historical scikit-learn version. Shards missing classes need scientific review. |
| [`src/utils/energy_noise_estimation.py:23`](../src/utils/energy_noise_estimation.py#L23), `compute_noise_estimation` | Uses `min(min_effect * n, n - noisy)` for affected client weights. No reinterpretation as a lower bound or change from `min` to `max` has been made. |
| [`src/utils/shard.py:3`](../src/utils/shard.py#L3), `shard_data` | Resets NumPy's global seed and randomly assigns samples to shards; this is not a guarantee of equal shard sizes or a configurable non-IID benchmark split. |
| [`src/utils/distiller.py:13`](../src/utils/distiller.py#L13), `Distiller.compile/train_step` | Retains MAE feature distillation with `alpha=10`; the stored temperature is not applied in the shown loss calculation. |

Changing these behaviors may change experimental outcomes. Such fixes need an explicit methodological decision and comparisons against the retained implementation. In particular, the passing maintenance tests do not certify the KNN class-mapping or seed behavior as correct.

## Cache lifecycle

NNC, AKD, NA-FedAvg, and confident learning allocate a unique child directory under `--temp_dir` for each simulation. They clean only that directory on normal exit or Python exception and restore the original parent argument. A forced process termination can leave a child directory behind. Existing parent contents are not reused as experiment caches and are not deleted. Distributed worker lifecycle under this change still needs a real Flower/Ray smoke test.

## Publication metadata and reported results

The [ACM record](https://doi.org/10.1145/3626242) identifies the final article as TIST **15(2), Article 22, pages 1–26 (2024)**, published 22 February 2024; its accepted manuscript was online 9 October 2023. This explains the 2023 citation retained in the historical repository.

The original project page and README had different numeric summaries from the final journal abstract. The refreshed overview omits those conflicting numbers; retained figures are identified as historical research figures. Consult the final paper for the precise protocol, comparator, dataset, and metric. No figure or reported result was regenerated during maintenance.
