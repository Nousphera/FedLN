# Recovered SpeechCommands path

`src/data/spcm.py` is adapted from the author's recovered HomeLab archive
`FedLN/data/spcm.py`, inspected 2026-10-15; source SHA-256 `73bfa53c0bba349a9596c4bf0695bf75cbea12c833abb83eccab8a9b8a3b6804`.
That archive has no Git metadata. Its original loader was omitted from the
public release and did not run: return unpacking, record-path/ID handling,
label lookup and embedding IDs were inconsistent. The maintained adapter fixes
those issues and keeps the original 12-class CNN preprocessing. It uses whole
prepared-record shards, with deterministic balanced assignment of shard files
to clients. This is not the paper's controlled 25% sample-count variation.

## Prepared input and invocation

Set `TFDS_DATA_DIR` to a directory containing `speech_commands/`. Each split
uses files named `speech_commands-train.tfrecord-00000-of-00100` (and `test`),
compressed with ZLIB. Each serialized `tf.train.Example` has:

| Feature | Type/content |
| --- | --- |
| `audio` | bytes from `tf.io.serialize_tensor` of a 1D int64 PCM waveform |
| `shape` | int64 waveform length |
| `label` | int64 class ID in 0–11 |
| `file_id` | int64 nonnegative row index in the split's embedding array |
| `speaker_id` | bytes; retained by the format, unused by this loader |

Each record file has an adjacent `.npz` sidecar containing `file_id` and
`label` arrays in exactly its record order. IDs must be unique within a split.
The loader verifies these IDs and labels against actual records in their exact order.
There must be at least as many train record shards as clients. Neither corpus
record conversion nor the historical embedding extractor was recovered; maintained
`prepare_audio.py` and `extract_features.py` now provide these steps with explicit
TFDS/teacher versions and recorded hashes. See [PAPER_DATA.md](PAPER_DATA.md).
The generated examples in `tests/test_audio_data.py::write_records` demonstrate
the exact schema without requiring any dataset download.

```bash
export TFDS_DATA_DIR=/path/to/prepared/root
python run.py --method nnc --dataset_name spcm --model_name cnn \
  --embeddings_dir /path/to/features --knn_neighbors 100 \
  --num_clients 30 --num_rounds 200 --batch_size 128 --train_epochs 1 \
  --output runs/spcm-nnc
```

NNC/AKD read `/path/to/features/spcm.npy`: row `file_id` must contain the
corresponding train waveform's teacher embedding. Features may not be shuffled
independently. AKD's `--embeddings_dims` must equal the array's column count.
The final paper uses TRILLsson v3; generated test features are not substitutes
for this teacher. `fedavg` and `na_fedavg` need no embedding file. Choose noise
levels appropriate to the actual corpus; the historical synthetic noise helper
checks that the achieved fraction is close to the requested fraction.

Training preserves the original 15,690-sample random window, L2 waveform
normalization, 64 mel bins, and log-magnitude spectrogram. Evaluation retains
full padded waveforms and batches one example, allowing variable duration.
The exact historical unknown/silence recipe, splits and teacher preprocessing
were not recovered. The maintained exporter explicitly uses TFDS 4.9.4
Speech Commands 0.0.3 and its twelve-label recipe; the extractor obtains the
official TRILLsson 3 SavedModel. These executable paths are not benchmark
reproductions.

## Checks

Three focused tests passed: generated ZLIB records/sidecars preserve class IDs,
counts and corrected labels; NNC and AKD each complete two actual CPU rounds
with two clients and 16 generated waveforms, stock Flower messages and changed
parameters. These are execution checks, not benchmark accuracy measurements.
The AKD audio fixture has clean labels; noisy-client distillation is covered
separately by the existing method tests.
