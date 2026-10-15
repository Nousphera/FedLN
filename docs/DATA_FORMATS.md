# Speech Commands record format

The audio loader reads prepared ZLIB TFRecords from
`${TFDS_DATA_DIR}/speech_commands/`. The [preparation guide](DATA_SETUP.md)
creates these files from TFDS Speech Commands 0.0.3; the
[teacher guide](PAPER_DATA.md) extracts aligned TRILLsson features.

## Records and identities

A split contains files such as
`speech_commands-train.tfrecord-00000-of-00100`. Each `tf.train.Example` has:

| Feature | Type and content |
| --- | --- |
| `audio` | Bytes from `tf.io.serialize_tensor` of a 1D int64 PCM waveform |
| `shape` | int64 waveform length |
| `label` | int64 class ID in 0–11 |
| `file_id` | int64 nonnegative row index in the split's embedding array |
| `speaker_id` | Bytes; retained in the schema and unused by the loader |

Each record file has an adjacent `.npz` sidecar with `file_id` and `label`
arrays in record order. IDs must be unique within a split. The loader verifies
record IDs and labels against these sidecars. Provide at least as many training
shards as clients; whole shards are assigned deterministically across clients.

```bash
export TFDS_DATA_DIR=/path/to/prepared/root
python run.py --method nnc --dataset_name spcm --model_name cnn \
  --embeddings_dir /path/to/features --knn_neighbors 100 \
  --num_clients 30 --num_rounds 200 --batch_size 128 --train_epochs 1 \
  --output runs/spcm-nnc
```

NNC and AKD read `/path/to/features/spcm.npy`; row `file_id` must represent the
matching training waveform. AKD's `--embeddings_dims` equals the feature width
(1024 for TRILLsson 3). Features, identity sidecars, and source manifests stay
together. FedAvg and NA-FedAvg use no embedding file.

## Audio frontend

Training uses a 15,690-sample random window, L2 waveform normalization, 64 mel
bins, and a log-magnitude spectrogram. Evaluation keeps full padded waveforms
and batches one example to accommodate variable duration. The exporter uses the
TFDS twelve-label recipe, including its unknown and silence classes. Validation
is exported separately from training and test; training consumes train/test.
TRILLsson receives full mono 16 kHz PCM waveforms through its own frontend,
separately from the client's spectrogram preprocessing.

`tests/test_audio_data.py::write_records` demonstrates the record schema using
generated waveforms. The tests also exercise NNC and AKD training on those files.
