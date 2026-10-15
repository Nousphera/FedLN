# Flower execution

```bash
flwr run .
python scripts/check_run.py runs/flower-smoke --rounds 2
```

`pyproject.toml` registers the ClientApp and ServerApp. The ServerApp uses
Flower 1.13's strategy API and the same training core as the sequential runner;
the ClientApp constructs clients from their SuperNode `partition-id`.
See Flower's [1.13 migration guide](https://flower.ai/docs/framework/how-to-upgrade-to-flower-1.13.html).

## Identity and state

The server resolves logical IDs with `get_properties`, validates a complete
unique partition mapping, samples with the configured seed, and sorts returned
updates by logical ID. Worker restarts preserve client identity. Run-owned caches
live on the shared local filesystem and persist across calls. Every run needs a
new output directory. Client failures abort aggregation; small positive
participation still selects at least one client. NA-FedAvg additionally controls its all-client noise-estimation stage.

## Resources and configuration

The default local federation allocates one Ray CPU, so one worker trains at a
time. Increase `options.backend.init-args.num-cpus` to allow concurrent workers.
The selected federation's `options.num-supernodes` must equal the configured
client count. CPU is the default; GPU runs also require matching backend GPU
resources and the training CLI's device option.

Install `requirements.txt` in the README's Python 3.11 environment.
Flower 1.13.1 and Ray 2.10.0 are pinned. Create the checkout and virtual
environment at final paths without spaces: Ray launches subprocesses using
those paths, and a shell alias or symlink does not reliably change them.

TensorFlow 2.15 requires a protobuf range compatible with Flower 1.13. Keep the
dependencies in `requirements.txt` together.

`--run-config 'options="..."'` replaces the complete training argument string.
Use the `paper` federation for the client count shown in the README, or edit a
federation to match your experiment. This profile uses local workers with shared
data and output directories.

## Results

The server saves results after each centralized evaluation and exports the
evaluated global parameters. The output checker requires the requested number
of completed rounds and finite final arrays, so it also detects incomplete
training when the Flower CLI itself exits successfully. Use the appropriate
`--rounds` value for your run. The default generated dataset provides a short
installation and training check. Weight exports contain model state; optimizer
and client caches are scoped to the running experiment.
