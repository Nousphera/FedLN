# Native Flower execution

`flwr run .` reads the two real App entry points from `pyproject.toml`. The
ServerApp uses Flower 1.13's strategy compatibility API, exactly as that version's
ServerApp does internally, with the maintained trainer's server. The ClientApp
creates the maintained client from its SuperNode `partition-id`. No training
algorithm is reimplemented in the transport adapter. See Flower's
[1.13 migration guide](https://flower.ai/docs/framework/how-to-upgrade-to-flower-1.13.html).

The server resolves logical IDs with `get_properties` before seeded sampling,
validates a complete unique partition mapping, and sorts returned updates by
logical ID. Worker restarts do not change partition identity. Run-owned caches
persist across calls on the shared local filesystem; use one new output directory
per run. Client failures abort the run instead of silently aggregating a smaller
population. Tiny positive participation selects at least one client. FedLN's
NA-FedAvg strategy still controls its all-client noise-estimation stage.

For constrained machines, the default federation grants Ray one CPU, so only one
worker trains at a time. Increasing `options.backend.init-args.num-cpus` permits
concurrent workers without changing the method. The baseline is macOS ARM64 CPU,
Python 3.11, Flower 1.13.1 and Ray 2.10.0. Multi-host filesystem/cache ownership,
GPU scheduling, failure recovery and throughput measurements are unvalidated.

Install `requirements-simulation.txt` in the dedicated environment from the
README. Both checkout and environment paths should contain no spaces. The
modern Flower 1.40 dependency stack cannot be combined with TensorFlow 2.15's
protobuf requirements. A shell alias or a symlink to a spaced virtual environment
does not reliably fix Ray's process startup; create the environment at its final
no-space path. No patched Flower installation is needed.

CLI `options` replaces the whole string in the project configuration. All normal
method/data options remain available; `flwr run . --run-config 'options="..."'`
is the native invocation, and the script CLI remains the sequential alternative.

A native run writes results after each centralized evaluation and checkpoints
the effective global model, including server-distilled FedCompress weights.
Synthetic results establish execution only, not scientific benchmark claims.
