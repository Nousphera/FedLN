"""Validate completed native smoke artifacts independently of the CLI exit code."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--rounds", type=int, required=True)
a = p.parse_args()
r = json.loads((a.output / "results.json").read_text())
assert r["backend"] == "flower-clientapp-serverapp-local-simulation"
assert [x["round"] for x in r["rounds"]] == list(range(1, a.rounds + 1))
assert all(x["examples"] > 0 and x["clients"] and len(x["clients"]) == len(set(x["clients"])) for x in r["rounds"])
assert any(x["parameters_changed"] for x in r["rounds"])
with np.load(a.output / "final-parameters.npz", allow_pickle=False) as data:
    assert data.files and all(np.isfinite(data[key]).all() for key in data.files)
print(f"Verified {a.rounds} completed native rounds and finite final parameters")
