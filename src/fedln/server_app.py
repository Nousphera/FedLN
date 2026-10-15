"""Native Flower orchestration with checked updates and central evaluation."""

import json
import math
from pathlib import Path
import platform
import time
from importlib.metadata import version
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import ServerApp, Grid
from .config import defaults, validate
from .data import load_data
from .methods import aggregate, energy_weights
from .models import build_model
from .training import evaluate, seed_everything

app = ServerApp()


def wait_for_nodes(grid, count, timeout):
    deadline = time.monotonic() + timeout
    while True:
        nodes = sorted(grid.get_node_ids())
        if len(set(nodes)) != len(nodes) or len(nodes) > count:
            raise ValueError("Unexpected Flower SuperNodes")
        if len(nodes) == count:
            return nodes
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Only {len(nodes)}/{count} Flower SuperNodes registered")
        time.sleep(.25)


def replies(grid, messages, timeout):
    expected = {m.metadata.dst_node_id: m for m in messages}
    seen = set()
    for reply in grid.send_and_receive(messages, timeout=timeout):
        node = reply.metadata.src_node_id
        if reply.has_error() or node not in expected or node in seen:
            raise RuntimeError("Each selected client must return one successful update")
        if reply.metadata.group_id != expected[node].metadata.group_id:
            raise ValueError("Reply round mismatch")
        seen.add(node)
        yield reply
    if seen != set(expected):
        raise RuntimeError("Missing client replies; round was not aggregated")


def run_server(grid, cfg):
    cfg = validate(defaults() | cfg)
    seed_everything(cfg["seed"], cfg["torch-threads"])
    bundle = load_data(cfg["dataset"], cfg["data-root"], training=False, human_noise_key=cfg["human-noise-key"] or None)
    model = build_model(cfg["dataset"], cfg["model"])
    count = 1 if cfg["method"] == "centralized" else cfg["num-clients"]
    nodes = wait_for_nodes(grid, count, cfg["registration-timeout"])

    def message(node, rnd):
        content = RecordDict({"config": ConfigRecord({"round": rnd, "phase": "train" if rnd else "identify"})})
        if rnd:
            content["model"] = ArrayRecord(model.state_dict())
        return Message(content, dst_node_id=node, message_type="train", group_id=str(rnd))

    mapping = {}
    for reply in replies(grid, [message(node, 0) for node in nodes], cfg["reply-timeout"]):
        cid = reply.content["metrics"]["client-id"]
        if type(cid) != int or cid in mapping or not 0 <= cid < count:
            raise ValueError("Client partition IDs must be unique and complete")
        mapping[cid] = reply.metadata.src_node_id
    rng = np.random.default_rng(cfg["seed"])
    impacts, noise, cutoff, history = None, {}, None, []
    for rnd in range(1, cfg["num-rounds"] + 1):
        score_round = cfg["method"] == "nafedavg" and rnd == cfg["noise-round"]
        sample_size = count if score_round else max(1, math.floor(count * cfg["participation"]))
        selected = sorted(rng.choice(count, sample_size, replace=False).tolist())
        messages = [message(mapping[cid], rnd) for cid in selected]
        updates, weights, scores, metrics = [], [], {}, []
        for reply in replies(grid, messages, cfg["reply-timeout"]):
            metric = dict(reply.content["metrics"])
            cid, n = metric["client-id"], metric["num-examples"]
            if (cid not in selected or mapping[cid] != reply.metadata.src_node_id
                    or type(n) != int or n <= 0 or not all(math.isfinite(v) for v in metric.values())):
                raise ValueError("Invalid client metrics")
            updates.append(reply.content["model"].to_torch_state_dict())
            metrics.append(metric)
            if score_round:
                vectors = reply.content["energy"].to_numpy_ndarrays()
                if len(vectors) != 2 or any(len(s) != n for s in vectors):
                    raise ValueError("Energy scores must cover every local example")
                scores[cid] = tuple(vectors)
        if score_round:
            if set(scores) != set(range(count)):
                raise ValueError("Noise estimation requires all clients")
            impacts, noise, cutoff = energy_weights(scores, cfg["percentile"])
        weights = [impacts[m["client-id"]] if impacts is not None else m["num-examples"] for m in metrics]
        model.load_state_dict(aggregate(updates, weights, model.state_dict()))
        evaluation = evaluate(model, bundle.test, cfg)
        total = sum(m["num-examples"] for m in metrics)
        entry = {"round": rnd, "clients": metrics, "train-loss": sum(m["loss"] * m["num-examples"] for m in metrics) / total,
                 "aggregation-skipped": sum(weights) == 0, **evaluation}
        history.append(entry)
        print(json.dumps({"round": rnd, **evaluation}), flush=True)
    return model, {
        "schema": 1, "config": cfg, "data-fingerprint": bundle.fingerprint,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "flower": version("flwr")},
        "rounds": history, "noise-estimates": noise, "energy-threshold": cutoff,
    }


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = validate(defaults() | context.run_config)
    path = Path(cfg["output"])
    checkpoint = path.with_suffix(".pt")
    if path.exists() or checkpoint.exists():
        raise FileExistsError("Output exists; choose a new --output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    model, result = run_server(grid, cfg)
    with checkpoint.open("xb") as stream:
        torch.save({"model": model.state_dict(), "config": cfg}, stream)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
