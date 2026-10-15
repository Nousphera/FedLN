"""Stateful Flower clients for FedLN and its baselines."""

from copy import deepcopy
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .config import defaults, validate
from .data import load_data
from .methods import client_data, neighbor_labels
from .models import build_model
from .training import LocalData, confident_keep, predict, seed_everything, train_local

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = validate(defaults() | context.run_config)
    cid = context.node_config["partition-id"]
    limit = 1 if cfg["method"] == "centralized" else cfg["num-clients"]
    if type(cid) != int or not 0 <= cid < limit:
        raise ValueError("Invalid client partition")
    config = msg.content["config"]
    rnd, phase = config["round"], config["phase"]
    if type(rnd) != int or phase not in {"identify", "train"} or not 0 <= rnd <= cfg["num-rounds"]:
        raise ValueError("Invalid experiment message")
    if (phase == "identify") != (rnd == 0):
        raise ValueError("Invalid initialization round")
    if phase == "identify":
        return Message(RecordDict({"metrics": MetricRecord({"client-id": cid})}), reply_to=msg)

    seed_everything(cfg["seed"] + cid, cfg["torch-threads"])
    bundle = load_data(cfg["dataset"], cfg["data-root"], training=True, human_noise_key=cfg["human-noise-key"] or None)
    clean_view = load_data(cfg["dataset"], cfg["data-root"], training=False, human_noise_key=cfg["human-noise-key"] or None)
    indices, labels = client_data(bundle.labels, bundle.classes, cfg, cid)
    features = None
    if cfg["method"] in {"nnc", "akd"}:
        from .teachers import load_features
        features = load_features(cfg["dataset"], cfg["feature-root"], bundle)
    if "local-data" not in context.state:
        noise = 0.
        if features is not None:
            voted, noise = neighbor_labels(features[indices], labels, cfg["neighbors"], cfg["seed"] + cid)
            if cfg["method"] == "nnc":
                labels = voted
        elif cfg["method"] == "confident-learning":
            keep = confident_keep(bundle.train, clean_view.train, indices, labels, cfg, cfg["seed"] + cid)
            indices, labels = indices[keep], labels[keep]
        context.state["local-data"] = ArrayRecord([indices, labels])
        context.state["noise"] = ConfigRecord({"estimate": noise})
    else:
        indices, labels = context.state["local-data"].to_numpy_ndarrays()
    noise = context.state["noise"]["estimate"]
    local = LocalData(bundle.train, indices, labels)
    evaluation = LocalData(clean_view.train, indices, labels)
    model = build_model(cfg["dataset"], cfg["model"])
    state = msg.content["model"].to_torch_state_dict()
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Received nonfinite model")
    model.load_state_dict(state, strict=True)
    active = cfg["method"] == "akd" and noise > cfg["noise-threshold"]
    projection, teacher = None, None
    if active and cfg["teacher"] == "embeddings":
        projection = torch.nn.Linear(model.feature_dim, features.shape[1])
        torch.nn.init.xavier_uniform_(projection.weight)
        torch.nn.init.zeros_(projection.bias)
        if "projection" in context.state:
            projection.load_state_dict(context.state["projection"].to_torch_state_dict())
    elif active:
        teacher = deepcopy(model)
    score_round = cfg["method"] == "nafedavg" and rnd == cfg["noise-round"]
    if score_round:
        global_energy = predict(model, evaluation, cfg, energies=True)[0].numpy()
    optimizer = context.state["optimizer"].to_torch_state_dict() if "optimizer" in context.state else None
    metrics, optimizer = train_local(model, local, cfg, cfg["seed"] + rnd * cfg["num-clients"] + cid,
                                     features=features, projection=projection, teacher=teacher,
                                     optimizer_state=optimizer)
    if cfg["method"] == "centralized":
        context.state["optimizer"] = ArrayRecord(optimizer)
    if projection is not None:
        context.state["projection"] = ArrayRecord(projection.state_dict())
    content = RecordDict({
        "model": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"client-id": cid, "num-examples": len(local),
                                  "noise-estimate": noise, "distillation-active": int(active), **metrics}),
    })
    if score_round:
        local_energy = predict(model, evaluation, cfg, energies=True)[0].numpy()
        content["energy"] = ArrayRecord([global_energy, local_energy])
    return Message(content, reply_to=msg)
