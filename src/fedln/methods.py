"""Client partitions, class-conditional noise and FedLN interventions."""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def partitions(size, clients, variation=.25, seed=42):
    """Disjoint random shards; variation is the target coefficient of variation."""
    if size < clients or clients < 1 or not 0 <= variation <= 1:
        raise ValueError("Each client needs at least one training sample")
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log1p(variation ** 2))
    weights = rng.lognormal(-sigma ** 2 / 2, sigma, clients)
    shares = (size - clients) * weights / weights.sum()
    counts = np.floor(shares).astype(np.int64) + 1
    remainder = size - counts.sum()
    counts[np.argsort(-(shares % 1), kind="stable")[:remainder]] += 1
    return np.split(rng.permutation(size), np.cumsum(counts)[:-1])


def corrupt_labels(labels, classes, level, sparsity, seed):
    labels = np.asarray(labels, dtype=np.int64)
    if not 0 <= level <= 1 or not 0 <= sparsity <= 1:
        raise ValueError("Noise level and sparsity must be in [0, 1]")
    if not len(labels) or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Invalid labels")
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    destinations = max(1, int(np.ceil((classes - 1) * (1 - sparsity))))
    for label in range(classes):
        alternatives = rng.choice(np.delete(np.arange(classes), label), destinations, replace=False)
        rows = np.flatnonzero((labels == label) & (rng.random(len(labels)) < level))
        noisy[rows] = rng.choice(alternatives, len(rows))
    return noisy


def client_data(labels, classes, cfg, cid):
    shards = partitions(len(labels), cfg["num-clients"], cfg["quantity-variation"], cfg["seed"])
    noisy = set(np.random.default_rng(cfg["seed"] + 1).choice(
        cfg["num-clients"], round(cfg["noisy-clients"] * cfg["num-clients"]), replace=False).tolist())
    synthetic = cfg["dataset"] not in {"cifar10n", "cifar100n"}

    def local(i):
        values = np.asarray(labels)[shards[i]].copy()
        if synthetic and i in noisy:
            values = corrupt_labels(values, classes, cfg["noise-level"], cfg["noise-sparsity"], cfg["seed"] + 1009 * (i + 1))
        return values

    if cfg["method"] == "centralized":
        return np.concatenate(shards), np.concatenate([local(i) for i in range(len(shards))])
    return shards[cid], local(cid)


def neighbor_labels(features, labels, neighbors=100, seed=42):
    """Euclidean vote over other local samples, with seeded random tie breaking."""
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels) or not np.isfinite(features).all() or len(labels) < 2:
        raise ValueError("kNN requires finite aligned embeddings for at least two local samples")
    k = min(neighbors, len(labels) - 1)
    if k < 1:
        raise ValueError("neighbors must be positive")
    index = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1).fit(features)
    rng = np.random.default_rng(seed)
    corrected = np.empty_like(labels)
    # Bound distance-matrix memory for clients with large local datasets.
    for start in range(0, len(labels), 512):
        rows = index.kneighbors(features[start:start + 512], return_distance=False)
        for offset, row in enumerate(rows):
            own = start + offset
            row = row[row != own][:k]
            classes, counts = np.unique(labels[row], return_counts=True)
            corrected[own] = rng.choice(classes[counts == counts.max()])
    return corrected, float(np.mean(corrected != labels))


def energy_weights(scores, percentile=75.):
    """Low local energy indicates noise; aggregate using effective clean counts."""
    if not scores:
        raise ValueError("Energy scores are required")
    for global_scores, local_scores in scores.values():
        if (np.ndim(global_scores) != 1 or np.shape(global_scores) != np.shape(local_scores)
                or not len(global_scores) or not np.isfinite(global_scores).all() or not np.isfinite(local_scores).all()):
            raise ValueError("Expected finite aligned global/local score vectors")
    cutoff = float(np.percentile(np.concatenate([g for g, _ in scores.values()]), percentile))
    noise = {cid: float(np.mean(local < cutoff)) for cid, (_, local) in scores.items()}
    weights = {cid: len(local) * (1 - noise[cid]) for cid, (_, local) in scores.items()}
    return weights, noise, cutoff


def aggregate(states, weights, reference):
    """Sample/noise weighted mean, retaining valid integer BatchNorm counters."""
    if len(states) != len(weights) or not states:
        raise ValueError("Missing client updates")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid aggregation weights")
    for state in states:
        if set(state) != set(reference):
            raise ValueError("Client model keys differ")
        for key, value in state.items():
            if value.shape != reference[key].shape or value.dtype != reference[key].dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid client model tensor: {key}")
    if weights.sum() == 0:
        return {k: v.clone() for k, v in reference.items()}
    weights /= weights.sum()
    out = {}
    for key, value in reference.items():
        if value.is_floating_point():
            out[key] = sum(state[key].to(torch.float64) * float(w) for state, w in zip(states, weights)).to(value.dtype)
        else:
            out[key] = torch.stack([state[key] for state, w in zip(states, weights) if w > 0]).amax(0)
    return out
