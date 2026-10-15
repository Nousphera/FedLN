"""Single-host Flower transport; algorithms stay in the research core.

Flower 1.13 rebuilds clients between messages. Stable partition IDs and run-owned
on-disk state preserve client state across worker lifetimes. The local
simulation shares an absolute output directory; multi-host deployment is not
supported by this adapter.
"""
import json
import os
from pathlib import Path
import random
from contextlib import contextmanager
import numpy as np
from flwr.client import Client
from flwr.common import Code, Status, GetPropertiesRes, GetPropertiesIns, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        if path is not None:
            os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class PartitionClient(Client):
    """Defer data/model loading until fit, retaining the SuperNode partition ID."""
    def __init__(self, cid, factory, seed, clients, work=None, framework="tensorflow"):
        self.cid, self.factory = int(cid), factory
        self.seed, self.clients, self.work, self.framework = seed, clients, work, framework

    def get_properties(self, ins):
        return GetPropertiesRes(Status(Code.OK, ""), {"partition-id": self.cid})

    def fit(self, ins):
        seed = self.seed + int(ins.config["round"]) * self.clients + self.cid
        if self.framework == "tensorflow":
            import tensorflow as tf
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
        else:
            import torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        with working_directory(self.work):
            client = self.factory()
            if self.framework == "tensorflow":
                tf.keras.utils.set_random_seed(seed)
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
            response = client.to_client().fit(ins)
        response.metrics["partition-id"] = self.cid
        return response


class StableClientManager(SimpleClientManager):
    """Sample logical partitions, never random Flower transport node IDs."""
    def __init__(self, count, seed):
        super().__init__()
        self.count = count
        self.rng = random.Random(seed)
        self.identities = {}

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        if not self.wait_for(self.count, timeout=120):
            raise RuntimeError("Expected SuperNodes did not connect; match num-supernodes to client count")
        for node, proxy in self.all().items():
            if node not in self.identities:
                result = proxy.get_properties(GetPropertiesIns({}), timeout=120, group_id=0)
                if result.status.code != Code.OK:
                    raise RuntimeError("Could not resolve partition identity")
                self.identities[node] = int(result.properties["partition-id"])
        if sorted(self.identities.values()) != list(range(self.count)):
            raise ValueError("SuperNode partition IDs must exactly cover the configured clients")
        candidates = sorted(self.all().values(), key=lambda proxy: self.identities[proxy.cid])
        if criterion is not None:
            candidates = [proxy for proxy in candidates if criterion.select(proxy)]
        if len(candidates) < num_clients:
            raise ValueError("Not enough eligible clients")
        return self.rng.sample(candidates, num_clients)


def run_native(driver, server, rounds, count, seed, output, report):
    """Run the existing strategy through Flower and persist evaluated parameters."""
    from flwr.server.compat import start_driver
    from flwr.server import ServerConfig
    output = Path(output)
    strategy = server.strategy
    original_configure, original_aggregate, original_evaluate = strategy.configure_fit, strategy.aggregate_fit, strategy.evaluate
    report["backend"] = "flower-clientapp-serverapp-local-simulation"
    pending = {}
    before = []

    def configure(server_round, parameters, client_manager):
        rnd, manager = server_round, client_manager
        nonlocal before
        effective = getattr(strategy, "parameters", None) or parameters
        before = parameters_to_ndarrays(effective)
        return original_configure(rnd, parameters, manager)

    def aggregate(server_round, results, failures):
        rnd = server_round
        if failures:
            raise RuntimeError(f"Round {rnd} failed on {len(failures)} client(s): {failures}")
        if not results:
            raise RuntimeError("No client updates returned")
        # Scheduling completion order must not change aggregation order.
        results.sort(key=lambda item: int(item[1].metrics["partition-id"]))
        ids = [int(response.metrics["partition-id"]) for _, response in results]
        if len(ids) != len(set(ids)) or any(cid < 0 or cid >= count for cid in ids):
            raise ValueError("Duplicate or invalid client identity in fit results")
        parameters, metrics = original_aggregate(rnd, results, failures)
        if parameters is None:
            raise RuntimeError("Strategy did not return model parameters")
        pending.update(round=rnd, clients=ids, examples=sum(r.num_examples for _, r in results),
                       client_examples=[r.num_examples for _, r in results], aggregation_metrics=metrics)
        return parameters, metrics

    def evaluate(server_round, parameters):
        rnd = server_round
        evaluation = original_evaluate(rnd, parameters)
        if rnd and evaluation is not None:
            effective = getattr(strategy, "parameters", None) or parameters
            arrays = parameters_to_ndarrays(effective)
            loss, metrics = evaluation
            row = {**pending, "loss": float(loss), "metrics": metrics,
                   "parameters_changed": any(not np.array_equal(a, b) for a, b in zip(before, arrays))}
            report["rounds"].append(row)
            (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            np.savez_compressed(output / "final-parameters.npz", *arrays)
        return evaluation

    strategy.configure_fit, strategy.aggregate_fit, strategy.evaluate = configure, aggregate, evaluate
    server._client_manager = StableClientManager(count, seed)
    start_driver(driver=driver, server=server, config=ServerConfig(num_rounds=rounds))
    if len(report["rounds"]) != rounds:
        raise RuntimeError("Native Flower run ended without all round evaluations")
    return report
