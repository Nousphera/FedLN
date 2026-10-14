from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from flwr.common import Code, Status, GetPropertiesRes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_transport import StableClientManager, PartitionClient


class Proxy:
    def __init__(self, node, partition):
        self.cid, self.partition = node, partition
    def get_properties(self, ins, timeout, group_id):
        return GetPropertiesRes(Status(Code.OK,""), {"partition-id":self.partition})


def test_sampling_uses_partition_identity_across_transport_node_ids():
    selections = []
    for nodes in [["random-a","random-b","random-c"], ["other-z","other-b","other-a"]]:
        manager = StableClientManager(3,42)
        for cid in [2,0,1]:
            manager.register(Proxy(nodes[cid],cid))
        selections.append([[p.partition for p in manager.sample(2)] for _ in range(5)])
    assert selections[0] == selections[1]
    assert len(set(sum(selections[0],[]))) == 3


def test_reject_duplicate_partition_ids():
    manager = StableClientManager(2,42)
    manager.register(Proxy("a",0)); manager.register(Proxy("b",0))
    with pytest.raises(ValueError, match="exactly cover"):
        manager.sample(1)


def test_properties_do_not_initialize_training():
    client = PartitionClient(1, lambda: pytest.fail("unneeded initialization"), 42, 2)
    assert client.get_properties(None).properties["partition-id"] == 1


def test_native_keyword_calls_save_effective_post_evaluation_weights(tmp_path, monkeypatch):
    import json
    import numpy as np
    import flwr.server.compat
    from flwr.common import FitRes, ndarrays_to_parameters
    from native_transport import run_native
    initial = ndarrays_to_parameters([np.zeros(2, np.float32)])
    result = FitRes(Status(Code.OK, ""), initial, 4, {"partition-id": 1})
    strategy = SimpleNamespace(
        parameters=None,
        configure_fit=lambda rnd, parameters, manager: [],
        aggregate_fit=lambda rnd, results, failures: (initial, {"loss": 1.0}),
    )
    def evaluate(rnd, parameters):
        strategy.parameters = ndarrays_to_parameters([np.ones(2, np.float32)])
        return 0.5, {"accuracy":0.5}
    strategy.evaluate = evaluate
    server = SimpleNamespace(strategy=strategy)
    def fake_start(*, driver, server, config):
        strategy.configure_fit(server_round=1, parameters=initial, client_manager=None)
        strategy.aggregate_fit(server_round=1, results=[(None,result)], failures=[])
        strategy.evaluate(server_round=1, parameters=initial)
    monkeypatch.setattr(flwr.server.compat, "start_driver", fake_start)
    report = run_native(None,server,1,2,42,tmp_path,{"rounds":[]})
    assert report["rounds"][0]["clients"] == [1]
    assert report["rounds"][0]["parameters_changed"]
    with np.load(tmp_path/"final-parameters.npz") as saved:
        np.testing.assert_array_equal(saved["arr_0"], np.ones(2))
