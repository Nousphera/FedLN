"""Native Flower App entry points for the maintained FedLN core."""
import shlex
from pathlib import Path
from flwr.client import ClientApp
from flwr.server import ServerApp
from native_transport import PartitionClient
from run import main


def options(context):
    return shlex.split(str(context.run_config["options"]))


def client_fn(context):
    argv = options(context)
    cid = int(context.node_config["partition-id"])
    # These transport settings share the exact command-line options of the runner.
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_clients", type=int, default=3)
    settings, _ = p.parse_known_args(argv)
    return PartitionClient(cid, lambda: main(argv, client_id=cid), settings.seed, settings.num_clients)

client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp()


@server_app.main()
def server_main(driver, context):
    main(options(context), driver=driver)
