"""Flower client for federated disposable income prediction."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, get_input_dim, load_data, set_seed
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

set_seed()

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    # Initialize model with global weights
    input_dim = get_input_dim()
    model = Net(input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load this client's partitioned data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = int(context.run_config["batch-size"])
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Read config from server (SmartFedProx sends per-round adaptive values)
    config = msg.content["config"]
    mu = float(config["mu"])

    # Local epochs: per-round from server if available, else from run config
    try:
        epochs = int(config["epochs"])
    except (KeyError, TypeError):
        epochs = int(context.run_config["local-epochs"])

    # Gradient clipping: SmartFedProx sends this to prevent non-IID gradient explosion
    try:
        grad_clip = float(config["grad_clip"])
    except (KeyError, TypeError):
        grad_clip = 0.0

    # Snapshot global params before local training (needed for proximal term)
    global_params = (
        [p.clone().detach().to(device) for p in model.parameters()]
        if mu > 0.0
        else None
    )
    model.to(device)

    # Local training
    train_loss = train_fn(
        model,
        trainloader,
        epochs,
        float(config["lr"]),
        device,
        mu=mu,
        global_params=global_params,
        grad_clip=grad_clip,
    )

    # Compute model divergence from global model (L2 norm of param diff)
    divergence = 0.0
    if global_params is not None:
        divergence = sum(
            ((lp - gp) ** 2).sum().item()
            for lp, gp in zip(model.parameters(), global_params)
        ) ** 0.5

    # Return updated model and metrics
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "train_loss": train_loss,
            "num-examples": float(len(trainloader.dataset)),
            "divergence": divergence,
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate model on local test data."""
    input_dim = get_input_dim()
    model = Net(input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = int(context.run_config["batch-size"])
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_r2 = test_fn(model, valloader, device)

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": eval_loss,
            "eval_r2": eval_r2,
            "num-examples": float(len(valloader.dataset)),
        }),
    })
    return Message(content=content, reply_to=msg)
