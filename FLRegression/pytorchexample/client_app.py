"""pytorchexample: A Flower / PyTorch app for Personal Finance Prediction."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, get_input_dim, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    input_dim = get_input_dim()
    model = Net(input_dim=input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Get adaptive μ configuration from server
    config = msg.content["config"]
    adaptive_mu_enabled = config.get("adaptive_mu_enabled", False)

    adaptive_mu_config = None
    if adaptive_mu_enabled:
        # Retrieve historical divergence for this client from context state
        # context.state stores ConfigRecord, so we need to extract the value
        state_record = context.state.get("adaptive_state", None)
        if state_record is not None:
            historical_divergence = state_record.get("historical_divergence", 0.0)
        else:
            historical_divergence = 0.0
        adaptive_mu_config = {
            "enabled": True,
            "historical_divergence": historical_divergence,
            "mu_min": config.get("mu_min", 0.001),
            "mu_max": config.get("mu_max", 1.0),
        }

    # Call the training function with FedProx proximal term
    train_result = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        config["lr"],
        device,
        proximal_mu=config["proximal_mu"],
        adaptive_mu_config=adaptive_mu_config,
    )

    # Update historical divergence with exponential moving average
    current_divergence = train_result["divergence"]
    alpha = 0.3  # EMA smoothing factor
    state_record = context.state.get("adaptive_state", None)
    if state_record is not None:
        old_divergence = state_record.get("historical_divergence", current_divergence)
    else:
        old_divergence = current_divergence
    new_historical = alpha * current_divergence + (1 - alpha) * old_divergence
    # Store in ConfigRecord as required by Flower's context.state
    context.state["adaptive_state"] = ConfigRecord({"historical_divergence": new_historical})

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_result["train_loss"],
        "divergence": current_divergence,
        "effective_mu": train_result["effective_mu"],
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    input_dim = get_input_dim()
    model = Net(input_dim=input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_r2 = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_r2": eval_r2,  # R² score for regression
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
