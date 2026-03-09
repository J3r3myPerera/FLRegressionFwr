# Flower Client for Federated Learning

import flwr as fl
from module import (
    Net, get_input_dim, train, test,
    get_model_parameters, set_model_parameters,
    DEVICE, LOCAL_EPOCHS, LEARNING_RATE,
)


class FlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient for federated regression training."""

    def __init__(self, client_id, trainloader, testloader):
        self.client_id = client_id
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = len(trainloader.dataset)

    def get_parameters(self, config):
        """Return current model parameters."""
        model = Net(input_dim=get_input_dim())
        return get_model_parameters(model)

    def fit(self, parameters, config):
        """Train local model and return updated weights + metrics."""
        model = Net(input_dim=get_input_dim())
        set_model_parameters(model, parameters)

        # Read training config from server
        proximal_mu = config.get("proximal_mu", 0.0)
        adaptive_mu_enabled = config.get("adaptive_mu_enabled", False)
        local_epochs = config.get("local_epochs", LOCAL_EPOCHS)
        lr = config.get("lr", LEARNING_RATE)
        global_avg_divergence = config.get("global_avg_divergence", 0.0)
        historical_divergence = config.get("historical_divergence", 0.0)

        # Build adaptive mu config if enabled
        adaptive_mu_config = None
        if adaptive_mu_enabled:
            adaptive_mu_config = {
                "enabled": True,
                "historical_divergence": historical_divergence,
                "global_avg_divergence": global_avg_divergence,
                "mu_min": 0.01,
                "mu_max": 0.5,
            }

        result = train(
            model,
            self.trainloader,
            epochs=local_epochs,
            lr=lr,
            device=DEVICE,
            proximal_mu=proximal_mu,
            adaptive_mu_config=adaptive_mu_config,
        )

        return (
            get_model_parameters(model),
            self.num_examples,
            {
                "train_loss": float(result["train_loss"]),
                "divergence": float(result["divergence"]),
                "effective_mu": float(result["effective_mu"]),
                "client_id": int(self.client_id),
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        model = Net(input_dim=get_input_dim())
        set_model_parameters(model, parameters)
        loss, r2 = test(model, self.testloader, DEVICE)
        return float(loss), len(self.testloader.dataset), {"r2": float(r2)}
