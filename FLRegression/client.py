#Client

from module import Net, load_data, train, test, DEVICE


class SimulatedClient:

    def __init__(self, client_id: int, num_clients: int, batch_size: int):
        self.client_id = client_id
        self.trainloader, self.testloader = load_data(client_id, num_clients, batch_size)
        self.historical_divergence = 0.0
        self.num_examples = len(self.trainloader.dataset)

    def train(self, model_state_dict, config, global_avg_divergence: float = 0.0):
        #Train local model and return updated weights + metrics.
        model = Net()
        model.load_state_dict(model_state_dict)

        adaptive_mu_config = None
        if config.get("adaptive_mu_enabled", False):
            adaptive_mu_config = {
                "enabled": True,
                "historical_divergence": self.historical_divergence,
                "global_avg_divergence": global_avg_divergence,
                "mu_min": 0.001,
                "mu_max": 1.0,
            }

        result = train(
            model,
            self.trainloader,
            epochs=config["local_epochs"],
            lr=config["lr"],
            device=DEVICE,
            proximal_mu=config["proximal_mu"],
            adaptive_mu_config=adaptive_mu_config,
        )

        # Update historical divergence with EMA. Client returns metrics to server.
        alpha = 0.3
        self.historical_divergence = alpha * result["divergence"] + (1 - alpha) * self.historical_divergence

        return {
            "state_dict": model.state_dict(),
            "num_examples": self.num_examples,
            "train_loss": result["train_loss"],
            "divergence": result["divergence"],
            "effective_mu": result["effective_mu"],
        }

    def evaluate(self, model_state_dict):
        """Evaluate model on local test data."""
        model = Net()
        model.load_state_dict(model_state_dict)

        loss, accuracy = test(model, self.testloader, DEVICE)
        return {"loss": loss, "accuracy": accuracy, "num_examples": len(self.testloader.dataset)}
