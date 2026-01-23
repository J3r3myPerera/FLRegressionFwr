"""Custom FedProx strategy with smart client selection."""

import math
import random
from collections import defaultdict
from typing import Iterable

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedProx


class SelectionStrategy:
    """Selection strategy types."""

    RANDOM = "random"
    DIVERSITY = "diversity"  # Prefer high-divergence clients
    HYBRID = "hybrid"  # Balanced mix of high/low divergence


class ClientHistory:
    """Tracks per-client metrics across rounds."""

    def __init__(self):
        self.divergence: list[float] = []
        self.train_loss: list[float] = []
        self.effective_mu: list[float] = []
        self.num_examples: int = 0
        self.participation_count: int = 0

    @property
    def avg_divergence(self) -> float:
        return sum(self.divergence) / len(self.divergence) if self.divergence else 0.0

    @property
    def latest_divergence(self) -> float:
        return self.divergence[-1] if self.divergence else 0.0

    @property
    def latest_loss(self) -> float:
        return self.train_loss[-1] if self.train_loss else float("inf")


class SmartFedProx(FedProx):
    """FedProx strategy with intelligent client selection based on divergence metrics."""

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        proximal_mu: float = 0.1,
        # Smart selection parameters
        selection_strategy: str = SelectionStrategy.RANDOM,
        selection_temperature: float = 1.0,
        hybrid_high_ratio: float = 0.5,
        cold_start_rounds: int = 2,
        exploration_rate: float = 0.1,
    ):
        # FedProx uses fraction_train, not fraction_fit
        super().__init__(
            fraction_train=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_fit_clients,
            min_evaluate_nodes=min_evaluate_clients,
            min_available_nodes=min_available_clients,
            proximal_mu=proximal_mu,
        )

        # Store our own fraction_fit for use in configure_train
        self.fraction_fit = fraction_fit

        # Selection configuration
        self.selection_strategy = selection_strategy
        self.selection_temperature = selection_temperature
        self.hybrid_high_ratio = hybrid_high_ratio
        self.cold_start_rounds = cold_start_rounds
        self.exploration_rate = exploration_rate
        self.min_fit_clients = min_fit_clients

        # State tracking
        self.client_history: dict[int, ClientHistory] = defaultdict(ClientHistory)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure training with smart client selection."""

        # Get all available node IDs
        all_node_ids = list(grid.get_node_ids())

        # Calculate number of clients to select
        num_to_select = max(
            self.min_fit_clients, int(len(all_node_ids) * self.fraction_fit)
        )
        num_to_select = min(num_to_select, len(all_node_ids))

        # Select clients based on strategy
        selected_nodes = self._select_clients(all_node_ids, num_to_select, server_round)

        # Add proximal_mu to config
        config_dict = dict(config)
        config_dict["proximal_mu"] = self.proximal_mu

        # Create messages only for selected nodes (using Message constructor, not deprecated grid.create_message)
        messages = []
        for node_id in selected_nodes:
            content = RecordDict(
                {
                    "arrays": arrays,
                    "config": ConfigRecord(config_dict),
                }
            )
            msg = Message(
                content=content,
                message_type="train",
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(msg)

        print(
            f"    Smart selection ({self.selection_strategy}): "
            f"Selected {len(selected_nodes)}/{len(all_node_ids)} clients"
        )

        return messages

    def _select_clients(
        self,
        all_node_ids: list[int],
        num_to_select: int,
        server_round: int,
    ) -> list[int]:
        """Select clients based on configured strategy."""

        # Cold start: use random selection for first few rounds
        if server_round <= self.cold_start_rounds:
            print(f"    Cold start round {server_round}/{self.cold_start_rounds}")
            return random.sample(all_node_ids, num_to_select)

        # Exploration: occasionally select randomly
        if random.random() < self.exploration_rate:
            print("    Exploration: using random selection")
            return random.sample(all_node_ids, num_to_select)

        # Get clients with history
        clients_with_history = [
            nid
            for nid in all_node_ids
            if nid in self.client_history and self.client_history[nid].divergence
        ]

        # If not enough history, mix with random
        if len(clients_with_history) < num_to_select:
            clients_without_history = [
                nid for nid in all_node_ids if nid not in clients_with_history
            ]
            remaining = num_to_select - len(clients_with_history)
            random_selection = random.sample(
                clients_without_history, min(remaining, len(clients_without_history))
            )
            return clients_with_history + random_selection

        # Apply selection strategy
        if self.selection_strategy == SelectionStrategy.DIVERSITY:
            return self._select_diversity(clients_with_history, num_to_select)
        elif self.selection_strategy == SelectionStrategy.HYBRID:
            return self._select_hybrid(clients_with_history, num_to_select)
        else:
            return random.sample(all_node_ids, num_to_select)

    def _select_diversity(self, node_ids: list[int], k: int) -> list[int]:
        """Select clients with highest divergence (most unique/non-IID data)."""
        # Sort by divergence descending
        sorted_nodes = sorted(
            node_ids,
            key=lambda nid: self.client_history[nid].latest_divergence,
            reverse=True,
        )
        return self._softmax_sample(sorted_nodes, k)

    def _select_hybrid(self, node_ids: list[int], k: int) -> list[int]:
        """Select a balanced mix of high and low divergence clients."""
        sorted_by_div = sorted(
            node_ids,
            key=lambda nid: self.client_history[nid].latest_divergence,
            reverse=True,
        )

        # Split into high and low divergence groups
        num_high = max(1, int(k * self.hybrid_high_ratio))
        num_low = k - num_high

        mid = len(sorted_by_div) // 2
        high_div = sorted_by_div[:mid]
        low_div = sorted_by_div[mid:]

        selected = []
        if high_div:
            selected.extend(random.sample(high_div, min(num_high, len(high_div))))
        if low_div:
            selected.extend(random.sample(low_div, min(num_low, len(low_div))))

        # Fill remaining if needed
        remaining = k - len(selected)
        if remaining > 0:
            available = [n for n in node_ids if n not in selected]
            if available:
                selected.extend(
                    random.sample(available, min(remaining, len(available)))
                )

        return selected

    def _softmax_sample(self, sorted_nodes: list[int], k: int) -> list[int]:
        """Sample k nodes using softmax probability weighting.

        Temperature controls randomness:
        - Low temperature (<1): More deterministic (closer to top-k)
        - Temperature = 1: Balanced probability weighting
        - High temperature (>1): More uniform sampling
        """
        if k >= len(sorted_nodes):
            return sorted_nodes[:k]

        n = len(sorted_nodes)
        # Position-based scores (higher for earlier positions = higher divergence)
        scores = [n - i for i in range(n)]

        # Apply temperature-scaled softmax
        max_score = max(scores)
        exp_scores = [
            math.exp((s - max_score) / max(self.selection_temperature, 0.01))
            for s in scores
        ]
        total = sum(exp_scores)
        probs = [s / total for s in exp_scores]

        # Weighted sampling without replacement
        selected = []
        remaining_nodes = list(sorted_nodes)
        remaining_probs = list(probs)

        for _ in range(k):
            if not remaining_nodes:
                break
            # Normalize remaining probabilities
            total_prob = sum(remaining_probs)
            norm_probs = [p / total_prob for p in remaining_probs]

            # Sample one
            r = random.random()
            cumsum = 0
            for i, p in enumerate(norm_probs):
                cumsum += p
                if r <= cumsum:
                    selected.append(remaining_nodes[i])
                    remaining_nodes.pop(i)
                    remaining_probs.pop(i)
                    break

        return selected

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate training results and update client history."""

        replies_list = list(replies)

        # Extract and store client metrics
        for reply in replies_list:
            node_id = reply.metadata.src_node_id
            metrics = reply.content.get("metrics", {})

            history = self.client_history[node_id]

            if "divergence" in metrics:
                history.divergence.append(float(metrics["divergence"]))
            if "train_loss" in metrics:
                history.train_loss.append(float(metrics["train_loss"]))
            if "effective_mu" in metrics:
                history.effective_mu.append(float(metrics["effective_mu"]))
            if "num-examples" in metrics:
                history.num_examples = int(metrics["num-examples"])

            history.participation_count += 1

        # Call parent aggregation
        return super().aggregate_train(server_round, replies_list)

    def get_client_stats(self) -> dict:
        """Get summary statistics for all tracked clients."""
        stats = {}
        for node_id, history in self.client_history.items():
            stats[node_id] = {
                "avg_divergence": history.avg_divergence,
                "latest_divergence": history.latest_divergence,
                "participation_count": history.participation_count,
                "num_examples": history.num_examples,
            }
        return stats
