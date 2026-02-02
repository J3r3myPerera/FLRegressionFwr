#Server
import torch
import numpy as np
from collections import defaultdict
from module import Net, get_input_dim, load_centralized_dataset, test, NUM_CLIENTS, BATCH_SIZE, FRACTION_FIT, LOCAL_EPOCHS, LEARNING_RATE, DEVICE
from client import SimulatedClient


class FederatedSimulator:
    
    def __init__(self, strategy_name: str, config: dict):
        self.strategy_name = strategy_name
        self.config = config
        self.clients = []
        self.client_stats = defaultdict(lambda: {"divergences": [], "participation": 0})
        
        # Initialize clients
        print(f"\n  Initializing {NUM_CLIENTS} clients...")
        for i in range(NUM_CLIENTS):
            self.clients.append(SimulatedClient(i, NUM_CLIENTS, BATCH_SIZE))
    
    def select_clients(self, round_num: int) -> list:
        """Select clients based on strategy."""
        num_to_select = max(2, int(NUM_CLIENTS * FRACTION_FIT))
        
        if self.config["selection_strategy"] == "random":
            return list(np.random.choice(NUM_CLIENTS, num_to_select, replace=False))
        
        elif self.config["selection_strategy"] == "hybrid":
            # Cold start: random for first 3 rounds to build better history
            if round_num <= 3:
                return list(np.random.choice(NUM_CLIENTS, num_to_select, replace=False))
            
            # Exploration: 15% chance of random selection to prevent lock-in
            if np.random.random() < 0.15:
                return list(np.random.choice(NUM_CLIENTS, num_to_select, replace=False))
            
            # Get clients with divergence history uses smoothed average. Calculates the smooth divergence for each client
            clients_with_history = []
            for i in range(NUM_CLIENTS):
                divs = self.client_stats[i]["divergences"]
                if len(divs) >= 2:
                    # Weighted average favoring recent
                    avg_div = 0.5 * divs[-1] + 0.3 * divs[-2] + 0.2 * np.mean(divs[:-2]) if len(divs) > 2 else 0.6 * divs[-1] + 0.4 * divs[-2]
                elif len(divs) == 1:
                    avg_div = divs[-1]
                else:
                    avg_div = 0
                clients_with_history.append((i, avg_div))
            
            # Sort by divergence
            sorted_clients = sorted(clients_with_history, key=lambda x: x[1], reverse=True)
            
            # Balanced selection: prioritize middle-divergence clients for stability
            # with some high and low for adaptation
            # 30% high, 50% middle, 20% low
            num_high = max(1, num_to_select * 3 // 10)
            num_low = max(1, num_to_select * 2 // 10)
            num_mid = num_to_select - num_high - num_low
            
            high_div = [c[0] for c in sorted_clients[:num_high]]
            low_div = [c[0] for c in sorted_clients[-num_low:] if c[0] not in high_div]
            
            # Middle clients for stability
            mid_start = num_high
            mid_end = len(sorted_clients) - num_low
            mid_candidates = [c[0] for c in sorted_clients[mid_start:mid_end] if c[0] not in high_div + low_div]
            if len(mid_candidates) > num_mid:
                mid_div = list(np.random.choice(mid_candidates, num_mid, replace=False))
            else:
                mid_div = mid_candidates
            
            selected = high_div + mid_div + low_div
            
            # Fill remaining slots randomly if needed
            while len(selected) < num_to_select:
                remaining = [i for i in range(NUM_CLIENTS) if i not in selected]
                if remaining:
                    selected.append(np.random.choice(remaining))
                else:
                    break
            
            return selected[:num_to_select]
        
        return list(range(num_to_select))
    
    def aggregate(self, client_results: list) -> dict:
        
        total_examples = sum(r["num_examples"] for r in client_results)
        
        # Weighted average of model parameters
        aggregated_state = {}
        for key in client_results[0]["state_dict"].keys():
            weighted_sum = torch.zeros_like(client_results[0]["state_dict"][key], dtype=torch.float32)
            for result in client_results:
                weight = result["num_examples"] / total_examples
                weighted_sum += result["state_dict"][key].float() * weight
            aggregated_state[key] = weighted_sum
        
        return aggregated_state
    
    def evaluate_global(self, model_state_dict) -> tuple:
        """Evaluate global model on centralized test set."""
        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)
        model.load_state_dict(model_state_dict)
        
        test_dataloader = load_centralized_dataset()
        loss, r2 = test(model, test_dataloader, DEVICE)
        return loss, r2
    
    def run(self, num_rounds: int) -> dict:
        """Run federated learning simulation."""
        print(f"\n{'='*60}")
        print(f"Running: {self.strategy_name}")
        print(f"Config: {self.config['description']}")
        print(f"{'='*60}")
        
        # Initialize global model
        input_dim = get_input_dim()
        global_model = Net(input_dim=input_dim)
        global_state = global_model.state_dict()
        
        # Metrics storage
        metrics = {
            "rounds": [],
            "r2_scores": [],
            "mse_losses": [],
            "avg_train_loss": [],
            "avg_divergence": [],
            "avg_effective_mu": [],
        }
        
        # Training config
        train_config = {
            "local_epochs": LOCAL_EPOCHS,
            "lr": LEARNING_RATE,
            "proximal_mu": self.config["proximal_mu"],
            "adaptive_mu_enabled": self.config["adaptive_mu_enabled"],
        }
        
        # Track global average divergence across rounds
        global_avg_divergence = 0.0
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n  Round {round_num}/{num_rounds}")
            
            # Select clients
            selected_ids = self.select_clients(round_num)
            print(f"    Selected clients: {selected_ids}")
            
            # Train on selected clients. server passes the global context to each client
            client_results = []
            for client_id in selected_ids:
                result = self.clients[client_id].train(global_state, train_config, global_avg_divergence)
                client_results.append(result)
                
                # Track stats
                self.client_stats[client_id]["divergences"].append(result["divergence"])
                self.client_stats[client_id]["participation"] += 1
            
            # Aggregate
            global_state = self.aggregate(client_results)
            
            # Evaluate global model
            loss, r2 = self.evaluate_global(global_state)
            
            # Compute round metrics
            avg_train_loss = np.mean([r["train_loss"] for r in client_results])
            avg_divergence = np.mean([r["divergence"] for r in client_results])
            avg_mu = np.mean([r["effective_mu"] for r in client_results])
            
            # Update global average divergence for next round (EMA)
            if global_avg_divergence == 0:
                global_avg_divergence = avg_divergence
            else:
                global_avg_divergence = 0.7 * global_avg_divergence + 0.3 * avg_divergence
            
            # Store metrics
            metrics["rounds"].append(round_num)
            metrics["r2_scores"].append(r2)
            metrics["mse_losses"].append(loss)
            metrics["avg_train_loss"].append(avg_train_loss)
            metrics["avg_divergence"].append(avg_divergence)
            metrics["avg_effective_mu"].append(avg_mu)
            
            print(f"    R² = {r2:.4f}, MSE = {loss:.4f}, Avg μ = {avg_mu:.4f}")
        
        print(f"\n  Final R²: {metrics['r2_scores'][-1]:.4f}")
        return metrics
