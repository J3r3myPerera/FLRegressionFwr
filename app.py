"""
Streamlit Frontend for Federated Learning Regression
Indian Personal Finance - Disposable Income Prediction
"""

import streamlit as st
import numpy as np
import torch
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import sys
import os
from pathlib import Path

# Set DATA_PATH if not already set
if not os.getenv("DATA_PATH"):
    # Try to find the data file relative to the app location
    app_dir = Path(__file__).parent
    data_path = app_dir / "data" / "indianPersonalFinanceAndSpendingHabits.csv"
    if data_path.exists():
        os.environ["DATA_PATH"] = str(data_path)
    else:
        # Fallback: try parent directory
        data_path = app_dir.parent / "data" / "indianPersonalFinanceAndSpendingHabits.csv"
        if data_path.exists():
            os.environ["DATA_PATH"] = str(data_path)

# Add FLRegression to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FLRegression'))

from module import (
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS,
    LEARNING_RATE, BATCH_SIZE, DEVICE, STRATEGIES,
    get_input_dim, _load_and_preprocess_data, reset_data_cache
)
from server import FederatedSimulator

# Page configuration
st.set_page_config(
    page_title="Federated Learning Regression",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'simulation_complete' not in st.session_state:
        st.session_state.simulation_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'config' not in st.session_state:
        st.session_state.config = None


def run_single_simulation(strategy_name, config, num_rounds):
    """Run a single simulation for a strategy."""
    try:
        simulator = FederatedSimulator(strategy_name, config)
        metrics = simulator.run(num_rounds)
        return metrics
    except Exception as e:
        st.error(f"Error running {strategy_name}: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


def create_metrics_plots(all_results):
    """Create interactive plots for all metrics."""
    colors = {"FedAvg": "#e74c3c", "FedProx": "#3498db", "SmartFedProx": "#2ecc71"}
    markers = {"FedAvg": "circle", "FedProx": "square", "SmartFedProx": "triangle-up"}
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "R² Score Progression", "MSE Loss Progression", "Average Training Loss",
            "Average Model Divergence", "Average Effective μ", "Final Performance Comparison"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Plot 1: R² Score
    for name, metrics in all_results.items():
        fig.add_trace(
            go.Scatter(
                x=metrics["rounds"],
                y=metrics["r2_scores"],
                mode='lines+markers',
                name=name,
                line=dict(color=colors[name], width=2),
                marker=dict(symbol=markers[name], size=8)
            ),
            row=1, col=1
        )
    
    # Plot 2: MSE Loss
    for name, metrics in all_results.items():
        fig.add_trace(
            go.Scatter(
                x=metrics["rounds"],
                y=metrics["mse_losses"],
                mode='lines+markers',
                name=name,
                line=dict(color=colors[name], width=2),
                marker=dict(symbol=markers[name], size=8),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Plot 3: Training Loss
    for name, metrics in all_results.items():
        fig.add_trace(
            go.Scatter(
                x=metrics["rounds"],
                y=metrics["avg_train_loss"],
                mode='lines+markers',
                name=name,
                line=dict(color=colors[name], width=2),
                marker=dict(symbol=markers[name], size=8),
                showlegend=False
            ),
            row=1, col=3
        )
    
    # Plot 4: Divergence
    for name, metrics in all_results.items():
        fig.add_trace(
            go.Scatter(
                x=metrics["rounds"],
                y=metrics["avg_divergence"],
                mode='lines+markers',
                name=name,
                line=dict(color=colors[name], width=2),
                marker=dict(symbol=markers[name], size=8),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Plot 5: Effective μ
    for name, metrics in all_results.items():
        fig.add_trace(
            go.Scatter(
                x=metrics["rounds"],
                y=metrics["avg_effective_mu"],
                mode='lines+markers',
                name=name,
                line=dict(color=colors[name], width=2),
                marker=dict(symbol=markers[name], size=8),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Plot 6: Final Comparison
    names = list(all_results.keys())
    final_r2 = [all_results[n]["r2_scores"][-1] for n in names]
    final_mse = [all_results[n]["mse_losses"][-1] for n in names]
    
    fig.add_trace(
        go.Bar(
            x=names,
            y=final_r2,
            name="Final R²",
            marker_color=[colors[n] for n in names],
            text=[f"{r2:.4f}" for r2 in final_r2],
            textposition="outside"
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=names,
            y=final_mse,
            mode='lines+markers',
            name="Final MSE",
            line=dict(color='black', width=2, dash='dash'),
            marker=dict(size=10),
            yaxis='y8'
        ),
        row=2, col=3, secondary_y=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Federated Round", row=1, col=1)
    fig.update_yaxes(title_text="R² Score", row=1, col=1)
    fig.update_xaxes(title_text="Federated Round", row=1, col=2)
    fig.update_yaxes(title_text="MSE Loss", row=1, col=2)
    fig.update_xaxes(title_text="Federated Round", row=1, col=3)
    fig.update_yaxes(title_text="Avg Training Loss", row=1, col=3)
    fig.update_xaxes(title_text="Federated Round", row=2, col=1)
    fig.update_yaxes(title_text="Avg Divergence", row=2, col=1)
    fig.update_xaxes(title_text="Federated Round", row=2, col=2)
    fig.update_yaxes(title_text="Avg Effective μ", row=2, col=2)
    fig.update_xaxes(title_text="Strategy", row=2, col=3)
    fig.update_yaxes(title_text="R² Score", row=2, col=3)
    fig.update_yaxes(title_text="MSE Loss", row=2, col=3, secondary_y=True)
    
    fig.update_layout(
        height=900,
        title_text="Federated Learning Strategy Comparison<br><sub>Personal Finance - Disposable Income Prediction</sub>",
        title_x=0.5,
        showlegend=True
    )
    
    return fig


def create_individual_plot(all_results, metric_name, y_label, title):
    """Create an individual plot for a specific metric."""
    colors = {"FedAvg": "#e74c3c", "FedProx": "#3498db", "SmartFedProx": "#2ecc71"}
    markers = {"FedAvg": "circle", "FedProx": "square", "SmartFedProx": "triangle-up"}
    
    fig = go.Figure()
    
    for name, metrics in all_results.items():
        fig.add_trace(go.Scatter(
            x=metrics["rounds"],
            y=metrics[metric_name],
            mode='lines+markers',
            name=name,
            line=dict(color=colors[name], width=2.5),
            marker=dict(symbol=markers[name], size=10)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Federated Round",
        yaxis_title=y_label,
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Federated Learning Regression</h1>', unsafe_allow_html=True)
    st.markdown("### Indian Personal Finance - Disposable Income Prediction")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Simulation Configuration")
        
        # Dataset info
        st.subheader("Dataset")
        st.info("**Indian Personal Finance Dataset**\n\nPredicting Disposable Income using Federated Learning")
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        num_rounds = st.slider("Number of Rounds", min_value=5, max_value=50, value=10, step=1)
        num_clients = st.slider("Number of Clients", min_value=5, max_value=20, value=10, step=1)
        fraction_fit = st.slider("Fraction Fit (Clients per Round)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        local_epochs = st.slider("Local Epochs", min_value=1, max_value=10, value=3, step=1)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
        
        # Strategy selection
        st.subheader("Strategies to Run")
        run_fedavg = st.checkbox("FedAvg", value=True)
        run_fedprox = st.checkbox("FedProx", value=True)
        run_smartfedprox = st.checkbox("SmartFedProx", value=True)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            num_trials = st.slider("Number of Trials", min_value=1, max_value=5, value=1, step=1)
            use_fixed_seed = st.checkbox("Use Fixed Seed (Reproducible)", value=True)
            fixed_seed = st.number_input("Seed Value", min_value=0, max_value=10000, value=42, step=1) if use_fixed_seed else None
        
        # Run button
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
        
        # Reset button
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.simulation_complete = False
            st.session_state.results = None
            st.rerun()
    
    # Main content area
    if run_button:
        # Validate strategy selection
        selected_strategies = []
        if run_fedavg:
            selected_strategies.append("FedAvg")
        if run_fedprox:
            selected_strategies.append("FedProx")
        if run_smartfedprox:
            selected_strategies.append("SmartFedProx")
        
        if not selected_strategies:
            st.error("⚠️ Please select at least one strategy to run!")
            return
        
        # Note: Module constants are used from module.py
        # The UI parameters are for display/reference - actual values come from module.py
        st.info(f"ℹ️ Using configuration from module.py: {NUM_CLIENTS} clients, {FRACTION_FIT} fraction fit, {LOCAL_EPOCHS} local epochs, LR={LEARNING_RATE}, Batch={BATCH_SIZE}")
        
        # Check if data file exists
        data_path = os.getenv("DATA_PATH")
        if not data_path or not Path(data_path).exists():
            # Try to find it
            app_dir = Path(__file__).parent
            potential_paths = [
                app_dir / "data" / "indianPersonalFinanceAndSpendingHabits.csv",
                app_dir.parent / "data" / "indianPersonalFinanceAndSpendingHabits.csv",
            ]
            data_path = None
            for path in potential_paths:
                if path.exists():
                    data_path = str(path)
                    os.environ["DATA_PATH"] = data_path
                    break
            
            if not data_path:
                st.error("❌ Data file not found! Please ensure 'indianPersonalFinanceAndSpendingHabits.csv' exists in the 'data' directory.")
                return
        
        # Initialize data
        with st.spinner("Loading and preprocessing data..."):
            try:
                reset_data_cache()
                _load_and_preprocess_data()
                input_dim = get_input_dim()
                st.success(f"✅ Data loaded! Input dimension: {input_dim}")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.info("Please ensure the data file exists and is properly formatted.")
                return
        
        # Run simulations
        all_results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_simulations = len(selected_strategies) * num_trials
        current_simulation = 0
        
        all_trial_results = {name: [] for name in selected_strategies}
        
        # Set seed
        if use_fixed_seed:
            base_seed = fixed_seed
        else:
            base_seed = int(time.time()) % 10000
        
        for trial in range(num_trials):
            trial_seed = base_seed + trial * 100
            
            for strategy_name in selected_strategies:
                current_simulation += 1
                progress = current_simulation / total_simulations
                progress_bar.progress(progress)
                
                status_text.text(f"Running {strategy_name} - Trial {trial + 1}/{num_trials}...")
                
                # Set seed for this simulation
                np.random.seed(trial_seed)
                torch.manual_seed(trial_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(trial_seed)
                
                config = STRATEGIES[strategy_name]
                metrics = run_single_simulation(strategy_name, config, num_rounds)
                
                if metrics:
                    all_trial_results[strategy_name].append(metrics)
        
        # Aggregate results across trials
        if num_trials > 1:
            aggregated_results = {}
            for strategy_name in selected_strategies:
                trials = all_trial_results[strategy_name]
                if not trials:
                    continue
                
                aggregated_results[strategy_name] = {
                    "rounds": trials[0]["rounds"],
                    "r2_scores": [np.mean([t["r2_scores"][i] for t in trials]) for i in range(num_rounds)],
                    "mse_losses": [np.mean([t["mse_losses"][i] for t in trials]) for i in range(num_rounds)],
                    "avg_train_loss": [np.mean([t["avg_train_loss"][i] for t in trials]) for i in range(num_rounds)],
                    "avg_divergence": [np.mean([t["avg_divergence"][i] for t in trials]) for i in range(num_rounds)],
                    "avg_effective_mu": [np.mean([t["avg_effective_mu"][i] for t in trials]) for i in range(num_rounds)],
                }
            all_results = aggregated_results
        else:
            all_results = {name: all_trial_results[name][0] for name in selected_strategies if all_trial_results[name]}
        
        progress_bar.progress(1.0)
        status_text.text("✅ Simulation complete!")
        
        # Store results
        st.session_state.results = all_results
        st.session_state.simulation_complete = True
        st.session_state.config = {
            "num_rounds": num_rounds,
            "num_clients": NUM_CLIENTS,  # Use actual value from module
            "fraction_fit": FRACTION_FIT,  # Use actual value from module
            "local_epochs": LOCAL_EPOCHS,  # Use actual value from module
            "learning_rate": LEARNING_RATE,  # Use actual value from module
            "batch_size": BATCH_SIZE,  # Use actual value from module
            "num_trials": num_trials,
            "strategies": selected_strategies
        }
        
        st.rerun()
    
    # Display results if simulation is complete
    if st.session_state.simulation_complete and st.session_state.results:
        results = st.session_state.results
        config = st.session_state.config
        
        st.success("✅ Simulation completed successfully!")
        
        # Summary metrics
        st.header("📊 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Strategies", len(results))
        with col2:
            st.metric("Rounds", config["num_rounds"])
        with col3:
            st.metric("Clients", config["num_clients"])
        with col4:
            st.metric("Trials", config["num_trials"])
        
        # Final performance comparison table
        st.header("📈 Final Performance Comparison")
        comparison_data = []
        for name, metrics in results.items():
            comparison_data.append({
                "Strategy": name,
                "Final R²": f"{metrics['r2_scores'][-1]:.4f}",
                "Best R²": f"{max(metrics['r2_scores']):.4f}",
                "Final MSE": f"{metrics['mse_losses'][-1]:.4f}",
                "Lowest MSE": f"{min(metrics['mse_losses']):.4f}",
                "Final Avg μ": f"{metrics['avg_effective_mu'][-1]:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Determine winner
        winner = max(results.keys(), key=lambda x: results[x]["r2_scores"][-1])
        st.info(f"🏆 **Best performing strategy:** {winner} (Final R² = {results[winner]['r2_scores'][-1]:.4f})")
        
        # Comprehensive plots
        st.header("📉 Comprehensive Comparison")
        fig_comprehensive = create_metrics_plots(results)
        st.plotly_chart(fig_comprehensive, use_container_width=True)
        
        # Individual metric plots
        st.header("📊 Individual Metric Plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("R² Score Progression")
            fig_r2 = create_individual_plot(results, "r2_scores", "R² Score", 
                                           "R² Score Comparison: FedAvg vs FedProx vs SmartFedProx")
            st.plotly_chart(fig_r2, use_container_width=True)
            
            st.subheader("Average Model Divergence")
            fig_div = create_individual_plot(results, "avg_divergence", "Avg Divergence",
                                            "Average Model Divergence Over Rounds")
            st.plotly_chart(fig_div, use_container_width=True)
        
        with col2:
            st.subheader("MSE Loss Progression")
            fig_mse = create_individual_plot(results, "mse_losses", "MSE Loss",
                                            "MSE Loss Comparison: FedAvg vs FedProx vs SmartFedProx")
            st.plotly_chart(fig_mse, use_container_width=True)
            
            st.subheader("Average Effective μ")
            fig_mu = create_individual_plot(results, "avg_effective_mu", "Avg Effective μ",
                                           "Average Proximal Coefficient (μ) Over Rounds")
            st.plotly_chart(fig_mu, use_container_width=True)
        
        # Detailed metrics table
        st.header("📋 Detailed Metrics by Round")
        
        # Create a combined dataframe
        all_rounds_data = []
        for name, metrics in results.items():
            for i, round_num in enumerate(metrics["rounds"]):
                all_rounds_data.append({
                    "Round": round_num,
                    "Strategy": name,
                    "R² Score": metrics["r2_scores"][i],
                    "MSE Loss": metrics["mse_losses"][i],
                    "Avg Training Loss": metrics["avg_train_loss"][i],
                    "Avg Divergence": metrics["avg_divergence"][i],
                    "Avg Effective μ": metrics["avg_effective_mu"][i]
                })
        
        df_detailed = pd.DataFrame(all_rounds_data)
        st.dataframe(df_detailed, use_container_width=True, hide_index=True)
        
        # Download results
        st.header("💾 Download Results")
        csv = df_detailed.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Metrics (CSV)",
            data=csv,
            file_name=f"fl_results_{int(time.time())}.csv",
            mime="text/csv"
        )
    
    else:
        # Welcome message
        st.info("👈 Configure your simulation parameters in the sidebar and click 'Run Simulation' to start!")
        
        # Show information about strategies
        st.header("📚 About the Strategies")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("FedAvg")
            st.markdown("""
            - **Proximal μ**: 0.0
            - **Selection**: Random
            - **Description**: Baseline federated averaging algorithm
            """)
        
        with col2:
            st.subheader("FedProx")
            st.markdown("""
            - **Proximal μ**: 0.1
            - **Selection**: Random
            - **Description**: FedAvg with proximal term for handling non-IID data
            """)
        
        with col3:
            st.subheader("SmartFedProx")
            st.markdown("""
            - **Proximal μ**: 0.1 (adaptive)
            - **Selection**: Hybrid (divergence-based)
            - **Description**: FedProx with adaptive μ and intelligent client selection
            """)


if __name__ == "__main__":
    main()
