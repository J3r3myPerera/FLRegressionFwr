# Standalone script to visualize the data distribution across simulated FL clients.

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Add FLRegression to path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "FLRegression"))

# Reuse project config
from module import NUM_CLIENTS, NUM_ROUNDS, FRACTION_FIT, LOCAL_EPOCHS, LEARNING_RATE, BATCH_SIZE
from dataset import (
    _load_and_preprocess_data,
    _get_data_path,
    PARTITION_COLUMN,
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "dataset_stats"

# Partition index extraction (mirrors dataset.load_data but returns indices)

def get_partition_indices(partition_id: int, num_partitions: int) -> np.ndarray:
    """Return the row indices assigned to a given client."""
    data_cache, _ = _load_and_preprocess_data()
    X = data_cache["X"]
    combined_keys = data_cache["combined_keys"]
    unique_keys = data_cache["unique_keys"]
    incomes = data_cache["incomes"]

    np.random.seed(42 + partition_id)

    num_keys = len(unique_keys)
    primary_key_idx = partition_id % num_keys
    secondary_key_idx = (partition_id + num_keys // 2) % num_keys

    primary_key = unique_keys[primary_key_idx]
    secondary_key = unique_keys[secondary_key_idx]

    primary_indices = np.where(combined_keys == primary_key)[0]
    secondary_indices = np.where(combined_keys == secondary_key)[0]

    income_percentile = np.percentile(incomes, [25, 75])
    if partition_id % 2 == 0:
        income_mask_primary = incomes[primary_indices] < income_percentile[1]
        income_mask_secondary = incomes[secondary_indices] < income_percentile[1]
    else:
        income_mask_primary = incomes[primary_indices] > income_percentile[0]
        income_mask_secondary = incomes[secondary_indices] > income_percentile[0]

    if income_mask_primary.sum() > len(primary_indices) * 0.3:
        primary_indices = primary_indices[income_mask_primary]
    if income_mask_secondary.sum() > len(secondary_indices) * 0.3:
        secondary_indices = secondary_indices[income_mask_secondary]

    quantity_factor = 0.5 + (partition_id % 5) * 0.2
    n_primary = min(len(primary_indices), int(800 * quantity_factor * 0.7))
    n_secondary = min(len(secondary_indices), int(800 * quantity_factor * 0.3))

    if len(primary_indices) > n_primary:
        np.random.shuffle(primary_indices)
        primary_indices = primary_indices[:n_primary]
    if len(secondary_indices) > n_secondary:
        np.random.shuffle(secondary_indices)
        secondary_indices = secondary_indices[:n_secondary]

    partition_indices = np.concatenate([primary_indices, secondary_indices])
    np.random.shuffle(partition_indices)

    if len(partition_indices) < 50:
        all_indices = np.arange(len(X))
        remaining = np.setdiff1d(all_indices, partition_indices)
        extra = np.random.choice(remaining, min(100, len(remaining)), replace=False)
        partition_indices = np.concatenate([partition_indices, extra])

    return partition_indices


def build_client_dataframes(df: pd.DataFrame, num_clients: int) -> dict:
    """Return {client_id: DataFrame_subset} for every client."""
    return {
        cid: df.iloc[get_partition_indices(cid, num_clients)]
        for cid in range(num_clients)
    }

# outputs

def _save(fig, name: str):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def plot_samples_per_client(client_dfs: dict):
    counts = {f"Client {k}": len(v) for k, v in client_dfs.items()}
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.keys(), counts.values(), color=plt.cm.tab10.colors[:len(counts)])
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Count per Client", fontsize=14, fontweight="bold")
    ax.set_xlabel("Client")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    _save(fig, "fig_samples_per_client")


def plot_income_per_client(client_dfs: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [cdf["Income"].values for cdf in client_dfs.values()]
    labels = [f"Client {k}" for k in client_dfs]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    colors = plt.cm.tab10.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor((*color[:3], 0.6))
    ax.set_ylabel("Income")
    ax.set_title("Income Distribution per Client", fontsize=14, fontweight="bold")
    ax.set_xlabel("Client")
    fig.tight_layout()
    _save(fig, "fig_income_per_client")


def plot_target_per_client(client_dfs: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [cdf[TARGET_COLUMN].values for cdf in client_dfs.values()]
    labels = [f"Client {k}" for k in client_dfs]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    colors = plt.cm.tab10.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor((*color[:3], 0.6))
    ax.set_ylabel(TARGET_COLUMN.replace("_", " "))
    ax.set_title(f"{TARGET_COLUMN.replace('_', ' ')} Distribution per Client", fontsize=14, fontweight="bold")
    ax.set_xlabel("Client")
    fig.tight_layout()
    _save(fig, "fig_target_per_client")


def plot_noniid_heatmap(client_dfs: dict):
    """Heatmap: rows = clients, columns = Occupation × City_Tier groups."""
    # Build proportion matrix
    groups = sorted({
        f"{row['Occupation']}_{row['City_Tier']}"
        for cdf in client_dfs.values()
        for _, row in cdf.iterrows()
    })
    matrix = np.zeros((len(client_dfs), len(groups)))
    for cid, cdf in client_dfs.items():
        total = len(cdf)
        for _, row in cdf.iterrows():
            g = f"{row['Occupation']}_{row['City_Tier']}"
            matrix[cid, groups.index(g)] += 1
        matrix[cid] /= max(total, 1)

    fig, ax = plt.subplots(figsize=(max(12, len(groups) * 0.9), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(client_dfs)))
    ax.set_yticklabels([f"Client {k}" for k in client_dfs])
    ax.set_title("Non-IID Heatmap (Occupation × City Tier proportion per Client)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Proportion")
    fig.tight_layout()
    _save(fig, "fig_noniid_heatmap")


def plot_table_per_client(client_dfs: dict, df_full: pd.DataFrame):
    """Render a per-client summary as a figure table."""
    rows = []
    for cid, cdf in client_dfs.items():
        rows.append({
            "Client": cid,
            "Samples": len(cdf),
            "Mean Income": f"{cdf['Income'].mean():,.0f}",
            "Med Income": f"{cdf['Income'].median():,.0f}",
            f"Mean {TARGET_COLUMN}": f"{cdf[TARGET_COLUMN].mean():,.0f}",
            "Top Occupation": cdf["Occupation"].mode().iloc[0] if len(cdf) else "-",
            "Top City Tier": cdf["City_Tier"].mode().iloc[0] if len(cdf) else "-",
        })
    tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 0.4 * len(tbl) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Per-Client Data Summary", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, "fig_table_per_client")


def plot_table_summary(df_full: pd.DataFrame):
    """Render overall dataset summary stats as a figure table."""
    num_cols = [c for c in NUMERICAL_COLUMNS if c in df_full.columns][:10]
    desc = df_full[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc = desc.round(2)
    desc.columns = ["Count", "Mean", "Std", "Min", "Median", "Max"]

    fig, ax = plt.subplots(figsize=(14, 0.4 * len(desc) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=desc.values,
        rowLabels=desc.index,
        colLabels=desc.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Dataset Feature Summary (Top 10 Numerical)", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, "fig_table_summary")

# LaTeX tables

def _save_tex(content: str, name: str):
    path = OUTPUT_DIR / f"{name}.tex"
    path.write_text(content)
    print(f"  Saved {name}.tex")


def tex_city_tier(df: pd.DataFrame):
    ct = df["City_Tier"].value_counts().sort_index()
    lines = [
        r"\begin{tabular}{lrr}", r"\hline",
        r"City Tier & Count & \% \\", r"\hline",
    ]
    for tier, count in ct.items():
        lines.append(f"{tier} & {count} & {count / len(df) * 100:.1f}\\% \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    _save_tex("\n".join(lines), "table_city_tier")


def tex_occupation(df: pd.DataFrame):
    occ = df["Occupation"].value_counts().sort_index()
    lines = [
        r"\begin{tabular}{lrr}", r"\hline",
        r"Occupation & Count & \% \\", r"\hline",
    ]
    for name, count in occ.items():
        lines.append(f"{name} & {count} & {count / len(df) * 100:.1f}\\% \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    _save_tex("\n".join(lines), "table_occupation")


def tex_feature_summary(df: pd.DataFrame):
    num_cols = [c for c in NUMERICAL_COLUMNS if c in df.columns]
    desc = df[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
    lines = [
        r"\begin{tabular}{lrrrrrr}", r"\hline",
        r"Feature & Count & Mean & Std & Min & Median & Max \\", r"\hline",
    ]
    for feat, row in desc.iterrows():
        vals = " & ".join(f"{v:,.2f}" for v in row)
        lines.append(f"{feat} & {vals} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    _save_tex("\n".join(lines), "table_feature_summary")


def tex_fl_config():
    lines = [
        r"\begin{tabular}{lr}", r"\hline",
        r"Parameter & Value \\", r"\hline",
        f"Num Clients & {NUM_CLIENTS} \\\\",
        f"Num Rounds & {NUM_ROUNDS} \\\\",
        f"Fraction Fit & {FRACTION_FIT} \\\\",
        f"Local Epochs & {LOCAL_EPOCHS} \\\\",
        f"Learning Rate & {LEARNING_RATE} \\\\",
        f"Batch Size & {BATCH_SIZE} \\\\",
        f"Partition Column & {PARTITION_COLUMN} \\\\",
        f"Target Column & {TARGET_COLUMN} \\\\",
        r"\hline", r"\end{tabular}",
    ]
    _save_tex("\n".join(lines), "table_fl_config")


def tex_overall_stats(df: pd.DataFrame):
    lines = [
        r"\begin{tabular}{lr}", r"\hline",
        r"Statistic & Value \\", r"\hline",
        f"Total Samples & {len(df):,} \\\\",
        f"Features & {len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS)} \\\\",
        f"Numerical Features & {len(NUMERICAL_COLUMNS)} \\\\",
        f"Categorical Features & {len(CATEGORICAL_COLUMNS)} \\\\",
        f"Target & {TARGET_COLUMN} \\\\",
        f"City Tiers & {df['City_Tier'].nunique()} \\\\",
        f"Occupations & {df['Occupation'].nunique()} \\\\",
        r"\hline", r"\end{tabular}",
    ]
    _save_tex("\n".join(lines), "table_overall_stats")


def tex_per_client(client_dfs: dict):
    lines = [
        r"\begin{tabular}{rrrrrll}", r"\hline",
        r"Client & Samples & Mean Inc. & Med Inc. & Mean Target & Top Occ. & Top Tier \\",
        r"\hline",
    ]
    for cid, cdf in client_dfs.items():
        top_occ = cdf["Occupation"].mode().iloc[0] if len(cdf) else "-"
        top_tier = cdf["City_Tier"].mode().iloc[0] if len(cdf) else "-"
        lines.append(
            f"{cid} & {len(cdf)} & {cdf['Income'].mean():,.0f} & "
            f"{cdf['Income'].median():,.0f} & {cdf[TARGET_COLUMN].mean():,.0f} & "
            f"{top_occ} & {top_tier} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    _save_tex("\n".join(lines), "table_per_client")

# Main

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(_get_data_path())

    print(f"Building partitions for {NUM_CLIENTS} clients...")
    client_dfs = build_client_dataframes(df, NUM_CLIENTS)

    print("\nGenerating figures...")
    plot_samples_per_client(client_dfs)
    plot_income_per_client(client_dfs)
    plot_target_per_client(client_dfs)
    plot_noniid_heatmap(client_dfs)
    plot_table_per_client(client_dfs, df)
    plot_table_summary(df)

    print("\nGenerating LaTeX tables...")
    tex_city_tier(df)
    tex_occupation(df)
    tex_feature_summary(df)
    tex_fl_config()
    tex_overall_stats(df)
    tex_per_client(client_dfs)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()