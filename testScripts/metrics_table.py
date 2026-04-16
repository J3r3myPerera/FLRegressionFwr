"""
Per-round metrics table generator for federated learning simulations.

For each strategy, produces a table analogous to the classification-style table
in the project brief, but with regression metrics (R², MSE, RMSE, MAE, Train Loss).

Columns mirror the structure: metric_mean (min-max across trials) per federated round.
"""

import os
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _cell(values: list, fmt: str = ".4f") -> str:
    """Return  'mean (min-max)'  string for one metric across trials."""
    mean = np.mean(values)
    lo   = np.min(values)
    hi   = np.max(values)
    return f"{mean:{fmt}} ({lo:{fmt}}-{hi:{fmt}})"


def _separator(col_widths: list, char: str = "-") -> str:
    return "+" + "+".join(char * (w + 2) for w in col_widths) + "+"


def _row(cells: list, col_widths: list) -> str:
    padded = [f" {str(c).center(w)} " for c, w in zip(cells, col_widths)]
    return "|" + "|".join(padded) + "|"


# ── main table builder ────────────────────────────────────────────────────────

def generate_per_round_table(
    all_trial_results: dict,
    num_rounds: int,
    output_dir: str = "outputs/metrics_tables",
    save_txt: bool = True,
    save_latex: bool = True,
) -> None:
    """
    For every strategy in *all_trial_results*, print and (optionally) save a
    per-round metrics table showing mean (min-max across trials).

    Parameters
    ----------
    all_trial_results : dict
        {strategy_name: [trial_metrics_dict, …]}
        Each trial_metrics_dict must contain lists keyed by:
        'r2_scores', 'mse_losses', 'rmse_scores', 'mae_scores', 'avg_train_loss', 'avg_effective_mu'
    num_rounds : int
        Number of federated rounds per trial.
    output_dir : str
        Directory to save table files (created if missing).
    save_txt : bool
        Write a plain-text .txt file per strategy.
    save_latex : bool
        Write a LaTeX .tex file per strategy.
    """
    os.makedirs(output_dir, exist_ok=True)

    headers = [
        "Round",
        "R² (min-max)",
        "MSE (min-max)",
        "RMSE (min-max)",
        "MAE (min-max)",
        "Train Loss (min-max)",
        "Avg μ (min-max)",
    ]

    for strategy_name, trials in all_trial_results.items():
        rows = []
        for rnd in range(num_rounds):
            r2_vals   = [t["r2_scores"][rnd]       for t in trials]
            mse_vals  = [t["mse_losses"][rnd]      for t in trials]
            rmse_vals = [t["rmse_scores"][rnd]     for t in trials]
            mae_vals  = [t["mae_scores"][rnd]      for t in trials]
            tl_vals   = [t["avg_train_loss"][rnd]  for t in trials]
            mu_vals   = [t["avg_effective_mu"][rnd] for t in trials]

            rows.append([
                str(rnd + 1),
                _cell(r2_vals),
                _cell(mse_vals),
                _cell(rmse_vals),
                _cell(mae_vals),
                _cell(tl_vals),
                _cell(mu_vals),
            ])

        # ── determine column widths ──────────────────────────────────────────
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # ── build ASCII table ────────────────────────────────────────────────
        lines = []
        title = f"Strategy: {strategy_name}  |  Per-Round Metrics (mean, min-max across {len(trials)} trial(s))"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(_separator(col_widths))
        lines.append(_row(headers, col_widths))
        lines.append(_separator(col_widths, "="))
        for row in rows:
            lines.append(_row(row, col_widths))
            lines.append(_separator(col_widths))

        table_str = "\n".join(lines)

        # ── print ────────────────────────────────────────────────────────────
        print("\n" + table_str)

        # ── save .txt ────────────────────────────────────────────────────────
        if save_txt:
            txt_path = os.path.join(output_dir, f"table_{strategy_name}.txt")
            with open(txt_path, "w") as f:
                f.write(table_str + "\n")
            print(f"  ✓ Saved plain-text table → {txt_path}")

        # ── save .tex ────────────────────────────────────────────────────────
        if save_latex:
            tex_path = os.path.join(output_dir, f"table_{strategy_name}.tex")
            _write_latex_table(
                strategy_name, headers, rows, len(trials), tex_path
            )
            print(f"  ✓ Saved LaTeX table     → {tex_path}")


# ── LaTeX helper ──────────────────────────────────────────────────────────────

def _write_latex_table(
    strategy_name: str,
    headers: list,
    rows: list,
    num_trials: int,
    tex_path: str,
) -> None:
    """Write a LaTeX longtable for the given strategy."""
    col_spec = "|c|" + "|".join(["c"] * (len(headers) - 1)) + "|"
    header_row = " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\"

    tex_lines = [
        "% Auto-generated by metrics_table.py",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{Per-Round Regression Metrics — {strategy_name} "
        f"(mean, min--max across {num_trials} trial(s))}}",
        f"\\label{{tab:metrics_{strategy_name.lower().replace(' ', '_')}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header_row,
        "\\hline",
    ]
    for row in rows:
        escaped = [c.replace("-", "--") for c in row]
        tex_lines.append(" & ".join(escaped) + " \\\\")
        tex_lines.append("\\hline")

    tex_lines += [
        "\\end{tabular}",
        "\\end{table}",
    ]

    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")