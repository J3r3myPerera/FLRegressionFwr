import pandas as pd
import numpy as np

SEED = 42
INPUT_PATH = "data/indianPersonalFinanceAndSpendingHabits.csv"
OUTPUT_PATH = "data/indianPersonalFinanceAndSpendingHabits_expanded.csv"
TARGET_ROWS = 100_000

np.random.seed(SEED)

df = pd.read_csv(INPUT_PATH)
original_rows = len(df)
extra_needed = TARGET_ROWS - original_rows
print(f"Original rows: {original_rows}  |  Target: {TARGET_ROWS}  |  Generating: {extra_needed}")

EXPENSE_COLS = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous",
]

RATIO_COLS = [
    "Rent", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Miscellaneous",
]

RATIO_BOUNDS = {
    "Rent":          (0.15, 0.30),
    "Insurance":     (0.02, 0.05),
    "Groceries":     (0.10, 0.15),
    "Transport":     (0.05, 0.08),
    "Eating_Out":    (0.02, 0.05),
    "Entertainment": (0.02, 0.05),
    "Utilities":     (0.04, 0.08),
    "Healthcare":    (0.03, 0.05),
    "Miscellaneous": (0.01, 0.03),
}

LOAN_RATIO_BOUNDS = (0.005, 0.20)
EDU_RATIO_BOUNDS  = (0.001, 0.10)
DSP_BOUNDS        = (5.0, 24.999)


def logit_transform(x, lo, hi, eps=1e-6):
    p = np.clip((x - lo) / (hi - lo), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid_inverse(y, lo, hi):
    return lo + (hi - lo) / (1 + np.exp(-y))


synthetic_parts = []

for (occ, tier), group in df.groupby(["Occupation", "City_Tier"]):
    group = group.reset_index(drop=True)
    n_new = round(extra_needed * len(group) / original_rows)
    if n_new <= 0:
        continue

    # Sample log(Income), expense ratios, and DSP jointly — Income and DSP
    # are strongly correlated (r≈0.78) so joint sampling preserves the tails.
    log_income = np.log(group["Income"].values)
    ratio_data = {
        col: logit_transform((group[col] / group["Income"]).values, *RATIO_BOUNDS[col])
        for col in RATIO_COLS
    }
    dsp_logit = logit_transform(group["Desired_Savings_Percentage"].values, *DSP_BOUNDS)

    all_joint = np.column_stack([log_income] + list(ratio_data.values()) + [dsp_logit])
    mu  = all_joint.mean(axis=0)
    cov = np.cov(all_joint.T) + np.eye(all_joint.shape[1]) * 1e-8

    samples = np.random.multivariate_normal(mu, cov, size=n_new)

    income_samples = np.exp(samples[:, 0])
    ratio_samples  = {
        col: sigmoid_inverse(samples[:, 1 + i], *RATIO_BOUNDS[col])
        for i, col in enumerate(RATIO_COLS)
    }
    dsp_samples = np.clip(sigmoid_inverse(samples[:, 1 + len(RATIO_COLS)], *DSP_BOUNDS), *DSP_BOUNDS)

    age_new = np.clip(
        np.random.choice(group["Age"].values, n_new, replace=True) + np.random.randint(-2, 3, n_new),
        18, 64,
    ).astype(int)

    dep_new = np.clip(
        np.random.choice(group["Dependents"].values, n_new, replace=True), 0, 4
    ).astype(int)

    # Loan_Repayment: ~60% zero in original data
    has_loan = np.random.random(n_new) < (group["Loan_Repayment"] > 0).mean()
    loan_amounts = np.zeros(n_new)
    if has_loan.any():
        pos_ratios = (group.loc[group["Loan_Repayment"] > 0, "Loan_Repayment"]
                      / group.loc[group["Loan_Repayment"] > 0, "Income"]).values
        lr = logit_transform(pos_ratios, *LOAN_RATIO_BOUNDS)
        loan_amounts[has_loan] = sigmoid_inverse(
            np.random.normal(lr.mean(), lr.std(), has_loan.sum()), *LOAN_RATIO_BOUNDS
        ) * income_samples[has_loan]

    # Education is exactly 0 when Dependents == 0
    edu_amounts = np.zeros(n_new)
    has_dep = dep_new > 0
    if has_dep.any():
        pos_edu = group.loc[group["Dependents"] > 0]
        er = logit_transform((pos_edu["Education"] / pos_edu["Income"]).values, *EDU_RATIO_BOUNDS)
        edu_amounts[has_dep] = sigmoid_inverse(
            np.random.normal(er.mean(), er.std(), has_dep.sum()), *EDU_RATIO_BOUNDS
        ) * income_samples[has_dep]

    part = pd.DataFrame({
        "Income":      income_samples,
        "Age":         age_new,
        "Dependents":  dep_new,
        "Occupation":  occ,
        "City_Tier":   tier,
        **{col: ratio_samples[col] * income_samples for col in RATIO_COLS},
        "Loan_Repayment": loan_amounts,
        "Education":      edu_amounts,
        "Desired_Savings_Percentage": dsp_samples,
    })

    part["Disposable_Income"] = part["Income"] - part[EXPENSE_COLS].sum(axis=1)
    part["Desired_Savings"] = np.minimum(
        part["Income"] * part["Desired_Savings_Percentage"] / 100,
        part["Disposable_Income"].clip(lower=0),
    )

    for col in ["Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", "Miscellaneous"]:
        part[f"Potential_Savings_{col}"] = np.random.uniform(0.05, 0.30, n_new) * part[col]
    for col in ["Healthcare", "Education"]:
        part[f"Potential_Savings_{col}"] = np.random.uniform(0.00, 0.05, n_new) * part[col]

    synthetic_parts.append(part)

synthetic_df = pd.concat(synthetic_parts, ignore_index=True)[df.columns.tolist()]
combined = pd.concat([df, synthetic_df], ignore_index=True)
combined.to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. Total rows: {len(combined)}  →  {OUTPUT_PATH}")

print(f"\n{'Column':<37} {'Original':>12} {'Synthetic':>12}  {'Δ%':>6}")
print("-" * 70)
for col in df.select_dtypes(include="number").columns:
    om, sm = df[col].mean(), synthetic_df[col].mean()
    pct = abs(om - sm) / (abs(om) + 1e-9) * 100
    print(f"{col:<37} {om:>12.2f} {sm:>12.2f}  {pct:>5.1f}%{'  !' if pct > 10 else ''}")
