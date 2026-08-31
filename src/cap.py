import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CSV_PATH = ""

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

BATCH_SIZE = 16
MAX_LENGTH = 512

df = pd.read_csv(CSV_PATH)

required_cols = ["premise", "hypothesis", "label_semantic"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
model.eval()

id2label = model.config.id2label
label_names = {v.lower(): k for k, v in id2label.items()}

print("Model labels:", id2label)

def find_label_id(name):
    name = name.lower()
    for i, label in id2label.items():
        if name in label.lower():
            return i
    raise ValueError(f"Could not find label id for {name}. Labels: {id2label}")

entailment_id = find_label_id("entail")
neutral_id = find_label_id("neutral")
contradiction_id = find_label_id("contrad")

all_results = []

for start in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = df.iloc[start:start+BATCH_SIZE]

    enc = tokenizer(
        batch["premise"].astype(str).tolist(),
        batch["hypothesis"].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits.float()
        probs = F.softmax(logits, dim=-1).cpu()

    for row_idx, prob in zip(batch.index, probs):
        entailment = float(prob[entailment_id])
        neutral = float(prob[neutral_id])
        contradiction = float(prob[contradiction_id])

        formula = min(1, entailment + 0.3*neutral)

        all_results.append({
            **df.loc[row_idx].to_dict(),
            "cap_definition1": formula,
            "p_entailment": entailment,
            "p_neutral": neutral,
            "p_contradiction": contradiction,
            "predicted_nli_label": id2label[int(prob.argmax())],
        })

results_df = pd.DataFrame(all_results)

import pandas as pd

prob_cols = ["p_entailment", "p_neutral", "p_contradiction", "cap_definition1"]
meta_cols = [
    "question", "dataset", "premise", "hypothesis",
    "label_semantic"] #"label_linguistic"

combined = pd.concat(
    [results_df, results_df],
    ignore_index=True
)

final_df = (
    combined
    .groupby(meta_cols, as_index=False)[prob_cols]
    .mean()
)

final_df["predicted_nli_label"] = (
    final_df[prob_cols]
    .idxmax(axis=1)
    .str.replace("p_", "", regex=False)
)

summary_df = (
    final_df
    .groupby("label_semantic")
    .agg(
        avg_cap =("cap_definition1", "mean"),
        min_cap =("cap_definition1", "min"),
        max_cap=("cap_definition1", "max"),
        count=("hypothesis" ,"count"),
    )
    .sort_values("avg_cap", ascending=False)
    .reset_index()
)

# display(final_df)
display(summary_df)



import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import spearmanr, kendalltau
import numpy as np

CSV_PATH = ""

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

BATCH_SIZE = 16
MAX_LENGTH = 512

df = pd.read_csv(CSV_PATH)

required_cols = ["premise", "hypothesis", "label_semantic"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

model.eval()

id2label = model.config.id2label
print("Model labels:", id2label)

def find_label_id(name):
    name = name.lower()
    for i, label in id2label.items():
        if name in label.lower():
            return i
    raise ValueError(f"Could not find label id for {name}. Labels: {id2label}")

entailment_id = find_label_id("entail")
neutral_id = find_label_id("neutral")
contradiction_id = find_label_id("contrad")

def run_nli_scores(left_texts, right_texts, prefix):
    """
    Computes NLI probabilities for left_text -> right_text.

    For XNLI-style models:
    left_text = premise
    right_text = hypothesis
    """
    all_scores = []

    for start in tqdm(range(0, len(left_texts), BATCH_SIZE), desc=f"NLI {prefix}"):
        left_batch = left_texts[start:start+BATCH_SIZE]
        right_batch = right_texts[start:start+BATCH_SIZE]

        enc = tokenizer(
            left_batch,
            right_batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits.float()
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        for prob in probs:
            entailment = float(prob[entailment_id])
            neutral = float(prob[neutral_id])
            contradiction = float(prob[contradiction_id])

            directional_cap = entailment + 0.3 * neutral

            all_scores.append({
                f"p_entailment_{prefix}": entailment,
                f"p_neutral_{prefix}": neutral,
                f"p_contradiction_{prefix}": contradiction,
                f"directional_cap_{prefix}": directional_cap,
                f"predicted_nli_label_{prefix}": id2label[int(prob.argmax())],
            })

    return pd.DataFrame(all_scores)


premises = df["premise"].astype(str).tolist()
hypotheses = df["hypothesis"].astype(str).tolist()

# Forward: premise -> hypothesis
forward_scores = run_nli_scores(
    premises,
    hypotheses,
    prefix="premise_to_hypothesis"
)

# Reverse: hypothesis -> premise
reverse_scores = run_nli_scores(
    hypotheses,
    premises,
    prefix="hypothesis_to_premise"
)

results_df = pd.concat(
    [
        df.reset_index(drop=True),
        forward_scores.reset_index(drop=True),
        reverse_scores.reset_index(drop=True)
    ],
    axis=1
)

results_df["cap_definition1"] = (
    0.85 * (
        results_df["p_entailment_premise_to_hypothesis"]
        + 0.3 * results_df["p_neutral_premise_to_hypothesis"]
    )
    + 0.15 * (
          results_df["p_entailment_hypothesis_to_premise"]
         + 0.3 * results_df["p_neutral_hypothesis_to_premise"]
    )
)


summary_df = (
    results_df
    .groupby("label_semantic")
    .agg(
        avg_cap=("cap_definition1", "mean"),
        std_cap=("cap_definition1", "std"),
        min_cap=("cap_definition1", "min"),
        max_cap=("cap_definition1", "max"),
        count=("hypothesis", "count"),
    )
    .sort_values("avg_cap", ascending=False)
    .reset_index()
)

display(summary_df)

# -----------------------------
# Human ordinal severity mapping
# -----------------------------

severity_map = {
    "exact": 7,
    "equivalent": 6,
    "alternative_correct": 6,
    "overinclusive_valid": 5,
    "partial": 4,
    "overinclusive_invalid": 2,
    "invalid": 1,
    "contradictory": 0,
}

results_df["label_semantic_clean"] = (
    results_df["label_semantic"]
    .astype(str)
    .str.strip()
    .str.lower()
)

results_df["human_ordinal"] = results_df["label_semantic_clean"].map(severity_map)

missing_labels = sorted(
    results_df.loc[results_df["human_ordinal"].isna(), "label_semantic"]
    .astype(str)
    .unique()
)

if missing_labels:
    raise ValueError(
        f"These label_semantic values are not in severity_map: {missing_labels}"
    )


# -----------------------------
# Correlation tests
# -----------------------------

def correlation_report(df, score_col, human_col="human_ordinal"):
    temp = df[[score_col, human_col]].dropna()

    if len(temp) < 3:
        return {
            "metric": score_col,
            "n": len(temp),
            "spearman": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
        }

    sp = spearmanr(temp[score_col], temp[human_col])
    kt = kendalltau(temp[score_col], temp[human_col])

    return {
        "metric": score_col,
        "n": len(temp),
        "spearman": sp.statistic,
        "spearman_p": sp.pvalue,
        "kendall_tau": kt.statistic,
        "kendall_p": kt.pvalue,
    }


metric_cols = ["cap_definition1"]

correlation_df = pd.DataFrame([
    correlation_report(results_df, col)
    for col in metric_cols
])

correlation_df = correlation_df.sort_values("spearman", ascending=False)

display(correlation_df)

# -----------------------------
# Pairwise ranking accuracy
# -----------------------------

def pairwise_ranking_accuracy(
    df,
    score_col="cap_definition1",
    label_col="label_semantic_clean",
    ordinal_col="human_ordinal",
    label_a=None,
    label_b=None,
):
    """
    For every pair with different human ordinal scores:
    correct if the example with higher human ordinal also has higher CAP.

    Ties in CAP are reported separately and counted as incorrect in accuracy.
    """

    temp = df[[score_col, label_col, ordinal_col]].dropna().copy()

    if label_a is not None and label_b is not None:
        allowed = {label_a, label_b}
        temp = temp[temp[label_col].isin(allowed)]

    scores = temp[score_col].to_numpy()
    ordinals = temp[ordinal_col].to_numpy()

    total = 0
    correct = 0
    incorrect = 0
    cap_ties = 0

    n = len(temp)

    for i in range(n):
        for j in range(i + 1, n):
            if ordinals[i] == ordinals[j]:
                continue

            total += 1

            human_order = np.sign(ordinals[i] - ordinals[j])
            cap_order = np.sign(scores[i] - scores[j])

            if cap_order == 0:
                cap_ties += 1
            elif cap_order == human_order:
                correct += 1
            else:
                incorrect += 1

    accuracy = correct / total if total > 0 else np.nan

    return {
        "comparison": (
            "all_different_label_pairs"
            if label_a is None
            else f"{label_a} vs {label_b}"
        ),
        "n_examples": n,
        "comparable_pairs": total,
        "correct_ordered_pairs": correct,
        "incorrect_ordered_pairs": incorrect,
        "cap_ties": cap_ties,
        "accuracy": accuracy,
    }


overall_pairwise = pairwise_ranking_accuracy(results_df)

hard_pairs = [
    ("equivalent", "partial"),
    ("partial", "overinclusive_valid"),
    ("overinclusive_valid", "overinclusive_invalid"),
    ("alternative_correct", "invalid"),
]

pairwise_rows = [overall_pairwise]

for a, b in hard_pairs:
    pairwise_rows.append(
        pairwise_ranking_accuracy(
            results_df,
            label_a=a,
            label_b=b,
        )
    )

pairwise_df = pd.DataFrame(pairwise_rows)

display(pairwise_df)

# -----------------------------
# Pairwise ranking comparison across metrics
# -----------------------------

def pairwise_ranking_accuracy_for_metric(
    df,
    score_col,
    label_col="label_semantic_clean",
    ordinal_col="human_ordinal",
):
    temp = df[[score_col, label_col, ordinal_col]].dropna().copy()

    scores = temp[score_col].to_numpy()
    ordinals = temp[ordinal_col].to_numpy()

    total = 0
    correct = 0
    incorrect = 0
    ties = 0

    n = len(temp)

    for i in range(n):
        for j in range(i + 1, n):
            if ordinals[i] == ordinals[j]:
                continue

            total += 1

            human_order = np.sign(ordinals[i] - ordinals[j])
            metric_order = np.sign(scores[i] - scores[j])

            if metric_order == 0:
                ties += 1
            elif metric_order == human_order:
                correct += 1
            else:
                incorrect += 1

    return {
        "metric": score_col,
        "n_examples": n,
        "comparable_pairs": total,
        "correct_ordered_pairs": correct,
        "incorrect_ordered_pairs": incorrect,
        "ties": ties,
        "pairwise_accuracy": correct / total if total > 0 else np.nan,
    }


ranking_metric_cols = ["cap_definition1"] + existing_metric_cols

metric_pairwise_df = pd.DataFrame([
    pairwise_ranking_accuracy_for_metric(results_df, col)
    for col in ranking_metric_cols
])

metric_pairwise_df = metric_pairwise_df.sort_values(
    "pairwise_accuracy",
    ascending=False
)

display(metric_pairwise_df)

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kendalltau
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# -----------------------------
# Bootstrap helpers
# -----------------------------

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
CI_LOW = 2.5
CI_HIGH = 97.5

rng = np.random.default_rng(BOOTSTRAP_SEED)


def bootstrap_ci(
    values,
    stat_fn,
    n_boot=BOOTSTRAP_N,
    ci_low=CI_LOW,
    ci_high=CI_HIGH,
    seed=BOOTSTRAP_SEED,
    desc=None,
):
    """
    Generic nonparametric bootstrap CI.

    values can be:
    - a DataFrame
    - a NumPy array
    - a list

    stat_fn receives the bootstrapped sample.
    """

    local_rng = np.random.default_rng(seed)

    if isinstance(values, pd.DataFrame):
        n = len(values)
    else:
        values = np.asarray(values)
        n = len(values)

    if n == 0:
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": 0,
        }

    try:
        estimate = stat_fn(values)
    except Exception:
        estimate = np.nan

    boot_stats = []

    iterator = range(n_boot)
    if desc:
        iterator = tqdm(iterator, desc=desc)

    for _ in iterator:
        idx = local_rng.integers(0, n, size=n)

        if isinstance(values, pd.DataFrame):
            sample = values.iloc[idx].reset_index(drop=True)
        else:
            sample = values[idx]

        try:
            stat = stat_fn(sample)
            if pd.notna(stat):
                boot_stats.append(stat)
        except Exception:
            continue

    if len(boot_stats) == 0:
        return {
            "estimate": estimate,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": n,
        }

    boot_stats = np.asarray(boot_stats)

    return {
        "estimate": estimate,
        "ci_low": np.percentile(boot_stats, ci_low),
        "ci_high": np.percentile(boot_stats, ci_high),
        "n": n,
        "valid_bootstraps": len(boot_stats),
    }


def format_ci(row, estimate_col="estimate", low_col="ci_low", high_col="ci_high"):
    return f"{row[estimate_col]:.4f} [{row[low_col]:.4f}, {row[high_col]:.4f}]"

# -----------------------------
# Bootstrap CI: Spearman / Kendall
# -----------------------------

def spearman_stat(sample_df, score_col="cap_definition1", human_col="human_ordinal"):
    temp = sample_df[[score_col, human_col]].dropna()

    if len(temp) < 3:
        return np.nan

    if temp[score_col].nunique() < 2 or temp[human_col].nunique() < 2:
        return np.nan

    return spearmanr(temp[score_col], temp[human_col]).statistic


def kendall_stat(sample_df, score_col="cap_definition1", human_col="human_ordinal"):
    temp = sample_df[[score_col, human_col]].dropna()

    if len(temp) < 3:
        return np.nan

    if temp[score_col].nunique() < 2 or temp[human_col].nunique() < 2:
        return np.nan

    return kendalltau(temp[score_col], temp[human_col]).statistic


corr_ci_rows = []

all_metric_cols = ["cap_definition1"] + existing_metric_cols

for metric in all_metric_cols:
    metric_df = results_df[[metric, "human_ordinal"]].dropna().copy()

    sp_ci = bootstrap_ci(
        metric_df,
        stat_fn=lambda x, m=metric: spearman_stat(x, score_col=m),
        desc=f"Bootstrap Spearman: {metric}",
        seed=BOOTSTRAP_SEED + 10,
    )

    kt_ci = bootstrap_ci(
        metric_df,
        stat_fn=lambda x, m=metric: kendall_stat(x, score_col=m),
        desc=f"Bootstrap Kendall: {metric}",
        seed=BOOTSTRAP_SEED + 11,
    )

    corr_ci_rows.append({
        "metric": metric,
        "n": len(metric_df),

        "spearman": sp_ci["estimate"],
        "spearman_ci_low": sp_ci["ci_low"],
        "spearman_ci_high": sp_ci["ci_high"],

        "kendall_tau": kt_ci["estimate"],
        "kendall_ci_low": kt_ci["ci_low"],
        "kendall_ci_high": kt_ci["ci_high"],
    })

correlation_ci_df = pd.DataFrame(corr_ci_rows)

correlation_ci_df["spearman_95ci"] = correlation_ci_df.apply(
    lambda r: f'{r["spearman"]:.4f} [{r["spearman_ci_low"]:.4f}, {r["spearman_ci_high"]:.4f}]',
    axis=1
)

correlation_ci_df["kendall_95ci"] = correlation_ci_df.apply(
    lambda r: f'{r["kendall_tau"]:.4f} [{r["kendall_ci_low"]:.4f}, {r["kendall_ci_high"]:.4f}]',
    axis=1
)

correlation_ci_df = correlation_ci_df.sort_values("spearman", ascending=False)

display(correlation_ci_df)

# -----------------------------
# Bootstrap CI: pairwise ranking accuracy
# -----------------------------

def pairwise_accuracy_stat(
    sample_df,
    score_col="cap_definition1",
    ordinal_col="human_ordinal",
):
    temp = sample_df[[score_col, ordinal_col]].dropna().reset_index(drop=True)

    scores = temp[score_col].to_numpy()
    ordinals = temp[ordinal_col].to_numpy()

    total = 0
    correct = 0

    n = len(temp)

    for i in range(n):
        for j in range(i + 1, n):
            if ordinals[i] == ordinals[j]:
                continue

            human_order = np.sign(ordinals[i] - ordinals[j])
            score_order = np.sign(scores[i] - scores[j])

            if score_order == human_order:
                correct += 1

            total += 1

    if total == 0:
        return np.nan

    return correct / total


pairwise_ci_rows = []

for metric in all_metric_cols:
    metric_df = results_df[[metric, "human_ordinal"]].dropna().copy()

    ci = bootstrap_ci(
        metric_df,
        stat_fn=lambda x, m=metric: pairwise_accuracy_stat(
            x,
            score_col=m,
            ordinal_col="human_ordinal",
        ),
        desc=f"Bootstrap pairwise accuracy: {metric}",
        seed=BOOTSTRAP_SEED + 20,
    )

    pairwise_ci_rows.append({
        "metric": metric,
        "n": len(metric_df),
        "pairwise_accuracy": ci["estimate"],
        "pairwise_accuracy_ci_low": ci["ci_low"],
        "pairwise_accuracy_ci_high": ci["ci_high"],
    })

pairwise_accuracy_ci_df = pd.DataFrame(pairwise_ci_rows)

pairwise_accuracy_ci_df["pairwise_accuracy_95ci"] = pairwise_accuracy_ci_df.apply(
    lambda r: f'{r["pairwise_accuracy"]:.4f} [{r["pairwise_accuracy_ci_low"]:.4f}, {r["pairwise_accuracy_ci_high"]:.4f}]',
    axis=1
)

pairwise_accuracy_ci_df = pairwise_accuracy_ci_df.sort_values(
    "pairwise_accuracy",
    ascending=False
)

display(pairwise_accuracy_ci_df)

# -----------------------------
# Bootstrap CI: hard neighboring pairs
# -----------------------------

hard_pairs = [
    ("equivalent", "partial"),
    ("partial", "overinclusive_valid"),
    ("overinclusive_valid", "overinclusive_invalid"),
    ("alternative_correct", "invalid"),
]


def pairwise_accuracy_for_label_pair_stat(
    sample_df,
    higher_label,
    lower_label,
    score_col="cap_definition1",
    label_col="label_semantic_clean",
):
    temp = sample_df[
        sample_df[label_col].isin([higher_label, lower_label])
    ][[score_col, label_col]].dropna().reset_index(drop=True)

    higher_scores = temp.loc[temp[label_col] == higher_label, score_col].to_numpy()
    lower_scores = temp.loc[temp[label_col] == lower_label, score_col].to_numpy()

    if len(higher_scores) == 0 or len(lower_scores) == 0:
        return np.nan

    total = 0
    correct = 0

    for hs in higher_scores:
        for ls in lower_scores:
            total += 1
            if hs > ls:
                correct += 1

    if total == 0:
        return np.nan

    return correct / total


hard_pair_ci_rows = []

for metric in all_metric_cols:
    for higher_label, lower_label in hard_pairs:
        pair_df = results_df[
            results_df["label_semantic_clean"].isin([higher_label, lower_label])
        ][[metric, "label_semantic_clean"]].dropna().copy()

        ci = bootstrap_ci(
            pair_df,
            stat_fn=lambda x, h=higher_label, l=lower_label, m=metric:
                pairwise_accuracy_for_label_pair_stat(
                    x,
                    higher_label=h,
                    lower_label=l,
                    score_col=m,
                ),
            desc=f"Bootstrap hard pair: {metric}, {higher_label} > {lower_label}",
            seed=BOOTSTRAP_SEED + 30,
        )

        hard_pair_ci_rows.append({
            "metric": metric,
            "comparison": f"{higher_label} > {lower_label}",
            "n": len(pair_df),
            "pairwise_accuracy": ci["estimate"],
            "pairwise_accuracy_ci_low": ci["ci_low"],
            "pairwise_accuracy_ci_high": ci["ci_high"],
        })

hard_pairwise_ci_df = pd.DataFrame(hard_pair_ci_rows)

hard_pairwise_ci_df["pairwise_accuracy_95ci"] = hard_pairwise_ci_df.apply(
    lambda r: f'{r["pairwise_accuracy"]:.4f} [{r["pairwise_accuracy_ci_low"]:.4f}, {r["pairwise_accuracy_ci_high"]:.4f}]',
    axis=1
)

hard_pairwise_ci_df = hard_pairwise_ci_df.sort_values(
    ["comparison", "pairwise_accuracy"],
    ascending=[True, False]
)

display(hard_pairwise_ci_df)

# -----------------------------
# Pairwise AUC between classes
# -----------------------------

ordered_labels = [
    "exact",
    "equivalent",
    "alternative_correct",
    "overinclusive_valid",
    "partial",
    "overinclusive_invalid",
    "invalid",
    "contradictory",
]

label_pairs = []

for a in ordered_labels:
    for b in ordered_labels:
        if a == b:
            continue

        if severity_map[a] > severity_map[b]:
            label_pairs.append((a, b))


def pairwise_auc_between_classes(
    df,
    higher_label,
    lower_label,
    score_col="cap_definition1",
    label_col="label_semantic_clean",
):
    temp = df[
        df[label_col].isin([higher_label, lower_label])
    ][[score_col, label_col]].dropna().copy()

    higher_scores = temp.loc[temp[label_col] == higher_label, score_col].to_numpy()
    lower_scores = temp.loc[temp[label_col] == lower_label, score_col].to_numpy()

    if len(higher_scores) == 0 or len(lower_scores) == 0:
        return np.nan

    total = 0
    wins = 0
    ties = 0

    for hs in higher_scores:
        for ls in lower_scores:
            total += 1

            if hs > ls:
                wins += 1
            elif hs == ls:
                ties += 1

    if total == 0:
        return np.nan

    return (wins + 0.5 * ties) / total


pairwise_auc_rows = []

for metric in all_metric_cols:
    for higher_label, lower_label in label_pairs:
        auc = pairwise_auc_between_classes(
            results_df,
            higher_label=higher_label,
            lower_label=lower_label,
            score_col=metric,
        )

        n_high = (results_df["label_semantic_clean"] == higher_label).sum()
        n_low = (results_df["label_semantic_clean"] == lower_label).sum()

        pairwise_auc_rows.append({
            "metric": metric,
            "higher_label": higher_label,
            "lower_label": lower_label,
            "comparison": f"{higher_label} > {lower_label}",
            "n_higher": n_high,
            "n_lower": n_low,
            "pairwise_auc": auc,
        })

pairwise_auc_df = pd.DataFrame(pairwise_auc_rows)

pairwise_auc_df = pairwise_auc_df.sort_values(
    ["metric", "pairwise_auc"],
    ascending=[True, False]
)

display(pairwise_auc_df)

# -----------------------------
# Bootstrap CI: pairwise AUC between classes
# -----------------------------

pairwise_auc_ci_rows = []

for metric in all_metric_cols:
    for higher_label, lower_label in label_pairs:
        pair_df = results_df[
            results_df["label_semantic_clean"].isin([higher_label, lower_label])
        ][[metric, "label_semantic_clean"]].dropna().copy()

        ci = bootstrap_ci(
            pair_df,
            stat_fn=lambda x, h=higher_label, l=lower_label, m=metric:
                pairwise_auc_between_classes(
                    x,
                    higher_label=h,
                    lower_label=l,
                    score_col=m,
                ),
            desc=f"Bootstrap AUC: {metric}, {higher_label} > {lower_label}",
            seed=BOOTSTRAP_SEED + 40,
        )

        pairwise_auc_ci_rows.append({
            "metric": metric,
            "higher_label": higher_label,
            "lower_label": lower_label,
            "comparison": f"{higher_label} > {lower_label}",
            "n": len(pair_df),
            "pairwise_auc": ci["estimate"],
            "pairwise_auc_ci_low": ci["ci_low"],
            "pairwise_auc_ci_high": ci["ci_high"],
        })

pairwise_auc_ci_df = pd.DataFrame(pairwise_auc_ci_rows)

pairwise_auc_ci_df["pairwise_auc_95ci"] = pairwise_auc_ci_df.apply(
    lambda r: f'{r["pairwise_auc"]:.4f} [{r["pairwise_auc_ci_low"]:.4f}, {r["pairwise_auc_ci_high"]:.4f}]',
    axis=1
)

pairwise_auc_ci_df = pairwise_auc_ci_df.sort_values(
    ["metric", "pairwise_auc"],
    ascending=[True, False]
)

display(pairwise_auc_ci_df)

# -----------------------------
# 3. Monotonicity violations
# -----------------------------

def monotonicity_violation_rate_between_classes(
    df,
    better_label,
    worse_label,
    score_col="cap_definition1",
    label_col="label_semantic_clean",
):
    """
    Violation:
        score(worse_label) > score(better_label)

    Tie:
        score(worse_label) == score(better_label)

    Non-violation:
        score(better_label) > score(worse_label)
    """

    temp = df[
        df[label_col].isin([better_label, worse_label])
    ][[score_col, label_col]].dropna().copy()

    better_scores = temp.loc[temp[label_col] == better_label, score_col].to_numpy()
    worse_scores = temp.loc[temp[label_col] == worse_label, score_col].to_numpy()

    if len(better_scores) == 0 or len(worse_scores) == 0:
        return {
            "n_better": len(better_scores),
            "n_worse": len(worse_scores),
            "comparable_pairs": 0,
            "violations": np.nan,
            "ties": np.nan,
            "correct": np.nan,
            "violation_rate": np.nan,
            "tie_rate": np.nan,
            "correct_rate": np.nan,
        }

    total = 0
    violations = 0
    ties = 0
    correct = 0

    for bs in better_scores:
        for ws in worse_scores:
            total += 1

            if ws > bs:
                violations += 1
            elif ws == bs:
                ties += 1
            else:
                correct += 1

    return {
        "n_better": len(better_scores),
        "n_worse": len(worse_scores),
        "comparable_pairs": total,
        "violations": violations,
        "ties": ties,
        "correct": correct,
        "violation_rate": violations / total if total > 0 else np.nan,
        "tie_rate": ties / total if total > 0 else np.nan,
        "correct_rate": correct / total if total > 0 else np.nan,
    }


ordered_labels_by_severity = sorted(
    severity_map.keys(),
    key=lambda x: severity_map[x],
    reverse=True
)

monotonicity_rows = []

for metric in all_metric_cols:
    for better_label in ordered_labels_by_severity:
        for worse_label in ordered_labels_by_severity:
            if better_label == worse_label:
                continue

            if severity_map[better_label] <= severity_map[worse_label]:
                continue

            stats = monotonicity_violation_rate_between_classes(
                results_df,
                better_label=better_label,
                worse_label=worse_label,
                score_col=metric,
            )

            monotonicity_rows.append({
                "metric": metric,
                "better_label": better_label,
                "worse_label": worse_label,
                "comparison": f"{better_label} > {worse_label}",
                **stats,
            })

monotonicity_df = pd.DataFrame(monotonicity_rows)

monotonicity_df = monotonicity_df.sort_values(
    ["metric", "violation_rate"],
    ascending=[True, False]
)

display(monotonicity_df)

# -----------------------------
# Mean CAP table based on monotonicity violations
# -----------------------------

cap_means = (
    results_df
    .groupby("label_semantic_clean")["cap_definition1"]
    .mean()
    .rename("mean_cap")
    .reset_index()
)

cap_monotonicity_means_df = monotonicity_df[
    monotonicity_df["metric"] == "cap_definition1"
].copy()

cap_monotonicity_means_df = cap_monotonicity_means_df.merge(
    cap_means.rename(columns={
        "label_semantic_clean": "better_label",
        "mean_cap": "Mean(better)"
    }),
    on="better_label",
    how="left"
)

cap_monotonicity_means_df = cap_monotonicity_means_df.merge(
    cap_means.rename(columns={
        "label_semantic_clean": "worse_label",
        "mean_cap": "Mean(worse)"
    }),
    on="worse_label",
    how="left"
)

cap_monotonicity_means_df["Mean difference better-worse"] = (
    cap_monotonicity_means_df["Mean(better)"]
    - cap_monotonicity_means_df["Mean(worse)"]
)

cap_monotonicity_means_df["Mean violation?"] = (
    cap_monotonicity_means_df["Mean(worse)"]
    > cap_monotonicity_means_df["Mean(better)"]
)

cap_monotonicity_means_df = cap_monotonicity_means_df[
    [
        "better_label",
        "worse_label",
        "Mean(better)",
        "Mean(worse)",
        "Mean difference better-worse",
        "Mean violation?",
        "violation_rate",
        "comparable_pairs",
        "violations",
        "ties",
        "correct",
    ]
].rename(columns={
    "better_label": "Better class",
    "worse_label": "Worse class",
})

cap_monotonicity_means_df = cap_monotonicity_means_df.sort_values(
    ["Mean violation?", "violation_rate"],
    ascending=[False, False]
)

cap_monotonicity_means_df[
    ["Mean(better)", "Mean(worse)", "Mean difference better-worse", "violation_rate"]
] = cap_monotonicity_means_df[
    ["Mean(better)", "Mean(worse)", "Mean difference better-worse", "violation_rate"]
].round(4)

display(cap_monotonicity_means_df)

display(
    cap_monotonicity_means_df[
        [
            "Better class",
            "Worse class",
            "Mean(better)",
            "Mean(worse)",
        ]
    ]
)

bad_mean_violations_df = cap_monotonicity_means_df[
    cap_monotonicity_means_df["Mean violation?"] == True
].copy()

display(
    bad_mean_violations_df[
        [
            "Better class",
            "Worse class",
            "Mean(better)",
            "Mean(worse)",
            "Mean difference better-worse",
            "violation_rate",
            "comparable_pairs",
        ]
    ]
)