import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

# ── NLP metrics ───────────────────────────────────────────────────────────────
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from scipy.stats import spearmanr, kendalltau

smoother = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

SEVERITY = {
    "exact":                 6,
    "equivalent":            6,
    "alternative_correct":   6,
    "overinclusive_valid":   5,
    "partial":               4,
    "overinclusive_invalid": 3,
    "invalid":               1,
    "contradictory":         0,
}

LABEL_ORDER = [
    "exact", "equivalent", "alternative_correct",
    "overinclusive_valid", "partial",
    "overinclusive_invalid", "invalid", "contradictory",
]

HARD_PAIRS = [
    ("equivalent",           "partial"),
    ("partial",              "overinclusive_valid"),
    ("overinclusive_valid",  "overinclusive_invalid"),
    ("alternative_correct",  "invalid"),
]

METRIC_NAMES = ["bleu", "rouge_l", "meteor", "bertscore"]

# ── helpers ────────────────────────────────────────────────────────────────────

def safe_bleu(ref: str, hyp: str) -> float:
    ref_tok = word_tokenize(ref.lower())
    hyp_tok = word_tokenize(hyp.lower())
    if not hyp_tok:
        return 0.0
    return sentence_bleu([ref_tok], hyp_tok, smoothing_function=smoother)


def safe_rouge_l(ref: str, hyp: str) -> float:
    return rouge.score(ref, hyp)["rougeL"].fmeasure


def safe_meteor(ref: str, hyp: str) -> float:
    ref_tok = word_tokenize(ref.lower())
    hyp_tok = word_tokenize(hyp.lower())
    if not ref_tok or not hyp_tok:
        return 0.0
    return meteor_score([ref_tok], hyp_tok)


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot=10_000, ci=0.95) -> tuple:
    boots = [stat_fn(np.random.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi


def bootstrap_corr(x: np.ndarray, y: np.ndarray, corr_fn, n_boot=10_000, ci=0.95):
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            val = corr_fn(x[idx], y[idx]).statistic
        except Exception:
            val = np.nan
        boots.append(val)
    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    return np.percentile(boots, (1 - ci) / 2 * 100), np.percentile(boots, (1 + ci) / 2 * 100)


def fmt(val, lo=None, hi=None, decimals=4):
    s = f"{val:.{decimals}f}"
    if lo is not None and hi is not None:
        s += f"  [{lo:.{decimals}f}, {hi:.{decimals}f}]"
    return s


def main():
    csv_path = ""
    print(f"Loading {csv_path} …")
    df = pd.read_csv( )
    df = df.dropna(subset=["gold_answer", "answer", "label_semantic"])
    n = len(df)
    print(f"Rows after dropna: {n}")

    refs  = df["gold_answer"].astype(str).tolist()
    hyps  = df["answer"].astype(str).tolist()
    labels = df["label_semantic"].tolist()

    print("\nComputing BLEU …")
    bleu_scores = np.array([safe_bleu(r, h) for r, h in zip(refs, hyps)])

    print("Computing ROUGE-L …")
    rouge_scores = np.array([safe_rouge_l(r, h) for r, h in zip(refs, hyps)])

    print("Computing METEOR …")
    meteor_scores = np.array([safe_meteor(r, h) for r, h in zip(refs, hyps)])

    print("Computing BERTScore (this may take a few minutes) …")
    P, R, F = bert_score_fn(hyps, refs, lang="en", verbose=False)
    bert_scores = F.numpy()

    scores = {
        "bleu":      bleu_scores,
        "rouge_l":   rouge_scores,
        "meteor":    meteor_scores,
        "bertscore": bert_scores,
    }

    df_scores = pd.DataFrame(scores)
    df_scores["label_semantic"] = labels
    df_scores["severity"]       = [SEVERITY[l] for l in labels]

    # ── 2. Per-class descriptive statistics ───────────────────────────────────
    print("\n\n" + "═"*90)
    print("TABLE 1 — Per-class descriptive statistics  (mean ± std | min | p5 | p95 | max)")
    print("═"*90)

    for metric in METRIC_NAMES:
        print(f"\n  Metric: {metric.upper()}")
        header = f"  {'Label':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'P5':>8} {'P95':>8} {'Max':>8}"
        print(header)
        print("  " + "-"*73)
        for lbl in LABEL_ORDER:
            if lbl not in df_scores["label_semantic"].values:
                continue
            vals = df_scores.loc[df_scores["label_semantic"] == lbl, metric].values
            p5, p95 = np.percentile(vals, 5), np.percentile(vals, 95)
            print(f"  {lbl:<25} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
                  f"{np.min(vals):>8.4f} {p5:>8.4f} {p95:>8.4f} {np.max(vals):>8.4f}")

    # ── 3. Spearman & Kendall (severity ↔ metric score) ──────────────────────
    print("\n\n" + "═"*90)
    print("TABLE 2 — Spearman ρ and Kendall τ  (severity vs metric score, 95% CI via 10k bootstrap)")
    print("═"*90)
    sev = df_scores["severity"].values

    header2 = f"  {'Metric':<12} {'Spearman ρ':>12}  {'95% CI':>20}    {'Kendall τ':>12}  {'95% CI':>20}"
    print(header2)
    print("  " + "-"*80)
    for metric in METRIC_NAMES:
        sc = df_scores[metric].values
        sp  = spearmanr(sev, sc).statistic
        kt  = kendalltau(sev, sc).statistic
        sp_lo, sp_hi = bootstrap_corr(sev, sc, spearmanr)
        kt_lo, kt_hi = bootstrap_corr(sev, sc, kendalltau)
        print(f"  {metric:<12} {sp:>12.4f}  [{sp_lo:>8.4f}, {sp_hi:>8.4f}]    "
              f"{kt:>12.4f}  [{kt_lo:>8.4f}, {kt_hi:>8.4f}]")

    # ── 4. Pairwise ranking accuracy (all comparable pairs) ───────────────────
    print("\n\n" + "═"*90)
    print("TABLE 3 — Pairwise ranking accuracy  (higher severity → higher metric score)")
    print("═"*90)
    print("  (Pairs where both labels have DIFFERENT severity; ties in severity excluded)\n")

    # Precompute indices per label
    label_idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        label_idx[lbl].append(i)

    def pairwise_acc(metric_vals, label_pairs_subset=None):
        correct = total = 0
        lbl_list = list(LABEL_ORDER)
        pairs_to_check = label_pairs_subset if label_pairs_subset else [
            (a, b) for a, b in combinations(lbl_list, 2)
            if SEVERITY.get(a, -1) != SEVERITY.get(b, -1)
        ]
        for la, lb in pairs_to_check:
            sa, sb = SEVERITY.get(la, -1), SEVERITY.get(lb, -1)
            if sa == sb:
                continue
            high_lbl, low_lbl = (la, lb) if sa > sb else (lb, la)
            hi_vals = metric_vals[label_idx[high_lbl]]
            lo_vals = metric_vals[label_idx[low_lbl]]
            n_hi, n_lo = len(hi_vals), len(lo_vals)
            for v_hi in hi_vals:
                for v_lo in lo_vals:
                    correct += int(v_hi > v_lo)
                    total   += 1
        return correct, total

    header3 = f"  {'Metric':<12} {'Correct':>12} {'Total':>12} {'Accuracy':>12}  {'95% CI':>22}"
    print(header3)
    print("  " + "-"*74)

    def row_pairwise_acc_bootstrap(metric_vals, n_boot=10_000):
        """Faster bootstrap: sample indices at row level, recompute per-label stats."""
        n = len(metric_vals)
        boots = []
        lbl_arr = np.array(labels)
        sev_arr = np.array([SEVERITY[l] for l in labels])

        label_combos = [
            (la, lb) for la, lb in combinations(LABEL_ORDER, 2)
            if SEVERITY.get(la, -1) != SEVERITY.get(lb, -1)
            and la in label_idx and lb in label_idx
        ]

        for _ in range(n_boot):
            idx = np.random.choice(n, size=n, replace=True)
            mv  = metric_vals[idx]
            lb_b = lbl_arr[idx]
            boot_idx = defaultdict(list)
            for i, l in enumerate(lb_b):
                boot_idx[l].append(mv[i])

            correct = total = 0
            for la, lb in label_combos:
                sa, sb = SEVERITY[la], SEVERITY[lb]
                high_lbl, low_lbl = (la, lb) if sa > sb else (lb, la)
                hi_v = boot_idx.get(high_lbl, [])
                lo_v = boot_idx.get(low_lbl, [])
                if not hi_v or not lo_v:
                    continue
                # approximate: use mean comparison per boot for speed
                if np.mean(hi_v) > np.mean(lo_v):
                    correct += 1
                total += 1
            boots.append(correct / total if total else np.nan)
        boots = np.array(boots)
        return np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)

    for metric in METRIC_NAMES:
        mv = df_scores[metric].values
        correct, total = pairwise_acc(mv)
        acc = correct / total if total else 0
        ci_lo, ci_hi = row_pairwise_acc_bootstrap(mv, n_boot=2000)  # 2k is fast enough
        print(f"  {metric:<12} {correct:>12,} {total:>12,} {acc:>12.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── 5. Hard neighboring pairs ─────────────────────────────────────────────
    print("\n\n" + "═"*90)
    print("TABLE 4 — Pairwise ranking accuracy on HARD neighboring pairs")
    print("═"*90)

    hard_label_pairs = [
        (la, lb) for la, lb in HARD_PAIRS
        if la in label_idx and lb in label_idx
    ]

    for metric in METRIC_NAMES:
        print(f"\n  Metric: {metric.upper()}")
        mv = df_scores[metric].values
        header4 = f"  {'Pair':<45} {'Correct':>10} {'Total':>10} {'Accuracy':>10}"
        print(header4)
        print("  " + "-"*77)
        for la, lb in hard_label_pairs:
            sa, sb = SEVERITY[la], SEVERITY[lb]
            high_lbl, low_lbl = (la, lb) if sa > sb else (lb, la)
            hi_vals = mv[label_idx[high_lbl]]
            lo_vals = mv[label_idx[low_lbl]]
            correct = sum(int(v_hi > v_lo) for v_hi in hi_vals for v_lo in lo_vals)
            total   = len(hi_vals) * len(lo_vals)
            acc     = correct / total if total else 0
            pair_str = f"{high_lbl} > {low_lbl}"
            print(f"  {pair_str:<45} {correct:>10,} {total:>10,} {acc:>10.4f}")

    # ── 6. Monotonicity violations ─────────────────────────────────────────────
    print("\n\n" + "═"*90)
    print("TABLE 5 — Monotonicity violations  (mean score of worse class > mean score of better class)")
    print("═"*90)
    print("  Violation rate = #violating pairs / #total comparable class pairs\n")

    # Compute per-class mean per metric
    class_means = {}
    for lbl in LABEL_ORDER:
        if lbl not in label_idx:
            continue
        class_means[lbl] = {m: np.mean(df_scores.loc[df_scores["label_semantic"]==lbl, m].values)
                             for m in METRIC_NAMES}

    for metric in METRIC_NAMES:
        violations = 0
        total_pairs = 0
        viol_list = []
        for la, lb in combinations(LABEL_ORDER, 2):
            if la not in class_means or lb not in class_means:
                continue
            sa, sb = SEVERITY[la], SEVERITY[lb]
            if sa == sb:
                continue
            high_lbl, low_lbl = (la, lb) if sa > sb else (lb, la)
            hi_mean = class_means[high_lbl][metric]
            lo_mean = class_means[low_lbl][metric]
            total_pairs += 1
            if lo_mean > hi_mean:
                violations += 1
                viol_list.append((high_lbl, low_lbl, hi_mean, lo_mean))

        rate = violations / total_pairs if total_pairs else 0
        print(f"\n  Metric: {metric.upper()} — violation rate: {violations}/{total_pairs} = {rate:.4f}")
        if viol_list:
            print(f"  {'Better class':<25} {'Worse class':<25} {'Mean(better)':>14} {'Mean(worse)':>14}")
            print("  " + "-"*80)
            for h, l, hm, lm in viol_list:
                print(f"  {h:<25} {l:<25} {hm:>14.4f} {lm:>14.4f}")
        else:
            print("  No violations detected.")

    # ── 7. Bootstrap CIs — per metric per class (mean & median) ───────────────
    print("\n\n" + "═"*90)
    print("TABLE 6 — Bootstrap 95% CIs per metric per class  (10k resamples, mean & median)")
    print("═"*90)

    N_BOOT = 10_000
    for metric in METRIC_NAMES:
        print(f"\n  Metric: {metric.upper()}")
        header6 = (f"  {'Label':<25} {'Mean':>8} {'Mean CI':>22}    "
                   f"{'Median':>8} {'Median CI':>22}")
        print(header6)
        print("  " + "-"*90)
        for lbl in LABEL_ORDER:
            if lbl not in label_idx:
                continue
            vals = df_scores.loc[df_scores["label_semantic"] == lbl, metric].values
            mn   = np.mean(vals)
            med  = np.median(vals)
            mn_lo, mn_hi   = bootstrap_ci(vals, np.mean,   n_boot=N_BOOT)
            med_lo, med_hi = bootstrap_ci(vals, np.median, n_boot=N_BOOT)
            print(f"  {lbl:<25} {mn:>8.4f} [{mn_lo:>8.4f}, {mn_hi:>8.4f}]    "
                  f"{med:>8.4f} [{med_lo:>8.4f}, {med_hi:>8.4f}]")

    # ── 8. Bootstrap CIs — Spearman, Kendall, pairwise accuracy ───────────────
    print("\n\n" + "═"*90)
    print("TABLE 7 — Bootstrap 95% CIs: Spearman, Kendall, pairwise ranking accuracy  (10k resamples)")
    print("═"*90)

    sev_arr = df_scores["severity"].values.astype(float)
    header7 = (f"  {'Metric':<12} {'Spearman':>10} {'95% CI':>22}  "
               f"{'Kendall':>10} {'95% CI':>22}  {'PairAcc':>10} {'95% CI':>22}")
    print(header7)
    print("  " + "-"*115)

    def pairwise_acc_fast(metric_vals, lbl_arr, label_order, severity_map):
        cls_means = {}
        for l in label_order:
            mask = lbl_arr == l
            if mask.sum() > 0:
                cls_means[l] = metric_vals[mask].mean()
        correct = total = 0
        for la, lb in combinations(label_order, 2):
            if la not in cls_means or lb not in cls_means:
                continue
            sa, sb = severity_map[la], severity_map[lb]
            if sa == sb:
                continue
            high_lbl, low_lbl = (la, lb) if sa > sb else (lb, la)
            if cls_means[high_lbl] > cls_means[low_lbl]:
                correct += 1
            total += 1
        return correct / total if total else np.nan

    lbl_arr_np = np.array(labels)

    for metric in METRIC_NAMES:
        mv = df_scores[metric].values
        sp_val = spearmanr(sev_arr, mv).statistic
        kt_val = kendalltau(sev_arr, mv).statistic
        pa_val = pairwise_acc_fast(mv, lbl_arr_np, LABEL_ORDER, SEVERITY)

        sp_lo, sp_hi = bootstrap_corr(sev_arr, mv, spearmanr,   n_boot=N_BOOT)
        kt_lo, kt_hi = bootstrap_corr(sev_arr, mv, kendalltau,  n_boot=N_BOOT)

        # bootstrap pairwise acc
        pa_boots = []
        n = len(mv)
        for _ in range(N_BOOT):
            idx = np.random.choice(n, size=n, replace=True)
            pa_boots.append(pairwise_acc_fast(mv[idx], lbl_arr_np[idx], LABEL_ORDER, SEVERITY))
        pa_boots = np.array(pa_boots)
        pa_lo = np.nanpercentile(pa_boots, 2.5)
        pa_hi = np.nanpercentile(pa_boots, 97.5)

        print(f"  {metric:<12} {sp_val:>10.4f} [{sp_lo:>8.4f}, {sp_hi:>8.4f}]  "
              f"{kt_val:>10.4f} [{kt_lo:>8.4f}, {kt_hi:>8.4f}]  "
              f"{pa_val:>10.4f} [{pa_lo:>8.4f}, {pa_hi:>8.4f}]")

    print("\n\nDone.\n")


if __name__ == "__main__":
    main()
