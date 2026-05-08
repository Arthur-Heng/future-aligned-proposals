#!/usr/bin/env python3
"""
Compute correlation between human evaluation preferences and automated FAS scores.

For each human-evaluated pair, we look up the automated evaluation scores for
both sides and test whether the side preferred by humans also has a higher FAS.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_eval_results(path: str) -> dict:
    """Load evaluation results and index by root_title."""
    with open(path) as f:
        data = json.load(f)
    by_title = {}
    for r in data.get("results", []):
        title = r.get("root_title", "")
        by_title[title] = r
    return by_title


def load_annotations(annotations_dir: str, batches_dir: str):
    """Load all annotations and pair metadata, returning per-pair majority votes."""
    pair_votes = {}

    for batch_letter in ["A", "B", "C", "D"]:
        batch_path = os.path.join(batches_dir, f"comparison_batch_{batch_letter}.json")
        if not os.path.exists(batch_path):
            continue
        with open(batch_path) as f:
            batch_data = json.load(f)
        pairs = batch_data.get("pairs", [])

        ann_files = {}
        for fname in sorted(os.listdir(annotations_dir)):
            if f"batch_{batch_letter}_annotations_" in fname:
                annotator = fname.split("_annotations_")[1].replace(".json", "")
                with open(os.path.join(annotations_dir, fname)) as f:
                    ann_files[annotator] = json.load(f)

        for local_idx, pair in enumerate(pairs):
            pair_id = pair.get("id", f"{batch_letter}_{local_idx}")
            for metric in ["overall", "soundness", "excitement"]:
                votes = []
                for ann_data in ann_files.values():
                    key = str(local_idx)
                    if key in ann_data and metric in ann_data[key]:
                        votes.append(ann_data[key][metric])

                if len(votes) < 3:
                    continue

                count_a = sum(1 for v in votes if v == "A")
                count_b = sum(1 for v in votes if v == "B")
                if count_a > count_b:
                    majority = "A"
                elif count_b > count_a:
                    majority = "B"
                else:
                    majority = "tie"

                if pair_id not in pair_votes:
                    pair_votes[pair_id] = {"pair": pair}
                pair_votes[pair_id][metric] = majority

    return pair_votes


def point_biserial_corr(x_continuous, y_binary):
    """Point-biserial correlation between continuous x and binary y (0/1)."""
    x = np.array(x_continuous, dtype=float)
    y = np.array(y_binary, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt((n - 2) / (1 - r**2))
    from scipy import stats
    p = 2 * stats.t.sf(abs(t), n - 2)
    return r, p


def main():
    base_dir = Path(__file__).parent.parent

    stepwise_eval_path = base_dir / "evaluation/results/eval_qwen-14b-tuned-stepwise-cot-v4_neurips_2025_icml_2025_plus1_20260227_154033.json"
    prompting_eval_path = base_dir / "evaluation/results/eval_qwen-14b-v4_neurips_2025_icml_2025_plus1_20260227_165811.json"
    human_data_path = base_dir / "baselines/comparison_human_annotation.json"
    annotations_dir = base_dir / "baselines/annotations"
    batches_dir = base_dir / "baselines"

    print("Loading automated evaluation results...")
    stepwise_by_title = load_eval_results(str(stepwise_eval_path))
    prompting_by_title = load_eval_results(str(prompting_eval_path))
    print(f"  Stepwise-CoT: {len(stepwise_by_title)} results")
    print(f"  Prompting: {len(prompting_by_title)} results")

    print("Loading human annotations...")
    pair_votes = load_annotations(str(annotations_dir), str(batches_dir))
    print(f"  {len(pair_votes)} pairs with votes")

    with open(human_data_path) as f:
        human_data = json.load(f)
    pairs_meta = {p["id"]: p for p in human_data["pairs"]}

    # =========================================================================
    # Analysis 1: Score difference predicts human preference
    # For stepwise_vs_prompting pairs: does (stepwise_score - prompting_score)
    # correlate with human preferring stepwise?
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: FAS score difference vs. human preference")
    print("(stepwise_vs_prompting pairs only)")
    print("=" * 80)

    for metric in ["overall", "soundness", "excitement"]:
        score_diffs = []
        human_prefs = []  # 1 = stepwise preferred, 0 = prompting preferred
        details = []

        for pair_id, vote_data in pair_votes.items():
            pair = vote_data["pair"]
            if pair.get("comparison_type") != "stepwise_vs_prompting":
                continue
            if metric not in vote_data:
                continue

            title = pair.get("root_title", "")
            stepwise_is_a = pair.get("stepwise_is_a", True)

            s_eval = stepwise_by_title.get(title)
            p_eval = prompting_by_title.get(title)
            if not s_eval or not p_eval:
                continue

            s_score = s_eval.get("max_subfield_scores", {}).get("overall", 0)
            p_score = p_eval.get("max_subfield_scores", {}).get("overall", 0)
            diff = s_score - p_score

            majority = vote_data[metric]
            if majority == "A":
                human_pref = 1 if stepwise_is_a else 0
            elif majority == "B":
                human_pref = 0 if stepwise_is_a else 1
            else:
                continue  # skip ties for correlation

            score_diffs.append(diff)
            human_prefs.append(human_pref)
            details.append((title[:50], s_score, p_score, diff, human_pref))

        if len(score_diffs) < 5:
            print(f"\n  {metric}: too few non-tie pairs ({len(score_diffs)})")
            continue

        r, p = point_biserial_corr(score_diffs, human_prefs)
        concordant = sum(1 for d, h in zip(score_diffs, human_prefs) if (d > 0 and h == 1) or (d < 0 and h == 0))
        discordant = sum(1 for d, h in zip(score_diffs, human_prefs) if (d > 0 and h == 0) or (d < 0 and h == 1))
        tied_auto = sum(1 for d in score_diffs if d == 0)

        print(f"\n  {metric.upper()} (n={len(score_diffs)} non-tie pairs):")
        print(f"    Point-biserial r = {r:.3f} (p = {p:.4f})")
        print(f"    Concordant: {concordant}, Discordant: {discordant}, Auto-tied: {tied_auto}")
        if concordant + discordant > 0:
            concordance = concordant / (concordant + discordant)
            print(f"    Concordance rate (excl. auto-ties): {concordance:.1%} ({concordant}/{concordant+discordant})")

    # =========================================================================
    # Analysis 2: Absolute FAS predicts competitiveness with human proposals
    # For stepwise_vs_human pairs: does higher stepwise FAS correlate with
    # human judges finding stepwise competitive (win or tie)?
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: FAS score predicts competitiveness with human proposals")
    print("(stepwise_vs_human pairs)")
    print("=" * 80)

    for metric in ["overall", "soundness", "excitement"]:
        fas_scores = []
        human_outcomes = []  # 1 = stepwise wins, 0 = stepwise loses

        for pair_id, vote_data in pair_votes.items():
            pair = vote_data["pair"]
            if pair.get("comparison_type") != "stepwise_vs_human":
                continue
            if metric not in vote_data:
                continue

            title = pair.get("root_title", "")
            generated_is_a = pair.get("generated_is_a", True)

            s_eval = stepwise_by_title.get(title)
            if not s_eval:
                continue

            fas = s_eval.get("max_subfield_scores", {}).get("overall", 0)

            majority = vote_data[metric]
            if majority == "A":
                outcome = 1 if generated_is_a else 0
            elif majority == "B":
                outcome = 0 if generated_is_a else 1
            else:
                continue  # skip ties

            fas_scores.append(fas)
            human_outcomes.append(outcome)

        if len(fas_scores) < 5:
            print(f"\n  {metric}: too few non-tie pairs ({len(fas_scores)})")
            continue

        r, p = point_biserial_corr(fas_scores, human_outcomes)

        wins = [f for f, o in zip(fas_scores, human_outcomes) if o == 1]
        losses = [f for f, o in zip(fas_scores, human_outcomes) if o == 0]

        print(f"\n  {metric.upper()} (n={len(fas_scores)} non-tie pairs):")
        print(f"    Point-biserial r = {r:.3f} (p = {p:.4f})")
        print(f"    Mean FAS when stepwise wins: {np.mean(wins):.2f} (n={len(wins)})")
        print(f"    Mean FAS when stepwise loses: {np.mean(losses):.2f} (n={len(losses)})")

    # =========================================================================
    # Analysis 3: Per-subfield FAS difference vs. human preference
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Per-subfield FAS difference vs. human overall preference")
    print("(stepwise_vs_prompting pairs)")
    print("=" * 80)

    subfields = ["research_question", "hypothesis", "proposed_method", "novelty_claims", "experiment_details", "overall"]

    for sf in subfields:
        score_diffs = []
        human_prefs = []

        for pair_id, vote_data in pair_votes.items():
            pair = vote_data["pair"]
            if pair.get("comparison_type") != "stepwise_vs_prompting":
                continue
            if "overall" not in vote_data:
                continue

            title = pair.get("root_title", "")
            stepwise_is_a = pair.get("stepwise_is_a", True)

            s_eval = stepwise_by_title.get(title)
            p_eval = prompting_by_title.get(title)
            if not s_eval or not p_eval:
                continue

            s_score = s_eval.get("max_subfield_scores", {}).get(sf, 0)
            p_score = p_eval.get("max_subfield_scores", {}).get(sf, 0)
            diff = s_score - p_score

            majority = vote_data["overall"]
            if majority == "A":
                human_pref = 1 if stepwise_is_a else 0
            elif majority == "B":
                human_pref = 0 if stepwise_is_a else 1
            else:
                continue

            score_diffs.append(diff)
            human_prefs.append(human_pref)

        if len(score_diffs) < 5:
            continue

        r, p = point_biserial_corr(score_diffs, human_prefs)
        concordant = sum(1 for d, h in zip(score_diffs, human_prefs) if (d > 0 and h == 1) or (d < 0 and h == 0))
        discordant = sum(1 for d, h in zip(score_diffs, human_prefs) if (d > 0 and h == 0) or (d < 0 and h == 1))

        label = sf.replace("_", " ").title()
        conc = concordant / (concordant + discordant) if (concordant + discordant) > 0 else 0
        print(f"  {label:25s}  r={r:+.3f} (p={p:.3f})  concordance={conc:.1%} ({concordant}/{concordant+discordant})")

    # =========================================================================
    # Analysis 4: Rank-biserial correlation (non-parametric)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Rank-biserial correlation (Mann-Whitney)")
    print("=" * 80)

    from scipy.stats import mannwhitneyu

    for comp_type, comp_label in [("stepwise_vs_prompting", "vs Prompting"), ("stepwise_vs_human", "vs Human")]:
        print(f"\n  --- {comp_label} ---")
        for metric in ["overall", "soundness", "excitement"]:
            win_scores = []
            lose_scores = []

            for pair_id, vote_data in pair_votes.items():
                pair = vote_data["pair"]
                if pair.get("comparison_type") != comp_type:
                    continue
                if metric not in vote_data:
                    continue

                title = pair.get("root_title", "")
                s_eval = stepwise_by_title.get(title)
                if not s_eval:
                    continue

                fas = s_eval.get("max_subfield_scores", {}).get("overall", 0)

                if comp_type == "stepwise_vs_prompting":
                    stepwise_is_a = pair.get("stepwise_is_a", True)
                else:
                    stepwise_is_a = pair.get("generated_is_a", True)

                majority = vote_data[metric]
                if majority == "A":
                    stepwise_wins = stepwise_is_a
                elif majority == "B":
                    stepwise_wins = not stepwise_is_a
                else:
                    continue

                if stepwise_wins:
                    win_scores.append(fas)
                else:
                    lose_scores.append(fas)

            if len(win_scores) < 3 or len(lose_scores) < 3:
                print(f"    {metric}: insufficient data")
                continue

            U, p = mannwhitneyu(win_scores, lose_scores, alternative="two-sided")
            n1, n2 = len(win_scores), len(lose_scores)
            rank_biserial = 1 - (2 * U) / (n1 * n2)

            print(f"    {metric:12s}  win_FAS={np.mean(win_scores):.2f} (n={n1}), lose_FAS={np.mean(lose_scores):.2f} (n={n2}), rank-biserial r={rank_biserial:+.3f}, p={p:.3f}")

    # =========================================================================
    # Summary for LaTeX
    # =========================================================================
    print("\n" + "=" * 80)
    print("LATEX SUMMARY")
    print("=" * 80)

    # Recompute key numbers for stepwise_vs_prompting overall
    score_diffs = []
    human_prefs = []
    for pair_id, vote_data in pair_votes.items():
        pair = vote_data["pair"]
        if pair.get("comparison_type") != "stepwise_vs_prompting":
            continue
        if "overall" not in vote_data:
            continue
        title = pair.get("root_title", "")
        stepwise_is_a = pair.get("stepwise_is_a", True)
        s_eval = stepwise_by_title.get(title)
        p_eval = prompting_by_title.get(title)
        if not s_eval or not p_eval:
            continue
        s_score = s_eval.get("max_subfield_scores", {}).get("overall", 0)
        p_score = p_eval.get("max_subfield_scores", {}).get("overall", 0)
        diff = s_score - p_score
        majority = vote_data["overall"]
        if majority == "A":
            human_pref = 1 if stepwise_is_a else 0
        elif majority == "B":
            human_pref = 0 if stepwise_is_a else 1
        else:
            continue
        score_diffs.append(diff)
        human_prefs.append(human_pref)

    r, p = point_biserial_corr(score_diffs, human_prefs)
    concordant = sum(1 for d, h in zip(score_diffs, human_prefs) if (d > 0 and h == 1) or (d < 0 and h == 0))
    discordant = sum(1 for d, h in zip(score_diffs, human_prefs) if (d > 0 and h == 0) or (d < 0 and h == 1))
    if concordant + discordant > 0:
        conc = concordant / (concordant + discordant)
    else:
        conc = 0
    print(f"Stepwise vs Prompting (overall): r={r:.3f}, p={p:.4f}, concordance={conc:.1%}")
    print(f"  n={len(score_diffs)} non-tie pairs, concordant={concordant}, discordant={discordant}")


if __name__ == "__main__":
    main()
