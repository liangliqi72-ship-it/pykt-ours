import os
import json
import math
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


FEATURE_NAMES = [
    "hist_len_norm",
    "prefix_correct_rate",
    "recent_correct_rate",
    "recent_volatility",
    "consecutive_wrong_norm",
    "current_concept_count_norm",
    "current_concept_correct_rate",
    "gap_since_last_same_concept_norm",
    "decayed_correct_rate",
    "behavior_risk",
]


ALPHA_FEATURE_NAMES = [
    "hist_len_norm",
    "prefix_correct_rate",
    "recent_correct_rate",
    "recent_volatility",
    "behavior_risk",
]


def parse_one_token(v):
    """
    Convert one sequence token to int.

    Supports:
        "12"
        "12.0"
        "-1"
        "12_34"  -> use first concept id 12
    """
    if v is None:
        return -1

    s = str(v).strip()

    if len(s) == 0 or s.lower() == "nan":
        return -1

    # pyKT sometimes uses multi-concept format like "12_34"
    if "_" in s:
        s = s.split("_")[0]

    return int(float(s))


def parse_seq(x):
    """
    Parse pyKT sequence string.

    Supports:
        "1,2,3"
        "[1,2,3]"
        "1"
        "12_34,56_78"
        NaN
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return [parse_one_token(v) for v in x]

    if pd.isna(x):
        return []

    s = str(x).strip()
    if len(s) == 0:
        return []

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    if len(s.strip()) == 0:
        return []

    return [parse_one_token(v) for v in s.split(",") if str(v).strip() != ""]



def seq_to_str(seq):
    return ",".join(str(int(x)) for x in seq)


def feat_to_str(feats):
    """
    feats: List[List[float]], shape [L, F]

    Output format:
        timestep delimiter: ","
        feature delimiter: "|"

    Example:
        0.1|0.2|0.3|0.4|0.5,0.2|0.3|0.4|0.5|0.6
    """
    rows = []
    for row in feats:
        rows.append("|".join(f"{float(v):.6f}" for v in row))
    return ",".join(rows)


def safe_rate(num, den, default=0.0):
    if den <= 0:
        return default
    return float(num) / float(den)


def align_concepts_and_responses(concepts, responses):
    raw_len = len(responses)

    if len(concepts) < raw_len:
        concepts = concepts + [-1] * (raw_len - len(concepts))
    elif len(concepts) > raw_len:
        concepts = concepts[:raw_len]

    return concepts, responses


def build_prefix_features(
    concepts,
    responses,
    max_len=200,
    window=5,
    decay_lambda=0.1,
):
    """
    Build causal prefix features.

    At timestep t:
        allowed:
            q_0 ... q_{t-1}
            r_0 ... r_{t-1}
            current concept c_t

        forbidden:
            current response r_t
            future responses r_{t+1:}

    Returns:
        group_profile: [L, 10]
        alpha_feat:    [L, 5]
        valid_mask:    [L]
        hist_lens:     [L], exact causal history length before t
    """

    L = len(responses)

    group_profile = []
    alpha_feat = []
    valid_mask = []
    hist_lens = []

    # Global prefix states
    hist_len = 0
    correct_sum = 0
    recent_rs = []

    # Per-concept prefix states
    concept_count = {}
    concept_correct = {}
    concept_last_pos = {}
    concept_past_events = {}

    consecutive_wrong = 0

    for t in range(L):
        c = concepts[t] if t < len(concepts) else -1
        r = responses[t]

        is_valid = int(r in [0, 1] and c >= 0)

        # Exact causal history length before observing r_t
        hist_lens.append(hist_len)

        # ---------- causal features before observing r_t ----------
        hist_len_norm = math.log1p(hist_len) / math.log1p(max_len)
        hist_len_norm = min(max(hist_len_norm, 0.0), 1.0)

        prefix_correct_rate = safe_rate(correct_sum, hist_len, default=0.0)

        if len(recent_rs) > 0:
            recent_correct_rate = float(np.mean(recent_rs[-window:]))
        else:
            recent_correct_rate = 0.0

        recent_volatility = math.sqrt(
            max(recent_correct_rate * (1.0 - recent_correct_rate), 0.0)
        )

        consecutive_wrong_norm = min(consecutive_wrong / float(window), 1.0)

        c_count = concept_count.get(c, 0)
        c_correct = concept_correct.get(c, 0)

        current_concept_count_norm = math.log1p(c_count) / math.log1p(max_len)
        current_concept_count_norm = min(max(current_concept_count_norm, 0.0), 1.0)

        current_concept_correct_rate = safe_rate(
            c_correct,
            c_count,
            default=0.0,
        )

        if c in concept_last_pos:
            gap = t - concept_last_pos[c]
            gap_since_last_same_concept_norm = math.log1p(gap) / math.log1p(max_len)
            gap_since_last_same_concept_norm = min(
                max(gap_since_last_same_concept_norm, 0.0),
                1.0,
            )
        else:
            # No previous practice on this concept.
            # Treat as high memory uncertainty.
            gap_since_last_same_concept_norm = 1.0

        # Decayed correctness for current concept.
        # Only uses previous events of current concept.
        past_events = concept_past_events.get(c, [])

        if len(past_events) == 0:
            decayed_correct_rate = 0.0
        else:
            weights = []
            values = []

            for pos, rr in past_events:
                dist = t - pos
                w = math.exp(-decay_lambda * dist)
                weights.append(w)
                values.append(rr)

            decayed_correct_rate = float(
                np.dot(weights, values) / max(np.sum(weights), 1e-8)
            )

        short_history_risk = 1.0 - hist_len_norm
        low_recent_perf = 1.0 - recent_correct_rate

        behavior_risk = (
            0.4 * short_history_risk
            + 0.3 * low_recent_perf
            + 0.3 * recent_volatility
        )
        behavior_risk = min(max(behavior_risk, 0.0), 1.0)

        # 10-d profile for KMeans group assignment:
        # cognitive + memory + behavior
        z = [
            hist_len_norm,
            prefix_correct_rate,
            recent_correct_rate,
            recent_volatility,
            consecutive_wrong_norm,
            current_concept_count_norm,
            current_concept_correct_rate,
            gap_since_last_same_concept_norm,
            decayed_correct_rate,
            behavior_risk,
        ]

        # 5-d feature for alpha gate:
        # intentionally smaller than group profile
        a = [
            hist_len_norm,
            prefix_correct_rate,
            recent_correct_rate,
            recent_volatility,
            behavior_risk,
        ]

        group_profile.append(z)
        alpha_feat.append(a)
        valid_mask.append(is_valid)

        # ---------- update prefix states after observing r_t ----------
        if is_valid:
            rr = int(r)

            hist_len += 1
            correct_sum += rr
            recent_rs.append(rr)

            concept_count[c] = concept_count.get(c, 0) + 1
            concept_correct[c] = concept_correct.get(c, 0) + rr
            concept_last_pos[c] = t

            if c not in concept_past_events:
                concept_past_events[c] = []
            concept_past_events[c].append((t, rr))

            if rr == 0:
                consecutive_wrong += 1
            else:
                consecutive_wrong = 0

    return group_profile, alpha_feat, valid_mask, hist_lens


def detect_columns(df):
    """
    Detect concept/question/response columns for pyKT sequence csv.
    """
    concept_candidates = [
        "concepts",
        "skills",
        "skill_ids",
        "cseqs",
        "concept_seq",
        "concept_ids",
    ]

    question_candidates = [
        "questions",
        "qseqs",
        "question_ids",
        "problem_ids",
        "items",
    ]

    response_candidates = [
        "responses",
        "rseqs",
        "corrects",
        "labels",
        "answer",
    ]

    concept_col = None
    for c in concept_candidates:
        if c in df.columns:
            concept_col = c
            break

    question_col = None
    for c in question_candidates:
        if c in df.columns:
            question_col = c
            break

    response_col = None
    for c in response_candidates:
        if c in df.columns:
            response_col = c
            break

    # Prefer concepts if available. Otherwise use questions as concept proxy.
    if concept_col is None:
        concept_col = question_col

    if concept_col is None:
        raise ValueError(
            f"Cannot find concept/question column. Existing columns: {list(df.columns)}"
        )

    if response_col is None:
        raise ValueError(
            f"Cannot find response column. Existing columns: {list(df.columns)}"
        )

    return concept_col, response_col


def filter_fit_df(df, args):
    """
    Strict paper-level usage:
        scaler and KMeans should be fit on train split only.

    If --fit_fold_col is not provided, this function returns full df and warns.
    """
    fit_df = df.copy()

    if args.fit_fold_col is None:
        print("[WARN] --fit_fold_col is not provided.")
        print("[WARN] Fitting scaler/KMeans on the full fit file.")
        print("[WARN] This is only safe if the fit file contains train-only data.")
        return fit_df

    if args.fit_fold_col not in fit_df.columns:
        raise ValueError(
            f"fit_fold_col={args.fit_fold_col} not found. "
            f"Existing columns: {list(fit_df.columns)}"
        )

    if args.fit_fold_values is None:
        raise ValueError(
            "--fit_fold_col is provided, but --fit_fold_values is None."
        )

    values = [v.strip() for v in args.fit_fold_values.split(",")]

    fit_df = fit_df[
        fit_df[args.fit_fold_col].astype(str).isin(values)
    ].copy()

    print(
        f"[INFO] Using train-only split for fitting: "
        f"{args.fit_fold_col} in {values}, rows={len(fit_df)}"
    )

    if len(fit_df) == 0:
        raise ValueError("No rows left after fit split filtering.")

    return fit_df


def collect_fit_features(
    df,
    concept_col,
    response_col,
    max_len,
    window,
    min_hist,
    decay_lambda,
):
    """
    Collect prefix features for scaler/KMeans fitting.

    Important:
        Only valid positions with hist_len >= min_hist are used.
        Cold-start positions are reserved for unknown group.
    """
    all_feats = []

    for _, row in df.iterrows():
        concepts = parse_seq(row[concept_col])
        responses = parse_seq(row[response_col])
        concepts, responses = align_concepts_and_responses(concepts, responses)

        profile, _, valid_mask, hist_lens = build_prefix_features(
            concepts=concepts,
            responses=responses,
            max_len=max_len,
            window=window,
            decay_lambda=decay_lambda,
        )

        for z, is_valid, hist_len in zip(profile, valid_mask, hist_lens):
            if is_valid and hist_len >= min_hist:
                all_feats.append(z)

    return np.asarray(all_feats, dtype=np.float32)


def make_empty_counter(k):
    return {str(i): 0 for i in range(k + 1)}


def add_counter(counter, gid):
    key = str(int(gid))
    if key not in counter:
        counter[key] = 0
    counter[key] += 1


def transform_df(df, concept_col, response_col, scaler, kmeans, args):
    group_ids_all = []
    alpha_feats_all = []

    unknown_group = args.k

    mismatch_count = 0

    group_len_min = None
    group_len_max = None
    alpha_len_min = None
    alpha_len_max = None
    raw_len_min = None
    raw_len_max = None

    group_counter_all = make_empty_counter(args.k)
    group_counter_valid = make_empty_counter(args.k)
    group_counter_known_valid = make_empty_counter(args.k)

    total_positions = 0
    valid_positions = 0
    known_valid_positions = 0
    unknown_all_positions = 0
    unknown_valid_positions = 0

    transition_count = 0
    transition_base = 0

    for _, row in df.iterrows():
        concepts = parse_seq(row[concept_col])
        responses = parse_seq(row[response_col])
        concepts, responses = align_concepts_and_responses(concepts, responses)

        raw_len = len(responses)

        profile, alpha_feat, valid_mask, hist_lens = build_prefix_features(
            concepts=concepts,
            responses=responses,
            max_len=args.max_len,
            window=args.window,
            decay_lambda=args.decay_lambda,
        )

        group_ids = []

        for t, z in enumerate(profile):
            hist_len = hist_lens[t]

            if valid_mask[t] == 0:
                gid = unknown_group
            elif hist_len < args.min_hist:
                gid = unknown_group
            else:
                zz = np.asarray(z, dtype=np.float32).reshape(1, -1)
                zz = scaler.transform(zz)
                gid = int(kmeans.predict(zz)[0])

            group_ids.append(gid)

            total_positions += 1
            add_counter(group_counter_all, gid)

            if gid == unknown_group:
                unknown_all_positions += 1

            if valid_mask[t] == 1:
                valid_positions += 1
                add_counter(group_counter_valid, gid)

                if gid == unknown_group:
                    unknown_valid_positions += 1
                else:
                    known_valid_positions += 1
                    add_counter(group_counter_known_valid, gid)

        # For invalid / padding positions, keep alpha features as zeros.
        for t, a in enumerate(alpha_feat):
            if valid_mask[t] == 0:
                alpha_feat[t] = [0.0 for _ in a]

        if len(group_ids) != raw_len or len(alpha_feat) != raw_len:
            mismatch_count += 1

        if raw_len_min is None:
            raw_len_min = raw_len
            raw_len_max = raw_len
            group_len_min = len(group_ids)
            group_len_max = len(group_ids)
            alpha_len_min = len(alpha_feat)
            alpha_len_max = len(alpha_feat)
        else:
            raw_len_min = min(raw_len_min, raw_len)
            raw_len_max = max(raw_len_max, raw_len)
            group_len_min = min(group_len_min, len(group_ids))
            group_len_max = max(group_len_max, len(group_ids))
            alpha_len_min = min(alpha_len_min, len(alpha_feat))
            alpha_len_max = max(alpha_len_max, len(alpha_feat))

        # Dynamic group transition rate.
        # Only count non-unknown valid groups.
        prev = None
        for gid in group_ids:
            if gid != unknown_group:
                if prev is not None:
                    transition_base += 1
                    if gid != prev:
                        transition_count += 1
                prev = gid

        group_ids_all.append(seq_to_str(group_ids))
        alpha_feats_all.append(feat_to_str(alpha_feat))

    out_df = df.copy()
    out_df["group_ids"] = group_ids_all
    out_df["alpha_feats"] = alpha_feats_all

    out_df["dugp_group_ids"] = group_ids_all
    out_df["dugp_alpha_feats"] = alpha_feats_all

    stats = {
        "num_sequences": int(len(df)),
        "mismatch_count": int(mismatch_count),

        "raw_len_min": int(raw_len_min) if raw_len_min is not None else 0,
        "raw_len_max": int(raw_len_max) if raw_len_max is not None else 0,
        "group_len_min": int(group_len_min) if group_len_min is not None else 0,
        "group_len_max": int(group_len_max) if group_len_max is not None else 0,
        "alpha_len_min": int(alpha_len_min) if alpha_len_min is not None else 0,
        "alpha_len_max": int(alpha_len_max) if alpha_len_max is not None else 0,

        "total_positions": int(total_positions),
        "valid_positions": int(valid_positions),
        "known_valid_positions": int(known_valid_positions),

        "unknown_all_positions": int(unknown_all_positions),
        "unknown_valid_positions": int(unknown_valid_positions),

        "unknown_ratio_all_positions": float(
            unknown_all_positions / max(total_positions, 1)
        ),
        "unknown_ratio_valid_positions": float(
            unknown_valid_positions / max(valid_positions, 1)
        ),

        "group_counter_all": group_counter_all,
        "group_counter_valid": group_counter_valid,
        "group_counter_known_valid": group_counter_known_valid,

        "transition_count": int(transition_count),
        "transition_base": int(transition_base),
        "transition_rate": float(transition_count / max(transition_base, 1)),
    }

    return out_df, stats


def describe_center(row):
    """
    Coarse semantic label for cluster center.
    This is only for diagnostics and paper interpretation assistance.
    """
    hist = row["hist_len_norm"]
    pcr = row["prefix_correct_rate"]
    recent = row["recent_correct_rate"]
    vol = row["recent_volatility"]
    cwr = row["consecutive_wrong_norm"]
    gap = row["gap_since_last_same_concept_norm"]
    concept_cr = row["current_concept_correct_rate"]
    decayed = row["decayed_correct_rate"]
    risk = row["behavior_risk"]

    labels = []

    if hist < 0.25:
        labels.append("short-history")
    elif hist > 0.65:
        labels.append("long-history")
    else:
        labels.append("mid-history")

    if pcr >= 0.7 and recent >= 0.7:
        labels.append("high-cognitive")
    elif pcr <= 0.4 and recent <= 0.4:
        labels.append("low-cognitive")
    else:
        labels.append("mid-cognitive")

    if gap >= 0.75:
        labels.append("high-memory-risk")
    elif gap <= 0.35 and decayed >= 0.6:
        labels.append("low-memory-risk")
    else:
        labels.append("mid-memory-risk")

    if risk >= 0.65 or vol >= 0.45 or cwr >= 0.5:
        labels.append("high-behavior-risk")
    elif risk <= 0.35:
        labels.append("low-behavior-risk")
    else:
        labels.append("mid-behavior-risk")

    if concept_cr >= 0.7:
        labels.append("concept-strong")
    elif concept_cr <= 0.35:
        labels.append("concept-weak")

    return ";".join(labels)


def save_cluster_centers(scaler, kmeans, data_dir, args):
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    centers_df = pd.DataFrame(centers, columns=FEATURE_NAMES)
    centers_df.insert(0, "group_id", list(range(args.k)))

    centers_df["semantic_description"] = centers_df.apply(
        describe_center,
        axis=1,
    )

    centers_file = os.path.join(
        data_dir,
        f"dugp_prefix_kmeans_centers_k{args.k}.csv",
    )

    centers_df.to_csv(centers_file, index=False)

    print(f"[INFO] Saved cluster centers to: {centers_file}")

    return centers_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--data_root", type=str, default="../data")

    parser.add_argument("--fit_file", type=str, default="train_valid_sequences.csv")
    parser.add_argument(
        "--transform_files",
        type=str,
        nargs="+",
        default=[
            "train_valid_sequences.csv",
            "test_sequences.csv",
            "test_window_sequences.csv",
        ],
    )

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--unknown_group", type=int, default=8)

    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_hist", type=int, default=3)
    parser.add_argument("--decay_lambda", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    # Use these for strict train-only fitting.
    # Example:
    #   --fit_fold_col fold --fit_fold_values 0,1,2,3
    # or:
    #   --fit_fold_col split --fit_fold_values train
    parser.add_argument("--fit_fold_col", type=str, default=None)
    parser.add_argument("--fit_fold_values", type=str, default=None)

    args = parser.parse_args()

    if args.unknown_group != args.k:
        raise ValueError("This script assumes unknown_group == k.")

    data_dir = os.path.join(args.data_root, args.dataset_name)

    fit_file_path = os.path.join(data_dir, args.fit_file)

    scaler_file = os.path.join(
        data_dir,
        f"dugp_prefix_scaler_k{args.k}.pkl",
    )
    kmeans_file = os.path.join(
        data_dir,
        f"dugp_prefix_kmeans_k{args.k}.pkl",
    )
    stats_file = os.path.join(
        data_dir,
        f"dugp_prefix_group_stats_k{args.k}.json",
    )

    print(f"[INFO] Dataset directory: {data_dir}")
    print(f"[INFO] Loading fit file: {fit_file_path}")

    fit_source_df = pd.read_csv(fit_file_path)
    concept_col, response_col = detect_columns(fit_source_df)

    print(f"[INFO] Detected concept_col={concept_col}, response_col={response_col}")

    fit_df = filter_fit_df(fit_source_df, args)

    print("[INFO] Collecting train-only causal prefix features for scaler/KMeans...")

    fit_feats = collect_fit_features(
        fit_df,
        concept_col=concept_col,
        response_col=response_col,
        max_len=args.max_len,
        window=args.window,
        min_hist=args.min_hist,
        decay_lambda=args.decay_lambda,
    )

    print(f"[INFO] fit feature shape: {fit_feats.shape}")

    if fit_feats.shape[0] == 0:
        raise ValueError("No valid features collected for KMeans.")

    if fit_feats.shape[0] < args.k:
        raise ValueError(
            f"Not enough fit samples for KMeans: "
            f"num_samples={fit_feats.shape[0]}, k={args.k}"
        )

    scaler = StandardScaler()
    fit_feats_scaled = scaler.fit_transform(fit_feats)

    kmeans = KMeans(
        n_clusters=args.k,
        random_state=args.seed,
        n_init=20,
        max_iter=300,
    )
    kmeans.fit(fit_feats_scaled)

    joblib.dump(scaler, scaler_file)
    joblib.dump(kmeans, kmeans_file)

    print(f"[INFO] Saved scaler to: {scaler_file}")
    print(f"[INFO] Saved KMeans to: {kmeans_file}")

    centers_file = save_cluster_centers(
        scaler=scaler,
        kmeans=kmeans,
        data_dir=data_dir,
        args=args,
    )

    all_stats = {
        "config": {
            "dataset_name": args.dataset_name,
            "data_root": args.data_root,
            "fit_file": args.fit_file,
            "transform_files": args.transform_files,
            "k": args.k,
            "unknown_group": args.unknown_group,
            "max_len": args.max_len,
            "window": args.window,
            "min_hist": args.min_hist,
            "decay_lambda": args.decay_lambda,
            "seed": args.seed,
            "fit_fold_col": args.fit_fold_col,
            "fit_fold_values": args.fit_fold_values,
            "feature_names": FEATURE_NAMES,
            "alpha_feature_names": ALPHA_FEATURE_NAMES,
            "scaler_file": scaler_file,
            "kmeans_file": kmeans_file,
            "centers_file": centers_file,
        },
        "splits": {},
    }

    for file_name in args.transform_files:
        in_file = os.path.join(data_dir, file_name)

        base_name = os.path.basename(file_name)
        stem, ext = os.path.splitext(base_name)

        out_file = os.path.join(
            data_dir,
            f"{stem}_dugp_causal_group_k{args.k}{ext}",
        )

        split_name = stem

        print(f"[INFO] Transforming {split_name}: {in_file}")

        if not os.path.exists(in_file):
            print(f"[WARN] File does not exist, skip: {in_file}")
            continue

        df = pd.read_csv(in_file)
        c_col, r_col = detect_columns(df)

        out_df, stats = transform_df(
            df,
            concept_col=c_col,
            response_col=r_col,
            scaler=scaler,
            kmeans=kmeans,
            args=args,
        )

        out_df.to_csv(out_file, index=False)

        all_stats["splits"][split_name] = stats

        print(f"[INFO] Saved: {out_file}")
        print(f"[CHECK] {split_name} mismatch_count = {stats['mismatch_count']}")
        print(
            f"[CHECK] {split_name} raw len min/max = "
            f"{stats['raw_len_min']} / {stats['raw_len_max']}"
        )
        print(
            f"[CHECK] {split_name} group len min/max = "
            f"{stats['group_len_min']} / {stats['group_len_max']}"
        )
        print(
            f"[CHECK] {split_name} alpha len min/max = "
            f"{stats['alpha_len_min']} / {stats['alpha_len_max']}"
        )
        print(
            f"[CHECK] {split_name} unknown ratio all positions = "
            f"{stats['unknown_ratio_all_positions']:.6f}"
        )
        print(
            f"[CHECK] {split_name} unknown ratio valid positions = "
            f"{stats['unknown_ratio_valid_positions']:.6f}"
        )
        print(
            f"[CHECK] {split_name} group distribution valid = "
            f"{stats['group_counter_valid']}"
        )
        print(
            f"[CHECK] {split_name} group distribution known valid = "
            f"{stats['group_counter_known_valid']}"
        )
        print(
            f"[CHECK] {split_name} transition_rate = "
            f"{stats['transition_rate']:.6f}"
        )

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved stats to: {stats_file}")
    print("[DONE] DUGP causal group generation finished.")


if __name__ == "__main__":
    main()
