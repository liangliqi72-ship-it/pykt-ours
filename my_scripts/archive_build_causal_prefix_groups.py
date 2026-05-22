import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/assist2015")
K = 8
UNKNOWN_GROUP = K
MIN_HISTORY = 3
RECENT_WINDOW = 10

TRAIN_FILE = DATA_DIR / "train_valid_sequences.csv"
TEST_FILES = [
    DATA_DIR / "train_valid_sequences.csv",
    DATA_DIR / "test_sequences.csv",
    DATA_DIR / "test_window_sequences.csv",
]


def find_response_col(df):
    candidates = [
        "responses",
        "rseqs",
        "response",
        "corrects",
        "correct_seq",
        "r_seq",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    print("Available columns:", list(df.columns))
    raise ValueError("Cannot find response sequence column.")


def parse_seq(x):
    if pd.isna(x):
        return []
    parts = str(x).split(",")
    out = []
    for p in parts:
        p = p.strip()
        if p == "" or p.lower() == "nan":
            continue
        try:
            v = int(float(p))
            if v in [0, 1]:
                out.append(v)
        except Exception:
            pass
    return out


def prefix_feature(responses, t):
    """
    Build features using responses before position t.
    Do not use responses[t] or future responses.
    """
    hist = responses[:t]

    if len(hist) < MIN_HISTORY:
        return None

    arr = np.asarray(hist, dtype=np.float32)
    recent = arr[-RECENT_WINDOW:]

    hist_len = len(arr)
    correct_rate = float(arr.mean())
    wrong_rate = 1.0 - correct_rate
    recent_correct_rate = float(recent.mean())
    recent_wrong_rate = 1.0 - recent_correct_rate
    recent_volatility = float(recent.std()) if len(recent) > 1 else 0.0
    log_hist_len = float(np.log1p(hist_len))

    return [
        log_hist_len,
        correct_rate,
        wrong_rate,
        recent_correct_rate,
        recent_wrong_rate,
        recent_volatility,
    ]


def collect_train_prefix_features(train_path):
    df = pd.read_csv(train_path)
    rcol = find_response_col(df)

    feats = []
    for _, row in df.iterrows():
        responses = parse_seq(row[rcol])
        for t in range(len(responses)):
            f = prefix_feature(responses, t)
            if f is not None:
                feats.append(f)

    if len(feats) == 0:
        raise ValueError("No prefix features collected. Check response column and sequence format.")

    return np.asarray(feats, dtype=np.float32)


def build_group_sequences(csv_path, scaler, kmeans):
    df = pd.read_csv(csv_path)
    rcol = find_response_col(df)

    all_group_seqs = []
    all_alpha_feats = []

    for _, row in df.iterrows():
        responses = parse_seq(row[rcol])

        group_seq = []
        alpha_feat_seq = []

        for t in range(len(responses)):
            f = prefix_feature(responses, t)

            if f is None:
                group_seq.append(UNKNOWN_GROUP)

                # alpha_feat: [log_hist_len, recent_volatility, behavior_risk]
                # distance will be computed inside the model.
                alpha_feat_seq.append([0.0, 0.0, 1.0])
            else:
                x = scaler.transform(np.asarray([f], dtype=np.float32))
                gid = int(kmeans.predict(x)[0])
                group_seq.append(gid)

                log_hist_len = f[0]
                recent_volatility = f[5]
                behavior_risk = f[4]  # recent_wrong_rate
                alpha_feat_seq.append([
                    log_hist_len,
                    recent_volatility,
                    behavior_risk,
                ])

        all_group_seqs.append(",".join(map(str, group_seq)))
        all_alpha_feats.append(
            ";".join(",".join(f"{v:.6f}" for v in feat) for feat in alpha_feat_seq)
        )

    out = pd.DataFrame({
        "uid": df["uid"].astype(str).values if "uid" in df.columns else np.arange(len(df)),
        "group_ids": all_group_seqs,
        "alpha_feats": all_alpha_feats,
    })

    out_path = csv_path.with_name(csv_path.stem + "_causal_group_k8_unknown.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    return out_path


def main():
    print("Collecting training prefix features...")
    X_train = collect_train_prefix_features(TRAIN_FILE)
    print("Train prefix feature shape:", X_train.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    kmeans = KMeans(n_clusters=K, random_state=3407, n_init=10)
    kmeans.fit(X_scaled)

    model_path = DATA_DIR / "causal_prefix_kmeans_k8.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "kmeans": kmeans,
            "K": K,
            "unknown_group": UNKNOWN_GROUP,
            "min_history": MIN_HISTORY,
            "recent_window": RECENT_WINDOW,
        }, f)
    print("Saved:", model_path)

    for path in TEST_FILES:
        if path.exists():
            build_group_sequences(path, scaler, kmeans)
        else:
            print("Skip missing:", path)

    print("Done.")
    print("Use num_groups =", K + 1)


if __name__ == "__main__":
    main()

