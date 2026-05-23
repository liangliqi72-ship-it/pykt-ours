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
SEQ_FILES = [
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

    out = []
    for p in str(x).split(","):
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

    for idx, row in df.iterrows():
        responses = parse_seq(row[rcol])

        for t in range(len(responses)):
            f = prefix_feature(responses, t)
            if f is not None:
                feats.append(f)

        if (idx + 1) % 500 == 0:
            print(f"Collected train rows: {idx + 1}/{len(df)}")

    if len(feats) == 0:
        raise ValueError("No prefix features collected.")

    return np.asarray(feats, dtype=np.float32)


def build_group_sequences_fast(csv_path, scaler, kmeans):
    df = pd.read_csv(csv_path)
    rcol = find_response_col(df)

    print(f"Building causal groups for: {csv_path}")
    print("Rows:", len(df))

    all_group_seqs = []
    all_alpha_feats = []

    for idx, row in df.iterrows():
        responses = parse_seq(row[rcol])

        valid_positions = []
        valid_features = []

        group_seq = [UNKNOWN_GROUP] * len(responses)
        alpha_feat_seq = [[0.0, 0.0, 1.0] for _ in responses]

        for t in range(len(responses)):
            f = prefix_feature(responses, t)

            if f is not None:
                valid_positions.append(t)
                valid_features.append(f)

                log_hist_len = f[0]
                recent_volatility = f[5]
                behavior_risk = f[4]
                alpha_feat_seq[t] = [
                    log_hist_len,
                    recent_volatility,
                    behavior_risk,
                ]

        if len(valid_features) > 0:
            X = np.asarray(valid_features, dtype=np.float32)
            X_scaled = scaler.transform(X)
            gids = kmeans.predict(X_scaled)

            for pos, gid in zip(valid_positions, gids):
                group_seq[pos] = int(gid)

        all_group_seqs.append(",".join(map(str, group_seq)))
        all_alpha_feats.append(
            ";".join(",".join(f"{v:.6f}" for v in feat) for feat in alpha_feat_seq)
        )

        if (idx + 1) % 500 == 0:
            print(f"Processed rows: {idx + 1}/{len(df)}")

    out = pd.DataFrame({
        "uid": df["uid"].astype(str).values if "uid" in df.columns else np.arange(len(df)),
        "group_ids": all_group_seqs,
        "alpha_feats": all_alpha_feats,
    })

    out_path = csv_path.with_name(csv_path.stem + "_causal_group_k8_unknown.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)


def main():
    model_path = DATA_DIR / "causal_prefix_kmeans_k8.pkl"

    if model_path.exists():
        print("Loading existing KMeans:", model_path)
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        scaler = obj["scaler"]
        kmeans = obj["kmeans"]
    else:
        print("Collecting training prefix features...")
        X_train = collect_train_prefix_features(TRAIN_FILE)
        print("Train prefix feature shape:", X_train.shape)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        kmeans = KMeans(n_clusters=K, random_state=3407, n_init=10)
        kmeans.fit(X_scaled)

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

    for path in SEQ_FILES:
        if path.exists():
            build_group_sequences_fast(path, scaler, kmeans)
        else:
            print("Skip missing:", path)

    print("Done.")
    print("Use num_groups =", K + 1)


if __name__ == "__main__":
    main()

