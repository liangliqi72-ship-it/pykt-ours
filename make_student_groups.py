import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data_dir = "data/assist2015"
seq_path = os.path.join(data_dir, "train_valid_sequences.csv")
out_path = os.path.join(data_dir, "student_group_k8.csv")

num_groups = 8

df = pd.read_csv(seq_path)

if "uid" not in df.columns:
    raise ValueError(
        "train_valid_sequences.csv does not contain uid column. "
        "Please check the csv header first."
    )

features = []

for uid, g in df.groupby("uid"):
    all_responses = []
    all_usetimes = []

    for _, row in g.iterrows():
        responses = [int(x) for x in str(row["responses"]).split(",") if x != ""]
        selectmasks = [int(x) for x in str(row["selectmasks"]).split(",") if x != ""]

        valid_responses = []
        for r, m in zip(responses, selectmasks):
            if m != -1:
                valid_responses.append(r)

        all_responses.extend(valid_responses)

        if "usetimes" in df.columns and not pd.isna(row["usetimes"]):
            usetimes = [float(x) for x in str(row["usetimes"]).split(",") if x != ""]
            valid_usetimes = []
            for t, m in zip(usetimes, selectmasks):
                if m != -1:
                    valid_usetimes.append(t)
            all_usetimes.extend(valid_usetimes)

    if len(all_responses) == 0:
        avg_correct_rate = 0.0
        wrong_rate = 1.0
        recent_correct_rate = 0.0
        interaction_count = 0
    else:
        arr = np.array(all_responses, dtype=float)
        avg_correct_rate = arr.mean()
        wrong_rate = 1.0 - avg_correct_rate
        interaction_count = len(arr)
        recent_correct_rate = arr[-min(20, len(arr)):].mean()

    if len(all_usetimes) > 0:
        avg_response_time = np.mean(all_usetimes)
    else:
        avg_response_time = 0.0

    features.append([
        uid,
        avg_correct_rate,
        avg_response_time,
        np.log1p(interaction_count),
        wrong_rate,
        recent_correct_rate
    ])

feat_df = pd.DataFrame(
    features,
    columns=[
        "uid",
        "avg_correct_rate",
        "avg_response_time",
        "interaction_count",
        "wrong_rate",
        "recent_correct_rate"
    ]
)

X = feat_df[
    [
        "avg_correct_rate",
        "avg_response_time",
        "interaction_count",
        "wrong_rate",
        "recent_correct_rate"
    ]
].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=num_groups,
    random_state=42,
    n_init=10
)

group_ids = kmeans.fit_predict(X_scaled)

out_df = pd.DataFrame({
    "uid": feat_df["uid"],
    "group_id": group_ids
})

out_df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(out_df["group_id"].value_counts().sort_index())

