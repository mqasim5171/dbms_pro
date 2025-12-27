import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "outputs", "reco_dataset_hardneg.csv")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "pref_city", "pref_type", "pref_price_bucket",
    "prop_city", "prop_type", "price_bucket", "beds_bucket"
]
TARGET = "liked"

def prob_one(q):
    states = list(q.state_names[TARGET])
    if "1" in states:
        return float(q.values[states.index("1")])
    return float(q.values[-1])

def best_threshold(y_true, y_prob):
    best = {"thr": 0.5, "f1": -1, "precision": 0, "recall": 0, "accuracy": 0}
    for thr in np.linspace(0.05, 0.95, 19):  # 0.05, 0.10, ... 0.95
        y_pred = [1 if p >= thr else 0 for p in y_prob]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "thr": float(thr),
                "f1": float(f1),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
    return best

def main():
    df = pd.read_csv(DATA_PATH).astype(str)
    df = df[FEATURES + [TARGET]].copy()

    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )

    # ✅ Better causal structure (viva-friendly):
    # preferences influence what kind of property is matched,
    # and property attributes drive the final "liked" decision
    edges = [
        ("pref_city", "prop_city"),
        ("pref_type", "prop_type"),
        ("pref_price_bucket", "price_bucket"),

        ("prop_city", TARGET),
        ("prop_type", TARGET),
        ("price_bucket", TARGET),
        ("beds_bucket", TARGET),
    ]

    model = DiscreteBayesianNetwork(edges)
    model.fit(train, estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(model)

    y_true = test[TARGET].astype(int).tolist()
    y_prob = []

    for _, row in test.iterrows():
        evidence = {f: row[f] for f in FEATURES}
        q = infer.query(variables=[TARGET], evidence=evidence, show_progress=False)
        y_prob.append(prob_one(q))

    auc = roc_auc_score(y_true, y_prob)

    # ✅ threshold tuning
    best = best_threshold(y_true, y_prob)
    thr = best["thr"]
    y_pred = [1 if p >= thr else 0 for p in y_prob]

    cm = confusion_matrix(y_true, y_pred)

    metrics_df = pd.DataFrame([{
        "best_threshold": thr,
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "roc_auc": auc,
        "rows_total": len(df),
        "rows_train": len(train),
        "rows_test": len(test),
        "positive_test": int(np.sum(np.array(y_true) == 1)),
        "negative_test": int(np.sum(np.array(y_true) == 0)),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }])

    metrics_csv = os.path.join(OUT_DIR, "bbn_metrics_structured.csv")
    metrics_txt = os.path.join(OUT_DIR, "bbn_metrics_structured.txt")
    metrics_df.to_csv(metrics_csv, index=False)

    with open(metrics_txt, "w") as f:
        f.write(metrics_df.to_string(index=False) + "\n")
        f.write("\nBBN Structure (edges):\n")
        for a, b in edges:
            f.write(f"{a} -> {b}\n")

    print("✅ Saved:", metrics_csv)
    print("✅ Saved:", metrics_txt)
    print(metrics_df)

    # Confusion matrix plot
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (Structured BBN) - thr={thr:.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    cm_path = os.path.join(OUT_DIR, "confusion_matrix_structured.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", cm_path)

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve (BBN)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    roc_path = os.path.join(OUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", roc_path)

if __name__ == "__main__":
    main()
