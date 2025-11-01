# ======================================================
# 🧠 Multi-Model ROC–PR–Threshold Explorer (Merged Plots)
# ======================================================
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, f1_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact, FloatSlider, Dropdown
from IPython.display import clear_output

# 1️⃣ Create imbalanced dataset
X, y = make_classification(
    n_samples=8000,
    n_features=12,
    n_informative=5,
    n_redundant=0,
    n_classes=3,
    weights=[0.65, 0.25, 0.10],  # imbalance here
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# 2️⃣ Define models
models = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=500)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": OneVsRestClassifier(XGBClassifier(
        n_estimators=250, learning_rate=0.1, max_depth=4,
        subsample=0.8, eval_metric='logloss',
        random_state=42
    ))
}

colors = {
    "Logistic Regression": "royalblue",
    "Random Forest": "forestgreen",
    "XGBoost": "darkorange"
}

# 3️⃣ Train models
model_data = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)
    model_data[name] = {"model": model, "scores": y_scores}
print("✅ Training complete.\n")

# 4️⃣ Compute per-class metrics
def compute_metrics(y_true_bin, y_scores):
    fpr, tpr, prec, rec, roc_auc, avg_prec = {}, {}, {}, {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        prec[i], rec[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
        avg_prec[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    return fpr, tpr, prec, rec, roc_auc, avg_prec

for name, data in model_data.items():
    data["fpr"], data["tpr"], data["prec"], data["rec"], data["roc_auc"], data["avg_prec"] = compute_metrics(y_test_bin, data["scores"])

# 5️⃣ Metric evolution helper
def compute_metrics_for_class(y_true_bin, y_scores, class_idx):
    thresholds = np.linspace(0, 1, 201)
    recs = []
    for th in thresholds:
        y_pred = (y_scores[:, class_idx] >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin[:, class_idx], y_pred).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        PREC = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = f1_score(y_true_bin[:, class_idx], y_pred)
        recs.append((th, TPR, FPR, PREC, F1))
    return np.array(recs).T  # th, TPR, FPR, PREC, F1

# 6️⃣ Add line + label
def add_vline(fig, x, color, text, y=0.9):
    fig.add_vline(x=x, line_dash="dot", line_color=color)
    fig.add_annotation(x=x, y=y, yref="paper", text=text, showarrow=False,
                       font=dict(color=color, size=12), xanchor="left")

# 7️⃣ Interactive explorer
def explore_class(threshold, class_idx):
    clear_output(wait=True)
    print(f"🎯 Class {class_idx} | Threshold = {threshold:.2f}")

    # Shared subplots (3 in row + 1 below)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("ROC Curve", "Precision–Recall Curve", "Metric Evolution", "", "", "Score Distribution"),
        specs=[[{"type":"xy"}, {"type":"xy"}, {"type":"xy"}],
               [{"colspan":3, "type":"xy"}, None, None]],
        vertical_spacing=0.18
    )

    # Loop models and add traces
    for name, data in model_data.items():
        scores = data["scores"]
        fpr, tpr, prec, rec = data["fpr"], data["tpr"], data["prec"], data["rec"]
        color = colors[name]

        # ROC
        fig.add_trace(go.Scatter(x=fpr[class_idx], y=tpr[class_idx], mode="lines",
                                 name=name, line=dict(color=color, width=2)),
                      row=1, col=1)
        # PR
        fig.add_trace(go.Scatter(x=rec[class_idx], y=prec[class_idx], mode="lines",
                                 name=name, showlegend=False, line=dict(color=color, width=2)),
                      row=1, col=2)

        # Metric evolution
        th, TPRs, FPRs, PRECs, F1s = compute_metrics_for_class(y_test_bin, scores, class_idx)
        fig.add_trace(go.Scatter(x=th, y=TPRs, mode="lines", name=f"{name} TPR", line=dict(color=color, dash="solid")),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=th, y=F1s, mode="lines", name=f"{name} F1", line=dict(color=color, dash="dot")),
                      row=1, col=3)

        # Histogram
        fig.add_trace(go.Histogram(x=scores[y_test == class_idx, class_idx],
                                   name=f"{name} | Class {class_idx}", opacity=0.5,
                                   marker_color=color),
                      row=2, col=1)
        fig.add_trace(go.Histogram(x=scores[y_test != class_idx, class_idx],
                                   name=f"{name} | Other", opacity=0.3,
                                   marker_color=color, showlegend=False),
                      row=2, col=1)

    # --- Baseline diagonal for ROC ---
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"),
                  row=1, col=1)

    # --- Style ---
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_xaxes(title_text="Threshold", row=1, col=3)
    fig.update_yaxes(title_text="Metric", row=1, col=3, range=[0,1])
    fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_layout(
        title=f"📊 Multi-Model Comparison for Class {class_idx} @ Threshold {threshold:.2f}",
        height=800, width=1200,
        barmode="overlay",
        legend=dict(orientation="h", y=-0.2),
    )

    fig.show()

# 8️⃣ Run interactively
interact(
    explore_class,
    threshold=FloatSlider(min=0, max=1, step=0.01, value=0.5, continuous_update=False),
    class_idx=Dropdown(options=[0,1,2], value=0, description="Class:")
);
