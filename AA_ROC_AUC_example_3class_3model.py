# ======================================================
# 🧠 Multi-Model Interactive ROC–PR–Threshold Explorer
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
from ipywidgets import interact, FloatSlider, Dropdown
from IPython.display import clear_output

# 1️⃣ Create imbalanced 3-class dataset
X, y = make_classification(
    n_samples=8000,
    n_features=12,
    n_informative=5,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.65, 0.25, 0.10],  # imbalance here
    random_state=42
)

# 2️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# 3️⃣ Train all models
models = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=500)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": OneVsRestClassifier(XGBClassifier(
        n_estimators=250, learning_rate=0.1, max_depth=4,
        subsample=0.8, eval_metric='logloss', use_label_encoder=False,
        random_state=42
    ))
}

model_data = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)
    model_data[name] = {"model": model, "scores": y_scores}

print("✅ All models trained.\n")

# 4️⃣ Helper metrics per class
def compute_metrics(y_true_bin, y_scores):
    fpr, tpr, prec, rec, roc_auc, avg_prec = {}, {}, {}, {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        prec[i], rec[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
        avg_prec[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    return fpr, tpr, prec, rec, roc_auc, avg_prec

# Precompute all metrics
for name, data in model_data.items():
    data["fpr"], data["tpr"], data["prec"], data["rec"], data["roc_auc"], data["avg_prec"] = compute_metrics(y_test_bin, data["scores"])

# 5️⃣ Metric computation helper
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
    arr = np.array(recs)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

# Helper to add labeled vertical lines
def add_vline(fig, x, color, text, y=0.95):
    fig.add_vline(x=x, line_dash="dot", line_color=color)
    fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                       font=dict(color=color, size=12), xanchor="left")

# 6️⃣ Interactive visualization
def explore_models(threshold, class_idx):
    clear_output(wait=True)
    print(f"🎯 Class {class_idx} | Threshold = {threshold:.2f}")
    figs = []
    colors = {"Logistic Regression": "royalblue", "Random Forest": "forestgreen", "XGBoost": "darkorange"}

    # Compare each model
    for model_name, data in model_data.items():
        scores = data["scores"]
        model = data["model"]
        fpr, tpr, prec, rec = data["fpr"], data["tpr"], data["prec"], data["rec"]

        # Metric evolution
        th, TPRs, FPRs, PRECs, F1s = compute_metrics_for_class(y_test_bin, scores, class_idx)
        idx_youden = np.argmax(TPRs - FPRs)
        idx_f1 = np.argmax(F1s)
        th_youden, th_f1 = th[idx_youden], th[idx_f1]

        # Current threshold metrics
        y_pred = (scores[:, class_idx] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_bin[:, class_idx], y_pred).ravel()
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)
        PREC = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = f1_score(y_test_bin[:, class_idx], y_pred)
        print(f"📊 {model_name:18} | TPR={TPR:.3f} | FPR={FPR:.3f} | Prec={PREC:.3f} | F1={F1:.3f}")

        # === Build subplots side-by-side ===
        # ROC
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr[class_idx], y=tpr[class_idx],
                                     mode="lines", name=f"{model_name}", line=dict(color=colors[model_name])))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="gray"), name="Random"))
        roc_fig.add_trace(go.Scatter(x=[FPR], y=[TPR], mode="markers", marker=dict(size=10, color="red"), name="Current"))
        roc_fig.update_layout(title=f"ROC – {model_name}", xaxis_title="FPR", yaxis_title="TPR", width=400, height=350)

        # PR
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=rec[class_idx], y=prec[class_idx],
                                    mode="lines", name=f"{model_name}", line=dict(color=colors[model_name])))
        pr_fig.add_trace(go.Scatter(x=[TPR], y=[PREC], mode="markers",
                                    marker=dict(size=10, color="red"), name="Current"))
        pr_fig.update_layout(title=f"PR Curve – {model_name}", xaxis_title="Recall", yaxis_title="Precision", width=400, height=350)

        # Metric evolution
        metric_fig = go.Figure()
        metric_fig.add_trace(go.Scatter(x=th, y=TPRs, name="TPR"))
        metric_fig.add_trace(go.Scatter(x=th, y=FPRs, name="FPR"))
        metric_fig.add_trace(go.Scatter(x=th, y=PRECs, name="Precision"))
        metric_fig.add_trace(go.Scatter(x=th, y=F1s, name="F1", line=dict(width=3)))
        add_vline(metric_fig, th_youden, "green", "Youden J", y=0.95)
        add_vline(metric_fig, th_f1, "orange", "F1-max", y=0.90)
        metric_fig.update_layout(title=f"Metric Evolution – {model_name}", xaxis_title="Threshold", yaxis_title="Metric", width=500, height=350, yaxis_range=[0, 1])

        # Histogram
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=scores[y_test == class_idx, class_idx],
                                        name=f"Class {class_idx}", opacity=0.6))
        hist_fig.add_trace(go.Histogram(x=scores[y_test != class_idx, class_idx],
                                        name="Others", opacity=0.6))
        hist_fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        hist_fig.update_layout(title=f"Score Distribution – {model_name}",
                               barmode="overlay", width=500, height=350)

        # Confusion Matrix (3×3)
        y_pred_multi = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_multi)
        labels = [f"Class {i}" for i in range(3)]
        cm_fig = go.Figure(data=go.Heatmap(z=cm, text=cm, texttemplate="%{text}",
                                           colorscale="Blues", showscale=False))
        cm_fig.update_layout(
            title=dict(text=f"Confusion Matrix – {model_name}", x=0.5),
            xaxis=dict(tickvals=[0, 1, 2], ticktext=labels, side="top"),
            yaxis=dict(tickvals=[0, 1, 2], ticktext=labels, autorange="reversed"),
            width=400, height=350
        )

        # Display all together
        roc_fig.show()
        pr_fig.show()
        metric_fig.show()
        hist_fig.show()
        cm_fig.show()

# 7️⃣ Interactive controls
interact(
    explore_models,
    threshold=FloatSlider(min=0, max=1, step=0.01, value=0.5, continuous_update=False),
    class_idx=Dropdown(options=[0, 1, 2], value=0, description="Class:")
);
