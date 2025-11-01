# ============================================================
# 🧠 Multi-Model, Multi-Class Visualization — Clean Layout
# ============================================================
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
from IPython.display import clear_output, display

# --- 1️⃣ Create imbalanced dataset ---
X, y = make_classification(
    n_samples=9000,
    n_features=12,
    n_informative=5,
    n_redundant=0,
    n_classes=3,
    weights=[0.65, 0.25, 0.10],
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# --- 2️⃣ Define models ---
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

# --- 3️⃣ Train models ---
model_data = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)
    model_data[name] = {"model": model, "scores": y_scores}
print("✅ Training complete.\n")

# --- 4️⃣ Compute per-class ROC/PR ---
def compute_curves(y_true_bin, y_scores):
    fpr, tpr, prec, rec, roc_auc, avg_prec = {}, {}, {}, {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        prec[i], rec[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
        avg_prec[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    return fpr, tpr, prec, rec, roc_auc, avg_prec

for name, data in model_data.items():
    data["fpr"], data["tpr"], data["prec"], data["rec"], data["roc_auc"], data["avg_prec"] = compute_curves(y_test_bin, data["scores"])



# --- 5️⃣ Metric evolution per class ---
# --- Helper: compute metrics array once per model/class to reuse ---
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
    arr = np.array(recs).T  # shape (5, n_thresh)
    return arr  # rows: th, TPR, FPR, PREC, F1

# cache metric evolution per model/class to avoid recompute every call
_metric_cache = {}
def get_metric_evolution(model_name, scores, class_idx):
    key = (model_name, class_idx)
    if key not in _metric_cache:
        _metric_cache[key] = compute_metrics_for_class(y_test_bin, scores, class_idx)
    return _metric_cache[key]

# --- Updated explore_class with legend placement, F1/Youden lines, and red dots ---
# --- Helper: base plots ---
def add_vline_with_label(fig, x, color, label, y=0.95, line_dash="dot"):
    fig.add_vline(x=x, line_dash=line_dash, line_color=color)
    fig.add_annotation(
        x=x, y=y, text=label, showarrow=False,
        font=dict(color=color, size=12),
        xanchor="left"
    )

# --- Main interactive function ---
def explore_class(threshold, class_idx):
    clear_output(wait=True)
    print(f"🎯 Class {class_idx} | Threshold = {threshold:.2f}")

    # ==============================
    # 1️⃣ ROC + Precision–Recall
    # ==============================
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ROC Curve", "Precision–Recall Curve"),
        horizontal_spacing=0.08
    )

    metrics_table = []
    roc_traces, pr_traces = [], []
    ys = np.linspace(0.55, 0.05, len(model_data))  # stagger Youden labels

    for name, data in model_data.items():

        color = colors[name]
        scores = data["scores"]
        fpr, tpr, prec, rec = data["fpr"], data["tpr"], data["prec"], data["rec"]

        # --- ROC ---
        roc_traces.append(go.Scatter(
            x=fpr[class_idx], y=tpr[class_idx],
            mode="lines", name=name,
            line=dict(color=color, width=2)
        ))

        # --- Current threshold marker ---
        y_pred_th = (scores[:, class_idx] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_bin[:, class_idx], y_pred_th).ravel()
        tpr_pt = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0
        roc_traces.append(go.Scatter(
            x=[fpr_pt], y=[tpr_pt],
            mode="markers", showlegend=False,
            marker=dict(color="red", size=8)
        ))

        # --- Youden J marker + label on ROC ---
        th_arr, TPRs, FPRs, PRECs, F1s = get_metric_evolution(name, scores, class_idx)
        idx_youden = np.nanargmax(TPRs - FPRs)
        roc_traces.append(go.Scatter(
            x=[FPRs[idx_youden]], y=[TPRs[idx_youden]],
            mode="markers", showlegend=False,
            marker=dict(color=color, symbol="x", size=10, line=dict(width=2))
        ))

        # --- PR ---
        pr_traces.append(go.Scatter(
            x=rec[class_idx], y=prec[class_idx],
            mode="lines", name=name,
            line=dict(color=color, width=2)
        ))
        prec_pt = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_pt = tp / (tp + fn) if (tp + fn) > 0 else 0
        pr_traces.append(go.Scatter(
            x=[rec_pt], y=[prec_pt],
            mode="markers", showlegend=False,
            marker=dict(color="red", size=8)
        ))

        # --- Metrics table ---
        auc = data["roc_auc"][class_idx]
        ap = data["avg_prec"][class_idx]
        f1_val = f1_score(y_test_bin[:, class_idx], y_pred_th)
        metrics_table.append((name, auc, ap, f1_val))

    # --- Random reference line ---
    roc_traces.append(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random"
    ))

    for tr in roc_traces: fig.add_trace(tr, row=1, col=1)
    for tr in pr_traces: fig.add_trace(tr, row=1, col=2)

    fig.update_xaxes(title_text="FPR", row=1, col=1)
    fig.update_yaxes(title_text="TPR", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)

    fig.update_layout(
        title=f"📊 ROC and Precision–Recall Curves — Class {class_idx}",
        width=1000, height=500,
        margin=dict(t=80, b=140),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    )
    fig.for_each_annotation(lambda a: a.update(font=dict(size=13)))
    fig.show()

    # ==============================
    # 2️⃣ Metric Evolution (Standalone)
    # ==============================
    metric_fig = go.Figure()
    ys = np.linspace(0.55, 0.05, len(model_data))  # stagger Youden labels

    for i, (name, data) in enumerate(model_data.items()):
        color = colors[name]
        scores = data["scores"]
        th_arr, TPRs, FPRs, PRECs, F1s = get_metric_evolution(name, scores, class_idx)

        # --- Curves ---
        metric_fig.add_trace(go.Scatter(
            x=th_arr, y=TPRs, name=f"{name} TPR", line=dict(color=color)
        ))
        metric_fig.add_trace(go.Scatter(
            x=th_arr, y=F1s, name=f"{name} F1", line=dict(color=color, dash="dot")
        ))

        # --- Youden J vertical line with helper ---
        idx_youden = np.nanargmax(TPRs - FPRs)
        add_vline_with_label(metric_fig, x=th_arr[idx_youden], color=color,
                             label=f"{name} Youden J", y=ys[i], line_dash="dash")

        # --- Red dots for current threshold ---
        idx_th = np.argmin(np.abs(th_arr - threshold))
        metric_fig.add_trace(go.Scatter(
            x=[th_arr[idx_th]], y=[TPRs[idx_th]],
            mode="markers", marker=dict(color="red", size=8), showlegend=False
        ))
        metric_fig.add_trace(go.Scatter(
            x=[th_arr[idx_th]], y=[F1s[idx_th]],
            mode="markers", marker=dict(color="red", size=8), showlegend=False
        ))

    metric_fig.update_layout(
        title=f"📈 Metric Evolution vs Threshold — Class {class_idx}",
        xaxis_title="Threshold",
        yaxis_title="Metric Value",
        width=950, height=550,
        legend=dict(
            orientation="v", x=1.05, y=0.5,
            xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(t=80, r=220)
    )
    metric_fig.show()

    # ==============================
    # 3️⃣ Histogram
    # ==============================
    hist_fig = go.Figure()
    for name, data in model_data.items():
        color = colors[name]
        hist_fig.add_trace(go.Histogram(
            x=data["scores"][y_test == class_idx, class_idx],
            name=f"{name} | Class {class_idx}",
            opacity=0.6, marker_color=color
        ))
    hist_fig.update_layout(
        title=f"Score Distributions for Class {class_idx}",
        barmode="overlay", width=800, height=400,
        xaxis_title="Predicted Probability", yaxis_title="Count",
        legend=dict(orientation="v", y=0.7, x=1.2, xanchor="center")
    )
    hist_fig.show()

    # ==============================
    # 4️⃣ Confusion Matrices
    # ==============================
    cm_fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.05,
                           subplot_titles=tuple(models.keys()))
    labels = [f"Class {i}" for i in range(3)]
    for idx, (model_name, data) in enumerate(model_data.items(), start=1):
        model = data["model"]
        y_pred_multi = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_multi)
        cm_fig.add_trace(go.Heatmap(
            z=cm, text=cm, texttemplate="%{text}",
            colorscale="Blues", showscale=False
        ), row=1, col=idx)
        cm_fig.update_xaxes(
            tickvals=[0, 1, 2], ticktext=labels,
            title_text="Predicted Label", side="bottom", row=1, col=idx
        )
        cm_fig.update_yaxes(
            tickvals=[0, 1, 2], ticktext=labels,
            title_text="Actual Label" if idx == 1 else "",
            autorange="reversed", row=1, col=idx
        )
    cm_fig.update_layout(
    title=f"Confusion Matrices (All Models) — Class {class_idx}",
    width=1000, height=350, margin=dict(t=80),
    )
    cm_fig.show()

    # ==============================
    # 5️⃣ Summary Table
    # ==============================
    df = pd.DataFrame(
        metrics_table,
        columns=["Model", "ROC AUC", "Avg Precision", f"F1@{threshold:.2f}"]
    )
    display(df.style.background_gradient(cmap="Blues").format(precision=3))

    # ==============================
    # 6️⃣ Textual Summary
    # ==============================
    for name, data in model_data.items():
        scores = data["scores"]
        th_arr, TPRs, FPRs, PRECs, F1s = get_metric_evolution(name, scores, class_idx)

        # Best thresholds
        best_youden_idx = np.nanargmax(TPRs - FPRs)
        best_youden_th = th_arr[best_youden_idx]
        best_f1_idx = np.nanargmax(F1s)
        best_f1_th = th_arr[best_f1_idx]

        # Confusion matrix at current threshold
        y_pred_th = (scores[:, class_idx] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_bin[:, class_idx], y_pred_th).ravel()
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_val = f1_score(y_test_bin[:, class_idx], y_pred_th)

        # Print summary
        print("\n"+'='*60)
        print(f"Model: {name} | Class {class_idx} | Threshold = {threshold:.2f}")
        print("="*60)
        print(f"ROC AUC = {data['roc_auc'][class_idx]:.3f} | Average Precision = {data['avg_prec'][class_idx]:.3f}")
        print(f"Best Youden’s J threshold = {best_youden_th:.2f}")
        print(f"Best F1 threshold = {best_f1_th:.2f}")
        print(f"Threshold = {threshold:.2f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"TPR={tpr_val:.3f} | FPR={fpr_val:.3f} | Precision={precision_val:.3f} | Recall={tpr_val:.3f} | F1={f1_val:.3f}")
        print(f"(Best Youden J ≈ {best_youden_th:.2f}, Best F1 ≈ {best_f1_th:.2f})\n")

# --- 7️⃣ Interactive Controls (same as before) ---
interact(
    explore_class,
    threshold=FloatSlider(min=0, max=1, step=0.01, value=0.5, continuous_update=False),
    class_idx=Dropdown(options=[0,1,2], value=0, description="Class:")
);
