# ======================================================
# 🧠 Interactive Multi-class ROC, PR, Threshold Explorer
# ======================================================
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, f1_score
)
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider, Dropdown
from IPython.display import clear_output
from sklearn.multiclass import OneVsRestClassifier

# 1️⃣ Generate 3-class dataset
X, y = make_classification(
    n_samples=100000,
    n_features=10,
    n_informative=6,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.65, 0.3, 0.05],   # 👈 adjust imbalance here
    random_state=42
)

# 2️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3️⃣ Train classifier
# clf = LogisticRegression(multi_class='ovr', max_iter=500)
clf = OneVsRestClassifier(LogisticRegression(max_iter=500))
clf.fit(X_train, y_train)
y_scores = clf.predict_proba(X_test)   # shape (n_samples, 3)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# 4️⃣ Compute per-class ROC/PR
fpr, tpr, prec, rec, roc_auc, avg_prec = {}, {}, {}, {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
    prec[i], rec[i], _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_scores[:, i])
    avg_prec[i] = average_precision_score(y_test_bin[:, i], y_scores[:, i])
    print(f"Class {i}: ROC AUC = {roc_auc[i]:.3f} | Avg Precision = {avg_prec[i]:.3f}")

# 5️⃣ Helper function to build per-class metrics across thresholds
def compute_metrics_for_class(i):
    thresholds = np.linspace(0, 1, 201)
    records = []
    for th in thresholds:
        y_pred = (y_scores[:, i] >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_bin[:, i], y_pred).ravel()
        tpr_v = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0
        prec_v = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_v = tpr_v
        f1_v = f1_score(y_test_bin[:, i], y_pred)
        records.append((th, tpr_v, fpr_v, prec_v, rec_v, f1_v))
    data = np.array(records)
    th_vals, TPRs, FPRs, PRECs, RECs, F1s = data.T
    return th_vals, TPRs, FPRs, PRECs, RECs, F1s

# --- Helper: add vertical lines with labels ---
def add_vline_with_label(fig, x, color, label, y=0.95, line_dash="dot"):
    fig.add_vline(x=x, line_dash=line_dash, line_color=color)
    fig.add_annotation(
        x=x, y=y, text=label, showarrow=False,
        font=dict(color=color, size=12),
        xanchor="left"
    )

# --- Interactive visualization ---
def show_class_threshold(threshold, class_idx):
    clear_output(wait=True)
    print(f"🧩 Class {class_idx} — exploring threshold {threshold:.2f}")

    # Compute per-class metrics
    th_vals, TPRs, FPRs, PRECs, RECs, F1s = compute_metrics_for_class(class_idx)
    idx_youden = np.argmax(TPRs - FPRs)
    th_youden = th_vals[idx_youden]
    idx_f1 = np.argmax(F1s)
    th_f1 = th_vals[idx_f1]

    # Predict for this class at chosen threshold
    y_pred = (y_scores[:, class_idx] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_bin[:, class_idx], y_pred).ravel()
    TPR = tp / (tp + fn)
    FPR = fp / (fp + tn)
    PREC = tp / (tp + fp) if (tp + fp) > 0 else 0
    F1 = f1_score(y_test_bin[:, class_idx], y_pred)
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"TPR={TPR:.3f} | FPR={FPR:.3f} | Precision={PREC:.3f} | F1={F1:.3f}")
    print(f"(Best Youden J ≈ {th_youden:.2f}, Best F1 ≈ {th_f1:.2f})")

    # --- ROC curve ---
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr[class_idx], y=tpr[class_idx], mode="lines", name=f"Class {class_idx}"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
    add_vline_with_label(roc_fig, FPRs[idx_youden], "green", "Youden J", y=0.95)
    add_vline_with_label(roc_fig, FPRs[idx_f1], "orange", "F1-max", y=0.90)
    roc_fig.add_trace(go.Scatter(x=[FPR], y=[TPR], mode="markers", marker=dict(size=12, color="red"), name="Current"))
    roc_fig.update_layout(title=f"ROC Curve (Class {class_idx})", xaxis_title="FPR", yaxis_title="TPR", width=500, height=400)

    # --- PR curve ---
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=rec[class_idx], y=prec[class_idx], mode="lines", name=f"Class {class_idx}"))
    pr_fig.add_trace(go.Scatter(x=[RECs[idx_f1]], y=[PRECs[idx_f1]], mode="markers", marker=dict(size=10, color="orange"), name="Best F1"))
    pr_fig.add_trace(go.Scatter(x=[RECs[idx_youden]], y=[PRECs[idx_youden]], mode="markers", marker=dict(size=10, color="green"), name="Youden J"))
    pr_fig.update_layout(title=f"Precision–Recall Curve (Class {class_idx})", xaxis_title="Recall", yaxis_title="Precision", width=500, height=400)

    # --- Metric evolution ---
    metric_fig = go.Figure()
    metric_fig.add_trace(go.Scatter(x=th_vals, y=TPRs, name="TPR"))
    metric_fig.add_trace(go.Scatter(x=th_vals, y=FPRs, name="FPR"))
    metric_fig.add_trace(go.Scatter(x=th_vals, y=PRECs, name="Precision"))
    metric_fig.add_trace(go.Scatter(x=th_vals, y=F1s, name="F1", line=dict(width=3)))
    add_vline_with_label(metric_fig, th_youden, "green", "Youden J", y=0.95)
    add_vline_with_label(metric_fig, th_f1, "orange", "F1-max", y=0.90)
    metric_fig.add_trace(go.Scatter(x=[threshold], y=[F1], mode="markers", marker=dict(size=12, color="red"), name="Current"))
    metric_fig.update_layout(title="Metric Evolution vs Threshold", xaxis_title="Threshold", yaxis_title="Metric", width=700, height=400, yaxis_range=[0,1])

    # --- Histogram (simple, clean, with annotations) ---
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=y_scores[y_test==class_idx, class_idx], name=f"Class {class_idx}", opacity=0.6))
    hist_fig.add_trace(go.Histogram(x=y_scores[y_test!=class_idx, class_idx], name="Other classes", opacity=0.6))
    hist_fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    hist_fig.add_vline(x=th_youden, line_dash="dot", line_color="green")
    hist_fig.add_vline(x=th_f1, line_dash="dot", line_color="orange")
    hist_fig.add_annotation(x=threshold, y=0.95, yref="paper", text="Current", showarrow=False, font=dict(color="red"))
    hist_fig.add_annotation(x=th_youden, y=0.90, yref="paper", text="Youden J", showarrow=False, font=dict(color="green"))
    hist_fig.add_annotation(x=th_f1, y=0.85, yref="paper", text="F1-max", showarrow=False, font=dict(color="orange"))
    hist_fig.update_layout(title="Score Distribution and Thresholds", barmode="overlay", width=600, height=400)

    # --- 3×3 Confusion Matrix ---
    y_pred_multiclass = clf.predict(X_test)
    cm3 = confusion_matrix(y_test, y_pred_multiclass)
    labels = [f"Class {i}" for i in range(3)]
    cm_fig = go.Figure(data=go.Heatmap(z=cm3, text=cm3, texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm_fig.update_layout(
        title=dict(text="Confusion Matrix (3-Class)", x=0.5),
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        xaxis=dict(tickvals=[0,1,2], ticktext=labels, side='top'),
        yaxis=dict(tickvals=[0,1,2], ticktext=labels, autorange='reversed'),
        width=450,
        height=400
    )

    # ======================================================
    # 📊 Macro vs Micro Average ROC & PR Visualization
    # ======================================================

    # 1️⃣ Compute macro/micro ROC and PR curves
    # (We already have fpr, tpr, prec, rec, etc. for each class)
    # Now aggregate across classes.

    # Micro-average: flatten all one-vs-rest decisions
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_scores.ravel())
    roc_auc_micro = roc_auc_score(y_test_bin, y_scores, average='micro')

    prec_micro, rec_micro, _ = precision_recall_curve(y_test_bin.ravel(), y_scores.ravel())
    avg_prec_micro = average_precision_score(y_test_bin, y_scores, average='micro')

    # Macro-average: mean of per-class interpolated curves
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    roc_auc_macro = roc_auc_score(y_test_bin, y_scores, average='macro')

    # Similarly for PR
    all_rec = np.unique(np.concatenate([rec[i] for i in range(3)]))
    mean_prec = np.zeros_like(all_rec)
    for i in range(3):
        mean_prec += np.interp(all_rec, rec[i][::-1], prec[i][::-1])
    mean_prec /= 3
    avg_prec_macro = average_precision_score(y_test_bin, y_scores, average='macro')

    print("\n===========================================")
    print("Macro vs Micro ROC/PR Summary:")
    print(f"Micro ROC AUC = {roc_auc_micro:.3f} | Micro AP = {avg_prec_micro:.3f}")
    print(f"Macro ROC AUC = {roc_auc_macro:.3f} | Macro AP = {avg_prec_macro:.3f}")
    print("===========================================\n")

    # 2️⃣ Plot combined ROC
    roc_combined = go.Figure()

    # Per-class ROC
    for i in range(3):
        roc_combined.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode="lines", name=f"Class {i}"))

    # Micro & Macro ROC
    roc_combined.add_trace(go.Scatter(x=fpr_micro, y=tpr_micro, mode="lines", name=f"Micro-average (AUC={roc_auc_micro:.2f})", line=dict(width=3, color="black", dash="dash")))
    roc_combined.add_trace(go.Scatter(x=all_fpr, y=mean_tpr, mode="lines", name=f"Macro-average (AUC={roc_auc_macro:.2f})", line=dict(width=3, color="purple", dash="dot")))

    roc_combined.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
    roc_combined.update_layout(
        title="Macro vs Micro ROC Curves (3-Class)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700, height=500
    )

    # 3️⃣ Plot combined Precision–Recall
    pr_combined = go.Figure()

    for i in range(3):
        pr_combined.add_trace(go.Scatter(x=rec[i], y=prec[i], mode="lines", name=f"Class {i}"))

    pr_combined.add_trace(go.Scatter(x=rec_micro, y=prec_micro, mode="lines", name=f"Micro-average (AP={avg_prec_micro:.2f})", line=dict(width=3, color="black", dash="dash")))
    pr_combined.add_trace(go.Scatter(x=all_rec, y=mean_prec, mode="lines", name=f"Macro-average (AP={avg_prec_macro:.2f})", line=dict(width=3, color="purple", dash="dot")))

    pr_combined.update_layout(
        title="Macro vs Micro Precision–Recall Curves (3-Class)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=700, height=500
    )


    # 4️⃣ 🧠 Intuitive summary
    print("""
    🧠 Intuition:
    • Micro-average: aggregates all classes before computing metrics → reflects *overall* sample-level performance.
    • Macro-average: computes each class's metric first, then averages them → treats all classes *equally*.

    💡 Interpretation:
    - If Micro AUC >> Macro AUC → the model performs much better on majority classes.
    - If Macro AUC ≈ Micro AUC → performance is balanced across classes.
    """)


    # --- Show all ---
    roc_fig.show()
    pr_fig.show()
    metric_fig.show()
    hist_fig.show()
    cm_fig.show()
    roc_combined.show()
    pr_combined.show()

# --- Interactive controls ---
interact(
    show_class_threshold,
    threshold=FloatSlider(min=0, max=1, step=0.01, value=0.5, continuous_update=False),
    class_idx=Dropdown(options=[0, 1, 2], value=0, description="Class:")
);
