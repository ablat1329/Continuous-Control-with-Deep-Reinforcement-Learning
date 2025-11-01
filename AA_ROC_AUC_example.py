import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix, f1_score
)
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider
from IPython.display import clear_output

# 1️⃣ Generate dataset
X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=8,
    n_redundant=0,
    weights=[0.55, 0.45],
    random_state=42
)

# 2️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3️⃣ Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_scores = clf.predict_proba(X_test)[:, 1]

# 4️⃣ Compute curves
fpr, tpr, _ = roc_curve(y_test, y_scores)
prec, rec, _ = precision_recall_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)
avg_prec = average_precision_score(y_test, y_scores)
print(f"ROC AUC = {roc_auc:.3f} | Average Precision = {avg_prec:.3f}")

# 5️⃣ Metrics across thresholds
thresholds = np.linspace(0, 1, 201)
records = []
for th in thresholds:
    y_pred = (y_scores >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr_v = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec_v = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_v = tpr_v
    f1_v = f1_score(y_test, y_pred)
    records.append((th, tpr_v, fpr_v, prec_v, rec_v, f1_v))
data = np.array(records)
th_vals, TPRs, FPRs, PRECs, RECs, F1s = data.T

# 6️⃣ Auto-optimal thresholds
idx_youden = np.argmax(TPRs - FPRs)
th_youden = th_vals[idx_youden]
idx_f1 = np.argmax(F1s)
th_f1 = th_vals[idx_f1]
print(f"Best Youden’s J threshold = {th_youden:.2f}")
print(f"Best F1 threshold = {th_f1:.2f}")

# --- Helper: base plots ---
def add_vline_with_label(fig, x, color, label, y=0.95, line_dash="dot"):
    fig.add_vline(x=x, line_dash=line_dash, line_color=color)
    fig.add_annotation(
        x=x, y=y, text=label, showarrow=False,
        font=dict(color=color, size=12),
        xanchor="left"
    )


def base_figs():
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))

    # roc_fig.add_vline(x=FPRs[idx_youden], line_dash="dot", line_color="green", annotation_text="Youden J")
    # roc_fig.add_vline(x=FPRs[idx_f1], line_dash="dot", line_color="orange", annotation_text="F1-max")

    add_vline_with_label(roc_fig, FPRs[idx_youden], "green", "Youden J", y=0.95)
    add_vline_with_label(roc_fig, FPRs[idx_f1], "orange", "F1-max", y=0.90)

    roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=500, height=400)

    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR curve"))
    pr_fig.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision", width=500, height=400)

    metric_fig = go.Figure()
    metric_fig.add_trace(go.Scatter(x=th_vals, y=TPRs, name="TPR"))
    metric_fig.add_trace(go.Scatter(x=th_vals, y=FPRs, name="FPR"))
    metric_fig.add_trace(go.Scatter(x=th_vals, y=PRECs, name="Precision"))
    metric_fig.add_trace(go.Scatter(x=th_vals, y=RECs, name="Recall"))
    metric_fig.add_trace(go.Scatter(x=th_vals, y=F1s, name="F1-Score", line=dict(width=3)))
    # metric_fig.add_vline(x=th_youden, line_dash="dot", line_color="green", annotation_text="Youden J")
    # metric_fig.add_vline(x=th_f1, line_dash="dot", line_color="orange", annotation_text="F1-max")
    add_vline_with_label(metric_fig, th_youden, "green", "Youden J", y=0.95)
    add_vline_with_label(metric_fig, th_f1, "orange", "F1-max", y=0.90)
    metric_fig.update_layout(title="Metric Evolution vs Threshold", xaxis_title="Threshold", yaxis_title="Metric Value", width=700, height=400, yaxis_range=[0,1])
    return roc_fig, pr_fig, metric_fig

# --- Display per threshold ---
def show_threshold(threshold):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    TPR = tp / (tp + fn)
    FPR = fp / (fp + tn)
    PREC = tp / (tp + fp) if (tp + fp) > 0 else 0
    REC = TPR
    F1 = f1_score(y_test, y_pred)

    clear_output(wait=True)
    print(f"Threshold = {threshold:.2f}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"TPR={TPR:.3f} | FPR={FPR:.3f} | Precision={PREC:.3f} | Recall={REC:.3f} | F1={F1:.3f}")
    print(f"(Best Youden J ≈ {th_youden:.2f}, Best F1 ≈ {th_f1:.2f})")

    # --- Clear labeled confusion matrix with percentages ---
    total = tn + fp + fn + tp
    cm = np.array([[tn, fp], [fn, tp]])
    cm_perc = cm / total * 100
    cm_labels = [
        [f"TN: {tn} ({cm_perc[0,0]:.1f}%)", f"FP: {fp} ({cm_perc[0,1]:.1f}%)"],
        [f"FN: {fn} ({cm_perc[1,0]:.1f}%)", f"TP: {tp} ({cm_perc[1,1]:.1f}%)"]
    ]

    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        text=cm_labels,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False
    ))
    cm_fig.update_layout(
        title=dict(text="Confusion Matrix", x=0.5),
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        xaxis=dict(tickmode='array', tickvals=[0,1], ticktext=["Predicted Negative", "Predicted Positive"]),
        yaxis=dict(tickmode='array', tickvals=[0,1], ticktext=["Actual Negative", "Actual Positive"]),
        width=450,
        height=400,
        font=dict(size=12)
    )

    # --- Clear labeled confusion matrix (Predicted labels on top) ---
    total = tn + fp + fn + tp
    cm = np.array([[tn, fp], [fn, tp]])
    cm_perc = cm / total * 100

    # Each cell shows count and percentage
    cm_text = [
        [f"TN: {tn}<br>({cm_perc[0,0]:.1f}%)", f"FP: {fp}<br>({cm_perc[0,1]:.1f}%)"],
        [f"FN: {fn}<br>({cm_perc[1,0]:.1f}%)", f"TP: {tp}<br>({cm_perc[1,1]:.1f}%)"]
    ]

    cm_fig1 = go.Figure(data=go.Heatmap(
        z=cm,
        text=cm_text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False
    ))

    # ✅ Predicted on top, Actual on left
    cm_fig1.update_layout(
        title=dict(text="Confusion Matrix", x=0.5),
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=["Predicted Negative", "Predicted Positive"],
            side='top'  # 👈 places Predicted labels at the top
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=["Actual Negative", "Actual Positive"],
            autorange='reversed'  # 👈 ensures top-left = TN, bottom-right = TP
        ),
        width=450,
        height=400,
        font=dict(size=12)
    )

    # --- ROC, PR, metric, histogram plots ---
    roc_fig, pr_fig, metric_fig = base_figs()
    roc_fig.add_trace(go.Scatter(x=[FPR], y=[TPR], mode="markers", marker=dict(size=12, color="red"), name="Current"))
    pr_fig.add_trace(go.Scatter(x=[REC], y=[PREC], mode="markers", marker=dict(size=12, color="red"), name="Current"))
    metric_fig.add_trace(go.Scatter(x=[threshold], y=[F1], mode="markers", marker=dict(size=12, color="red"), name="Current"))

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=y_scores[y_test==1], name="Positive", opacity=0.6))
    hist_fig.add_trace(go.Histogram(x=y_scores[y_test==0], name="Negative", opacity=0.6))
    hist_fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Current")
    hist_fig.add_vline(x=th_youden, line_dash="dot", line_color="green", annotation_text="Youden J")
    hist_fig.add_vline(x=th_f1, line_dash="dot", line_color="orange", annotation_text="F1-max")
    hist_fig.update_layout(title="Score Distribution and Thresholds", barmode="overlay", width=600, height=400)

    # --- Show all ---
    roc_fig.show()
    pr_fig.show()
    metric_fig.show()
    hist_fig.show()
    # cm_fig.show()
    cm_fig1.show()

# --- Interactive threshold slider ---
@interact(threshold=FloatSlider(min=0, max=1, step=0.01, value=0.5, continuous_update=False))
def manual_threshold(threshold=0.5):
    show_threshold(threshold)
