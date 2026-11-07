"""BEED EEG Classification | Streamlit evaluation dashboard.

Run after generating model/*.joblib and test_data.csv with train_models.py:
    streamlit run app.py

This app never calls fit() or retrains a model. It loads trusted local joblib
artifacts and evaluates the CSV uploaded by the user.
"""
from pathlib import Path
import html

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
FEATURES = [f"X{i}" for i in range(1, 17)]
TARGET = "y"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
MODEL_ABBR = {
    "Logistic Regression": "LR",
    "Decision Tree": "DT",
    "K-Nearest Neighbors": "KNN",
    "Gaussian Naive Bayes": "NB",
    "Support Vector Machine": "SVM",
}

st.set_page_config(
    page_title="NeuroLens | EEG Model Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {--ink:#edf4ff; --muted:#9eaec6; --teal:#5eead4; --violet:#a5b4fc;}
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 85% 0%,#192e46 0%,transparent 38%),linear-gradient(135deg,#081523,#101b2f 60%,#0c1726);color:#edf4ff;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#0c1b2c;border-right:1px solid #253b50;}
[data-testid="stSidebar"] * {color:#e5edf8;}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1480px;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:#f1f7ff!important;}
h1 {font-size:2.55rem!important;font-weight:800!important;}
h3 {font-size:1.2rem!important;}
[data-testid="stMetric"] {background:#13263b;padding:18px 20px;border:1px solid #2a415a;border-radius:16px;min-height:123px;}
[data-testid="stMetricLabel"] {color:#a9bbd2!important;}
[data-testid="stMetricValue"] {color:#f3f9ff!important;font-weight:700!important;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#2b4058!important;border-radius:17px!important;}
[data-testid="stFileUploader"] {border-radius:14px;}
div.stButton > button {background:#1bcbb2;color:#042d30;border:0;border-radius:10px;font-weight:700;}
[data-testid="stTabs"] [role="tab"] {font-weight:700;color:#b5c6de;}
[data-testid="stTabs"] [aria-selected="true"] {color:#69ead1;}
.hero {padding:30px 32px;background:linear-gradient(115deg,#123f53,#1e3252 60%,#26264c);border:1px solid #426079;border-radius:22px;margin-bottom:22px;}
.hero .eyebrow {font-size:.78rem;letter-spacing:.18em;text-transform:uppercase;color:#6de4d0;font-weight:800;}
.hero h1 {margin:9px 0 7px;}
.hero p {color:#bed0e7;margin:0;max-width:740px;}
.pill {display:inline-block;padding:6px 11px;border:1px solid #477385;border-radius:99px;background:#1b4052;color:#8ef7e0;font-size:.77rem;font-weight:700;margin-right:7px;margin-top:14px;}
.section-kicker {color:#69e7cf;font-weight:800;font-size:.78rem;letter-spacing:.13em;text-transform:uppercase;margin:24px 0 3px;}
.sidebar-brand {font-size:1.5rem;font-family:'Manrope',sans-serif;font-weight:800;margin-top:8px;letter-spacing:-.04em;}
.sidebar-sub {font-size:.82rem;color:#a4b7cf;margin-top:3px;margin-bottom:22px;}
.info-card {padding:19px;border:1px solid #2c4560;border-radius:16px;background:#14283e;margin:12px 0;}
.info-card strong {color:#f1f6ff;}
.info-card span {color:#adc2d9;font-size:.88rem;}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading saved classification models...")
def load_models():
    """Load only the model artifacts created by your own trusted training script."""
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained model: model/{filename}")
        loaded[name] = joblib.load(path)
    return loaded


def load_and_validate_csv(file_obj):
    frame = pd.read_csv(file_obj)
    if frame.empty:
        raise ValueError("The uploaded CSV has no rows.")
    missing = [col for col in FEATURES + [TARGET] if col not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    inputs = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if inputs.isna().any().any() or not np.isfinite(inputs.to_numpy()).all():
        raise ValueError("EEG feature values must be complete, numeric and finite.")
    labels = frame[TARGET]
    if labels.isna().any():
        raise ValueError("Target column y contains missing labels.")
    return frame, inputs, labels


def evaluate(model, inputs, labels):
    predicted = model.predict(inputs)
    if not labels.isin(model.classes_).all():
        raise ValueError("The CSV has target labels not recognized by the saved model.")
    probabilities = model.predict_proba(inputs)
    classes = model.classes_
    if len(classes) == 2:
        target_binary = (np.asarray(labels) == classes[1]).astype(int)
        auc = (roc_auc_score(target_binary, probabilities[:, 1])
               if np.unique(target_binary).size == 2 else float("nan"))
    else:
        auc = (roc_auc_score(labels, probabilities, labels=classes,
                             multi_class="ovr", average="weighted")
               if set(labels.unique()) == set(classes) else float("nan"))
    scores = {
        "Accuracy": accuracy_score(labels, predicted),
        "AUC": auc,
        "Precision": precision_score(labels, predicted, average="weighted", zero_division=0),
        "Recall": recall_score(labels, predicted, average="weighted", zero_division=0),
        "F1": f1_score(labels, predicted, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(labels, predicted),
    }
    return scores, predicted


def draw_confusion_matrix(y_true, y_pred, classes, normalize):
    cm = confusion_matrix(y_true, y_pred, labels=classes,
                          normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    fig.patch.set_facecolor("#112338")
    ax.set_facecolor("#112338")
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    display.plot(ax=ax, cmap="BuPu", values_format=".1%" if normalize else "g",
                 colorbar=False)
    ax.set_xlabel("Predicted label", color="#c8d5e9")
    ax.set_ylabel("Actual label", color="#c8d5e9")
    ax.tick_params(colors="#c8d5e9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#53647a")
    fig.tight_layout()
    return fig


st.sidebar.markdown('<div class="sidebar-brand">🧠 NeuroLens</div><div class="sidebar-sub">EEG Classification · Model Studio</div>', unsafe_allow_html=True)
st.sidebar.markdown("### 01 / Test data")
uploaded = st.sidebar.file_uploader("Upload labeled test CSV", type=["csv"], help="Required columns: X1–X16 and y. Use test_data.csv generated by train_models.py.")
st.sidebar.markdown("### 02 / Classifier")
selected_model = st.sidebar.selectbox("Choose trained model", list(MODEL_FILES), index=0)
st.sidebar.divider()
st.sidebar.caption("BEED · 16 EEG features · 5 saved classifiers")
st.sidebar.caption("Educational research demo only. Not for clinical diagnosis.")

st.markdown(
    '<div class="hero"><div class="eyebrow">Machine learning • EEG analytics</div>'
    '<h1>NeuroLens <span style="color:#68ead4">/ Model Studio</span></h1>'
    '<p>Explore and compare five machine learning classifiers for EEG-based epilepsy classification. Upload labeled test data to inspect performance, errors, and predictions.</p>'
    '<span class="pill">5 ML MODELS</span><span class="pill">6 EVALUATION METRICS</span>'
    '<span class="pill">NO RETRAINING</span></div>', unsafe_allow_html=True,
)

if uploaded is None:
    first, second = st.columns([1.35, 1], gap="large")
    with first:
        st.markdown('<div class="section-kicker">Get started</div>', unsafe_allow_html=True)
        st.subheader("Upload your test dataset")
        st.write("Use the **CSV upload control in the sidebar** and choose the held-out `test_data.csv` generated by your training script. Every row must include 16 EEG inputs and the true target `y`.")
        st.info("Your five `.joblib` files must already exist in `model/`. The app evaluates saved models—training happens only in `train_models.py`.")
    with second:
        st.markdown('<div class="info-card"><strong>Expected CSV format</strong><br><span>X1, X2, X3, ... X16, y</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card"><strong>Included analytics</strong><br><span>Model leaderboard • 6 scores • confusion matrix • class-wise report • prediction explorer</span></div>', unsafe_allow_html=True)
    st.stop()

try:
    df, X, y = load_and_validate_csv(uploaded)
    models = load_models()
    records = []
    predictions_by_model = {}
    for name, model in models.items():
        scores, prediction = evaluate(model, X, y)
        records.append({"Model": name, **scores})
        predictions_by_model[name] = prediction
    leaderboard = pd.DataFrame(records)
except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
    st.error(f"Unable to evaluate this test dataset: {exc}")
    st.stop()

selected_scores = leaderboard.set_index("Model").loc[selected_model]
selected_predictions = predictions_by_model[selected_model]
selected_pipeline = models[selected_model]

st.markdown('<div class="section-kicker">Dataset snapshot</div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4, gap="medium")
d1.metric("Test records", f"{len(df):,}")
d2.metric("Input features", len(FEATURES))
d3.metric("Target classes", int(y.nunique()))
d4.metric("Selected model", MODEL_ABBR[selected_model])

st.markdown('<div class="section-kicker">Performance overview</div>', unsafe_allow_html=True)
st.subheader(selected_model)
metric_items = list(selected_scores.items())
for start in (0, 3):
    cols = st.columns(3, gap="medium")
    for col, (label, value) in zip(cols, metric_items[start:start + 3]):
        col.metric(label, "N/A" if pd.isna(value) else f"{value:.4f}")
st.caption("Precision, Recall and F1 use weighted averaging. AUC uses binary ROC-AUC or weighted one-vs-rest multiclass ROC-AUC as appropriate; N/A means the uploaded test set lacks the classes needed for AUC.")

tab_compare, tab_diagnostics, tab_explore = st.tabs([
    "◈  Model comparison", "▦  Error diagnostics", "▤  Data explorer"
])

with tab_compare:
    st.markdown("### All five models · same uploaded test set")
    st.dataframe(leaderboard.style.format({key: "{:.4f}" for key in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")}, na_rep="N/A"), hide_index=True, use_container_width=True)
    chart_col, context_col = st.columns([1.75, 1], gap="large")
    with chart_col:
        choice = st.selectbox("Compare models by", ["F1", "Accuracy", "AUC", "Precision", "Recall", "MCC"], index=0)
        chart_data = leaderboard.set_index("Model")[[choice]].dropna().sort_values(choice)
        st.bar_chart(chart_data, color="#54d6c7", horizontal=True, height=320)
    with context_col:
        st.markdown('<div class="section-kicker">How to interpret</div>', unsafe_allow_html=True)
        st.write("**Accuracy** measures total correct predictions. **F1** balances precision and recall. **MCC** is a useful overall correlation measure. **AUC** reflects class separation from model scores.")
        st.write("For EEG classifications, also inspect the class-wise report and confusion matrix before choosing a winner.")
    st.download_button("Download comparison as CSV", leaderboard.to_csv(index=False), "model_comparison.csv", "text/csv")

with tab_diagnostics:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("### Confusion matrix")
        normalize = st.toggle("Show row-normalized percentages", value=False)
        fig = draw_confusion_matrix(y, selected_predictions, selected_pipeline.classes_, normalize)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("### Class-wise performance")
        report = classification_report(y, selected_predictions, labels=selected_pipeline.classes_, output_dict=True, zero_division=0)
        report_frame = pd.DataFrame(report).transpose()
        st.dataframe(report_frame.style.format(precision=3), use_container_width=True)
        st.caption("Support counts how many true test examples belong to each class. The full six-metric summary is shown above.")

with tab_explore:
    st.markdown("### Prediction explorer")
    preview = df.copy()
    preview["Predicted"] = selected_predictions
    preview["Correct"] = (y.to_numpy() == np.asarray(selected_predictions))
    show_only_errors = st.checkbox("Show only misclassified rows", value=False)
    display_df = preview.loc[~preview["Correct"]] if show_only_errors else preview
    st.caption(f"Showing {len(display_df):,} of {len(preview):,} records")
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=390)
    st.download_button("Download predictions as CSV", preview.to_csv(index=False), "eeg_predictions.csv", "text/csv")

st.divider()
st.caption("NeuroLens • BEED EEG comparison • Saved models loaded from model/ • For education, not medical diagnosis")
