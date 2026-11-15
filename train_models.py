"""BITS ML Assignment 2 — BEED EEG classification.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# 1. PROJECT CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "BEED_Data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
FEATURES = [f"X{i}" for i in range(1, 17)]
TARGET = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.20
SUBJECT_COLUMN = None  # Replace only with a verified subject/recording group column.


def heading(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def calculate_metrics(model, x_eval, y_eval):
    """Return six required metrics and predictions for any supported class count."""
    predictions = model.predict(x_eval)
    probabilities = model.predict_proba(x_eval)
    labels = model.classes_
    actual_classes = set(pd.unique(y_eval))

    if len(labels) == 2:
        binary_target = (np.asarray(y_eval) == labels[1]).astype(int)
        auc = (
            roc_auc_score(binary_target, probabilities[:, 1])
            if len(np.unique(binary_target)) == 2
            else float("nan")
        )
    elif actual_classes == set(labels):
        auc = roc_auc_score(
            y_eval,
            probabilities,
            labels=labels,
            multi_class="ovr",
            average="weighted",
        )
    else:
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy_score(y_eval, predictions),
        "AUC": auc,
        "Precision": precision_score(y_eval, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_eval, predictions, average="weighted", zero_division=0),
        "F1": f1_score(y_eval, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_eval, predictions),
    }
    return metrics, predictions


def main():
    # 2. LOAD AND VALIDATE THE ORIGINAL DATASET
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET] + ([SUBJECT_COLUMN] if SUBJECT_COLUMN else [])
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}. Inspect your CSV header.")
    if len(df) < 500 or len(FEATURES) < 12:
        raise ValueError("Dataset does not meet the 500-row/12-feature minimum.")

    group_candidates = [
        col for col in df.columns
        if any(word in col.lower() for word in (
            "subject", "patient", "person", "participant", "recording", "session"
        ))
    ]
    if group_candidates and SUBJECT_COLUMN is None:
        raise ValueError(
            f"Possible subject/recording identifiers: {group_candidates}. "
            "Review them; set SUBJECT_COLUMN if appropriate."
        )

    x = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET]
    if y.isna().any() or y.nunique() < 2:
        raise ValueError("Target labels must contain at least two classes and no missing values.")
    if x.isna().any().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Features contain missing, non-numeric or infinite values. Inspect the CSV.")

    heading("A. DATASET DESCRIPTION")
    print("Dataset: BEED - Bangalore EEG Epilepsy Dataset")
    print("Source: UCI Machine Learning Repository")
    print("Original dataset rows:", len(df))
    print("Total dataset columns:", len(df.columns))
    print("Number of model input features:", len(FEATURES))
    print("Feature names:", ", ".join(FEATURES))
    print("Target:", TARGET)
    print("Number of classes:", y.nunique())
    print("Target label frequencies:\n" + y.value_counts(dropna=False).to_string())
    print("Missing values (entire dataset):", int(df.isna().sum().sum()))
    print("Duplicate rows (whole-row matches):", int(df.duplicated().sum()))
    print("Missing predictor values:", int(x.isna().sum().sum()))
    print("Subject/group identifier used:", SUBJECT_COLUMN or "None available/configured")
    print("NOTE: Duplicate observations were counted, not automatically discarded.")

    # 3. SPLIT THE COMPLETE DATASET — DO NOT SAMPLE/DISCARD ROWS
    if SUBJECT_COLUMN is not None:
        groups = df[SUBJECT_COLUMN]
        if groups.isna().any():
            raise ValueError("The chosen subject/group column has missing identifiers.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(x, y, groups=groups))
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split_method = "GroupShuffleSplit (subject/recording-aware)"
        overlap = set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))
        if overlap:
            raise ValueError("Group leakage detected across train/test sets.")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        split_method = "Stratified row-level split (NOT patient-independent)"

    classes = set(pd.unique(y))
    if set(pd.unique(y_train)) != classes or set(pd.unique(y_test)) != classes:
        raise ValueError("At least one target class missing from train or test; revise splitting.")
    if len(x_train) + len(x_test) != len(df):
        raise ValueError("Not all original records were retained in train/test partitions.")

    heading("B. TRAIN/TEST DETAILS ")
    print("Split method:", split_method)
    print("Train records:", len(x_train))
    print("Test records:", len(x_test))
    print("Train + test = original:", len(x_train) + len(x_test) == len(df))
    print("Test fraction requested:", TEST_SIZE)
    print("Random state:", RANDOM_STATE)
    print("Training target counts:\n" + y_train.value_counts().to_string())
    print("Test target counts:\n" + y_test.value_counts().to_string())
    print("Preprocessing: StandardScaler fitted on training data for LR, KNN and SVM.")
    print("No scaling for Decision Tree or Gaussian Naive Bayes.")
    if SUBJECT_COLUMN is None:
        print("LIMITATION: Subject-independent performance cannot be claimed without subject IDs.")

    # 4. SAVE THE HELD-OUT TEST DATA FOR STREAMLIT
    test_data = x_test.copy()
    test_data[TARGET] = y_test.to_numpy()
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)
    print("Saved: test_data.csv (features plus target)")

    # 5. DEFINE FIVE ASSIGNMENT MODELS
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE)
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": make_pipeline(GaussianNB()),
        "SVM": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE)
        ),
    }
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "SVM": "svm.joblib",
    }

    # 6. TRAIN, EVALUATE, SAVE
    results = []
    for name, model in models.items():
        heading(f"C. MODEL DETAILS — {name}")
        started = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        metrics, prediction = calculate_metrics(model, x_test, y_test)
        print("Model:", name)
        print("Training time (seconds):", round(elapsed, 3))
        print("Parameters:", model.get_params())
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}" if np.isfinite(score) else f"{metric_name}: N/A")
        print("Confusion matrix (rows actual, columns predicted; labels:", model.classes_, "):")
        print(confusion_matrix(y_test, prediction, labels=model.classes_))
        print("Classification report:\n" + classification_report(
            y_test, prediction, labels=model.classes_, zero_division=0
        ))
        file_path = MODEL_DIR / model_files[name]
        joblib.dump(model, file_path)
        print("Saved model:", file_path.relative_to(BASE_DIR))
        results.append({"Model": name, **metrics})

    # 7. PRODUCE ASSIGNMENT COMPARISON TABLE + README-READY MARKDOWN
    comparison = pd.DataFrame(results)
    comparison.to_csv(BASE_DIR / "results.csv", index=False)
    heading("D. MODEL COMPARISON — SIX REQUIRED METRICS")
    print(comparison.round(4).to_string(index=False))
    heading("E. README MARKDOWN TABLE ")
    columns = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, result in comparison.iterrows():
        cells = [result["Model"]] + [
            f"{result[key]:.4f}" if pd.notna(result[key]) else "N/A"
            for key in columns[1:]
        ]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
