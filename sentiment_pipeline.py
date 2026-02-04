# ============================================================
# STEP 0: Imports & Environment
# ============================================================

import numpy as np
import pandas as pd
import re
import os
import time
import joblib
import warnings

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

# ============================================================
# STEP 1: Load & Prepare Data
# ============================================================

def standardize_columns(df):
    if "Review text" in df.columns:
        df["review_text"] = df["Review text"]
    if "Ratings" in df.columns:
        df["reviewer_rating"] = df["Ratings"]
    return df

df = pd.concat([
    standardize_columns(pd.read_csv("data/review_badmention.csv")),
    standardize_columns(pd.read_csv("data/review_tawa.csv")),
    standardize_columns(pd.read_csv("data/review_tea.csv"))
])

df = df[["review_text", "reviewer_rating"]].dropna().drop_duplicates()
df["sentiment"] = (df["reviewer_rating"].astype(int) >= 4).astype(int)

X = df["review_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ============================================================
# STEP 2: Custom Text Cleaner
# ============================================================

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [
            re.sub(r"\s+", " ",
                   re.sub(r"[^a-zA-Z ]", " ", str(x).lower())
            ).strip() for x in X
        ]

# ============================================================
# STEP 3: Pipeline Skeleton
# ============================================================

pipeline = Pipeline([
    ("cleaner", TextCleaner()),
    ("vectorizer", TfidfVectorizer(stop_words="english")),
    ("model", LogisticRegression())
])

f1_safe = make_scorer(f1_score, average="weighted", zero_division=0)

# ============================================================
# STEP 4: OPTUNA OBJECTIVES (ONE PER MODEL)
# ============================================================

def objective_lr(trial):
    pipeline.set_params(
        vectorizer__max_features=trial.suggest_int("max_features", 2000, 8000, step=2000),
        vectorizer__ngram_range=trial.suggest_categorical("ngram_range", [(1,1),(1,2)]),
        model=LogisticRegression(
            C=trial.suggest_float("C", 1e-2, 10, log=True),
            max_iter=1000,
            class_weight="balanced"
        )
    )
    return cross_val_score(
        pipeline, X_train, y_train,
        scoring=f1_safe,
        cv=StratifiedKFold(3, shuffle=True)
    ).mean()

def objective_nb(trial):
    pipeline.set_params(
        vectorizer__max_features=trial.suggest_int("max_features", 2000, 8000, step=2000),
        vectorizer__ngram_range=trial.suggest_categorical("ngram_range", [(1,1),(1,2)]),
        model=MultinomialNB(
            alpha=trial.suggest_float("alpha", 0.01, 1.0)
        )
    )
    return cross_val_score(
        pipeline, X_train, y_train,
        scoring=f1_safe,
        cv=StratifiedKFold(3, shuffle=True)
    ).mean()

def objective_svm(trial):
    pipeline.set_params(
        vectorizer__max_features=trial.suggest_int("max_features", 2000, 8000, step=2000),
        vectorizer__ngram_range=trial.suggest_categorical("ngram_range", [(1,1),(1,2)]),
        model=LinearSVC(
            C=trial.suggest_float("C", 0.01, 10, log=True),
            class_weight="balanced"
        )
    )
    return cross_val_score(
        pipeline, X_train, y_train,
        scoring=f1_safe,
        cv=StratifiedKFold(3, shuffle=True)
    ).mean()

def objective_rf(trial):
    pipeline.set_params(
        vectorizer__max_features=trial.suggest_int("max_features", 2000, 8000, step=2000),
        model=RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            class_weight="balanced",
            random_state=42
        )
    )
    return cross_val_score(
        pipeline, X_train, y_train,
        scoring=f1_safe,
        cv=StratifiedKFold(3, shuffle=True)
    ).mean()

# ============================================================
# STEP 5: MODEL LOOP (EXACTLY LIKE IRIS)
# ============================================================

objectives = {
    "LogisticRegression": objective_lr,
    "NaiveBayes": objective_nb,
    "SVM": objective_svm,
    "RandomForest": objective_rf
}

mlflow.set_experiment("FLIPKART_SENTIMENT_ANALYSIS")

results = {}

for model_name, obj_fn in objectives.items():

    print(f"\n--- Optimizing {model_name} ---")

    mlflow_cb = MLflowCallback(
        metric_name="cv_f1_weighted",
        mlflow_kwargs={"nested": True}
    )

    study = optuna.create_study(direction="maximize")
    start_fit = time.time()
    study.optimize(obj_fn, n_trials=20, callbacks=[mlflow_cb])
    fit_time = time.time() - start_fit

    best_params = study.best_params

    print(f"Best CV F1: {study.best_value:.4f}")

    pipeline.fit(X_train, y_train)

    train_f1 = f1_score(y_train, pipeline.predict(X_train),
                        average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, pipeline.predict(X_test),
                       average="weighted", zero_division=0)

    model_path = f"{model_name}_model.pkl"
    joblib.dump(pipeline, model_path)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(best_params)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("fit_time", fit_time)
        mlflow.log_metric("model_size", os.path.getsize(model_path))
        mlflow.sklearn.log_model(pipeline, name=f"{model_name}_sentiment_model")

    os.remove(model_path)

    results[model_name] = (train_f1, test_f1)

# ============================================================
# STEP 6: SUMMARY
# ============================================================

print("\n--- FINAL SUMMARY ---")
for k, v in results.items():
    print(f"{k}: Train F1={v[0]:.4f}, Test F1={v[1]:.4f}")
