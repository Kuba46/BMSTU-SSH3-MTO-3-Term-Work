"""
models/svm_clf.py
=================
Классификатор тональности на основе метода опорных векторов (SVM).

Запуск:
    python -m models.svm_clf
    python -m models.svm_clf --report
    python -m models.svm_clf --prefix comments_
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import LinearSVC

from config.settings import (
    COMMENTS_LABELED_CSV,
    COMMENTS_METRICS_JSON,
    COMMENTS_PROCESSED_CSV,
    COMMENTS_SVM_MODEL,
    LABELED_CSV,
    METRICS_JSON,
    MODELS_DIR,
    PROCESSED_CSV,
    RANDOM_STATE,
    SENTIMENT_LABEL_NAMES,
    SVM_MODEL,
    SVM_PARAMS,
    TEST_SIZE,
    TFIDF_PARAMS,
)

log = logging.getLogger(__name__)

_VECTORIZER_PATH = MODELS_DIR / "svm_vectorizer.pkl"


def _resolve_artifacts(prefix: str = "") -> tuple[Path, Path, Path, Path, Path]:
    if prefix == "comments_":
        return (
            COMMENTS_PROCESSED_CSV,
            COMMENTS_LABELED_CSV,
            COMMENTS_SVM_MODEL,
            MODELS_DIR / "comments_svm_vectorizer.pkl",
            COMMENTS_METRICS_JSON,
        )
    return (
        PROCESSED_CSV,
        LABELED_CSV,
        SVM_MODEL,
        _VECTORIZER_PATH,
        METRICS_JSON,
    )


def _merge_processed_and_labeled(processed_path: Path, labeled_path: Path) -> pd.DataFrame:
    df_corpus = pd.read_csv(processed_path, encoding="utf-8")
    df_labels = pd.read_csv(labeled_path, encoding="utf-8")

    id_candidates = [
        col for col in ["post_id", "comment_id"]
        if col in df_corpus.columns and col in df_labels.columns
    ]
    if not id_candidates:
        raise ValueError("Не найдена общая ID-колонка (post_id/comment_id).")
    id_col = id_candidates[0]

    df_labels = df_labels[["channel_username", id_col, "sentiment"]].copy()
    df_labels["sentiment"] = pd.to_numeric(df_labels["sentiment"], errors="coerce")
    df_labels = df_labels.dropna(subset=["sentiment"])
    df_labels["sentiment"] = df_labels["sentiment"].astype(int)

    df = df_corpus.merge(df_labels, on=["channel_username", id_col], how="inner")
    df["text_lemma"] = df["text_lemma"].fillna("").astype(str)
    df = df[df["text_lemma"].str.strip().ne("")].reset_index(drop=True)
    return df


def load_labeled_corpus(processed_path: Path, labeled_path: Path) -> pd.DataFrame:
    if not processed_path.exists():
        raise FileNotFoundError(f"Файл не найден: {processed_path}")
    if not labeled_path.exists():
        raise FileNotFoundError(f"Файл разметки не найден: {labeled_path}")
    return _merge_processed_and_labeled(processed_path, labeled_path)


def train(
    df: pd.DataFrame,
    tfidf_params: dict | None = None,
    svm_params: dict | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_folds: int = 5,
) -> tuple[TfidfVectorizer, CalibratedClassifierCV, dict]:
    if tfidf_params is None:
        tfidf_params = TFIDF_PARAMS
    if svm_params is None:
        svm_params = SVM_PARAMS

    X_text = df["text_lemma"].values
    y = df["sentiment"].values

    vectorizer = TfidfVectorizer(**tfidf_params)
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    lsvc_params = {k: v for k, v in svm_params.items() if k != "kernel"}
    base_svc = LinearSVC(**lsvc_params)
    model = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)

    cv_model = CalibratedClassifierCV(LinearSVC(**lsvc_params), method="sigmoid", cv=3)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels = sorted(set(y))
    label_names = [SENTIMENT_LABEL_NAMES[l] for l in labels]

    metrics = {
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }

    return vectorizer, model, metrics


def save_model(
    vectorizer: TfidfVectorizer,
    model: CalibratedClassifierCV,
    model_path: Path,
    vectorizer_path: Path,
) -> None:
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    log.info("SVM модель сохранена → %s", model_path)
    log.info("SVM векторизатор сохранён → %s", vectorizer_path)


def load_model(
    model_path=SVM_MODEL,
    vectorizer_path=_VECTORIZER_PATH,
) -> tuple[TfidfVectorizer, CalibratedClassifierCV]:
    if not model_path.exists():
        raise FileNotFoundError(f"SVM модель не найдена: {model_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer, model


def comparison_table(metrics_path: Path = METRICS_JSON) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Файл метрик не найден: {metrics_path}")

    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    metric_keys = ["precision_macro", "recall_macro", "f1_macro", "cv_f1_mean", "cv_f1_std"]
    for metric in metric_keys:
        lr_val = data.get("logreg", {}).get(metric, float("nan"))
        svm_val = data.get("svm", {}).get(metric, float("nan"))
        winner = "logreg" if (lr_val <= svm_val if metric.endswith("_std") else lr_val >= svm_val) else "svm"
        rows.append({"metric": metric, "logreg": round(lr_val, 4), "svm": round(svm_val, 4), "winner": winner})

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Обучение классификатора тональности (SVM / LinearSVC).")
    parser.add_argument("--report", action="store_true", help="Вывести classification_report.")
    parser.add_argument("--compare", action="store_true", help="Вывести сравнение LogReg vs SVM.")
    parser.add_argument("--processed", type=str, default=None, help="Путь к processed CSV.")
    parser.add_argument("--labeled", type=str, default=None, help="Путь к labeled CSV.")
    parser.add_argument("--prefix", type=str, default="", help="Префикс набора артефактов (например comments_).")
    parser.add_argument("--model-path", type=str, default=None, help="Явный путь для модели.")
    parser.add_argument("--vectorizer-path", type=str, default=None, help="Явный путь для векторизатора.")
    parser.add_argument("--metrics-path", type=str, default=None, help="Явный путь для metrics JSON.")
    args = parser.parse_args()

    default_processed, default_labeled, default_model, default_vectorizer, default_metrics = _resolve_artifacts(args.prefix)
    processed_path = default_processed if args.processed is None else Path(args.processed)
    labeled_path = default_labeled if args.labeled is None else Path(args.labeled)
    model_path = default_model if args.model_path is None else Path(args.model_path)
    vectorizer_path = default_vectorizer if args.vectorizer_path is None else Path(args.vectorizer_path)
    metrics_path = default_metrics if args.metrics_path is None else Path(args.metrics_path)

    df = load_labeled_corpus(processed_path=processed_path, labeled_path=labeled_path)
    vectorizer, model, metrics = train(df)
    save_model(vectorizer, model, model_path=model_path, vectorizer_path=vectorizer_path)

    if args.report:
        print("\nClassification Report (SVM / LinearSVC):")
        print(metrics["classification_report"])
        print("Confusion Matrix:")
        print(np.array(metrics["confusion_matrix"]))

    existing = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing["svm"] = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    log.info("Метрики SVM сохранены → %s", metrics_path)

    if args.compare:
        try:
            print("\nСравнение LogisticRegression vs SVM:")
            print(comparison_table(metrics_path=metrics_path).to_string(index=False))
        except FileNotFoundError as exc:
            log.warning(str(exc))


if __name__ == "__main__":
    main()