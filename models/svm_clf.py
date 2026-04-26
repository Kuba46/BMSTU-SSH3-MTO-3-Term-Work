"""
models/svm_clf.py
=================
Классификатор тональности на основе метода опорных векторов (SVM).

Интерфейс намеренно идентичен sentiment.py — это позволяет
сравнивать модели «один к одному» в evaluation/metrics.py.

Отличия от sentiment.py:
    - Использует LinearSVC (быстрее SVC для больших словарей)
    - Для получения вероятностей оборачивается в CalibratedClassifierCV (нужно для мягкого голосования при ансамблировании)
    - Сохраняет модель в отдельный файл SVM_MODEL

Запуск:
    python -m models.svm_clf           # обучить и оценить
    python -m models.svm_clf --report  # + полный classification_report
"""

import argparse
import json
import logging

import joblib
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
    LOGREG_MODEL,          # путь к LogReg — для сравнительной таблицы
    METRICS_JSON,
    MODELS_DIR,
    RANDOM_STATE,
    SENTIMENT_LABEL_NAMES,
    SVM_PARAMS,
    TEST_SIZE,
    TFIDF_PARAMS,
    SVM_MODEL,
)


log = logging.getLogger(__name__)

_VECTORIZER_PATH = MODELS_DIR / "svm_vectorizer.pkl"


# ── Загрузка данных ───────────────────────────────────────────────────────────

def load_labeled_corpus() -> pd.DataFrame:
    """
    Идентична функции в sentiment.py — дублирование намеренное,
    чтобы модули были независимыми и запускались без импорта друг друга.
    """
    from models.sentiment import load_labeled_corpus as _load
    return _load()


# ── Обучение ──────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    tfidf_params: dict | None = None,
    svm_params: dict | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_folds: int = 5,
) -> tuple[TfidfVectorizer, CalibratedClassifierCV, dict]:
    """
    Полный цикл обучения SVM.

    LinearSVC не поддерживает predict_proba нативно — оборачиваем
    в CalibratedClassifierCV с method='sigmoid' (Platts scaling).

    Returns:
        (vectorizer, calibrated_model, metrics_dict)
    """
    if tfidf_params is None:
        tfidf_params = TFIDF_PARAMS
    if svm_params is None:
        svm_params = SVM_PARAMS

    X_text = df["text_lemma"].values
    y      = df["sentiment"].values

    # ── Векторизация ──────────────────────────────────────────────────────────
    log.info("Векторизация размеченной выборки (TF-IDF)...")
    vectorizer = TfidfVectorizer(**tfidf_params)
    X = vectorizer.fit_transform(X_text)
    log.info("Матрица TF-IDF: %s", X.shape)

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    log.info("Train: %d  Test: %d", X_train.shape[0], X_test.shape[0])

    # ── Базовый LinearSVC ─────────────────────────────────────────────────────
    # Извлекаем параметры, совместимые с LinearSVC
    # (убираем kernel — он не нужен для LinearSVC)
    lsvc_params = {k: v for k, v in svm_params.items() if k != "kernel"}
    base_svc = LinearSVC(**lsvc_params)

    # ── Калибровка вероятностей ───────────────────────────────────────────────
    # cv=3 внутри калибратора — не путать с внешней кросс-валидацией
    model = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)

    # ── Кросс-валидация ───────────────────────────────────────────────────────
    log.info("Кросс-валидация (%d-fold, метрика: f1_macro)...", cv_folds)
    cv_model = CalibratedClassifierCV(LinearSVC(**lsvc_params), method="sigmoid", cv=3)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(cv_model, X_train, y_train,
                                cv=cv, scoring="f1_macro", n_jobs=-1)
    log.info(
        "CV F1-macro: %.4f ± %.4f  (по фолдам: %s)",
        cv_scores.mean(), cv_scores.std(),
        " ".join(f"{s:.3f}" for s in cv_scores),
    )

    # ── Итоговое обучение ─────────────────────────────────────────────────────
    log.info("Обучение SVM (LinearSVC + калибровка) на train...")
    model.fit(X_train, y_train)

    # ── Оценка на test ────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    labels      = sorted(set(y))
    label_names = [SENTIMENT_LABEL_NAMES[l] for l in labels]

    metrics = {
        "precision_macro": float(precision_score(y_test, y_pred, average="macro",
                                                  zero_division=0)),
        "recall_macro":    float(recall_score(y_test, y_pred, average="macro",
                                              zero_division=0)),
        "f1_macro":        float(f1_score(y_test, y_pred, average="macro",
                                          zero_division=0)),
        "cv_f1_mean":      float(cv_scores.mean()),
        "cv_f1_std":       float(cv_scores.std()),
        "n_train":         int(X_train.shape[0]),
        "n_test":          int(X_test.shape[0]),
        "n_features":      int(X.shape[1]),
        "classification_report": classification_report(
            y_test, y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred,
                                              labels=labels).tolist(),
    }

    log.info(
        "Результаты SVM на тестовой выборке:\n"
        "  Precision (macro): %.4f\n"
        "  Recall    (macro): %.4f\n"
        "  F1        (macro): %.4f",
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
    )

    return vectorizer, model, metrics


# ── Сохранение / загрузка ─────────────────────────────────────────────────────

def save_model(
    vectorizer: TfidfVectorizer,
    model: CalibratedClassifierCV,
    model_path=SVM_MODEL,
    vectorizer_path=_VECTORIZER_PATH,
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
        raise FileNotFoundError(
            f"SVM модель не найдена: {model_path}\n"
            "Сначала запустите: python -m models.svm_clf"
        )
    model      = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    log.info("SVM модель загружена: %s", model_path)
    return vectorizer, model


# ── Сравнительная таблица LogReg vs SVM ──────────────────────────────────────

def comparison_table() -> pd.DataFrame:
    """
    Загружает METRICS_JSON и строит сравнительную таблицу
    LogisticRegression vs SVM по ключевым метрикам.

    Returns:
        DataFrame с колонками: metric, logreg, svm, winner
    """
    if not METRICS_JSON.exists():
        raise FileNotFoundError(
            f"Файл метрик не найден: {METRICS_JSON}\n"
            "Сначала обучите обе модели."
        )

    with open(METRICS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    metric_keys = ["precision_macro", "recall_macro", "f1_macro",
                   "cv_f1_mean", "cv_f1_std"]

    for metric in metric_keys:
        lr_val  = data.get("logreg", {}).get(metric, float("nan"))
        svm_val = data.get("svm",    {}).get(metric, float("nan"))

        # Для std меньше = лучше (стабильность); для остальных больше = лучше
        if metric.endswith("_std"):
            winner = "logreg" if lr_val <= svm_val else "svm"
        else:
            winner = "logreg" if lr_val >= svm_val else "svm"

        rows.append({
            "metric": metric,
            "logreg": round(lr_val, 4),
            "svm":    round(svm_val, 4),
            "winner": winner,
        })

    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Обучение классификатора тональности (SVM / LinearSVC)."
    )
    parser.add_argument("--report", action="store_true",
                        help="Вывести полный classification_report.")
    parser.add_argument("--compare", action="store_true",
                        help="Вывести сравнительную таблицу LogReg vs SVM.")
    args = parser.parse_args()

    df = load_labeled_corpus()
    vectorizer, model, metrics = train(df)
    save_model(vectorizer, model)

    if args.report:
        print("\nClassification Report (SVM / LinearSVC):")
        print(metrics["classification_report"])
        print("Confusion Matrix:")
        import numpy as np
        print(np.array(metrics["confusion_matrix"]))

    # Сохраняем метрики
    existing = {}
    if METRICS_JSON.exists():
        with open(METRICS_JSON, encoding="utf-8") as f:
            existing = json.load(f)
    existing["svm"] = {
        k: v for k, v in metrics.items()
        if k != "classification_report"
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    log.info("Метрики SVM сохранены → %s", METRICS_JSON)

    if args.compare:
        try:
            tbl = comparison_table()
            print("\nСравнение LogisticRegression vs SVM:")
            print(tbl.to_string(index=False))
        except FileNotFoundError as e:
            log.warning(str(e))


if __name__ == "__main__":
    main()