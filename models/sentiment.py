"""
models/sentiment.py
===================
Классификатор тональности на основе логистической регрессии.

Этапы:
    1. Загружает размеченную выборку (labeled/posts_labeled.csv)
    2. Строит TF-IDF-матрицу только на размеченных постах
    3. Делит на train/test (стратифицированно по классам)
    4. Обучает LogisticRegression
    5. Оценивает на тестовой выборке (Precision, Recall, F1 macro)
    6. Сохраняет модель и векторизатор через joblib

Запуск:
    python -m models.sentiment           # обучить и оценить
    python -m models.sentiment --report  # + полный classification_report
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import inspect
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import (
    PROCESSED_CSV,
    LABELED_CSV,
    COMMENTS_PROCESSED_CSV,
    COMMENTS_LABELED_CSV,
    LOGREG_MODEL,
    COMMENTS_LOGREG_MODEL,
    TFIDF_PARAMS,
    LOGREG_PARAMS,
    TEST_SIZE,
    RANDOM_STATE,
    SENTIMENT_LABEL_NAMES,
    MODELS_DIR,
    METRICS_JSON,
    COMMENTS_METRICS_JSON,
)

log = logging.getLogger(__name__)

# Путь для сохранения векторизатора, обученного на размеченной выборке
_VECTORIZER_PATH = MODELS_DIR / "logreg_vectorizer.pkl"


def _resolve_artifacts(prefix: str = "") -> tuple[Path, Path, Path, Path, Path]:
    if prefix == "comments_":
        return (
            COMMENTS_PROCESSED_CSV,
            COMMENTS_LABELED_CSV,
            COMMENTS_LOGREG_MODEL,
            MODELS_DIR / "comments_logreg_vectorizer.pkl",
            COMMENTS_METRICS_JSON,
        )
    return (
        PROCESSED_CSV,
        LABELED_CSV,
        LOGREG_MODEL,
        _VECTORIZER_PATH,
        METRICS_JSON,
    )


def _build_logreg(params: dict) -> LogisticRegression:
    """Попытаться создать LogisticRegression с `params`. При ошибке TypeError
    фильтруем неподдерживаемые ключи по сигнатуре конструктора и пробуем снова.
    Это делает код совместимее с разными версиями scikit-learn.
    """
    try:
        return LogisticRegression(**params)
    except TypeError as e:
        log.warning("LogisticRegression init failed: %s. Trying fallback.", e)
        try:
            sig = inspect.signature(LogisticRegression)
            allowed = set(sig.parameters.keys()) - {"self"}
            filtered = {k: v for k, v in params.items() if k in allowed}
            if not filtered:
                raise
            log.info("Using filtered LogisticRegression params: %s", ", ".join(filtered.keys()))
            return LogisticRegression(**filtered)
        except Exception:
            log.exception("Failed to construct LogisticRegression after filtering parameters.")
            raise


# ── Загрузка данных ───────────────────────────────────────────────────────────

def load_labeled_corpus(
    processed_path=PROCESSED_CSV,
    labeled_path=LABELED_CSV,
) -> pd.DataFrame:
    """
    Объединяет обработанный корпус (text_lemma) с разметкой тональности.

    Returns:
        DataFrame с колонками: text_lemma, sentiment, channel_username,
        channel_label, orientation, date
    """
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Обработанный корпус не найден: {processed_path}\n"
            "Запустите NLP-пайплайн: python -m nlp.preprocessor && "
            "python -m nlp.lemmatizer"
        )
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Разметка не найдена: {labeled_path}\n"
            "Запустите разметчик для соответствующего корпуса."
        )

    df_corpus = pd.read_csv(processed_path, encoding="utf-8")
    df_labels = pd.read_csv(labeled_path, encoding="utf-8")

    id_candidates = [
        col for col in ["post_id", "comment_id"]
        if col in df_corpus.columns and col in df_labels.columns
    ]
    if not id_candidates:
        raise ValueError(
            "Не найдена общая ID-колонка для объединения. "
            "Ожидались 'post_id' или 'comment_id'."
        )
    id_col = id_candidates[0]

    df_labels = df_labels[["channel_username", id_col, "sentiment"]].copy()

    df_labels["sentiment"] = pd.to_numeric(df_labels["sentiment"], errors="coerce")
    df_labels = df_labels.dropna(subset=["sentiment"])
    df_labels["sentiment"] = df_labels["sentiment"].astype(int)

    df = df_corpus.merge(df_labels, on=["channel_username", id_col], how="inner")

    if df.empty:
        corpus_channels = set(df_corpus["channel_username"].dropna().astype(str).unique())
        labels_channels = set(df_labels["channel_username"].dropna().astype(str).unique())
        overlap_channels = sorted(corpus_channels & labels_channels)

        raise ValueError(
            "После объединения PROCESSED_CSV и LABELED_CSV не найдено ни одной размеченной записи.\n"
            "Проверьте, что вы размечаете посты из processed-корпуса, а не из другого набора.\n"
            f"Каналы в processed: {len(corpus_channels)}, в labeled: {len(labels_channels)}, "
            f"пересечение: {len(overlap_channels)}."
        )

    df["text_lemma"] = df["text_lemma"].fillna("").astype(str)

    # Убираем записи без лемматизированного текста
    df = df[df["text_lemma"].str.strip().ne("")].reset_index(drop=True)

    log.info(
        "Загружено размеченных постов: %d  "
        "(позитивных: %d, нейтральных: %d, негативных: %d)",
        len(df),
        (df["sentiment"] == 1).sum(),
        (df["sentiment"] == 0).sum(),
        (df["sentiment"] == -1).sum(),
    )
    return df


# ── Обучение ──────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    tfidf_params: dict | None = None,
    logreg_params: dict | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_folds: int = 5,
) -> tuple[TfidfVectorizer, LogisticRegression, dict]:
    """
    Полный цикл обучения: векторизация → split → кросс-валидация → обучение.

    Args:
        df:            DataFrame с колонками text_lemma, sentiment
        tfidf_params:  параметры TF-IDF (по умолчанию из settings)
        logreg_params: параметры LogReg (по умолчанию из settings)
        test_size:     доля тестовой выборки
        random_state:  seed для воспроизводимости
        cv_folds:      число фолдов кросс-валидации

    Returns:
        (vectorizer, model, metrics_dict)
    """
    if tfidf_params is None:
        tfidf_params = TFIDF_PARAMS
    if logreg_params is None:
        logreg_params = LOGREG_PARAMS

    if df.empty:
        raise ValueError(
            "Пустая обучающая выборка. Сначала создайте/обновите разметку для записей из processed-корпуса."
        )

    X_text = df["text_lemma"].values
    y      = df["sentiment"].values

    # ── Векторизация ──────────────────────────────────────────────────────────
    log.info("Векторизация размеченной выборки (TF-IDF)...")
    vectorizer = TfidfVectorizer(**tfidf_params)
    X = vectorizer.fit_transform(X_text)
    log.info("Матрица TF-IDF: %s", X.shape)

    # ── Разбивка train/test (стратифицированная) ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    log.info(
        "Train: %d  Test: %d  (соотношение %.0f/%.0f)",
        X_train.shape[0], X_test.shape[0],
        (1 - test_size) * 100, test_size * 100,
    )

    # ── Кросс-валидация (на train) ────────────────────────────────────────────
    log.info("Кросс-валидация (%d-fold, метрика: f1_macro)...", cv_folds)
    model_cv = _build_logreg(logreg_params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model_cv, X_train, y_train,
                                cv=cv, scoring="f1_macro", n_jobs=-1)
    log.info(
        "CV F1-macro: %.4f ± %.4f  (по фолдам: %s)",
        cv_scores.mean(), cv_scores.std(),
        " ".join(f"{s:.3f}" for s in cv_scores),
    )

    # ── Итоговое обучение на всём train ──────────────────────────────────────
    log.info("Обучение LogisticRegression на train...")
    model = _build_logreg(logreg_params)
    model.fit(X_train, y_train)

    # ── Оценка на test ────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    labels     = sorted(set(y))
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
        "Результаты LogReg на тестовой выборке:\n"
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
    model: LogisticRegression,
    model_path=LOGREG_MODEL,
    vectorizer_path=_VECTORIZER_PATH,
) -> None:
    """Сохраняет модель и векторизатор через joblib."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    log.info("Модель сохранена → %s", model_path)
    log.info("Векторизатор сохранён → %s", vectorizer_path)


def load_model(
    model_path=LOGREG_MODEL,
    vectorizer_path=_VECTORIZER_PATH,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Загружает сохранённые модель и векторизатор."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}\n"
            "Сначала запустите: python -m models.sentiment"
        )
    model      = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    log.info("Модель загружена: %s", model_path)
    return vectorizer, model


# ── Интерпретация модели ──────────────────────────────────────────────────────

def top_features(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    n: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Топ-N слов с наибольшим весом для каждого класса тональности.
    Работает только для линейной модели (коэффициенты интерпретируемы).

    Returns:
        Словарь {'positive': df, 'neutral': df, 'negative': df}
    """
    terms  = vectorizer.get_feature_names_out()
    result = {}

    for i, cls in enumerate(model.classes_):
        coefs = model.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:n]
        result[SENTIMENT_LABEL_NAMES[cls]] = pd.DataFrame({
            "term":        terms[top_idx],
            "coefficient": coefs[top_idx],
        })

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Обучение классификатора тональности (LogisticRegression)."
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Вывести полный classification_report.",
    )
    parser.add_argument(
        "--features", type=int, default=0, metavar="N",
        help="Вывести топ-N признаков для каждого класса.",
    )
    parser.add_argument(
        "--processed", type=str, default=None,
        help="Путь к processed CSV (если нужно указать нестандартный файл).",
    )
    parser.add_argument(
        "--labeled", type=str, default=None,
        help="Путь к labeled CSV (если нужно указать нестандартный файл).",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Префикс набора артефактов (например, comments_).",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Явный путь для сохранения модели.",
    )
    parser.add_argument(
        "--vectorizer-path", type=str, default=None,
        help="Явный путь для сохранения векторизатора.",
    )
    parser.add_argument(
        "--metrics-path", type=str, default=None,
        help="Явный путь для сохранения метрик JSON.",
    )
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
        print("\nClassification Report (LogisticRegression):")
        print(metrics["classification_report"])
        print("Confusion Matrix:")
        print(np.array(metrics["confusion_matrix"]))

    if args.features > 0:
        feats = top_features(vectorizer, model, n=args.features)
        for cls_name, df_feat in feats.items():
            print(f"\nТоп-{args.features} признаков [{cls_name}]:")
            print(df_feat.to_string(index=False))

    # Сохраняем метрики в JSON для последующего сравнения с SVM
    import json
    existing = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing["logreg"] = {
        k: v for k, v in metrics.items()
        if k != "classification_report"   # отчёт — только в лог
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    log.info("Метрики сохранены → %s", metrics_path)


if __name__ == "__main__":
    main()
    