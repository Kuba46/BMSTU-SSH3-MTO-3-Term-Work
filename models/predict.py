"""
models/predict.py
=================
Авторазметка тональности всего корпуса лучшей из обученных моделей.

Логика выбора модели:
    - Загружает METRICS_JSON
    - Сравнивает f1_macro LogReg и SVM
    - Использует модель с более высоким f1_macro
    - Если метрики одинаковы — предпочитает LogReg (более интерпретируема)

Результат:
    - Добавляет колонки sentiment_pred, sentiment_proba_pos, sentiment_proba_neu, sentiment_proba_neg в корпус
    - Сохраняет results/predictions.csv

Запуск:
    python -m models.predict              # авторазметка всего корпуса
    python -m models.predict --model svm  # принудительно использовать SVM
    python -m models.predict --model logreg
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from config.settings import (
    PROCESSED_CSV,
    PREDICTIONS_CSV,
    METRICS_JSON,
    LOGREG_MODEL,
    SVM_MODEL,
    SENTIMENT_LABEL_NAMES,
    MODELS_DIR,
)

log = logging.getLogger(__name__)


# ── Выбор модели ──────────────────────────────────────────────────────────────

def _select_best_model() -> str:
    """
    Читает METRICS_JSON и возвращает имя лучшей модели ('logreg' или 'svm').
    При отсутствии файла — возвращает 'logreg' по умолчанию.
    """
    if not METRICS_JSON.exists():
        log.warning(
            "Файл метрик не найден (%s). Используется LogReg по умолчанию.",
            METRICS_JSON,
        )
        return "logreg"

    with open(METRICS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    lr_f1  = data.get("logreg", {}).get("f1_macro", 0.0)
    svm_f1 = data.get("svm",    {}).get("f1_macro", 0.0)

    if svm_f1 > lr_f1:
        log.info(
            "Выбрана модель: SVM (F1=%.4f) vs LogReg (F1=%.4f)", svm_f1, lr_f1
        )
        return "svm"
    else:
        log.info(
            "Выбрана модель: LogReg (F1=%.4f) vs SVM (F1=%.4f)", lr_f1, svm_f1
        )
        return "logreg"


def _load_model(model_name: str):
    """
    Загружает векторизатор и модель по имени.

    Returns:
        (vectorizer, model)
    """
    if model_name == "svm":
        from models.svm_clf import load_model
        return load_model()
    else:
        from models.sentiment import load_model
        return load_model()


# ── Предсказание ──────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    """
    Размечает тональность всего DataFrame.

    Args:
        df:         DataFrame с колонкой text_lemma
        model_name: 'logreg' | 'svm' | None (автовыбор)

    Returns:
        df с добавленными колонками:
          sentiment_pred      — предсказанная метка (-1, 0, 1)
          sentiment_label     — строковое название класса
          proba_positive      — P(позитивная)
          proba_neutral       — P(нейтральная)
          proba_negative      — P(негативная)
          sentiment_confidence— уверенность модели (max proba)
    """
    if model_name is None:
        model_name = _select_best_model()

    vectorizer, model = _load_model(model_name)

    df = df.copy()
    df["text_lemma"] = df["text_lemma"].fillna("").astype(str)

    # Убираем строки с пустым текстом (они не могут быть размечены)
    empty_mask = df["text_lemma"].str.strip().eq("")
    n_empty = empty_mask.sum()
    if n_empty:
        log.warning(
            "%d записей с пустым text_lemma пропущены при предсказании", n_empty
        )

    df_valid = df[~empty_mask].copy()
    df_empty = df[empty_mask].copy()
    df_empty["sentiment_pred"] = np.nan
    df_empty["sentiment_label"] = "unknown"
    df_empty["proba_positive"] = np.nan
    df_empty["proba_neutral"] = np.nan
    df_empty["proba_negative"] = np.nan
    df_empty["sentiment_confidence"] = np.nan

    if df_valid.empty:
        log.error("Нет записей для предсказания!")
        return df_empty

    log.info("Векторизация %d документов...", len(df_valid))
    X = vectorizer.transform(df_valid["text_lemma"])

    log.info("Предсказание тональности (модель: %s)...", model_name)
    y_pred = model.predict(X)

    # Вероятности классов
    # model.classes_ содержит метки в том же порядке, что и columns proba
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    def _proba_for_class(cls_val: int) -> np.ndarray:
        if cls_val in classes:
            return proba[:, classes.index(cls_val)]
        return np.zeros(len(proba))

    df_valid["sentiment_pred"]       = y_pred
    df_valid["sentiment_label"]      = [SENTIMENT_LABEL_NAMES[p] for p in y_pred]
    df_valid["proba_positive"]       = _proba_for_class(1)
    df_valid["proba_neutral"]        = _proba_for_class(0)
    df_valid["proba_negative"]       = _proba_for_class(-1)
    df_valid["sentiment_confidence"] = proba.max(axis=1)
    df_valid["model_used"]           = model_name

    # Объединяем обратно
    result = pd.concat([df_valid, df_empty], ignore_index=True)
    result = result.sort_index()

    # Статистика предсказаний
    counts = df_valid["sentiment_label"].value_counts()
    total  = len(df_valid)
    log.info(
        "Авторазметка завершена (%d документов):\n"
        "   позитивных: %d (%.1f%%)\n"
        "   нейтральных: %d (%.1f%%)\n"
        "   негативных: %d (%.1f%%)\n"
        "   ср. уверенность: %.3f",
        total,
        counts.get("positive", 0), counts.get("positive", 0) / total * 100,
        counts.get("neutral",  0), counts.get("neutral",  0) / total * 100,
        counts.get("negative", 0), counts.get("negative", 0) / total * 100,
        df_valid["sentiment_confidence"].mean(),
    )

    return result


# ── Агрегированная тональность ────────────────────────────────────────────────

def sentiment_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегированная тональность по каналам.

    Returns:
        DataFrame: channel_label, orientation, n_posts,
                   pct_positive, pct_neutral, pct_negative,
                   mean_confidence
    """
    df = df.dropna(subset=["sentiment_pred"])
    rows = []
    for (label, orient), grp in df.groupby(["channel_label", "orientation"]):
        n = len(grp)
        rows.append({
            "channel_label":  label,
            "orientation":    orient,
            "n_posts":        n,
            "pct_positive":   (grp["sentiment_pred"] == 1).sum()  / n * 100,
            "pct_neutral":    (grp["sentiment_pred"] == 0).sum()  / n * 100,
            "pct_negative":   (grp["sentiment_pred"] == -1).sum() / n * 100,
            "mean_confidence": grp["sentiment_confidence"].mean(),
        })
    return pd.DataFrame(rows).sort_values("n_posts", ascending=False)


def sentiment_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Динамика тональности по месяцам.

    Returns:
        DataFrame: period (YYYY-MM), n_posts,
                   pct_positive, pct_neutral, pct_negative
    """
    df = df.dropna(subset=["sentiment_pred"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["period"] = df["date"].dt.to_period("M").astype(str)

    rows = []
    for period, grp in df.groupby("period"):
        n = len(grp)
        rows.append({
            "period":       period,
            "n_posts":      n,
            "pct_positive": (grp["sentiment_pred"] == 1).sum()  / n * 100,
            "pct_neutral":  (grp["sentiment_pred"] == 0).sum()  / n * 100,
            "pct_negative": (grp["sentiment_pred"] == -1).sum() / n * 100,
        })
    return pd.DataFrame(rows).sort_values("period")


# ── Основной пайплайн ─────────────────────────────────────────────────────────

def run_pipeline(model_name: str | None = None) -> pd.DataFrame:
    """
    Загружает обработанный корпус → размечает → сохраняет PREDICTIONS_CSV.
    """
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"Обработанный корпус не найден: {PROCESSED_CSV}\n"
            "Запустите NLP-пайплайн перед предсказанием."
        )

    df = pd.read_csv(PROCESSED_CSV, encoding="utf-8")
    df["text_lemma"] = df["text_lemma"].fillna("").astype(str)
    log.info("Загружено документов для разметки: %d", len(df))

    df_pred = predict(df, model_name=model_name)
    df_pred.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")
    log.info("Предсказания сохранены → %s  (%d строк)", PREDICTIONS_CSV, len(df_pred))

    # Краткие агрегаты в лог
    log.info("\nТональность по каналам:\n%s",
             sentiment_by_channel(df_pred).to_string(index=False))
    log.info("\nТональность по месяцам:\n%s",
             sentiment_by_month(df_pred).to_string(index=False))

    return df_pred


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Авторазметка тональности всего корпуса."
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "svm"],
        default=None,
        help="Принудительно выбрать модель (по умолчанию — лучшая по F1).",
    )
    args = parser.parse_args()
    run_pipeline(model_name=args.model)


if __name__ == "__main__":
    main()