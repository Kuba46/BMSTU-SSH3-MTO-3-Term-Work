"""
evaluation/metrics.py
=====================
Оценка качества классификаторов и сравнительный анализ моделей.

Что делает:
    evaluate_model()      — Precision, Recall, F1 (macro/weighted/per-class)
                          + матрица ошибок для одной модели
    compare_models()      — сводная таблица LogReg vs SVM по всем метрикам
    error_analysis()      — анализ ошибок: какие посты классифицированы неверно
    confidence_analysis() — распределение уверенности модели по классам
    save_report()         — сохраняет полный отчёт в RESULTS_DIR/metrics_report.txt

Запуск:
    python -m evaluation.metrics           # полный отчёт по обеим моделям
    python -m evaluation.metrics --model logreg   # только LogReg
    python -m evaluation.metrics --errors  # + анализ ошибок
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import (
    LABELED_CSV,
    METRICS_JSON,
    PROCESSED_CSV,
    RESULTS_DIR,
    SENTIMENT_LABEL_NAMES,
    TEST_SIZE,
    RANDOM_STATE,
)

log = logging.getLogger(__name__)

REPORT_PATH = RESULTS_DIR / "metrics_report.txt"


# Загрузка тестовой выборки
def _load_test_set(model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает размеченную выборку, векторизует и возвращает
    (y_true, y_pred, y_proba_max) на тестовой части.

    Использует тот же split, что и при обучении (random_state из settings).
    """

    # Загружаем корпус и разметку
    df_corpus = pd.read_csv(PROCESSED_CSV, encoding="utf-8")
    df_labels = pd.read_csv(LABELED_CSV, encoding="utf-8",
                             usecols=["channel_username", "post_id", "sentiment"])
    df_labels["sentiment"] = pd.to_numeric(df_labels["sentiment"], errors="coerce")
    df_labels = df_labels.dropna(subset=["sentiment"])
    df_labels["sentiment"] = df_labels["sentiment"].astype(int)

    df = df_corpus.merge(df_labels, on=["channel_username", "post_id"], how="inner")
    df["text_lemma"] = df["text_lemma"].fillna("").astype(str)
    df = df[df["text_lemma"].str.strip().ne("")].reset_index(drop=True)

    X_text = df["text_lemma"].values
    y      = df["sentiment"].values

    # Воспроизводим тот же split
    _, X_test_text, _, y_true = train_test_split(
        X_text, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Загружаем модель и векторизатор
    if model_name == "svm":
        from models.svm_clf import load_model
    else:
        from models.sentiment import load_model

    vectorizer, model = load_model()
    X_test = vectorizer.transform(X_test_text)
    y_pred = model.predict(X_test)

    try:
        proba      = model.predict_proba(X_test)
        proba_max  = proba.max(axis=1)
    except AttributeError:
        proba_max  = np.ones(len(y_pred))

    return y_true, y_pred, proba_max, X_test_text


# Оценка одной модели
def evaluate_model(model_name: str = "logreg") -> dict:
    """
    Полная оценка модели на тестовой выборке.

    Returns:
        Словарь с метриками: precision/recall/f1 (macro, weighted, per-class),
        матрица ошибок, classification_report (строка).
    """
    log.info("Оценка модели: %s", model_name)
    y_true, y_pred, proba_max, _ = _load_test_set(model_name)

    labels      = sorted(set(y_true) | set(y_pred))
    label_names = [SENTIMENT_LABEL_NAMES[l] for l in labels]

    metrics = {
        "model":              model_name,
        "n_test":             len(y_true),
        "precision_macro":    float(precision_score(y_true, y_pred,
                                                    average="macro",
                                                    zero_division=0)),
        "recall_macro":       float(recall_score(y_true, y_pred,
                                                 average="macro",
                                                 zero_division=0)),
        "f1_macro":           float(f1_score(y_true, y_pred,
                                             average="macro",
                                             zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred,
                                                    average="weighted",
                                                    zero_division=0)),
        "recall_weighted":    float(recall_score(y_true, y_pred,
                                                 average="weighted",
                                                 zero_division=0)),
        "f1_weighted":        float(f1_score(y_true, y_pred,
                                             average="weighted",
                                             zero_division=0)),
        "mean_confidence":    float(proba_max.mean()),
        "confusion_matrix":   confusion_matrix(y_true, y_pred,
                                               labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        ),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None,
                            labels=labels, zero_division=0)
    for lbl, f1_val in zip(label_names, per_class_f1):
        metrics[f"f1_{lbl}"] = float(f1_val)

    log.info(
        "Результат [%s]: P=%.4f  R=%.4f  F1=%.4f (macro)",
        model_name,
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
    )
    log.info("\n%s", metrics["classification_report"])

    return metrics


# Сравнительная таблица
def compare_models() -> pd.DataFrame:
    """
    Строит сравнительную таблицу LogReg vs SVM.
    Использует сохранённый METRICS_JSON (результаты обучения),
    а не переоценку на тестовой выборке — для скорости.

    Returns:
        DataFrame: metric, logreg, svm, best_model, delta
    """
    if not METRICS_JSON.exists():
        raise FileNotFoundError(
            f"Файл метрик не найден: {METRICS_JSON}\n"
            "Обучите обе модели: python -m models.sentiment && "
            "python -m models.svm_clf"
        )

    with open(METRICS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    metrics_to_compare = [
        ("Precision (macro)",  "precision_macro",  False),
        ("Recall (macro)",     "recall_macro",     False),
        ("F1 (macro)",         "f1_macro",         False),
        ("CV F1 (среднее)",    "cv_f1_mean",       False),
        ("CV F1 (разброс)",    "cv_f1_std",        True),   # меньше = лучше
        ("Обучающих примеров", "n_train",          False),
        ("Тестовых примеров",  "n_test",           False),
        ("Признаков (TF-IDF)", "n_features",       False),
    ]

    rows = []
    for display_name, key, lower_is_better in metrics_to_compare:
        lr_val  = data.get("logreg", {}).get(key, float("nan"))
        svm_val = data.get("svm",    {}).get(key, float("nan"))

        if np.isnan(lr_val) or np.isnan(svm_val):
            best = "—"
            delta = float("nan")
        elif lower_is_better:
            best  = "logreg" if lr_val <= svm_val else "svm"
            delta = svm_val - lr_val
        else:
            best  = "logreg" if lr_val >= svm_val else "svm"
            delta = lr_val - svm_val     # > 0 означает LogReg лучше

        rows.append({
            "Метрика":   display_name,
            "LogReg":    round(lr_val,  4) if not np.isnan(lr_val)  else "—",
            "SVM":       round(svm_val, 4) if not np.isnan(svm_val) else "—",
            "Лучшая":   best,
            "Δ (LR−SVM)": round(delta, 4) if not np.isnan(delta) else "—",
        })

    result = pd.DataFrame(rows)
    log.info("\nСравнение LogReg vs SVM:\n%s", result.to_string(index=False))
    return result


# Анализ ошибок
def error_analysis(model_name: str = "logreg", n: int = 20) -> pd.DataFrame:
    """
    Возвращает n постов, классифицированных неверно с наибольшей уверенностью.
    Это самые «трудные» ошибки модели — полезны для качественного анализа.

    Returns:
        DataFrame: text (первые 200 символов), true_label, pred_label,
                   confidence
    """
    y_true, y_pred, proba_max, texts = _load_test_set(model_name)

    wrong_mask  = y_true != y_pred
    wrong_texts = texts[wrong_mask]
    wrong_true  = y_true[wrong_mask]
    wrong_pred  = y_pred[wrong_mask]
    wrong_conf  = proba_max[wrong_mask]

    # Сортируем по убыванию уверенности (самые «уверенные» ошибки — наверх)
    sort_idx = np.argsort(wrong_conf)[::-1][:n]

    rows = []
    for i in sort_idx:
        rows.append({
            "text_preview": str(wrong_texts[i])[:200],
            "true_label":   SENTIMENT_LABEL_NAMES.get(wrong_true[i], str(wrong_true[i])),
            "pred_label":   SENTIMENT_LABEL_NAMES.get(wrong_pred[i], str(wrong_pred[i])),
            "confidence":   round(float(wrong_conf[i]), 4),
        })

    result = pd.DataFrame(rows)
    log.info(
        "Анализ ошибок [%s]: всего ошибок=%d, показано=%d",
        model_name, wrong_mask.sum(), len(result),
    )
    return result


# Анализ уверенности
def confidence_analysis(model_name: str = "logreg") -> pd.DataFrame:
    """
    Распределение уверенности модели по классам и по правильности предсказания.

    Returns:
        DataFrame: bin (интервал уверенности), correct, incorrect, accuracy
    """
    y_true, y_pred, proba_max, _ = _load_test_set(model_name)

    bins   = np.arange(0.0, 1.05, 0.1)
    labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

    correct   = (y_true == y_pred).astype(int)
    bin_idx   = np.digitize(proba_max, bins) - 1
    bin_idx   = np.clip(bin_idx, 0, len(labels) - 1)

    rows = []
    for i, lbl in enumerate(labels):
        mask   = bin_idx == i
        n_corr = correct[mask].sum()
        n_incorr = mask.sum() - n_corr
        acc    = n_corr / mask.sum() if mask.sum() > 0 else float("nan")
        rows.append({
            "bin":       lbl,
            "correct":   int(n_corr),
            "incorrect": int(n_incorr),
            "total":     int(mask.sum()),
            "accuracy":  round(float(acc), 3) if not np.isnan(acc) else "—",
        })

    return pd.DataFrame(rows)


# Сохранение отчёта
def save_report(
    metrics_lr: dict,
    metrics_svm: dict,
    compare_df: pd.DataFrame,
) -> None:
    """Сохраняет текстовый отчёт с результатами обеих моделей."""
    lines = [
        "=" * 70,
        "ОТЧЁТ ОБ ОЦЕНКЕ КАЧЕСТВА КЛАССИФИКАТОРОВ ТОНАЛЬНОСТИ",
        "=" * 70,
        "",
        "LogisticRegression ",
        metrics_lr.get("classification_report", ""),
        "",
        "SVM (LinearSVC + калибровка) ",
        metrics_svm.get("classification_report", ""),
        "",
        "Сравнительная таблица",
        compare_df.to_string(index=False),
        "",
        "=" * 70,
    ]
    text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("Отчёт сохранён → %s", REPORT_PATH)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Оценка качества классификаторов тональности."
    )
    parser.add_argument(
        "--model", choices=["logreg", "svm", "both"], default="both",
        help="Модель для оценки (по умолчанию — обе).",
    )
    parser.add_argument(
        "--errors", action="store_true",
        help="Вывести анализ ошибок.",
    )
    parser.add_argument(
        "--confidence", action="store_true",
        help="Вывести анализ уверенности модели.",
    )
    args = parser.parse_args()

    model_names = (
        ["logreg", "svm"] if args.model == "both"
        else [args.model]
    )

    all_metrics = {}
    for name in model_names:
        try:
            all_metrics[name] = evaluate_model(name)
        except Exception as exc:
            log.error("Ошибка при оценке модели %s: %s", name, exc)

    if not all_metrics:
        raise RuntimeError(
            "Не удалось оценить ни одну модель. "
            "Проверьте наличие разметки/моделей и корректность входных данных."
        )

    if len(all_metrics) == 2:
        try:
            cmp = compare_models()
            save_report(
                all_metrics.get("logreg", {}),
                all_metrics.get("svm",    {}),
                cmp,
            )
        except Exception as exc:
            log.warning("Не удалось построить сравнительную таблицу: %s", exc)

    if args.errors:
        for name in model_names:
            try:
                err_df = error_analysis(name, n=10)
                print(f"\nАнализ ошибок [{name}]:")
                print(err_df.to_string(index=False))
            except Exception as exc:
                log.warning(str(exc))

    if args.confidence:
        for name in model_names:
            try:
                conf_df = confidence_analysis(name)
                print(f"\nУверенность модели [{name}]:")
                print(conf_df.to_string(index=False))
            except Exception as exc:
                log.warning(str(exc))


if __name__ == "__main__":
    main()