"""
nlp/vectorizer.py
=================
Третий шаг NLP-пайплайна: TF-IDF-векторизация лемматизированного корпуса.

Что делает:
    - Обучает TfidfVectorizer на колонке text_lemma
    - Сохраняет разреженную матрицу (scipy NPZ) и словарь (JSON)
    - Предоставляет функции для анализа значимости терминов:
        top_terms_global()   — топ-N слов по средневзвешенному TF-IDF
        top_terms_per_doc()  — топ-N слов для конкретного документа
        top_terms_per_group()— топ-N слов по группе (канал, ориентация, период)
    - Умеет строить динамику TF-IDF ключевых слов по временным периодам

Запуск:
    python -m nlp.vectorizer          # обучает и сохраняет матрицу
    python -m nlp.vectorizer --top 20 # выводит топ-20 терминов корпуса
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import (
    PROCESSED_CSV,
    TFIDF_MATRIX,
    TFIDF_VOCAB,
    TFIDF_PARAMS,
)

log = logging.getLogger(__name__)


# ── Обучение и сохранение ─────────────────────────────────────────────────────

def fit_vectorizer(
    texts: list[str] | pd.Series,
    params: dict | None = None,
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """
    Обучает TfidfVectorizer на корпусе.

    Args:
        texts:  итерируемая коллекция лемматизированных текстов
        params: параметры векторизатора (по умолчанию из settings)

    Returns:
        (vectorizer, tfidf_matrix) — обученный объект и разреженная матрица
    """
    if params is None:
        params = TFIDF_PARAMS

    log.info(
        "Обучение TF-IDF: документов=%d, max_features=%s, ngram_range=%s",
        len(texts), params.get("max_features"), params.get("ngram_range"),
    )

    vectorizer = TfidfVectorizer(**params)
    matrix = vectorizer.fit_transform(texts)

    log.info(
        "Матрица TF-IDF: форма=%s, ненулевых элементов=%d (плотность=%.4f%%)",
        matrix.shape,
        matrix.nnz,
        matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100,
    )
    return vectorizer, matrix


def save_matrix(
    vectorizer: TfidfVectorizer,
    matrix: sp.csr_matrix,
    matrix_path=TFIDF_MATRIX,
    vocab_path=TFIDF_VOCAB,
) -> None:
    """Сохраняет матрицу (NPZ) и словарь (JSON)."""
    sp.save_npz(str(matrix_path), matrix)
    log.info("Матрица сохранена → %s", matrix_path)

    vocab = {
        term: int(idx)
        for term, idx in vectorizer.vocabulary_.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    log.info("Словарь сохранён → %s  (%d терминов)", vocab_path, len(vocab))


def load_matrix(
    matrix_path=TFIDF_MATRIX,
    vocab_path=TFIDF_VOCAB,
) -> tuple[sp.csr_matrix, dict[str, int]]:
    """
    Загружает сохранённую матрицу и словарь.

    Returns:
        (matrix, vocabulary) — разреженная матрица и словарь {term: index}
    """
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Матрица не найдена: {matrix_path}\n"
            "Сначала запустите: python -m nlp.vectorizer"
        )
    matrix = sp.load_npz(str(matrix_path))
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    log.info("Матрица загружена: %s, словарь: %d терминов", matrix.shape, len(vocab))
    return matrix, vocab


# ── Анализ терминов ───────────────────────────────────────────────────────────

def top_terms_global(
    vectorizer: TfidfVectorizer,
    matrix: sp.csr_matrix,
    n: int = 30,
) -> pd.DataFrame:
    """
    Топ-N терминов по среднему TF-IDF-весу по всему корпусу.

    Returns:
        DataFrame с колонками ['term', 'mean_tfidf']
    """
    mean_tfidf = np.asarray(matrix.mean(axis=0)).flatten()
    terms = vectorizer.get_feature_names_out()

    idx = np.argsort(mean_tfidf)[::-1][:n]
    df = pd.DataFrame({
        "term":       terms[idx],
        "mean_tfidf": mean_tfidf[idx],
    })
    return df


def top_terms_per_group(
    df_corpus: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: sp.csr_matrix,
    group_col: str,
    n: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Топ-N терминов для каждой группы (например, по каналу или ориентации).

    Args:
        df_corpus: DataFrame корпуса (строки соответствуют строкам матрицы)
        group_col: колонка для группировки ('channel_label', 'orientation' и т.д.)
        n:         число топ-терминов

    Returns:
        Словарь {group_value: DataFrame(term, mean_tfidf)}
    """
    terms = vectorizer.get_feature_names_out()
    result = {}

    for group_val, group_idx in df_corpus.groupby(group_col).groups.items():
        sub_matrix = matrix[group_idx]
        mean_tfidf = np.asarray(sub_matrix.mean(axis=0)).flatten()
        idx = np.argsort(mean_tfidf)[::-1][:n]
        result[group_val] = pd.DataFrame({
            "term":       terms[idx],
            "mean_tfidf": mean_tfidf[idx],
        })

    return result


def tfidf_dynamics(
    df_corpus: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: sp.csr_matrix,
    keywords: list[str],
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Строит временну́ю динамику TF-IDF-весов для заданного списка ключевых слов.

    Позволяет отследить, как менялась значимость терминов
    (например, «схема», «реституция», «добросовестный») по месяцам.

    Args:
        df_corpus: DataFrame с колонкой 'date' (datetime)
        keywords:  список лемматизированных слов для отслеживания
        freq:      частота агрегации (по умолчанию 'ME' — конец месяца)

    Returns:
        DataFrame: индекс — период, колонки — ключевые слова, значения — mean TF-IDF
    """
    if "date" not in df_corpus.columns:
        raise ValueError("В корпусе нет колонки 'date'.")

    terms = vectorizer.get_feature_names_out()
    term_to_idx = {t: i for i, t in enumerate(terms)}

    # Оставляем только те слова, которые есть в словаре
    valid_kw = [kw for kw in keywords if kw in term_to_idx]
    if not valid_kw:
        log.warning("Ни одно из ключевых слов не найдено в словаре TF-IDF.")
        return pd.DataFrame()

    missing = set(keywords) - set(valid_kw)
    if missing:
        log.warning("Слова не найдены в словаре: %s", missing)

    df2 = df_corpus.copy()
    df2["period"] = pd.to_datetime(df2["date"], utc=True).dt.to_period(freq)

    records = []
    for period, grp in df2.groupby("period"):
        row = {"period": period}
        sub = matrix[grp.index]
        for kw in valid_kw:
            col_idx = term_to_idx[kw]
            row[kw] = float(sub[:, col_idx].mean())
        records.append(row)

    result = pd.DataFrame(records).set_index("period")
    result.index = result.index.astype(str)
    return result


# ── Основной пайплайн ─────────────────────────────────────────────────────────

def run_pipeline(
    input_path=PROCESSED_CSV,
    top_n: int = 30,
) -> tuple[TfidfVectorizer, sp.csr_matrix, pd.DataFrame]:
    """
    Загружает PROCESSED_CSV (с колонкой text_lemma) →
    обучает TF-IDF → сохраняет матрицу и словарь →
    выводит топ-N терминов.

    Returns:
        (vectorizer, matrix, df_corpus)
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {input_path}\n"
            "Сначала запустите: python -m nlp.lemmatizer"
        )

    df = pd.read_csv(input_path, encoding="utf-8")
    df["text_lemma"] = df["text_lemma"].fillna("").astype(str)

    # Убираем строки без лемматизированного текста
    df = df[df["text_lemma"].str.strip().ne("")].reset_index(drop=True)
    log.info("Корпус для векторизации: %d документов", len(df))

    vectorizer, matrix = fit_vectorizer(df["text_lemma"])
    save_matrix(vectorizer, matrix)

    # Топ-термины глобально
    top_df = top_terms_global(vectorizer, matrix, n=top_n)
    log.info("\nТоп-%d терминов корпуса:\n%s", top_n, top_df.to_string(index=False))

    return vectorizer, matrix, df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="TF-IDF векторизация корпуса.")
    parser.add_argument(
        "--top", type=int, default=30,
        help="Вывести топ-N терминов (по умолчанию 30).",
    )
    args = parser.parse_args()
    run_pipeline(top_n=args.top)


if __name__ == "__main__":
    main()