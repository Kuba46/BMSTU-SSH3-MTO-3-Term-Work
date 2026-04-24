"""
nlp/lemmatizer.py
=================
Второй шаг NLP-пайплайна: морфологический анализ и лемматизация.

Что делает:
    - Токенизирует очищенный текст по пробелам
    - Для каждого токена определяет лемму через pymorphy2
    - Фильтрует токены по части речи (ALLOWED_POS из settings)
    - Удаляет стоп-слова (встроенные NLTK + пользовательский список)
    - Удаляет слишком короткие токены (< MIN_TOKEN_LEN символов)
    - Возвращает строку лемматизированных токенов через пробел

Запуск:
    python -m nlp.lemmatizer          # обрабатывает PROCESSED_CSV
    python -m nlp.lemmatizer --demo   # примеры в терминале
"""

import argparse
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pymorphy2
from nltk.corpus import stopwords

from config.settings import (
    PROCESSED_CSV,
    STOPWORDS_PATH,
    MIN_TOKEN_LENGTH,
    ALLOWED_POS_TAGS,
)

log = logging.getLogger(__name__)

# ── Инициализация анализатора ─────────────────────────────────────────────────
# MorphAnalyzer создаётся один раз на уровне модуля — это дорогая операция
_morph = pymorphy2.MorphAnalyzer()


def _load_stopwords() -> frozenset[str]:
    """
    Загружает объединённый список стоп-слов:
      1. NLTK Russian stopwords
      2. Пользовательский файл STOPWORDS_PATH (если существует)

    Возвращает frozenset лемматизированных стоп-слов.
    """
    # NLTK
    try:
        nltk_stops = set(stopwords.words("russian"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        nltk_stops = set(stopwords.words("russian"))

    # Пользовательский файл
    custom_stops: set[str] = set()
    if Path(STOPWORDS_PATH).exists():
        with open(STOPWORDS_PATH, encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith("#"):
                    custom_stops.add(word)
        log.debug("Загружено %d пользовательских стоп-слов", len(custom_stops))
    else:
        log.warning("Файл стоп-слов не найден: %s (используются только NLTK)", STOPWORDS_PATH)

    combined = nltk_stops | custom_stops
    log.info("Стоп-слов всего: %d (NLTK=%d, custom=%d)",
             len(combined), len(nltk_stops), len(custom_stops))
    return frozenset(combined)


# Загружаем стоп-слова при импорте модуля
_STOPWORDS: frozenset[str] = _load_stopwords()


@lru_cache(maxsize=50_000)
def _lemmatize_token(token: str) -> tuple[str, str]:
    """
    Лемматизирует один токен через pymorphy2.
    Возвращает (лемма, POS-тег).

    Кэшируется через lru_cache — при повторяющихся словах
    экономит значительное время.
    """
    parsed = _morph.parse(token)
    if not parsed:
        return token, "UNKN"
    best = parsed[0]
    lemma = best.normal_form
    pos   = best.tag.POS or "UNKN"
    return lemma, pos


def lemmatize_text(text: str) -> str:
    """
    Полный конвейер лемматизации одного текста.

    Args:
        text: очищенный текст (после preprocessor.py)

    Returns:
        Строка лемматизированных токенов, разделённых пробелами.
        Может быть пустой строкой, если все токены отфильтрованы.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    tokens = text.split()
    result = []

    for token in tokens:
        # Слишком короткие токены отбрасываем до лемматизации (быстро)
        if len(token) < MIN_TOKEN_LENGTH:
            continue

        lemma, pos = _lemmatize_token(token)

        # Фильтр по части речи
        if pos not in ALLOWED_POS_TAGS:
            continue

        # Фильтр по стоп-словам (проверяем и оригинал, и лемму)
        if lemma in _STOPWORDS or token in _STOPWORDS:
            continue

        # Повторная проверка длины леммы
        if len(lemma) < MIN_TOKEN_LENGTH:
            continue

        result.append(lemma)

    return " ".join(result)


def lemmatize_dataframe(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    out_col: str = "text_lemma",
) -> pd.DataFrame:
    """
    Применяет lemmatize_text ко всему DataFrame.

    Args:
        df:       DataFrame с очищенным текстом
        text_col: колонка с очищенным текстом (после preprocessor)
        out_col:  название новой колонки с лемматизированным текстом

    Returns:
        DataFrame с добавленной колонкой out_col.
        Строки с пустым результатом удаляются.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Колонка '{text_col}' не найдена. "
            "Убедитесь, что preprocessor.py уже выполнен."
        )

    n0 = len(df)
    log.info("Лемматизация: %d записей...", n0)

    df = df.copy()
    df[out_col] = df[text_col].apply(lemmatize_text)

    # Удаляем строки, где после лемматизации ничего не осталось
    empty_mask = df[out_col].str.strip().eq("")
    n_empty = empty_mask.sum()
    if n_empty:
        log.warning("Удалено %d записей с пустым результатом лемматизации", n_empty)
    df = df[~empty_mask].reset_index(drop=True)

    # Статистика
    token_counts = df[out_col].str.split().str.len()
    log.info(
        "Лемматизация завершена: было %d → стало %d записей. "
        "Токенов: min=%d median=%.0f max=%d",
        n0, len(df),
        token_counts.min(),
        token_counts.median(),
        token_counts.max(),
    )

    return df


def run_pipeline(input_path=PROCESSED_CSV, output_path=PROCESSED_CSV) -> pd.DataFrame:
    """
    Загружает PROCESSED_CSV (после preprocessor) → лемматизирует →
    перезаписывает тот же файл с добавленной колонкой text_lemma.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {input_path}\n"
            "Сначала запустите: python -m nlp.preprocessor"
        )

    df = pd.read_csv(input_path, encoding="utf-8")
    df["text_clean"] = df["text_clean"].fillna("").astype(str)

    df_lemma = lemmatize_dataframe(df)
    df_lemma.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Сохранено → %s", output_path)

    return df_lemma


def _demo() -> None:
    """Демонстрирует лемматизацию на примерах."""
    examples = [
        "схема долиной добросовестный покупатель реституция суд квартира",
        "верховный суд отменил несправедливое решение хамовнического суда",
        "общественное мнение сформировалось в телеграм-каналах за несколько месяцев",
        "логистическая регрессия классификация тональности постов каналов",
        "анализ кластеризация kmeans dbscan визуализация результаты",
    ]
    print("\n── ДЕМО: лемматизация ───────────────────────────────────────────")
    for i, text in enumerate(examples, 1):
        lemmatized = lemmatize_text(text)
        print(f"\n[{i}] ИСХОДНЫЙ:\n  {text!r}")
        print(f"    ЛЕММАТИЗИРОВАННЫЙ:\n  {lemmatized!r}")
    print("\n─────────────────────────────────────────────────────────────────\n")

    # Показываем статистику кэша
    ci = _lemmatize_token.cache_info()
    print(f"  Кэш лемматизатора: hits={ci.hits}, misses={ci.misses}, "
          f"size={ci.currsize}/{ci.maxsize}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Лемматизация текста постов.")
    parser.add_argument("--demo", action="store_true",
                        help="Показать примеры лемматизации без запуска пайплайна.")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()