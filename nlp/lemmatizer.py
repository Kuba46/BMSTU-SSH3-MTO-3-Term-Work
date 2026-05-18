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

Зависимости:
    pip install pymorphy3

Запуск:
    python -m nlp.lemmatizer          # обрабатывает PROCESSED_CSV
    python -m nlp.lemmatizer --input data/processed/comments_processed.csv --output data/processed/comments_processed.csv
    python -m nlp.lemmatizer --demo   # примеры в терминале
"""

import argparse
import logging
import pandas as pd

from functools import lru_cache
from pathlib import Path

# ── Совместимость pymorphy2 с Python 3.12 ────────────────────────────────────
# В Python 3.12 удалён pkg_resources (часть setuptools).
# pymorphy2 использует его для поиска словарей через entry_points.
# Решение 1 (рекомендуется): pip install setuptools pymorphy2-dicts-ru
# Решение 2 (fallback): заглушка pkg_resources через importlib.metadata
import sys
import types

try:
    import pymorphy3 as _pymorphy
    _MORPH_LIB = "pymorphy3"
except ImportError:
    try:
        import pymorphy3 as _pymorphy
        _MORPH_LIB = "pymorphy3"
    except ImportError:
        raise ImportError(
            "Не найдена библиотека морфологического анализа.\n"
            "Установите одну из:\n"
            "  pip install setuptools pymorphy2-dicts-ru   (рекомендуется)\n"
            "  pip install pymorphy3                       (Python 3.12+)"
        )

from config.settings import (
    PROCESSED_CSV,
    STOPWORDS_PATH,
    MIN_TOKEN_LENGTH,
    ALLOWED_POS,
)

log = logging.getLogger(__name__)

# ── Инициализация анализатора ─────────────────────────────────────────────────
# MorphAnalyzer создаётся один раз на уровне модуля — это дорогая операция.
# Используем _pymorphy — алиас, поддерживающий pymorphy2 и pymorphy3.
import logging as _log
_log.getLogger(__name__).debug("Морфологический анализатор: %s", _MORPH_LIB)
_morph = _pymorphy.MorphAnalyzer()


# Встроенный базовый список русских стоп-слов.
# Используется если stopwords_ru.txt не найден или пуст.
# Покрывает служебные части речи и наиболее частотные слова.
_BUILTIN_STOPWORDS = frozenset({
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у",
    "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот",
    "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть",
    "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "сказал",
    "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они",
    "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их",
    "чем", "была", "сам", "чтоб", "без", "будто", "человек", "чего",
    "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот",
    "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом",
    "один", "почти", "мой", "тем", "чтобы", "нее", "кажется", "сейчас",
    "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец",
    "два", "об", "другой", "хоть", "после", "над", "больше", "тот",
    "через", "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между",
    # вспомогательные глаголы
    "быть", "стать", "являться", "иметь", "делать", "говорить",
    "мочь", "хотеть", "знать", "думать", "видеть", "получить",
})


def _load_stopwords() -> frozenset[str]:
    """
    Загружает стоп-слова из двух источников (без зависимости от NLTK):
      1. Встроенный базовый список (_BUILTIN_STOPWORDS)
      2. Пользовательский файл nlp/stopwords_ru.txt

    NLTK намеренно не используется — избегаем зависимости от
    сетевого скачивания и проблем с SSL-сертификатами.
    """
    # Пользовательский файл
    custom_stops: set[str] = set()
    sw_path = Path(STOPWORDS_PATH)
    if sw_path.exists():
        with open(sw_path, encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith("#"):
                    custom_stops.add(word)
        log.debug("Загружено %d стоп-слов из %s", len(custom_stops), sw_path.name)
    else:
        log.warning(
            "Файл стоп-слов не найден: %s — используется только встроенный список.",
            STOPWORDS_PATH,
        )

    combined = _BUILTIN_STOPWORDS | custom_stops
    log.info(
        "Стоп-слов загружено: %d (встроенных=%d, из файла=%d)",
        len(combined), len(_BUILTIN_STOPWORDS), len(custom_stops),
    )
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
        if pos not in ALLOWED_POS:
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
    ]
    print("\n── ДЕМО: лемматизация ───────────────────────────────────────────")
    for i, text in enumerate(examples, 1):
        lemmatized = lemmatize_text(text)
        print(f"\n[{i}] ИСХОДНЫЙ:\n  {text!r}")
        print(f"    ЛЕММАТИЗИРОВАННЫЙ:\n  {lemmatized!r}")

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
    parser.add_argument("--demo", action="store_true", help="Показать примеры лемматизации без запуска пайплайна.")
    parser.add_argument("--input", type=str, default=None, help="Путь к входному CSV (по умолчанию PROCESSED_CSV).")
    parser.add_argument("--output", type=str, default=None, help="Путь к выходному CSV (по умолчанию PROCESSED_CSV).")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        input_path = PROCESSED_CSV if args.input is None else Path(args.input)
        output_path = PROCESSED_CSV if args.output is None else Path(args.output)
        run_pipeline(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()