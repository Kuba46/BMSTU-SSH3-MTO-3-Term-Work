"""
nlp/preprocessor.py
===================
Первый шаг NLP-пайплайна: очистка сырого текста Telegram-постов.
 
Что делает:
    - Приводит к нижнему регистру
    - Удаляет URL, упоминания (@username), хэштеги (#tag)
    - Удаляет эмодзи и прочие не-ASCII / не-кириллические символы
    - Удаляет служебные Telegram-паттерны (forwarded from, via @bot и т.д.)
    - Нормализует пробелы и переносы строк
    - Удаляет посты, которые стали пустыми после очистки
 
Запуск:
    python -m nlp.preprocessor          # читает RAW_CSV, пишет в PROCESSED_CSV
    python -m nlp.preprocessor --demo   # показывает примеры очистки в терминале
"""


import re
import argparse
import logging
import unicodedata
import pandas as pd

from config import RAW_CSV, PROCESSED_CSV


log = logging.getLogger(__name__)


_RE_URL = re.compile(
    r"https?://\S+|www\.\S+|t\.me/\S+",
    flags=re.IGNORECASE
)

_RE_MENTION = re.compile(r"@\w+")

_RE_HASHTAG = re.compile(r"#\w+")

_RE_FORWARDED = re.compile(
    r"(переслано\s+от|forwarded\s+from|via\s+@\w+)",
    flags=re.IGNORECASE,
)

# Числа — удаляем отдельностоящие числа, но оставляем в составе слов
# (например «2025» удалим, «covid19» оставим)
_RE_STANDALONE_NUM = re.compile(r"\b\d+\b")

# Диапазоны номеров дел (оставляем, они могут быть ключевыми)
# — не удаляем паттерны вида «02-0387/2025»

# Пунктуация — оставляем только буквы (кириллица + латиница), цифры, пробел
# и дефис внутри слов; всё остальное → пробел
_RE_PUNCT = re.compile(r"[^\w\s\-]", flags=re.UNICODE)

# Множественные пробелы / переносы строк → один пробел
_RE_SPACES = re.compile(r"\s+")

# Дефис в начале/конце слова (артефакт после очистки)
_RE_LOOSE_DASH = re.compile(r"(?<!\w)-|-(?!\w)")


# Удаляет эмодзи и прочие символы вне базового многоязычного плана Unicode.
def remove_emoji(text: str) -> str:
    result = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith(("L", "N", "Z", "P", "M")) or char in " \n\t-":
            result.append(char)
        elif ord(char) >= 0x1F300:
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    text = _RE_FORWARDED.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)
    text = _RE_HASHTAG.sub(" ", text)
    text = remove_emoji(text)
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_STANDALONE_NUM.sub(" ", text)
    text = _RE_LOOSE_DASH.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


# Применяет очистку к каждому посту в DataFrame, удаляя пустые после очистки.
def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    n0 = len(df)
    df = df.copy()

    log.info("Очистка текста: %d записей...", n0)
    df["text_clean"] = df[text_col].apply(clean_text)

    empty_mask = df["text_clean"].str.strip().eq("")
    n_empty = empty_mask.sum()
    if n_empty:
        log.warning("Удалено %d записей с пустым текстом после очистки", n_empty)
    df = df[~empty_mask].reset_index(drop=True)

    log.info("Очистка завершена: было %d → стало %d записей", n0, len(df))
    return df


# Основная функция: читает сырой CSV, очищает текст, сохраняет результат в новый CSV.
def run_pipeline(input_path=RAW_CSV, output_path=PROCESSED_CSV) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Входной файл не найден: {input_path}\n"
            "Сначала запустите: python -m data.collector"
        )

    df = pd.read_csv(input_path, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)

    df_clean = preprocess_dataframe(df)
    df_clean.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Сохранено → %s", output_path)

    return df_clean
 
 
def _demo() -> None:
    """Показывает примеры очистки на типичных Telegram-текстах."""
    examples = [
        "🔥 СРОЧНО! Верховный суд отменил решение по делу Долиной https://t.me/shot_shot/12345",
        "Переслано от @rian_ru\n#суд #долина #лурье — сегодня важный день",
        "via @rian_ru Квартира стоимостью 112 млн рублей возвращена Долиной 😤👇",
        "Схема Долиной: как это работает? Подробнее: www.pravo.ru/story/12345",
        "RT @davankov: Верховный суд принял правильное решение! Лурье получит квартиру обратно.",
    ]
    print("\n── ДЕМО: очистка текста ─────────────────────────────────────────")
    for i, text in enumerate(examples, 1):
        cleaned = clean_text(text)
        print(f"\n[{i}] ИСХОДНЫЙ:\n  {text!r}")
        print(f"    ОЧИЩЕННЫЙ:\n  {cleaned!r}")
    print("\n─────────────────────────────────────────────────────────────────\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Очистка текста Telegram-постов.")
    parser.add_argument("--demo", action="store_true",
                        help="Показать примеры очистки без запуска пайплайна.")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()