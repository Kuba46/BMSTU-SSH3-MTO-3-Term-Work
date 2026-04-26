"""
data/dataset.py
===============
Сборка, валидация и статистический обзор корпуса.
 
Функции:
    load_raw()         — загрузить сырые данные из CSV
    load_comments()    — загрузить сырые комментарии
    validate()         — проверить корпус на дубликаты, пустые тексты и т.д.
    split_by_channel() — разбить DataFrame по каналам
    corpus_stats()     — вывести сводную статистику
    merge_labeled()    — объединить посты с ручной разметкой
 
Запуск:
    python -m data.dataset      # выведет статистику корпуса
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import (
    RAW_CSV,
    LABELED_CSV,
    PROCESSED_CSV,
    CORPUS_START_DATE,
    CORPUS_END_DATE,
)

log = logging.getLogger(__name__)


# Загружает сырой CSV с постами. Приводит типы: date → datetime, числовые колонки → int.
def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден {path}.\n"
                                f"Пожалуйста, сначала запустите сборщик: python -m data.collector.")
    
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    for col in ["views", "forwards", "reactions_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0
    df["text"] = df["text"].fillna("").astype(str)
    log.info("Загружено %d постов из %s", len(df), path)
    return df


# Загружает CSV с комментариями (если существует).
def load_comments(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_CSV.parent / "comments_raw.csv"

    if not path.exists():
        log.warning("Файл комментариев не найден: %s", path)
        return pd.DataFrame(columns=["post_id", "comment_id", "text", "date", "views"])
    
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)
    log.info("Загружено %d комментариев из %s", len(df), path)
    return df


# Проверяет DataFrame на дубликаты, пустые тексты и посты вне диапазона. Удаляет их и возвращает очищенный DataFrame.
def validate(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates(subset=["channel_username", "post_id"])
    n_dup = n0 - len(df)
    if n_dup:
        log.warning("Удалено дубликатов: %d", n_dup)
    
    mask_empty = df["text"].str.strip().eq("")
    n_empty = mask_empty.sum()
    if n_empty:
        log.warning("Удалено постов с пустым текстом: %d", n_empty)
    df = df[~mask_empty]

    start_ts = pd.Timestamp(CORPUS_START_DATE, tz="UTC")
    end_ts = pd.Timestamp(CORPUS_END_DATE, tz="UTC").replace(
        hour=23, minute=59, second=59
    )
    mask_time = df["date"].between(start_ts, end_ts)
    n_out = (~mask_time).sum()
    if n_out:
        log.warning("Удалено постов вне диапазона корпуса: %d", n_out)
    df = df[mask_time]
 
    log.info(
        "Валидация: было %d → стало %d записей (удалено %d)",
        n0, len(df), n0 - len(df),
    )
    return df.reset_index(drop=True)


# Возвращает словарь {username: DataFrame} для каждого канала.
def split_by_channel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        username: group.reset_index(drop=True)
        for username, group in df.groupby("channel_username")
    }


# Возвращает (state_df, public_df) по ориентации канала.
def split_by_orientation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state  = df[df["orientation"] == "state"].reset_index(drop=True)
    public = df[df["orientation"] == "public"].reset_index(drop=True)
    return state, public


# Подтягивает метки тональности из файла ручной разметки. Ожидаемые колонки labeled_path: channel_username, post_id, sentiment. Возвращает объединённый DataFrame только с размеченными постами.
def merge_labeled(df_raw: pd.DataFrame, labeled_path: Path = LABELED_CSV) -> pd.DataFrame:
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Файл разметки не найден: {labeled_path}\n"
            "Запустите: python -m data.labeler"
        )
    
    df_labels = pd.read_csv(labeled_path, encoding="utf-8", usecols=["channel_username", "post_id", "sentiment"])
    df_labels["sentiment"] = pd.to_numeric(df_labels["sentiment"], errors="coerce")
    df_labels = df_labels.dropna(subset=["sentiment"])
    df_labels["sentiment"] = df_labels["sentiment"].astype(int)

    df_merged = df_raw.merge(
        df_labels,
        on=["channel_username", "post_id"],
        how="inner",
    )
    log.info(
        "Разметка подтянута: %d постов из %d размеченных",
        len(df_merged), len(df_labels),
    )
    return df_merged.reset_index(drop=True)



# Выводит сводную статистику корпуса в лог
def corpus_stats(df: pd.DataFrame) -> None:
    if df.empty:
        log.warning("DataFrame пуст — статистика недоступна.")
        return
    
    total = len(df)
    log.info("=" * 60)
    log.info("СТАТИСТИКА КОРПУСА")
    log.info("=" * 60)
    log.info("Всего постов: %d", total)
    log.info("Период:       %s → %s",
             df["date"].min().date(), df["date"].max().date())
    log.info("Каналов:      %d", df["channel_username"].nunique())
 
    log.info("\nПо ориентации: ")
    orient = (
        df.groupby("orientation")
        .agg(
            posts=("post_id", "count"),
            views=("views", "sum"),
            reactions=("reactions_total", "sum"),
        )
    )
    log.info("\n%s", orient.to_string())
 
    log.info("\nПо каналам: ")
    by_ch = (
        df.groupby(["channel_label", "orientation"])
        .agg(
            posts=("post_id", "count"),
            views=("views", "sum"),
            reactions=("reactions_total", "sum"),
            forwards=("forwards", "sum"),
        )
        .sort_values("posts", ascending=False)
    )
    log.info("\n%s", by_ch.to_string())

    log.info("\nВременная динамика (по месяцам): ")
    df2 = df.copy()
    df2["month"] = df2["date"].dt.to_period("M")
    monthly = df2.groupby("month").size().rename("posts")
    log.info("\n%s", monthly.to_string())
 
    log.info("\nДлина текста (символов): ")
    lengths = df["text"].str.len()
    log.info(
        "min=%d  median=%d  mean=%.0f  max=%d",
        lengths.min(), lengths.median(), lengths.mean(), lengths.max(),
    )
    log.info("=" * 60)


# Сохраняет обработанный DataFrame в CSV. Если файл уже существует, перезаписывает его.
def save_processed(df: pd.DataFrame, path: Path = PROCESSED_CSV) -> None:
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Сохранено: %s  (%d строк)", path, len(df))


# Загружает обработанный DataFrame. Если файла нет, выдаёт ошибку с инструкцией по запуску NLP-пайплайна.
def load_processed(path: Path = PROCESSED_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Обработанный файл не найден: {path}\n"
            "Запустите NLP-пайплайн: python -m nlp.preprocessor"
        )
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    df_raw = load_raw()
    df_valid = validate(df_raw)
    corpus_stats(df_valid)
    save_processed(df_valid)


if __name__ == "__main__":
    main()