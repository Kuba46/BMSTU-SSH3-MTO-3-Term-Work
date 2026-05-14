"""
data/labeler.py
===============
Интерактивный инструмент ручной разметки тональности постов.

Логика работы:
    1. Загружает обработанный корпус (PROCESSED_CSV).
    2. Если файл разметки уже существует — подгружает его и пропускает
    уже размеченные посты (можно прерваться и продолжить).
    3. Показывает оператору текст поста и ждёт нажатия:
        p  → позитивная  (+1)
        n  → нейтральная (0)
        g  → негативная  (-1)   [от «грустная»]
        s  → пропустить (не добавлять в разметку)
        q  → сохранить и выйти
    4. Каждые AUTOSAVE_EVERY записей автоматически сохраняет прогресс.
    5. При завершении сохраняет labeled/posts_labeled.csv.

Запуск:
    python -m data.labeler                 # разметить все неразмеченные
    python -m data.labeler --n 100         # разметить первые 100 (для теста)
    python -m data.labeler --ch shot_shot  # разметить только конкретный канал
"""


import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from config.settings import (
    PROCESSED_CSV,
    LABELED_CSV,
    SENTIMENT_LABELS,
    SENTIMENT_LABEL_NAMES,
)

log = logging.getLogger(__name__)

AUTOSAVE_EVERY = 25   # сохранять прогресс каждые N размеченных постов

# Управляющие клавиши
KEY_MAP = {
    "p": SENTIMENT_LABELS["positive"],   # +1
    "n": SENTIMENT_LABELS["neutral"],    #  0
    "g": SENTIMENT_LABELS["negative"],   # -1
}

HELP_LINE = (
    "[p] позитивная  [n] нейтральная  [g] негативная"
    "[s] пропустить  [q] сохранить и выйти"
)


# Очистка терминала для удобства чтения. Работает на любой ОС.
def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# Символы-подсказки для отображения эмодзи-категорий
_CAT_ICONS = {"positive": "🟢", "negative": "🔴", "neutral": "🟡", "unknown": "⚪"}


def _emoji_hint(row: pd.Series) -> str:
    """
    Формирует строку-подсказку на основе эмодзи-реакций под постом.
    Помогает оператору быстрее определить тональность.

    Логика: смотрим на reactions_top и классифицируем каждое эмодзи
    по словарю EMOJI_SENTIMENT. Показываем доминирующую категорию.
    """
    try:
        from config.settings import EMOJI_SENTIMENT, EMOJI_LIKE_DISLIKE
        rtype = None
        for ch in __import__("config.settings", fromlist=["CHANNELS"]).CHANNELS:
            if ch["username"] == row.get("channel_username", ""):
                rtype = ch.get("reaction_type", "none")
                break

        if rtype == "none":
            return "  💬 Реакций нет (только комментарии)"

        emoji_dict = EMOJI_LIKE_DISLIKE if rtype == "like_dislike" else EMOJI_SENTIMENT
        reactions_top = str(row.get("reactions_top", "")).strip()

        if not reactions_top:
            return "  ⚪ Реакции отсутствуют"

        counts = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
        emojis = reactions_top.split()
        for em in emojis:
            cat = emoji_dict.get(em, "unknown")
            counts[cat] += 1

        # Доминирующая категория
        dominant = max(
            ["positive", "negative", "neutral"],
            key=lambda c: counts[c]
        )
        icon = _CAT_ICONS.get(dominant, "⚪")
        total_react = int(row.get("reactions_total", 0))

        parts = []
        for em in emojis:
            cat = emoji_dict.get(em, "unknown")
            parts.append(f"{em}{_CAT_ICONS.get(cat, '⚪')}")

        hint = f"  Реакции: {' '.join(parts)}  ({total_react:,} всего)"
        hint += f"  →  Подсказка ESI: {icon} {dominant.upper()}"
        return hint

    except Exception:
        return ""


# Отображает пост для разметки. Показывает канал, дату, статистику и текст (обрезая слишком длинные).
def _show_post(idx: int, total: int, row: pd.Series) -> None:
    _clear()
    print(f"{'─' * 70}")
    print(f"  Пост {idx + 1} / {total}")
    print(f"  Канал:  {row['channel_label']}  (@{row['channel_username']})")
    print(f"  Дата:   {str(row['date'])[:10]}")
    print(f"  Просм.: {int(row.get('views', 0)):,}   "
          f"Репосты: {int(row.get('forwards', 0)):,}")
    # Подсказка на основе эмодзи — выделяем цветом
    hint = _emoji_hint(row)
    if hint:
        print(f"\\033[33m{hint}\\033[0m")   # жёлтый цвет в терминале
    print(f"{'─' * 70}")
    # Обрезаем слишком длинные тексты для удобства чтения
    text = row["text"]
    if len(text) > 800:
        text = text[:800] + "\\n[...текст обрезан...]"
    print(f"\\n{text}\\n")
    print(f"{'─' * 70}")
    print(HELP_LINE)
    print()


# Загружает уже существующую разметку (если есть), чтобы не размечать одни и те же посты повторно.
def _load_existing_labels(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except pd.errors.EmptyDataError:
            log.warning("Файл разметки пуст: %s", path)
            return pd.DataFrame(columns=["channel_username", "post_id", "sentiment"])
        log.info("Найдена существующая разметка: %d записей (%s)", len(df), path)
        return df
    return pd.DataFrame(columns=["channel_username", "post_id", "sentiment"])


# Сохраняет накопленные метки в CSV. Если файла нет, создаёт новый.
def _save_labels(labels: list[dict], path: Path) -> None:
    if not labels:
        return
    df = pd.DataFrame(labels)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Разметка сохранена: %d записей → %s", len(df), path)


# Основной цикл ручной разметки. Загружает сырые посты, пропускает уже размеченные, показывает по одному и ждёт ввода.
def run_labeling(n_limit: int | None = None, channel_filter: list[str] | None = None,) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Загрузка обработанного корпуса
    if not PROCESSED_CSV.exists():
        print(f"[ОШИБКА] Обработанный файл не найден: {PROCESSED_CSV}")
        print("Сначала запустите: python -m nlp.preprocessor && python -m nlp.lemmatizer")
        sys.exit(1)

    df_raw = pd.read_csv(PROCESSED_CSV, encoding="utf-8")
    df_raw["text"] = df_raw["text"].fillna("").astype(str)

    # Фильтр по каналу
    if channel_filter:
        df_raw = df_raw[df_raw["channel_username"].isin(channel_filter)]
        log.info("Фильтр по каналам: %s (%d постов)", channel_filter, len(df_raw))

    # Убираем уже размеченные
    df_existing = _load_existing_labels(LABELED_CSV)
    if not df_existing.empty:
        already_done = set(
            zip(df_existing["channel_username"], df_existing["post_id"])
        )
        mask = df_raw.apply(
            lambda r: (r["channel_username"], r["post_id"]) not in already_done,
            axis=1,
        )
        df_raw = df_raw[mask]
        log.info("Пропущено уже размеченных: %d. Осталось: %d", len(already_done), len(df_raw))

    if df_raw.empty:
        print("Все посты уже размечены!")
        return

    # Применяем лимит
    if n_limit:
        df_raw = df_raw.head(n_limit)

    total = len(df_raw)
    print(f"\n  Начинаем разметку. Постов для разметки: {total}")
    print(HELP_LINE)
    input("\n  Нажмите Enter для старта...")

    # Накапливаем новые метки; начинаем с уже существующих
    all_labels: list[dict] = df_existing.to_dict("records")
    new_labels: list[dict] = []

    stats = {"p": 0, "n": 0, "g": 0, "s": 0}
    df_raw = df_raw.reset_index(drop=True)

    for idx, row in df_raw.iterrows():
        _show_post(idx, total, row)

        # Ожидаем валидный ввод
        while True:
            key = input("  Ваш выбор: ").strip().lower()
            if key in KEY_MAP:
                label = KEY_MAP[key]
                record = {
                    "channel_username": row["channel_username"],
                    "post_id":          row["post_id"],
                    "sentiment":        label,
                }
                new_labels.append(record)
                all_labels.append(record)
                stats[key] += 1
                print(
                    f"  → {SENTIMENT_LABEL_NAMES[label].upper()}  "
                    f"(позитивных: {stats['p']}, "
                    f"нейтральных: {stats['n']}, "
                    f"негативных: {stats['g']}, "
                    f"пропущено: {stats['s']})"
                )
                break
            elif key == "s":
                stats["s"] += 1
                print("  → Пост пропущен.")
                break
            elif key == "q":
                print("\n  Прерывание по команде пользователя. Сохраняем...")
                _save_labels(all_labels, LABELED_CSV)
                _print_summary(stats)
                return
            else:
                print("  Неверный ввод. Используйте: p / n / g / s / q")

        # Автосохранение
        if len(new_labels) % AUTOSAVE_EVERY == 0 and new_labels:
            _save_labels(all_labels, LABELED_CSV)
            print(f"\n  [Автосохранение: {len(all_labels)} записей]\n")

    # Финальное сохранение
    _save_labels(all_labels, LABELED_CSV)
    _print_summary(stats)


# Выводит сводную статистику по итогам разметки.
def _print_summary(stats: dict) -> None:
    total_labeled = stats["p"] + stats["n"] + stats["g"]
    print(f"\n{'═' * 50}")
    print("  ИТОГИ РАЗМЕТКИ")
    print(f"{'═' * 50}")
    print(f"  Размечено:    {total_labeled}")
    print(f"  Позитивных:   {stats['p']}")
    print(f"  Нейтральных:  {stats['n']}")
    print(f"  Негативных:   {stats['g']}")
    print(f"  Пропущено:    {stats['s']}")
    if total_labeled:
        neg_pct = stats["g"] / total_labeled * 100
        pos_pct = stats["p"] / total_labeled * 100
        print(f"\n  Доля негативных: {neg_pct:.1f}%")
        print(f"  Доля позитивных: {pos_pct:.1f}%")
    print(f"{'═' * 50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Интерактивная ручная разметка тональности постов."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="Разметить не более N постов (по умолчанию — все нераzmеченные).",
    )
    parser.add_argument(
        "--ch",
        nargs="*",
        metavar="USERNAME",
        help="Разметить только указанные каналы (по username без @).",
    )
    args = parser.parse_args()
    run_labeling(n_limit=args.n, channel_filter=args.ch)


if __name__ == "__main__":
    main()