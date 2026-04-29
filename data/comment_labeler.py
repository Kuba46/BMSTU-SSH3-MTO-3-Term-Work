"""
data/comment_labeler.py
=======================
Интерактивный инструмент ручной разметки тональности комментариев.

Логика работы:
    1. Загружает обработанный корпус комментариев (COMMENTS_PROCESSED_CSV).
    2. Подхватывает существующую разметку (COMMENTS_LABELED_CSV), чтобы можно
       было продолжать с места остановки.
    3. Показывает текст комментария и ожидает ввод:
        p  → позитивная (+1)
        n  → нейтральная (0)
        g  → негативная (-1)
        s  → пропустить
        q  → сохранить и выйти
    4. Автосохраняет прогресс каждые AUTOSAVE_EVERY записей.

Запуск:
    python -m data.comment_labeler
    python -m data.comment_labeler --n 300
    python -m data.comment_labeler --ch shot_shot davankov
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from config.settings import (
    COMMENTS_PROCESSED_CSV,
    COMMENTS_LABELED_CSV,
    SENTIMENT_LABELS,
    SENTIMENT_LABEL_NAMES,
)

log = logging.getLogger(__name__)

AUTOSAVE_EVERY = 50

KEY_MAP = {
    "p": SENTIMENT_LABELS["positive"],
    "n": SENTIMENT_LABELS["neutral"],
    "g": SENTIMENT_LABELS["negative"],
}

HELP_LINE = (
    "[p] позитивная  [n] нейтральная  [g] негативная  "
    "[s] пропустить  [q] сохранить и выйти"
)


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _show_comment(idx: int, total: int, row: pd.Series) -> None:
    _clear()
    print(f"{'─' * 80}")
    print(f"  Комментарий {idx + 1} / {total}")
    print(f"  Канал:  {row.get('channel_label', 'unknown')}  (@{row['channel_username']})")
    print(f"  Дата:   {str(row.get('date', ''))[:19]}")
    print(f"  post_id: {row.get('post_id', 'NA')}   comment_id: {row.get('comment_id', 'NA')}")
    print(f"{'─' * 80}")
    text = str(row.get("text", ""))
    if len(text) > 1200:
        text = text[:1200] + "\n[...текст обрезан...]"
    print(f"\n{text}\n")
    print(f"{'─' * 80}")
    print(HELP_LINE)
    print()


def _load_existing_labels(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, encoding="utf-8")
        if "comment_id" not in df.columns:
            raise ValueError(
                f"Файл {path} не содержит колонку comment_id. "
                "Проверьте, что это разметка комментариев."
            )
        log.info("Найдена существующая разметка комментариев: %d (%s)", len(df), path)
        return df
    return pd.DataFrame(columns=["channel_username", "post_id", "comment_id", "sentiment"])


def _save_labels(labels: list[dict], path: Path) -> None:
    if not labels:
        return
    df = pd.DataFrame(labels)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Разметка комментариев сохранена: %d записей → %s", len(df), path)


def run_labeling(n_limit: int | None = None, channel_filter: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not COMMENTS_PROCESSED_CSV.exists():
        print(f"[ОШИБКА] Файл не найден: {COMMENTS_PROCESSED_CSV}")
        print(
            "Сначала запустите:\n"
            "  python -m data.dataset --comments\n"
            "  python -m nlp.preprocessor --input data/raw/comments_raw.csv --output data/processed/comments_processed.csv\n"
            "  python -m nlp.lemmatizer --input data/processed/comments_processed.csv --output data/processed/comments_processed.csv"
        )
        sys.exit(1)

    df_raw = pd.read_csv(COMMENTS_PROCESSED_CSV, encoding="utf-8")
    df_raw["text"] = df_raw["text"].fillna("").astype(str)

    required_cols = {"channel_username", "comment_id", "text"}
    missing_cols = required_cols - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"В {COMMENTS_PROCESSED_CSV.name} отсутствуют колонки: {sorted(missing_cols)}")

    if channel_filter:
        df_raw = df_raw[df_raw["channel_username"].isin(channel_filter)]
        log.info("Фильтр по каналам: %s (%d комментариев)", channel_filter, len(df_raw))

    df_existing = _load_existing_labels(COMMENTS_LABELED_CSV)
    if not df_existing.empty:
        already_done = set(
            zip(
                df_existing["channel_username"].astype(str),
                df_existing["comment_id"].astype(str),
            )
        )
        mask = df_raw.apply(
            lambda r: (str(r["channel_username"]), str(r["comment_id"])) not in already_done,
            axis=1,
        )
        df_raw = df_raw[mask]
        log.info("Пропущено уже размеченных: %d. Осталось: %d", len(already_done), len(df_raw))

    df_raw = df_raw[df_raw["text"].str.strip().ne("")]
    if df_raw.empty:
        print("Все комментарии уже размечены (или нет текстов для разметки).")
        return

    if n_limit:
        df_raw = df_raw.head(n_limit)

    total = len(df_raw)
    print(f"\n  Начинаем разметку комментариев. Записей: {total}")
    print(HELP_LINE)
    input("\n  Нажмите Enter для старта...")

    all_labels: list[dict] = df_existing.to_dict("records")
    new_labels: list[dict] = []
    stats = {"p": 0, "n": 0, "g": 0, "s": 0}
    df_raw = df_raw.reset_index(drop=True)

    for idx, row in df_raw.iterrows():
        _show_comment(idx, total, row)

        while True:
            key = input("  Ваш выбор: ").strip().lower()
            if key in KEY_MAP:
                label = KEY_MAP[key]
                record = {
                    "channel_username": row["channel_username"],
                    "post_id": row.get("post_id", None),
                    "comment_id": row["comment_id"],
                    "sentiment": label,
                }
                new_labels.append(record)
                all_labels.append(record)
                stats[key] += 1
                print(
                    f"  → {SENTIMENT_LABEL_NAMES[label].upper()}  "
                    f"(позитивных: {stats['p']}, нейтральных: {stats['n']}, "
                    f"негативных: {stats['g']}, пропущено: {stats['s']})"
                )
                break
            if key == "s":
                stats["s"] += 1
                print("  → Комментарий пропущен.")
                break
            if key == "q":
                print("\n  Прерывание по команде пользователя. Сохраняем...")
                _save_labels(all_labels, COMMENTS_LABELED_CSV)
                _print_summary(stats)
                return
            print("  Неверный ввод. Используйте: p / n / g / s / q")

        if len(new_labels) % AUTOSAVE_EVERY == 0 and new_labels:
            _save_labels(all_labels, COMMENTS_LABELED_CSV)
            print(f"\n  [Автосохранение: {len(all_labels)} записей]\n")

    _save_labels(all_labels, COMMENTS_LABELED_CSV)
    _print_summary(stats)


def _print_summary(stats: dict) -> None:
    total_labeled = stats["p"] + stats["n"] + stats["g"]
    print(f"\n{'═' * 50}")
    print("  ИТОГИ РАЗМЕТКИ КОММЕНТАРИЕВ")
    print(f"{'═' * 50}")
    print(f"  Размечено:    {total_labeled}")
    print(f"  Позитивных:   {stats['p']}")
    print(f"  Нейтральных:  {stats['n']}")
    print(f"  Негативных:   {stats['g']}")
    print(f"  Пропущено:    {stats['s']}")
    print(f"{'═' * 50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Интерактивная ручная разметка тональности комментариев."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="Разметить не более N комментариев.",
    )
    parser.add_argument(
        "--ch",
        nargs="*",
        metavar="USERNAME",
        help="Разметить только указанные каналы (username без @).",
    )
    args = parser.parse_args()
    run_labeling(n_limit=args.n, channel_filter=args.ch)


if __name__ == "__main__":
    main()
