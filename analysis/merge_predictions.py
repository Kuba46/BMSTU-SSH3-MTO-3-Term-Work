"""
analysis/merge_predictions.py
=============================
Объединение предсказаний тональности постов и комментариев в общий файл.

Запуск:
    python -m analysis.merge_predictions
    python -m analysis.merge_predictions --posts results/predictions.csv \
        --comments results/comments_predictions.csv --output results/predictions_all.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from config.settings import (
    PREDICTIONS_CSV,
    COMMENTS_PREDICTIONS_CSV,
    PREDICTIONS_ALL_CSV,
)

log = logging.getLogger(__name__)


def _load_with_type(path: Path, item_type: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    if "item_type" not in df.columns:
        df["item_type"] = item_type
    return df


def run_pipeline(
    posts_path: Path = PREDICTIONS_CSV,
    comments_path: Path = COMMENTS_PREDICTIONS_CSV,
    output_path: Path = PREDICTIONS_ALL_CSV,
) -> pd.DataFrame:
    df_posts = _load_with_type(posts_path, "post")
    df_comments = _load_with_type(comments_path, "comment")
    combined = pd.concat([df_posts, df_comments], ignore_index=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Общий файл предсказаний сохранён → %s  (%d строк)", output_path, len(combined))
    return combined


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Объединение предсказаний постов и комментариев.")
    parser.add_argument("--posts", type=str, default=None,
                        help="Путь к predictions.csv (посты).")
    parser.add_argument("--comments", type=str, default=None,
                        help="Путь к comments_predictions.csv.")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для сохранения объединённого файла.")
    args = parser.parse_args()

    posts_path = PREDICTIONS_CSV if args.posts is None else Path(args.posts)
    comments_path = COMMENTS_PREDICTIONS_CSV if args.comments is None else Path(args.comments)
    output_path = PREDICTIONS_ALL_CSV if args.output is None else Path(args.output)

    run_pipeline(posts_path=posts_path, comments_path=comments_path, output_path=output_path)


if __name__ == "__main__":
    main()
