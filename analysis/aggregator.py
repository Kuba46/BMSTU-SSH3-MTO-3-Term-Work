"""
analysis/aggregator.py
======================
Агрегация тональности на уровне всего корпуса.

Вычисляет взвешенный индекс тональности S(t) по формуле:
    S(t) = Σ sentiment(m) · reach(m) / Σ reach(m)

где reach(m) = views(m) + k·reactions(m) — взвешенный охват поста,
k — коэффициент усиления реакций (по умолчанию k=5,
так как реакция — более активное действие, чем просмотр).

Функции:
    weighted_sentiment_index()  — S(t) для группы постов
    aggregate_by_period()       — S(t) по временны́м периодам
    aggregate_by_channel()      — S(t) и структура тональности по каналам
    aggregate_by_orientation()  — сравнение гос. vs общ.
    top_influential_posts()     — наиболее охватные посты каждой тональности

Запуск:
    python -m analysis.aggregator          # полная агрегация
    python -m analysis.aggregator --table  # только сводная таблица
"""

import logging

import pandas as pd

from config.settings import (
    PREDICTIONS_CSV,
    RESULTS_DIR,
    SENTIMENT_LABEL_NAMES,
)

log = logging.getLogger(__name__)

# Коэффициент усиления реакции относительно просмотра
REACTION_WEIGHT = 5


# ── Загрузка ──────────────────────────────────────────────────────────────────
def load_predictions() -> pd.DataFrame:
    """Загружает predictions.csv с предсказанной тональностью."""
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Предсказания не найдены: {PREDICTIONS_CSV}\n"
            "Запустите: python -m models.predict"
        )
    df = pd.read_csv(PREDICTIONS_CSV, encoding="utf-8")
    df["date"]           = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["sentiment_pred"] = pd.to_numeric(df["sentiment_pred"], errors="coerce")
    df["views"]          = pd.to_numeric(df.get("views", 0), errors="coerce").fillna(0)
    df["reactions_total"] = pd.to_numeric(
        df.get("reactions_total", 0), errors="coerce"
    ).fillna(0)
    df = df.dropna(subset=["date", "sentiment_pred"])
    return df


# ── Взвешенный индекс тональности ────────────────────────────────────────────
def weighted_sentiment_index(
    df: pd.DataFrame,
    reaction_weight: int = REACTION_WEIGHT,
) -> float:
    """
    Вычисляет взвешенный индекс тональности S для группы постов.

    S = Σ sentiment(m) · reach(m) / Σ reach(m)

    где reach(m) = views(m) + reaction_weight · reactions(m)

    Args:
        df:               DataFrame с колонками sentiment_pred, views,
                          reactions_total
        reaction_weight:  коэффициент усиления реакций

    Returns:
        float ∈ [-1, +1]; NaN если группа пуста или все охваты = 0
    """
    if df.empty:
        return float("nan")

    reach = df["views"] + reaction_weight * df["reactions_total"]

    total_reach = reach.sum()
    if total_reach == 0:
        # Равновзвешенный индекс, если нет данных об охвате
        return float(df["sentiment_pred"].mean())

    return float((df["sentiment_pred"] * reach).sum() / total_reach)


# ── Агрегация по периодам ─────────────────────────────────────────────────────
def aggregate_by_period(
    df: pd.DataFrame,
    freq: str = "W",
    reaction_weight: int = REACTION_WEIGHT,
) -> pd.DataFrame:
    """
    Взвешенный индекс тональности S(t) по временны́м периодам.

    Returns:
        DataFrame: period, n_posts, weighted_si, unweighted_si,
                   total_reach, pct_positive, pct_neutral, pct_negative
    """
    rows = []
    for period, grp in df.set_index("date").resample(freq):
        n = len(grp)
        if n == 0:
            continue

        reach = grp["views"] + reaction_weight * grp["reactions_total"]
        n_pos = (grp["sentiment_pred"] == 1).sum()
        n_neu = (grp["sentiment_pred"] == 0).sum()
        n_neg = (grp["sentiment_pred"] == -1).sum()

        rows.append({
            "period":        period,
            "n_posts":       n,
            "weighted_si":   weighted_sentiment_index(grp, reaction_weight),
            "unweighted_si": float(grp["sentiment_pred"].mean()),
            "total_reach":   int(reach.sum()),
            "pct_positive":  n_pos / n * 100,
            "pct_neutral":   n_neu / n * 100,
            "pct_negative":  n_neg / n * 100,
        })

    result = pd.DataFrame(rows)
    log.info("aggregate_by_period: %d периодов (freq=%s)", len(result), freq)
    return result


# ── Агрегация по каналам ──────────────────────────────────────────────────────
def aggregate_by_channel(
    df: pd.DataFrame,
    reaction_weight: int = REACTION_WEIGHT,
) -> pd.DataFrame:
    """
    Индекс тональности и структура распределения по каналам.

    Returns:
        DataFrame: channel_label, orientation, n_posts,
                   weighted_si, pct_positive, pct_neutral, pct_negative,
                   total_reach, mean_views, mean_reactions
    """
    rows = []
    for (label, orient), grp in df.groupby(
        ["channel_label", "orientation"], sort=False
    ):
        n = len(grp)
        n_pos = (grp["sentiment_pred"] == 1).sum()
        n_neu = (grp["sentiment_pred"] == 0).sum()
        n_neg = (grp["sentiment_pred"] == -1).sum()
        reach = grp["views"] + reaction_weight * grp["reactions_total"]

        rows.append({
            "channel_label":  label,
            "orientation":    orient,
            "n_posts":        n,
            "weighted_si":    weighted_sentiment_index(grp, reaction_weight),
            "pct_positive":   n_pos / n * 100,
            "pct_neutral":    n_neu / n * 100,
            "pct_negative":   n_neg / n * 100,
            "total_reach":    int(reach.sum()),
            "mean_views":     float(grp["views"].mean()),
            "mean_reactions": float(grp["reactions_total"].mean()),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("n_posts", ascending=False)
        .reset_index(drop=True)
    )
    log.info("aggregate_by_channel:\n%s", result.to_string(index=False))
    return result


# ── Агрегация по ориентации ───────────────────────────────────────────────────
def aggregate_by_orientation(
    df: pd.DataFrame,
    reaction_weight: int = REACTION_WEIGHT,
) -> pd.DataFrame:
    """
    Сравнение государственных и общественных каналов по ключевым метрикам.

    Returns:
        DataFrame: orientation, n_channels, n_posts, weighted_si,
                   pct_positive, pct_neutral, pct_negative, total_reach
    """
    rows = []
    for orient, grp in df.groupby("orientation"):
        n = len(grp)
        n_channels = grp["channel_label"].nunique()
        n_pos = (grp["sentiment_pred"] == 1).sum()
        n_neu = (grp["sentiment_pred"] == 0).sum()
        n_neg = (grp["sentiment_pred"] == -1).sum()
        reach = grp["views"] + reaction_weight * grp["reactions_total"]

        rows.append({
            "orientation":   orient,
            "n_channels":    n_channels,
            "n_posts":       n,
            "weighted_si":   weighted_sentiment_index(grp, reaction_weight),
            "pct_positive":  n_pos / n * 100,
            "pct_neutral":   n_neu / n * 100,
            "pct_negative":  n_neg / n * 100,
            "total_reach":   int(reach.sum()),
        })

    result = pd.DataFrame(rows)
    log.info("aggregate_by_orientation:\n%s", result.to_string(index=False))
    return result


# ── Наиболее охватные посты ───────────────────────────────────────────────────
def top_influential_posts(
    df: pd.DataFrame,
    n: int = 5,
    reaction_weight: int = REACTION_WEIGHT,
) -> dict[str, pd.DataFrame]:
    """
    Топ-N постов по охвату для каждой тональности.
    Используется для качественного анализа: какие именно тексты
    сформировали доминирующую тональность дискуссии.

    Returns:
        {'positive': df, 'neutral': df, 'negative': df}
        каждый df содержит: channel_label, date, text, views,
        reactions_total, reach, sentiment_pred
    """
    df2 = df.copy()
    df2["reach"] = df2["views"] + reaction_weight * df2["reactions_total"]

    result = {}
    for label_val, label_name in SENTIMENT_LABEL_NAMES.items():
        sub = df2[df2["sentiment_pred"] == label_val].copy()
        sub = sub.nlargest(n, "reach")
        cols = [c for c in ["channel_label", "orientation", "date",
                             "text", "views", "reactions_total", "reach"]
                if c in sub.columns]
        result[label_name] = sub[cols].reset_index(drop=True)

    log.info("top_influential_posts: топ-%d постов для каждого класса", n)
    return result


# ── Итоговая сводная таблица ──────────────────────────────────────────────────
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Единая сводная таблица по всему корпусу.

    Returns:
        DataFrame: метрика, значение
    """
    n = len(df)
    n_pos = (df["sentiment_pred"] == 1).sum()
    n_neu = (df["sentiment_pred"] == 0).sum()
    n_neg = (df["sentiment_pred"] == -1).sum()
    reach = df["views"] + REACTION_WEIGHT * df["reactions_total"]
    wsi   = weighted_sentiment_index(df)

    rows = [
        ("Всего постов",               n),
        ("Позитивных постов",           n_pos),
        ("Нейтральных постов",          n_neu),
        ("Негативных постов",           n_neg),
        ("Доля позитивных, %",          round(n_pos / n * 100, 1)),
        ("Доля нейтральных, %",         round(n_neu / n * 100, 1)),
        ("Доля негативных, %",          round(n_neg / n * 100, 1)),
        ("Взвешенный индекс S",         round(wsi, 4)),
        ("Суммарный охват (reach)",      int(reach.sum())),
        ("Суммарные просмотры",          int(df["views"].sum())),
        ("Суммарные реакции",            int(df["reactions_total"].sum())),
        ("Каналов в выборке",            df["channel_label"].nunique()),
        ("Период: начало",              str(df["date"].min().date())),
        ("Период: конец",               str(df["date"].max().date())),
    ]
    return pd.DataFrame(rows, columns=["Метрика", "Значение"])


# ── Основной пайплайн ─────────────────────────────────────────────────────────
def run_pipeline(table_only: bool = False) -> dict[str, pd.DataFrame]:
    """
    Запускает полную агрегацию и сохраняет результаты.
    """
    df = load_predictions()

    results = {
        "by_period_weekly":  aggregate_by_period(df, freq="W"),
        "by_period_monthly": aggregate_by_period(df, freq="ME"),
        "by_channel":        aggregate_by_channel(df),
        "by_orientation":    aggregate_by_orientation(df),
        "summary":           summary_table(df),
    }

    if not table_only:
        for name, frame in results.items():
            path = RESULTS_DIR / f"agg_{name}.csv"
            frame.to_csv(path, index=False, encoding="utf-8")
            log.info("Сохранено → %s", path)

    # Сводная таблица в лог
    log.info("\n%s\nСВОДНАЯ ТАБЛИЦА КОРПУСА\n%s",
             "═" * 50, "═" * 50)
    log.info("\n%s", results["summary"].to_string(index=False))

    # Наиболее охватные посты
    top = top_influential_posts(df, n=3)
    for cls_name, top_df in top.items():
        log.info("\nТоп-3 охватных постов [%s]:", cls_name)
        for _, row in top_df.iterrows():
            text_preview = str(row.get("text", ""))[:120].replace("\n", " ")
            log.info(
                "  [%s] reach=%s  %s...",
                row.get("channel_label", "?"),
                f"{int(row.get('reach', 0)):,}",
                text_preview,
            )

    return results


def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Агрегация тональности корпуса."
    )
    parser.add_argument(
        "--table", action="store_true",
        help="Только сводная таблица в терминал, без сохранения CSV.",
    )
    args = parser.parse_args()
    run_pipeline(table_only=args.table)


if __name__ == "__main__":
    main()