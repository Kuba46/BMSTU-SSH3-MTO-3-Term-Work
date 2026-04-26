"""
analysis/event_analysis.py
==========================
Ивент-анализ: наложение временно́й шкалы событий дела Долиной
на динамику тональности и публикационной активности.

Что строит:
    activity_timeline()     — число публикаций по дням/неделям/месяцам
    sentiment_timeline()    — динамика тональности по периодам
    channel_comparison()    — сравнение каналов по тональности во времени
    reaction_timeline()     — динамика реакций и просмотров
    event_impact()          — «окна» вокруг ключевых событий: δ тональности
    orientation_divergence()— расхождение гос. vs общ. каналов по времени

Все функции возвращают DataFrame, готовый к передаче в viz/plotter.py.

Запуск:
    python -m analysis.event_analysis        # полный анализ
    python -m analysis.event_analysis --summary  # только сводка в терминал
"""

import logging

import numpy as np
import pandas as pd

from config.settings import (
    EVENTS,
    PREDICTIONS_CSV,
    RESULTS_DIR,
)

log = logging.getLogger(__name__)


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    """
    Загружает predictions.csv и приводит типы.
    Обязательные колонки: date, sentiment_pred, channel_label,
    orientation, views, reactions_total.
    """
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Предсказания не найдены: {PREDICTIONS_CSV}\n"
            "Запустите: python -m models.predict"
        )
    df = pd.read_csv(PREDICTIONS_CSV, encoding="utf-8")
    df["date"]          = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["sentiment_pred"] = pd.to_numeric(df["sentiment_pred"], errors="coerce")
    df["views"]          = pd.to_numeric(df.get("views", 0), errors="coerce").fillna(0)
    df["reactions_total"] = pd.to_numeric(
        df.get("reactions_total", 0), errors="coerce"
    ).fillna(0)
    df = df.dropna(subset=["date", "sentiment_pred"])
    log.info("Загружено %d записей с предсказаниями", len(df))
    return df


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _events_df() -> pd.DataFrame:
    """Конвертирует список EVENTS из settings в DataFrame."""
    rows = []
    for ev in EVENTS:
        rows.append({
            "date":        pd.Timestamp(ev["date"], tz="UTC"),
            "label":       ev["label"],
            "short":       ev["short"],
            "description": ev["description"],
        })
    return pd.DataFrame(rows)


def _resample(
    df: pd.DataFrame,
    date_col: str = "date",
    freq: str = "W",
) -> pd.core.resample.DatetimeIndexResampler:
    """Устанавливает date_col как индекс и ресэмплирует."""
    return df.set_index(date_col).resample(freq)


# ── Динамика активности ───────────────────────────────────────────────────────

def activity_timeline(
    df: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Число публикаций за каждый период (неделя по умолчанию).

    Returns:
        DataFrame: period (datetime), n_posts, total_views,
                   total_reactions, total_forwards
    """
    df2 = df.copy()
    agg_cols = {
        "post_id":        ("n_posts",        "count"),
        "views":          ("total_views",     "sum"),
        "reactions_total":("total_reactions", "sum"),
    }
    if "forwards" in df2.columns:
        agg_cols["forwards"] = ("total_forwards", "sum")

    result = (
        df2.set_index("date")
        .resample(freq)
        .agg(**{new: (old, fn) for old, (new, fn) in agg_cols.items()})
        .reset_index()
        .rename(columns={"date": "period"})
    )
    log.info("activity_timeline: %d периодов (freq=%s)", len(result), freq)
    return result


# ── Динамика тональности ──────────────────────────────────────────────────────

def sentiment_timeline(
    df: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Доля позитивных / нейтральных / негативных публикаций по периодам.
    Также вычисляет взвешенный индекс тональности:
        sentiment_index = (n_pos - n_neg) / n_total  ∈ [-1, +1]

    Returns:
        DataFrame: period, n_posts, pct_positive, pct_neutral, pct_negative,
                   sentiment_index
    """
    df2 = df.dropna(subset=["sentiment_pred"]).copy()

    rows = []
    for period, grp in df2.set_index("date").resample(freq):
        n = len(grp)
        if n == 0:
            continue
        n_pos = (grp["sentiment_pred"] == 1).sum()
        n_neu = (grp["sentiment_pred"] == 0).sum()
        n_neg = (grp["sentiment_pred"] == -1).sum()
        rows.append({
            "period":          period,
            "n_posts":         n,
            "pct_positive":    n_pos / n * 100,
            "pct_neutral":     n_neu / n * 100,
            "pct_negative":    n_neg / n * 100,
            "sentiment_index": (n_pos - n_neg) / n,
        })

    result = pd.DataFrame(rows)
    log.info("sentiment_timeline: %d периодов (freq=%s)", len(result), freq)
    return result


# ── Сравнение каналов ─────────────────────────────────────────────────────────

def channel_comparison(
    df: pd.DataFrame,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Индекс тональности по каждому каналу по месяцам.

    Returns:
        DataFrame в wide-формате: period как индекс,
        каждый канал — отдельная колонка с sentiment_index
    """
    df2 = df.dropna(subset=["sentiment_pred", "channel_label"]).copy()

    rows = []
    for (period, channel), grp in (
        df2.set_index("date")
        .groupby([pd.Grouper(freq=freq), "channel_label"])
    ):
        n = len(grp)
        if n == 0:
            continue
        n_pos = (grp["sentiment_pred"] == 1).sum()
        n_neg = (grp["sentiment_pred"] == -1).sum()
        rows.append({
            "period":          period,
            "channel_label":   channel,
            "sentiment_index": (n_pos - n_neg) / n,
            "n_posts":         n,
        })

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df

    # Pivot в wide-формат
    wide_df = long_df.pivot_table(
        index="period",
        columns="channel_label",
        values="sentiment_index",
        aggfunc="mean",
    )
    wide_df.index = wide_df.index.strftime("%Y-%m")
    log.info("channel_comparison: %d периодов × %d каналов",
             len(wide_df), len(wide_df.columns))
    return wide_df


# ── Расхождение гос. vs общ. ──────────────────────────────────────────────────

def orientation_divergence(
    df: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Разница индексов тональности между государственными
    и общественными каналами по периодам.

        divergence = sentiment_index(state) - sentiment_index(public)

    Положительное значение → государственные каналы позитивнее.
    Отрицательное → общественные позитивнее.

    Returns:
        DataFrame: period, state_index, public_index, divergence
    """
    df2 = df.dropna(subset=["sentiment_pred", "orientation"]).copy()

    rows = []
    for period, grp in df2.set_index("date").resample(freq):
        if len(grp) == 0:
            continue
        result_row = {"period": period}
        for orient in ["state", "public"]:
            sub = grp[grp["orientation"] == orient]
            n   = len(sub)
            if n == 0:
                result_row[f"{orient}_index"] = np.nan
            else:
                n_pos = (sub["sentiment_pred"] == 1).sum()
                n_neg = (sub["sentiment_pred"] == -1).sum()
                result_row[f"{orient}_index"] = (n_pos - n_neg) / n
        rows.append(result_row)

    result = pd.DataFrame(rows)
    result["divergence"] = result["state_index"] - result["public_index"]
    log.info("orientation_divergence: %d периодов (freq=%s)", len(result), freq)
    return result


# ── Реакции и просмотры ───────────────────────────────────────────────────────

def reaction_timeline(
    df: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Суммарные реакции и просмотры по периодам.
    Дополнительно вычисляет engagement_rate:
        engagement_rate = reactions / views  (при views > 0)

    Returns:
        DataFrame: period, total_views, total_reactions, engagement_rate
    """
    df2 = df.copy()
    result = (
        df2.set_index("date")
        .resample(freq)
        .agg(
            total_views=("views", "sum"),
            total_reactions=("reactions_total", "sum"),
        )
        .reset_index()
        .rename(columns={"date": "period"})
    )
    result["engagement_rate"] = np.where(
        result["total_views"] > 0,
        result["total_reactions"] / result["total_views"],
        0.0,
    )
    return result


# ── Оконный анализ вокруг событий ────────────────────────────────────────────

def event_impact(
    df: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """
    Для каждого ключевого события из EVENTS вычисляет:
      - средний sentiment_index за window_days ДО события
      - средний sentiment_index за window_days ПОСЛЕ события
      - delta = after - before (изменение тональности)
      - n_before, n_after (число постов в окне)

    Это позволяет количественно оценить влияние каждого
    судебного решения на тональность дискуссии.

    Returns:
        DataFrame: event_label, event_date, si_before, si_after,
                   delta, n_before, n_after
    """
    df2 = df.dropna(subset=["sentiment_pred"]).copy()

    def _window_si(center_dt: pd.Timestamp, before: bool) -> tuple[float, int]:
        if before:
            mask = (df2["date"] >= center_dt - pd.Timedelta(days=window_days)) & \
                   (df2["date"] <  center_dt)
        else:
            mask = (df2["date"] >  center_dt) & \
                   (df2["date"] <= center_dt + pd.Timedelta(days=window_days))
        sub = df2[mask]
        n   = len(sub)
        if n == 0:
            return np.nan, 0
        n_pos = (sub["sentiment_pred"] == 1).sum()
        n_neg = (sub["sentiment_pred"] == -1).sum()
        return float((n_pos - n_neg) / n), n

    rows = []
    for ev in EVENTS:
        ev_ts = pd.Timestamp(ev["date"], tz="UTC")
        si_before, n_before = _window_si(ev_ts, before=True)
        si_after,  n_after  = _window_si(ev_ts, before=False)

        delta = (si_after - si_before) if not (
            np.isnan(si_before) or np.isnan(si_after)
        ) else np.nan

        rows.append({
            "event_label": ev["short"],
            "event_date":  ev["date"],
            "si_before":   round(si_before, 4) if not np.isnan(si_before) else None,
            "si_after":    round(si_after,  4) if not np.isnan(si_after)  else None,
            "delta":       round(delta, 4)     if not np.isnan(delta)     else None,
            "n_before":    n_before,
            "n_after":     n_after,
            "window_days": window_days,
        })

    result = pd.DataFrame(rows)
    log.info("event_impact (window=%d дней):\n%s",
             window_days, result.to_string(index=False))
    return result


# ── Основной пайплайн ─────────────────────────────────────────────────────────

def run_pipeline(summary_only: bool = False) -> dict[str, pd.DataFrame]:
    """
    Запускает весь ивент-анализ и сохраняет результаты в RESULTS_DIR.

    Returns:
        Словарь {name: DataFrame} для передачи в viz/plotter.py
    """
    df = load_predictions()

    results = {
        "activity_weekly":       activity_timeline(df, freq="W"),
        "activity_monthly":      activity_timeline(df, freq="ME"),
        "sentiment_weekly":      sentiment_timeline(df, freq="W"),
        "sentiment_monthly":     sentiment_timeline(df, freq="ME"),
        "channel_comparison":    channel_comparison(df, freq="ME"),
        "orientation_divergence": orientation_divergence(df, freq="W"),
        "reaction_timeline":     reaction_timeline(df, freq="W"),
        "event_impact":          event_impact(df, window_days=7),
    }

    if not summary_only:
        for name, frame in results.items():
            path = RESULTS_DIR / f"{name}.csv"
            frame.to_csv(path, index=True, encoding="utf-8")
            log.info("Сохранено → %s", path)

    # Краткая сводка в лог
    log.info("\n%s", "═" * 60)
    log.info("СВОДКА ИВЕНТ-АНАЛИЗА")
    log.info("═" * 60)

    si = results["sentiment_monthly"]
    if not si.empty:
        peak_neg = si.loc[si["pct_negative"].idxmax()]
        peak_act = results["activity_monthly"].loc[
            results["activity_monthly"]["n_posts"].idxmax()
        ]
        log.info("Пик негативной тональности: %s  (%.1f%%)",
                 peak_neg["period"], peak_neg["pct_negative"])
        log.info("Пик активности: %s  (%d постов)",
                 peak_act["period"], peak_act["n_posts"])

    log.info("\nВлияние событий на тональность:")
    log.info(results["event_impact"].to_string(index=False))

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Ивент-анализ корпуса.")
    parser.add_argument(
        "--summary", action="store_true",
        help="Только сводка в терминал, без сохранения CSV.",
    )
    args = parser.parse_args()
    run_pipeline(summary_only=args.summary)


if __name__ == "__main__":
    main()