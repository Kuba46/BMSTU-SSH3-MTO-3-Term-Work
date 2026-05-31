"""
analysis/emoji_analyzer.py
==========================
Анализ эмодзи-реакций под постами Telegram-каналов.

Что делает:
    - Парсит колонку reactions_top (строка эмодзи через пробел)
    - Классифицирует каждое эмодзи по словарю EMOJI_SENTIMENT из settings
    - Вычисляет emoji_sentiment_index (ESI) для каждого поста:
        ESI = (n_positive - n_negative) / n_total ∈ [-1, +1]
    - Сравнивает ESI с тональностью текста (sentiment_pred) 
    для выявления расхождений «текст позитивный — реакции негативные»

Функции:
    parse_emoji_reactions()     — разбор строки реакций в словарь
    emoji_sentiment_score()     — ESI для одного поста
    analyze_corpus()            — ESI для всего датасета
    compare_text_vs_emoji()     — сравнение тональности текста и реакций
    channel_emoji_profile()     — профиль реакций по каналу
    timeline_esi()              — динамика ESI по неделям с маркерами событий

Запуск:
    python -m analysis.emoji_analyzer
    python -m analysis.emoji_analyzer --channel rian_ru
"""


import argparse
import logging
from collections import Counter

import numpy as np
import pandas as pd

from config.settings import (
    CHANNELS,
    EMOJI_SENTIMENT,
    EMOJI_LIKE_DISLIKE,
    PREDICTIONS_CSV,
    CLEANED_CSV,
    RESULTS_DIR,
)

log = logging.getLogger(__name__)


def _get_reaction_type(channel_username: str) -> str:
    """Возвращает тип реакций для канала из конфига."""
    for ch in CHANNELS:
        if ch["username"] == channel_username:
            return ch.get("reaction_type", "none")
    return "none"


def _emoji_dict_for_channel(channel_username: str) -> dict[str, str]:
    """
    Возвращает подходящий словарь классификации эмодзи для канала.
    РИА Новости (like_dislike) → только 👍/👎.
    Остальные → полный EMOJI_SENTIMENT.
    """
    rtype = _get_reaction_type(channel_username)
    if rtype == "like_dislike":
        return EMOJI_LIKE_DISLIKE
    elif rtype == "emoji_full":
        return EMOJI_SENTIMENT
    return {}


def parse_emoji_reactions(
    reactions_top: str,
    emoji_dict: dict[str, str] | None = None,
) -> dict[str, int]:
    """
    Разбирает строку reactions_top (формат: «👍 🔥 ❤️» через пробел)
    в словарь {sentiment: count}.

    Каждое эмодзи считается за 1 реакцию (collector сохраняет
    только топ-3 уникальных эмодзи, не их количество).
    Для количественного анализа используется reactions_total.

    Args:
        reactions_top: строка с эмодзи через пробел
        emoji_dict:    словарь {emoji: sentiment}; None → EMOJI_SENTIMENT
    Returns:
        {"positive": N, "negative": N, "neutral": N, "unknown": N}
    """
    if emoji_dict is None:
        emoji_dict = EMOJI_SENTIMENT

    counts = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
    if not isinstance(reactions_top, str) or not reactions_top.strip():
        return counts

    for emoji in reactions_top.strip().split():
        sentiment = emoji_dict.get(emoji, "unknown")
        counts[sentiment] += 1
    return counts


def emoji_sentiment_score(
    reactions_top: str,
    reactions_total: int,
    emoji_dict: dict[str, str] | None = None,
) -> float | None:
    """
    Вычисляет Emoji Sentiment Index (ESI) для одного поста.

    ESI = (n_positive - n_negative) / n_classified  ∈ [-1, +1]

    где n_classified = n_positive + n_negative (нейтральные не учитываются,
    так как они не несут аффективной нагрузки).

    Возвращает None если:
      - реакций нет (reactions_total == 0)
      - все эмодзи неизвестны или нейтральны
    """
    if not reactions_total or not isinstance(reactions_top, str):
        return None

    parsed = parse_emoji_reactions(reactions_top, emoji_dict)
    n_pos = parsed["positive"]
    n_neg = parsed["negative"]
    n_classified = n_pos + n_neg

    if n_classified == 0:
        return None
    return (n_pos - n_neg) / n_classified


def analyze_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет к датафрейму колонки анализа эмодзи-реакций.

    Добавляемые колонки:
        reaction_type       — тип реакций канала (emoji_full / like_dislike / none)
        emoji_positive      — число позитивных эмодзи в топ-3
        emoji_negative      — число негативных эмодзи в топ-3
        emoji_neutral       — число нейтральных эмодзи в топ-3
        emoji_unknown       — число нераспознанных эмодзи
        emoji_dominant      — доминирующая категория (positive/negative/neutral/none)
        esi                 — Emoji Sentiment Index ∈ [-1, +1] или NaN
        top_emoji_list      — список топ-3 эмодзи как Python list (для агрегации)
    """
    df = df.copy()

    # Добавляем тип реакций для каждой строки
    channel_rtype = {
        ch["username"]: ch.get("reaction_type", "none")
        for ch in CHANNELS
    }
    df["reaction_type"] = df["channel_username"].map(channel_rtype).fillna("none")

    # Парсим эмодзи с учётом типа реакций канала
    parsed_rows = []
    for _, row in df.iterrows():
        ch_username = row.get("channel_username", "")
        emoji_dict  = _emoji_dict_for_channel(ch_username)
        rtype       = row.get("reaction_type", "none")

        if rtype == "none" or not emoji_dict:
            parsed_rows.append({
                "emoji_positive": 0, "emoji_negative": 0,
                "emoji_neutral":  0, "emoji_unknown":  0,
                "emoji_dominant": "none", "esi": np.nan,
                "top_emoji_list": [],
            })
            continue

        reactions_top   = str(row.get("reactions_top", ""))
        reactions_total = int(row.get("reactions_total", 0))
        parsed = parse_emoji_reactions(reactions_top, emoji_dict)

        # Доминирующая категория
        pos, neg, neu = parsed["positive"], parsed["negative"], parsed["neutral"]
        if pos == 0 and neg == 0 and neu == 0:
            dominant = "none"
        else:
            dominant = max(
                [("positive", pos), ("negative", neg), ("neutral", neu)],
                key=lambda x: x[1]
            )[0]

        # ESI
        esi = emoji_sentiment_score(reactions_top, reactions_total, emoji_dict)

        # Список эмодзи для частотного анализа
        emoji_list = reactions_top.strip().split() if reactions_top.strip() else []
        parsed_rows.append({
            "emoji_positive": pos,
            "emoji_negative": neg,
            "emoji_neutral":  neu,
            "emoji_unknown":  parsed["unknown"],
            "emoji_dominant": dominant,
            "esi":            esi if esi is not None else np.nan,
            "top_emoji_list": emoji_list,
        })

    df_parsed = pd.DataFrame(parsed_rows, index=df.index)
    df = pd.concat([df, df_parsed], axis=1)

    log.info(
        "Анализ эмодзи завершён: %d постов | "
        "с ESI: %d | без реакций: %d",
        len(df),
        df["esi"].notna().sum(),
        (df["reaction_type"] == "none").sum(),
    )
    return df


def compare_text_vs_emoji(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сравнивает тональность текста (sentiment_pred) с ESI эмодзи-реакций.

    Выявляет посты с расхождением:
      — текст позитивный, но реакции негативные (аудитория не согласна)
      — текст негативный, но реакции позитивные (аудитория одобряет)

    Колонка divergence:
       "aligned"    — текст и эмодзи согласуются
       "divergent"  — значимое расхождение (|ESI - text_score| > 0.5)
       "na"         — нет ESI для сравнения

    Returns:
        DataFrame с добавленными колонками text_score, divergence, delta
    """
    df = df.copy()

    if "sentiment_pred" not in df.columns:
        log.warning("Колонка sentiment_pred отсутствует — "
                    "запустите models/predict.py перед сравнением.")
        return df

    if "esi" not in df.columns:
        df = analyze_corpus(df)

    # Нормализуем sentiment_pred [-1, 0, 1] → [-1, 0, 1] (уже нормализован)
    df["text_score"] = df["sentiment_pred"].astype(float)

    def _divergence(row) -> str:
        esi = row.get("esi")
        txt = row.get("text_score")
        if pd.isna(esi) or pd.isna(txt):
            return "na"
        delta = abs(esi - txt)
        return "divergent" if delta > 0.5 else "aligned"

    df["delta"]      = (df["esi"] - df["text_score"]).abs()
    df["divergence"] = df.apply(_divergence, axis=1)

    n_div = (df["divergence"] == "divergent").sum()
    n_ali = (df["divergence"] == "aligned").sum()
    log.info(
        "Сравнение текст vs эмодзи: согласованных=%d | расходящихся=%d",
        n_ali, n_div,
    )

    # Топ-10 наиболее расходящихся постов
    divergent = (
        df[df["divergence"] == "divergent"]
        .nlargest(10, "delta")[
            ["channel_label", "date", "text", "sentiment_pred",
             "esi", "delta", "reactions_top"]
        ]
    )
    if not divergent.empty:
        log.info("Топ расходящихся постов (текст vs реакции):\n%s",
                 divergent.assign(
                     text=lambda x: x["text"].str[:80]
                 ).to_string(index=False))

    return df


def channel_emoji_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Частотный профиль эмодзи-реакций для каждого канала.

    Показывает: какие эмодзи чаще всего встречаются, их долю
    в каждой категории (positive/negative/neutral) и средний ESI.

    Returns:
        DataFrame: channel_label, reaction_type, mean_esi,
                   pct_positive, pct_negative, pct_neutral,
                   top_emojis (5 самых частых)
    """
    if "esi" not in df.columns:
        df = analyze_corpus(df)

    rows = []
    for (label, rtype), grp in df.groupby(["channel_label", "reaction_type"]):
        if rtype == "none":
            rows.append({
                "channel_label":  label,
                "reaction_type":  rtype,
                "mean_esi":       np.nan,
                "pct_positive":   np.nan,
                "pct_negative":   np.nan,
                "pct_neutral":    np.nan,
                "top_5_emojis":   "—",
                "n_posts":        len(grp),
            })
            continue

        # Частоты эмодзи
        all_emojis = []
        for emoji_list in grp["top_emoji_list"]:
            if isinstance(emoji_list, list):
                all_emojis.extend(emoji_list)

        counter = Counter(all_emojis)
        top5    = " ".join(e for e, _ in counter.most_common(5))

        # Доли по категориям
        total_classified = (
            grp["emoji_positive"].sum() +
            grp["emoji_negative"].sum() +
            grp["emoji_neutral"].sum()
        )
        pct_pos = grp["emoji_positive"].sum() / total_classified * 100 if total_classified else np.nan
        pct_neg = grp["emoji_negative"].sum() / total_classified * 100 if total_classified else np.nan
        pct_neu = grp["emoji_neutral"].sum()  / total_classified * 100 if total_classified else np.nan

        rows.append({
            "channel_label":  label,
            "reaction_type":  rtype,
            "mean_esi":       round(grp["esi"].mean(), 4) if grp["esi"].notna().any() else np.nan,
            "pct_positive":   round(pct_pos, 1),
            "pct_negative":   round(pct_neg, 1),
            "pct_neutral":    round(pct_neu, 1),
            "top_5_emojis":   top5 or "—",
            "n_posts":        len(grp),
        })

    result = pd.DataFrame(rows).sort_values("mean_esi", ascending=False, na_position="last")
    log.info("Профиль реакций по каналам:\n%s", result.to_string(index=False))
    return result


def timeline_esi(
    df: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """
    Динамика среднего ESI по неделям/месяцам.

    Позволяет сравнить:
      - динамику ESI реакций vs динамику тональности текста (sentiment_index)
      - реакцию аудитории на события дела через эмодзи

    Returns:
        DataFrame: period, mean_esi, median_esi, n_posts_with_esi,
                   pct_positive_dominant, pct_negative_dominant
    """
    if "esi" not in df.columns:
        df = analyze_corpus(df)

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], utc=True, errors="coerce")
    df2 = df2.dropna(subset=["date"])

    rows = []
    for period, grp in df2.set_index("date").resample(freq):
        grp_esi = grp[grp["esi"].notna()]
        n_esi   = len(grp_esi)
        if n_esi == 0:
            continue

        dom = grp_esi["emoji_dominant"]
        rows.append({
            "period":                period,
            "mean_esi":              round(grp_esi["esi"].mean(), 4),
            "median_esi":            round(grp_esi["esi"].median(), 4),
            "n_posts_with_esi":      n_esi,
            "pct_positive_dominant": (dom == "positive").sum() / n_esi * 100,
            "pct_negative_dominant": (dom == "negative").sum() / n_esi * 100,
        })

    return pd.DataFrame(rows)


def run_pipeline() -> dict[str, pd.DataFrame]:
    """
    Полный анализ эмодзи-реакций.
    Загружает predictions.csv (или raw), обогащает ESI, сохраняет результаты.
    """
    source = PREDICTIONS_CSV if PREDICTIONS_CSV.exists() else CLEANED_CSV
    if not source.exists():
        raise FileNotFoundError(
            f"Файл данных не найден: {source}\n"
            "Запустите сначала: python -m data.collector, затем python -m data.cleaner"
        )

    df = pd.read_csv(source, encoding="utf-8")
    df["reactions_top"]   = df.get("reactions_top",   pd.Series(dtype=str)).fillna("")
    df["reactions_total"] = pd.to_numeric(df.get("reactions_total", 0), errors="coerce").fillna(0)

    # Основной анализ
    df = analyze_corpus(df)

    # Сравнение текст vs эмодзи
    if "sentiment_pred" in df.columns:
        df = compare_text_vs_emoji(df)

    # Профиль по каналам
    profile = channel_emoji_profile(df)

    # Динамика ESI
    esi_weekly  = timeline_esi(df, freq="W")
    esi_monthly = timeline_esi(df, freq="ME")

    # Сохранение
    results = {
        "emoji_corpus":      df,
        "emoji_profile":     profile,
        "esi_weekly":        esi_weekly,
        "esi_monthly":       esi_monthly,
    }

    for name, frame in results.items():
        if name == "emoji_corpus":
            continue   # не сохраняем полный корпус повторно
        path = RESULTS_DIR / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        log.info("Сохранено → %s", path)

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Анализ эмодзи-реакций.")
    parser.add_argument("--channel", default=None, metavar="USERNAME",
                        help="Анализировать только один канал.")
    args = parser.parse_args()

    results = run_pipeline()

    if args.channel:
        ch_profile = results["emoji_profile"]
        ch_profile = ch_profile[
            ch_profile["channel_label"].str.contains(args.channel, case=False)
        ]
        print(f"\nПрофиль канала {args.channel}:")
        print(ch_profile.to_string(index=False))


if __name__ == "__main__":
    main()