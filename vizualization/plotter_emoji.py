"""
vizualization/plotter_emoji.py
==============================
Визуализация эмодзи-реакций под постами.

Графики:
    1. plot_emoji_frequency_by_channel()
    Горизонтальный bar chart топ-N эмодзи для каждого выбранного канала.
    Один subplot на канал, эмодзи окрашены по категории (positive/negative/neutral).

    2. plot_emoji_monthly_heatmap()
    Тепловая карта: ось X — месяцы, ось Y — эмодзи,
    значение — нормированная частота упоминания.
    Отдельный график для каждого канала.

    3. plot_emoji_sentiment_distribution()
    Stacked bar: доля позитивных / негативных / нейтральных эмодзи
    по каналам в сравнении.

    4. plot_esi_timeline()
    Динамика Emoji Sentiment Index (ESI) по неделям с маркерами событий.
    Отдельные ряды для state и public каналов.

    5. save_all_emoji_plots()
    Генерирует все четыре типа графиков и сохраняет в FIGURES_DIR.

Запуск:
    python -m vizualization.plotter_emoji
    python -m vizualization.plotter_emoji --show
    python -m vizualization.plotter_emoji --channel shot_shot
"""


import argparse
import logging
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import (
    CHANNELS,
    EMOJI_SENTIMENT,
    EMOJI_LIKE_DISLIKE,
    EMOJI_BY_SENTIMENT,
    EVENTS,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_STATE,
    COLOR_PUBLIC,
    FIGURE_DPI,
    FIGURES_DIR,
    CLEANED_CSV,
    RESULTS_DIR,
)

from analysis.emoji_analyzer import analyze_corpus


log = logging.getLogger(__name__)


def _register_emoji_fonts() -> list[str]:
    """Регистрирует системные emoji-шрифты и возвращает доступные имена."""
    env_font = os.getenv("EMOJI_FONT_PATH")
    if env_font:
        candidates = [Path(env_font)]
    else:
        candidates = []

    candidates += [
        Path("/System/Library/Fonts/Apple Color Emoji.ttc"),
        Path("/System/Library/Fonts/Supplemental/Apple Color Emoji.ttc"),
        Path("/Library/Fonts/Apple Color Emoji.ttc"),
        Path.home() / "Library/Fonts/Apple Color Emoji.ttc",
        Path("/Library/Fonts/NotoEmoji-Regular.ttf"),
        Path.home() / "Library/Fonts/NotoEmoji-Regular.ttf",
        Path("/Library/Fonts/Symbola.ttf"),
        Path.home() / "Library/Fonts/Symbola.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
        Path("/usr/share/fonts/truetype/ancient-scripts/Symbola.ttf"),
        Path("C:/Windows/Fonts/seguiemj.ttf"),
    ]

    registered: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            registered.append(name)
        except Exception as exc:
            log.debug("Не удалось зарегистрировать шрифт %s: %s", path, exc)

    if not registered:
        try:
            font_manager._load_fontmanager(try_read_cache=False)
        except Exception as exc:
            log.debug("Не удалось пересобрать кэш шрифтов: %s", exc)

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in [
        "Apple Color Emoji",
        "Segoe UI Emoji",
        "Noto Emoji",
        "Noto Color Emoji",
        "Symbola",
    ]:
        if name in available and name not in registered:
            registered.append(name)

    return registered

emoji_fonts = _register_emoji_fonts()
if not emoji_fonts:
    log.warning(
        "Emoji-шрифты не найдены. Установите, например, 'Noto Emoji' или 'Symbola'."
    )
    emoji_fonts = ["DejaVu Sans"]
else:
    log.info("Emoji-шрифты: %s", ", ".join(emoji_fonts))

# ── Глобальный стиль ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       emoji_fonts + ["DejaVu Sans", "sans-serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


_CAT_COLOR = {
    "positive": COLOR_POSITIVE,
    "negative": COLOR_NEGATIVE,
    "neutral":  COLOR_NEUTRAL,
    "unknown":  "#bdc3c7",
}


def _save(fig: plt.Figure, name: str, show: bool = False) -> Path:
    """Сохраняет фигуру в FIGURES_DIR."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    log.info("Сохранён → %s", path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def _emoji_dict_for(channel_username: str) -> dict[str, str]:
    """Возвращает словарь эмодзи→категория с учётом типа реакций канала."""
    for ch in CHANNELS:
        if ch["username"] == channel_username:
            rtype = ch.get("reaction_type", "none")
            if rtype == "like_dislike":
                return EMOJI_LIKE_DISLIKE
            elif rtype == "emoji_full":
                return EMOJI_SENTIMENT
            return {}
    return EMOJI_SENTIMENT


def _load_corpus(source_path: Path | None = None) -> pd.DataFrame:
    """
    Загружает корпус с реакциями.
    Приоритет: results/emoji_corpus.csv → results/predictions.csv → raw CSV.
    """
    candidates = [
        RESULTS_DIR / "emoji_corpus.csv",
        RESULTS_DIR / "predictions.csv",
        CLEANED_CSV,
    ]
    if source_path:
        candidates.insert(0, source_path)
 
    for path in candidates:
        if path and path.exists():
            log.info("Загружаем данные: %s", path)
            df = pd.read_csv(path, encoding="utf-8")
            df["reactions_top"]   = df.get("reactions_top",   pd.Series(dtype=str)).fillna("")
            df["reactions_total"] = pd.to_numeric(
                df.get("reactions_total", 0), errors="coerce"
            ).fillna(0).astype(int)
            df["date"] = pd.to_datetime(df.get("date", pd.Series(dtype=str)),
                                        utc=True, errors="coerce")
            return df
 
    raise FileNotFoundError(
        "Файл данных не найден. Запустите: python -m data.collector"
    )
 
 
def _count_emojis_for_df(
    df: pd.DataFrame,
    channel_username: str,
) -> Counter:
    """
    Подсчитывает суммарную частоту каждого эмодзи
    в строках reactions_top для заданного канала.
    """
    ch_df = df[df["channel_username"] == channel_username]
    emoji_counter: Counter = Counter()
    for row_emojis in ch_df["reactions_top"]:
        if isinstance(row_emojis, str) and row_emojis.strip():
            for em in row_emojis.strip().split():
                emoji_counter[em] += 1
    return emoji_counter
 
 
def _add_event_markers(ax: plt.Axes, ymin: float = 0, ymax: float = 1) -> None:
    """Добавляет вертикальные линии ключевых событий."""
    for ev in EVENTS:
        x = pd.Timestamp(ev["date"], tz="UTC")
        ax.axvline(x=x, color="#e67e22", linestyle="--",
                   linewidth=0.9, alpha=0.7)
        ax.text(x, ymax * 0.96, ev["short"],
                rotation=90, fontsize=6.5, color="#e67e22",
                alpha=0.85, va="top", ha="right")
 
 
def plot_emoji_frequency_by_channel(
    df: pd.DataFrame,
    channels: list[str] | None = None,
    top_n: int = 15,
    show: bool = False,
) -> Path:
    """
    Горизонтальный bar chart топ-N эмодзи для каждого канала.
    Столбцы окрашены по категории: зелёный=positive, красный=negative, серый=neutral.
 
    Args:
        df:       корпус с колонками channel_username, reactions_top
        channels: список username каналов (по умолчанию — все с реакциями)
        top_n:    сколько эмодзи показывать на канал
    """
    if channels is None:
        channels = [
            ch["username"] for ch in CHANNELS
            if ch.get("reaction_type") != "none"
        ]
 
    # Собираем счётчики для каждого канала
    channel_counters: dict[str, Counter] = {}
    for username in channels:
        cnt = _count_emojis_for_df(df, username)
        if cnt:
            channel_counters[username] = cnt
 
    if not channel_counters:
        log.warning("Нет данных о реакциях для выбранных каналов.")
        return FIGURES_DIR / "no_data.txt"
 
    n_channels = len(channel_counters)
    cols = min(3, n_channels)
    rows = (n_channels + cols - 1) // cols
 
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 6, rows * 5),
        squeeze=False,
    )
    axes_flat = axes.flatten()
 
    for ax_idx, (username, counter) in enumerate(channel_counters.items()):
        ax = axes_flat[ax_idx]
 
        # Метаданные канала
        ch_meta  = next((c for c in CHANNELS if c["username"] == username), {})
        label    = ch_meta.get("label", username)
        rtype    = ch_meta.get("reaction_type", "emoji_full")
        emoji_dict = _emoji_dict_for(username)
 
        # Топ-N эмодзи
        top_items = counter.most_common(top_n)
        if not top_items:
            ax.set_visible(False)
            continue
 
        emojis = [item[0] for item in top_items]
        counts = [item[1] for item in top_items]
        colors = [_CAT_COLOR.get(emoji_dict.get(e, "unknown"), "#bdc3c7")
                  for e in emojis]
 
        # Рисуем горизонтальные бары
        y_pos = range(len(emojis))
        bars = ax.barh(
            list(y_pos), counts,
            color=colors, alpha=0.85, height=0.65,
            edgecolor="white", linewidth=0.5,
        )
 
        # Подписи значений
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8.5,
            )
 
        # Эмодзи как метки оси Y
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(emojis, fontsize=14)   # крупнее для эмодзи
        ax.invert_yaxis()
 
        orient_label = "гос." if ch_meta.get("orientation") == "state" else "общ."
        rtype_label  = "👍👎 only" if rtype == "like_dislike" else "full emoji"
        ax.set_title(
            f"{label}\n({orient_label} | {rtype_label})",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Число постов с данным эмодзи в топе")
        ax.set_xlim(0, max(counts) * 1.18)

        legend_handles = [
            Patch(color=COLOR_POSITIVE, label="позитивные"),
            Patch(color=COLOR_NEGATIVE, label="негативные"),
            Patch(color=COLOR_NEUTRAL,  label="нейтральные"),
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=7.5, framealpha=0.7)

    for i in range(len(channel_counters), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Топ эмодзи-реакций по каналам выборки\n"
        "(март–декабрь 2025 г., дело Долиной)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return _save(fig, "fig_2_11_emoji_frequency_by_channel", show)


def plot_emoji_monthly_heatmap(
    df: pd.DataFrame,
    channels: list[str] | None = None,
    top_n: int = 12,
    show: bool = False,
) -> list[Path]:
    """
    Для каждого канала строит отдельную тепловую карту:
      ось X — месяц (апр 2025 … дек 2025),
      ось Y — топ-N эмодзи канала,
      значение — доля постов в данном месяце, где эмодзи присутствует в топ-3.
 
    Нормировка по столбцу (месяцу): позволяет сравнить
    относительную популярность эмодзи внутри каждого месяца.
    """
    if channels is None:
        channels = [
            ch["username"] for ch in CHANNELS
            if ch.get("reaction_type") != "none"
        ]
 
    saved_paths = []
 
    for username in channels:
        ch_meta  = next((c for c in CHANNELS if c["username"] == username), {})
        label    = ch_meta.get("label", username)
        rtype    = ch_meta.get("reaction_type", "emoji_full")
 
        ch_df = df[
            (df["channel_username"] == username) &
            (df["reactions_top"].str.strip() != "")
        ].copy()
 
        if ch_df.empty:
            log.info("@%s: нет данных о реакциях, пропускаем.", username)
            continue
 
        # Определяем топ-N эмодзи канала
        counter = _count_emojis_for_df(df, username)
        top_emojis = [em for em, _ in counter.most_common(top_n)]
        if not top_emojis:
            continue
 
        # Группируем по месяцу
        ch_df = ch_df.dropna(subset=["date"]).copy()
        dates = ch_df["date"]
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert(None)
        ch_df["month"] = dates.dt.to_period("M").astype(str)
        months = sorted(ch_df["month"].unique())
 
        # Строим матрицу: строки — эмодзи, столбцы — месяцы
        # Значение = число постов где эмодзи встречается в топ-3 / всего постов в месяц
        matrix = pd.DataFrame(index=top_emojis, columns=months, dtype=float).fillna(0.0)
 
        for month, month_df in ch_df.groupby("month"):
            n_month = len(month_df)
            if n_month == 0:
                continue
            for _, row in month_df.iterrows():
                emojis_in_post = str(row.get("reactions_top", "")).strip().split()
                for em in emojis_in_post:
                    if em in top_emojis:
                        matrix.at[em, month] += 1
            matrix[month] = matrix[month] / n_month * 100  # → %
 
        # Метки эмодзи без категорий
        row_labels = [em for em in top_emojis]
 
        # ── Рисуем ────────────────────────────────────────────────────────────
        fig_h = max(4, len(top_emojis) * 0.55 + 2)
        fig_w = max(8, len(months) * 1.1 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
 
        # Кастомный колормап: белый → цвет ориентации канала
        orient = ch_meta.get("orientation", "public")
        base_color = COLOR_STATE if orient == "state" else COLOR_PUBLIC
        cmap = LinearSegmentedColormap.from_list(
            "custom", ["#ffffff", base_color], N=256
        )
 
        sns.heatmap(
            matrix.astype(float),
            ax=ax,
            cmap=cmap,
            vmin=0, vmax=matrix.values.max() if matrix.values.max() > 0 else 1,
            linewidths=0.4,
            linecolor="#ecf0f1",
            annot=True,
            fmt=".0f",
            annot_kws={"fontsize": 8},
            cbar_kws={"label": "% постов в месяце", "shrink": 0.8},
            yticklabels=row_labels,
        )
 
        orient_label = "государственный" if orient == "state" else "общественный"
        rtype_label  = "только 👍/👎" if rtype == "like_dislike" else "полный набор реакций"
        ax.set_title(
            f"Динамика эмодзи-реакций по месяцам\n"
            f"{label}  ({orient_label} | {rtype_label})\n"
            f"март–декабрь 2025 г.",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Эмодзи")
 
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=12)

        fig.subplots_adjust(left=0.18)
        fig.tight_layout()
        fname = f"fig_emoji_monthly_{username}"
        path  = _save(fig, fname, show)
        saved_paths.append(path)
 
    return saved_paths
 

def plot_emoji_sentiment_distribution(
    df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Stacked horizontal bar chart: доля позитивных / нейтральных / негативных
    эмодзи для каждого канала.
 
    Каналы отсортированы по доле позитивных реакций.
    РИА Новости (like_dislike) выделен отдельным паттерном.
    """
    rows = []
    for ch in CHANNELS:
        username = ch["username"]
        rtype    = ch.get("reaction_type", "none")
        if rtype == "none":
            continue
 
        counter    = _count_emojis_for_df(df, username)
        emoji_dict = _emoji_dict_for(username)
 
        n_pos = n_neg = n_neu = 0
        for em, cnt in counter.items():
            cat = emoji_dict.get(em, "unknown")
            if cat == "positive":
                n_pos += cnt
            elif cat == "negative":
                n_neg += cnt
            elif cat == "neutral":
                n_neu += cnt
 
        total = n_pos + n_neg + n_neu
        if total == 0:
            continue
 
        rows.append({
            "label":       ch["label"],
            "orientation": ch["orientation"],
            "rtype":       rtype,
            "pct_pos":     n_pos / total * 100,
            "pct_neu":     n_neu / total * 100,
            "pct_neg":     n_neg / total * 100,
            "total_emojis": total,
        })

    if not rows:
        log.warning("Нет данных для plot_emoji_sentiment_distribution.")
        return FIGURES_DIR / "no_data.txt"

    df_plot = pd.DataFrame(rows).sort_values("pct_pos", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5, len(df_plot) * 0.75 + 1.5)))
 
    y = range(len(df_plot))
    height = 0.55

    # Три сегмента stacked bar
    bars_pos = ax.barh(list(y), df_plot["pct_pos"], height=height, color=COLOR_POSITIVE, alpha=0.85, label="Позитивные")
    bars_neu = ax.barh(list(y), df_plot["pct_neu"], height=height,
                       left=df_plot["pct_pos"], color=COLOR_NEUTRAL, alpha=0.75, label="Нейтральные")
    bars_neg = ax.barh(list(y), df_plot["pct_neg"], height=height,
                       left=df_plot["pct_pos"] + df_plot["pct_neu"],
                       color=COLOR_NEGATIVE, alpha=0.85, label="Негативные")

    # Подписи значений (только если сегмент достаточно широкий)
    for bars, col in [(bars_pos, "pct_pos"), (bars_neg, "pct_neg")]:
        for bar, val in zip(bars, df_plot[col]):
            if val > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )

    # Метки каналов
    labels_y = []
    for _, row in df_plot.iterrows():
        orient_mark = "🏛" if row["orientation"] == "state" else "📢"
        rtype_mark  = " [👍👎]" if row["rtype"] == "like_dislike" else ""
        labels_y.append(f"{orient_mark} {row['label']}{rtype_mark}")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels_y, fontsize=9.5)
    ax.set_xlabel("Доля эмодзи по категории, %")
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.axvline(50, color="#bdc3c7", linewidth=0.8, linestyle=":")
    ax.set_title("Распределение эмодзи-реакций по категориям тональности\n"
        "(по каналам выборки, март–декабрь 2025 г.)\n"
        "🏛 — государственные каналы  |  📢 — общественные каналы",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    return _save(fig, "fig_2_12_emoji_sentiment_distribution", show)


def plot_esi_timeline(
    df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Динамика Emoji Sentiment Index (ESI) по неделям.
    Два ряда: state-каналы (синий) и public-каналы (фиолетовый).
    Вертикальные маркеры ключевых событий дела Долиной.
 
    ESI = (n_pos_emojis − n_neg_emojis) / n_classified  ∈ [−1, +1]
    """

    if "esi" not in df.columns:
        df = analyze_corpus(df)
 
    df2 = df.dropna(subset=["date", "esi"]).copy()
    if df2.empty:
        log.warning("Нет данных ESI для построения временного ряда.")
        return FIGURES_DIR / "no_esi_data.txt"
 
    df2["date"] = pd.to_datetime(df2["date"], utc=True, errors="coerce")
 
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
 
    # ── Верхний: ESI по ориентации ────────────────────────────────────────────
    for orient, color, lbl in [
        ("state",  COLOR_STATE,  "Государственные каналы"),
        ("public", COLOR_PUBLIC, "Общественные каналы"),
    ]:
        sub = df2[df2["orientation"] == orient]
        if sub.empty:
            continue
        weekly = (
            sub.set_index("date")
            .resample("W")["esi"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        weekly.columns = ["period", "mean_esi", "sem_esi"]
        weekly = weekly.dropna(subset=["mean_esi"])
 
        ax_top.plot(
            weekly["period"], weekly["mean_esi"],
            color=color, linewidth=2.0, label=lbl,
        )
        # Доверительный интервал ±1 SEM
        ax_top.fill_between(
            weekly["period"],
            weekly["mean_esi"] - weekly["sem_esi"],
            weekly["mean_esi"] + weekly["sem_esi"],
            color=color, alpha=0.15,
        )

    ax_top.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="-")

    y_abs = max(df2["esi"].abs().max() * 1.2, 0.2)
    _add_event_markers(ax_top, ymin=-y_abs, ymax=y_abs)
    ax_top.set_ylim(-y_abs, y_abs)
    ax_top.set_ylabel("ESI (Emoji Sentiment Index)")
    ax_top.set_title("Динамика Emoji Sentiment Index (ESI) по неделям\n"
        "(государственные vs общественные каналы, март–декабрь 2025 г.)",
        fontsize=11, fontweight="bold",
    )
    ax_top.legend(loc="upper left", fontsize=9)

    # Зоны
    ax_top.fill_between(
        [df2["date"].min(), df2["date"].max()],
        [0, 0], [y_abs, y_abs],
        color=COLOR_POSITIVE, alpha=0.05,
    )
    ax_top.fill_between(
        [df2["date"].min(), df2["date"].max()],
        [-y_abs, -y_abs], [0, 0],
        color=COLOR_NEGATIVE, alpha=0.05,
    )

    # ── Нижний: число постов с реакциями ─────────────────────────────────────
    weekly_all = (
        df2.set_index("date")
        .resample("W")
        .size()
        .reset_index(name="n_posts")
    )
    ax_bot.bar(
        weekly_all["date"], weekly_all["n_posts"],
        width=5, color="#3498db", alpha=0.6,
    )
    _add_event_markers(ax_bot, ymax=weekly_all["n_posts"].max() * 1.1)
    ax_bot.set_ylabel("Постов с ESI")
    ax_bot.set_xlabel("Дата")

    # Формат оси X
    for ax in [ax_top, ax_bot]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, "fig_2_13_esi_timeline", show)


def save_all_emoji_plots(
    channels: list[str] | None = None,
    show: bool = False,
    source_path: Path | None = None,
) -> list[Path]:
    """
    Генерирует все четыре типа графиков и сохраняет в FIGURES_DIR.
 
    Args:
        channels:    список username каналов (None = все с реакциями)
        show:        открывать в окне (нужен GUI)
        source_path: путь к CSV с данными (None = автопоиск)
 
    Returns:
        Список путей к сохранённым PNG.
    """
    df = _load_corpus(source_path)
    saved = []

    log.info("Генерация графика 1: топ эмодзи по каналам...")
    try:
        p = plot_emoji_frequency_by_channel(df, channels=channels, show=show)
        saved.append(p)
    except Exception as e:
        log.error("График 1 не создан: %s", e)

    log.info("Генерация графиков 2: тепловые карты по месяцам...")
    try:
        paths = plot_emoji_monthly_heatmap(df, channels=channels, show=show)
        saved.extend(paths)
    except Exception as e:
        log.error("Графики 2 не созданы: %s", e)

    log.info("Генерация графика 3: распределение категорий по каналам...")
    try:
        p = plot_emoji_sentiment_distribution(df, show=show)
        saved.append(p)
    except Exception as e:
        log.error("График 3 не создан: %s", e)

    log.info("Генерация графика 4: динамика ESI...")
    try:
        p = plot_esi_timeline(df, show=show)
        saved.append(p)
    except Exception as e:
        log.error("График 4 не создан: %s", e)

    log.info(
        "\nВсего сохранено графиков: %d\nПапка: %s",
        len(saved), FIGURES_DIR,
    )
    return saved


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Визуализация эмодзи-реакций Telegram-каналов."
    )
    parser.add_argument(
        "--channel", nargs="*", metavar="USERNAME",
        help="Каналы для анализа (без @). По умолчанию — все с реакциями.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Открывать графики в окне (требует GUI).",
    )
    parser.add_argument(
        "--top", type=int, default=15,
        help="Число топ-эмодзи на канал (по умолчанию 15).",
    )
    parser.add_argument(
        "--only", choices=["freq", "heatmap", "dist", "esi"],
        default=None,
        help="Построить только один тип графика.",
    )
    args = parser.parse_args()
    if args.show:
        matplotlib.use("TkAgg")

    df = _load_corpus()

    if args.only == "freq":
        plot_emoji_frequency_by_channel(df, channels=args.channel, top_n=args.top, show=args.show)
    elif args.only == "heatmap":
        plot_emoji_monthly_heatmap(df, channels=args.channel, show=args.show)
    elif args.only == "dist":
        plot_emoji_sentiment_distribution(df, show=args.show)
    elif args.only == "esi":
        plot_esi_timeline(df, show=args.show)
    else:
        save_all_emoji_plots(channels=args.channel, show=args.show)
    print(f"\nГотово. Графики сохранены в: {FIGURES_DIR}")


if __name__ == "__main__":
    main()