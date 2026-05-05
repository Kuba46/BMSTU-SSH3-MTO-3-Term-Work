"""
vizualization/plotter_posts.py
==============================
Все визуализации по постам.

Функции:
    plot_activity_timeline()     — динамика публикационной активности
    plot_sentiment_timeline()    — динамика тональности (stacked area)
    plot_sentiment_index()       — взвешенный индекс S(t) с маркерами событий
    plot_channel_heatmap()       — тепловая карта тональности по каналам × месяцы
    plot_orientation_divergence()— расхождение гос. vs общ. каналов
    plot_cluster_tsne()          — t-SNE проекция кластеров
    plot_top_terms()             — топ-слова для каждого кластера (bar chart)
    plot_confusion_matrix()      — матрица ошибок классификатора
    plot_model_comparison()      — сравнение LogReg vs SVM
    plot_event_impact()          — δ тональности вокруг каждого события
    save_all()                   — сохраняет все графики в FIGURES_DIR

Запуск:
    python -m vizualization.plotter_posts         # генерирует все графики по постам
    python -m vizualization.plotter_posts --show  # открывает графики в окне (требует GUI)
"""

import argparse
import logging
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import (
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_PUBLIC,
    COLOR_STATE,
    EVENTS,
    FIGURE_DPI,
    FIGURE_SIZE_SQUARE,
    FIGURE_SIZE_WIDE,
    FIGURES_DIR,
    SENTIMENT_LABEL_NAMES,
    RESULTS_DIR,
    CLUSTERS_CSV,
    METRICS_JSON,
)

log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        FIGURE_DPI,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

sns.set_theme(style="whitegrid", context="notebook", palette="muted")


def _add_event_markers(
    ax: plt.Axes,
    events: list[dict] = EVENTS,
    ymin: float = 0.0,
    ymax: float = 1.0,
    color: str = "#e67e22",
    alpha: float = 0.7,
) -> None:
    for ev in events:
        x = pd.Timestamp(ev["date"], tz="UTC")
        ax.axvline(x=x, color=color, linestyle="--", linewidth=1.0, alpha=alpha)
        ax.text(
            x, ymax * 0.97,
            ev["short"],
            rotation=90,
            fontsize=7,
            color=color,
            alpha=0.85,
            va="top",
            ha="right",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
            },
        )


def _save(fig: plt.Figure, name: str, show: bool = False) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    bbox_inches = None if fig.get_constrained_layout() else "tight"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches=bbox_inches)
    log.info("График сохранён → %s", path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_activity_timeline(df: pd.DataFrame, show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    df2 = df.copy()
    df2["period"] = pd.to_datetime(df2["period"])

    ax.bar(df2["period"], df2["n_posts"],
           width=5, color="#3498db", alpha=0.75, label="Число постов")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Число публикаций")
    ax.set_title("Динамика публикационной активности в Telegram-каналах\n"
                 "(март–декабрь 2025 г., недельная агрегация)")

    y_max = df2["n_posts"].max() * 1.15
    _add_event_markers(ax, ymax=y_max)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()

    return _save(fig, "fig_2_1_activity_timeline", show)


def plot_sentiment_timeline(df: pd.DataFrame, show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    df2 = df.copy()
    df2["period"] = pd.to_datetime(df2["period"])
    df2 = df2.sort_values("period")

    ax.stackplot(
        df2["period"],
        df2["pct_positive"],
        df2["pct_neutral"],
        df2["pct_negative"],
        labels=["Позитивная", "Нейтральная", "Негативная"],
        colors=[COLOR_POSITIVE, COLOR_NEUTRAL, COLOR_NEGATIVE],
        alpha=0.75,
    )

    _add_event_markers(ax, ymax=100)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Доля публикаций, %")
    ax.set_ylim(0, 100)
    ax.set_title("Динамика тональности публикаций в Telegram-каналах\n"
                 "(март–декабрь 2025 г., недельная агрегация)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left")
    fig.tight_layout()

    return _save(fig, "fig_2_2_sentiment_timeline", show)


def plot_sentiment_index(df: pd.DataFrame, show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    df2 = df.copy()
    df2["period"] = pd.to_datetime(df2["period"])
    df2 = df2.sort_values("period")

    periods = df2["period"].values
    si = df2["weighted_si"].values

    ax.fill_between(periods, si, 0,
                    where=(si >= 0),
                    alpha=0.3, color=COLOR_POSITIVE, label="Позитивная зона")
    ax.fill_between(periods, si, 0,
                    where=(si < 0),
                    alpha=0.3, color=COLOR_NEGATIVE, label="Негативная зона")
    ax.plot(periods, si, color="#2c3e50", linewidth=1.5)
    ax.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="-")

    y_abs = max(abs(si).max() * 1.2, 0.1)
    _add_event_markers(ax, ymin=-y_abs, ymax=y_abs)
    ax.set_ylim(-y_abs, y_abs)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Взвешенный индекс тональности S(t)")
    ax.set_title("Взвешенный индекс тональности S(t)\n"
                 "(с учётом охвата публикаций; диапазон [-1, +1])")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left")
    fig.tight_layout()

    return _save(fig, "fig_2_3_sentiment_index", show)


def plot_channel_heatmap(df_wide: pd.DataFrame, show: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    show_annotations = df_wide.shape[0] * df_wide.shape[1] <= 60

    sns.heatmap(
        df_wide.T,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="#ecf0f1",
        annot=show_annotations,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Индекс тональности S"},
    )
    ax.set_title("Тепловая карта индекса тональности\n"
                 "по каналам и месяцам (март–декабрь 2025 г.)")
    ax.set_xlabel("Период (месяц)")
    ax.set_ylabel("Telegram-канал")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.subplots_adjust(bottom=0.18, left=0.22, right=0.96, top=0.92)

    return _save(fig, "fig_2_4_channel_heatmap", show)


def plot_orientation_divergence(df: pd.DataFrame, show: bool = False) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    df2 = df.copy()
    df2["period"] = pd.to_datetime(df2["period"])
    df2 = df2.sort_values("period")
    periods = df2["period"].values

    if df2["state_index"].notna().any():
        ax1.plot(periods, df2["state_index"], color=COLOR_STATE,
                 linewidth=1.8, label="Государственные каналы")
    if df2["public_index"].notna().any():
        ax1.plot(periods, df2["public_index"], color=COLOR_PUBLIC,
                 linewidth=1.8, linestyle="--", label="Общественные каналы")
    ax1.axhline(0, color="#bdc3c7", linewidth=0.8)
    ax1.set_ylabel("Индекс тональности S(t)")
    ax1.set_title("Сравнение тональности государственных\n"
                  "и общественных Telegram-каналов")
    ax1.legend()
    ax1.set_ylim(-1.0, 1.0)
    _add_event_markers(ax1, ymax=1.0)

    valid_div = df2.dropna(subset=["divergence"])
    if valid_div.empty:
        ax2.axis("off")
        ax2.text(
            0.5, 0.5,
            "Недостаточно данных для расчёта расхождения",
            ha="center", va="center",
            fontsize=10,
        )
    else:
        div_periods = pd.to_datetime(valid_div["period"]).values
        divergence = valid_div["divergence"].values
        divergence_series = pd.Series(divergence, index=div_periods)
        smooth = divergence_series.rolling(window=3, center=True, min_periods=1).mean().values

        ax2.bar(div_periods, divergence,
                width=5,
                color=[COLOR_STATE if d >= 0 else COLOR_PUBLIC for d in divergence],
                alpha=0.25)
        ax2.plot(div_periods, smooth, color="#2c3e50", linewidth=1.8, label="Сглаженный тренд")
        ax2.fill_between(div_periods, smooth, 0,
                         where=(smooth >= 0),
                         color=COLOR_STATE, alpha=0.12)
        ax2.fill_between(div_periods, smooth, 0,
                         where=(smooth < 0),
                         color=COLOR_PUBLIC, alpha=0.12)
        ax2.axhline(0, color="#bdc3c7", linewidth=0.8)
        ax2.set_xlabel("Дата")
        ax2.set_ylabel("Расхождение Δ S(t)")
        ax2.set_title("Расхождение (гос. − общ.): >0 → гос. позитивнее")
        _add_event_markers(ax2, alpha=0.35)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30, ha="right")
        ax2.legend(loc="upper left", fontsize=8, frameon=False)
        max_abs = float(np.nanmax(np.abs(divergence))) if len(divergence) else 0.1
        ax2.set_ylim(-max_abs * 1.25 if max_abs > 0 else -0.1, max_abs * 1.25 if max_abs > 0 else 0.1)
    fig.subplots_adjust(hspace=0.28, bottom=0.14, top=0.92)

    return _save(fig, "fig_2_5_orientation_divergence", show)


def plot_cluster_tsne(df: pd.DataFrame, show: bool = False) -> Path:
    if "tsne_x" not in df.columns or "tsne_y" not in df.columns:
        log.warning("t-SNE координаты отсутствуют — график пропущен.")
        return FIGURES_DIR / "fig_2_6_tsne_skipped.txt"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)

    clusters = sorted(df["kmeans_cluster"].unique())
    palette = sns.color_palette("tab10", len(clusters))

    for cl, color in zip(clusters, palette):
        sub = df[df["kmeans_cluster"] == cl]
        lbl = sub["kmeans_label"].iloc[0][:40] if len(sub) else str(cl)
        ax.scatter(
            sub["tsne_x"], sub["tsne_y"],
            s=15, alpha=0.55, color=color,
            label=f"К{cl}: {lbl}",
        )

    ax.set_title("t-SNE проекция тематических кластеров\n"
                 "(K-Means, TF-IDF пространство)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    return _save(fig, "fig_2_6_cluster_tsne", show)


def plot_top_terms(cluster_labels: dict[int, list[str]], km_model=None, show: bool = False) -> Path:
    n_clusters = len(cluster_labels)
    cols = min(3, n_clusters)
    rows = (n_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    palette = sns.color_palette("tab10", n_clusters)

    def _coerce_terms(raw_items: list) -> pd.DataFrame:
        if raw_items and isinstance(raw_items[0], dict):
            frame = pd.DataFrame(raw_items)
            if {"term", "weight"}.issubset(frame.columns):
                return frame[["term", "weight"]].copy()

        terms = list(raw_items)
        if not terms:
            return pd.DataFrame(columns=["term", "weight"])
        weights = np.linspace(len(terms), 1, len(terms), dtype=float)
        return pd.DataFrame({"term": terms, "weight": weights})

    for cl_id, words in sorted(cluster_labels.items()):
        ax = axes[cl_id]
        frame = _coerce_terms(words).head(10).sort_values("weight", ascending=True)
        y_pos = np.arange(len(frame))

        ax.barh(
            y_pos,
            frame["weight"],
            color=palette[cl_id],
            alpha=0.80,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(frame["term"].tolist(), fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Вес центроида")
        ax.set_title(f"Кластер {cl_id}", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.margins(x=0.05)

    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Топ-слова тематических кластеров K-Means",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    return _save(fig, "fig_2_7_top_terms", show)


def plot_confusion_matrix(
    cm: list[list[int]],
    model_name: str = "logreg",
    labels: list[str] | None = None,
    show: bool = False,
) -> Path:
    if labels is None:
        labels = ["Негативная", "Нейтральная", "Позитивная"]

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    row_sums = cm_arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.divide(
            cm_arr,
            row_sums,
            out=np.zeros_like(cm_arr, dtype=float),
            where=row_sums != 0,
        )

    annot = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            count = cm_arr[i, j]
            pct = cm_pct[i, j]
            annot[i, j] = f"{count}\n{pct:.0%}"

    sns.heatmap(
        cm_pct,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        cbar=True,
        cbar_kws={"label": "Доля по истинному классу"},
    )
    ax.set_xlabel("Предсказанная метка")
    ax.set_ylabel("Истинная метка")
    model_display = "LogisticRegression" if model_name == "logreg" else "SVM (LinearSVC)"
    ax.set_title(f"Матрица ошибок классификатора\n({model_display}, доли по строкам)")
    fig.tight_layout()

    return _save(fig, f"fig_2_8_confusion_matrix_{model_name}", show)


def plot_model_comparison(compare_df: pd.DataFrame, show: bool = False) -> Path:
    metrics_to_plot = [
        "Precision (macro)", "Recall (macro)", "F1 (macro)", "CV F1 (среднее)"
    ]
    metric_col = "Метрика" if "Метрика" in compare_df.columns else "metric"
    logreg_col = "LogReg" if "LogReg" in compare_df.columns else "logreg"
    svm_col = "SVM" if "SVM" in compare_df.columns else "svm"

    label_map = {
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
        "cv_f1_mean": "CV F1 (среднее)",
    }

    sub = compare_df.copy()
    sub[metric_col] = sub[metric_col].replace(label_map)
    sub = sub[sub[metric_col].isin(metrics_to_plot)].copy()

    x = np.arange(len(sub))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    lr_vals = pd.to_numeric(sub[logreg_col], errors="coerce").values
    svm_vals = pd.to_numeric(sub[svm_col], errors="coerce").values

    bars1 = ax.bar(x - width / 2, lr_vals, width, label="LogisticRegression",
                   color=COLOR_STATE, alpha=0.8)
    bars2 = ax.bar(x + width / 2, svm_vals, width, label="SVM (LinearSVC)",
                   color=COLOR_PUBLIC, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sub[metric_col].values, rotation=15, ha="right")
    ax.set_ylim(0, max(1.1, float(np.nanmax([lr_vals, svm_vals])) * 1.15))
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение качества классификаторов тональности\n"
                 "(LogisticRegression vs SVM, тестовая выборка)")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout()

    return _save(fig, "fig_2_9_model_comparison", show)


def plot_event_impact(df: pd.DataFrame, show: bool = False) -> Path:
    df2 = df.dropna(subset=["delta"]).copy()
    df2 = df2.sort_values("delta")

    fig, ax = plt.subplots(figsize=(10, 6))

    if df2.empty:
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Нет данных для оценки влияния событий",
            ha="center", va="center",
            fontsize=11,
        )
        return _save(fig, "fig_2_10_event_impact", show)

    if np.allclose(df2["delta"].values, 0):
        labels = df2["event_label"].astype(str)
        if "n_before" in df2.columns and "n_after" in df2.columns:
            labels = labels + " (n=" + df2["n_before"].astype(int).astype(str) + "/" + df2["n_after"].astype(int).astype(str) + ")"
        y_pos = np.arange(len(df2))
        ax.scatter(np.zeros(len(df2)), y_pos, s=60, color="#7f8c8d", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.axvline(0, color="#7f8c8d", linewidth=0.8)
        ax.set_xlim(-0.1, 0.1)
        ax.set_xlabel("Δ индекса тональности (после − до события)")
        ax.set_title("Изменение тональности дискуссии\n"
                     "в ±7 дней вокруг ключевых событий дела Долиной")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return _save(fig, "fig_2_10_event_impact", show)

    colors = [COLOR_POSITIVE if d >= 0 else COLOR_NEGATIVE
              for d in df2["delta"]]
    bars = ax.barh(df2["event_label"], df2["delta"],
                   color=colors, alpha=0.8, height=0.5)
    ax.axvline(0, color="#7f8c8d", linewidth=0.8)
    ax.bar_label(bars, fmt="%+.3f", fontsize=9, padding=3)
    ax.set_xlabel("Δ индекса тональности (после − до события)")
    ax.set_title("Изменение тональности дискуссии\n"
                 "в ±7 дней вокруг ключевых событий дела Долиной")
    ax.invert_yaxis()
    max_abs = float(np.nanmax(np.abs(df2["delta"])))
    ax.set_xlim(-max_abs * 1.15 if max_abs > 0 else -0.1, max_abs * 1.15 if max_abs > 0 else 0.1)
    fig.tight_layout()

    return _save(fig, "fig_2_10_event_impact", show)


def save_all(show: bool = False) -> list[Path]:
    saved = []

    def _try_load(name: str) -> pd.DataFrame | None:
        path = RESULTS_DIR / f"{name}.csv"
        if path.exists():
            return pd.read_csv(path)
        log.warning("Файл не найден, пропускаем: %s", path)
        return None

    df_act = _try_load("activity_weekly")
    if df_act is not None:
        saved.append(plot_activity_timeline(df_act, show))

    df_sent = _try_load("sentiment_weekly")
    if df_sent is not None:
        saved.append(plot_sentiment_timeline(df_sent, show))

    df_agg = _try_load("agg_by_period_weekly")
    if df_agg is not None:
        saved.append(plot_sentiment_index(df_agg, show))

    df_ch = _try_load("channel_comparison")
    if df_ch is not None:
        df_ch = df_ch.set_index(df_ch.columns[0])
        saved.append(plot_channel_heatmap(df_ch, show))

    df_div = _try_load("orientation_divergence")
    if df_div is not None:
        saved.append(plot_orientation_divergence(df_div, show))

    if CLUSTERS_CSV.exists():
        df_cl = pd.read_csv(CLUSTERS_CSV)
        saved.append(plot_cluster_tsne(df_cl, show))

        cluster_labels_path = RESULTS_DIR / "cluster_labels.json"
        if cluster_labels_path.exists():
            with open(cluster_labels_path, encoding="utf-8") as f:
                cl_labels = {int(k): v for k, v in json.load(f).items()}
            saved.append(plot_top_terms(cl_labels, show=show))

    if METRICS_JSON.exists():
        with open(METRICS_JSON, encoding="utf-8") as f:
            metrics_data = json.load(f)

        for model_name in ["logreg", "svm"]:
            cm = metrics_data.get(model_name, {}).get("confusion_matrix")
            if cm:
                saved.append(plot_confusion_matrix(cm, model_name, show=show))

        from models.svm_clf import comparison_table
        try:
            cmp_df = comparison_table()
            saved.append(plot_model_comparison(cmp_df, show))
        except Exception as exc:
            log.warning("Не удалось построить сравнение моделей: %s", exc)

    df_impact = _try_load("event_impact")
    if df_impact is not None:
        saved.append(plot_event_impact(df_impact, show))

    log.info("Всего графиков сохранено: %d", len(saved))
    return saved


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Генерация всех графиков по постам.")
    parser.add_argument(
        "--show", action="store_true",
        help="Открывать графики в окне (требует GUI-окружения).",
    )
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    paths = save_all(show=args.show)
    print(f"\nГотово. Сохранено {len(paths)} графиков в {FIGURES_DIR}")


if __name__ == "__main__":
    main()
