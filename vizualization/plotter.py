"""
viz/plotter.py
==============
Все визуализации проекта в одном модуле.

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
    python -m vizualization.plotter         # генерирует все графики
    python -m vizualization.plotter --show  # открывает графики в окне (требует GUI)
"""

import argparse
import logging
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # безголовый режим (нет GUI) — всегда работает на сервере
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

# ── Глобальные настройки стиля ────────────────────────────────────────────────
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

sns.set_palette("muted")


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _add_event_markers(
    ax: plt.Axes,
    events: list[dict] = EVENTS,
    ymin: float = 0.0,
    ymax: float = 1.0,
    color: str = "#e67e22",
    alpha: float = 0.7,
) -> None:
    """
    Добавляет вертикальные линии-маркеры ключевых событий на оси ax.
    Метки событий подписываются наклонно над линией.
    """
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
        )


def _save(fig: plt.Figure, name: str, show: bool = False) -> Path:
    """Сохраняет фигуру в FIGURES_DIR и опционально показывает."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    log.info("График сохранён → %s", path)
    if show:
        plt.show()
    plt.close(fig)
    return path


# ── 1. Динамика активности ────────────────────────────────────────────────────

def plot_activity_timeline(
    df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Число публикаций по неделям с маркерами событий.
    df должен содержать: period (datetime), n_posts.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    df2 = df.copy()
    df2["period"] = pd.to_datetime(df2["period"])

    ax.bar(df2["period"], df2["n_posts"],
           width=5, color="#3498db", alpha=0.75, label="Число постов")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Число публикаций")
    ax.set_title("Рис. 2.1. Динамика публикационной активности в Telegram-каналах\n"
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


# ── 2. Динамика тональности (stacked area) ───────────────────────────────────

def plot_sentiment_timeline(
    df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Стекированная диаграмма долей тональности по неделям.
    df: period, pct_positive, pct_neutral, pct_negative.
    """
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
    ax.set_title("Рис. 2.2. Динамика тональности публикаций в Telegram-каналах\n"
                 "(март–декабрь 2025 г., недельная агрегация)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left")
    fig.tight_layout()

    return _save(fig, "fig_2_2_sentiment_timeline", show)


# ── 3. Взвешенный индекс тональности S(t) ────────────────────────────────────

def plot_sentiment_index(
    df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Взвешенный индекс тональности S(t) ∈ [-1, +1] по неделям.
    df: period, weighted_si.
    Нулевая линия выделена; область выше/ниже закрашена.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    df2 = df.copy()
    df2["period"] = pd.to_datetime(df2["period"])
    df2 = df2.sort_values("period")

    periods = df2["period"].values
    si      = df2["weighted_si"].values

    # Позитивная область — зелёная, негативная — красная
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
    ax.set_title("Рис. 2.3. Взвешенный индекс тональности S(t)\n"
                 "(с учётом охвата публикаций; диапазон [-1, +1])")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper left")
    fig.tight_layout()

    return _save(fig, "fig_2_3_sentiment_index", show)


# ── 4. Тепловая карта: каналы × месяцы ───────────────────────────────────────

def plot_channel_heatmap(
    df_wide: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Тепловая карта sentiment_index.
    df_wide: строки — периоды (YYYY-MM), колонки — каналы.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.heatmap(
        df_wide.T,            # каналы по оси Y, месяцы по оси X
        ax=ax,
        cmap="RdYlGn",        # красный → жёлтый → зелёный
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="#ecf0f1",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Индекс тональности S"},
    )
    ax.set_title("Рис. 2.4. Тепловая карта индекса тональности\n"
                 "по каналам и месяцам (март–декабрь 2025 г.)")
    ax.set_xlabel("Период (месяц)")
    ax.set_ylabel("Telegram-канал")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    return _save(fig, "fig_2_4_channel_heatmap", show)


# ── 5. Расхождение гос. vs общ. ───────────────────────────────────────────────

def plot_orientation_divergence(
    df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Два ряда S(t) (state и public) + divergence на двух подграфиках.
    df: period, state_index, public_index, divergence.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    df2 = df.copy()
    df2["period"] = pd.to_datetime(df2["period"])
    df2 = df2.sort_values("period")
    periods = df2["period"].values

    # Верхний: два ряда тональности
    ax1.plot(periods, df2["state_index"],  color=COLOR_STATE,
             linewidth=1.8, label="Государственные каналы")
    ax1.plot(periods, df2["public_index"], color=COLOR_PUBLIC,
             linewidth=1.8, linestyle="--", label="Общественные каналы")
    ax1.axhline(0, color="#bdc3c7", linewidth=0.8)
    ax1.set_ylabel("Индекс тональности S(t)")
    ax1.set_title("Рис. 2.5. Сравнение тональности государственных\n"
                  "и общественных Telegram-каналов")
    ax1.legend()
    _add_event_markers(ax1, ymax=1.0)

    # Нижний: расхождение
    divergence = df2["divergence"].values
    ax2.bar(periods, divergence,
            width=5,
            color=[COLOR_STATE if d >= 0 else COLOR_PUBLIC for d in divergence],
            alpha=0.7)
    ax2.axhline(0, color="#bdc3c7", linewidth=0.8)
    ax2.set_xlabel("Дата")
    ax2.set_ylabel("Расхождение Δ S(t)")
    ax2.set_title("Расхождение (гос. − общ.): >0 → гос. позитивнее")
    _add_event_markers(ax2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    return _save(fig, "fig_2_5_orientation_divergence", show)


# ── 6. t-SNE проекция кластеров ───────────────────────────────────────────────

def plot_cluster_tsne(
    df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Точечный график t-SNE с раскраской по кластеру K-Means.
    df: tsne_x, tsne_y, kmeans_cluster, kmeans_label, channel_label.
    """
    if "tsne_x" not in df.columns or "tsne_y" not in df.columns:
        log.warning("t-SNE координаты отсутствуют — график пропущен.")
        return FIGURES_DIR / "fig_2_6_tsne_skipped.txt"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)

    clusters = sorted(df["kmeans_cluster"].unique())
    palette  = sns.color_palette("tab10", len(clusters))

    for cl, color in zip(clusters, palette):
        sub  = df[df["kmeans_cluster"] == cl]
        lbl  = sub["kmeans_label"].iloc[0][:40] if len(sub) else str(cl)
        ax.scatter(
            sub["tsne_x"], sub["tsne_y"],
            s=15, alpha=0.55, color=color,
            label=f"К{cl}: {lbl}",
        )

    ax.set_title("Рис. 2.6. t-SNE проекция тематических кластеров\n"
                 "(K-Means, TF-IDF пространство)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    return _save(fig, "fig_2_6_cluster_tsne", show)


# ── 7. Топ-слова по кластерам ─────────────────────────────────────────────────

def plot_top_terms(
    cluster_labels: dict[int, list[str]],
    km_model=None,
    show: bool = False,
) -> Path:
    """
    Горизонтальные bar-chart'ы с топ-словами для каждого кластера.
    cluster_labels: {cluster_id: [word1, word2, ...]}
    """
    n_clusters = len(cluster_labels)
    cols       = min(3, n_clusters)
    rows       = (n_clusters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    palette = sns.color_palette("tab10", n_clusters)

    for cl_id, words in sorted(cluster_labels.items()):
        ax    = axes[cl_id]
        words = words[:10]
        y_pos = range(len(words))

        # Если есть модель — используем веса центроида; иначе — равные
        if km_model is not None:
            centroid = km_model.cluster_centers_[cl_id]
        else:
            centroid = None

        ax.barh(y_pos, [1] * len(words),
                color=palette[cl_id], alpha=0.75)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Вес (TF-IDF)")
        ax.set_title(f"Кластер {cl_id}", fontsize=10, fontweight="bold")
        ax.set_xticks([])

    # Скрываем лишние подграфики
    for i in range(n_clusters, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Рис. 2.7. Топ-слова тематических кластеров K-Means",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    return _save(fig, "fig_2_7_top_terms", show)


# ── 8. Матрица ошибок ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list[list[int]],
    model_name: str = "logreg",
    labels: list[str] | None = None,
    show: bool = False,
) -> Path:
    """
    Матрица ошибок классификатора в виде аннотированной тепловой карты.
    cm: список списков (confusion_matrix.tolist())
    """
    if labels is None:
        labels = ["Негативная", "Нейтральная", "Позитивная"]

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_arr,
        ax=ax,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        cbar=False,
    )
    ax.set_xlabel("Предсказанная метка")
    ax.set_ylabel("Истинная метка")
    model_display = "LogisticRegression" if model_name == "logreg" else "SVM (LinearSVC)"
    ax.set_title(f"Рис. 2.8. Матрица ошибок классификатора\n({model_display})")
    fig.tight_layout()

    return _save(fig, f"fig_2_8_confusion_matrix_{model_name}", show)


# ── 9. Сравнение LogReg vs SVM ────────────────────────────────────────────────

def plot_model_comparison(
    compare_df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Grouped bar chart: LogReg vs SVM по метрикам Precision/Recall/F1.
    compare_df: Метрика, LogReg, SVM (строки — метрики, значения — числа).
    """
    metrics_to_plot = [
        "Precision (macro)", "Recall (macro)", "F1 (macro)", "CV F1 (среднее)"
    ]
    sub = compare_df[compare_df["Метрика"].isin(metrics_to_plot)].copy()

    x     = np.arange(len(sub))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    lr_vals  = pd.to_numeric(sub["LogReg"], errors="coerce").values
    svm_vals = pd.to_numeric(sub["SVM"],    errors="coerce").values

    bars1 = ax.bar(x - width / 2, lr_vals,  width, label="LogisticRegression",
                   color=COLOR_STATE,  alpha=0.8)
    bars2 = ax.bar(x + width / 2, svm_vals, width, label="SVM (LinearSVC)",
                   color=COLOR_PUBLIC, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sub["Метрика"].values, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Рис. 2.9. Сравнение качества классификаторов тональности\n"
                 "(LogisticRegression vs SVM, тестовая выборка)")
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout()

    return _save(fig, "fig_2_9_model_comparison", show)


# ── 10. Влияние событий на тональность ───────────────────────────────────────

def plot_event_impact(
    df: pd.DataFrame,
    show: bool = False,
) -> Path:
    """
    Горизонтальный bar-chart: δ тональности для каждого события.
    df: event_label, delta (может быть None).
    """
    df2 = df.dropna(subset=["delta"]).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLOR_POSITIVE if d >= 0 else COLOR_NEGATIVE
              for d in df2["delta"]]
    bars = ax.barh(df2["event_label"], df2["delta"],
                   color=colors, alpha=0.8, height=0.5)
    ax.axvline(0, color="#7f8c8d", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_xlabel("Δ индекса тональности (после − до события)")
    ax.set_title("Рис. 2.10. Изменение тональности дискуссии\n"
                 "в ±7 дней вокруг ключевых событий дела Долиной")
    ax.invert_yaxis()
    fig.tight_layout()

    return _save(fig, "fig_2_10_event_impact", show)


# ── Генерация всех графиков ───────────────────────────────────────────────────

def save_all(show: bool = False) -> list[Path]:
    """
    Загружает все необходимые данные из RESULTS_DIR
    и генерирует полный набор графиков.

    Returns:
        Список путей к сохранённым файлам.
    """

    saved = []

    def _try_load(name: str) -> pd.DataFrame | None:
        path = RESULTS_DIR / f"{name}.csv"
        if path.exists():
            return pd.read_csv(path)
        log.warning("Файл не найден, пропускаем: %s", path)
        return None

    # 1. Динамика активности
    df_act = _try_load("activity_weekly")
    if df_act is not None:
        saved.append(plot_activity_timeline(df_act, show))

    # 2. Динамика тональности (stacked)
    df_sent = _try_load("sentiment_weekly")
    if df_sent is not None:
        saved.append(plot_sentiment_timeline(df_sent, show))

    # 3. Взвешенный индекс S(t)
    df_agg = _try_load("agg_by_period_weekly")
    if df_agg is not None:
        saved.append(plot_sentiment_index(df_agg, show))

    # 4. Тепловая карта каналов
    df_ch = _try_load("channel_comparison")
    if df_ch is not None:
        df_ch = df_ch.set_index(df_ch.columns[0])
        saved.append(plot_channel_heatmap(df_ch, show))

    # 5. Расхождение ориентаций
    df_div = _try_load("orientation_divergence")
    if df_div is not None:
        saved.append(plot_orientation_divergence(df_div, show))

    # 6. t-SNE кластеров
    if CLUSTERS_CSV.exists():
        df_cl = pd.read_csv(CLUSTERS_CSV)
        saved.append(plot_cluster_tsne(df_cl, show))

        # 7. Топ-слова кластеров (из файла меток)
        cluster_labels_path = RESULTS_DIR / "cluster_labels.json"
        if cluster_labels_path.exists():
            with open(cluster_labels_path, encoding="utf-8") as f:
                cl_labels = {int(k): v for k, v in json.load(f).items()}
            saved.append(plot_top_terms(cl_labels, show=show))

    # 8–9. Матрица ошибок и сравнение моделей
    if METRICS_JSON.exists():
        with open(METRICS_JSON, encoding="utf-8") as f:
            metrics_data = json.load(f)

        for model_name in ["logreg", "svm"]:
            cm = metrics_data.get(model_name, {}).get("confusion_matrix")
            if cm:
                saved.append(plot_confusion_matrix(cm, model_name, show=show))

        # Сравнение моделей
        from models.svm_clf import comparison_table
        try:
            cmp_df = comparison_table()
            saved.append(plot_model_comparison(cmp_df, show))
        except Exception as exc:
            log.warning("Не удалось построить сравнение моделей: %s", exc)

    # 10. Влияние событий
    df_impact = _try_load("event_impact")
    if df_impact is not None:
        saved.append(plot_event_impact(df_impact, show))

    log.info("Всего графиков сохранено: %d", len(saved))
    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Генерация всех графиков проекта.")
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