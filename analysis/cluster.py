"""
analysis/cluster.py
===================
Кластеризация корпуса: тематические группы (K-Means) и
выявление аномальных всплесков активности (DBSCAN).

Функции:
    run_kmeans()        — разбивает корпус на k тематических кластеров
    elbow_analysis()    — метод «локтя» для подбора оптимального k
    run_dbscan()        — находит всплески/аномалии в TF-IDF-пространстве
    label_clusters()    — присваивает каждому кластеру топ-слова как метку
    tsne_projection()   — проецирует матрицу в 2D для визуализации

Запуск:
    python -m analysis.cluster            # K-Means + DBSCAN
    python -m analysis.cluster --elbow    # только метод «локтя»
    python -m analysis.cluster --k 6      # задать число кластеров явно
"""


import argparse
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from config.settings import (
    CLUSTERS_CSV,
    COMMENTS_PREDICTIONS_CSV,
    COMMENTS_PROCESSED_CSV,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    KMEANS_K_RANGE,
    KMEANS_N_CLUSTERS,
    PREDICTIONS_ALL_CSV,
    PREDICTIONS_CSV,
    PROCESSED_CSV,
    RANDOM_STATE,
    RESULTS_DIR,
    TFIDF_MATRIX,
    TFIDF_VOCAB,
    TSNE_PARAMS,
)

log = logging.getLogger(__name__)


# ── Загрузка артефактов ───────────────────────────────────────────────────────
def _resolve_paths(prefix: str = "") -> tuple[Path, Path, Path, Path, Path, Path]:
    if prefix == "comments_":
        return (
            TFIDF_MATRIX.with_name("comments_tfidf_matrix.npz"),
            TFIDF_VOCAB.with_name("comments_tfidf_vocab.json"),
            COMMENTS_PREDICTIONS_CSV,
            COMMENTS_PROCESSED_CSV,
            RESULTS_DIR / "comments_clusters.csv",
            RESULTS_DIR / "comments_cluster_labels.json",
        )
    if prefix == "all_":
        return (
            TFIDF_MATRIX.with_name("all_tfidf_matrix.npz"),
            TFIDF_VOCAB.with_name("all_tfidf_vocab.json"),
            PREDICTIONS_ALL_CSV,
            PROCESSED_CSV,
            RESULTS_DIR / "all_clusters.csv",
            RESULTS_DIR / "all_cluster_labels.json",
        )
    return (
        TFIDF_MATRIX,
        TFIDF_VOCAB,
        PREDICTIONS_CSV,
        PROCESSED_CSV,
        CLUSTERS_CSV,
        RESULTS_DIR / "cluster_labels.json",
    )


def _load_matrix_and_corpus(
    matrix_path: Path,
    vocab_path: Path,
    predictions_path: Path,
    processed_path: Path,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """
    Загружает TF-IDF-матрицу и корпус.
    Если есть predictions.csv — берёт его (содержит sentiment_pred),
    иначе — processed.csv.
    """
    from nlp.vectorizer import load_matrix

    matrix, vocab = load_matrix(matrix_path, vocab_path)

    source = predictions_path if predictions_path.exists() else processed_path
    if not source.exists():
        raise FileNotFoundError(
            f"Корпус не найден: {source}\n"
            "Запустите NLP-пайплайн и models/predict.py"
        )
    df = pd.read_csv(source, encoding="utf-8")
    df["text_lemma"] = df["text_lemma"].fillna("").astype(str)

    # Матрица и корпус должны совпадать по числу строк
    if matrix.shape[0] != len(df):
        raise ValueError(
            f"Несоответствие размеров: матрица {matrix.shape[0]} строк, "
            f"корпус {len(df)} строк.\n"
            "Перезапустите vectorizer.py после всех изменений корпуса."
        )
    return matrix, df
 
 
# ── Метод «локтя» ─────────────────────────────────────────────────────────────
def elbow_analysis(
    matrix: sp.csr_matrix,
    k_range=KMEANS_K_RANGE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Вычисляет inertia (сумму внутрикластерных квадратов расстояний)
    для каждого k из k_range.
 
    Returns:
        DataFrame с колонками ['k', 'inertia']
        (используется в viz/plotter.py для построения графика «локтя»)
    """
    log.info("Метод «локтя»: проверяем k = %s...", list(k_range))

    # L2-нормализуем матрицу — это улучшает качество K-Means
    # (после нормализации евклидово расстояние ≈ косинусное)
    X = normalize(matrix, norm="l2")

    records = []
    for k in k_range:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state,
        )
        km.fit(X)
        records.append({"k": k, "inertia": km.inertia_})
        log.info("  k=%2d  inertia=%.2f", k, km.inertia_)

    return pd.DataFrame(records)


# ── K-Means ───────────────────────────────────────────────────────────────────
def run_kmeans(
    matrix: sp.csr_matrix,
    n_clusters: int = KMEANS_N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> tuple[KMeans, np.ndarray]:
    """
    Запускает K-Means с k-means++ инициализацией.

    Args:
        matrix:     TF-IDF матрица (разреженная)
        n_clusters: число кластеров
        random_state: seed

    Returns:
        (fitted_model, labels) где labels — массив меток кластеров
        длиной = числу документов
    """
    log.info("K-Means: n_clusters=%d...", n_clusters)

    X = normalize(matrix, norm="l2")

    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=15,          # больше инициализаций → стабильнее результат
        max_iter=500,
        random_state=random_state,
        verbose=0,
    )
    km.fit(X)

    labels = km.labels_
    unique, counts = np.unique(labels, return_counts=True)

    log.info("K-Means завершён. Распределение по кластерам:")
    for cl, cnt in zip(unique, counts):
        log.info("  Кластер %d: %d документов (%.1f%%)", cl, cnt,
                 cnt / len(labels) * 100)
    return km, labels


def label_clusters(
    km: KMeans,
    vocab: dict[str, int],
    n_top: int = 10,
) -> dict[int, list[str]]:
    """
    Присваивает каждому кластеру список топ-N слов по близости
    к центроиду кластера.

    Args:
        km:    обученная модель KMeans
        vocab: словарь {term: column_index} из vectorizer
        n_top: число слов-меток
    
    Returns:
        {cluster_id: [word1, word2, ...]}
    """
    # Инвертируем словарь: {col_idx: term}
    idx_to_term = {v: k for k, v in vocab.items()}
    n_terms     = len(vocab)
    result      = {}

    for cl_id, centroid in enumerate(km.cluster_centers_):
        # Берём только те индексы, что реально есть в словаре
        valid_idx = [i for i in range(min(len(centroid), n_terms))]
        top_idx   = np.argsort(centroid[valid_idx])[::-1][:n_top]
        result[cl_id] = [idx_to_term.get(i, f"term_{i}") for i in top_idx]

    log.info("Метки кластеров (топ-%d слов):", n_top)
    for cl_id, words in result.items():
        log.info("  Кластер %d: %s", cl_id, ", ".join(words))
    return result


def save_cluster_labels(
    km: KMeans,
    vocab: dict[str, int],
    n_top: int = 10,
    output_path=RESULTS_DIR / "cluster_labels.json",
) -> dict[int, list[dict[str, float]]]:
    """
    Сохраняет топ-слова кластеров вместе с весами центроидов.

    Это позволяет визуализациям показывать реальные значения,
    а не одинаковые декоративные бары.
    """
    idx_to_term = {v: k for k, v in vocab.items()}
    n_terms = len(vocab)
    result: dict[int, list[dict[str, float]]] = {}

    for cl_id, centroid in enumerate(km.cluster_centers_):
        valid_idx = range(min(len(centroid), n_terms))
        top_idx = np.argsort(centroid[list(valid_idx)])[::-1][:n_top]
        result[cl_id] = [
            {
                "term": idx_to_term.get(i, f"term_{i}"),
                "weight": float(centroid[i]),
            }
            for i in top_idx
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info("Метки кластеров сохранены → %s", output_path)
    return result


# ── DBSCAN ────────────────────────────────────────────────────────────────────
def run_dbscan(
    matrix: sp.csr_matrix,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """
    Запускает DBSCAN для выявления плотных скоплений
    и «шумовых» документов (аномалий/всплесков).

    Метрика — косинусное расстояние: подходит для TF-IDF пространства,
    так как не зависит от длины документа.

    Labels:
       ≥ 0  → номер кластера (плотная группа)
      -1    → шум (документ не вошёл ни в один кластер)

    Returns:
        labels — массив меток длиной = числу документов
    """
    log.info(
        "DBSCAN: eps=%.3f, min_samples=%d, метрика=cosine...",
        eps, min_samples,
    )

    # DBSCAN с косинусной метрикой требует плотную матрицу
    # Для больших корпусов используем algorithm='ball_tree'
    X_dense = matrix.toarray()

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        algorithm="brute",   # brute — единственный поддерживающий косинус
        n_jobs=-1,
    )
    labels = dbscan.fit_predict(X_dense)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    log.info(
        "DBSCAN завершён: кластеров=%d, шумовых документов=%d (%.1f%%)",
        n_clusters, n_noise, n_noise / len(labels) * 100,
    )
    return labels


def spike_documents(
    df: pd.DataFrame,
    dbscan_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Извлекает «шумовые» документы DBSCAN — потенциальные всплески активности.
    Сортирует по дате для удобства ивент-анализа.

    Returns:
        DataFrame с колонками: date, channel_label, text, views, reactions_total
    """
    df = df.copy()
    df["dbscan_label"] = dbscan_labels

    noise = df[df["dbscan_label"] == -1].copy()
    noise["date"] = pd.to_datetime(noise["date"], utc=True, errors="coerce")
    noise = noise.sort_values("date")

    cols = [c for c in ["date", "channel_label", "orientation",
                         "text", "views", "reactions_total", "forwards"]
            if c in noise.columns]
    log.info("Выявлено аномальных документов (DBSCAN noise): %d", len(noise))
    return noise[cols].reset_index(drop=True)


# ── t-SNE проекция ────────────────────────────────────────────────────────────
def tsne_projection(
    matrix: sp.csr_matrix,
    params: dict | None = None,
) -> np.ndarray:
    """
    Снижает размерность TF-IDF матрицы до 2D методом t-SNE.
    Используется для визуализации кластеров в viz/plotter.py.

    Важно: t-SNE работает только с плотными матрицами.
    Для больших корпусов (>5000 документов) предварительно
    применяем TruncatedSVD (LSA) до 50 компонент.

    Returns:
        ndarray формы (n_docs, 2)
    """
    if params is None:
        params = TSNE_PARAMS

    n_docs = matrix.shape[0]
    log.info("t-SNE проекция: %d документов → 2D...", n_docs)

    if n_docs > 5_000:
        log.info("Корпус >5000 — предварительно применяем TruncatedSVD(50)...")
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
        X = svd.fit_transform(matrix)
        log.info("SVD: объяснённая дисперсия = %.2f%%",
                 svd.explained_variance_ratio_.sum() * 100)
    else:
        X = matrix.toarray()

    tsne   = TSNE(**params)
    coords = tsne.fit_transform(X)
    log.info("t-SNE завершён. KL-дивергенция = %.4f", tsne.kl_divergence_)
    return coords


# ── Сборка итогового DataFrame с кластерами ───────────────────────────────────
def build_cluster_dataframe(
    df: pd.DataFrame,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    cluster_labels: dict[int, list[str]],
    tsne_coords: np.ndarray | None = None,
    output_path: Path = CLUSTERS_CSV,
) -> pd.DataFrame:
    """
    Собирает итоговый DataFrame с результатами кластеризации.
    Сохраняет его в CLUSTERS_CSV.

    Колонки результата:
        все исходные + kmeans_cluster, kmeans_label,
        dbscan_label, tsne_x, tsne_y (если есть)
    """
    df = df.copy()
    df["kmeans_cluster"] = kmeans_labels
    df["kmeans_label"]   = [
        ", ".join(cluster_labels.get(cl, [])[:5])
        for cl in kmeans_labels
    ]
    df["dbscan_label"] = dbscan_labels

    if tsne_coords is not None:
        df["tsne_x"] = tsne_coords[:, 0]
        df["tsne_y"] = tsne_coords[:, 1]
    df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Кластеры сохранены → %s  (%d строк)", output_path, len(df))
    return df


# ── Основной пайплайн ─────────────────────────────────────────────────────────
def run_pipeline(
    n_clusters: int = KMEANS_N_CLUSTERS,
    run_tsne: bool = True,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Полный пайплайн кластеризации:
        1. Загрузка матрицы и корпуса
        2. K-Means
        3. Присвоение меток кластерам
        4. DBSCAN
        5. t-SNE (опционально)
        6. Сборка и сохранение результата
    """
    matrix_path, vocab_path, predictions_path, processed_path, clusters_path, labels_path = _resolve_paths(prefix)
    matrix, df = _load_matrix_and_corpus(matrix_path, vocab_path, predictions_path, processed_path)

    # Загружаем словарь
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    # K-Means
    km, km_labels = run_kmeans(matrix, n_clusters=n_clusters)
    cl_labels     = label_clusters(km, vocab)
    save_cluster_labels(km, vocab, output_path=labels_path)

    # DBSCAN
    db_labels = run_dbscan(matrix)

    # t-SNE
    tsne_coords = None
    if run_tsne:
        tsne_coords = tsne_projection(matrix)

    # Сборка
    df_result = build_cluster_dataframe(
        df, km_labels, db_labels, cl_labels, tsne_coords, output_path=clusters_path
    )

    # Выводим всплески для ивент-анализа
    spikes = spike_documents(df_result, db_labels)
    spikes_path = RESULTS_DIR / f"{prefix}spikes.csv"
    spikes.to_csv(spikes_path, index=False, encoding="utf-8")
    log.info("Аномальные документы сохранены → %s", spikes_path)
    return df_result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Кластеризация корпуса: K-Means и DBSCAN."
    )
    parser.add_argument(
        "--k", type=int, default=KMEANS_N_CLUSTERS,
        help=f"Число кластеров K-Means (по умолчанию {KMEANS_N_CLUSTERS}).",
    )
    parser.add_argument(
        "--elbow", action="store_true",
        help="Только метод «локтя» — вычислить inertia для диапазона k.",
    )
    parser.add_argument(
        "--no-tsne", action="store_true",
        help="Пропустить t-SNE (экономит время на больших корпусах).",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Префикс набора артефактов (например comments_ или all_).",
    )
    args = parser.parse_args()

    if args.elbow:
        matrix_path, vocab_path, predictions_path, processed_path, _, _ = _resolve_paths(args.prefix)
        matrix, _ = _load_matrix_and_corpus(matrix_path, vocab_path, predictions_path, processed_path)
        elbow_df  = elbow_analysis(matrix)
        print("\nМетод «локтя»:")
        print(elbow_df.to_string(index=False))
    else:
        run_pipeline(n_clusters=args.k, run_tsne=not args.no_tsne, prefix=args.prefix)


if __name__ == "__main__":
    main()