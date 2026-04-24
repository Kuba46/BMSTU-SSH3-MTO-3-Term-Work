"""
config/settings.py
==================
Централизованное хранилище всех констант проекта.
Все остальные модули импортируют нужные значения отсюда.
"""


from pathlib import Path
from datetime import date

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LABELED_DATA_DIR = DATA_DIR / "labeled"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

for dir in [RAW_DATA_DIR, LABELED_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

TELEGRAM_API_ID = "your_telegram_api_id"
TELEGRAM_API_HASH = "your_telegram_api_hash"
TELEGRAM_SESSION = "dolina_session"


CHANNELS = [
    {
        "username": "shot_shot",
        "label": "SHOT",
        "orientation": "public",
        "has_comments": True,
        "has_reactions": True,
    },
    {
        "username": "AlexCarrier",
        "label": "Александр Картавых",
        "orientation": "public",
        "has_comments": True,
        "has_reactions": True,
    },
    {
        "username": "topor",
        "label": "ТОПОР — Горячие новости",
        "orientation": "public",
        "has_comments": True,
        "has_reactions": False,
    },
    {
        "username": "Cbpub",
        "label": "КБ",
        "orientation": "public",
        "has_comments": False,
        "has_reactions": True,
    },
    {
        "username": "toporlive",
        "label": "Топор Live",
        "orientation": "public",
        "has_comments": False,
        "has_reactions": True,
    },
    {
        "username": "rian_ru",
        "label": "РИА Новости",
        "orientation": "state",
        "has_comments": False,
        "has_reactions": True,
    },
    {
        "username": "rbc_news",
        "label": "РБК",
        "orientation": "state",
        "has_comments": False,
        "has_reactions": True,
    },
    {
        "username": "rt_russian",
        "label": "RT на русском",
        "orientation": "state",
        "has_comments": False,
        "has_reactions": True,
    },
    {
        "username": "uranews",
        "label": "URA.RU",
        "orientation": "state",
        "has_comments": False,
        "has_reactions": True,
    },
    {
        "username": "davankov",
        "label": "ДАВАНКОВ // Вице-спикер Госдумы",
        "orientation": "state",
        "has_comments": True,
        "has_reactions": True,
    }
]

CHANNEL_USERNAMES = [ch["username"] for ch in CHANNELS]

CORPUS_START_DATE = date(2025, 3, 1)
CORPUS_END_DATE = date(2025, 12, 31)

KEYWORDS = [
    "долина",
    "лурье",
    "схема долиной",
    "добросовестный покупатель",
    "реституция",
    "хамовнический суд",
    "верховный суд",
    "02-0387",
    "мошенники",
    "квартира долин",
    "казус долиной",
    "эффект долиной",
]

MIN_TOKEN_LENGTH = 3

ALLOWED_POS_TAGS = {"NOUN", "VERB", "ADJF", "ADJS", "ADVB", "INFN"}

STOPWORDS_PATH = ROOT_DIR / "nlp" / "stopwords_ru.txt"

TFIDF_PARAMS = {
    "max_features": 5_000, # размер словаря
    "ngram_range": (1, 2), # uni- и биграммы
    "min_df": 3, # минимум 3 документа для включения термина
    "max_df": 0.90, # игнорируем слова из >90 % документов
    "sublinear_tf": True, # логарифмическое масштабирование TF
}

SENTIMENT_LABELS = {
    "positive": 1,
    "neutral":  0,
    "negative": -1,
}
SENTIMENT_LABEL_NAMES = {v: k for k, v in SENTIMENT_LABELS.items()}
 
# Параметры логистической регрессии
LOGREG_PARAMS = {
    "C":            1.0,
    "max_iter":     1_000,
    "solver":       "lbfgs",
    "multi_class":  "multinomial",
    "random_state": 42,
}

SVM_PARAMS = {
    "C": 1.0,
    "kernel": "linear",
    "max_iter": 5_000,
    "random_state": 42,
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

KMEANS_K_RANGE = range(2, 12) # диапазон количества кластеров для KMeans
KMEANS_N_CLUSTERS = 6 # оптимальное количество кластеров для KMeans (по результатам анализа)

DBSCAN_EPS = 0.5 # радиус окрестности для DBSCAN
DBSCAN_MIN_SAMPLES = 5 # минимальное количество точек для формирования кластера

TSNE_PARAMS = {
    "n_components": 2,
    "perplexity": 30,
    "random_state": 42,
    "n_iter": 1_000,
}

EVENTS = [
    {
        "date": date(2025, 3, 15),
        "label": "Решение Хамовнического суда",
        "short": "Хам. суд",
        "description": "Сделка признана недействительной; Лурье без жилья и денег",
    },
    {
        "date": date(2025, 9, 10),
        "label": "Решение Мосгорсуда",
        "short": "Мосгорсуд",
        "description": "Апелляция оставила решение в силе",
    },
    {
        "date": date(2025, 10, 1),
        "label": "Эффект Долиной",
        "short": "Эффект Долиной",
        "description": "Аналогичные иски по всей России; пик дискуссии",
    },
    {
        "date": date(2025, 11, 20),
        "label": "Второй кассационный суд",
        "short": "2-й кассац.",
        "description": "Жалоба Лурье отклонена",
    },
    {
        "date": date(2025, 12, 2),
        "label": "Жалоба в Верховный суд",
        "short": "ВС: жалоба",
        "description": "ВС истребовал материалы за сутки",
    },
    {
        "date": date(2025, 12, 16),
        "label": "Решение Верховного суда",
        "short": "ВС: решение",
        "description": "Все три предыдущих решения отменены; квартира за Лурье",
    },
    {
        "date": date(2025, 12, 18),
        "label": "Публикация текста решения ВС",
        "short": "Текст ВС",
        "description": "Сайт ВС: 2 млн посещений за 1 минуту",
    },
]

COLOR_POSITIVE = "#2ecc5d"  # зелёный — позитивная тональность
COLOR_NEUTRAL  = "#95a5a6"  # серый   — нейтральная
COLOR_NEGATIVE = "#e74c3c"  # красный — негативная
COLOR_STATE    = "#2980b9"  # синий   — гос. каналы
COLOR_PUBLIC   = "#8e44ad"  # фиолетовый — общ. каналы

FIGURE_DPI = 150
FIGURE_SIZE_WIDE = (14, 5)
FIGURE_SIZE_SQUARE = (8, 8)

RAW_CSV = RAW_DATA_DIR / "posts_raw.csv"
LABELED_CSV = LABELED_DATA_DIR / "posts_labeled.csv"
PROCESSED_CSV = PROCESSED_DATA_DIR / "posts_processed.csv"
TFIDF_MATRIX = PROCESSED_DATA_DIR / "tfidf_matrix.npz"
TFIDF_VOCAB = PROCESSED_DATA_DIR / "tfidf_vocab.json"
LOGREG_MODEL = MODELS_DIR / "logreg_sentiment.pkl"
SVM_MODEL = MODELS_DIR / "svm_sentiment.pkl"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"
CLUSTERS_CSV = RESULTS_DIR / "clusters.csv"
METRICS_JSON = RESULTS_DIR / "metrics.json"