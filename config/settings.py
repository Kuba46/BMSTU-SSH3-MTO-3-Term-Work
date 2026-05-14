"""
config/settings.py
==================
Централизованное хранилище всех констант проекта.
Все остальные модули импортируют нужные значения отсюда.
"""


from pathlib import Path
from datetime import date

# ── Корневые пути ─────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
CLEANED_DIR    = DATA_DIR / "cleaned"
REMOVED_DIR    = DATA_DIR / "removed"
LABELED_DIR    = DATA_DIR / "labeled"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = ROOT_DIR / "models" / "saved"
RESULTS_DIR    = ROOT_DIR / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"
NOTEBOOKS_DIR  = ROOT_DIR / "notebooks"

# Создаём папки при импорте (если не существуют)
for _d in [RAW_DIR, CLEANED_DIR, REMOVED_DIR, LABELED_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


try:
    from config.telegram_credentials_local import (
        TELEGRAM_API_ID as _TELEGRAM_API_ID,
        TELEGRAM_API_HASH as _TELEGRAM_API_HASH,
        TELEGRAM_SESSION as _TELEGRAM_SESSION,
    )

    TELEGRAM_API_ID = _TELEGRAM_API_ID
    TELEGRAM_API_HASH = _TELEGRAM_API_HASH
    TELEGRAM_SESSION = _TELEGRAM_SESSION
except ImportError:
    try:
        from telegram_credentials_local import (
            TELEGRAM_API_ID as _TELEGRAM_API_ID,
            TELEGRAM_API_HASH as _TELEGRAM_API_HASH,
            TELEGRAM_SESSION as _TELEGRAM_SESSION,
        )

        TELEGRAM_API_ID = _TELEGRAM_API_ID
        TELEGRAM_API_HASH = _TELEGRAM_API_HASH
        TELEGRAM_SESSION = _TELEGRAM_SESSION
    except ImportError:
        pass


# ── Выборка каналов ───────────────────────────────────────────────────────────
# Каждый канал описывается словарём:
#   username   — @-имя канала (без @)
#   label      — человекочитаемое название
#   orientation— "public" (общественный) | "state" (гос./провластный)
#   has_comments — True, если у канала открыты комментарии
#   has_reactions— True, если у канала включены реакции
# Типы реакций (reaction_type):
#   "emoji_full"  — полный набор эмодзи-реакций (Telegram Premium + стандартные)
#   "like_dislike"— только 👍 и 👎 (ограниченный режим, напр. РИА Новости)
#   "none"        — реакции отключены (напр. ТОПОР — только комментарии)
#
# Тип комментариев (comment_type):
#   "open"        — комментарии открыты для всех подписчиков
#   "none"        — комментарии отключены
#
# interaction_note — текстовое описание ограничений канала для документации

CHANNELS = [
    {
        "username":       "shot_shot",
        "label":          "SHOT",
        "orientation":    "public",
        "has_comments":   True,
        "comment_type":   "open",
        "has_reactions":  True,
        "reaction_type":  "emoji_full",
        "interaction_note": "Полный набор реакций; комментарии открыты. "
                            "На момент периода исследования (март–дек. 2025) "
                            "комментарии были временно закрыты под рядом постов.",
    },
    {
        "username":       "AlexCarrier",
        "label":          "Александр Картавых",
        "orientation":    "public",
        "has_comments":   True,
        "comment_type":   "open",
        "has_reactions":  True,
        "reaction_type":  "emoji_full",
        "interaction_note": "Полный набор реакций; комментарии открыты.",
    },
    {
        "username":       "topor",
        "label":          "ТОПОР — Горячие новости",
        "orientation":    "public",
        "has_comments":   True,
        "comment_type":   "open",
        "has_reactions":  False,
        "reaction_type":  "none",
        "interaction_note": "Реакции полностью отключены. "
                            "Единственный канал выборки с комментариями "
                            "но без реакций — анализ вовлечённости ведётся "
                            "исключительно на основе текстов комментариев.",
    },
    {
        "username":       "Cbpub",
        "label":          "КБ",
        "orientation":    "public",
        "has_comments":   False,
        "comment_type":   "none",
        "has_reactions":  True,
        "reaction_type":  "emoji_full",
        "interaction_note": "Комментарии закрыты; доступны только реакции.",
    },
    {
        "username":       "toporlive",
        "label":          "Топор Live",
        "orientation":    "public",
        "has_comments":   False,
        "comment_type":   "none",
        "has_reactions":  True,
        "reaction_type":  "emoji_full",
        "interaction_note": "Комментарии закрыты; доступны только реакции.",
    },
    {
        "username":       "rian_ru",
        "label":          "РИА Новости",
        "orientation":    "state",
        "has_comments":   False,
        "comment_type":   "none",
        "has_reactions":  True,
        "reaction_type":  "like_dislike",
        "interaction_note": "Ограниченный набор реакций: только 👍 (лайк) "
                            "и 👎 (дизлайк). Это методологически значимо: "
                            "бинарная шкала одобрения/неодобрения даёт "
                            "прямую метрику тональности аудитории, "
                            "не требующую классификации эмодзи.",
    },
    {
        "username":       "rbc_news",
        "label":          "РБК",
        "orientation":    "state",
        "has_comments":   False,
        "comment_type":   "none",
        "has_reactions":  True,
        "reaction_type":  "emoji_full",
        "interaction_note": "Комментарии закрыты; доступны только реакции.",
    },
    {
        "username":       "rt_russian",
        "label":          "RT на русском",
        "orientation":    "state",
        "has_comments":   False,
        "comment_type":   "none",
        "has_reactions":  True,
        "reaction_type":  "emoji_full",
        "interaction_note": "Комментарии закрыты; доступны только реакции.",
    },
    {
        "username":       "uranews",
        "label":          "URA.RU",
        "orientation":    "state",
        "has_comments":   False,
        "comment_type":   "none",
        "has_reactions":  True,
        "reaction_type":  "emoji_full",
        "interaction_note": "Комментарии закрыты; доступны только реакции.",
    },
    {
        "username":       "davankov",
        "label":          "ДАВАНКОВ // Вице-спикер Госдумы",
        "orientation":    "state",
        "has_comments":   True,
        "comment_type":   "open",
        "has_reactions":  True,
        "reaction_type":  "emoji_full",
        "interaction_note": "Полный набор реакций; комментарии открыты.",
    },
]

# Быстрый доступ: список username-ов
CHANNEL_USERNAMES = [ch["username"] for ch in CHANNELS]

# ── Временные рамки корпуса ───────────────────────────────────────────────────
# Март 2025 — первое решение Хамовнического суда
# Декабрь 2025 — решение Верховного суда
CORPUS_START_DATE = date(2025, 3, 1)
CORPUS_END_DATE = date(2025, 12, 31)

# ── Ключевые слова для фильтрации постов ─────────────────────────────────────
# Пост считается релевантным, если содержит хотя бы одно из этих слов.
# Регистронезависимый поиск (применяется к lower-case тексту).
KEYWORDS = [
    "долина",
    "лурье",
    "схема долиной",
    "добросовестный покупатель",
    "реституция",
    "хамовнический суд",
    "верховный суд",     # в сочетании с другими — фильтруется в dataset.py
    "02-0387",           # номер дела
    "мошенники",         # в контексте дела
    "квартира долин",
    "казус долиной",
    "эффект долиной",
]

# ── NLP / предобработка ───────────────────────────────────────────────────────
# Минимальная длина токена (символов) после лемматизации
MIN_TOKEN_LENGTH = 3

# Разрешённые части речи (pymorphy2 POS-теги).
# Оставляем существительные, прилагательные, глаголы, наречия.
ALLOWED_POS = {"NOUN", "ADJF", "ADJS", "VERB", "INFN", "ADVB"}

# Путь к пользовательскому списку стоп-слов
STOPWORDS_PATH = ROOT_DIR / "nlp" / "stopwords_ru.txt"

# ── TF-IDF ────────────────────────────────────────────────────────────────────
TFIDF_PARAMS = {
    "max_features":  5_000,   # размер словаря
    "ngram_range":   (1, 2),  # uni- и биграммы
    "min_df":        3,       # минимум 3 документа для включения термина
    "max_df":        0.90,    # игнорируем слова из >90 % документов
    "sublinear_tf":  True,    # логарифмическое масштабирование TF
}

# ── Классификаторы тональности ────────────────────────────────────────────────
# Метки классов
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

# Параметры SVM
SVM_PARAMS = {
    "C":            1.0,
    "kernel":       "linear",
    "max_iter":     5_000,
    "random_state": 42,
}

# Доля тестовой выборки при обучении
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Кластеризация ─────────────────────────────────────────────────────────────
# K-Means: диапазон k для метода «локтя»
KMEANS_K_RANGE = range(2, 12)

# K-Means: итоговое число кластеров (устанавливается после анализа «локтя»)
KMEANS_N_CLUSTERS = 6

# DBSCAN: параметры для поиска всплесков активности
DBSCAN_EPS     = 0.5   # радиус окрестности в пространстве TF-IDF
DBSCAN_MIN_SAMPLES = 5  # минимум соседей для формирования ядра

# t-SNE: параметры визуализации кластеров
TSNE_PARAMS = {
    "n_components": 2,
    "perplexity":   30,
    "random_state": 42,
    "max_iter":     1_000,
}

# ── Ивент-анализ ──────────────────────────────────────────────────────────────
# Ключевые события дела Долиной с датами и описаниями.
# Используются в event_analysis.py для наложения на временны́е ряды тональности.
EVENTS = [
    {
        "date":        date(2025, 3, 15),
        "label":       "Решение Хамовнического суда",
        "short":       "Хам. суд",
        "description": "Сделка признана недействительной; Лурье без жилья и денег",
    },
    {
        "date":        date(2025, 9, 10),
        "label":       "Решение Мосгорсуда",
        "short":       "Мосгорсуд",
        "description": "Апелляция оставила решение в силе",
    },
    {
        "date":        date(2025, 10, 1),
        "label":       "Эффект Долиной",
        "short":       "Эффект Долиной",
        "description": "Аналогичные иски по всей России; пик дискуссии",
    },
    {
        "date":        date(2025, 11, 20),
        "label":       "Второй кассационный суд",
        "short":       "2-й кассац.",
        "description": "Жалоба Лурье отклонена",
    },
    {
        "date":        date(2025, 12, 2),
        "label":       "Жалоба в Верховный суд",
        "short":       "ВС: жалоба",
        "description": "ВС истребовал материалы за сутки",
    },
    {
        "date":        date(2025, 12, 16),
        "label":       "Решение Верховного суда",
        "short":       "ВС: решение",
        "description": "Все три предыдущих решения отменены; квартира за Лурье",
    },
    {
        "date":        date(2025, 12, 18),
        "label":       "Публикация текста решения ВС",
        "short":       "Текст ВС",
        "description": "Сайт ВС: 2 млн посещений за 1 минуту",
    },
]

# ── Визуализация ──────────────────────────────────────────────────────────────
# Палитра для графиков (matplotlib named colors)
COLOR_POSITIVE = "#2ecc71"   # зелёный — позитивная тональность
COLOR_NEUTRAL  = "#95a5a6"   # серый   — нейтральная
COLOR_NEGATIVE = "#e74c3c"   # красный — негативная
COLOR_STATE    = "#2980b9"   # синий   — гос. каналы
COLOR_PUBLIC   = "#8e44ad"   # фиолетовый — общ. каналы

# DPI для сохранения графиков
FIGURE_DPI = 150

# Размер фигур по умолчанию (дюймы)
FIGURE_SIZE_WIDE  = (14, 5)
FIGURE_SIZE_SQUARE = (8, 8)

# ── Имена файлов артефактов ───────────────────────────────────────────────────

# Функция для получения пути к raw-файлу конкретного канала.
# Пример: raw_csv_for("shot_shot") → data/raw/shot_shot_raw.csv
def raw_csv_for(username: str):
    return RAW_DIR / f"{username}_raw.csv"

def comments_raw_for(username: str):
    return RAW_DIR / f"{username}_comments_raw.csv"

def cleaned_csv_for(username: str):
    return CLEANED_DIR / f"{username}_cleaned.csv"

def comments_cleaned_for(username: str):
    return CLEANED_DIR / f"{username}_comments_cleaned.csv"

def removed_csv_for(username: str):
    return REMOVED_DIR / f"{username}_removed.csv"

def comments_removed_for(username: str):
    return REMOVED_DIR / f"{username}_comments_removed.csv"

# Сводный файл — объединение всех каналов (создаётся в dataset.py)
RAW_CSV        = RAW_DIR       / "posts_raw.csv"
CLEANED_CSV    = CLEANED_DIR   / "posts_cleaned.csv"
COMMENTS_RAW_CSV = RAW_DIR       / "comments_raw.csv"
COMMENTS_CLEANED_CSV = CLEANED_DIR / "comments_cleaned.csv"
LABELED_CSV    = LABELED_DIR   / "posts_labeled.csv"
PROCESSED_CSV  = PROCESSED_DIR / "posts_processed.csv"
COMMENTS_PROCESSED_CSV = PROCESSED_DIR / "comments_processed.csv"
TFIDF_MATRIX   = PROCESSED_DIR / "tfidf_matrix.npz"
TFIDF_VOCAB    = PROCESSED_DIR / "tfidf_vocab.json"
LOGREG_MODEL   = MODELS_DIR    / "logreg_sentiment.pkl"
SVM_MODEL      = MODELS_DIR    / "svm_sentiment.pkl"
COMMENTS_SVM_MODEL = MODELS_DIR / "comments_svm_sentiment.pkl"
PREDICTIONS_CSV = RESULTS_DIR  / "predictions.csv"
COMMENTS_PREDICTIONS_CSV = RESULTS_DIR / "comments_predictions.csv"
PREDICTIONS_ALL_CSV = RESULTS_DIR / "predictions_all.csv"
CLUSTERS_CSV   = RESULTS_DIR   / "clusters.csv"
METRICS_JSON   = RESULTS_DIR   / "metrics.json"
COMMENTS_METRICS_JSON = RESULTS_DIR / "comments_metrics.json"

# Быстрый доступ: список username-ов
CHANNEL_USERNAMES = [ch["username"] for ch in CHANNELS]

# Удобные вспомогательные срезы по типу взаимодействия
CHANNELS_WITH_COMMENTS  = [ch for ch in CHANNELS if ch["has_comments"]]
CHANNELS_WITH_REACTIONS = [ch for ch in CHANNELS if ch["has_reactions"]]
CHANNELS_LIKE_DISLIKE   = [ch for ch in CHANNELS if ch.get("reaction_type") == "like_dislike"]
CHANNELS_EMOJI_FULL     = [ch for ch in CHANNELS if ch.get("reaction_type") == "emoji_full"]

# ── Классификация эмодзи-реакций ──────────────────────────────────────────────
#
# Словарь сопоставляет конкретные эмодзи с семантическими категориями.
# Используется в analysis/emoji_analyzer.py для подсчёта
# доли позитивных/негативных/нейтральных реакций под постом.
#
# Источник классификации: анализ наиболее распространённых реакций
# в новостных Telegram-каналах России (Brand Analytics, 2025).

EMOJI_SENTIMENT: dict[str, str] = {
    # ── Позитивные ────────────────────────────────────────────────────────────
    "👍": "positive",   # лайк — базовое одобрение
    "❤️": "positive",   # сердце — одобрение, симпатия
    "🔥": "positive",   # огонь — восторг, «горячая» новость
    "🎉": "positive",   # конфетти — радость, праздник
    "👏": "positive",   # аплодисменты — одобрение действия
    "😍": "positive",   # влюблённые глаза — восхищение
    "🥰": "positive",   # улыбка с сердечками — теплота
    "💯": "positive",   # сто баллов — полное согласие/одобрение
    "🤩": "positive",   # звёздные глаза — восторг
    "😊": "positive",   # улыбка — мягкое одобрение
    "✅": "positive",   # галочка — согласие/подтверждение
    "🙏": "positive",   # молитва/благодарность — одобрение решения
    "💪": "positive",   # бицепс — сила, поддержка
    "⚡": "positive",   # молния — «мощно», экспрессия одобрения
    "🕊️": "positive",  # голубь — справедливость, мир (в контексте суда)

    # ── Негативные ────────────────────────────────────────────────────────────
    "👎": "negative",   # дизлайк — базовое неодобрение
    "😡": "negative",   # злость — возмущение
    "🤬": "negative",   # ругань — сильное возмущение
    "💔": "negative",   # разбитое сердце — разочарование
    "😢": "negative",   # слёзы — грусть, сочувствие жертве
    "😭": "negative",   # рыдание — сильная грусть/возмущение
    "🤦": "negative",   # фейспалм — разочарование, «как так можно»
    "🤮": "negative",   # тошнота — отвращение
    "💩": "negative",   # куча — пренебрежение
    "🚫": "negative",   # запрет — отрицание
    "❌": "negative",   # крест — несогласие
    "🙈": "negative",   # обезьяна, закрывающая глаза — нежелание видеть
    "🤡": "negative",   # клоун — насмешка, унижение (в контексте суда)


    # ── Нейтральные / информационные ─────────────────────────────────────────
    "👀": "neutral",    # глаза — «смотрю», интерес без оценки
    "🤔": "neutral",    # раздумье — сомнение, неоднозначность
    "🤯": "neutral",    # взрыв мозга — шок (без чёткой валентности)
}

# Специальный случай: РИА Новости — только лайк/дизлайк
# При анализе этого канала используется упрощённая схема:
#   👍 → positive, 👎 → negative (других реакций нет)
EMOJI_LIKE_DISLIKE: dict[str, str] = {
    "👍": "positive",
    "👎": "negative",
}

# Обратный словарь: категория → список эмодзи
EMOJI_BY_SENTIMENT: dict[str, list[str]] = {}
for _emoji, _sent in EMOJI_SENTIMENT.items():
    EMOJI_BY_SENTIMENT.setdefault(_sent, []).append(_emoji)
