# Анализ социальных медиа как инструмента формирования общественного мнения

Курсовая работа по направлению «Прикладная информатика», профиль «Информационная аналитика» (МГТУ им. Н. Э. Баумана, кафедра СГН3).

**Тема:** Анализ социальных медиа как инструмента формирования общественного мнения на примере Telegram-каналов и «дела Долиной», 2025 г.

Проект реализует полный цикл автоматизированного исследования:
сбор сообщений из Telegram, лингвистическая предобработка, векторизация TF‑IDF,
классификация тональности (Logistic Regression / SVM), тематическая кластеризация (K‑Means, DBSCAN),
ивент‑анализ и визуализация результатов.

---

## Структура проекта

```text
Algorithm/
├── data/                   # Сбор и первичная обработка данных
│   ├── raw/                # Сырые CSV с постами и комментариями
│   ├── labeled/            # Файл ручной разметки (posts_labeled.csv)
│   ├── processed/          # Обработанные данные (после NLP)
│   ├── cleaner.py          # Очистка данных.
│   ├── collector.py        # Асинхронный сборщик через Telethon
│   ├── dataset.py          # Загрузка, валидация, статистика
│   └── labeler.py          # Интерактивная разметка тональности постов
├── nlp/                    # Предобработка и векторизация текста
│   ├── preprocessor.py     # Очистка: URL, эмодзи, хештеги и пр.
│   ├── lemmatizer.py       # Лемматизация (pymorphy2) + фильтрация стоп‑слов
│   ├── vectorizer.py       # Обучение TF‑IDF, сохранение матрицы
│   └── stopwords_ru.txt    # Пользовательский список стоп‑слов
├── models/                 # Классификаторы тональности
│   ├── sentiment.py        # Logistic Regression
│   ├── svm_clf.py          # SVM (LinearSVC + калибровка)
│   ├── predict.py          # Авторазметка всего корпуса
│   └── saved/              # Сериализованные модели (.pkl)
├── analysis/               # Кластеризация, событийный и emoji-анализ
│   ├── cluster.py          # K‑Means и DBSCAN, t‑SNE
│   ├── event_analysis.py   # Временные ряды тональности и метрики событий
│   ├── aggregator.py       # Агрегация по каналам, периодам, ориентации
│   └── merge_predictions.py # Объединение предсказаний постов и комментариев
├── evaluation/             # Оценка качества моделей
│   └── metrics.py          # Precision, Recall, F1, сравнение моделей
├── vizualization/          # Визуализация
│   ├── plotter_posts.py    # Графики по постам
│   ├── plotter_comments.py # Графики по комментариям
│   └── plotter_emoji.py    # Графики по эмодзи-реакциям
├── config/
│   ├── settings.py         # Общие константы, пути, параметры моделей
│   └── telegram_credentials_local.py  # Учётные данные Telegram API (не под версионным контролем)
├── results/                # Результаты (CSV, метрики, графики)
├── main.py                 # (опционально) точка входа для полного пайплайна
├── requirements.txt
└── README.md
```

---

## Требования

- Python 3.10 или выше
- Рекомендуется использовать виртуальное окружение

### Зависимости

Основные библиотеки, указанные в `requirements.txt`

Дополнительно для совместимости с Python 3.12+ может потребоваться `setuptools`.

Установить всё сразу:

```bash
pip install -r requirements.txt
```

---

## Подготовка к работе

### 1. Учётные данные Telegram API

Для сбора данных необходимо зарегистрировать приложение на [my.telegram.org](https://my.telegram.org/apps) и получить `api_id` и `api_hash`.

Создайте файл `config/telegram_credentials_local.py`:

```python
TELEGRAM_API_ID = 12345678          # Ваш api_id (целое число)
TELEGRAM_API_HASH = "abcdef..."     # Ваш api_hash (строка)
TELEGRAM_SESSION = "session_name"   # Имя файла сессии
```

Этот файл добавлен в `.gitignore` и не должен попадать в репозиторий.

### 2. Настройка ключевых слов и периода сбора

При необходимости отредактируйте `config/settings.py`:

- `CHANNELS` – список каналов (можно добавить / убрать)
- `CORPUS_START_DATE`, `CORPUS_END_DATE` – временной интервал
- `KEYWORDS` – фильтр релевантности


---

## Запуск пайплайна

Все команды выполняются из корня проекта.

### Шаг 1. Сбор данных

```bash
python -m data.collector
```

Будут загружены посты и комментарии из заданных каналов за указанный период.
Результат – CSV‑файлы в `data/raw/`.

### Шаг 2. Очистка, обработка и валидация

```bash
python -m data.cleaner
python -m data.dataset
```

Выполняется сборка сводного файла, проверка дубликатов, статистика корпуса.

Если нужны комментарии отдельным файлом, объедините их:

```bash
python -m data.dataset --comments
```

Результат – `data/raw/comments_raw.csv`.

### Шаг 3. Предобработка текста

```bash
python -m nlp.preprocessor
python -m nlp.lemmatizer
```

Очистка, лемматизация, удаление стоп‑слов. Создаётся колонка `text_lemma` в `data/processed/posts_processed.csv`.

Для комментариев используйте те же шаги, но с входным/выходным файлом:

```bash
python -m nlp.preprocessor --input data/raw/comments_raw.csv --output data/processed/comments_processed.csv
python -m nlp.lemmatizer --input data/processed/comments_processed.csv --output data/processed/comments_processed.csv
```

### Шаг 4. Векторизация

```bash
python -m nlp.vectorizer
```

Строится TF‑IDF матрица и словарь, сохраняются в `data/processed/`.

### Шаг 5. Обучение классификаторов тональности

Перед этим необходимо подготовить файл ручной разметки `data/labeled/posts_labeled.csv`.
Он должен содержать колонки `channel_username`, `post_id`, `sentiment` (1, 0, -1).
Создать его можно с помощью скрипта `data/labeler.py` (интерактивная разметка в терминале).

Подсказки в разметчике:

- `p` — позитивная, `n` — нейтральная, `g` — негативная, `s` — пропустить, `q` — сохранить и выйти
- отображается подсказка по эмодзи‑реакциям (если они есть)

Запуск разметчика:

```bash
python -m data.labeler
```

Обучение моделей:

```bash
python -m models.sentiment        # Logistic Regression
python -m models.svm_clf          # SVM
```

Метрики сохраняются в `results/metrics.json`.

### Шаг 6. Автоматическая разметка всего корпуса

```bash
python -m models.predict
```

Лучшая модель (по F1) применяется ко всем постам. Результат – `results/predictions.csv`.

Для комментариев:

```bash
python -m models.predict --input data/processed/comments_processed.csv --output results/comments_predictions.csv
```

### Шаг 7. Агрегация, emoji-анализ и событийный анализ

```bash
python -m analysis.aggregator
python -m analysis.emoji_analyzer
python -m analysis.event_analysis
```

Рассчитываются индексы тональности по каналам, периодам, ориентации.
Выполняется оконный анализ вокруг ключевых событий.

Для комментариев можно сохранить результаты с отдельным префиксом:

```bash
python -m analysis.aggregator --input results/comments_predictions.csv --prefix comments_
python -m analysis.event_analysis --input results/comments_predictions.csv --prefix comments_
```

Для общего анализа постов + комментариев:

```bash
python -m analysis.merge_predictions
python -m analysis.aggregator --input results/predictions_all.csv --prefix all_
python -m analysis.event_analysis --input results/predictions_all.csv --prefix all_
```

### Шаг 8. Кластеризация

```bash
python -m analysis.cluster
```

K‑Means и DBSCAN, проекция t‑SNE. Сохраняется `results/clusters.csv`.

### Шаг 9. Визуализация

```bash
python -m vizualization.plotter_posts
python -m vizualization.plotter_comments
python -m vizualization.plotter_emoji
```

Генерируются все графики в папку `results/figures/`:

- динамика публикационной активности
- динамика тональности и взвешенный индекс S(t)
- тепловая карта каналов по месяцам
- расхождение государственных и общественных каналов
- t‑SNE визуализация кластеров
- матрицы ошибок и сравнение моделей
- влияние событий на тональность
- эмодзи‑реакции и Emoji Sentiment Index (ESI)

#### Примечание про эмодзи‑шрифты

Если эмодзи не отображаются и появляются предупреждения `findfont`, установите emoji‑шрифт и сбросьте кэш:

```bash
brew tap homebrew/cask-fonts
brew install --cask font-noto-emoji
rm -f ~/.matplotlib/fontlist-v*.json
rm -rf ~/.cache/matplotlib
```

Либо задайте путь явно через переменную окружения:

```bash
export EMOJI_FONT_PATH="/Library/Fonts/NotoEmoji-Regular.ttf"
```


### Шаг 10. Запуск всего пайплайна (опционально)

Если реализован `main.py`, можно выполнить:

```bash
python main.py
```
