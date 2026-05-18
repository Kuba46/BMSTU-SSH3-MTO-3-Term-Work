"""
data/cleaner.py
===============
Очистка корпуса после первичного сбора.

Три механизма очистки:
    1. HARD EXCLUSIONS — стоп-фразы: если пост содержит такую фразу,
    он удаляется безусловно (смерти артистов, зарубежные события, не связанные с делом и т.д.)
    2. DOMINANCE CHECK — доминирование темы: подсчитывает число попаданий
    по ключевым словам дела vs число стоп-слов; удаляет пост если нерелевантная тема явно доминирует
    3. MANUAL REVIEW — интерактивный просмотр пограничных случаев (посты с низким score) для ручного решения

Запуск:
    python -m data.cleaner                     # авто-очистка всего корпуса
    python -m data.cleaner --review            # + ручной просмотр пограничных
    python -m data.cleaner --ch shot_shot      # только один канал
    python -m data.cleaner --dry-run           # показать что будет удалено

Выход:
    data/cleaned/{username}_cleaned.csv
    data/cleaned/posts_cleaned.csv
    data/cleaned/{username}_comments_cleaned.csv
    data/cleaned/comments_cleaned.csv
    data/removed/{username}_removed.csv
    data/removed/{username}_comments_removed.csv
    (raw-файлы не изменяются)
"""

import argparse
import logging
import os
import re

import pandas as pd

from config.settings import (
    CLEANED_DIR,
    CLEANED_CSV,
    REMOVED_DIR,
    COMMENTS_CLEANED_CSV,
    CHANNELS,
    raw_csv_for,
    comments_raw_for,
    cleaned_csv_for,
    comments_cleaned_for,
    removed_csv_for,
    comments_removed_for,
)

log = logging.getLogger(__name__)

HARD_EXCLUSIONS: list[str] = [
    # Зарубежные события без связи с делом
    "чарли",
    "чарли кирк",
    "charlie kirk",
    "трамп",
    "байден",
    "конгресс сша",
    "верховный суд сша",
    "верховный суд израил",
    "верховный суд украин",
    "верховный суд германи",
    "верховный суд франци",

    # Смерти и некрологи без связи с делом
    "умер певец",
    "умерла певица",
    "скончался певец",
    "скончалась певица",
    "смерть певца",
    "смерть певицы",
    "похороны певца",
    "похороны певицы",
    "прощание с певц",
    "умер",
    "умерла",
    "скончался",
    "скончалась",
    "умер исполнитель",
    "умерла исполнительница",
    "смерть исполнител",
    "умер музыкант",
    "умерла музыкантша",
    "скончался музыкант",
    "прощание с артист",
    "кладбищ",
    "троекур",

    "кобзон",
    "зыкина",
    "магомаев",
    "лещенко",
    "голубкин",

    # Спорт
    "чемпионат мира по футболу",
    "лига чемпионов",
    "финал кубка",
    "роналду",
    "ан-наср",
    "футболистом-миллиардером",

    # Криминальные новости без связи с делом
    "маньяк задержан",
    "серийный убийца",
    "теракт в",
    "взрыве в газпромбанке",
    "газпромбанк",

    "ozon",
    "aliexpress",
    "китай",
    "blablacar",

    # Религиозные новости без связи с делом
    "пап римск",
    "свят престол",
    "кандидат на место папы",

    # Рекламные/стройка без связи с делом
    "школа № 1329",
    "гбоу школа № 1329",
    "жилые кварталы set",
    "жилые кварталы veer",
    "жк бизнес-класса",
    "девелопером mr",

    # География/топонимы с "долиной"
    "долина скаджит",
    "долине скаджит",
    "долина кашмир",
    "долине кашмир",
    "долина рой",
    "долина ройя",
    "долине рой",
    "долина царей",
    "долина цариц",
    "долине царей",
    "долине цариц",
    "долина смерти",
    "кремниевая долина",
    "долина кремния",
    "долина гейзеров",

    # Пропажи/розыск
    "пропал",
    "пропала",
    "пропавш",
    "пропажа",
    "разыскивают",
    "розыск",
    "бесследно",
    "поиски",
    "волонтёр",
    "волонтер",
    "потеря",

    "иноагент",
    "иностран агент",
]

PROMO_EXCLUSIONS: list[str] = [
    # Политическая/рекламная повестка без связи с делом
    "региональное отделение",
    "реготделение",
    "команда",
    "депутат",
    "депутаты",
    "выборы",
    "партия",
    "проект",
    "инициатив",
    "избирател",
    "кампания",
    "да — переменам",
    "да переменам",
    "голос города",
    "я в деле",
    "старательский фарт",
    "магадан",
    "камчатк",
    "долина гейзеров",

    "читайте самые интересные публикации",
    "можайск район",
    "школ",
]

NEWS_DIGEST_PHRASES: list[str] = [
    "итоги дня",
    "главное за день",
    "главные новости",
    "главные события",
    "сводка новостей",
    "сводка дня",
    "дайджест",
    "дайджест новостей",
    "утренняя сводка",
    "вечерняя сводка",
    "коротко о главном",
    "за сутки",
    "что известно к этому часу",
    "подборка новостей",
    "интересные публикации за неделю:",
    "главные материалы агентства к утру",
    "Главные материалы агентства",
    "читайте",
]

NEGATIVE_TOPICS: list[str] = [
    # Военные/боевые темы
    "войн",
    "спецоперац",
    "сво",
    "обстрел",
    "удар",
    "фронт",
    "дрон",
    "ракета",
    "минобороны",
    "вражеск",
    "всу",
    "украин",
    "одесс",
    "кишинев",
    "границ",
    "погранслужб",
    "днестр",
    "киев",
    "донецк",
    "луганск",
    "харьков",
    "запорож",
    "мариупол",
    # Криминальные сводки
    "убийств",
    "убит",
    "труп",
    "зарезал",
    "расстрел",
    "стрельб",
    "поножов",
    "ограблен",
    "краж",
    "мошенничеств",
    "похитил",
    "изнасилован",
    "террор",
    "теракт",
    "подстанц",
    "закладк",
    "дропер",
    "мессенджер",
    "whatsapp",
    "роскомнадзор",
]

CONTEXTUAL_EXCLUSIONS: list[tuple[str, set[str]]] = [
    ("умер",            {"долин", "лурье", "квартир", "суд", "реституц", "покупател"}),
    ("скончал",         {"долин", "лурье", "квартир", "суд", "реституц"}),
    ("смерть",          {"долин", "лурье", "суд долин", "реституц", "покупател"}),
    ("сво",             {"долин", "лурье", "квартир", "реституц", "суд"}),
    ("войн",            {"долин", "лурье", "квартир", "реституц", "суд"}),
    ("убийств",         {"долин", "лурье", "квартир", "реституц", "суд"}),
    ("труп",            {"долин", "лурье", "квартир", "реституц", "суд"}),
    ("whatsapp",        {"долин", "лурье"}),
    ("мессенджер",      {"долин", "лурье"}),
    ("роскомнадзор",    {"долин", "лурье"}),
    ("подстанц",        {"долин", "лурье"}),
    ("теракт",          {"долин", "лурье"}),
    ("дропер",          {"долин", "лурье"}),
    ("закладк",         {"долин", "лурье"}),
    ("главные новости", {"долин", "лурье"}),
    ("главные события", {"долин", "лурье"}),
    ("итоги дня",       {"долин", "лурье"}),
]

ANCHORS: list[str] = [
    "долин",
    "лурье",
    "реституц",
    "покупател",
    "02-0387",
    "хамовнически",
    "схема долин",
    "мошенник",
    "112 млн",
    "175 млн",
    "верховный суд",
    "судебное решение",
]

# Минимальный score для автоматического принятия поста
# score = число попаданий по ANCHORS
MIN_ANCHOR_SCORE = 1

_RE_HTTP_URL = re.compile(r"https?://\S+", flags=re.IGNORECASE)


def _sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _RE_HTTP_URL.sub(" ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score(text: str) -> int:
    """Подсчитывает число якорных слов в тексте (признаки релевантности)."""
    t = text.lower()
    score = 0
    for anchor in ANCHORS:
        if anchor not in t:
            continue
        if anchor == "долин" and any(x in t for x in ["долинск", "долинское", "долинский", "долинская"]):
            continue
        score += 1
    return score


def _has_hard_exclusion(text: str) -> str | None:
    """
    Возвращает первую найденную стоп-фразу или None.
    Проверка по точному вхождению (lower-case).
    """
    t = text.lower()
    for phrase in HARD_EXCLUSIONS:
        if phrase in t:
            return phrase
    return None


def _has_promo_exclusion(text: str) -> str | None:
    """
    Возвращает первую найденную промо-фразу или None.
    Проверка по точному вхождению (lower-case).
    """
    t = text.lower()
    for phrase in PROMO_EXCLUSIONS:
        if phrase in t:
            return phrase
    return None


def _has_contextual_exclusion(text: str) -> str | None:
    """
    Проверяет контекстные стоп-фразы.
    Возвращает фразу-исключение если она найдена И якоря отсутствуют.
    """
    t = text.lower()
    for phrase, required_anchors in CONTEXTUAL_EXCLUSIONS:
        if phrase in t:
            # Если хотя бы один якорь есть — пост остаётся
            if not any(anchor in t for anchor in required_anchors):
                return phrase
    return None


def _negative_score(text: str) -> int:
    t = text.lower()
    return sum(1 for phrase in NEGATIVE_TOPICS if phrase in t)


def _is_news_digest(text: str) -> bool:
    t = text.lower()
    if any(phrase in t for phrase in NEWS_DIGEST_PHRASES):
        return True
    # Много пунктов/маркеров — типичная сводка
    bullet_lines = re.findall(r"(?m)^\s*[-•—]\s", text)
    if len(bullet_lines) >= 3:
        return True
    # Много коротких строк с двоеточиями ("Тема: описание")
    colon_lines = [
        line for line in text.splitlines()
        if ":" in line and 0 < len(line.strip()) <= 120
    ]
    if len(colon_lines) >= 4:
        return True
    return False


def classify_post(text: str) -> tuple[str, str]:
    """
    Классифицирует пост по трём категориям:
      'keep'    — релевантный, оставляем
      'remove'  — нерелевантный, удаляем
      'review'  — пограничный, нужен ручной просмотр

    Returns:
        (decision, reason) — решение и причина
    """
    if not isinstance(text, str) or not text.strip():
        return "remove", "пустой текст"

    score = _score(text)

    # 1. Жёсткие стоп-фразы
    hard = _has_hard_exclusion(text)
    if hard:
        return "remove", f"стоп-фраза: «{hard}»"

    # 1.1 Промо/агитация — удаляем только если нет якорей
    promo = _has_promo_exclusion(text)
    if promo and score == 0:
        return "remove", f"промо-стоп-фраза: «{promo}»"

    # 2. Контекстные стоп-фразы
    contextual = _has_contextual_exclusion(text)
    if contextual:
        return "remove", f"контекстное исключение: «{contextual}»"

    # 2.1 Сводки/дайджесты — удаляем даже при единичном якоре
    if _is_news_digest(text) and score <= 1:
        return "remove", "сводка/дайджест новостей"

    # 3. Dominance check по якорным словам
    negative = _negative_score(text)
    if score >= MIN_ANCHOR_SCORE:
        if negative >= max(2, score + 1):
            return "remove", f"доминирование нерелевантной темы (neg={negative}, anchors={score})"
        return "keep", f"якорей: {score}"

    # 4. Пограничный случай — нет ни стоп-фраз ни якорей
    return "review", "нет якорных слов, нет стоп-фраз — требует просмотра"


def clean_dataframe(
    df: pd.DataFrame,
    dry_run: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Применяет очистку ко всему DataFrame.

    Args:
        df:       DataFrame с колонкой 'text'
        dry_run:  если True — не удаляет, только помечает
        verbose:  логировать детали
    Returns:
        (df_clean, df_removed) — очищенный и удалённый датафреймы
    """
    df = df.copy()
    df["_decision"] = ""
    df["_reason"]   = ""

    for idx, row in df.iterrows():
        decision, reason = classify_post(str(row.get("text", "")))
        df.at[idx, "_decision"] = decision
        df.at[idx, "_reason"]   = reason

    keep_mask   = df["_decision"] == "keep"
    remove_mask = df["_decision"] == "remove"
    review_mask = df["_decision"] == "review"

    n_keep   = keep_mask.sum()
    n_remove = remove_mask.sum()
    n_review = review_mask.sum()

    if verbose:
        log.info(
            "Результат очистки: оставить=%d | удалить=%d | на проверку=%d | итого=%d",
            n_keep, n_remove, n_review, len(df),
        )
        if n_remove > 0:
            reasons = df[remove_mask]["_reason"].value_counts().head(10)
            log.info("Топ причин удаления:\n%s", reasons.to_string())

    df_removed = df[remove_mask | review_mask].copy()
    df_clean   = df[keep_mask].copy()
    df_clean = df_clean.drop(columns=["_decision", "_reason"])
    return df_clean, df_removed


def interactive_review(
    df_review: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Показывает пограничные посты для ручного решения.
    Оператор выбирает: оставить (k), удалить (d), пропустить (s), выйти (q).

    Returns:
        Обновлённый df_clean с добавленными вручную одобренными постами.
    """
    review_only = df_review[df_review["_decision"] == "review"].reset_index(drop=True)

    if review_only.empty:
        log.info("Нет пограничных постов для ручного просмотра.")
        return df_clean

    approved = []
    print(f"\n{'═'*65}")
    print(f"  РУЧНОЙ ПРОСМОТР: {len(review_only)} пограничных постов")
    print(f"  [k] оставить  [d] удалить  [s] пропустить  [q] выйти")
    print(f"{'═'*65}")

    for idx, row in review_only.iterrows():
        os.system("cls" if os.name == "nt" else "clear")
        print(f"\n  Пост {idx + 1} / {len(review_only)}")
        print(f"  Канал: {row.get('channel_label', '?')} | Дата: {str(row.get('date', ''))[:10]}")
        print(f"  Причина: {row.get('_reason', '—')}")
        print(f"  Просм.: {int(row.get('views', 0)):,}  | "
              f"Реакции: {int(row.get('reactions_total', 0)):,}")
        print(f"\n{'─'*65}")
        text = str(row.get("text", ""))
        print(text[:600] + ("\n[...обрезано...]" if len(text) > 600 else ""))
        print(f"{'─'*65}")

        while True:
            key = input("  Решение: ").strip().lower()
            if key == "k":
                clean_row = row.drop(labels=["_decision", "_reason"], errors="ignore")
                approved.append(clean_row.to_dict())
                print("  → ОСТАВЛЕН")
                break
            elif key == "d":
                print("  → УДАЛЁН")
                break
            elif key == "s":
                print("  → Пропущен")
                break
            elif key == "q":
                print("\n  Выход из ручного просмотра.")
                if approved:
                    df_approved = pd.DataFrame(approved)
                    df_clean = pd.concat([df_clean, df_approved], ignore_index=True)
                return df_clean
            else:
                print("  Используйте: k / d / s / q")

    if approved:
        df_approved = pd.DataFrame(approved)
        df_clean = pd.concat([df_clean, df_approved], ignore_index=True)
        log.info("Вручную одобрено дополнительно: %d постов", len(approved))

    return df_clean


def clean_channel_file(
    username: str,
    dry_run: bool = False,
    with_review: bool = False,
) -> tuple[int, int, set[int]]:
    """
    Очищает файл одного канала {username}_raw.csv.
    Сохраняет очищенный файл в data/cleaned/{username}_cleaned.csv.
    Исходные raw-файлы не изменяются.

    Returns:
        (n_kept, n_removed, removed_post_ids)
    """
    raw_path = raw_csv_for(username)
    if not raw_path.exists():
        log.warning("Файл не найден, пропускаем: %s", raw_path)
        return 0, 0

    df = pd.read_csv(raw_path, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    n_before = len(df)

    log.info("── Очистка @%s (%d постов) ──", username, n_before)

    df_clean, df_removed = clean_dataframe(df, dry_run=dry_run)

    if not df_clean.empty:
        df_clean["text"] = df_clean["text"].apply(_sanitize_text)

    # Ручной просмотр пограничных случаев
    if with_review and not dry_run:
        df_clean = interactive_review(df_removed, df_clean)

    n_kept    = len(df_clean)
    n_removed = n_before - n_kept
    removed_post_ids = set(df["post_id"]) - set(df_clean["post_id"])

    if not dry_run:
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        REMOVED_DIR.mkdir(parents=True, exist_ok=True)
        cleaned_path = cleaned_csv_for(username)
        df_clean.to_csv(cleaned_path, index=False, encoding="utf-8")
        log.info("  Очищенные → %s (%d строк)", cleaned_path.name, len(df_clean))

        removed_path = removed_csv_for(username)
        df_removed.to_csv(removed_path, index=False, encoding="utf-8")
        log.info("  Удалённые → %s (%d строк)", removed_path.name, len(df_removed))

    log.info(
        "  @%s: было %d → осталось %d | удалено %d (%.1f%%)",
        username, n_before, n_kept, n_removed,
        n_removed / n_before * 100 if n_before else 0,
    )
    return n_kept, n_removed, removed_post_ids


def clean_comments_for_channel(
    username: str,
    removed_post_ids: set[int],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Удаляет комментарии, относящиеся к удалённым постам.
    Сохраняет очищенные комментарии в data/cleaned и удалённые в data/removed.
    """
    comments_path = comments_raw_for(username)
    if not comments_path.exists():
        return 0, 0

    df = pd.read_csv(comments_path, encoding="utf-8")
    if "post_id" not in df.columns:
        return 0, 0

    n_before = len(df)
    if removed_post_ids:
        mask_remove = df["post_id"].isin(removed_post_ids)
    else:
        mask_remove = pd.Series(False, index=df.index)

    df_removed = df[mask_remove].copy()
    df_clean = df[~mask_remove].copy()

    if not dry_run:
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        REMOVED_DIR.mkdir(parents=True, exist_ok=True)
        cleaned_path = comments_cleaned_for(username)
        df_clean.to_csv(cleaned_path, index=False, encoding="utf-8")
        removed_path = comments_removed_for(username)
        df_removed.to_csv(removed_path, index=False, encoding="utf-8")
        log.info("  Комментарии удалённые → %s (%d строк)", removed_path.name, len(df_removed))
        log.info("  Комментарии очищенные → %s (%d строк)", cleaned_path.name, len(df_clean))

    return len(df_clean), len(df_removed)


def rebuild_summary(channels: list[str]) -> None:
    """
    Пересобирает сводный posts_cleaned.csv из очищенных файлов каналов.
    Вызывается после очистки всех каналов.
    """
    dfs = []
    for username in channels:
        path = cleaned_csv_for(username)
        if path.exists():
            dfs.append(pd.read_csv(path, encoding="utf-8"))

    if not dfs:
        log.warning("Нет очищенных файлов для сборки сводного.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(CLEANED_CSV, index=False, encoding="utf-8")
    log.info(
        "Сводный файл пересобран → %s (%d постов из %d каналов)",
        CLEANED_CSV, len(df_all), len(dfs),
    )


def rebuild_comments_summary(channels: list[str]) -> None:
    """
    Пересобирает сводный comments_cleaned.csv из очищенных файлов комментариев.
    """
    dfs = []
    for username in channels:
        path = comments_cleaned_for(username)
        if path.exists():
            dfs.append(pd.read_csv(path, encoding="utf-8"))

    if not dfs:
        log.warning("Нет очищенных комментариев для сборки сводного.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(COMMENTS_CLEANED_CSV, index=False, encoding="utf-8")
    log.info(
        "Сводный файл комментариев пересобран → %s (%d строк из %d каналов)",
        COMMENTS_CLEANED_CSV, len(df_all), len(dfs),
    )


def run_pipeline(
    target_usernames: list[str] | None = None,
    dry_run: bool = False,
    with_review: bool = False,
) -> None:
    """
    Запускает очистку для всех каналов (или указанных).
    После очистки пересобирает сводный posts_cleaned.csv.
    """
    channels = [
        ch["username"] for ch in CHANNELS
        if target_usernames is None or ch["username"] in target_usernames
    ]

    if not channels:
        log.error("Каналы не найдены. Проверьте --ch аргументы.")
        return

    if dry_run:
        log.info("DRY-RUN режим: файлы не будут изменены.")

    total_kept    = 0
    total_removed = 0
    removed_by_channel: dict[str, set[int]] = {}

    for username in channels:
        kept, removed, removed_post_ids = clean_channel_file(
            username, dry_run=dry_run, with_review=with_review
        )
        total_kept    += kept
        total_removed += removed
        removed_by_channel[username] = removed_post_ids

    log.info(
        "\n%s\nИТОГО: оставлено %d | удалено %d | всего было %d\n%s",
        "═" * 50,
        total_kept, total_removed, total_kept + total_removed,
        "═" * 50,
    )

    if not dry_run:
        rebuild_summary(channels)

        total_comments_removed = 0
        total_comments_kept = 0
        for username in channels:
            kept_c, removed_c = clean_comments_for_channel(
                username, removed_by_channel.get(username, set()), dry_run=dry_run
            )
            total_comments_kept += kept_c
            total_comments_removed += removed_c
        rebuild_comments_summary(channels)
        log.info(
            "Комментариев: оставлено %d | удалено %d",
            total_comments_kept, total_comments_removed,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Очистка корпуса от нерелевантных постов."
    )
    parser.add_argument(
        "--ch", nargs="*", metavar="USERNAME",
        help="Очистить только указанные каналы (без @). По умолчанию — все.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Показать что будет удалено без изменения файлов.",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Запустить ручной просмотр пограничных постов.",
    )
    args = parser.parse_args()
    run_pipeline(
        target_usernames=args.ch,
        dry_run=args.dry_run,
        with_review=args.review,
    )


if __name__ == "__main__":
    main()
