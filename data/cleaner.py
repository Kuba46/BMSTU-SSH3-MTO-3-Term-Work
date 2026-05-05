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
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from config.settings import (
    RAW_DIR,
    RAW_CSV,
    CHANNELS,
    raw_csv_for,
)

log = logging.getLogger(__name__)

HARD_EXCLUSIONS: list[str] = [
    # ── Зарубежные события ────────────────────────────────────────────────────
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

    # ── Смерти и некрологи ────────────────────────────────────────────────────
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

    "кобзон",
    "зыкина",
    "магомаев",
    "лещенко",

    # ── Спорт ─────────────────────────────────────────────────────────────────
    "чемпионат мира по футболу",
    "лига чемпионов",
    "финал кубка",

    # ── Криминальные новости без связи с делом ────────────────────────────────
    "маньяк задержан",
    "серийный убийца",
    "теракт в",
]

CONTEXTUAL_EXCLUSIONS: list[tuple[str, set[str]]] = [
    ("трамп",           {"долин", "лурье", "квартир", "суд долин", "реституц"}),
    ("байден",          {"долин", "лурье", "квартир", "реституц"}),
    ("умер",            {"долин", "лурье", "квартир", "суд", "реституц", "покупател"}),
    ("скончал",         {"долин", "лурье", "квартир", "суд", "реституц"}),
    ("смерть",          {"долин", "лурье", "суд долин", "реституц", "покупател"}),
    ("похороны",        {"долин", "лурье", "квартир", "суд"}),
    ("верховный суд сша",{"долин", "лурье", "квартир"}),
]

ANCHORS: list[str] = [
    "квартир",
    "реституц",
    "добросовестный",
    "покупател",
    "02-0387",
    "хамовнически",
    "схема долин",
    "мошенник",
    "112 млн",
    "175 млн",
    "верховный суд",
    "иск",
    "судебное решение",
    "апелляц",
    "кассац",
]

# Минимальный score для автоматического принятия поста
# score = число попаданий по ANCHORS
MIN_ANCHOR_SCORE = 1


def _score(text: str) -> int:
    """Подсчитывает число якорных слов в тексте (признаки релевантности)."""
    t = text.lower()
    return sum(1 for anchor in ANCHORS if anchor in t)


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

    # 1. Жёсткие стоп-фразы
    hard = _has_hard_exclusion(text)
    if hard:
        return "remove", f"стоп-фраза: «{hard}»"

    # 2. Контекстные стоп-фразы
    contextual = _has_contextual_exclusion(text)
    if contextual:
        return "remove", f"контекстное исключение: «{contextual}»"

    # 3. Dominance check по якорным словам
    score = _score(text)
    if score >= MIN_ANCHOR_SCORE:
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

        # Показываем топ-5 причин удаления
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
) -> tuple[int, int]:
    """
    Очищает файл одного канала {username}_raw.csv.
    Сохраняет очищенный файл поверх оригинала.
    Нерелевантные посты сохраняет в {username}_removed.csv для аудита.

    Returns:
        (n_kept, n_removed)
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

    # Ручной просмотр пограничных случаев
    if with_review and not dry_run:
        df_clean = interactive_review(df_removed, df_clean)

    n_kept    = len(df_clean)
    n_removed = n_before - n_kept

    if not dry_run:
        # Перезаписываем оригинал очищенными данными
        df_clean.to_csv(raw_path, index=False, encoding="utf-8")

        # Сохраняем удалённые для аудита
        if not df_removed.empty:
            removed_path = RAW_DIR / f"{username}_removed.csv"
            df_removed.to_csv(removed_path, index=False, encoding="utf-8")
            log.info("  Удалённые → %s (%d строк)", removed_path.name, len(df_removed))

    log.info(
        "  @%s: было %d → осталось %d | удалено %d (%.1f%%)",
        username, n_before, n_kept, n_removed,
        n_removed / n_before * 100 if n_before else 0,
    )
    return n_kept, n_removed


def rebuild_summary(channels: list[str]) -> None:
    """
    Пересобирает сводный posts_raw.csv из очищенных файлов каналов.
    Вызывается после очистки всех каналов.
    """
    dfs = []
    for username in channels:
        path = raw_csv_for(username)
        if path.exists():
            dfs.append(pd.read_csv(path, encoding="utf-8"))

    if not dfs:
        log.warning("Нет очищенных файлов для сборки сводного.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(RAW_CSV, index=False, encoding="utf-8")
    log.info(
        "Сводный файл пересобран → %s (%d постов из %d каналов)",
        RAW_CSV, len(df_all), len(dfs),
    )


def run_pipeline(
    target_usernames: list[str] | None = None,
    dry_run: bool = False,
    with_review: bool = False,
) -> None:
    """
    Запускает очистку для всех каналов (или указанных).
    После очистки пересобирает сводный posts_raw.csv.
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

    for username in channels:
        kept, removed = clean_channel_file(
            username, dry_run=dry_run, with_review=with_review
        )
        total_kept    += kept
        total_removed += removed

    log.info(
        "\n%s\nИТОГО: оставлено %d | удалено %d | всего было %d\n%s",
        "═" * 50,
        total_kept, total_removed, total_kept + total_removed,
        "═" * 50,
    )

    if not dry_run:
        rebuild_summary(channels)


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