"""
data/collector.py
=================
Асинхронный сборщик данных из Telegram-каналов через Telethon MTProto API.

Логика сбора:
    - Надёжное получение linked discussion group (3 стратегии fallback)
    - Комментарии собираются ТОЛЬКО под релевантными постами
    - Жёсткая фильтрация комментариев по дате (только период корпуса)
    - CSV с комментариями разбит на секции по постам через is_section_header

Запуск:
    python -m data.collector
    python -m data.collector --ch topor davankov
"""

import asyncio
import argparse
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from telethon import TelegramClient
from telethon.errors import (
    MsgIdInvalidError,
    ChannelPrivateError,
    ChatAdminRequiredError,
    FloodWaitError,
)
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.functions.messages import GetDiscussionMessageRequest
from telethon.tl.types import (
    Message,
    MessageReactions,
    ReactionEmoji,
)

from config.settings import (
    CHANNELS,
    CORPUS_START_DATE,
    CORPUS_END_DATE,
    RAW_CSV,
    RAW_DIR,
    raw_csv_for,
)

from config.telegram_credentials_local import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_SESSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_START_DT = datetime(
    CORPUS_START_DATE.year, CORPUS_START_DATE.month, CORPUS_START_DATE.day,
    tzinfo=timezone.utc,
)
_END_DT = datetime(
    CORPUS_END_DATE.year, CORPUS_END_DATE.month, CORPUS_END_DATE.day,
    23, 59, 59, tzinfo=timezone.utc,
)


def _cleanup_stale_session_journal() -> None:
    session_db = Path(f"{TELEGRAM_SESSION}.session")
    session_journal = Path(f"{TELEGRAM_SESSION}.session-journal")

    if not session_journal.exists() or not session_db.exists():
        return

    try:
        conn = sqlite3.connect(session_db, timeout=1)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        log.warning("Session DB занята другим процессом: %s", session_db)
        return

    session_journal.unlink(missing_ok=True)
    log.warning("Удалён stale session journal: %s", session_journal)

# Фильтрация релевантности

_KEYWORDS_SPECIFIC = {
    "долин", "лурье",
    "схема долин", "казус долин", "эффект долин",
    "бабушкина схема",
    "добросовестный покупател",
    "двусторонняя реституц",
    "02-0387",
    "народная артистка",
    "певица долин", "квартира долин",
    "полина лурье", "лариса долин",
    "175 млн", "112 млн",
    "хамовнически",
}

_KEYWORDS_CONTEXTUAL = [
    ("реституц",            {"долин", "лурье", "покупател", "продавец", "квартир"}),
    ("верховный суд",       {"долин", "лурье", "квартир", "реституц", "покупател"}),
    ("мосгорсуд",           {"долин", "лурье", "квартир"}),
    ("кассационн",          {"долин", "лурье", "квартир", "покупател"}),
    ("апелляц",             {"долин", "лурье", "квартир"}),
    ("мошенник",            {"долин", "лурье", "квартир", "певиц"}),
    ("телефонный мошенник", {"долин", "квартир", "певиц"}),
    ("недвижимост",         {"долин", "лурье", "реституц", "схема"}),
]


def _is_relevant(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.lower()
    if any(kw in t for kw in _KEYWORDS_SPECIFIC):
        return True
    for broad_kw, context_kws in _KEYWORDS_CONTEXTUAL:
        if broad_kw in t and any(ctx in t for ctx in context_kws):
            return True
    return False


def _in_period(dt: datetime | None) -> bool:
    if dt is None:
        return False
    return _START_DT <= dt <= _END_DT


# Парсинг реакций к посту — извлекаем общее количество и топ-3 эмодзи (если есть)
def _parse_reactions(reactions: MessageReactions | None) -> dict:
    """
    Docstring для _parse_reactions
    
    :param reactions: Описание
    :type reactions: MessageReactions | None
    :return: Описание
    :rtype: dict
    """
    if reactions is None or not reactions.results:
        return {"reactions_total": 0, "reactions_top": ""}
    total = 0
    counts = []
    for r in reactions.results:
        total += r.count
        if isinstance(r.reaction, ReactionEmoji):
            counts.append((r.reaction.emoticon, r.count))
    counts.sort(key=lambda x: x[1], reverse=True)
    return {
        "reactions_total": total,
        "reactions_top":   " ".join(e for e, _ in counts[:3]),
    }


# Linked discussion group
async def _get_discussion_peer(
    client: TelegramClient,
    channel_username: str,
) -> tuple[object | None, int | None]:
    """
    Три стратегии получения linked chat:
      1. GetFullChannelRequest -> linked_chat_id -> get_entity(id)
      2. Поиск по id в chats из ответа GetFullChannel (если get_entity упал)
      3. None, None — linked chat не существует
    """
    try:
        entity = await client.get_entity(channel_username)
        full   = await client(GetFullChannelRequest(entity))
        linked_id = getattr(full.full_chat, "linked_chat_id", None)

        if not linked_id:
            log.debug("@%s: linked_chat_id отсутствует", channel_username)
            return None, None

        # Стратегия 1: get_entity по id
        try:
            peer = await client.get_entity(linked_id)
            return peer, linked_id
        except Exception:
            pass

        # Стратегия 2: поиск в chats ответа
        for chat in getattr(full, "chats", []):
            if chat.id == abs(linked_id):
                log.debug("@%s: linked chat найден через chats[]", channel_username)
                return chat, linked_id

        return None, None

    except (ChannelPrivateError, ChatAdminRequiredError):
        log.warning("@%s: нет доступа к linked chat", channel_username)
        return None, None
    except Exception as exc:
        log.warning("@%s: ошибка получения discussion peer: %s", channel_username, exc)
        return None, None


# Комментарии к одному посту
async def _collect_comments_for_post(
    client: TelegramClient,
    channel_peer: object,
    discussion_peer: object,
    post_id: int,
    post_date: datetime,
    channel_username: str,
) -> list[dict]:
    """
    Собирает комментарии к одному релевантному посту.

    Ключевые правила:
      1. Только комментарии в период [CORPUS_START, CORPUS_END].
         Это исправляет @davankov: linked group хранит все треды
         с момента создания канала, включая 2023 год.
      2. Первая строка результата — секция-заголовок (is_section_header=True)
         для структурирования CSV файла.

    Формат секции в CSV:
      is_section_header=True  → "=== POST 90737 | 2025-12-16 | 42 комм. ==="
      is_section_header=False → обычный комментарий
    """
    if discussion_peer is None:
        return []

    collected: list[dict] = []

    try:
        discussion = await client(GetDiscussionMessageRequest(
            peer=channel_peer,
            msg_id=post_id,
        ))
        if not discussion or not discussion.messages:
            return []

        discussion_chat_id = getattr(discussion_peer, "id", None)
        root_msg_id = None
        for message in discussion.messages:
            peer_id = getattr(message, "peer_id", None)
            if getattr(peer_id, "channel_id", None) == discussion_chat_id:
                root_msg_id = message.id
                break

        if root_msg_id is None:
            return []

        async for msg in client.iter_messages(
            discussion_peer,
            reply_to=root_msg_id,
            limit=None,
        ):
            if not isinstance(msg, Message):
                continue
            if not _in_period(msg.date):
                continue

            collected.append({
                "channel_username": channel_username,
                "post_id":          post_id,
                "post_date":        post_date.isoformat(),
                "comment_id":       msg.id,
                "author_id":        getattr(msg.from_id, "user_id", None),
                "text":             msg.text or "",
                "date":             msg.date.isoformat() if msg.date else None,
                "views":            getattr(msg, "views", 0) or 0,
                "is_section_header": False,
            })

    except MsgIdInvalidError:
        pass  # у поста нет обсуждения — нормально
    except FloodWaitError as e:
        log.info("FloodWait (пост %d): ждём %ds", post_id, e.seconds)
        await asyncio.sleep(e.seconds + 1)
    except Exception as exc:
        log.warning("Ошибка комментариев к посту %d (@%s): %s",
                    post_id, channel_username, exc)
        return []

    if not collected:
        return []

    # Секция-заголовок перед комментариями поста
    header = {
        "channel_username": channel_username,
        "post_id":          post_id,
        "post_date":        post_date.isoformat(),
        "comment_id":       None,
        "author_id":        None,
        "text": (
            f"=== POST {post_id}"
            f" | {post_date.strftime('%Y-%m-%d %H:%M')}"
            f" | {len(collected)} комментариев ==="
        ),
        "date":             None,
        "views":            None,
        "is_section_header": True,
    }
    return [header] + collected


# Сбор одного канала
async def collect_channel(
    client: TelegramClient,
    channel_meta: dict,
) -> tuple[list[dict], list[dict]]:
    username      = channel_meta["username"]
    label         = channel_meta["label"]
    orientation   = channel_meta["orientation"]
    has_comments  = channel_meta["has_comments"]
    has_reactions = channel_meta["has_reactions"]

    log.info("▶ Канал: %s (@%s)", label, username)

    posts:    list[dict] = []
    comments: list[dict] = []
    skipped_rel  = 0
    skipped_date = 0

    # Получаем сущности один раз для всего канала
    channel_peer = None
    discussion_peer = None
    if has_comments:
        try:
            channel_peer = await client.get_entity(username)
        except Exception as exc:
            log.warning("  @%s: не удалось получить channel peer: %s", username, exc)

        discussion_peer, linked_id = await _get_discussion_peer(client, username)
        if discussion_peer is not None:
            log.info("  @%s: discussion group найдена (linked_id=%d)",
                     username, linked_id)
        else:
            log.info("  @%s: discussion group не найдена — "
                     "комментарии пропускаются.", username)

    # Итерируем посты от END к START
    async for msg in client.iter_messages(
        username,
        offset_date=_END_DT,
        reverse=False,
        limit=None,
    ):
        if not isinstance(msg, Message):
            continue

        msg_date = msg.date

        # Прерываем как только ушли раньше START
        if msg_date is not None and msg_date < _START_DT:
            log.debug("  @%s: достигли границы START (%s), прерываем.",
                      username, msg_date.date())
            break

        # Пропускаем посты позже END
        if msg_date is None or msg_date > _END_DT:
            skipped_date += 1
            continue

        text = msg.text or ""

        if not _is_relevant(text):
            skipped_rel += 1
            continue

        react = _parse_reactions(msg.reactions if has_reactions else None)

        posts.append({
            "channel_username": username,
            "channel_label":    label,
            "orientation":      orientation,
            "has_comments":     has_comments,
            "has_reactions":    has_reactions,
            "post_id":          msg.id,
            "text":             text,
            "date":             msg_date.isoformat(),
            "views":            getattr(msg, "views",    0) or 0,
            "forwards":         getattr(msg, "forwards", 0) or 0,
            **react,
        })

        # Комментарии — только под этим релевантным постом,
        # только за период корпуса
        if has_comments and discussion_peer is not None and channel_peer is not None:
            post_comments = await _collect_comments_for_post(
                client, channel_peer, discussion_peer,
                msg.id, msg_date, username,
            )
            comments.extend(post_comments)

    n_real_comments = sum(1 for c in comments if not c.get("is_section_header"))
    log.info(
        "  ✓ %s: постов=%d | нерелев.=%d | вне периода=%d | комментариев=%d",
        label, len(posts), skipped_rel, skipped_date, n_real_comments,
    )
    return posts, comments


# Сохранение
def _save_channel_files(
    username: str,
    posts: list[dict],
    comments: list[dict],
) -> None:
    """
    Сохраняет данные канала в два CSV-файла.
    posts:    data/raw/{username}_raw.csv
    comments: data/raw/{username}_comments_raw.csv

    Формат comments CSV:
        Строки с is_section_header=True — разделители секций между постами.
        Остальные строки — обычные комментарии.

    Пример чтения в pandas:
        df = pd.read_csv("shot_shot_comments_raw.csv")
        posts_sections = df[df["is_section_header"] == True]
        real_comments  = df[df["is_section_header"] == False]
    """
    if posts:
        path = raw_csv_for(username)
        pd.DataFrame(posts).to_csv(path, index=False, encoding="utf-8")
        log.info("    Посты → %s (%d строк)", path.name, len(posts))

    if comments:
        n_headers  = sum(1 for c in comments if c.get("is_section_header"))
        n_comments = len(comments) - n_headers
        path = RAW_DIR / f"{username}_comments_raw.csv"
        pd.DataFrame(comments).to_csv(path, index=False, encoding="utf-8")
        log.info(
            "    Комментарии → %s (%d строк в %d секциях)",
            path.name, n_comments, n_headers,
        )


# ── Главный пайплайн ──────────────────────────────────────────────────────────

async def run_collection(target_usernames: list[str] | None = None) -> None:
    channels_to_collect = [
        ch for ch in CHANNELS
        if target_usernames is None or ch["username"] in target_usernames
    ]

    if not channels_to_collect:
        log.error("Каналы не найдены. Проверьте --ch аргументы.")
        return

    log.info("Период: %s → %s | Каналов: %d", CORPUS_START_DATE, CORPUS_END_DATE, len(channels_to_collect))

    all_posts: list[dict] = []

    max_connect_attempts = 2
    for attempt in range(1, max_connect_attempts + 1):
        try:
            async with TelegramClient(
                TELEGRAM_SESSION, TELEGRAM_API_ID, TELEGRAM_API_HASH
            ) as client:
                await client.start()

                for ch_meta in channels_to_collect:
                    posts, comments = await collect_channel(client, ch_meta)

                    # Сохраняем сразу после каждого канала — защита от падений
                    _save_channel_files(ch_meta["username"], posts, comments)
                    all_posts.extend(posts)
            break

        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt >= max_connect_attempts:
                raise

            log.warning(
                "SQLite session заблокирована (%s). Попытка восстановления %d/%d...",
                exc,
                attempt,
                max_connect_attempts,
            )
            _cleanup_stale_session_journal()
            await asyncio.sleep(1)

    # Сводный файл
    if all_posts:
        df_new = pd.DataFrame(all_posts)

        if RAW_CSV.exists():
            df_existing = pd.read_csv(RAW_CSV, encoding="utf-8")
        else:
            df_existing = pd.DataFrame()

        updated_channels = {ch["username"] for ch in channels_to_collect}
        if not df_existing.empty and "channel_username" in df_existing.columns:
            df_existing = df_existing[
                ~df_existing["channel_username"].isin(updated_channels)
            ]

        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(
            subset=["channel_username", "post_id"], keep="last"
        )

        if "date" in df_all.columns:
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
            df_all = df_all.sort_values("date")

        df_all.to_csv(RAW_CSV, index=False, encoding="utf-8")

        stats = (
            df_all
            .groupby(["channel_label", "orientation"])
            .agg(
                постов    = ("post_id",        "count"),
                просмотров = ("views",          "sum"),
                реакций   = ("reactions_total", "sum"),
            )
            .reset_index()
            .sort_values("постов", ascending=False)
        )
        log.info("\nСводная статистика:\n%s", stats.to_string(index=False))
        log.info("\nСводный файл → %s (%d постов)", RAW_CSV, len(df_all))
    else:
        log.warning("Релевантных постов не найдено.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сбор постов и комментариев из Telegram-каналов."
    )
    parser.add_argument(
        "--ch", nargs="*", metavar="USERNAME",
        help="Собрать только эти каналы (без @). По умолчанию — все.",
    )
    args = parser.parse_args()
    asyncio.run(run_collection(args.ch))


if __name__ == "__main__":
    main()