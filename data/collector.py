"""
data/collector.py
=================
Асинхронный сборщик данных из Telegram-каналов через Telethon MTProto API.
 
Что собирает:
    - Посты канала за период CORPUS_START — CORPUS_END
    - Реакции под постами (суммарно и по типу эмодзи)
    - Комментарии (только для каналов с has_comments=True)
 
Запуск:
    python -m data.collector  # все каналы
    python -m data.collector --ch shot_shot rian_ru   # выборочно
"""

import argparse
import logging
import pandas as pd

from datetime import datetime, timezone

from telethon import TelegramClient
from telethon.tl.types import Message, MessageReactions, ReactionEmoji

from config.settings import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_SESSION,
    CHANNELS,
    CORPUS_START_DATE,
    CORPUS_END_DATE,
    RAW_CSV,
    KEYWORDS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

_START_DT = datetime(CORPUS_START_DATE.year, CORPUS_START_DATE.month, CORPUS_START_DATE.day, tzinfo=timezone.utc)
_END_DT   = datetime(CORPUS_END_DATE.year,   CORPUS_END_DATE.month,   CORPUS_END_DATE.day, 23, 59, 59, tzinfo=timezone.utc)


# Проверяем релевантность поста по наличию ключевых слов. Если текста нет — считаем нерелевантным.
def _is_relevant(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    return any(keyword in text for keyword in KEYWORDS)


# Парсим реакции к посту. Если реакций нет — возвращаем 0 и пустой топ. Иначе считаем общее количество и топ-3 эмодзи.
def _parse_reactions(reactions: MessageReactions | None) -> dict:
    if reactions is None or not reactions.results:
        return {"reactions_total": 0, "reactions_top": ""}
 
    total = 0
    counts = []
    for r in reactions.results:
        total += r.count
        if isinstance(r.reaction, ReactionEmoji):
            counts.append((r.reaction.emoticon, r.count))
 
    counts.sort(key=lambda x: x[1], reverse=True)
    top = " ".join(e for e, _ in counts[:3])
    return {"reactions_total": total, "reactions_top": top}


async def _collect_comments(client: TelegramClient, channel_username: str, post_id: int) -> list[dict]:
    comments = []
    try:
        async for message in client.iter_messages(channel_username, reply_to=post_id, limit=None):
            if not isinstance(message, Message):
                continue
            comments.append({
                "post_id": post_id,
                "comment_id": message.id,
                "comment_text": message.text or "",
                "comment_date": message.date.isoformat() if message.date else None,
                "views": getattr(message, "views", 0) or 0,
            })
    except Exception as e:
        log.warning(f"Ошибка при сборе комментариев для поста {post_id} в канале {channel_username}: {e}")
    return comments


# Собираем все релевантные посты одного канала за период корпуса. Возвращаем посты и комментарии.
async def collect_channel(
    client: TelegramClient,
    channel_meta: dict,
) -> tuple[list[dict], list[dict]]:
    username     = channel_meta["username"]
    label        = channel_meta["label"]
    orientation  = channel_meta["orientation"]
    has_comments = channel_meta["has_comments"]
    has_reactions = channel_meta["has_reactions"]
 
    log.info("▶ Канал: %s (@%s)", label, username)
 
    posts: list[dict]    = []
    comments: list[dict] = []
    skipped = 0
 
    async for msg in client.iter_messages(
        username,
        offset_date=_END_DT,
        reverse=False, # от новых к старым; прерываем, когда уходим за START
        limit=None,
    ):
        if not isinstance(msg, Message):
            continue
 
        # Прерываем обход, как только вышли за левую границу периода
        if msg.date and msg.date < _START_DT:
            break
 
        # Пропускаем посты вне правой границы (могут попасться при офсете)
        if msg.date and msg.date > _END_DT:
            continue
 
        text = msg.text or ""
 
        # Фильтрация по ключевым словам
        if not _is_relevant(text):
            skipped += 1
            continue
 
        react = _parse_reactions(msg.reactions if has_reactions else None)
 
        post_row = {
            "channel_username": username,
            "channel_label":    label,
            "orientation":      orientation,
            "has_comments":     has_comments,
            "has_reactions":    has_reactions,
            "post_id":          msg.id,
            "text":             text,
            "date":             msg.date.isoformat() if msg.date else None,
            "views":            getattr(msg, "views", 0) or 0,
            "forwards":         getattr(msg, "forwards", 0) or 0,
            **react,
        }
        posts.append(post_row)
 
        # Собираем комментарии (если канал их поддерживает)
        if has_comments:
            coms = await _collect_comments(client, username, msg.id)
            comments.extend(coms)
 
    log.info(
        "Выполнено! %s: собрано %d постов, пропущено %d нерелевантных, "
        "комментариев: %d",
        label, len(posts), skipped, len(comments),
    )
    return posts, comments
 

# Главный асинхронный сборщик. Если target_usernames задан — собираем только эти каналы.
async def run_collection(target_usernames: list[str] | None = None) -> None:
    channels_to_collect = [
        ch for ch in CHANNELS
        if target_usernames is None or ch["username"] in target_usernames
    ]
 
    if not channels_to_collect:
        log.error("Ни одного подходящего канала не найдено. Проверьте username-ы.")
        return
 
    async with TelegramClient(TELEGRAM_SESSION, TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
        await client.start()
        log.info("Сессия Telegram открыта. Сбор за период %s — %s.", CORPUS_START_DATE, CORPUS_END_DATE)

        all_posts:    list[dict] = []
        all_comments: list[dict] = []

        for ch_meta in channels_to_collect:
            posts, comments = await collect_channel(client, ch_meta)
            all_posts.extend(posts)
            all_comments.extend(comments)
    
    df_posts = pd.DataFrame(all_posts)
    df_posts.to_csv(RAW_CSV, index=False, encoding="utf-8")
    log.info("Посты сохранены → %s  (%d строк)", RAW_CSV, len(df_posts))

    if all_comments:
        from config.settings import RAW_DIR
        comments_csv = RAW_DIR / "comments_raw.csv"
        df_comments = pd.DataFrame(all_comments)
        df_comments.to_csv(comments_csv, index=False, encoding="utf-8")
        log.info("Комментарии сохранены → %s  (%d строк)", comments_csv, len(df_comments))

    if not df_posts.empty:
        stats = (
            df_posts.groupby(["channel_label", "orientation"])
            .agg(posts=("post_id", "count"), views=("views", "sum"))
            .reset_index()
            .sort_values("posts", ascending=False)
        )
        log.info("\nСтатистика по каналам:\n%s", stats.to_string(index=False))