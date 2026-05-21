"""Telegram 群管理 bot — 接入 Claude 判别。

命令:
  /ping  — 验活
  /id    — 打印当前 chat/user id

行为:
  群里每条普通消息(非管理员、非 bot 自己),走 Claude 判别。
  根据 verdict.action 执行:删除/禁言/封禁/通知管理员/忽略。
  默认 DRY_RUN=true,只打日志不真删 — 先看判别合理后再放开。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from telegram import (
    ChatMemberUpdated,
    ChatPermissions,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import MessageEntityType
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    JobQueue,
    MessageHandler,
    filters,
)

from classifier import Verdict, classify

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() in ("1", "true", "yes")
ADMIN_USER_IDS = {
    int(x) for x in os.environ.get("ADMIN_USER_IDS", "").split(",") if x.strip()
}
GROUP_CHAT_ID = (
    int(os.environ["GROUP_CHAT_ID"]) if os.environ.get("GROUP_CHAT_ID") else None
)
NOTIFY_ADMIN_ID = next(iter(ADMIN_USER_IDS), None)  # DM 第一个管理员

claude = AsyncAnthropic()  # 自动读 ANTHROPIC_API_KEY

# 累犯升级:同一 user 2 小时内被自动删 ≥ 2 次,直接 ban
RECIDIVIST_WINDOW = timedelta(hours=2)
RECIDIVIST_THRESHOLD = 2  # >=2 次删触发 ban
_delete_history: dict[int, deque[datetime]] = defaultdict(deque)

# 入群验证
VERIFY_ENABLED = os.environ.get("VERIFY_ENABLED", "true").lower() in ("1", "true", "yes")
VERIFY_TIMEOUT_SECONDS = int(os.environ.get("VERIFY_TIMEOUT_SECONDS", "90"))
CAS_ENABLED = os.environ.get("CAS_ENABLED", "true").lower() in ("1", "true", "yes")
EMOJI_POOL = [
    "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼",
    "🐨", "🐯", "🦁", "🐮", "🐷", "🐸", "🐵", "🐔",
    "🦄", "🐝", "🐢", "🦋",
]
NO_PERMS = ChatPermissions(can_send_messages=False)
FULL_PERMS = ChatPermissions(
    can_send_messages=True,
    can_send_audios=True,
    can_send_documents=True,
    can_send_photos=True,
    can_send_videos=True,
    can_send_video_notes=True,
    can_send_voice_notes=True,
    can_send_polls=True,
    can_send_other_messages=True,
    can_add_web_page_previews=True,
    can_change_info=False,
    can_invite_users=True,
    can_pin_messages=False,
    can_manage_topics=False,
)

# 待验证用户:user_id -> {correct_emoji, chat_id, welcome_msg_id, ...}
_pending_verifications: dict[int, dict] = {}


async def _cas_check(user_id: int) -> bool:
    """查 CAS (combot anti-spam) 黑名单。True = 在黑名单。出错时返回 False(fail-open)。"""
    if not CAS_ENABLED:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"https://api.cas.chat/check?user_id={user_id}")
            data = r.json()
            return bool(data.get("ok"))
    except Exception:
        log.warning("CAS check failed for %s", user_id, exc_info=True)
        return False

# 数据日志(后续可用来挖正则预过滤规则)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _record_delete(user_id: int, when: datetime) -> int:
    """记录一次删除,返回窗口内累计次数(含本次)。"""
    history = _delete_history[user_id]
    cutoff = when - RECIDIVIST_WINDOW
    while history and history[0] < cutoff:
        history.popleft()
    history.append(when)
    return len(history)


def _log_verdict(update: Update, verdict: Verdict) -> None:
    """把每条判别写到 data/verdicts-YYYY-MM-DD.jsonl,后续挖规律用。"""
    msg = update.effective_message
    user = msg.from_user if msg else None
    chat = update.effective_chat
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "chat_id": chat.id if chat else None,
        "user_id": user.id if user else None,
        "user_name": user.full_name if user else None,
        "user_username": user.username if user else None,
        "user_is_bot": user.is_bot if user else None,
        "message_id": msg.message_id if msg else None,
        "text": msg.text if msg else None,
        "forwarded": bool(msg.forward_origin) if msg else False,
        "verdict": {
            "is_spam": verdict.is_spam,
            "category": verdict.category,
            "confidence": verdict.confidence,
            "reason": verdict.reason,
            "action": verdict.action,
        },
    }
    fname = DATA_DIR / f"verdicts-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    try:
        with fname.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("data log write failed: %s", e)


async def cmd_ping(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    mode = "DRY_RUN" if DRY_RUN else "LIVE"
    await update.effective_message.reply_text(f"pong ({mode})")


async def cmd_id(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    user = update.effective_user
    lines = [
        f"chat_id: `{chat.id}`",
        f"chat_type: `{chat.type}`",
        f"chat_title: {chat.title or '-'}",
        f"your user_id: `{user.id}`",
        f"your username: @{user.username or '-'}",
    ]
    await update.effective_message.reply_text(
        "\n".join(lines), parse_mode="Markdown"
    )


async def _notify_admin(context: ContextTypes.DEFAULT_TYPE, body: str) -> None:
    if NOTIFY_ADMIN_ID is None:
        return
    try:
        await context.bot.send_message(NOTIFY_ADMIN_ID, body, parse_mode="Markdown")
    except TelegramError as e:
        log.warning("notify admin failed: %s", e)


def _has_link(msg) -> bool:
    if not msg.entities:
        return False
    link_types = {
        MessageEntityType.URL,
        MessageEntityType.TEXT_LINK,
        MessageEntityType.MENTION,
        MessageEntityType.TEXT_MENTION,
    }
    return any(e.type in link_types for e in msg.entities)


async def _act(
    context: ContextTypes.DEFAULT_TYPE,
    update: Update,
    verdict: Verdict,
) -> None:
    chat = update.effective_chat
    msg = update.effective_message
    user = msg.from_user
    prefix = "[DRY] " if DRY_RUN else ""

    action_label = {
        "delete_ban": "删 + 封号",
        "delete_mute": "删 + 禁言 24h",
        "delete": "仅删",
        "flag": "通知管理员",
        "ignore": "忽略",
    }.get(verdict.action, verdict.action)

    log.warning(
        "%sVERDICT %s/%s conf=%.2f action=%s reason=%s | from=%s(@%s) text=%r",
        prefix,
        "SPAM" if verdict.is_spam else "OK",
        verdict.category,
        verdict.confidence,
        verdict.action,
        verdict.reason,
        user.full_name if user else "?",
        user.username if user else "?",
        msg.text[:200] if msg.text else "",
    )

    _log_verdict(update, verdict)

    if verdict.action == "ignore":
        return

    # 决定最终 action:累犯升级
    final_action = verdict.action
    upgraded_reason = ""
    if user and verdict.action in ("delete", "delete_mute") and not DRY_RUN:
        count = _record_delete(user.id, msg.date or datetime.now(timezone.utc))
        if count >= RECIDIVIST_THRESHOLD:
            final_action = "delete_ban"
            upgraded_reason = f"累犯升级:2 小时内 {count} 次被删 → 封号"

    # 通知管理员(always, 不管 dry run)
    sender = f"{user.full_name} (@{user.username or '?'}, id={user.id})" if user else "?"
    action_shown = {
        "delete_ban": "删 + 封号",
        "delete_mute": "删 + 禁言 24h",
        "delete": "仅删",
        "flag": "通知管理员",
    }.get(final_action, final_action)
    body = (
        f"*{prefix}{action_shown}*  `{verdict.category}` conf=`{verdict.confidence:.2f}`\n"
        f"原因: {verdict.reason}"
    )
    if upgraded_reason:
        body += f"\n⚠️ {upgraded_reason}"
    body += (
        f"\n发送人: {sender}\n"
        f"原文:\n```\n{msg.text[:500] if msg.text else ''}\n```"
    )
    await _notify_admin(context, body)

    if DRY_RUN or final_action == "flag":
        return

    # 真删
    try:
        await msg.delete()
    except TelegramError as e:
        log.warning("delete failed: %s", e)

    if user is None:
        return
    if final_action == "delete_mute":
        try:
            from telegram import ChatPermissions
            until = (msg.date or datetime.now(timezone.utc)) + timedelta(hours=24)
            await context.bot.restrict_chat_member(
                chat_id=chat.id,
                user_id=user.id,
                permissions=ChatPermissions(can_send_messages=False),
                until_date=until,
            )
        except TelegramError as e:
            log.warning("mute failed: %s", e)
    elif final_action == "delete_ban":
        try:
            await context.bot.ban_chat_member(chat_id=chat.id, user_id=user.id)
        except TelegramError as e:
            log.warning("ban failed: %s", e)


async def _kick(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, why: str) -> None:
    """踢人(ban + unban,可重进)。"""
    try:
        await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        await context.bot.unban_chat_member(chat_id=chat_id, user_id=user_id, only_if_banned=True)
        log.warning("verify: kicked user=%s reason=%s", user_id, why)
    except TelegramError as e:
        log.warning("verify: kick failed user=%s: %s", user_id, e)


async def _cleanup_verification(
    context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: int
) -> None:
    state = _pending_verifications.pop(user_id, None)
    if state and (mid := state.get("welcome_msg_id")):
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=mid)
        except TelegramError:
            pass
    if context.job_queue:
        for job in context.job_queue.get_jobs_by_name(f"verify_{chat_id}_{user_id}"):
            job.schedule_removal()


async def _verify_timeout(context: ContextTypes.DEFAULT_TYPE) -> None:
    """超时未验证 → 踢。"""
    data = context.job.data
    chat_id = data["chat_id"]
    user_id = data["user_id"]
    if user_id not in _pending_verifications:
        return  # 已通过/已失败
    await _kick(context, chat_id, user_id, "超时未验证")
    await _cleanup_verification(context, user_id, chat_id)


async def on_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ChatMemberHandler — 新人进群触发 CAS 查 + emoji 验证。"""
    if not VERIFY_ENABLED:
        return

    cmu: ChatMemberUpdated = update.chat_member
    chat = update.effective_chat

    if GROUP_CHAT_ID is not None and chat.id != GROUP_CHAT_ID:
        return

    if not (
        cmu.old_chat_member.status in ("left", "kicked")
        and cmu.new_chat_member.status == "member"
    ):
        return

    user = cmu.new_chat_member.user
    if user.is_bot:
        return
    if user.id in ADMIN_USER_IDS:
        return

    # 1. CAS 黑名单查询
    if await _cas_check(user.id):
        await _kick(context, chat.id, user.id, "CAS 黑名单命中")
        # DM 管理员
        if NOTIFY_ADMIN_ID:
            try:
                await context.bot.send_message(
                    NOTIFY_ADMIN_ID,
                    f"🚫 *CAS 命中*\n用户: {user.full_name} (@{user.username or '?'}, id={user.id})\n已自动踢出",
                    parse_mode="Markdown",
                )
            except TelegramError:
                pass
        return

    # 2. 立即禁言
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat.id, user_id=user.id, permissions=NO_PERMS
        )
    except TelegramError as e:
        log.warning("verify: mute failed for %s: %s", user.id, e)
        return

    # 3. 生成 3-选 emoji 题
    choices = random.sample(EMOJI_POOL, 3)
    correct = random.choice(choices)
    _pending_verifications[user.id] = {
        "chat_id": chat.id,
        "correct_emoji": correct,
        "welcome_msg_id": None,
    }

    # 4. 发欢迎 + 3 个按钮
    mention = user.mention_html()
    text = (
        f"👋 欢迎 {mention} 进群!\n\n"
        f"为了防止广告机器人,请在 <b>{VERIFY_TIMEOUT_SECONDS} 秒</b>内点击下面的 <b>{correct}</b>。\n"
        "点错或超时都会被请出群(可重进)。"
    )
    keyboard = [
        InlineKeyboardButton(emoji, callback_data=f"verify:{user.id}:{emoji}")
        for emoji in choices
    ]
    try:
        welcome = await context.bot.send_message(
            chat_id=chat.id,
            text=text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([keyboard]),
        )
    except TelegramError as e:
        log.warning("verify: welcome send failed: %s", e)
        _pending_verifications.pop(user.id, None)
        return

    _pending_verifications[user.id]["welcome_msg_id"] = welcome.message_id

    # 5. 排定超时
    if context.job_queue:
        context.job_queue.run_once(
            _verify_timeout,
            VERIFY_TIMEOUT_SECONDS,
            name=f"verify_{chat.id}_{user.id}",
            data={"chat_id": chat.id, "user_id": user.id},
        )
    log.info(
        "verify: started user=%s correct=%s choices=%s timeout=%ds",
        user.id, correct, choices, VERIFY_TIMEOUT_SECONDS,
    )


async def on_verify_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """CallbackQueryHandler — emoji 验证。"""
    query = update.callback_query
    if not query or not query.data.startswith("verify:"):
        return

    parts = query.data.split(":", 2)
    if len(parts) != 3:
        await query.answer("按钮数据异常", show_alert=True)
        return
    try:
        target_uid = int(parts[1])
    except ValueError:
        await query.answer("按钮数据异常", show_alert=True)
        return
    clicked_emoji = parts[2]

    if query.from_user.id != target_uid:
        await query.answer("这不是给你的验证,别瞎点", show_alert=True)
        return

    state = _pending_verifications.get(target_uid)
    if not state:
        await query.answer("验证已过期或已完成", show_alert=True)
        return

    chat_id = state["chat_id"]

    if clicked_emoji != state["correct_emoji"]:
        await query.answer(f"❌ 错了,应该点 {state['correct_emoji']}", show_alert=True)
        await _kick(context, chat_id, target_uid, f"点错 emoji({clicked_emoji} vs {state['correct_emoji']})")
        await _cleanup_verification(context, target_uid, chat_id)
        return

    # 通过
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat_id, user_id=target_uid, permissions=FULL_PERMS
        )
    except TelegramError as e:
        log.warning("verify: unmute failed: %s", e)
        await query.answer("解禁失败,联系管理员", show_alert=True)
        return

    await query.answer("✅ 验证通过,欢迎加入!", show_alert=False)
    await _cleanup_verification(context, target_uid, chat_id)
    log.info("verify: passed user=%s", target_uid)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None or not msg.text:
        return
    chat = update.effective_chat
    user = msg.from_user

    # 仅处理目标群
    if GROUP_CHAT_ID is not None and chat.id != GROUP_CHAT_ID:
        return

    # 管理员豁免(不包括匿名管理员 1087968824 — 那是占位虚拟账号)
    if user and user.id in ADMIN_USER_IDS:
        return

    # bot 自己豁免
    if user and user.is_bot:
        return

    log.info(
        "[%s/%s] %s(@%s): %s",
        chat.type,
        chat.id,
        user.full_name if user else "?",
        user.username if user else "?",
        msg.text[:200],
    )

    try:
        verdict = await classify(
            claude,
            text=msg.text,
            sender_name=user.full_name if user else "?",
            sender_username=user.username if user else None,
            has_link=_has_link(msg),
            is_forwarded=bool(msg.forward_origin),
        )
    except Exception:
        log.exception("classify failed — leaving message alone")
        return

    await _act(context, update, verdict)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("handler error", exc_info=context.error)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing in .env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing in .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
            on_message,
        )
    )
    app.add_handler(
        ChatMemberHandler(on_new_member, ChatMemberHandler.CHAT_MEMBER)
    )
    app.add_handler(CallbackQueryHandler(on_verify_click, pattern=r"^verify:\d+:"))
    app.add_error_handler(on_error)

    log.info(
        "bot starting | DRY_RUN=%s | VERIFY=%s(%ds) | CAS=%s | target_group=%s | admins=%s",
        DRY_RUN,
        VERIFY_ENABLED,
        VERIFY_TIMEOUT_SECONDS,
        CAS_ENABLED,
        GROUP_CHAT_ID,
        ADMIN_USER_IDS,
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
