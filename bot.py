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
from openai import AsyncOpenAI
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query as claude_query,
)
from dotenv import load_dotenv
from telegram import (
    ChatMemberUpdated,
    ChatPermissions,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, MessageEntityType
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

from classifier import LLMRouter, Verdict

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

# LLM provider 路由 —
#   - 显式 LLM_PROVIDER=claude / openai → 单 provider,无 fallback
#   - 未显式设(默认):双 key 都配 → openai primary(便宜,DeepSeek 等)+ claude fallback
#                      只 openai key   → openai 单(无 fallback)
#                      只 claude key   → claude 单(无 fallback)
#                      都没           → SystemExit
def _make_claude() -> AsyncAnthropic:
    return AsyncAnthropic()  # 自动读 ANTHROPIC_API_KEY


def _make_openai() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,  # 空 → openai.com 官方
    )


_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").lower()
_has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
_has_openai = bool(os.environ.get("OPENAI_API_KEY"))

if _LLM_PROVIDER == "claude":
    if not _has_anthropic:
        raise SystemExit("ANTHROPIC_API_KEY missing(LLM_PROVIDER=claude)")
    llm_router = LLMRouter(_make_claude(), fallback=None)
elif _LLM_PROVIDER == "openai":
    if not _has_openai:
        raise SystemExit("OPENAI_API_KEY missing(LLM_PROVIDER=openai)")
    llm_router = LLMRouter(_make_openai(), fallback=None)
elif _LLM_PROVIDER == "":
    # 自动 — 双 key → openai 主 / claude 备;单 key → 单 provider
    if _has_openai and _has_anthropic:
        llm_router = LLMRouter(_make_openai(), fallback=_make_claude())
    elif _has_openai:
        llm_router = LLMRouter(_make_openai(), fallback=None)
    elif _has_anthropic:
        llm_router = LLMRouter(_make_claude(), fallback=None)
    else:
        raise SystemExit("需 ANTHROPIC_API_KEY 或 OPENAI_API_KEY 至少一个")
else:
    raise SystemExit(f"unknown LLM_PROVIDER={_LLM_PROVIDER!r},需 claude / openai / 空")

# 累犯升级:同一 user 2 小时内被自动删 ≥ 2 次,直接 ban
RECIDIVIST_WINDOW = timedelta(hours=2)
RECIDIVIST_THRESHOLD = 2  # >=2 次删触发 ban
_delete_history: dict[int, deque[datetime]] = defaultdict(deque)

# delete_ban 时同步清该用户近 24h 全部消息(包括之前 classify 没拦住的广告)。
# 内存 dict 重启丢,接受 — 24h 窗口 + bot 通常长跑。
USER_HISTORY_WINDOW = timedelta(hours=24)
_user_messages: dict[int, deque[tuple[int, datetime]]] = defaultdict(deque)

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


def _record_user_message(user_id: int, message_id: int, when: datetime) -> None:
    """记录用户每条非命令文本消息 + 自动淘汰 > USER_HISTORY_WINDOW 的老条目。"""
    history = _user_messages[user_id]
    cutoff = when - USER_HISTORY_WINDOW
    while history and history[0][1] < cutoff:
        history.popleft()
    history.append((message_id, when))


async def _purge_user_recent_messages(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int
) -> int:
    """delete_ban 时调用,删该用户近 USER_HISTORY_WINDOW 内全部 bot 看到的消息。
    返删除成功条数。Telegram 普通 bot 仅能删 < 48h 消息,super group + admin bot 可删该 user 任意 msg_id;
    单条 delete_message 失败(老 / 已删 / 权限)silent skip。"""
    history = _user_messages.pop(user_id, None)
    if not history:
        return 0
    deleted = 0
    for mid, _ in list(history):
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=mid)
            deleted += 1
        except TelegramError:
            pass
    return deleted


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


# === Admin private chat — Claude Agent SDK with full tools(Bash/Read/Write/Edit/etc) ===
# 验证(两层):
#   1. user.id ∈ ADMIN_USER_IDS
#   2. TOTP /auth 通过 → 24h session,每次活跃自动续期 24h(/logout 主动结束)
# permission_mode="bypassPermissions" = 完全跳过 y/n 确认,等于 telegram admin 远控 botuser 账号
# 全部能力(含 NOPASSWD sudo)。session_id 维持多轮;/reset 清当前 user 的 claude session。

import pyotp

_TOTP_SECRET_FILE = Path(__file__).parent / ".totp-secret"
_TOTP_LOCK_MAX_FAILS = 5
_TOTP_LOCK_DURATION = timedelta(minutes=15)
_SESSION_DURATION = timedelta(hours=24)
_BOT_NAME = "telegram-bot"


def _load_or_create_totp_secret() -> str:
    """启动时调用。文件存在 → 返已存 secret;不存在 → 生成 + 写 600 perm + 返。"""
    if _TOTP_SECRET_FILE.exists():
        return _TOTP_SECRET_FILE.read_text().strip()
    secret = pyotp.random_base32()
    _TOTP_SECRET_FILE.write_text(secret)
    _TOTP_SECRET_FILE.chmod(0o600)
    return secret


_TOTP_SECRET = _load_or_create_totp_secret()
_TOTP_NEEDS_SETUP_DM = not (Path(__file__).parent / ".totp-setup-done").exists()


def _make_totp_provisioning_uri() -> str:
    issuer = _BOT_NAME
    label = f"{issuer}:admin"
    return pyotp.totp.TOTP(_TOTP_SECRET).provisioning_uri(name=label, issuer_name=issuer)


# Claude Agent session(per user_id 续多轮)
_chat_sessions: dict[int, str] = {}
# Auth session:user_id -> last_active datetime;过期(now - last_active > 24h)= 需重 /auth
_auth_sessions: dict[int, datetime] = {}
# TOTP 失败 lock:user_id -> (fail_count, lock_until)
_totp_fails: dict[int, tuple[int, datetime | None]] = {}
# Chat model 选择(per user_id);None = SDK 默认(跟 Claude Code 配置)
_user_chat_model: dict[int, str] = {}


# === Session 状态持久化(bot 重启不丢 auth / 对话 / 模型选择) ===
_STATE_FILE = Path(__file__).parent / ".session-state.json"


def _load_state() -> None:
    """启动时调一次。文件不存在 / 损坏 → 静默从空开始。"""
    if not _STATE_FILE.exists():
        return
    try:
        data = json.loads(_STATE_FILE.read_text())
        for uid_s, iso in (data.get("auth_sessions") or {}).items():
            _auth_sessions[int(uid_s)] = datetime.fromisoformat(iso)
        for uid_s, sid in (data.get("chat_sessions") or {}).items():
            _chat_sessions[int(uid_s)] = sid
        for uid_s, model in (data.get("user_chat_model") or {}).items():
            _user_chat_model[int(uid_s)] = model
        for uid_s, item in (data.get("totp_fails") or {}).items():
            count, lock_iso = item
            _totp_fails[int(uid_s)] = (
                count,
                datetime.fromisoformat(lock_iso) if lock_iso else None,
            )
        log.info(
            "session state loaded: auth=%d chat=%d model=%d",
            len(_auth_sessions), len(_chat_sessions), len(_user_chat_model),
        )
    except Exception as e:
        log.warning("session state load failed (ignored, fresh start): %s", e)


def _save_state() -> None:
    """状态变更后调。简单同步 write,文件几 KB 无 perf 问题。"""
    try:
        data = {
            "auth_sessions": {str(k): v.isoformat() for k, v in _auth_sessions.items()},
            "chat_sessions": {str(k): v for k, v in _chat_sessions.items()},
            "user_chat_model": {str(k): v for k, v in _user_chat_model.items()},
            "totp_fails": {
                str(k): [c, (lu.isoformat() if lu else None)]
                for k, (c, lu) in _totp_fails.items()
            },
        }
        # 原子写:tmp + rename(防止 crash 时 partial file)
        tmp = _STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.chmod(0o600)
        tmp.replace(_STATE_FILE)
    except Exception as e:
        log.warning("session state save failed: %s", e)


_load_state()

# 预设可选 model 列表(`/model` 列出 + key 短输入)
_CHAT_MODEL_PRESETS = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


def _is_authed(user_id: int) -> bool:
    last = _auth_sessions.get(user_id)
    if last is None:
        return False
    return (datetime.now(timezone.utc) - last) < _SESSION_DURATION


def _touch_session(user_id: int) -> None:
    """活跃 → 重置 last_active(续期 24h)"""
    _auth_sessions[user_id] = datetime.now(timezone.utc)
    _save_state()

_AGENT_SYSTEM_PROMPT = """\
你是 telegram 群管理 bot 内嵌的助手,通过私聊跟管理员(botuser)交互。
你跑在管理员的 Ubuntu 机器上,拥有完整 Claude Code tools(Bash / Read / Write / Edit 等)。
回复尽量简洁、用 plain text(Telegram 不解析 markdown),避免过长输出。
"""


async def cmd_reset(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None or user.id not in ADMIN_USER_IDS:
        await update.effective_message.reply_text("权限不足。")
        return
    had = _chat_sessions.pop(user.id, None)
    if had:
        _save_state()
    txt = "✅ 会话已清,下条消息开新对话。" if had else "(本来就没活跃会话)"
    await update.effective_message.reply_text(txt)


async def cmd_auth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/auth <6 位 TOTP>` — 验证后开 24h session,活跃自动续期。"""
    user = update.effective_user
    msg = update.effective_message
    if user is None or user.id not in ADMIN_USER_IDS:
        await msg.reply_text("权限不足。")
        return

    # check lock
    fails, lock_until = _totp_fails.get(user.id, (0, None))
    now = datetime.now(timezone.utc)
    if lock_until and now < lock_until:
        remaining = int((lock_until - now).total_seconds() / 60) + 1
        await msg.reply_text(f"🔒 失败次数过多,锁定 {remaining} 分钟后重试。")
        return

    # 解析 /auth <code>
    parts = (msg.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await msg.reply_text("用法:`/auth <6 位数字>`(从 Authenticator app 看)", parse_mode="Markdown")
        return
    code = parts[1].strip()

    # 立即删 telegram 含 code 的消息(防 history 泄漏)
    try:
        await msg.delete()
    except TelegramError:
        pass

    # 验
    totp = pyotp.TOTP(_TOTP_SECRET)
    if totp.verify(code, valid_window=1):  # ±30s 容忍时钟漂移
        _touch_session(user.id)
        _totp_fails.pop(user.id, None)
        _save_state()
        await context.bot.send_message(
            user.id,
            f"✅ 验证通过。24h session 已开,每次发消息自动续 24h。\n`/logout` 主动结束。",
            parse_mode="Markdown",
        )
        return

    # 失败
    fails += 1
    if fails >= _TOTP_LOCK_MAX_FAILS:
        _totp_fails[user.id] = (0, now + _TOTP_LOCK_DURATION)
        _save_state()
        await context.bot.send_message(
            user.id,
            f"❌ 验证失败 {fails} 次,锁定 15 分钟。",
        )
        log.warning("totp lock user=%s", user.id)
    else:
        _totp_fails[user.id] = (fails, None)
        _save_state()
        await context.bot.send_message(
            user.id,
            f"❌ 验证失败({fails}/{_TOTP_LOCK_MAX_FAILS})。",
        )


async def cmd_logout(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    msg = update.effective_message
    if user is None or user.id not in ADMIN_USER_IDS:
        await msg.reply_text("权限不足。")
        return
    had = _auth_sessions.pop(user.id, None)
    _chat_sessions.pop(user.id, None)  # 顺便清 claude session
    _save_state()
    txt = "✅ 已 logout,session + 对话历史已清。" if had else "(本来就没 session)"
    await msg.reply_text(txt)


async def cmd_model(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """`/model` 显当前;`/model <preset|full-name>` 切换。仅 admin。"""
    user = update.effective_user
    msg = update.effective_message
    if user is None or user.id not in ADMIN_USER_IDS:
        await msg.reply_text("权限不足。")
        return

    parts = (msg.text or "").split(maxsplit=1)
    current = _user_chat_model.get(user.id) or "(默认,Claude Code 配置)"

    if len(parts) < 2:
        # 列预设 + 当前
        preset_lines = "\n".join(f"  `/model {k}` → `{v}`" for k, v in _CHAT_MODEL_PRESETS.items())
        await msg.reply_text(
            f"当前 chat model: `{current}`\n\n预设(切换):\n{preset_lines}\n\n"
            f"或自定:`/model <完整 model 名>` 或 `/model reset` 恢复默认。",
            parse_mode="Markdown",
        )
        return

    arg = parts[1].strip()
    if arg == "reset" or arg == "default":
        _user_chat_model.pop(user.id, None)
        _save_state()
        await msg.reply_text("✅ 已恢复默认 model。")
        return

    target = _CHAT_MODEL_PRESETS.get(arg, arg)  # 预设 key 或直接 full name
    _user_chat_model[user.id] = target
    _save_state()
    await msg.reply_text(f"✅ chat model 切为 `{target}`。", parse_mode="Markdown")


async def _send_long(message, text: str) -> None:
    """Telegram 单条 4096 chars 上限,长输出分段。"""
    CHUNK = 4000
    if not text:
        text = "(空响应)"
    for i in range(0, len(text), CHUNK):
        await message.reply_text(text[i : i + CHUNK])


_UPLOADS_DIR = Path(__file__).parent / "data" / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


async def _download_telegram_attachment(
    context: ContextTypes.DEFAULT_TYPE, msg, user_id: int
) -> tuple[Path | None, str]:
    """私聊图片 / 文件 → 下载到 data/uploads/。
    返 (本地 path, 描述文本)。无附件返 (None, "")。
    """
    file_obj = None
    suffix = ""
    label = ""

    if msg.photo:  # PhotoSize list,取最大分辨率
        file_obj = await context.bot.get_file(msg.photo[-1].file_id)
        suffix = ".jpg"
        label = "图片"
    elif msg.document:
        file_obj = await context.bot.get_file(msg.document.file_id)
        suffix = Path(msg.document.file_name or "").suffix or ".bin"
        label = f"文件({msg.document.file_name or '?'})"
    elif msg.video:
        file_obj = await context.bot.get_file(msg.video.file_id)
        suffix = ".mp4"
        label = "视频"
    elif msg.voice:
        file_obj = await context.bot.get_file(msg.voice.file_id)
        suffix = ".ogg"
        label = "语音"
    elif msg.audio:
        file_obj = await context.bot.get_file(msg.audio.file_id)
        suffix = Path(msg.audio.file_name or "").suffix or ".mp3"
        label = "音频"

    if file_obj is None:
        return None, ""

    dest = _UPLOADS_DIR / f"{user_id}_{msg.message_id}{suffix}"
    await file_obj.download_to_drive(custom_path=str(dest))
    return dest, label


async def on_admin_private(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    user = update.effective_user
    if msg is None or user is None:
        return
    # 任一支持类型:text / caption / photo / document / video / voice / audio
    has_attachment = bool(
        msg.photo or msg.document or msg.video or msg.voice or msg.audio
    )
    text_part = msg.text or msg.caption or ""
    if not text_part and not has_attachment:
        return
    if user.id not in ADMIN_USER_IDS:
        return

    # TOTP session gate
    if not _is_authed(user.id):
        await msg.reply_text(
            "🔒 未授权,请先 `/auth <6 位 TOTP>`(从 Authenticator app 看)。"
            "\n首次使用先扫 setup QR(bot 启动时已 DM 给你)。",
            parse_mode="Markdown",
        )
        return

    # 活跃 → 续 session 24h
    _touch_session(user.id)

    # 下载附件(如有)
    attachment_path, attachment_label = await _download_telegram_attachment(
        context, msg, user.id
    )

    # 构造 prompt — 含附件时告诉 Claude 路径(它用 Read tool 看图 / 文件)
    if attachment_path:
        path_rel = attachment_path.relative_to(Path(__file__).parent)
        prompt_parts = [f"用户发了一个{attachment_label},保存在 ./{path_rel}"]
        if text_part:
            prompt_parts.append(f"caption / 附加说明:{text_part}")
        prompt_parts.append("请用 Read tool 看这个文件并处理用户的需求。")
        prompt = "\n\n".join(prompt_parts)
    else:
        prompt = text_part

    session_id = _chat_sessions.get(user.id)

    try:
        await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)
    except TelegramError:
        pass

    options = ClaudeAgentOptions(
        system_prompt=_AGENT_SYSTEM_PROMPT,
        permission_mode="bypassPermissions",
        resume=session_id,
        max_turns=20,
        max_budget_usd=0.50,
        model=_user_chat_model.get(user.id),  # None = SDK 默认
    )

    collected: list[str] = []
    new_session_id: str | None = None
    err: str | None = None

    try:
        async for m in claude_query(prompt=prompt, options=options):
            if isinstance(m, AssistantMessage):
                for block in m.content:
                    if isinstance(block, TextBlock) and block.text:
                        collected.append(block.text)
                if m.session_id:
                    new_session_id = m.session_id
            elif isinstance(m, ResultMessage):
                if m.session_id:
                    new_session_id = m.session_id
                if m.is_error and m.errors:
                    err = "; ".join(m.errors)
    except Exception as e:
        log.exception("agent query failed for user=%s", user.id)
        err = f"agent 错误:{type(e).__name__}: {e}"

    if new_session_id:
        _chat_sessions[user.id] = new_session_id
        _save_state()

    response = "\n".join(s for s in collected if s).strip()
    if err and not response:
        response = f"⚠️ {err}"
    elif err:
        response += f"\n\n⚠️ {err}"

    await _send_long(msg, response)


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
        # 同步清该用户近 24h bot 看过的全部消息(覆盖之前漏拦的广告)
        purged = await _purge_user_recent_messages(context, chat.id, user.id)
        if purged > 0:
            log.warning("delete_ban: purged %d recent messages from user=%s", purged, user.id)
            await _notify_admin(context, f"🧹 已清该用户近 24h 内 {purged} 条历史消息")


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

    # 记进该用户 24h history(无论 spam 与否)— delete_ban 时一并清,
    # 覆盖之前 classify 没拦住的广告(用户需求)
    if user:
        _record_user_message(user.id, msg.message_id, msg.date or datetime.now(timezone.utc))

    log.info(
        "[%s/%s] %s(@%s): %s",
        chat.type,
        chat.id,
        user.full_name if user else "?",
        user.username if user else "?",
        msg.text[:200],
    )

    try:
        verdict = await llm_router.classify(
            text=msg.text,
            sender_name=user.full_name if user else "?",
            sender_username=user.username if user else None,
            has_link=_has_link(msg),
            is_forwarded=bool(msg.forward_origin),
        )
    except Exception:
        log.exception("classify failed(primary+fallback all errored) — leaving message alone")
        return

    await _act(context, update, verdict)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("handler error", exc_info=context.error)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing in .env")
    # LLM key 检查已在 module 加载时(llm_router 初始化)完成。

    # 启动时 log TOTP setup URL 到 stderr(journalctl 看;首次 setup 扫这个 URL)
    setup_uri = _make_totp_provisioning_uri()
    log.warning("=== TOTP setup URI(扫这个 URL / 输入 secret 到 Authenticator app)===")
    log.warning("URL: %s", setup_uri)
    log.warning("Secret: %s", _TOTP_SECRET)
    log.warning("ASCII QR(扫码):")
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(setup_uri)
        qr.make()
        # ASCII art 到 log
        import io
        buf = io.StringIO()
        qr.print_ascii(out=buf, invert=True)
        for line in buf.getvalue().splitlines():
            log.warning("  %s", line)
    except Exception as e:
        log.warning("(QR 生成失败 — 直接复制 secret 输入 app: %s)", e)
    log.warning("=== END TOTP setup ===")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("auth", cmd_auth))
    app.add_handler(CommandHandler("logout", cmd_logout))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
            on_message,
        )
    )
    # admin 私聊 → Claude Agent SDK chat(完整 tools,session 维持多轮)
    # 支持 text / photo / document / video / voice / audio(附 caption);命令走 CommandHandler 不走这条
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.Document.ALL
             | filters.VIDEO | filters.VOICE | filters.AUDIO)
            & ~filters.COMMAND
            & filters.ChatType.PRIVATE,
            on_admin_private,
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
