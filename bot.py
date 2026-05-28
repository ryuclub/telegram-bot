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
    ClaudeSDKClient,
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

import classifier
from classifier import ClaudeCodeCLI, LLMRouter, Verdict
import rules

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
#   - 显式 LLM_PROVIDER=claude / openai / claude-cli → 单 provider 无 fallback
#   - 未显式设(默认,推荐):
#       OPENAI_API_KEY 配了    → openai primary(DeepSeek 等)
#       ANTHROPIC_API_KEY 配了 → 加 Claude API 作 fallback
#       默认始终再加 Claude Code CLI 作最后兜底(OAuth 包月,API limit hit 也能用)
#   - 顺序:DeepSeek → Claude API → Claude CLI
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
elif _LLM_PROVIDER == "claude-cli":
    # 强制只用 CLI 包月,任何分类都走 claude-agent-sdk OAuth
    llm_router = LLMRouter(ClaudeCodeCLI(), fallback=None)
elif _LLM_PROVIDER == "":
    # 自动 — primary openai(快/便宜),fallback Claude CLI(包月,不算 API spend)
    # API key 路径默认不上 LLMRouter(API limit 一旦 hit 反而拖慢)
    if _has_openai:
        llm_router = LLMRouter(_make_openai(), fallback=ClaudeCodeCLI())
    elif _has_anthropic:
        # 没 OpenAI key,只能用 API(走 monthly limit),仍加 CLI 作兜底
        llm_router = LLMRouter(_make_claude(), fallback=ClaudeCodeCLI())
    else:
        # 啥 key 都没,只剩 CLI 包月
        llm_router = LLMRouter(ClaudeCodeCLI(), fallback=None)
else:
    raise SystemExit(f"unknown LLM_PROVIDER={_LLM_PROVIDER!r},需 claude / openai / claude-cli / 空")

# === 可热重载常量参考 rules.py ===
# 这里只保留运行时状态(用户历史 deque 等),阈值/窗口全部走 rules.X 动态读取
_delete_history: dict[int, deque[datetime]] = defaultdict(deque)
_user_messages: dict[int, deque[tuple[int, datetime]]] = defaultdict(deque)
_repeat_history: dict[int, dict[str, deque[datetime]]] = defaultdict(lambda: defaultdict(deque))


def _normalize_for_repeat(text: str) -> str:
    """规范化以识别"加个空格/标点"的变体刷屏。"""
    import unicodedata
    # 全角→半角 + 去所有空白 + 大小写统一
    t = unicodedata.normalize("NFKC", text or "")
    return "".join(t.split()).lower()


def _check_repeat_spam(user_id: int, text: str, now: datetime) -> int:
    """记录 + 检测。返回窗口内累计次数(含本次)。≥ REPEAT_THRESHOLD 即触发。"""
    norm = _normalize_for_repeat(text)
    if not norm or len(norm) < rules.REPEAT_MIN_LEN or len(norm) > rules.REPEAT_MAX_LEN:
        return 0  # 空 / 过短(噪音回复) / 过长(不算同条) 不参与
    bucket = _repeat_history[user_id][norm]
    cutoff = now - rules.REPEAT_WINDOW
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    bucket.append(now)
    return len(bucket)

# 入群验证
VERIFY_ENABLED = os.environ.get("VERIFY_ENABLED", "true").lower() in ("1", "true", "yes")
CAS_ENABLED = os.environ.get("CAS_ENABLED", "true").lower() in ("1", "true", "yes")

# bot 后缀验证用 — full mute + emoji 验证
NO_PERMS = ChatPermissions(can_send_messages=False)
EMOJI_POOL = [
    "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼",
    "🐨", "🐯", "🦁", "🐮", "🐷", "🐸", "🐵", "🐔",
    "🦄", "🐝", "🐢", "🦋",
]
_pending_verifications: dict[int, dict] = {}

# 新人 24h 软限制:只允许纯文字 + polls + 其它互动,禁所有媒体 / 链接预览
# Telegram 的 restrict_chat_member 加 until_date 会自动到点恢复 → 无需 job
NEW_USER_PERMS = ChatPermissions(
    can_send_messages=True,        # 文字 OK,走 LLM
    can_send_audios=False,
    can_send_documents=False,
    can_send_photos=False,
    can_send_videos=False,
    can_send_video_notes=False,
    can_send_voice_notes=False,
    can_send_polls=False,
    can_send_other_messages=False,  # 禁 sticker / GIF / 内嵌 game
    can_add_web_page_previews=False,
    can_change_info=False,
    can_invite_users=False,
    can_pin_messages=False,
    can_manage_topics=False,
)


async def _cas_check(user_id: int) -> dict | None:
    """查 CAS (combot anti-spam) 黑名单。
    命中 → 返 result dict(含 offenses / messages / time_added);
    干净 → 返 None;出错 → None(fail-open)。"""
    if not CAS_ENABLED:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"https://api.cas.chat/check?user_id={user_id}")
            data = r.json()
            if data.get("ok"):
                return data.get("result") or {}
            return None
    except Exception:
        log.warning("CAS check failed for %s", user_id, exc_info=True)
        return None


# CAS 全量本地 sqlite — export.csv 110 万 ID,12 MB
# 启动时 + 每 24h 后台刷新。on_message 走本地查询(微秒级)。
# 命中再调 per-user API 拿详情(offenses/samples),DM 才看得到。
_CAS_DB_PATH = Path(__file__).parent / "data" / "cas-ids.sqlite"
_CAS_EXPORT_URL = "https://api.cas.chat/export.csv"
_cas_last_refresh: datetime | None = None


def _cas_db_has(user_id: int) -> bool:
    """本地 sqlite 查 user_id 是否在 CAS 黑名单。db 不存在 → False(fail-open)。"""
    if not _CAS_DB_PATH.exists():
        return False
    try:
        import sqlite3
        conn = sqlite3.connect(_CAS_DB_PATH, timeout=2.0)
        cur = conn.execute("SELECT 1 FROM cas WHERE user_id=? LIMIT 1", (user_id,))
        hit = cur.fetchone() is not None
        conn.close()
        return hit
    except Exception:
        log.warning("CAS local db query failed for %s", user_id, exc_info=True)
        return False


async def _cas_refresh_db() -> bool:
    """下 export.csv → 重建 sqlite。原子替换(写到 .tmp 再 rename)。"""
    import sqlite3
    tmp_csv = _CAS_DB_PATH.with_suffix(".csv.tmp")
    tmp_db = _CAS_DB_PATH.with_suffix(".sqlite.tmp")
    for p in (tmp_csv, tmp_db):
        if p.exists():
            p.unlink()
    try:
        log.info("CAS db refresh: downloading %s", _CAS_EXPORT_URL)
        async with httpx.AsyncClient(timeout=60.0) as c:
            async with c.stream("GET", _CAS_EXPORT_URL) as r:
                r.raise_for_status()
                with tmp_csv.open("wb") as f:
                    async for chunk in r.aiter_bytes(64 * 1024):
                        f.write(chunk)
        # 解析 + 建库
        conn = sqlite3.connect(tmp_db)
        conn.execute("CREATE TABLE cas(user_id INTEGER PRIMARY KEY)")
        with tmp_csv.open() as f:
            batch = []
            for line in f:
                s = line.strip().split(",")[0]
                if s.isdigit():
                    batch.append((int(s),))
                if len(batch) >= 5000:
                    conn.executemany("INSERT OR IGNORE INTO cas VALUES(?)", batch)
                    batch.clear()
            if batch:
                conn.executemany("INSERT OR IGNORE INTO cas VALUES(?)", batch)
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM cas").fetchone()[0]
        conn.close()
        # 原子 rename
        tmp_db.replace(_CAS_DB_PATH)
        try:
            tmp_csv.unlink()
        except OSError:
            pass
        global _cas_last_refresh
        _cas_last_refresh = datetime.now(timezone.utc)
        log.info("CAS db refresh OK: %d rows", n)
        return True
    except Exception:
        log.exception("CAS db refresh failed")
        for p in (tmp_csv, tmp_db):
            if p.exists():
                try: p.unlink()
                except OSError: pass
        return False


async def _cas_refresh_loop() -> None:
    """后台 task: 每 24h 刷新一次 CAS db。启动时如 db 不存在或过期立刻刷。"""
    # 启动时按需刷一次
    age = None
    if _CAS_DB_PATH.exists():
        mtime = datetime.fromtimestamp(_CAS_DB_PATH.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(timezone.utc) - mtime
    if not _CAS_DB_PATH.exists() or (age and age > rules.CAS_REFRESH_INTERVAL):
        await _cas_refresh_db()
    while True:
        try:
            await asyncio.sleep(rules.CAS_REFRESH_INTERVAL.total_seconds())
            await _cas_refresh_db()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("CAS refresh loop err")
            await asyncio.sleep(3600)  # 出错后 1h 再试


async def _cas_check_cached(user_id: int) -> dict | None:
    """on_message 主入口:先查本地 sqlite,命中再走 API 拿详情。
    本地无命中 = 干净(覆盖 99% 流量,零 API 调用)。"""
    if not CAS_ENABLED:
        return None
    if not _cas_db_has(user_id):
        return None  # 本地说干净,信它
    # 本地命中 → 调 API 拿详情(offenses/samples/time_added)给 DM 用
    detail = await _cas_check(user_id)
    if detail is not None:
        return detail
    # API 暂时挂 → 至少回个最小 dict,让上层知道命中
    return {"offenses": "?", "time_added": "?", "messages": []}

# 数据日志(后续可用来挖正则预过滤规则)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _record_delete(user_id: int, when: datetime) -> int:
    """记录一次删除,返回窗口内累计次数(含本次)。"""
    history = _delete_history[user_id]
    cutoff = when - rules.RECIDIVIST_WINDOW
    while history and history[0] < cutoff:
        history.popleft()
    history.append(when)
    return len(history)


def _record_user_message(user_id: int, message_id: int, when: datetime) -> None:
    """记录用户每条非命令文本消息 + 自动淘汰 > USER_HISTORY_WINDOW 的老条目。"""
    history = _user_messages[user_id]
    cutoff = when - rules.USER_HISTORY_WINDOW
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
        "caption": msg.caption if msg else None,
        "enriched_text": _extract_classify_text(msg) if msg else None,
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
#   2. TOTP /auth 通过 → 72h session,每次活跃自动续期 72h(/logout 主动结束)
# permission_mode="bypassPermissions" = 完全跳过 y/n 确认,等于 telegram admin 远控 botuser 账号
# 全部能力(含 NOPASSWD sudo)。session_id 维持多轮;/reset 清当前 user 的 claude session。

import pyotp

_TOTP_SECRET_FILE = Path(__file__).parent / ".totp-secret"
_TOTP_LOCK_MAX_FAILS = 5
_TOTP_LOCK_DURATION = timedelta(minutes=15)
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
# Auth session:user_id -> last_active datetime;过期(now - last_active > 72h)= 需重 /auth
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
    return (datetime.now(timezone.utc) - last) < rules.SESSION_DURATION


def _touch_session(user_id: int) -> None:
    """活跃 → 重置 last_active(续期 72h)"""
    _auth_sessions[user_id] = datetime.now(timezone.utc)
    _save_state()

_AGENT_SYSTEM_PROMPT = """\
你是 telegram 群管理 bot 内嵌的助手,通过私聊跟管理员(botuser)交互。
你跑在管理员的 Ubuntu 机器上,拥有完整 Claude Code tools(Bash / Read / Write / Edit 等)。
回复尽量简洁、用 plain text(Telegram 不解析 markdown),避免过长输出。
"""


# === ClaudeSDKClient 长连接池 — per-user 复用 subprocess + prompt cache ===
# 首次消息 spawn `claude` 进程,后续消息复用同进程(prompt cache 命中);
# IDLE_TIMEOUT 无活动 → 关 subprocess 释放内存。
_IDLE_TIMEOUT = timedelta(minutes=30)
_CLIENT_CLEANUP_INTERVAL = 60  # 秒


class _ClientSession:
    """per-user ClaudeSDKClient + lock + last_used 追踪。"""

    def __init__(self, options: ClaudeAgentOptions):
        self.options = options
        self.client = ClaudeSDKClient(options=options)
        self.last_used = datetime.now(timezone.utc)
        self.lock = asyncio.Lock()  # 防同一 user 并发多消息
        self.connected = False

    async def ensure_connected(self) -> None:
        if not self.connected:
            await self.client.connect()
            self.connected = True

    async def close(self) -> None:
        if self.connected:
            try:
                await self.client.disconnect()
            except Exception as e:
                log.warning("client disconnect err: %s", e)
            self.connected = False


_client_sessions: dict[int, _ClientSession] = {}


async def _get_or_create_client_session(user_id: int) -> _ClientSession:
    """有 client 用现成的;无则按 user 当前 session_id + model 新建并 connect。"""
    s = _client_sessions.get(user_id)
    if s is not None:
        s.last_used = datetime.now(timezone.utc)
        return s
    options = ClaudeAgentOptions(
        system_prompt=_AGENT_SYSTEM_PROMPT,
        permission_mode="bypassPermissions",
        resume=_chat_sessions.get(user_id),  # 上次的 claude session_id;None = 新对话
        max_turns=50,
        max_budget_usd=20.0,
        model=_user_chat_model.get(user_id),
        # 关键:.env 里的 ANTHROPIC_API_KEY 让 CLI 走 API 计费(用户的 monthly spend limit)。
        # 清空让 CLI fall back 到 Claude Code OAuth 订阅(包月,不计 token)。
        # classifier 那条路在主进程内调 anthropic SDK,仍能用 env 里的 API key 作 fallback。
        env={"ANTHROPIC_API_KEY": ""},
    )
    s = _ClientSession(options)
    await s.ensure_connected()
    _client_sessions[user_id] = s
    return s


async def _close_client_session(user_id: int) -> None:
    s = _client_sessions.pop(user_id, None)
    if s is not None:
        await s.close()


async def _client_idle_cleanup_loop() -> None:
    """每分钟扫,关 30 min 无活动 client。"""
    while True:
        try:
            await asyncio.sleep(_CLIENT_CLEANUP_INTERVAL)
            now = datetime.now(timezone.utc)
            for uid in list(_client_sessions.keys()):
                s = _client_sessions.get(uid)
                if s is None or s.lock.locked():
                    continue
                if (now - s.last_used) > _IDLE_TIMEOUT:
                    log.info("client idle cleanup user=%s", uid)
                    await _close_client_session(uid)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("client idle cleanup loop err")


async def cmd_reset(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None or user.id not in ADMIN_USER_IDS:
        await update.effective_message.reply_text("权限不足。")
        return
    had = _chat_sessions.pop(user.id, None)
    if had:
        _save_state()
    # 关现有 client,下条消息会重建(无 resume = 全新对话)
    await _close_client_session(user.id)
    txt = "✅ 会话已清,下条消息开新对话。" if had else "(本来就没活跃会话)"
    await update.effective_message.reply_text(txt)


async def cmd_auth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/auth <6 位 TOTP>` — 验证后开 72h session,活跃自动续期。"""
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
            f"✅ 验证通过。72h session 已开,每次发消息自动续 72h。\n`/logout` 主动结束。",
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
    await _close_client_session(user.id)  # 关持久 client subprocess
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
    # 关现有 client,下次重建用新 model
    await _close_client_session(user.id)
    await msg.reply_text(f"✅ chat model 切为 `{target}`。", parse_mode="Markdown")


async def cmd_reload(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """/reload — 热重载 rules.py + classifier.py 模块。
    失败(语法/导入错)→ 保留旧规则继续运行,DM 报错。
    只有改 bot.py 骨架时才需要 systemctl restart。"""
    user = update.effective_user
    msg = update.effective_message
    if user is None or user.id not in ADMIN_USER_IDS:
        await msg.reply_text("权限不足。")
        return
    import importlib
    errs = []
    reloaded = []
    for modname, mod in (("rules", rules), ("classifier", classifier)):
        try:
            importlib.reload(mod)
            reloaded.append(modname)
        except Exception as e:
            errs.append(f"{modname}: {type(e).__name__}: {e}")
            log.exception("reload %s failed", modname)
    parts = []
    if reloaded:
        parts.append("✅ 热重载: " + ", ".join(reloaded))
        # 显示几个关键参数,确认新值
        parts.append(
            f"REPEAT={rules.REPEAT_THRESHOLD}/"
            f"{int(rules.REPEAT_WINDOW.total_seconds()/3600)}h "
            f"min={rules.REPEAT_MIN_LEN} max={rules.REPEAT_MAX_LEN}\n"
            f"RECIDIVIST={rules.RECIDIVIST_THRESHOLD}/"
            f"{int(rules.RECIDIVIST_WINDOW.total_seconds()/3600)}h\n"
            f"SESSION={int(rules.SESSION_DURATION.total_seconds()/3600)}h\n"
            f"USER_HISTORY={int(rules.USER_HISTORY_WINDOW.total_seconds()/3600)}h\n"
            f"CAS_REFRESH={int(rules.CAS_REFRESH_INTERVAL.total_seconds()/3600)}h"
        )
    if errs:
        parts.append("❌ 失败:\n" + "\n".join(errs))
    await msg.reply_text("\n\n".join(parts) or "(nothing changed)")


async def _send_long(message, text: str) -> None:
    """Telegram 单条 4096 chars 上限,长输出分段。空响应直接静默(typing 消失即是完成信号)。"""
    if not text:
        return  # 不发垃圾占位消息
    CHUNK = 4000
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

    # 写 INDEX.jsonl 一行 — Claude 后续 `Read data/uploads/INDEX.jsonl` 找历史素材
    caption = msg.caption or msg.text or ""
    index_entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "msg_id": msg.message_id,
        "type": label,
        "filename": dest.name,
        "path": str(dest.relative_to(Path(__file__).parent)),
        "caption": caption,
    }
    try:
        index_file = _UPLOADS_DIR / "INDEX.jsonl"
        with index_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("uploads index append failed: %s", e)

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

    # 活跃 → 续 session 72h
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

    # Typing keep-alive — ChatAction.TYPING 仅持续 ~5s,长查询需循环刷新
    async def _typing_loop():
        try:
            while True:
                try:
                    await context.bot.send_chat_action(
                        chat_id=msg.chat_id, action=ChatAction.TYPING
                    )
                except TelegramError:
                    pass
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            return

    typing_task = asyncio.create_task(_typing_loop())

    collected: list[str] = []
    new_session_id: str | None = None
    err: str | None = None

    try:
        session = await _get_or_create_client_session(user.id)
        async with session.lock:
            await session.client.query(prompt)
            async for m in session.client.receive_response():
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
            session.last_used = datetime.now(timezone.utc)
    except Exception as e:
        log.exception("agent query failed for user=%s", user.id)
        err = f"agent 错误:{type(e).__name__}: {e}"
        # 出错 → 关 client,下次重新 connect
        await _close_client_session(user.id)
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except (asyncio.CancelledError, Exception):
            pass

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
    # 先试 Markdown,失败 fall back plain text — 用户文本可能含 markdown 元字符炸 parser
    try:
        await context.bot.send_message(NOTIFY_ADMIN_ID, body, parse_mode="Markdown")
    except TelegramError as e:
        log.warning("notify admin Markdown failed (%s), retry plain text", e)
        try:
            await context.bot.send_message(NOTIFY_ADMIN_ID, body, disable_web_page_preview=True)
        except TelegramError as e2:
            log.warning("notify admin plain text also failed: %s", e2)


def _has_link(msg) -> bool:
    entities = list(msg.entities or ()) + list(msg.caption_entities or ())
    if not entities:
        # link_preview 也算
        lpo = getattr(msg, "link_preview_options", None)
        return bool(lpo and getattr(lpo, "url", None))
    link_types = {
        MessageEntityType.URL,
        MessageEntityType.TEXT_LINK,
        MessageEntityType.MENTION,
        MessageEntityType.TEXT_MENTION,
    }
    return any(e.type in link_types for e in entities)


def _extract_links_mentions(msg) -> list[str]:
    """从 entities + caption_entities + link_preview 抽 URL/text_link/@mention,
    给 LLM 看 — 防止"text='i' 但藏隐藏链接"这种诱饵广告漏判。"""
    out: list[str] = []
    text = msg.text or msg.caption or ""
    entities = list(msg.entities or ()) + list(msg.caption_entities or ())
    for e in entities:
        et = e.type
        if et == MessageEntityType.URL:
            out.append(text[e.offset : e.offset + e.length])
        elif et == MessageEntityType.TEXT_LINK and e.url:
            visible = text[e.offset : e.offset + e.length]
            out.append(f"{visible!r}→{e.url}")
        elif et == MessageEntityType.MENTION:
            out.append(text[e.offset : e.offset + e.length])
        elif et == MessageEntityType.TEXT_MENTION and e.user:
            uname = e.user.username or e.user.full_name or "?"
            out.append(f"@{uname}(id={e.user.id})")
    lpo = getattr(msg, "link_preview_options", None)
    if lpo and getattr(lpo, "url", None):
        out.append(f"[link_preview→{lpo.url}]")
    return out


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
        if count >= rules.RECIDIVIST_THRESHOLD:
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
    # 构造可点击的 t.me 链接(supergroup 有 username 用公开链接,否则用 c/<id> 私有链接)
    if chat.username:
        msg_link = f"https://t.me/{chat.username}/{msg.message_id}"
    elif str(chat.id).startswith("-100"):
        msg_link = f"https://t.me/c/{str(chat.id)[4:]}/{msg.message_id}"
    else:
        msg_link = ""

    body = (
        f"*{prefix}{action_shown}*  `{verdict.category}` conf=`{verdict.confidence:.2f}`\n"
        f"原因: {verdict.reason}"
    )
    if upgraded_reason:
        body += f"\n⚠️ {upgraded_reason}"
    body += (
        f"\n发送人: {sender}\n"
        f"msg_id: `{msg.message_id}`"
    )
    if msg_link:
        body += f"  [跳转]({msg_link})"
    body += f"\n原文:\n```\n{(msg.text or msg.caption or '')[:500]}\n```"
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
        return
    await _kick(context, chat_id, user_id, "超时未验证(bot 后缀账号)")
    await _cleanup_verification(context, user_id, chat_id)


async def on_verify_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """CallbackQueryHandler — bot 后缀账号 emoji 验证。"""
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
        await _kick(context, chat_id, target_uid, "点错 emoji(bot 后缀账号)")
        await _cleanup_verification(context, target_uid, chat_id)
        return
    # 通过 → 解禁到 NEW_USER_PERMS(24h 软限制),走标准新人流程
    try:
        until = datetime.now(timezone.utc) + rules.NEW_USER_RESTRICTION_DURATION
        await context.bot.restrict_chat_member(
            chat_id=chat_id, user_id=target_uid,
            permissions=NEW_USER_PERMS, until_date=until,
        )
    except TelegramError as e:
        log.warning("verify: unmute failed: %s", e)
        await query.answer("解禁失败,联系管理员", show_alert=True)
        return
    await query.answer("✅ 验证通过,欢迎加入!", show_alert=False)
    await _cleanup_verification(context, target_uid, chat_id)
    log.info("verify: passed user=%s", target_uid)


async def _start_bot_suffix_verification(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, user
) -> None:
    """username 末尾 bot 或 is_bot=True → 全静音 + emoji 三选一 + 90s 超时踢。"""
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat_id, user_id=user.id, permissions=NO_PERMS
        )
    except TelegramError as e:
        log.warning("verify: mute failed for %s: %s", user.id, e)
        return
    choices = random.sample(EMOJI_POOL, 3)
    correct = random.choice(choices)
    _pending_verifications[user.id] = {
        "chat_id": chat_id, "correct_emoji": correct, "welcome_msg_id": None,
    }
    mention = user.mention_html()
    text = (
        f"👋 {mention}\n\n"
        f"检测到账号是 bot 类型 / 末尾 bot,请在 <b>{rules.VERIFY_TIMEOUT_SECONDS} 秒</b>内点击 <b>{correct}</b> 证明你是真人。\n"
        "点错或超时会被请出群(可重进)。"
    )
    keyboard = [
        InlineKeyboardButton(emoji, callback_data=f"verify:{user.id}:{emoji}")
        for emoji in choices
    ]
    try:
        welcome = await context.bot.send_message(
            chat_id=chat_id, text=text, parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([keyboard]),
        )
    except TelegramError as e:
        log.warning("verify: welcome send failed: %s", e)
        _pending_verifications.pop(user.id, None)
        return
    _pending_verifications[user.id]["welcome_msg_id"] = welcome.message_id
    if context.job_queue:
        context.job_queue.run_once(
            _verify_timeout, rules.VERIFY_TIMEOUT_SECONDS,
            name=f"verify_{chat_id}_{user.id}",
            data={"chat_id": chat_id, "user_id": user.id},
        )
    log.info("verify(bot-suffix): started user=%s(@%s) is_bot=%s correct=%s",
             user.id, user.username, user.is_bot, correct)


async def on_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """新人进群:CAS 黑名单查询(命中即踢);末尾 bot → emoji 验证;否则 24h 软限制。"""
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
    # 仅豁免本 bot 自己,其它 bot(包括 is_bot=True 的 spam bot)正常审
    if context.bot and user.id == context.bot.id:
        return
    if user.id in ADMIN_USER_IDS:
        return

    # 1. CAS 黑名单查询(命中 → 踢 + DM 详情)
    cas_result = await _cas_check_cached(user.id)
    if cas_result is not None:
        await _kick(context, chat.id, user.id, "CAS 黑名单命中")
        if NOTIFY_ADMIN_ID:
            offenses = cas_result.get("offenses", "?")
            time_added = cas_result.get("time_added") or "?"
            sample_msgs = cas_result.get("messages") or []
            sample_str = ""
            if sample_msgs:
                sample_str = "\n样本:\n" + "\n".join(
                    f"  • `{(m or '')[:120]}`" for m in sample_msgs[:3]
                )
            lines = [
                "🚫 *CAS 命中*",
                f"用户: {user.full_name} (@{user.username or '?'}, id=`{user.id}`)",
                f"违规次数: {offenses}",
                f"上榜时间: {time_added}",
                f"CAS 链接: https://cas.chat/query?u={user.id}",
                "已自动踢出。" + sample_str,
            ]
            try:
                await context.bot.send_message(
                    NOTIFY_ADMIN_ID,
                    "\n".join(lines),
                    parse_mode="Markdown",
                    disable_web_page_preview=True,
                )
            except TelegramError as e:
                log.warning("CAS hit notify failed: %s", e)
                try:
                    await context.bot.send_message(
                        NOTIFY_ADMIN_ID,
                        "\n".join(lines),
                        disable_web_page_preview=True,
                    )
                except TelegramError:
                    pass
        return

    # 2. bot 后缀 / is_bot 账号 → emoji 验证(真 bot 无法点击 → 超时被踢)
    uname = (user.username or "").lower()
    if user.is_bot or uname.endswith("bot"):
        await _start_bot_suffix_verification(context, chat.id, user)
        return

    # 3. 通过 CAS → 24h 软限制:只能发纯文字(media/links/stickers 禁)
    # Telegram restrict_chat_member 的 until_date 到点自动解禁,不需要我们排 job
    until = datetime.now(timezone.utc) + rules.NEW_USER_RESTRICTION_DURATION
    try:
        await context.bot.restrict_chat_member(
            chat_id=chat.id,
            user_id=user.id,
            permissions=NEW_USER_PERMS,
            until_date=until,
        )
        log.info(
            "new member soft-restrict: user=%s(@%s) until=%s",
            user.id, user.username, until.isoformat(),
        )
    except TelegramError as e:
        log.warning("new member restrict failed for %s: %s", user.id, e)


def _extract_classify_text(msg) -> str:
    """拼丰富 context 给 classifier — 防单字符 / 媒体类广告漏检:
    text/caption + entities URL + link_preview + 转发 + reply quote + contact/sticker/media
    """
    parts: list[str] = []
    base = (msg.text or msg.caption or "").strip()
    if base:
        parts.append(base)

    # contact (vCard) — spammer 常用,卡片渲染像广告 banner;号码+姓名暴露其 spam 意图
    if msg.contact:
        c = msg.contact
        parts.append(
            f"[共享联系人] name={c.first_name or ''} {c.last_name or ''} "
            f"phone={c.phone_number or '?'} vcard={'有' if c.vcard else '无'}"
        )
    # sticker — 部分 sticker pack 是引流广告
    if msg.sticker:
        s = msg.sticker
        parts.append(
            f"[贴纸] emoji={s.emoji or '?'} set={s.set_name or '?'}"
        )
    # 媒体类 — 即使没 caption 也告诉 LLM 类型
    media_types = []
    if msg.photo: media_types.append("photo")
    if msg.video: media_types.append("video")
    if msg.animation: media_types.append("animation/GIF")
    if msg.document: media_types.append(f"document({msg.document.file_name or '?'})")
    if msg.location: media_types.append("location")
    if msg.venue:
        v = msg.venue
        media_types.append(f"venue({v.title or '?'})")
    if media_types and not base:
        parts.append(f"[纯媒体消息: {', '.join(media_types)}]")

    # 提 URL:text_link entity(显示文字 vs 实际 URL 不同的)+ url entity 在 text 已含
    extra_urls: list[str] = []
    for ent in list(msg.entities or []) + list(msg.caption_entities or []):
        if ent.type == "text_link" and ent.url:
            extra_urls.append(ent.url)
    # link preview 也给(广告常见:文字 "i" + 链接预览 t.me/...)
    if getattr(msg, "link_preview_options", None) and getattr(msg.link_preview_options, "url", None):
        extra_urls.append(msg.link_preview_options.url)
    if extra_urls:
        parts.append("[包含链接] " + " ".join(extra_urls))

    # Reply quote — Telegram 允许 reply 时 quote 任意外部消息片段(文本不必发到本群)
    # 广告号常用:本消息发 "o",quote 一段真广告;客户端渲染时显示 quote 内容 → 用户看到广告
    quote = getattr(msg, "quote", None)
    if quote and getattr(quote, "text", None):
        parts.append(f"[引用文本] {quote.text}")
    # external_reply — 引用外部 channel/group 消息;暴露来源 channel
    ext = getattr(msg, "external_reply", None)
    if ext:
        ext_chat = getattr(ext, "chat", None)
        if ext_chat:
            ext_name = f"@{ext_chat.username or ext_chat.title or '?'}"
            parts.append(f"[引用自外部频道 {ext_name}]")
    # 本群 reply
    if getattr(msg, "reply_to_message", None):
        r = msg.reply_to_message
        rtxt = (r.text or r.caption or "")
        if rtxt:
            parts.append(f"[reply: {rtxt[:200]}]")

    # 转发来源 — channel/user 名 是判 spam 的强信号
    if msg.forward_origin:
        try:
            origin = msg.forward_origin
            origin_name = type(origin).__name__
            if hasattr(origin, "chat") and origin.chat:
                origin_name = f"频道 @{origin.chat.username or origin.chat.title or '?'}"
            elif hasattr(origin, "sender_user") and origin.sender_user:
                origin_name = f"用户 @{origin.sender_user.username or origin.sender_user.full_name or '?'}"
            elif hasattr(origin, "sender_user_name"):
                origin_name = f"隐藏用户 {origin.sender_user_name}"
            parts.append(f"[转发自 {origin_name}]")
        except Exception:
            parts.append("[转发]")

    return "\n".join(parts) if parts else (msg.text or msg.caption or "")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None:
        return
    # 至少要 text/caption/forward/含 URL 之一才送审(纯媒体 / 纯反应不审)
    has_content = bool(
        msg.text or msg.caption
        or msg.forward_origin
        or msg.entities or msg.caption_entities
        or msg.contact or msg.sticker
        or msg.photo or msg.video or msg.animation or msg.document
        or msg.location or msg.venue
        or getattr(msg, "quote", None) or getattr(msg, "external_reply", None)
    )
    if not has_content:
        return
    chat = update.effective_chat
    user = msg.from_user

    # 仅处理目标群
    if GROUP_CHAT_ID is not None and chat.id != GROUP_CHAT_ID:
        return

    # 管理员豁免(不包括匿名管理员 1087968824 — 那是占位虚拟账号)
    if user and user.id in ADMIN_USER_IDS:
        return

    # bot 自己豁免 — 仅豁免本 bot,其它 bot(spam 广告 bot 一抓一把)正常审
    if user and user.is_bot and context.bot and user.id == context.bot.id:
        return

    # 记进该用户 24h history(无论 spam 与否)— delete_ban 时一并清,
    # 覆盖之前 classify 没拦住的广告(用户需求)
    if user:
        _record_user_message(user.id, msg.message_id, msg.date or datetime.now(timezone.utc))

    # CAS 黑名单前置(缓存 6h clean / 30d hit) — 覆盖"入群时干净后来上榜"的盲区
    # 命中 → 直接清近 24h + 永封,跳过 LLM
    from classifier import Verdict
    if user:
        cas_result = await _cas_check_cached(user.id)
        if cas_result is not None:
            offenses = cas_result.get("offenses", "?")
            time_added = cas_result.get("time_added") or "?"
            log.warning(
                "CAS pre-filter hit: user=%s(@%s) offenses=%s added=%s",
                user.id, user.username, offenses, time_added,
            )
            verdict = Verdict(
                is_spam=True,
                category="other_spam",
                confidence=1.0,
                reason=f"CAS 黑名单命中(offenses={offenses}, added={time_added})",
                action="delete_ban",
            )
            await _act(context, update, verdict)
            return

    # 硬规则:任何用户发 vCard 共享联系人 → 直接 delete_ban
    # 移民讨论群没人正经分享 vcard,基本都是广告 banner(假钞/客服/灰产联系卡)
    if user and msg.contact:
        log.warning(
            "vCard pre-filter: user=%s contact=%s phone=%s → delete_ban",
            user.id, (msg.contact.first_name or ''), msg.contact.phone_number,
        )
        verdict = Verdict(
            is_spam=True,
            category="other_spam",
            confidence=1.0,
            reason=f"分享 vCard 联系人卡片({msg.contact.phone_number or '?'})",
            action="delete_ban",
        )
        await _act(context, update, verdict)
        return

    enriched_text = _extract_classify_text(msg)
    log.info(
        "[%s/%s] msg=%s %s(@%s, id=%s, bot=%s): %s",
        chat.type,
        chat.id,
        msg.message_id,
        user.full_name if user else "?",
        user.username if user else "?",
        user.id if user else "?",
        user.is_bot if user else "?",
        enriched_text[:200],
    )

    # 预过滤:同一 user 短时间反复发同一短消息
    base_text = msg.text or msg.caption or ""
    if user and base_text:
        count = _check_repeat_spam(
            user.id, base_text, msg.date or datetime.now(timezone.utc)
        )
        if count >= rules.REPEAT_THRESHOLD:
            # ≥3 次 → 100% spam,跳过 LLM
            log.warning(
                "repeat-spam pre-filter triggered: user=%s text=%r count=%d/%d",
                user.id, base_text[:60], count, rules.REPEAT_THRESHOLD,
            )
            verdict = Verdict(
                is_spam=True,
                category="other_spam",
                confidence=1.0,
                reason=f"{rules.REPEAT_WINDOW.total_seconds()/3600:.0f}h 内重复发送同一短消息 {count} 次",
                action="delete_ban",
            )
            await _act(context, update, verdict)
            return
        elif count == 2:
            # 第 2 次相同 — 不一定 spam(可能正常重复),但提前 DM 管理员留意,
            # 不阻断 LLM 流程,继续走正常判断
            log.info(
                "repeat-spam 2nd hit (advisory): user=%s text=%r",
                user.id, base_text[:60],
            )
            asyncio.create_task(_notify_admin(
                context,
                f"⚠️ *重复刷屏预警*(2 次,未拦截)\n"
                f"用户: {user.full_name} (@{user.username or '?'}, id={user.id})\n"
                f"消息: `{base_text[:120]}`\n"
                f"_第 3 次会自动 delete_ban_"
            ))

    try:
        verdict = await llm_router.classify(
            text=enriched_text,
            sender_name=user.full_name if user else "?",
            sender_username=user.username if user else None,
            has_link=_has_link(msg),
            is_forwarded=bool(msg.forward_origin),
        )
    except Exception:
        log.exception(
            "classify failed (primary+fallback all errored) — flagging for manual review"
        )
        # 不要静默漏:给管理员 DM 一个 flag verdict,他能手动看 & 删
        verdict = Verdict(
            is_spam=False,
            category="other_spam",
            confidence=0.0,
            reason="LLM 分类器全部失败,需人工查看",
            action="flag",
        )

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

    async def _post_init(app_):
        # 启动 background cleanup loop(每分钟扫 idle ClaudeSDKClient,30 min 无活动 → 关)
        app_.bot_data["idle_cleanup_task"] = asyncio.create_task(_client_idle_cleanup_loop())
        # 启动 CAS db 后台刷新 task(启动时刷一次 + 每 24h 刷)
        app_.bot_data["cas_refresh_task"] = asyncio.create_task(_cas_refresh_loop())
        # 上线通知:DM 所有当前活跃 session 的 admin,免得他们不知道 bot 重启没回话
        # 之前若有未回复消息,admin 知道得自己重发
        now = datetime.now(timezone.utc)
        for uid, last in list(_auth_sessions.items()):
            if (now - last) >= rules.SESSION_DURATION:
                continue
            try:
                await app_.bot.send_message(
                    uid,
                    f"🔄 bot 重启上线\n"
                    f"上次活跃: {last:%Y-%m-%d %H:%M UTC}\n"
                    f"如刚才有消息我没回复,请重发。"
                )
            except Exception as e:
                log.warning("startup notify uid=%s failed: %s", uid, e)

    async def _post_shutdown(app_):
        # 关所有 active ClaudeSDKClient + 取消所有 background tasks
        for key in ("idle_cleanup_task", "cas_refresh_task"):
            task = app_.bot_data.get(key)
            if task:
                task.cancel()
        for uid in list(_client_sessions.keys()):
            await _close_client_session(uid)

    app = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("auth", cmd_auth))
    app.add_handler(CommandHandler("logout", cmd_logout))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.CONTACT | filters.Sticker.ALL
             | filters.PHOTO | filters.VIDEO | filters.ANIMATION
             | filters.Document.ALL | filters.LOCATION | filters.VENUE)
            & ~filters.COMMAND & filters.ChatType.GROUPS,
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
        "bot starting | DRY_RUN=%s | VERIFY=%s | CAS=%s | new-user soft-restrict=%s | target_group=%s | admins=%s",
        DRY_RUN,
        VERIFY_ENABLED,
        CAS_ENABLED,
        rules.NEW_USER_RESTRICTION_DURATION,
        GROUP_CHAT_ID,
        ADMIN_USER_IDS,
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
