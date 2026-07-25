"""易变的规则参数,bot.py 只通过 `import rules` 再 `rules.X` 引用,
这样 `/reload` 命令可以 importlib.reload(rules) 热生效,不需要重启服务。

加新参数时:在这里定义 + 在 bot.py 里把 `rules.X` 用上 + 改完测一下 /reload。
"""

from __future__ import annotations

from datetime import timedelta

# === 重复刷屏检测 ===
# 同一 user 在窗口内重复发同一规范化短消息 ≥ 阈值次 → 自动 delete_ban
# 规范化:NFKC + 去空白 + 小写。第 2 次重复 DM 预警,第 3 次自动封。
REPEAT_WINDOW = timedelta(hours=12)
REPEAT_THRESHOLD = 3
REPEAT_MIN_LEN = 4   # 防 "好的" / "+1" / "ok" 等正常短互动误伤
REPEAT_MAX_LEN = 50  # 长文本通常每次都不同,不参与

# === 累犯升级 ===
# 同一 user 在窗口内被自动删 ≥ 阈值次 → 升级为 delete_ban
RECIDIVIST_WINDOW = timedelta(hours=2)
RECIDIVIST_THRESHOLD = 2

# === 用户消息历史窗口 ===
# delete_ban 时清这个窗口内该用户全部消息(覆盖之前漏拦的广告)
USER_HISTORY_WINDOW = timedelta(hours=24)

# === Admin TOTP session ===
# /auth 通过后,session 在此时长内有效;每次活跃自动续到此时长
SESSION_DURATION = timedelta(days=30)  # 30 天;每次活跃自动续 30 天,/logout 主动退

# === CAS 缓存与刷新 ===
# 全量本地 sqlite 每隔多久重新下一次 export.csv
CAS_REFRESH_INTERVAL = timedelta(hours=24)

# === 新人入群软限制 ===
# CAS 通过后,新人在此时长内禁止发媒体/链接/sticker,纯文字仍能发(走 LLM)
# Telegram restrict_chat_member 的 until_date 会自动恢复权限,无需我们排 job
NEW_USER_RESTRICTION_DURATION = timedelta(hours=24)

# === 末尾 bot 账号入群验证 ===
# username 以 "bot" 结尾 OR is_bot=True → 必须 90s 内点对 emoji,否则踢
# 真正的 Telegram bot 账号无法点 callback button → 必然超时被踢
# 真人若昵称带 bot 后缀,可正常点过
VERIFY_TIMEOUT_SECONDS = 90
