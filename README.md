# telegram-bot

群管理 bot:
- 新人加群 emoji 验证
- LLM 驱动消息分类拦截广告 / 洗钱黑话
- delete_ban 时同步清该用户近 24h 全部消息
- **管理员私聊 = 远程 Claude Code 助手**(TOTP 验证 + 24h 自动续期 session,完整 tools)

LLM provider 支持:**Claude**(Anthropic 原生 + prompt caching)/ **OpenAI 兼容**(含 **DeepSeek**)。env 切换 / 自动 fallback。

## 部署

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # 填 TELEGRAM_BOT_TOKEN / ANTHROPIC_API_KEY / ADMIN_USER_IDS / GROUP_CHAT_ID
sudo cp telegram-bot.service /etc/systemd/system/
sudo systemctl enable --now telegram-bot
```

`DRY_RUN=true` 只打日志不真删真踢。

## 管理员私聊 = Claude Code 远程

**首次 setup**:bot 启动后 `journalctl -u telegram-bot` 看 stderr 里的 TOTP setup URL + ASCII QR → iPhone 用 Google Authenticator(或 Authy / 1Password)扫码,从此每 30s 一个 6 位数字。

**命令**:
| 命令 | 作用 |
|---|---|
| `/auth 123456` | 验证 TOTP → 开 24h session(消息后立即删 telegram 记录) |
| `/logout` | 主动结束 session + 清对话历史 |
| `/reset` | 仅清 Claude 对话历史,不退出 session |
| `/model` | 显当前 chat model |
| `/model opus` / `sonnet` / `haiku` | 切预设 |
| `/model <full-name>` | 切自定 model |
| `/model reset` | 恢复默认 |
| (任意私聊文本) | 调 Claude Agent SDK,带完整 tools(Bash/Read/Write/Edit/etc),session 多轮维持 |

**Session 行为**:
- 24h 有效,**每次活跃自动续 24h**(即长期使用永不过期)
- 完全不动 24h → 过期需 `/auth`
- TOTP 输错 5 次 → 锁 15min

**风险**:`permission_mode=bypassPermissions` = telegram admin 等于拿到 botuser 全部本机权限(含 NOPASSWD sudo)。TOTP 是第二道防线(防 telegram 账号被盗 / SIM swap)。**第一道防线**:bot 启动时 `.totp-secret` 文件 600 perm 只 botuser 可读 — 拿到机器读权限就拿到 TOTP secret,等于绕过 TOTP。

## LLM provider 路由

**自动模式(推荐,默认)** — `LLM_PROVIDER` 留空:
- 两 key 都配 → **openai 主**(便宜,DeepSeek 等)+ **claude 备**(primary 失败自动接手)
- 只 openai → openai 单
- 只 claude → claude 单

```
# 双配 — openai 主 / claude 备
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

failover 触发:openai 抛任何 exception(API 错 / 429 / 网络断)→ log warning → 自动重试 claude。log 看 `primary openai failed: ... — fallback to claude`。

**显式强制单 provider**(无 fallback):
```
LLM_PROVIDER=claude         # 强制 claude,即使配了 openai 也不用
LLM_PROVIDER=openai         # 强制 openai,失败不切 claude
```

**Claude 提供商详情**:Anthropic 原生 + prompt caching(cached_write ~5K tokens 复用)。
**OpenAI 兼容详情**:`OPENAI_BASE_URL` 决定 endpoint(DeepSeek / OpenAI / Together / etc),`OPENAI_MODEL` 决定 model。DeepSeek 价格比 Claude Haiku 便宜 3-5× + 中文洗钱黑话识别熟。

切换后 `update.sh` 重启即生效。

## 更新

代码改完 push GitHub 后,机器上一行:

```bash
/home/botuser/telegram-bot/update.sh
```

= `git pull --ff-only` + `.venv/bin/pip install -r requirements.txt` + `sudo systemctl restart telegram-bot` + 显前 5 行 status。

改了 `telegram-bot.service` 文件本身需要额外:`sudo cp telegram-bot.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart telegram-bot`。
改了 `.env` 不走 git,直接编辑后 `sudo systemctl restart telegram-bot`。

## 运行情况

`sudo systemctl status telegram-bot` / `journalctl -u telegram-bot -f`

判定历史落盘 `data/verdicts-YYYY-MM-DD.jsonl`(已 gitignore,含 chat / user 隐私)。
