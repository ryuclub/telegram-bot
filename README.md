# telegram-bot

群管理 bot — 新人加群 emoji 验证 + LLM 驱动消息分类拦截广告 / 洗钱黑话。

支持 **Claude**(Anthropic 原生 + prompt caching)/ **OpenAI 兼容**(含 **DeepSeek**)。env `LLM_PROVIDER` 切换。

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

## LLM provider 切换

**Claude(默认)**:
```
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
```
含 prompt caching,每次调用 cached_write ~5K tokens 命中复用。

**DeepSeek**(走 OpenAI 兼容 endpoint):
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...                  # DeepSeek API key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat             # 或 deepseek-reasoner
```
DeepSeek 价格比 Claude Haiku 便宜 3-5× + 中文洗钱黑话识别熟,但无 Anthropic-style prompt caching(DeepSeek 服务端有自动 cache,大部分 system prompt token 命中)。

**OpenAI 官方 / 其它 OpenAI 兼容服务**:同上,`OPENAI_BASE_URL` 留空 / 改成对应 endpoint,`OPENAI_MODEL` 改对应名。

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
