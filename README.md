# telegram-bot

群管理 bot — 新人加群 emoji 验证 + Claude 驱动消息分类拦截广告 / 洗钱黑话。

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

## 运行情况

`sudo systemctl status telegram-bot` / `journalctl -u telegram-bot -f`

判定历史落盘 `data/verdicts-YYYY-MM-DD.jsonl`(已 gitignore,含 chat / user 隐私)。
