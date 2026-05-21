#!/usr/bin/env bash
# 拉新代码 + 装依赖 + 重启 systemd 服务 + 显状态。
# 改了 telegram-bot.service 本身需要额外手动 sudo cp + daemon-reload(脚本不动 systemd unit)。
set -euo pipefail
cd "$(dirname "$0")"

echo "▶ git pull"
git pull --ff-only

echo "▶ pip install(若 requirements 变化)"
.venv/bin/pip install -q -r requirements.txt

echo "▶ systemctl restart telegram-bot"
sudo systemctl restart telegram-bot

echo "▶ status"
sudo systemctl status telegram-bot --no-pager | head -5
