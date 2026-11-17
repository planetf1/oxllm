#!/usr/bin/env bash
# 把 octolib 登录得到的 ChatGPT access token 同步进 oxllm 配置，并触发热重载。
#
#   ./oauth/sync-token.sh                  # 只同步 token
#   ./oauth/sync-token.sh --models         # 同时按当前套餐刷新 models 列表
#   ./oauth/sync-token.sh -c other.toml    # 指定配置
#
# 依赖：oauth/chatgpt-login 已编译（cargo build --release），且已完成 login。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/oxllm-chatgpt.toml"
HELPER="${HELPER:-$HERE/chatgpt-login/target/release/oxllm-chatgpt-login}"
REFRESH_MODELS=0
RELOAD_URL="${RELOAD_URL:-http://127.0.0.1:8080/reload}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) REFRESH_MODELS=1; shift ;;
    -c|--config) CONFIG="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

[[ -x "$HELPER" ]] || { echo "缺少 $HELPER，请先：cd $HERE/chatgpt-login && cargo build --release" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "配置不存在：$CONFIG" >&2; exit 1; }

TOKEN="$("$HELPER" token)"
[[ -n "$TOKEN" ]] || { echo "未取到 access_token，请先运行 $HELPER login" >&2; exit 1; }
chmod 600 "$CONFIG"

MODELS=""
if [[ "$REFRESH_MODELS" == "1" ]]; then
  MODELS="$("$HELPER" models | tail -n +2 | awk '{print $1}' | paste -sd ',' - | sed 's/,/, /g')"
  [[ -n "$MODELS" ]] || echo "warn: 没取到模型列表，保留配置里的 models" >&2
fi

CONFIG="$CONFIG" TOKEN="$TOKEN" MODELS="$MODELS" python3 - <<'PY'
import os, re, pathlib

path = pathlib.Path(os.environ["CONFIG"])
token = os.environ["TOKEN"]
models = os.environ.get("MODELS", "")
text = path.read_text()

# 只改 [[providers]] 中 chatgpt-oauth 这一段，避免误伤其它 provider
blocks = re.split(r'(?=^\[\[providers\]\])', text, flags=re.M)
updated = False
for i, block in enumerate(blocks):
    if 'name = "chatgpt-oauth"' in block.split('[virtual_models]')[0]:
        block = re.sub(r'^api_key = .*$', f'api_key = "{token}"', block, count=1, flags=re.M)
        if models:
            block = re.sub(r'^models = \[.*?\]$', f'models = [{models}]', block, count=1, flags=re.M)
        blocks[i] = block
        updated = True
        break

if not updated:
    raise SystemExit("配置里找不到 provider `chatgpt-oauth`")

path.write_text("".join(blocks))
print(f"已写入 token（{len(token)} 字符）" + (f" 与 {models}" if models else ""))
PY

if curl -fsS -m 3 -X POST "$RELOAD_URL" >/dev/null 2>&1; then
  echo "已触发热重载：$RELOAD_URL"
else
  echo "提示：oxllm 未在运行或地址不同，启动后会自动读取新配置（watch_config = true）"
fi
