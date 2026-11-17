# oxllm 接入 OpenAI OAuth（ChatGPT 订阅）

用 ChatGPT 账号（Plus / Pro / Team / Enterprise）的订阅额度替代 `sk-...` API key 走 oxllm。
登录与凭据刷新交给 [octolib](https://github.com/muvon/octolib) 的 `chatgpt` 提供方，
oxllm 只负责代理、路由和电路熔断。

## 为什么这么接

octolib 的 `chatgpt` 提供方实现的路径是：

| 环节 | 做法 |
|---|---|
| 登录 | `chatgpt::start_login`，OAuth2 authorization code + PKCE，loopback 回调，scope 含 `chatgpt.tokens.use.direct`（按 ChatGPT 套餐计费） |
| 凭据 | `~/.config/octolib/chatgpt/auth.json`（access/refresh token、client_id、expires_at） |
| 刷新 | 临近过期（提前 300s）或遇到 401 时用 refresh_token 换新并回写 |
| 上游 | `https://api.openai.com/v1/responses`，仅 `Authorization: Bearer` |
| 请求约定 | body 必须 `"store": false`，流式 SSE |

oxllm 侧的对应关系很直接：`base_url = https://api.openai.com/v1/` + provider `api = "openai"`，
它的 `/v1/responses` 路由会拼成 `.../v1/responses`，鉴权就是 Bearer —— 所以**不需要改 oxllm 代码**，
只需要把 token 写进配置。

## 目录

```
oauth/
├── chatgpt-login/           # Rust 助手（依赖 octolib），负责 login / token / models / status
├── oxllm-chatgpt.toml       # oxllm 配置模板（provider chatgpt-oauth + 虚拟模型 gpt）
└── sync-token.sh            # 取 token → 写入配置 → 触发热重载
```

## 步骤

### 1. 编译助手

```bash
cd oauth/chatgpt-login && cargo build --release
# 产物：oauth/chatgpt-login/target/release/oxllm-chatgpt-login
```

依赖的 octolib 在 `Cargo.toml` 里固定了 rev（`4b4f855`），所以没有提交助手的 `Cargo.lock`；
需要升级 octolib 时改 rev 再重新构建。

### 2. 登录（浏览器授权）

```bash
./oauth/chatgpt-login/target/release/oxllm-chatgpt-login login
```

它会打印授权 URL，并监听 `127.0.0.1:<随机端口>/callback`。**回调必须回到运行该命令的机器**：

- 在 oxllm 所在的机器上直接跑，浏览器也在同一台机器：直接开 URL；
- 远端/容器里跑：先按打印出的端口做转发，例如
  `ssh -L 1455:127.0.0.1:1455 user@host`，再在本地浏览器开 URL；
- 也可以在**自己的笔记本**上先登录，再把 `~/.config/octolib/chatgpt/auth.json` 拷到 oxllm 机器同一路径。
  千万不要把 auth.json 提交到仓库。

登录成功会打印账号邮箱与过期时间，凭据落在 `~/.config/octolib/chatgpt/auth.json`。

核对凭据（不发请求）：

```bash
./oauth/chatgpt-login/target/release/oxllm-chatgpt-login status
```

### 3. 选模型 slug

```bash
./oauth/chatgpt-login/target/release/oxllm-chatgpt-login models
```

把 `oxllm-chatgpt.toml` 里两处 `gpt-5.2-codex` 换成列表里的 slug（`models = [...]` 和
`[virtual_models]` 的 target）。套餐不同可用模型不同，不要照抄示例。

### 4. 写入 token 并热重载

```bash
./oauth/sync-token.sh              # 只同步 token
./oauth/sync-token.sh --models     # 顺手按当前套餐重写 models 列表
```

脚本做的事：调用 `token`（顺带触发 octolib 的到期刷新）→ 把 provider `chatgpt-oauth` 的 `api_key`
替换成新 token → 配置 `chmod 600` → `POST /reload`（oxllm 未运行时跳过，`watch_config = true`
会在下次启动时读到新值）。

### 5. 启动 oxllm

```bash
./target/release/oxllm validate -c oauth/oxllm-chatgpt.toml
./target/release/oxllm serve -c oauth/oxllm-chatgpt.toml
```

### 6. 调用

只走 **Responses API**（`/v1/responses`），并且 body 必须带 `"store": false`：

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/responses \
  -H 'content-type: application/json' \
  -d '{"model":"gpt","input":"用一句话介绍你自己","store":false,"stream":true}'
```

OpenAI SDK（Python）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="oauth")  # 占位 key，鉴权由 oxllm 注入
resp = client.responses.create(model="gpt", input="hello", store=False)
print(resp.output_text)
```

## 生命周期与自动化

- access token 有效期有限（octolib 提前 300s 刷新），refresh token 可长期续期。
- 定时续期（示例，每 6 小时）：

```cron
0 */6 * * * /path/to/oxllm/oauth/sync-token.sh >/dev/null 2>&1
```

- 若 refresh 失败（改了密码、撤销授权等），重新跑 `login` 即可。

## 注意

- **不要用 `/v1/chat/completions` 打这个 provider**：该 token 的套餐额度只覆盖 Responses API，
  Chat Completions 会 401/403。客户端请走 `/v1/responses`。
- `store: false` 是硬要求；oxllm 原样转发 body，不会替你补字段。
- token 以明文写在配置文件里，脚本已 `chmod 600`；仓库里的模板保持 `OAUTH_TOKEN_PLACEHOLDER`，
  建议把本地改过的配置加入 `.gitignore`。
- 这是把个人订阅额度接进本地代理，仅限本人使用；多账号/共享/转售不属于该方案的适用范围。
- 相关上游行为以 octolib 与 OpenAI 文档为准；这里只做配置和凭据同步，不绕过任何配额或鉴权。

## 已知限制（本机实测）

当前这台机器所在网络对 OpenAI 是不可用的，所以本仓库里只能验证到「配置合法 + 路由与凭据同步链路正确」，
真正的登录和推理要在能访问 OpenAI 的机器/网络上完成：

- `https://auth.openai.com` → 403 `unsupported_country_region_territory`（地区不支持，登录与刷新都会失败）
- `https://api.openai.com/v1/responses` → 连接超时（请求会挂在 oxllm 的上游超时上）

判定方法：

```bash
curl -sS -m 8 -o /dev/null -w '%{http_code}\n' https://auth.openai.com/
curl -sS -m 8 -o /dev/null -w '%{http_code}\n' https://api.openai.com/v1/models
```

两个都通（返回非 000/403）再按上面的步骤走。如果你的 oxllm 常驻在不支持的地区，就需要让
`login` / `sync-token.sh` 在支持地区的机器上执行，或让 oxllm 的上游出口走可达的网络路径，
并且注意 `sync-token.sh` 的刷新同样需要该网络环境。

## 已验证 / 未验证

| 项目 | 状态 |
|---|---|
| `oxllm validate -c oauth/oxllm-chatgpt.toml` | ✅ VALID |
| 助手编译（octolib 0.54，feature 仅 `llm`） | ✅ 通过 |
| `token` → 写配置 → 热重载（伪造凭据） | ✅ 通过 |
| oxllm 路由到 `/v1/responses` 并向上游发起请求 | ✅ 触发（上游因地区限制无响应） |
| 真实登录 + 真实推理 | ⛔ 需在支持地区的网络下执行 |
