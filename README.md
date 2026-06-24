# a.b.u.n.dance2api

OpenAI 兼容格式的 FastAPI 代理服务，后端请求转发到 a.b.u.n.dance。

## 本地运行

1. 复制环境变量示例：

```powershell
Copy-Item .env.example .env
```

2. 填写 `.env` 中的 `DEFAULT_API_KEY` 和上游登录 Cookie。

3. 启动服务：

```powershell
uv run uvicorn main:app --host 0.0.0.0 --port 18000
```

4. 检查服务：

```powershell
Invoke-RestMethod http://127.0.0.1:18000/healthz
```

## Zeabur 部署

项目根目录已经提供 `Dockerfile`，Zeabur 会优先按 Dockerfile 构建。

在 Zeabur 服务的环境变量里至少配置：

- `DEFAULT_API_KEY`：客户端访问本代理时使用的 Bearer token，必须配置，建议使用长随机值。
- `ABUNDANCE_COOKIE` 或 `ABUNDANCE_SESSION`：从 a.b.u.n.dance 登录态 Cookie 中复制。

推荐直接配置 `ABUNDANCE_COOKIE`，值为浏览器 Network 面板里的完整 `Cookie`
请求头。也可以只配置 `ABUNDANCE_SESSION`。

`ABUNDANCE_OIDC_TOKEN` 是可选项。它通常约 1 小时过期，本服务会在它过期
或被上游拒绝时自动尝试使用 `session-only` 请求。

多账号轮询可用 `ABUNDANCE_ACCOUNTS_JSON`：

```json
[
  {"name": "account-a", "cookie": "session=xxx; oidc_id_token=xxx"},
  {"name": "account-b", "cookie": "session=yyy; oidc_id_token=yyy"}
]
```

也可以用 `ABUNDANCE_COOKIE_1`、`ABUNDANCE_COOKIE_2` 这种编号环境变量。
新建的上游会话会按账号轮询；同一个已缓存会话会继续使用原账号。

可选配置：

- `ABUNDANCE_BASE_URL`：默认 `https://a.b.u.n.dance`。
- `ABUNDANCE_SEND_TUNING_FIELDS`：默认 `false`，通常不要开启。
- `ABUNDANCE_DEFAULT_SPEED`：默认 `standard`，可选 `standard`、`extended`。
- `ABUNDANCE_DEFAULT_INTELLIGENCE`：默认 `medium`，可选 `minimal`、`low`、`medium`、`high`。
- `ABUNDANCE_OIDC_TOKEN`：可选，通常不建议作为唯一登录凭据。
- `ABUNDANCE_REQUEST_TIMEOUT_SECONDS`：默认 `120`。
- `ABUNDANCE_CONNECT_KEEPALIVE_SECONDS`：默认 `15`，等待上游建立流式连接时发送 SSE keep-alive。
- `ABUNDANCE_CONNECT_WORKERS`：默认 `8`，等待上游连接的后台线程数。
- `MAX_FULL_PROMPT_CHARS`：默认 `32000`，冷启动全量历史超过后会自动压缩。
- `MAX_FULL_PROMPT_RECENT_MESSAGES`：默认 `12`，长历史压缩时保留的最近消息数。
- `MAX_FULL_PROMPT_MESSAGE_CHARS`：默认 `4000`，单条历史消息超过后会裁剪。
- `MAX_UPSTREAM_CONTENT_CHARS`：默认 `32000`，正常尝试发给上游的正文硬上限。
- `MAX_UPSTREAM_RETRY_CONTENT_CHARS`：默认 `16000`，长上下文被上游拒绝时自动降级重试的正文上限。角色扮演场景可以把 `MAX_UPSTREAM_CONTENT_CHARS` 调到 `40000` 或 `48000`，但建议保留这个降级值。
- `ABUNDANCE_HTTP_PROXY`：需要代理访问上游时设置。
- `PORT`：Zeabur 通常会自动注入；未设置时默认 `18000`。

部署后可访问：

- 健康检查：`https://你的域名/healthz`
- 模型列表：`https://你的域名/v1/models`
- 聊天接口：`https://你的域名/v1/chat/completions`

`/v1/*` 接口需要请求头：

```http
Authorization: Bearer 你的 DEFAULT_API_KEY
```

如果没有配置 `DEFAULT_API_KEY`，`/v1/*` 接口会拒绝访问。
