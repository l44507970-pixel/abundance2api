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

可选配置：

- `ABUNDANCE_BASE_URL`：默认 `https://a.b.u.n.dance`。
- `ABUNDANCE_DEFAULT_SPEED`：默认 `default`。
- `ABUNDANCE_DEFAULT_INTELLIGENCE`：默认 `standard`。
- `ABUNDANCE_OIDC_TOKEN`：可选，通常不建议作为唯一登录凭据。
- `ABUNDANCE_REQUEST_TIMEOUT_SECONDS`：默认 `120`。
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
