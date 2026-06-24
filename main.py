from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict, Any, Literal, Union
import curl_cffi.requests
import concurrent.futures
import time
import json
import uuid
import re
import shlex
import logging
import os
import hashlib
import secrets
import base64
from http.cookies import SimpleCookie
import xml.etree.ElementTree as ET
from pathlib import Path

app = FastAPI(title="OpenAI Compatible API (a.b.u.n.dance)", version="1.0.0")
logger = logging.getLogger("uvicorn.error")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


load_env_file(Path(__file__).with_name(".env"))


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# 代理自身的 API Key 校验（客户端 -> 本代理）
DEFAULT_API_KEY = os.getenv("DEFAULT_API_KEY", "").strip()
ACCEPT_ANY_API_KEY = os.getenv("ACCEPT_ANY_API_KEY", "false").lower() not in {"0", "false", "no"}

# a.b.u.n.dance 上游配置（本代理 -> a.b.u.n.dance）
ABUNDANCE_BASE_URL = os.getenv("ABUNDANCE_BASE_URL", "https://a.b.u.n.dance").rstrip("/")
ABUNDANCE_RAW_COOKIE = os.getenv("ABUNDANCE_COOKIE", "").strip()
ABUNDANCE_SESSION_COOKIE = os.getenv("ABUNDANCE_SESSION", "").strip()
ABUNDANCE_OIDC_TOKEN = os.getenv("ABUNDANCE_OIDC_TOKEN", "").strip()
ABUNDANCE_DEFAULT_SPEED = os.getenv("ABUNDANCE_DEFAULT_SPEED", "default")
ABUNDANCE_DEFAULT_INTELLIGENCE = os.getenv("ABUNDANCE_DEFAULT_INTELLIGENCE", "standard")
ABUNDANCE_IMPERSONATE = os.getenv("ABUNDANCE_IMPERSONATE", "chrome131")
ABUNDANCE_REQUEST_TIMEOUT_SECONDS = env_int("ABUNDANCE_REQUEST_TIMEOUT_SECONDS", 120)
ABUNDANCE_MAX_STATUS_RETRIES = env_int("ABUNDANCE_MAX_STATUS_RETRIES", 2)
ABUNDANCE_RETRY_BASE_DELAY_SECONDS = 0.5
ABUNDANCE_CONNECT_KEEPALIVE_SECONDS = max(1, env_int("ABUNDANCE_CONNECT_KEEPALIVE_SECONDS", 15))
ABUNDANCE_CONNECT_WORKERS = max(1, env_int("ABUNDANCE_CONNECT_WORKERS", 8))
ABUNDANCE_USER_AGENT = os.getenv(
    "ABUNDANCE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
)
UPSTREAM_CONNECT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=ABUNDANCE_CONNECT_WORKERS)

# 提示词注入模式下的工具调用触发信号（上游不支持原生 function calling）
TOOL_TRIGGER_SIGNAL = "<<<abundance_tool_call>>>"

# 支持的模型列表（来自 docs/对话请求.md）
AVAILABLE_MODELS = [
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.5",
    "openai/gpt-5.5-pro",
    "openai/gpt-5.4",
    "openai/gpt-5.2-codex",
    "openai/gpt-5.2",
    "openai/gpt-5.2-pro",
    "google-ai-studio/gemini-3.5-flash",
    "google-ai-studio/gemini-3.1-pro-preview",
    "grok/grok-4.3",
    "grok/grok-4-fast",
]

SUPPORTED_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# 会话 -> 上游会话的增量发送追踪（仅缓存指纹，不缓存正文，重启后自动回退到全量发送）
MAX_FINGERPRINT_SESSIONS = env_int("MAX_FINGERPRINT_SESSIONS", 500)
SESSION_FINGERPRINTS: Dict[str, List[str]] = {}

# 会话 -> 上游真实 conversation id 的缓存。上游没有"按 id 自动建会话"的语义，
# 必须先调用 POST /api/conversations 拿到真实 id 才能继续对话，所以这里缓存的是
# 已经真实创建过的会话，而不是本地随便派生的 UUID。
CONVERSATION_CACHE: Dict[str, str] = {}


# ===================== OpenAI 兼容的请求模型 =====================

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class Tool(BaseModel):
    type: Literal["function"]
    function: ToolFunction
    source_type: Optional[str] = None


class ToolChoice(BaseModel):
    type: Literal["function"]
    function: Dict[str, str]


class Message(BaseModel):
    role: str
    content: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    user: Optional[str] = None
    safety_identifier: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    client_metadata: Optional[Dict[str, Any]] = None
    prompt_cache_key: Optional[str] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")


# ===================== API Key 验证（代理自身） =====================

def is_valid_api_key(api_key: str) -> bool:
    if DEFAULT_API_KEY and secrets.compare_digest(api_key, DEFAULT_API_KEY):
        return True
    return ACCEPT_ANY_API_KEY and api_key.startswith("sk-")


def verify_api_key(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    api_key = authorization[7:]
    if not is_valid_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


# ===================== 会话身份识别（用于增量发送 + 上游会话 id 派生） =====================

def normalize_identifier(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            text = str(value)
        text = text.strip()
    return text or None


def first_identifier(*values: Any) -> Optional[str]:
    for value in values:
        identifier = normalize_identifier(value)
        if identifier:
            return identifier
    return None


def request_extra_value(request_payload: Any, key: str) -> Any:
    value = getattr(request_payload, key, None)
    if value is not None:
        return value

    extra = getattr(request_payload, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(key)
    return None


def request_mapping(request_payload: Any, key: str) -> Dict[str, Any]:
    value = request_extra_value(request_payload, key)
    return value if isinstance(value, dict) else {}


def mapping_value(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            identifier = normalize_identifier(mapping[key])
            if identifier:
                return identifier
    return None


def request_header(http_request: Request, name: str) -> Optional[str]:
    return normalize_identifier(http_request.headers.get(name))


def safe_headers_for_log(headers: Any) -> Dict[str, str]:
    sensitive_headers = {"authorization", "x-api-key", "cookie", "set-cookie", "proxy-authorization"}
    safe: Dict[str, str] = {}
    for key, value in headers.items():
        safe[key] = "<redacted>" if key.lower() in sensitive_headers else str(value)
    return safe


def resolve_request_user_id(request_payload: Any, http_request: Request) -> Optional[str]:
    metadata = request_mapping(request_payload, "metadata")
    client_metadata = request_mapping(request_payload, "client_metadata")
    return first_identifier(
        request_header(http_request, "x-user-id"),
        request_extra_value(request_payload, "safety_identifier"),
        request_extra_value(request_payload, "user"),
        mapping_value(metadata, "user_id", "user", "end_user_id", "account_id"),
        mapping_value(client_metadata, "user_id", "user", "end_user_id", "account_id"),
        request_header(http_request, "x-codex-installation-id"),
        mapping_value(client_metadata, "x-codex-installation-id", "codex_installation_id"),
        mapping_value(metadata, "x-codex-installation-id", "codex_installation_id")
    )


def resolve_request_conversation_id(request_payload: Any, http_request: Request) -> Optional[str]:
    metadata = request_mapping(request_payload, "metadata")
    client_metadata = request_mapping(request_payload, "client_metadata")
    return first_identifier(
        request_header(http_request, "x-claude-code-session-id"),
        request_header(http_request, "x-session-id"),
        request_header(http_request, "x-conversation-id"),
        request_extra_value(request_payload, "conversation_id"),
        request_extra_value(request_payload, "session_id"),
        request_extra_value(request_payload, "thread_id"),
        mapping_value(metadata, "conversation_id", "session_id", "thread_id"),
        request_extra_value(request_payload, "prompt_cache_key"),
        mapping_value(client_metadata, "prompt_cache_key", "x-codex-window-id", "codex_window_id", "window_id"),
        mapping_value(metadata, "prompt_cache_key", "x-codex-window-id", "codex_window_id", "window_id"),
        request_header(http_request, "x-codex-window-id"),
        request_header(http_request, "x-client-request-id"),
        request_header(http_request, "x-codex-parent-thread-id")
    )


def request_client_hint(request_payload: Any, http_request: Request) -> Optional[str]:
    metadata = request_mapping(request_payload, "metadata")
    client_metadata = request_mapping(request_payload, "client_metadata")
    return first_identifier(
        request_header(http_request, "x-title"),
        request_header(http_request, "http-referer"),
        request_header(http_request, "user-agent"),
        mapping_value(client_metadata, "app", "client", "source"),
        mapping_value(metadata, "app", "client", "source")
    )


def has_conversation_context(messages: List[Message]) -> bool:
    user_count = sum(1 for msg in messages if msg.role == "user" and content_to_text(msg.content))
    has_assistant_or_tool = any(msg.role in {"assistant", "tool", "function"} for msg in messages)
    return has_assistant_or_tool or user_count > 1


def derive_transcript_conversation_id(
    request_payload: Any,
    http_request: Request,
    messages: Optional[List[Message]]
) -> Optional[str]:
    if not messages or not has_conversation_context(messages):
        return None

    seed_messages = [
        {
            "role": msg.role,
            "content": content_to_text(msg.content),
            "name": msg.name or "",
            "tool_call_id": msg.tool_call_id or ""
        }
        for msg in messages
        if msg.role in {"user", "assistant"} and content_to_text(msg.content)
    ][:2]
    if len(seed_messages) < 2:
        return None

    seed = {
        "client": request_client_hint(request_payload, http_request) or "unknown-client",
        "messages": seed_messages
    }
    seed_text = json.dumps(seed, ensure_ascii=False, sort_keys=True)
    return f"transcript_{uuid.uuid5(uuid.NAMESPACE_URL, f'abundance-ai-transcript:{seed_text}').hex}"


def resolve_session_identity(
    request_payload: Any,
    http_request: Request,
    messages: Optional[List[Message]] = None
) -> tuple[Optional[str], str, bool]:
    """返回 (user_id, conversation_id, is_stable)。

    is_stable 表示 conversation_id 不是逐请求随机生成的兜底值——无论它来自显式的
    header/字段，还是从前两条消息内容派生的 transcript 哈希，都算稳定，可以用来做
    跨请求的增量缓存与上游会话 id 派生。
    """
    user_id = resolve_request_user_id(request_payload, http_request)
    conversation_id = resolve_request_conversation_id(request_payload, http_request)
    is_stable = conversation_id is not None
    if conversation_id is None:
        conversation_id = derive_transcript_conversation_id(request_payload, http_request, messages)
        is_stable = conversation_id is not None
    if conversation_id is None:
        conversation_id = f"request_{uuid.uuid4().hex}"
    return user_id, conversation_id, is_stable


def request_session_key(
    protocol: str,
    model: str,
    api_key: Optional[str],
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> str:
    tenant_id = api_key or DEFAULT_API_KEY
    return (
        f"protocol={protocol}:"
        f"tenant={tenant_id}:"
        f"user={user_id or 'anonymous'}:"
        f"conversation={conversation_id or 'legacy'}:"
        f"model={model}"
    )


# ===================== 多模态 content -> 纯文本 =====================

def image_placeholder(media_type: str, source_type: str) -> str:
    return f"[image attachment: {media_type or 'image'} via {source_type or 'unknown'}]"


def image_attachment_from_anthropic_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if block.get("type") != "image":
        return None

    source = block.get("source")
    if not isinstance(source, dict):
        return None

    source_type = normalize_identifier(source.get("type")) or "unknown"
    media_type = normalize_identifier(source.get("media_type")) or "image/png"
    if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
        return None

    if source_type == "base64":
        data = normalize_identifier(source.get("data"))
        if not data:
            return None
        return {"type": "image", "source_type": "base64", "media_type": media_type, "data": data}

    if source_type == "url":
        url = normalize_identifier(source.get("url"))
        if not url:
            return None
        return {"type": "image", "source_type": "url", "media_type": media_type, "url": url}

    return None


def image_attachment_from_openai_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item_type = block.get("type")
    if item_type not in {"image_url", "input_image"}:
        return None

    image_url = block.get("image_url") or block.get("url") or ""
    if isinstance(image_url, dict):
        image_url = image_url.get("url", "")
    image_url = normalize_identifier(image_url)
    if not image_url:
        return None

    match = re.match(r"^data:([^;]+);base64,(.+)$", image_url, flags=re.IGNORECASE | re.DOTALL)
    if match:
        media_type = match.group(1).lower()
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            return None
        return {"type": "image", "source_type": "base64", "media_type": media_type, "data": match.group(2)}

    return {"type": "image", "source_type": "url", "media_type": "image/png", "url": image_url}


def content_to_text_and_image_attachments(content: Any) -> tuple[str, List[Dict[str, Any]]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        try:
            return json.dumps(content, ensure_ascii=False), []
        except TypeError:
            return str(content), []

    parts: List[str] = []
    attachments: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            parts.append(str(item))
            continue

        item_type = item.get("type")
        if item_type in {"text", "input_text", "output_text"}:
            parts.append(str(item.get("text", "")))
        elif item_type in {"image_url", "input_image"}:
            attachment = image_attachment_from_openai_block(item)
            if attachment:
                attachments.append(attachment)
                parts.append(image_placeholder(attachment.get("media_type", "image"), attachment.get("source_type", "")))
            else:
                image_url = item.get("image_url") or item.get("url") or ""
                if isinstance(image_url, dict):
                    image_url = image_url.get("url", "")
                if image_url:
                    parts.append(f"[image_url: {image_url}]")
        elif item_type == "image":
            attachment = image_attachment_from_anthropic_block(item)
            if attachment:
                attachments.append(attachment)
                parts.append(image_placeholder(attachment.get("media_type", "image"), attachment.get("source_type", "")))
            else:
                source = item.get("source", {})
                if isinstance(source, dict):
                    parts.append(image_placeholder(source.get("media_type", "image"), source.get("type", "unknown")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
        elif item_type == "tool_result":
            result_text, result_attachments = content_to_text_and_image_attachments(item.get("content", ""))
            if result_text:
                parts.append(result_text)
            attachments.extend(result_attachments)
        else:
            parts.append(json.dumps(item, ensure_ascii=False))

    return "\n".join(part for part in parts if part), attachments


def content_to_text(content: Any) -> str:
    text, _attachments = content_to_text_and_image_attachments(content)
    return text


# ===================== 工具调用：提示词注入 =====================

def build_tool_call_index_from_messages(messages: List[Message]) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for msg in messages:
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        for tool_call in msg.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            function_info = tool_call.get("function", {})
            if not tool_call_id or not isinstance(function_info, dict):
                continue

            arguments = function_info.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            name = function_info.get("name", "")
            if name:
                index[tool_call_id] = {"name": name, "arguments": arguments}
    return index


def format_assistant_tool_calls_for_prompt(tool_calls: List[Dict[str, Any]]) -> str:
    lines = ["Assistant tool calls:"]
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_info = tool_call.get("function", {})
        if not isinstance(function_info, dict):
            continue
        arguments = function_info.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        lines.append(
            f"- id: {tool_call.get('id', '')}\n"
            f"  name: {function_info.get('name', '')}\n"
            f"  arguments: {arguments}"
        )
    return "\n".join(lines)


def validate_tools(tools: List[Tool]) -> None:
    seen = set()
    for tool in tools:
        name = tool.function.name
        if not name:
            raise HTTPException(status_code=400, detail="Tool function name must be non-empty")
        if name in seen:
            raise HTTPException(status_code=400, detail=f"Duplicate tool function name: {name}")
        seen.add(name)

        parameters = tool.function.parameters or {}
        if not isinstance(parameters, dict):
            raise HTTPException(status_code=400, detail=f"Tool '{name}' parameters must be an object")

        properties = parameters.get("properties", {})
        if properties is None:
            properties = {}
        if not isinstance(properties, dict):
            raise HTTPException(status_code=400, detail=f"Tool '{name}' parameters.properties must be an object")

        required = parameters.get("required", [])
        if required is None:
            required = []
        if not isinstance(required, list):
            raise HTTPException(status_code=400, detail=f"Tool '{name}' parameters.required must be a list")
        invalid_required = [item for item in required if not isinstance(item, str)]
        if invalid_required:
            raise HTTPException(status_code=400, detail=f"Tool '{name}' required entries must be strings")

        missing = [item for item in required if item not in properties]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Tool '{name}' required parameters {missing} are not defined in properties"
            )


def tool_choice_to_dict(tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]]) -> Any:
    if isinstance(tool_choice, BaseModel):
        return tool_choice.model_dump(exclude_none=True)
    return tool_choice


def is_tool_choice_none(tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]]) -> bool:
    return tool_choice_to_dict(tool_choice) == "none"


def tool_choice_prompt(tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]], tools: List[Tool]) -> str:
    choice = tool_choice_to_dict(tool_choice)
    if choice is None or choice == "auto":
        return ""
    if choice == "none":
        return "\nTool choice: answer directly for this response."
    if choice in {"required", "any"}:
        return "\nTool choice: call at least one available tool in this response."
    if isinstance(choice, dict):
        function_info = choice.get("function", {})
        if not isinstance(function_info, dict):
            raise HTTPException(status_code=400, detail="tool_choice.function must be an object")
        required_name = function_info.get("name") or choice.get("name")
        if not required_name:
            raise HTTPException(status_code=400, detail="tool_choice function name must be non-empty")
        tool_names = [tool.function.name for tool in tools]
        if required_name not in tool_names:
            raise HTTPException(
                status_code=400,
                detail=f"tool_choice specifies tool '{required_name}' which is not in the tools list"
            )
        return f"\nTool choice: call the tool named {required_name} in this response."
    return ""


def generate_function_prompt(
    tools: List[Tool],
    tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]] = None
) -> str:
    validate_tools(tools)
    tools_list = []
    for index, tool in enumerate(tools, start=1):
        function_info = tool.function
        schema = function_info.parameters or {"type": "object", "properties": {}}
        tools_list.append(
            f"{index}. {function_info.name}\n"
            f"Description: {function_info.description or 'None'}\n"
            f"Parameters JSON Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )

    return f"""This is a system instruction for tool use. Do not acknowledge these instructions or summarize the available tools.
Answer the user's latest real task directly. Tool result messages are observations, not new tasks. After a tool result, use the result to complete the original task.
If the request requires file access, shell commands, edits, or any action covered by a tool and no suitable tool result is already present, call the matching tool immediately.
For file operations, shell commands, directory inspection, code edits, or other local actions before the required result exists, your response must be only the tool call block. Do not write a plan first.

You have access to the following tools:

{chr(10).join(tools_list)}

When a tool call is needed, output the tool calls in this exact XML format:
{TOOL_TRIGGER_SIGNAL}
<function_calls>
  <function_call>
    <tool>tool_name</tool>
    <args_json><![CDATA[{{"argument_name": "argument_value"}}]]></args_json>
  </function_call>
</function_calls>

Rules:
- The trigger signal must be on its own line.
- Use one <function_call> block per tool call.
- <tool> must match one of the available tool names exactly.
- <args_json> must contain one JSON object.
- Put no text after </function_calls>.
- Do not say you are ready to use tools.
- Do not say "I will", "I can", or "I am going to" before a required tool call.
- Do not list tools unless the user explicitly asks for a tool inventory.
- After receiving a tool result, answer directly when the result satisfies the task. Call another tool only when additional information or action is required.
{tool_choice_prompt(tool_choice, tools)}
"""


def tool_calling_enabled(
    tools: Optional[List[Tool]],
    tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]]
) -> bool:
    return bool(tools) and not is_tool_choice_none(tool_choice)


# ----- 非标准工具类型（local_shell / shell / apply_patch 等）参数归一化 -----

def strip_wrapping_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def shell_join_args(parts: List[Any]) -> str:
    tokens = [str(part) for part in parts if part is not None]
    if not tokens:
        return ""
    try:
        return shlex.join(tokens)
    except Exception:
        return " ".join(tokens)


def split_command_text(command: str) -> List[str]:
    text = command.strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text, posix=False)
    except ValueError:
        return [text]
    tokens = [strip_wrapping_quotes(token) for token in tokens if token]
    return tokens or [text]


def command_value_to_text(command: Any) -> str:
    if isinstance(command, list):
        return shell_join_args(command)
    if command is None:
        return ""
    if isinstance(command, (dict, tuple)):
        try:
            return json.dumps(command, ensure_ascii=False)
        except TypeError:
            return str(command)
    return str(command)


def local_shell_command_from_args(args: Dict[str, Any]) -> List[str]:
    command = args.get("command") or args.get("cmd") or args.get("script") or args.get("input")
    if isinstance(command, list):
        return [str(part) for part in command if part is not None]
    if isinstance(command, str):
        return split_command_text(command)
    if command is not None:
        return [str(command)]
    commands = args.get("commands")
    if isinstance(commands, list) and commands:
        joined = " && ".join(command_value_to_text(item) for item in commands)
        return split_command_text(joined)
    return split_command_text(json.dumps(args, ensure_ascii=False))


def normalize_parsed_tool_args(parsed_tool: Dict[str, Any], tool: Tool) -> None:
    args = parsed_tool.get("args")
    if not isinstance(args, dict):
        return

    source_type = (tool.source_type or tool.function.name or "").lower()
    if source_type != "local_shell" and tool.function.name != "local_shell":
        return

    command = args.get("command")
    if isinstance(command, str):
        args["command"] = split_command_text(command)
    elif command is None and any(key in args for key in ("cmd", "script", "input", "commands")):
        args["command"] = local_shell_command_from_args(args)

    if "working_directory" not in args:
        workdir = args.get("workdir") or args.get("cwd")
        if workdir:
            args["working_directory"] = workdir


# ===================== 消息历史 -> 单条文本 prompt（全量路径） =====================

def build_tool_result_content(
    tool_name: str,
    tool_arguments: str,
    content: str,
    tool_call_id: Optional[str] = None,
    original_task: Optional[str] = None
) -> str:
    header = f"Tool result for {tool_name or 'tool'}"
    if tool_call_id:
        header += f" (tool_call_id: {tool_call_id})"

    parts = [
        "[Tool Result]",
        header,
        f"Arguments: {tool_arguments or '{}'}",
        f"Result:\n{content}",
    ]
    if original_task:
        parts.extend([
            "",
            "[Continuation Instruction]",
            f"Original user task: {original_task}",
            "Use this tool result to continue the original task. If the result satisfies the task, answer directly. If the task is still incomplete, call the next required tool immediately."
        ])
    else:
        parts.extend([
            "",
            "[Continuation Instruction]",
            "Use this tool result to continue the previous user task. If the result satisfies the task, answer directly. If the task is still incomplete, call the next required tool immediately."
        ])

    return "\n".join(parts)


def messages_to_prompt(
    messages: List[Message],
    tools: Optional[List[Tool]] = None,
    tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]] = None
) -> str:
    prompt_parts = []
    if tools:
        prompt_parts.append(generate_function_prompt(tools, tool_choice))

    tool_call_index = build_tool_call_index_from_messages(messages)

    latest_user_task: Optional[str] = None
    for msg in messages:
        content = content_to_text(msg.content)
        if msg.role in {"system", "developer"}:
            if content:
                prompt_parts.append(f"System: {content}")
        elif msg.role == "user":
            if content:
                latest_user_task = content
            prompt_parts.append(f"User: {content}")
        elif msg.role == "assistant":
            if content:
                prompt_parts.append(f"Assistant: {content}")
            if msg.tool_calls:
                prompt_parts.append(format_assistant_tool_calls_for_prompt(msg.tool_calls))
        elif msg.role == "tool":
            tool_name = msg.name or ""
            tool_arguments = ""
            if msg.tool_call_id and msg.tool_call_id in tool_call_index:
                indexed_call = tool_call_index[msg.tool_call_id]
                tool_name = tool_name or indexed_call["name"]
                tool_arguments = indexed_call["arguments"]

            prompt_parts.append(build_tool_result_content(
                tool_name, tool_arguments, content, msg.tool_call_id, latest_user_task
            ))
        elif msg.role == "function":
            prompt_parts.append(f"Function {msg.name or 'function'}: {content}")
        else:
            prompt_parts.append(f"{msg.role.title()}: {content}")
    return "\n\n".join(part for part in prompt_parts if part)


# ===================== 增量发送（按会话缓存指纹） =====================

def message_fingerprint(msg: Message) -> str:
    payload = {
        "role": msg.role,
        "content": content_to_text(msg.content),
        "tool_calls": msg.tool_calls,
        "tool_call_id": msg.tool_call_id,
        "name": msg.name,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def message_fingerprints(messages: List[Message]) -> List[str]:
    return [message_fingerprint(msg) for msg in messages]


def latest_user_task_text(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            content = content_to_text(msg.content).strip()
            if content:
                return content
    return ""


def build_incremental_content(
    full_messages: List[Message],
    new_messages: List[Message],
    tools: Optional[List[Tool]],
    tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]]
) -> str:
    """仅把"新出现"的 user/tool/system 消息拼成增量文本；assistant 消息跳过——
    上游会话本身已经记住了它刚生成的那一轮回复，没必要回传。"""
    parts: List[str] = []
    if tools:
        parts.append(generate_function_prompt(tools, tool_choice))

    tool_call_index = build_tool_call_index_from_messages(full_messages)
    latest_task = latest_user_task_text(full_messages)

    for msg in new_messages:
        if msg.role == "assistant":
            continue
        content = content_to_text(msg.content)
        if msg.role in {"system", "developer"}:
            if content:
                parts.append(f"System: {content}")
        elif msg.role == "user":
            parts.append(f"User: {content}")
        elif msg.role in {"tool", "function"}:
            tool_name = msg.name or ""
            tool_arguments = ""
            if msg.tool_call_id and msg.tool_call_id in tool_call_index:
                indexed_call = tool_call_index[msg.tool_call_id]
                tool_name = tool_name or indexed_call["name"]
                tool_arguments = indexed_call["arguments"]
            parts.append(build_tool_result_content(tool_name, tool_arguments, content, msg.tool_call_id, latest_task))
        else:
            parts.append(f"{msg.role.title()}: {content}")

    return "\n\n".join(part for part in parts if part)


def remember_fingerprints(session_key: Optional[str], fingerprints: List[str]) -> None:
    if not session_key:
        return
    if session_key not in SESSION_FINGERPRINTS and len(SESSION_FINGERPRINTS) >= MAX_FINGERPRINT_SESSIONS:
        SESSION_FINGERPRINTS.pop(next(iter(SESSION_FINGERPRINTS)), None)
    SESSION_FINGERPRINTS[session_key] = fingerprints


def remember_conversation_id(session_key: Optional[str], conversation_id: str) -> None:
    if not session_key:
        return
    if session_key not in CONVERSATION_CACHE and len(CONVERSATION_CACHE) >= MAX_FINGERPRINT_SESSIONS:
        CONVERSATION_CACHE.pop(next(iter(CONVERSATION_CACHE)), None)
    CONVERSATION_CACHE[session_key] = conversation_id


def resolve_conversation_and_content(
    session_key: Optional[str],
    messages: List[Message],
    active_tools: Optional[List[Tool]],
    tool_choice: Optional[Union[str, ToolChoice, Dict[str, Any]]]
) -> tuple[str, str, bool]:
    """返回 (上游 conversation id, 要发给上游的 content, 是否走了增量路径)。

    上游必须先创建会话才能发消息，所以"能否增量发送"和"用哪个会话 id"是绑在一起的：
    只有同时命中会话缓存与指纹前缀，才说明这是同一个已创建会话的延续，可以只发增量；
    否则（冷启动、历史分叉、或增量内容为空）就新建一个会话，发送完整拼接的内容。
    """
    fingerprints = message_fingerprints(messages)

    if session_key:
        cached_conversation_id = CONVERSATION_CACHE.get(session_key)
        previous_fingerprints = SESSION_FINGERPRINTS.get(session_key)
        if cached_conversation_id and previous_fingerprints and len(fingerprints) > len(previous_fingerprints) \
                and fingerprints[:len(previous_fingerprints)] == previous_fingerprints:
            new_messages = messages[len(previous_fingerprints):]
            incremental_content = build_incremental_content(messages, new_messages, active_tools, tool_choice)
            if incremental_content:
                remember_fingerprints(session_key, fingerprints)
                return cached_conversation_id, incremental_content, True

    conversation_id = create_abundance_conversation()
    content = messages_to_prompt(messages, active_tools, tool_choice)
    remember_conversation_id(session_key, conversation_id)
    remember_fingerprints(session_key, fingerprints)
    return conversation_id, content, False


# ===================== 工具调用：XML 输出解析 =====================

def remove_think_blocks(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start_pos = text.find("<think>")
        if start_pos == -1:
            break

        pos = start_pos + 7
        depth = 1
        while pos < len(text) and depth > 0:
            if text[pos:pos + 7] == "<think>":
                depth += 1
                pos += 7
            elif text[pos:pos + 8] == "</think>":
                depth -= 1
                pos += 8
            else:
                pos += 1

        if depth == 0:
            text = text[:start_pos] + text[pos:]
        else:
            break
    return text


def find_last_trigger_signal_outside_think(text: str, trigger_signal: str) -> int:
    if not text or not trigger_signal:
        return -1

    i = 0
    think_depth = 0
    last_pos = -1
    while i < len(text):
        if text.startswith("<think>", i):
            think_depth += 1
            i += 7
            continue
        if text.startswith("</think>", i):
            think_depth = max(0, think_depth - 1)
            i += 8
            continue
        if think_depth == 0 and text.startswith(trigger_signal, i):
            last_pos = i
            i += len(trigger_signal)
            continue
        i += 1
    return last_pos


def _extract_cdata_text(raw: str) -> str:
    if "<![CDATA[" not in raw:
        return raw
    parts = re.findall(r"<!\[CDATA\[(.*?)\]\]>", raw, flags=re.DOTALL)
    return "".join(parts) if parts else raw


def _parse_args_json_payload(payload: str) -> Optional[Dict[str, Any]]:
    payload = _extract_cdata_text(payload or "").strip()
    if not payload:
        return {}
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_function_calls_xml(content: str, trigger_signal: str = TOOL_TRIGGER_SIGNAL) -> Optional[List[Dict[str, Any]]]:
    if not content or trigger_signal not in content:
        return None

    cleaned_content = remove_think_blocks(content)
    trigger_pos = cleaned_content.rfind(trigger_signal)
    if trigger_pos == -1:
        return None

    after_trigger = cleaned_content[trigger_pos + len(trigger_signal):]
    calls_match = re.search(r"<function_calls>([\s\S]*?)</function_calls>", after_trigger)
    if not calls_match:
        return None

    calls_xml = calls_match.group(0)
    calls_content = calls_match.group(1)
    results: List[Dict[str, Any]] = []

    try:
        root = ET.fromstring(calls_xml)
        for function_call in root.findall("function_call"):
            tool_el = function_call.find("tool")
            name = (tool_el.text or "").strip() if tool_el is not None else ""
            if not name:
                continue

            args: Dict[str, Any] = {}
            args_json_el = function_call.find("args_json")
            if args_json_el is not None:
                parsed_args = _parse_args_json_payload(args_json_el.text or "")
                if parsed_args is None:
                    return None
                args = parsed_args
            else:
                args_el = function_call.find("args")
                if args_el is not None:
                    for child in list(args_el):
                        raw_value = child.text or ""
                        try:
                            args[child.tag] = json.loads(raw_value)
                        except json.JSONDecodeError:
                            args[child.tag] = raw_value

            results.append({"name": name, "args": args})
        return results or None
    except ET.ParseError:
        pass

    call_blocks = re.findall(r"<function_call>([\s\S]*?)</function_call>", calls_content)
    for block in call_blocks:
        tool_match = re.search(r"<tool>(.*?)</tool>", block, flags=re.DOTALL)
        if not tool_match:
            continue

        args: Dict[str, Any] = {}
        args_json_match = re.search(r"<args_json>([\s\S]*?)</args_json>", block, flags=re.DOTALL)
        if args_json_match:
            parsed_args = _parse_args_json_payload(args_json_match.group(1))
            if parsed_args is None:
                return None
            args = parsed_args
        else:
            args_match = re.search(r"<args>([\s\S]*?)</args>", block, flags=re.DOTALL)
            if args_match:
                for key, value in re.findall(r"<([^\s>/]+)>([\s\S]*?)</\1>", args_match.group(1)):
                    try:
                        args[key] = json.loads(value)
                    except json.JSONDecodeError:
                        args[key] = value

        results.append({"name": tool_match.group(1).strip(), "args": args})

    return results or None


def _schema_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_value_against_schema(value: Any, schema: Dict[str, Any], path: str = "args") -> List[str]:
    if not isinstance(schema, dict):
        return []

    errors: List[str] = []
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        return [f"{path}: expected one of {enum_values!r}, got {value!r}"]

    expected_type = schema.get("type")

    def type_ok(schema_type: str) -> bool:
        if schema_type == "object":
            return isinstance(value, dict)
        if schema_type == "array":
            return isinstance(value, list)
        if schema_type == "string":
            return isinstance(value, str)
        if schema_type == "boolean":
            return isinstance(value, bool)
        if schema_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if schema_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if schema_type == "null":
            return value is None
        return True

    if isinstance(expected_type, str) and not type_ok(expected_type):
        return [f"{path}: expected type '{expected_type}', got '{_schema_type_name(value)}'"]
    if isinstance(expected_type, list) and not any(type_ok(item) for item in expected_type if isinstance(item, str)):
        return [f"{path}: expected type in {expected_type!r}, got '{_schema_type_name(value)}'"]

    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for key in required:
            if isinstance(key, str) and key not in value:
                errors.append(f"{path}: missing required property '{key}'")
        for key, item_value in value.items():
            if key in properties:
                errors.extend(_validate_value_against_schema(item_value, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property '{key}'")

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(_validate_value_against_schema(item, schema["items"], f"{path}[{index}]"))

    return errors


def validate_parsed_tools(parsed_tools: List[Dict[str, Any]], tools: List[Tool]) -> Optional[str]:
    tools_by_name = {tool.function.name: tool for tool in tools}
    allowed = {name: tool.function.parameters or {} for name, tool in tools_by_name.items()}
    allowed_names = sorted(allowed.keys())

    for index, parsed_tool in enumerate(parsed_tools, start=1):
        name = parsed_tool.get("name")
        args = parsed_tool.get("args")
        if not isinstance(name, str) or not name:
            return f"Tool call #{index}: missing tool name"
        if name not in allowed:
            return f"Tool call #{index}: unknown tool '{name}'. Allowed tools: {allowed_names}"
        normalize_parsed_tool_args(parsed_tool, tools_by_name[name])
        if not isinstance(args, dict):
            return f"Tool call #{index} '{name}': arguments must be a JSON object"

        errors = _validate_value_against_schema(args, allowed[name], name)
        if errors:
            preview = "; ".join(errors[:6])
            return f"Tool call #{index} '{name}': schema validation failed: {preview}"

    return None


def extract_valid_tool_calls(content: str, tools: List[Tool]) -> Optional[List[Dict[str, Any]]]:
    parsed_tools = parse_function_calls_xml(content)
    if not parsed_tools:
        return None
    validation_error = validate_parsed_tools(parsed_tools, tools)
    if validation_error:
        return None
    return parsed_tools


def build_openai_tool_calls(parsed_tools: List[Dict[str, Any]], include_index: bool = False) -> List[Dict[str, Any]]:
    tool_calls = []
    for index, tool in enumerate(parsed_tools):
        tool_call = {
            "id": f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {
                "name": tool["name"],
                "arguments": json.dumps(tool["args"], ensure_ascii=False)
            }
        }
        if include_index:
            tool_call = {"index": index, **tool_call}
        tool_calls.append(tool_call)
    return tool_calls


def content_before_tool_call(content: str) -> Optional[str]:
    trigger_pos = find_last_trigger_signal_outside_think(content, TOOL_TRIGGER_SIGNAL)
    if trigger_pos == -1:
        return content or None
    prefix = content[:trigger_pos].rstrip()
    return prefix or None


# ===================== 上游 a.b.u.n.dance HTTP 调用 + SSE 解析 =====================

def abundance_request_proxies() -> Optional[Dict[str, str]]:
    proxy = os.getenv("ABUNDANCE_HTTP_PROXY", "").strip()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def parse_cookie_header(cookie_header: str) -> Dict[str, str]:
    if not cookie_header:
        return {}
    parsed = SimpleCookie()
    try:
        parsed.load(cookie_header)
    except Exception:
        logger.warning("Failed to parse ABUNDANCE_COOKIE; falling back to explicit cookie env vars")
        return {}
    return {key: morsel.value for key, morsel in parsed.items() if morsel.value}


def is_jwt_expired(token: str, leeway_seconds: int = 60) -> bool:
    parts = token.split(".")
    if len(parts) < 2:
        return False

    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
    except Exception:
        return False

    exp = data.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return exp <= time.time() + leeway_seconds


def abundance_cookies(include_oidc: bool = True) -> Dict[str, str]:
    cookies = parse_cookie_header(ABUNDANCE_RAW_COOKIE)
    if ABUNDANCE_SESSION_COOKIE:
        cookies["session"] = ABUNDANCE_SESSION_COOKIE
    if include_oidc and ABUNDANCE_OIDC_TOKEN:
        cookies["oidc_id_token"] = ABUNDANCE_OIDC_TOKEN

    # id_token 通常只有约 1 小时有效期；过期后优先让长期 session 独立工作。
    oidc_token = cookies.get("oidc_id_token")
    if oidc_token and is_jwt_expired(oidc_token):
        cookies.pop("oidc_id_token", None)
    return cookies


def abundance_cookie_variants() -> List[Dict[str, str]]:
    primary = abundance_cookies(include_oidc=True)
    if not primary:
        return []

    variants = [primary]
    if "session" in primary and "oidc_id_token" in primary:
        session_only = dict(primary)
        session_only.pop("oidc_id_token", None)
        variants.append(session_only)
    return variants


def describe_cookie_variant(cookies: Dict[str, str]) -> str:
    if "session" in cookies and "oidc_id_token" in cookies:
        return "session+oidc"
    if "session" in cookies:
        return "session-only"
    if "oidc_id_token" in cookies:
        return "oidc-only"
    return "custom-cookie"


def abundance_headers() -> Dict[str, str]:
    return {
        "accept": "*/*",
        "content-type": "application/json",
        "origin": ABUNDANCE_BASE_URL,
        "user-agent": ABUNDANCE_USER_AGENT,
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    }


def response_error_detail(resp, max_chars: int = 1200) -> str:
    try:
        text = resp.text
    except Exception:
        text = ""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return resp.reason or "empty response body"
    if len(text) > max_chars:
        return f"{text[:max_chars].rstrip()}..."
    return text


def should_retry_upstream_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def abundance_post(url: str, body: Dict[str, Any], stream: bool, action: str):
    """带重试地向 a.b.u.n.dance 发 POST；认证失败时自动尝试 session-only。"""
    cookie_variants = abundance_cookie_variants()
    if not cookie_variants:
        raise HTTPException(
            status_code=500,
            detail="Missing ABUNDANCE_COOKIE or ABUNDANCE_SESSION. Set them in .env (see .env.example)."
        )

    request_kwargs: Dict[str, Any] = {
        "data": json.dumps(body, ensure_ascii=False).encode("utf-8"),
        "headers": abundance_headers(),
        "stream": stream,
        "timeout": ABUNDANCE_REQUEST_TIMEOUT_SECONDS,
        "impersonate": ABUNDANCE_IMPERSONATE,
    }
    proxies = abundance_request_proxies()
    if proxies:
        request_kwargs["proxies"] = proxies

    max_attempts = ABUNDANCE_MAX_STATUS_RETRIES + 1
    last_auth_status: Optional[int] = None
    for variant_index, cookies in enumerate(cookie_variants):
        variant_kwargs = dict(request_kwargs)
        variant_kwargs["cookies"] = cookies

        for attempt in range(1, max_attempts + 1):
            resp = curl_cffi.requests.post(url, **variant_kwargs)

            if resp.status_code == 200:
                return resp

            if resp.status_code in (401, 403):
                last_auth_status = resp.status_code
                detail = response_error_detail(resp)
                try:
                    resp.close()
                except Exception:
                    logger.debug("Failed to close a.b.u.n.dance auth-error response", exc_info=True)
                if variant_index + 1 < len(cookie_variants):
                    logger.info(
                        "a.b.u.n.dance rejected %s cookie variant for %s; retrying with next variant",
                        describe_cookie_variant(cookies),
                        action
                    )
                    break
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"a.b.u.n.dance rejected the {action} request "
                        f"(HTTP {resp.status_code}). The session cookie may have expired, "
                        f"or the upstream requires a fresh oidc_id_token. Upstream detail: {detail}"
                    )
                )

            detail = response_error_detail(resp)
            if attempt == max_attempts or not should_retry_upstream_status(resp.status_code):
                try:
                    resp.close()
                except Exception:
                    logger.debug("Failed to close a.b.u.n.dance error response", exc_info=True)
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"a.b.u.n.dance {action} API returned HTTP {resp.status_code}: {detail}"
                    )
                )

            try:
                resp.close()
            except Exception:
                logger.debug("Failed to close a.b.u.n.dance retry response", exc_info=True)

            delay_seconds = ABUNDANCE_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "a.b.u.n.dance %s API returned HTTP %s on attempt %s/%s; retrying in %.2fs; detail=%s",
                action, resp.status_code, attempt, max_attempts, delay_seconds, detail
            )
            time.sleep(delay_seconds)

    if last_auth_status:
        raise HTTPException(
            status_code=502,
            detail=(
                f"a.b.u.n.dance rejected the {action} request (HTTP {last_auth_status}). "
                "Refresh ABUNDANCE_SESSION or provide a fresh ABUNDANCE_COOKIE."
            )
        )

    raise RuntimeError("a.b.u.n.dance API retry loop exited unexpectedly")


def call_abundance_api(
    conversation_id: str,
    content: str,
    model: str,
    speed: Optional[str] = None,
    intelligence: Optional[str] = None
):
    url = f"{ABUNDANCE_BASE_URL}/api/conversations/{conversation_id}/messages"
    body = {
        "content": content,
        "model": model,
        "speed": speed or ABUNDANCE_DEFAULT_SPEED,
        "intelligence": intelligence or ABUNDANCE_DEFAULT_INTELLIGENCE,
        "stream": True
    }
    return abundance_post(url, body, stream=True, action="send-message")


def create_abundance_conversation() -> str:
    """上游没有"按 id 自动建会话"的语义：必须先 POST /api/conversations 拿到真实 id，
    才能向 /api/conversations/{id}/messages 发消息，否则会失败。"""
    url = f"{ABUNDANCE_BASE_URL}/api/conversations"
    resp = abundance_post(url, {}, stream=False, action="create-conversation")
    try:
        payload = resp.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="a.b.u.n.dance returned a non-JSON response when creating a conversation."
        )
    finally:
        resp.close()

    conversation = payload.get("conversation") if isinstance(payload, dict) else None
    conversation_id = conversation.get("id") if isinstance(conversation, dict) else None
    if not conversation_id:
        raise HTTPException(
            status_code=502,
            detail="a.b.u.n.dance did not return a conversation id when creating a conversation."
        )
    return conversation_id


def parse_sse_block(block: str) -> Optional[tuple[str, Optional[Dict[str, Any]], str]]:
    event_name = "message"
    data_lines: List[str] = []
    for line in block.split("\n"):
        line = line.rstrip("\r")
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    if not data_lines:
        return None

    data_text = "\n".join(data_lines)
    payload: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(data_text)
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        payload = None
    return event_name, payload, data_text


def iter_abundance_sse_events(resp):
    """按 SSE 规范用空行切分事件块，对 event:/data: 字段做健壮解析。

    手动在原始字节上做缓冲拆分（而非依赖 iter_lines 对空行的处理方式），
    这样不论上游一次 flush 多少字节，都能正确还原出 event/delta/done 等事件块。
    """
    buffer = ""
    for chunk in resp.iter_content():
        if not chunk:
            continue
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        buffer += text.replace("\r\n", "\n")

        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event = parse_sse_block(block)
            if event:
                yield event

    tail = buffer.strip("\n")
    if tail:
        event = parse_sse_block(tail)
        if event:
            yield event


# ===================== 流式 / 非流式响应生成（OpenAI 格式） =====================

def emit_buffered_stream_response(
    model: str,
    request_id: str,
    content: str,
    tools: Optional[List[Tool]]
):
    parsed_tools = extract_valid_tool_calls(content, tools or []) if tools else None
    if parsed_tools:
        prefix_content = content_before_tool_call(content)
        if prefix_content:
            prefix_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": prefix_content},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(prefix_chunk, ensure_ascii=False)}\n\n"

        tool_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": build_openai_tool_calls(parsed_tools, include_index=True)
                },
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    if content:
        content_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def stream_error_chunks(model: str, request_id: str, message: str):
    content_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": f"[upstream error] {message}"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def stream_start_chunk(model: str, request_id: str) -> str:
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def wait_for_abundance_stream_response(conversation_id: str, content: str, model: str):
    future = UPSTREAM_CONNECT_EXECUTOR.submit(call_abundance_api, conversation_id, content, model)
    while True:
        try:
            yield future.result(timeout=ABUNDANCE_CONNECT_KEEPALIVE_SECONDS)
            return
        except concurrent.futures.TimeoutError:
            yield ": keep-alive\n\n"


def exception_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc) or exc.__class__.__name__


def stream_generator(
    model: str,
    conversation_id: str,
    content: str,
    request_id: str,
    tools: Optional[List[Tool]] = None
):
    """Starlette 在 StreamingResponse 拉取第一个分片之前就已经发出 200 状态行，
    所以这里必须把包括首次上游调用在内的一切异常都兜住，转成一段 SSE 错误增量
    再正常收尾——否则客户端只会看到一个提前结束的空 200 响应。"""
    buffered_content = ""
    resp = None
    try:
        yield stream_start_chunk(model, request_id)
        connect_waiter = wait_for_abundance_stream_response(conversation_id, content, model)
        while resp is None:
            item = next(connect_waiter)
            if isinstance(item, str):
                yield item
            else:
                resp = item

        for event_name, payload, _raw in iter_abundance_sse_events(resp):
            if event_name in {"connecting", "heartbeat"}:
                continue

            if event_name == "error":
                message = "a.b.u.n.dance upstream error"
                if isinstance(payload, dict):
                    message = normalize_identifier(payload.get("error") or payload.get("message")) or message
                if tools:
                    buffered_content += f"\n\n[upstream error] {message}"
                    continue
                yield from stream_error_chunks(model, request_id, message)
                return

            if event_name == "delta":
                text = payload.get("text", "") if isinstance(payload, dict) else ""
                if not text:
                    continue

                if tools:
                    buffered_content += text
                    continue

                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                continue

            if event_name == "done":
                if tools:
                    final_text = buffered_content
                    if isinstance(payload, dict):
                        assistant_message = payload.get("assistantMessage")
                        if isinstance(assistant_message, dict):
                            final_text = assistant_message.get("content") or buffered_content
                    yield from emit_buffered_stream_response(model, request_id, final_text, tools)
                    return

                final_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

        # 上游连接结束但没有收到 done 事件：兜底收尾，避免客户端挂起等待。
        if tools:
            yield from emit_buffered_stream_response(model, request_id, buffered_content, tools)
            return
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        message = exception_message(exc)
        logger.warning("stream_generator failed: %s", message)
        if tools and buffered_content:
            yield from emit_buffered_stream_response(
                model, request_id, f"{buffered_content}\n\n[upstream error] {message}", tools
            )
        else:
            yield from stream_error_chunks(model, request_id, message)
    finally:
        if resp is not None:
            resp.close()


def get_complete_response(
    model: str,
    conversation_id: str,
    content: str,
    tools: Optional[List[Tool]] = None
) -> Dict[str, Any]:
    resp = call_abundance_api(conversation_id, content, model)
    result = ""
    final_text: Optional[str] = None

    try:
        for event_name, payload, _raw in iter_abundance_sse_events(resp):
            if event_name in {"connecting", "heartbeat"}:
                continue
            if event_name == "delta":
                text = payload.get("text", "") if isinstance(payload, dict) else ""
                if text:
                    result += text
                continue
            if event_name == "error":
                message = "a.b.u.n.dance upstream error"
                if isinstance(payload, dict):
                    message = normalize_identifier(payload.get("error") or payload.get("message")) or message
                result += f"\n\n[upstream error] {message}"
                continue
            if event_name == "done":
                if isinstance(payload, dict):
                    assistant_message = payload.get("assistantMessage")
                    if isinstance(assistant_message, dict):
                        final_text = assistant_message.get("content")
                break
    finally:
        resp.close()

    raw_content = final_text if final_text is not None else result

    parsed_tools = extract_valid_tool_calls(raw_content, tools or []) if tools else None
    content_out = raw_content
    tool_calls: List[Dict[str, Any]] = []
    finish_reason = "stop"

    if parsed_tools:
        content_out = content_before_tool_call(raw_content)
        tool_calls = build_openai_tool_calls(parsed_tools)
        finish_reason = "tool_calls"

    return {
        "content": content_out,
        "raw_content": raw_content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason
    }


def build_chat_completion_payload(
    request_id: str,
    model: str,
    result: Dict[str, Any],
    prompt_text: str
) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": result["content"] or ""
    }
    if result["tool_calls"]:
        message = {
            "role": "assistant",
            "content": result["content"],
            "tool_calls": result["tool_calls"]
        }

    output_text = result["content"] or ""
    input_tokens = len(prompt_text.split())
    output_tokens = len(output_text.split())

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": result["finish_reason"]
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    }


# ===================== 请求准备 =====================

def prepare_chat_execution(
    request: ChatCompletionRequest,
    session_key: Optional[str]
) -> Dict[str, Any]:
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")

    choice = tool_choice_to_dict(request.tool_choice)
    if choice in {"required", "any"} and not request.tools:
        raise HTTPException(status_code=400, detail="tool_choice requires a non-empty tools list")
    if isinstance(choice, dict) and not request.tools:
        raise HTTPException(status_code=400, detail="tool_choice requires a non-empty tools list")

    active_tools = request.tools if tool_calling_enabled(request.tools, request.tool_choice) else None
    if active_tools:
        validate_tools(active_tools)

    abundance_conversation_id, content, is_incremental = resolve_conversation_and_content(
        session_key, request.messages, active_tools, request.tool_choice
    )

    return {
        "active_tools": active_tools,
        "abundance_conversation_id": abundance_conversation_id,
        "content": content,
        "is_incremental": is_incremental,
        "request_id": f"chatcmpl-{uuid.uuid4().hex[:24]}"
    }


# ===================== FastAPI 路由 =====================

@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None, alias="Authorization")):
    verify_api_key(authorization)

    models = [
        {
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "abundance"
        }
        for model in AVAILABLE_MODELS
    ]
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    authorization: Optional[str] = Header(None)
):
    logger.info(
        "Received /v1/chat/completions request: model=%s stream=%s messages=%s tools=%s",
        request.model, request.stream, len(request.messages), len(request.tools or [])
    )
    logger.debug("Headers: %s", safe_headers_for_log(http_request.headers))

    api_key = verify_api_key(authorization)
    user_id, conversation_id, is_stable = resolve_session_identity(request, http_request, request.messages)
    session_key = request_session_key("chat", request.model, api_key, user_id, conversation_id) if is_stable else None

    execution = prepare_chat_execution(request, session_key)
    active_tools = execution["active_tools"]
    content = execution["content"]
    request_id = execution["request_id"]
    abundance_conversation_id = execution["abundance_conversation_id"]
    logger.info(
        "session_key=%s abundance_conversation_id=%s incremental=%s content_len=%s",
        session_key, abundance_conversation_id, execution["is_incremental"], len(content)
    )

    if request.stream:
        return StreamingResponse(
            stream_generator(request.model, abundance_conversation_id, content, request_id, active_tools),
            media_type="text/event-stream"
        )

    result = get_complete_response(request.model, abundance_conversation_id, content, active_tools)
    response = build_chat_completion_payload(request_id, request.model, result, content)
    return JSONResponse(content=response)


@app.get("/v1")
@app.head("/v1")
async def v1_root():
    return {
        "message": "OpenAI Compatible API Server (a.b.u.n.dance backend)",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions"
        }
    }


@app.get("/")
@app.head("/")
async def root():
    return {
        "message": "OpenAI Compatible API Server (a.b.u.n.dance backend)",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "health": "/healthz"
        },
        "auth": "Bearer token required for /v1 endpoints"
    }


@app.get("/healthz")
@app.head("/healthz")
async def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=env_int("PORT", 18000)
    )
