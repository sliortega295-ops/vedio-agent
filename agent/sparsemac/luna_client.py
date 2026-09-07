from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_LUNA_MODEL = "gpt-5.6-luna"
DEFAULT_LUNA_BASE_URL = "https://bmapi.020212.xyz/v1"


class LunaClient:
    """Small OpenAI-compatible GPT-5.6 Luna text/vision JSON client.

    SparseMAC owns this client rather than importing implementation code from
    the VQQA baseline.  Configuration is read from ``.env.local`` or the
    process environment, and secrets never need to appear in generated remote
    scripts or command-line arguments.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 180,
        reasoning_effort: str | None = None,
    ):
        self.model = model or _first_env(
            "SPARSEMAC_LUNA_MODEL",
            "VQQA_VLM_MODEL",
            default=DEFAULT_LUNA_MODEL,
        )
        self.api_key = api_key or _first_env(
            "SPARSEMAC_LUNA_API_KEY",
            "VQQA_VLM_API_KEY",
            "OPENAI_API_KEY",
        )
        self.base_url = (
            base_url
            or _first_env(
                "SPARSEMAC_LUNA_BASE_URL",
                "VQQA_VLM_BASE_URL",
                default=DEFAULT_LUNA_BASE_URL,
            )
        ).rstrip("/")
        self.timeout = int(timeout)
        self.reasoning_effort = reasoning_effort or _first_env(
            "SPARSEMAC_LUNA_REASONING_EFFORT",
            default="medium",
        )
        self.max_retries = int(_first_env("SPARSEMAC_LUNA_MAX_RETRIES", default="3"))
        self.available = bool(self.api_key)
        self.last_error = "" if self.available else "SPARSEMAC_LUNA_API_KEY is not set"

    def generate_json(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        *,
        max_tokens: int = 6000,
        temperature: float = 0.0,
        system_prompt: str = (
            "You are a precise text-to-video planning and evaluation expert. "
            "Return only one valid JSON object and no markdown."
        ),
    ) -> tuple[dict[str, Any] | None, str]:
        if not self.available:
            return None, self.last_error

        content: str | list[dict[str, Any]]
        paths = image_paths or []
        if paths:
            parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for path_text in paths:
                path = Path(path_text)
                mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{encoded}",
                            "detail": "high",
                        },
                    }
                )
            content = parts
        else:
            content = prompt

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        errors: list[str] = []
        for attempt in range(max(1, self.max_retries)):
            raw_body, error = self._post_with_urllib(payload)
            if not raw_body:
                errors.append(error)
                raw_body, error = self._post_with_curl(payload)
            if raw_body:
                parsed, text = _parse_response(raw_body)
                if parsed is not None:
                    self.last_error = ""
                    return parsed, text
                errors.append("invalid JSON reply: " + _redact(text[:1200]))
            elif error:
                errors.append(error)
            if attempt + 1 < self.max_retries:
                time.sleep(2.0 * (attempt + 1))

        self.last_error = "GPT-5.6 Luna error: " + " | ".join(
            item for item in errors if item
        )[-1800:]
        return None, self.last_error

    def _post_with_urllib(self, payload: dict[str, Any]) -> tuple[str, str]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "SparseMAC/2.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8", errors="replace"), ""
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1200]
            return "", f"urllib HTTP {exc.code}: {_redact(detail)}"
        except Exception as exc:
            return "", f"urllib {_redact(str(exc))}"

    def _post_with_curl(self, payload: dict[str, Any]) -> tuple[str, str]:
        env = os.environ.copy()
        env["SPARSEMAC_LUNA_API_KEY_RUNTIME"] = self.api_key or ""
        command = (
            "curl -sS --http1.1 --max-time "
            + _shell_quote(str(self.timeout))
            + " "
            + _shell_quote(f"{self.base_url}/chat/completions")
            + ' -H "Authorization: Bearer $SPARSEMAC_LUNA_API_KEY_RUNTIME"'
            + " -H "
            + _shell_quote("Content-Type: application/json")
            + " -H "
            + _shell_quote("User-Agent: SparseMAC/2.0")
            + " --data-binary @-"
        )
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", command],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.timeout + 10,
                env=env,
                check=False,
            )
        except Exception as exc:
            return "", f"curl {_redact(str(exc))}"
        if proc.returncode != 0:
            return "", f"curl rc={proc.returncode}: {_redact(proc.stderr.strip()[:1200])}"
        return proc.stdout, ""


def _parse_response(raw: str) -> tuple[dict[str, Any] | None, str]:
    try:
        body = json.loads(raw)
    except Exception:
        return None, raw
    try:
        message = (body.get("choices") or [])[0].get("message", {})
        content = message.get("content") or body.get("output_text") or ""
    except Exception:
        return None, _redact(raw[:1200])
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    text = str(content).strip()
    return _extract_json(text), text


def _extract_json(text: str) -> dict[str, Any] | None:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(value[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    local_env = Path(__file__).resolve().parents[1] / ".env.local"
    if local_env.exists():
        wanted = set(names)
        for line in local_env.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() in wanted:
                value = raw_value.strip().strip("'\"")
                if value:
                    return value
    return default


def _redact(text: str) -> str:
    for name in (
        "SPARSEMAC_LUNA_API_KEY",
        "VQQA_VLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        key = _first_env(name)
        if key:
            text = text.replace(key, "<redacted>")
    return text


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"
