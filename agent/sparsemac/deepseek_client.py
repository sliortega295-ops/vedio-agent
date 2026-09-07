from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any


class DeepSeekClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 90,
    ):
        self.model = model or _env("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.api_key = api_key or _env("DEEPSEEK_API_KEY")
        self.base_url = (base_url or _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout = timeout
        self.max_retries = int(_env("DEEPSEEK_MAX_RETRIES", "3"))
        self.available = bool(self.api_key)
        self.last_error = "" if self.available else "DEEPSEEK_API_KEY is not set"

    def generate_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 5000,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any] | None, str]:
        if not self.available:
            return None, self.last_error

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise planning agent. Return only valid JSON and no markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        raw = ""
        errors = []
        for attempt in range(max(1, self.max_retries)):
            raw, error = self._post_with_urllib(payload)
            if raw:
                break
            errors.append(error)
            raw, error = self._post_with_curl(payload)
            if raw:
                break
            errors.append(error)
            if attempt + 1 < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
        if not raw:
            self.last_error = "DeepSeek error: " + " | ".join(error for error in errors if error)[-1200:]
            return None, self.last_error

        try:
            body = json.loads(raw)
            text = body["choices"][0]["message"]["content"]
        except Exception:
            return None, _redact(raw[:1000])
        parsed = _extract_json(text)
        return parsed, text

    def _post_with_urllib(self, payload: dict[str, Any]) -> tuple[str, str]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8"), ""
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            return "", f"urllib HTTP {exc.code}: {_redact(detail)}"
        except Exception as exc:
            return "", f"urllib {_redact(str(exc))}"

    def _post_with_curl(self, payload: dict[str, Any]) -> tuple[str, str]:
        env = os.environ.copy()
        env["DEEPSEEK_API_KEY"] = self.api_key or ""
        # Use a tiny shell only so curl expands the environment variable inside
        # the header. The key stays out of argv and process listings.
        shell_cmd = (
            "curl -sS --http1.1 --max-time "
            + _shell_quote(str(self.timeout))
            + " "
            + _shell_quote(f"{self.base_url}/chat/completions")
            + ' -H "Authorization: Bearer $DEEPSEEK_API_KEY"'
            + " -H "
            + _shell_quote("Content-Type: application/json")
            + " --data-binary @-"
        )
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", shell_cmd],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.timeout + 5,
                env=env,
                check=False,
            )
        except Exception as exc:
            return "", f"curl {_redact(str(exc))}"
        if proc.returncode != 0:
            return "", f"curl rc={proc.returncode}: {_redact(proc.stderr.strip()[:1000])}"
        return proc.stdout, ""


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    local_env = Path(__file__).resolve().parents[1] / ".env.local"
    if not local_env.exists():
        return default
    for line in local_env.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        if key.strip() == name:
            return raw_value.strip().strip("'\"") or default
    return default


def _redact(text: str) -> str:
    key = _env("DEEPSEEK_API_KEY")
    if key:
        text = text.replace(key, "<redacted>")
    return text
def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"
