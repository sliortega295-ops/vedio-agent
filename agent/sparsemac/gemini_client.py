from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.available = bool(self.api_key)
        self._client = None

    def _load_client(self):
        if self._client is not None:
            return self._client
        if not self.available:
            return None
        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except Exception:
            try:
                import google.generativeai as genai_old

                genai_old.configure(api_key=self.api_key)
                self._client = genai_old.GenerativeModel(self.model)
                return self._client
            except Exception:
                self.available = False
                return None

    def generate_json(self, prompt: str, image_paths: list[str] | None = None) -> tuple[dict[str, Any] | None, str]:
        client = self._load_client()
        if client is None:
            return None, ""

        image_paths = image_paths or []
        try:
            if client.__class__.__module__.startswith("google.genai"):
                parts: list[Any] = [prompt]
                for path in image_paths:
                    data = Path(path).read_bytes()
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(data).decode("ascii")}})
                resp = client.models.generate_content(model=self.model, contents=parts)
                text = getattr(resp, "text", "") or ""
            else:
                parts = [prompt]
                for path in image_paths:
                    parts.append({"mime_type": "image/jpeg", "data": Path(path).read_bytes()})
                resp = client.generate_content(parts)
                text = getattr(resp, "text", "") or ""
        except Exception as exc:
            return None, f"Gemini error: {exc}"

        parsed = _extract_json(text)
        return parsed, text


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None

