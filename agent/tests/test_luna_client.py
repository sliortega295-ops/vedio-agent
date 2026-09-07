from __future__ import annotations

import json

from sparsemac.luna_client import DEFAULT_LUNA_MODEL, LunaClient, _extract_json, _parse_response


def test_luna_client_uses_sparsemac_environment(monkeypatch):
    monkeypatch.setenv("SPARSEMAC_LUNA_API_KEY", "test-key")
    monkeypatch.setenv("SPARSEMAC_LUNA_BASE_URL", "https://example.invalid/v1/")
    monkeypatch.setenv("SPARSEMAC_LUNA_MODEL", "gpt-5.6-luna")
    client = LunaClient()
    assert client.available is True
    assert client.model == DEFAULT_LUNA_MODEL
    assert client.base_url == "https://example.invalid/v1"


def test_luna_client_parses_chat_completion_and_fenced_json():
    body = {
        "choices": [
            {"message": {"content": "```json\n{\"ok\": true}\n```"}}
        ]
    }
    parsed, text = _parse_response(json.dumps(body))
    assert parsed == {"ok": True}
    assert "ok" in text
    assert _extract_json("prefix {\"score\": 0.7} suffix") == {"score": 0.7}
