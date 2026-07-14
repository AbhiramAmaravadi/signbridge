from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class GeminiClient:
    api_key: str
    model: str = DEFAULT_GEMINI_MODEL

    @classmethod
    def from_env(cls) -> "GeminiClient | None":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        return cls(api_key=api_key, model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))

    def generate_text(self, prompt: str) -> str:
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        data = self._post_generate_content(payload)
        return self._extract_text(data)

    def analyze_image(self, prompt: str, image_base64: str, mime_type: str = "image/jpeg") -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": self._strip_data_url(image_base64),
                            }
                        },
                    ]
                }
            ]
        }
        data = self._post_generate_content(payload)
        return self._extract_text(data)

    def _post_generate_content(self, payload: dict) -> dict:
        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent?key={self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc

    @staticmethod
    def _extract_text(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(str(part.get("text", "")) for part in parts).strip()

    @staticmethod
    def _strip_data_url(image_base64: str) -> str:
        return image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
