from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class GeminiClient:
    api_key: str
    model: str = DEFAULT_GEMINI_MODEL

    @classmethod
    def from_env(cls) -> "GeminiClient | None":
        gemini_key = os.getenv("GEMINI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        api_key = gemini_key or google_key
        if not api_key:
            print("[GEMINI] No API key detected. Falling back to local text.", flush=True)
            return None

        source = "GEMINI_API_KEY" if gemini_key else "GOOGLE_API_KEY"
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        print(
            f"[GEMINI] API key detected via {source}: Yes (First 4 chars: {api_key[:4]}...). Model: {model}",
            flush=True,
        )
        return cls(api_key=api_key, model=model)

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

    def generate_json_with_optional_image(
        self,
        prompt: str,
        image_base64: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if image_base64:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": self._strip_data_url(image_base64),
                    }
                }
            )

        data = self._post_generate_content({"contents": [{"parts": parts}]})
        text = self._extract_text(data)
        return self._parse_json_object(text)

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
            print(f"[GEMINI] HTTP error during generateContent: {exc.code} {detail}", flush=True)
            raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc
        except Exception as exc:
            print(f"[GEMINI] Request failed during generateContent: {type(exc).__name__}: {exc}", flush=True)
            raise

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

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end >= start:
            cleaned = cleaned[start : end + 1]

        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Gemini response was not a JSON object.")
        return parsed
