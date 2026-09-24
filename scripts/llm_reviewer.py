"""Provider-neutral LLM reviewer for discovery candidates.

The script reads one candidate JSON object from stdin and writes one normalized
review JSON object to stdout. Provider and model selection are configuration:

    python scripts/llm_reviewer.py --provider openrouter --model openrouter/free
    python scripts/llm_reviewer.py --provider gemini --model gemini-2.5-flash-lite

When flags are omitted, ``LLM_PROVIDER``/``LLM_MODEL`` and then provider-specific
model environment variables are used. Adding a new model within a supported
provider therefore does not require a code change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = {
    "openrouter": "openrouter/free",
    "gemini": "gemini-2.5-flash-lite",
}
API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

VALID_ENTRY_TYPES = {
    "model", "dataset", "tool", "benchmark", "simulator", "paper_only", "irrelevant", "unclear"
}
VALID_DECISIONS = {"accept", "needs_review", "reject"}
ALLOWED_METADATA = {
    "categories": {"manipulation", "locomotion", "navigation", "dexterous", "whole-body", "aerial"},
    "hardware_targets": {"manipulator", "humanoid", "quadruped", "biped", "mobile", "drone", "hand"},
    "learning_methods": {"VLA", "IL", "RL", "diffusion", "world_model", "sim2real"},
    "framework": {"pytorch", "jax", "tensorflow", "other"},
    "communication": {"ros2", "grpc", "lcm", "zenoh", "other"},
}

REQUIRED_FIELDS = {
    "has_verified_model_link": False,
    "has_verified_artifact_link": False,
    "entry_type": "unclear",
    "decision": "needs_review",
    "entry_summary": "",
    "maintainer_summary": "",
    "reason": "",
    "model_name": "",
    "organization": "",
    "categories": [],
    "hardware_targets": [],
    "learning_methods": [],
    "framework": [],
    "communication": [],
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_verified_model_link": {"type": "boolean"},
        "has_verified_artifact_link": {"type": "boolean"},
        "entry_type": {"type": "string", "enum": sorted(VALID_ENTRY_TYPES)},
        "decision": {"type": "string", "enum": sorted(VALID_DECISIONS)},
        "entry_summary": {"type": "string"},
        "maintainer_summary": {"type": "string"},
        "reason": {"type": "string"},
        "model_name": {"type": "string"},
        "organization": {"type": "string"},
        **{
            field: {"type": "array", "items": {"type": "string", "enum": sorted(values)}}
            for field, values in ALLOWED_METADATA.items()
        },
    },
    "required": list(REQUIRED_FIELDS),
}

SYSTEM_PROMPT = """
You are reviewing candidates for an Awesome Physical AI repository.

Judge whether the candidate is relevant to Physical AI, robotics, embodied AI,
robot learning, manipulation, locomotion, simulation, tactile sensing, navigation,
or related physical-world AI systems.

Use the provided paper title, abstract, authors, links, duplicate matches,
keyword-based review, artifact_availability, and link check results.

Rules:
- Reject autonomous-driving-only, traffic-only, ADAS-only, unrelated, or duplicate entries.
- Reject clearly unofficial reimplementations, fine-tunes, converted models, or community-only artifacts.
- Mark paper-only entries as needs_review unless they are clearly irrelevant.
- Treat a verified official model link as the strongest positive signal for inclusion.
- Do not treat a generic project page as a verified model release unless artifact_availability explicitly shows a verified model/code/dataset/space artifact link.
- If has_verified_model_link is false, state that no verified model link was found in the reason and maintainer_summary.
- Prefer needs_review over reject when the candidate is plausibly Physical AI but model/artifact availability is unclear.
- entry_summary must be an evidence-based 2-3 sentence Awesome-list description and must not invent artifact availability.
- maintainer_summary must be a concise 2-3 sentence note for the generated model PR review.
- model_name and organization must be supported by the title or an official repository namespace; otherwise return an empty string.
- Metadata arrays are PR preparation annotations. Select only explicitly supported schema values.
- Do not invent links, stars, datasets, models, code releases, benchmarks, or claims not present in the input.
- Return only valid JSON matching the requested schema. Do not wrap it in markdown.
"""


def read_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("empty stdin")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("stdin JSON must be an object")
    return data


def build_user_prompt(candidate: dict[str, Any]) -> str:
    return "Candidate JSON:\n" + json.dumps(candidate, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("LLM returned non-object JSON")
    return data


def normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(REQUIRED_FIELDS)
    normalized.update(review)
    normalized["has_verified_model_link"] = bool(normalized.get("has_verified_model_link"))
    normalized["has_verified_artifact_link"] = bool(normalized.get("has_verified_artifact_link"))

    if normalized.get("entry_type") not in VALID_ENTRY_TYPES:
        normalized["entry_type"] = "unclear"
    if normalized.get("decision") not in VALID_DECISIONS:
        normalized["decision"] = "needs_review"

    for field in ("entry_summary", "maintainer_summary", "reason", "model_name", "organization"):
        normalized[field] = str(normalized.get(field) or "")
    for field, allowed in ALLOWED_METADATA.items():
        values = normalized.get(field)
        normalized[field] = [value for value in values if value in allowed] if isinstance(values, list) else []

    normalized["status"] = "ok"
    return normalized


def call_openrouter(*, api_key: str, model: str, candidate: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_HTTP_REFERER", "https://github.com/PyTorchKR/Awesome-Physical-AI"
            ),
            "X-OpenRouter-Title": os.environ.get("OPENROUTER_APP_TITLE", "Awesome Physical AI Discovery"),
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(candidate)},
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    choices = response.json().get("choices") or []
    if not choices:
        raise ValueError("OpenRouter returned no choices")
    content = (choices[0].get("message") or {}).get("content") or ""
    return normalize_review(extract_json_object(content))


def call_gemini(*, api_key: str, model: str, candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini provider requires: pip install -U google-genai") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=SYSTEM_PROMPT + "\n\n" + build_user_prompt(candidate))],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    return normalize_review(extract_json_object(response.text or ""))


def provider_api_key(provider: str) -> str:
    return os.environ.get(API_KEY_ENV[provider], "")


def provider_model(provider: str, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    generic_model = os.environ.get("LLM_MODEL", "").strip()
    if generic_model:
        return generic_model
    provider_env = "OPENROUTER_MODEL" if provider == "openrouter" else "GEMINI_MODEL"
    return os.environ.get(provider_env, "").strip() or DEFAULT_MODELS[provider]


def review_candidate(*, provider: str, model: str, api_key: str, candidate: dict[str, Any]) -> dict[str, Any]:
    if provider == "openrouter":
        return call_openrouter(api_key=api_key, model=model, candidate=candidate)
    if provider == "gemini":
        return call_gemini(api_key=api_key, model=model, candidate=candidate)
    raise ValueError(f"unsupported provider: {provider}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review one discovery candidate with a configured LLM provider.")
    parser.add_argument(
        "--provider",
        choices=tuple(DEFAULT_MODELS),
        default=os.environ.get("LLM_PROVIDER", "openrouter"),
        help="LLM API provider. Defaults to LLM_PROVIDER or openrouter.",
    )
    parser.add_argument("--model", help="Provider model ID. Defaults to LLM_MODEL or the provider default.")
    return parser


def error_result(reason: str) -> int:
    print(json.dumps({"status": "error", "reason": reason}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = provider_api_key(args.provider)
    if not api_key:
        return error_result(f"{API_KEY_ENV[args.provider]} environment variable is not set")
    if not api_key.isascii():
        return error_result(f"{API_KEY_ENV[args.provider]} must be an ASCII value")

    try:
        candidate = read_stdin_json()
    except (json.JSONDecodeError, ValueError) as exc:
        return error_result(f"invalid stdin JSON: {exc}")

    model = provider_model(args.provider, args.model)
    try:
        review = review_candidate(
            provider=args.provider,
            model=model,
            api_key=api_key,
            candidate=candidate,
        )
    except Exception as exc:
        return error_result(f"{args.provider} review failed: {exc}")

    print(json.dumps(review, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
