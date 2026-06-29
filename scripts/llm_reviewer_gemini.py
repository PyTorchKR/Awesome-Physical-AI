"""
Gemini-based LLM reviewer for scripts/discover_new.py.

This script reads one candidate-review JSON object from stdin and returns one
JSON object to stdout. It is intended to be used through:

python scripts/discover_new.py \
  --llm-review-mode ambiguous \
  --llm-review-command "python scripts/llm_reviewer_gemini.py"
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-2.5-flash-lite"


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_verified_model_link": {"type": "boolean"},
        "has_verified_artifact_link": {"type": "boolean"},
        "entry_type": {
            "type": "string",
            "enum": ["model", "dataset", "tool", "benchmark", "simulator", "paper_only", "irrelevant", "unclear"],
        },
        "decision": {
            "type": "string",
            "enum": ["accept", "needs_review", "reject"],
        },
        "entry_summary": {"type": "string"},
        "maintainer_summary": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "has_verified_model_link",
        "has_verified_artifact_link",
        "entry_type",
        "decision",
        "entry_summary",
        "maintainer_summary",
        "reason",
    ],
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
- The entry_summary should be a public-facing 2-3 sentence description that could be used as an Awesome-list item description.
- The entry_summary must summarize the candidate's task, method/artifact, and Physical AI relevance, but must not invent artifact availability.
- Always write maintainer_summary as a concise 2-3 sentence review note for weekly GitHub issue triage.
- The maintainer_summary should explain inclusion relevance, artifact availability, and any caution such as paper-only, unofficial, gated, placeholder, or inconclusive links.
- Do not invent links, stars, datasets, models, code releases, benchmarks, or claims not present in the input.
- Return only valid JSON matching the requested schema.
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
    return json.dumps(candidate, ensure_ascii=False, indent=2)


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": "GEMINI_API_KEY environment variable is not set",
                },
                ensure_ascii=False,
            )
        )
        return 0

    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    try:
        candidate = read_stdin_json()
    except (json.JSONDecodeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": f"invalid stdin JSON: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 0

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=SYSTEM_PROMPT + "\n\nCandidate JSON:\n" + build_user_prompt(candidate)
                        )
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )

        text = response.text or ""
        review = json.loads(text)
        if not isinstance(review, dict):
            raise ValueError("Gemini returned non-object JSON")

        review.setdefault("status", "ok")
        print(json.dumps(review, ensure_ascii=False))
        return 0

    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": f"Gemini review failed: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
