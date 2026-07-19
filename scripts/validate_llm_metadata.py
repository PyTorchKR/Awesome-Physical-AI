#!/usr/bin/env python3
"""
LLM-based validation for submitted tags and summaries.

This script compares repository metadata against README and paper abstract
evidence, asks Gemini for a structured JSON verdict, then writes a report
consumable by pytest and GitHub Actions.

Usage:
  python scripts/validate_llm_metadata.py
  python scripts/validate_llm_metadata.py --output reports/llm.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_REPORT_PATH = ROOT / "reports" / "llm_validation_report.json"
DEFAULT_SUMMARY_PATH = ROOT / "reports" / "llm_validation_summary.md"

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
REQUEST_TIMEOUT_SECONDS = 30
MAX_EVIDENCE_CHARS = 6000
VALID_VERDICTS = {"pass", "warning", "fail"}
RETRYABLE_STATUS_CODES = {429, 503}
MAX_GEMINI_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "tag_score": {"type": "NUMBER"},
        "summary_score": {"type": "NUMBER"},
        "final_verdict": {"type": "STRING", "enum": ["pass", "warning", "fail"]},
        "reason": {"type": "STRING"},
        "unsupported_tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "unsupported_claims": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": [
        "tag_score",
        "summary_score",
        "final_verdict",
        "reason",
        "unsupported_tags",
        "unsupported_claims",
    ],
}


@dataclass
class EvidenceBundle:
    readme_text: str = ""
    abstract_text: str = ""
    issues: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    entry_id: str
    entry_type: str
    name: str
    tags: list[str]
    summary: str
    summary_ko: str
    tag_score: float
    summary_score: float
    final_verdict: str
    reason: str
    unsupported_tags: list[str]
    unsupported_claims: list[str]
    evidence_sources: list[str]
    evidence_issues: list[str]


def load_yaml(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or []


def iter_entries() -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for entry_type, file_name in (
        ("model", "models.yaml"),
        ("dataset", "datasets.yaml"),
        ("tool", "tools.yaml"),
    ):
        for entry in load_yaml(DATA_DIR / file_name):
            entries.append((entry_type, entry))
    return entries


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, limit: int = MAX_EVIDENCE_CHARS) -> str:
    text = sanitize_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def parse_github_repo(github_url: str) -> tuple[str, str] | None:
    if not github_url:
        return None
    parsed = urlparse(github_url)
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


class EvidenceFetcher:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def fetch(self, entry: dict[str, Any]) -> EvidenceBundle:
        bundle = EvidenceBundle()

        github_repo = parse_github_repo(entry.get("github_url", ""))
        if github_repo:
            owner, repo = github_repo
            try:
                bundle.readme_text = self.fetch_github_readme(owner, repo)
            except Exception as exc:  # pragma: no cover - network behavior
                bundle.issues.append(f"README fetch failed: {exc}")
        else:
            bundle.issues.append("README evidence unavailable: missing or unsupported github_url")

        paper_url = entry.get("paper_url", "")
        if paper_url:
            try:
                bundle.abstract_text = self.fetch_paper_abstract(paper_url)
            except Exception as exc:  # pragma: no cover - network behavior
                bundle.issues.append(f"Abstract fetch failed: {exc}")
        else:
            bundle.issues.append("Abstract evidence unavailable: missing paper_url")

        bundle.readme_text = truncate_text(bundle.readme_text)
        bundle.abstract_text = truncate_text(bundle.abstract_text)
        return bundle

    def fetch_github_readme(self, owner: str, repo: str) -> str:
        url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        response = self.session.get(
            url,
            headers={
                "Accept": "application/vnd.github.raw+json",
                "User-Agent": "awesome-physical-ai-validator",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.text

    def fetch_paper_abstract(self, paper_url: str) -> str:
        if "arxiv.org" in paper_url:
            return self.fetch_arxiv_abstract(paper_url)
        response = self.session.get(
            paper_url,
            headers={"User-Agent": "awesome-physical-ai-validator"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.text

    def fetch_arxiv_abstract(self, paper_url: str) -> str:
        response = self.session.get(
            paper_url,
            headers={"User-Agent": "awesome-physical-ai-validator"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        html = response.text
        match = re.search(
            r'<meta\s+name="citation_abstract"\s+content="([^"]+)"',
            html,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r'<blockquote class="abstract[^"]*">\s*<span[^>]*>Abstract:</span>(.*?)</blockquote>',
                html,
                flags=re.IGNORECASE | re.DOTALL,
            )
        if not match:
            raise ValueError("could not find abstract in paper page")
        return sanitize_text(match.group(1))


class GeminiValidator:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.session = session or requests.Session()

    def validate(
        self,
        entry_type: str,
        entry: dict[str, Any],
        evidence: EvidenceBundle,
    ) -> dict[str, Any]:
        prompt = build_prompt(entry_type, entry, evidence)

        last_retryable_error: requests.HTTPError | None = None

        for attempt in range(1, MAX_GEMINI_RETRIES + 1):
            response = self.session.post(
                GEMINI_API_URL.format(model=self.model),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "responseMimeType": "application/json",
                        "responseSchema": VALIDATION_SCHEMA,
                    },
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                print("Gemini API request failed.")
                print(f"Model: {self.model}")
                print(f"Status code: {response.status_code}")
                print(f"Request URL: {response.request.url}")
                print(f"Response body: {response.text}")

                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_retryable_error = exc
                    if attempt < MAX_GEMINI_RETRIES:
                        backoff = RETRY_BACKOFF_SECONDS[
                            min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                        ]
                        print(
                            f"Retrying Gemini request for {entry.get('id', '<no-id>')} "
                            f"({attempt}/{MAX_GEMINI_RETRIES}) after {backoff:.1f}s"
                        )
                        time.sleep(backoff)
                        continue
                    raise RetryableGeminiError(
                        entry_id=entry.get("id", ""),
                        status_code=response.status_code,
                        attempts=attempt,
                        message=f"Gemini request failed after {attempt} attempts",
                    ) from exc

                raise exc

            payload = response.json()
            text = extract_gemini_text(payload)
            result = json.loads(text)
            return normalize_validation_result(result)

        if last_retryable_error is not None:
            raise RetryableGeminiError(
                entry_id=entry.get("id", ""),
                status_code=0,
                attempts=MAX_GEMINI_RETRIES,
                message="Gemini request exhausted retries",
            ) from last_retryable_error
        raise RuntimeError("Gemini validation reached an unexpected state")


class MockGeminiValidator:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses

    def validate(
        self,
        entry_type: str,
        entry: dict[str, Any],
        evidence: EvidenceBundle,
    ) -> dict[str, Any]:
        entry_id = entry.get("id", "")
        if entry_id not in self.responses:
            raise KeyError(f"missing mock response for '{entry_id}'")
        response = self.responses[entry_id]
        if isinstance(response, Exception):
            raise response
        return normalize_validation_result(response)


class RetryableGeminiError(Exception):
    def __init__(self, entry_id: str, status_code: int, attempts: int, message: str) -> None:
        super().__init__(message)
        self.entry_id = entry_id
        self.status_code = status_code
        self.attempts = attempts


def extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response did not include any candidates")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        raise ValueError("Gemini response did not include any content parts")
    text = parts[0].get("text", "")
    if not text:
        raise ValueError("Gemini response part did not include text")
    return text


def normalize_validation_result(result: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in VALIDATION_SCHEMA["required"] if key not in result]
    if missing:
        raise ValueError(f"Gemini JSON missing required keys: {missing}")

    normalized = {
        "tag_score": clamp_score(result["tag_score"]),
        "summary_score": clamp_score(result["summary_score"]),
        "final_verdict": str(result["final_verdict"]).lower(),
        "reason": str(result["reason"]).strip(),
        "unsupported_tags": [str(item) for item in result["unsupported_tags"]],
        "unsupported_claims": [str(item) for item in result["unsupported_claims"]],
    }

    if normalized["final_verdict"] not in VALID_VERDICTS:
        raise ValueError(
            f"Gemini final_verdict must be one of {sorted(VALID_VERDICTS)}, "
            f"got '{normalized['final_verdict']}'"
        )
    if not normalized["reason"]:
        raise ValueError("Gemini reason must not be empty")
    return normalized


def clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"score must be numeric, got {value!r}") from exc
    if numeric < 0:
        return 0.0
    if numeric > 1:
        return 1.0
    return round(numeric, 3)


def build_prompt(entry_type: str, entry: dict[str, Any], evidence: EvidenceBundle) -> str:
    return f"""
당신은 Awesome Physical AI 저장소에 등록된 항목의 메타데이터를 검증하는 엄격한 검수자입니다.

검증 목표:
사용자가 제출한 태그, 영어 요약문, 한국어 요약문이 GitHub README 및 논문 Abstract 근거에 비추어 적절한지 판단하세요.

반드시 지켜야 할 기준:
1. 제공된 README evidence와 Abstract evidence만 근거로 사용하세요.
2. 외부 지식이나 일반적인 추측을 사용하지 마세요.
3. 태그가 그럴듯해 보여도 README 또는 Abstract에서 뒷받침되지 않으면 unsupported_tags에 포함하세요.
4. 영어/한국어 요약문에 README 또는 Abstract에서 확인되지 않는 주장, 과장된 표현, 잘못된 적용 분야, 잘못된 성능 주장, 잘못된 모델 설명이 있으면 unsupported_claims에 포함하세요.
5. README는 구현 방식, 프레임워크, 설치/사용법, 코드 기반 기능을 판단할 때 우선 근거로 사용하세요.
6. 논문 Abstract는 연구 목적, 핵심 기여, 적용 도메인, 제안 방법, 실험 대상 등을 판단할 때 우선 근거로 사용하세요.
7. README 또는 Abstract가 누락되었거나 근거가 약한 경우, 명확히 틀린 주장이 아니라면 fail보다 warning을 우선 사용하세요.
8. 명확히 근거와 모순되는 태그나 요약문 주장이 있으면 fail을 사용할 수 있습니다.
9. tag_score와 summary_score는 0.0 이상 1.0 이하의 숫자로 작성하세요.
10. final_verdict는 반드시 pass, warning, fail 중 하나만 사용하세요.
11. reason은 반드시 한국어로 작성하세요.
12. 응답은 설명 문장 없이 JSON만 반환하세요.

판정 기준:
- pass: 태그와 요약문이 README 및 Abstract 근거로 충분히 뒷받침됨
- warning: 근거가 부족하거나 일부 태그/주장이 약하게만 뒷받침됨
- fail: 태그 또는 요약문이 근거와 명확히 모순되거나 핵심 내용이 잘못됨

반환해야 하는 JSON 스키마:
{json.dumps(VALIDATION_SCHEMA, ensure_ascii=False)}

검증 항목 정보:
- Entry type: {entry_type}
- Entry id: {entry.get("id", "")}
- Entry name: {entry.get("name", "")}
- Submitted tags: {json.dumps(entry.get("tags", []), ensure_ascii=False)}
- Submitted English summary: {entry.get("description_en", "")}
- Submitted Korean summary: {entry.get("description_ko", "")}

README evidence:
{evidence.readme_text or "(missing)"}

Abstract evidence:
{evidence.abstract_text or "(missing)"}
""".strip()


def validate_entry(
    entry_type: str,
    entry: dict[str, Any],
    fetcher: EvidenceFetcher,
    validator: GeminiValidator | MockGeminiValidator,
) -> ValidationResult:
    evidence = fetcher.fetch(entry)
    evidence_sources: list[str] = []
    if evidence.readme_text:
        evidence_sources.append("README")
    if evidence.abstract_text:
        evidence_sources.append("Abstract")

    try:
        llm_result = validator.validate(entry_type, entry, evidence)
    except RetryableGeminiError as exc:
        warning_reason = (
            f"Gemini validation temporarily unavailable after {exc.attempts} attempts "
            f"(status {exc.status_code}); marked as warning and continued."
        )
        issues = list(evidence.issues)
        issues.append(warning_reason)
        return ValidationResult(
            entry_id=entry.get("id", ""),
            entry_type=entry_type,
            name=entry.get("name", ""),
            tags=entry.get("tags", []),
            summary=entry.get("description_en", ""),
            summary_ko=entry.get("description_ko", ""),
            tag_score=0.0,
            summary_score=0.0,
            final_verdict="warning",
            reason=warning_reason,
            unsupported_tags=[],
            unsupported_claims=[],
            evidence_sources=evidence_sources,
            evidence_issues=issues,
        )

    return ValidationResult(
        entry_id=entry.get("id", ""),
        entry_type=entry_type,
        name=entry.get("name", ""),
        tags=entry.get("tags", []),
        summary=entry.get("description_en", ""),
        summary_ko=entry.get("description_ko", ""),
        tag_score=llm_result["tag_score"],
        summary_score=llm_result["summary_score"],
        final_verdict=llm_result["final_verdict"],
        reason=llm_result["reason"],
        unsupported_tags=llm_result["unsupported_tags"],
        unsupported_claims=llm_result["unsupported_claims"],
        evidence_sources=evidence_sources,
        evidence_issues=evidence.issues,
    )


def build_report(
    results: list[ValidationResult],
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    counts = {"pass": 0, "warning": 0, "fail": 0}
    for result in results:
        counts[result.final_verdict] += 1

    return {
        "status": "skipped" if skipped_reason else "completed",
        "skipped_reason": skipped_reason,
        "counts": counts,
        "results": [asdict(result) for result in results],
    }


def render_actions_summary(report: dict[str, Any]) -> str:
    lines = [
        "# LLM Metadata Validation",
        "",
        f"- Status: `{report['status']}`",
        f"- Pass: `{report['counts']['pass']}`",
        f"- Warning: `{report['counts']['warning']}`",
        f"- Fail: `{report['counts']['fail']}`",
    ]
    skipped_reason = report.get("skipped_reason")
    if skipped_reason:
        lines.extend(["", f"- Skipped reason: {skipped_reason}"])
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            "| Entry | Type | Verdict | Tag score | Summary score | Notes |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for result in report["results"]:
        notes: list[str] = [f"reason: {result['reason']}"]
        if result["unsupported_tags"]:
            notes.append("unsupported tags: " + ", ".join(result["unsupported_tags"]))
        if result["unsupported_claims"]:
            notes.append("unsupported claims: " + ", ".join(result["unsupported_claims"]))
        if result["evidence_issues"]:
            notes.append("evidence issues: " + "; ".join(result["evidence_issues"]))
        lines.append(
            f"| {result['entry_id']} | {result['entry_type']} | {result['final_verdict']} "
            f"| {result['tag_score']:.2f} | {result['summary_score']:.2f} | {' / '.join(notes)} |"
        )
    return "\n".join(lines) + "\n"


def determine_exit_code(report: dict[str, Any]) -> int:
    return 0


def write_report_files(report: dict[str, Any], output: Path, summary_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(render_actions_summary(report), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tags and summaries with Gemini")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--mock-response-file", type=Path, default=None)
    parser.add_argument(
        "--entry-ids",
        type=str,
        default="",
        help="Comma-separated entry IDs to validate. Empty means validate all entries.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_ids = {item.strip() for item in args.entry_ids.split(",") if item.strip()}
    entries = [
        (entry_type, entry)
        for entry_type, entry in iter_entries()
        if (
            (entry.get("tags") or entry.get("description_en") or entry.get("description_ko"))
            and (not requested_ids or entry.get("id", "") in requested_ids)
        )
    ]

    fetcher = EvidenceFetcher()

    if args.mock_response_file:
        mock_responses = json.loads(args.mock_response_file.read_text(encoding="utf-8"))
        validator: GeminiValidator | MockGeminiValidator = MockGeminiValidator(mock_responses)
    else:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            report = build_report([], skipped_reason="GEMINI_API_KEY is not configured")
            write_report_files(report, args.output, args.summary_output)
            print(f"LLM validation skipped: {report['skipped_reason']}")
            return determine_exit_code(report)
        validator = GeminiValidator(api_key=api_key)

    results: list[ValidationResult] = []
    for entry_type, entry in entries:
        result = validate_entry(entry_type, entry, fetcher=fetcher, validator=validator)
        results.append(result)
        print(
            f"[{result.final_verdict}] {result.entry_type}:{result.entry_id} "
            f"(tag={result.tag_score:.2f}, summary={result.summary_score:.2f})"
        )

    report = build_report(results)
    write_report_files(report, args.output, args.summary_output)
    print(f"Report written to {args.output}")
    return determine_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
