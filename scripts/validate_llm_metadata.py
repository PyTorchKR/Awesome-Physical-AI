#!/usr/bin/env python3
"""
LLM-based validation for submitted tags and summaries.

This script compares repository metadata against README, paper abstract, and
paper introduction evidence, asks Gemini for a structured JSON verdict, then writes a report
consumable by pytest and GitHub Actions.

Usage:
  python scripts/validate_llm_metadata.py
  python scripts/validate_llm_metadata.py --no-print-evidence
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
        "summary_specificity_score": {"type": "NUMBER"},
        "final_verdict": {"type": "STRING", "enum": ["pass", "warning", "fail"]},
        "reason": {"type": "STRING"},
        "unsupported_tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "unsupported_claims": {"type": "ARRAY", "items": {"type": "STRING"}},
        "evidence_citations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "claim": {"type": "STRING"},
                    "source": {"type": "STRING", "enum": ["README", "Abstract", "Introduction"]},
                    "quote": {"type": "STRING"},
                    "relation": {"type": "STRING", "enum": ["supports", "contradicts"]},
                },
                "required": ["claim", "source", "quote", "relation"],
            },
        },
    },
    "required": [
        "tag_score",
        "summary_score",
        "summary_specificity_score",
        "final_verdict",
        "reason",
        "unsupported_tags",
        "unsupported_claims",
        "evidence_citations",
    ],
}


@dataclass
class EvidenceBundle:
    readme_text: str = ""
    abstract_text: str = ""
    introduction_text: str = ""
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
    summary_specificity_score: float = 0.0


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
            try:
                bundle.introduction_text = self.fetch_paper_introduction(paper_url)
            except Exception as exc:  # pragma: no cover - network behavior
                bundle.issues.append(f"Introduction fetch failed: {exc}")
        else:
            bundle.issues.append("Abstract evidence unavailable: missing paper_url")
            bundle.issues.append("Introduction evidence unavailable: missing paper_url")

        bundle.readme_text = truncate_text(bundle.readme_text)
        bundle.abstract_text = truncate_text(bundle.abstract_text)
        bundle.introduction_text = truncate_text(bundle.introduction_text)
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

    def fetch_paper_introduction(self, paper_url: str) -> str:
        """Fetch the paper's Introduction section for evidence-bound review."""
        if "arxiv.org" in paper_url:
            return self.fetch_arxiv_introduction(paper_url)
        response = self.session.get(
            paper_url,
            headers={"User-Agent": "awesome-physical-ai-validator"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return self.extract_introduction(response.text)

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

    def fetch_arxiv_introduction(self, paper_url: str) -> str:
        match = re.search(r"arxiv\.org/(?:abs|html|pdf)/([^?#]+)", paper_url)
        if not match:
            raise ValueError("could not determine arXiv paper identifier")
        paper_id = match.group(1).removesuffix(".pdf")
        response = self.session.get(
            f"https://arxiv.org/html/{paper_id}",
            headers={"User-Agent": "awesome-physical-ai-validator"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return self.extract_introduction(response.text)

    @staticmethod
    def extract_introduction(html: str) -> str:
        """Extract text from the first HTML heading named Introduction."""
        headings = list(
            re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", html, flags=re.IGNORECASE | re.DOTALL)
        )
        for index, heading in enumerate(headings):
            title = sanitize_text(heading.group(2))
            title = re.sub(r"^(?:\d+(?:\.\d+)*|[IVXLC]+\b)\.?\s*", "", title).strip().lower()
            if title not in {"introduction", "intro"}:
                continue
            end = headings[index + 1].start() if index + 1 < len(headings) else len(html)
            introduction = sanitize_text(html[heading.end() : end])
            if introduction:
                return introduction
        raise ValueError("could not find Introduction section in paper page")


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
    missing = [
        key
        for key in VALIDATION_SCHEMA["required"]
        if key not in result and key not in {"summary_specificity_score", "evidence_citations"}
    ]
    if missing:
        raise ValueError(f"Gemini JSON missing required keys: {missing}")

    normalized = {
        "tag_score": clamp_score(result["tag_score"]),
        "summary_score": clamp_score(result["summary_score"]),
        "summary_specificity_score": clamp_score(
            result.get("summary_specificity_score", result["summary_score"])
        ),
        "final_verdict": str(result["final_verdict"]).lower(),
        "reason": str(result["reason"]).strip(),
        "unsupported_tags": [str(item) for item in result["unsupported_tags"]],
        "unsupported_claims": [str(item) for item in result["unsupported_claims"]],
        "evidence_citations": [
            {
                "claim": str(item.get("claim", "")).strip(),
                "source": str(item.get("source", "")).strip(),
                "quote": str(item.get("quote", "")).strip(),
                "relation": str(item.get("relation", "")).strip(),
            }
            for item in result.get("evidence_citations", [])
            if isinstance(item, dict)
        ],
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
    """Build the evidence-bound review prompt for Gemini."""
    return f"""
당신은 Awesome Physical AI 저장소에 등록된 항목의 메타데이터를 검증하는 엄격한 검수자입니다.

제출된 태그와 영어/한국어 요약문을 제공된 GitHub README, 논문 Abstract, 논문 Introduction 근거만으로 검증하세요. 일반 지식이나 추측은 사용하지 마세요.

검증 기준:
1. 근거로 뒷받침되지 않는 태그는 모두 unsupported_tags에 넣으세요.
2. 근거가 없거나 부정확·과장된 요약문 주장은 모두 unsupported_claims에 넣으세요.
3. 사실 여부와 별도로, 각 요약문이 이 항목만의 고유한 핵심 기여를 설명하는지 판단하세요. 사실과 일치해도 지나치게 일반적일 수 있습니다.
4. 넓은 분야, 일반적인 아키텍처·학습 방식·성능/기능만 말하고 이 항목의 제안 메커니즘, 설계 선택, 데이터셋 구성, 벤치마크/프로토콜 또는 구별되는 기여를 빠뜨리면 일반적인 요약문으로 판단하세요.
5. 짧다는 이유만으로 일반적이라고 판단하지 마세요. 근거에 기반한 고유 핵심 메서드나 아이디어를 하나 이상 명확히 설명하고, 관련 있다면 그것이 문제를 어떻게 해결하는지 드러나면 충분히 구체적입니다.
6. 일반적인 요약문이면, 사실 오류가 없더라도 빠진 고유 기여가 무엇인지 한국어로 간결하게 reason에 포함하세요.
7. summary_specificity_score는 0.0~1.0입니다. 1.0은 핵심 기여가 명확히 드러나는 경우, 0.5는 일부만 드러나는 경우, 0.0은 다른 많은 프로젝트와 거의 구별되지 않는 경우입니다.
8. 사실과 일치하지만 일반적인 요약문은 보통 warning으로 판정하세요. 근거와 모순되거나 핵심 내용이 틀린 경우에 fail을 사용하세요.
9. 태그와 주장이 근거에 부합하고 요약문이 충분히 고유할 때만 pass를 사용하세요. reason은 반드시 한국어로 작성하세요.
10. README, Abstract, Introduction 중 하나라도 제공되었으면 evidence_citations를 절대 빈 배열로 반환하지 마세요. unsupported_claims의 각 항목과 최종 판정의 핵심 근거에 대해 evidence_citations를 작성하세요. quote에는 제공된 README, Abstract, Introduction에서 찾은 연속된 문장을 바꾸지 말고 그대로 넣고, source와 그 문장이 주장을 지지하는지( supports ) 또는 반박하는지( contradicts )를 표시하세요. 근거 문장이 없으면 인용을 만들지 마세요.
11. 설명 문장 없이 아래 스키마에 맞는 JSON만 반환하세요.

판단 예시:
- 일반적인 요약문: "이중 시스템 VLA 구조에서 고수준 의도와 저수준 행동을 분리하고, VLM과 flow-matching 기반 정책을 사용한다." 이 설명은 사실일 수 있지만 많은 VLA 프로젝트에 적용될 수 있으므로, 해당 항목만의 기여를 보여 주지 못하면 warning으로 판정하고 누락된 고유 기여를 reason에 적으세요.
- 고유한 요약문: "행동이 잠재 의도를 반드시 통과하도록 병목을 설계해 기존 이중 시스템 VLA의 의도 우회(shortcut)를 차단하고, 행동 그래디언트가 VLM을 정교화하도록 한다." 이처럼 제안한 구조적 장치와 해결하려는 문제를 근거에 맞게 설명하면 충분히 구체적입니다.
- 위 예시는 판단 방식만 보여 주며, 실제 항목을 검증할 때는 반드시 해당 항목에 제공된 README, Abstract, Introduction 근거만 사용하세요.

반환 JSON 스키마:
{json.dumps(VALIDATION_SCHEMA, ensure_ascii=False)}

검증 항목:
- 항목 유형: {entry_type}
- 항목 ID: {entry.get("id", "")}
- 항목명: {entry.get("name", "")}
- 제출 태그: {json.dumps(entry.get("tags", []), ensure_ascii=False)}
- 제출 영어 요약문: {entry.get("description_en", "")}
- 제출 한국어 요약문: {entry.get("description_ko", "")}

README 근거:
{evidence.readme_text or "(missing)"}

Abstract 근거:
{evidence.abstract_text or "(missing)"}

Introduction 근거:
{evidence.introduction_text or "(missing)"}
""".strip()


def print_evidence_citations(entry_id: str, citations: list[dict[str, str]]) -> None:
    """Print LLM-selected verbatim citations without adding them to the JSON report."""
    print(f"\n=== LLM cited evidence for {entry_id} ===")
    if not citations:
        print("(No exact citations returned.)")
    for citation in citations:
        print(
            f"\n[{citation['relation']}] {citation['claim']}\n"
            f"Source: {citation['source']}\n"
            f"Quote: {citation['quote']}"
        )
    print("\n=== End cited evidence ===\n")


def validate_entry(
    entry_type: str,
    entry: dict[str, Any],
    fetcher: EvidenceFetcher,
    validator: GeminiValidator | MockGeminiValidator,
    print_evidence: bool = False,
) -> ValidationResult:
    evidence = fetcher.fetch(entry)
    evidence_sources: list[str] = []
    if evidence.readme_text:
        evidence_sources.append("README")
    if evidence.abstract_text:
        evidence_sources.append("Abstract")
    if evidence.introduction_text:
        evidence_sources.append("Introduction")

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
            summary_specificity_score=0.0,
            final_verdict="warning",
            reason=warning_reason,
            unsupported_tags=[],
            unsupported_claims=[],
            evidence_sources=evidence_sources,
            evidence_issues=issues,
        )

    if print_evidence:
        print_evidence_citations(entry.get("id", ""), llm_result["evidence_citations"])

    return ValidationResult(
        entry_id=entry.get("id", ""),
        entry_type=entry_type,
        name=entry.get("name", ""),
        tags=entry.get("tags", []),
        summary=entry.get("description_en", ""),
        summary_ko=entry.get("description_ko", ""),
        tag_score=llm_result["tag_score"],
        summary_score=llm_result["summary_score"],
        summary_specificity_score=llm_result["summary_specificity_score"],
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
            "| Entry | Type | Verdict | Tag score | Summary score | Specificity | Notes |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for result in report["results"]:
        notes: list[str] = [f"reason: {result['reason']}"]
        if result["unsupported_tags"]:
            notes.append("수정 필요 태그: " + ", ".join(result["unsupported_tags"]))
        if result["unsupported_claims"]:
            notes.append("수정 필요 문구: " + ", ".join(result["unsupported_claims"]))
        if result["evidence_issues"]:
            notes.append("evidence issues: " + "; ".join(result["evidence_issues"]))
        lines.append(
            f"| {result['entry_id']} | {result['entry_type']} | {result['final_verdict']} "
            f"| {result['tag_score']:.2f} | {result['summary_score']:.2f} | "
            f"{result.get('summary_specificity_score', 0.0):.2f} | {' / '.join(notes)} |"
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
    parser.add_argument(
        "--no-print-evidence",
        action="store_false",
        dest="print_evidence",
        default=True,
        help="Do not print LLM-selected evidence citations to the terminal.",
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
        result = validate_entry(
            entry_type,
            entry,
            fetcher=fetcher,
            validator=validator,
            print_evidence=args.print_evidence,
        )
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
