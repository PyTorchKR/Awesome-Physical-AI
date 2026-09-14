"""Submit high-confidence discovery candidates through the Add a Model flow.

The repository already turns an ``add-model`` issue into a pull request. This
script renders crawler output in that issue form's exact structure, so the
maintainer reviews the generated PR instead of triaging a separate free-form
discovery issue first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import requests

from discover_weekly import (
    canonicalize_seen_url,
    load_seen_cache,
    mark_seen_submitted,
    write_seen_cache,
)


GITHUB_API_URL = "https://api.github.com"
DISCOVERY_MARKER_RE = re.compile(r"<!--\s*discovery-key:\s*(.+?)\s*-->")

VALID_CATEGORIES = ("manipulation", "locomotion", "navigation", "dexterous", "whole-body", "aerial")
VALID_HARDWARE = ("manipulator", "humanoid", "quadruped", "biped", "mobile", "drone", "hand")
VALID_LEARNING = ("VLA", "IL", "RL", "diffusion", "world_model", "sim2real")
VALID_FRAMEWORK = ("pytorch", "jax", "tensorflow", "other")
VALID_COMMUNICATION = ("ros2", "grpc", "lcm", "zenoh", "other")

CATEGORY_TERMS = {
    "manipulation": ("manipulation", "manipulator", "grasp", "insertion", "pick-and-place", "bimanual"),
    "locomotion": ("locomotion", "gait", "walking", "running"),
    "navigation": ("navigation", "vln", "path planning", "mobile robot"),
    "dexterous": ("dexterous", "in-hand", "in hand manipulation"),
    "whole-body": ("whole-body", "whole body"),
    "aerial": ("aerial", "drone", "quadrotor", "uav"),
}
HARDWARE_TERMS = {
    "manipulator": ("manipulation", "manipulator", "robot arm", "grasp", "insertion", "bimanual"),
    "humanoid": ("humanoid",),
    "quadruped": ("quadruped",),
    "biped": ("biped",),
    "mobile": ("navigation", "vln", "mobile robot"),
    "drone": ("drone", "aerial", "quadrotor", "uav"),
    "hand": ("dexterous", "robot hand", "in-hand"),
}
LEARNING_TERMS = {
    "VLA": ("vision-language-action", "vision language action", "vla"),
    "IL": ("imitation learning", "behavior cloning", "behaviour cloning", "demonstration"),
    "RL": ("reinforcement learning",),
    "diffusion": ("diffusion",),
    "world_model": ("world model",),
    "sim2real": ("sim-to-real", "sim to real", "sim2real"),
}
FRAMEWORK_TERMS = {
    "pytorch": ("pytorch",),
    "jax": ("jax",),
    "tensorflow": ("tensorflow",),
}
COMMUNICATION_TERMS = {
    "ros2": ("ros2", "ros 2"),
    "grpc": ("grpc",),
    "lcm": ("lightweight communications and marshalling", "lcm"),
    "zenoh": ("zenoh",),
}


@dataclass(frozen=True)
class ModelSubmission:
    key: str
    id: str
    name: str
    organization: str
    year: int
    description_en: str
    description_ko: str
    github_url: str
    paper_url: str
    huggingface_url: str
    project_page_url: str
    categories: tuple[str, ...]
    hardware_targets: tuple[str, ...]
    learning_methods: tuple[str, ...]
    framework: tuple[str, ...]
    communication: tuple[str, ...]
    tags: tuple[str, ...]
    evidence: tuple[str, ...]


def load_candidates(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("discovery JSON must be a list")
    return [item for item in data if isinstance(item, dict)]


def candidate_submission_key(candidate: dict[str, Any]) -> str:
    return canonicalize_seen_url(str(candidate.get("url") or "").strip())


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-+", "-", value)


def clean_text(value: Any, *, limit: int = 1000) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def available_checks(candidate: dict[str, Any], kind: str) -> list[str]:
    return [
        clean_text(check.get("url"))
        for check in candidate.get("checks") or []
        if isinstance(check, dict)
        and check.get("status") == "available"
        and check.get("kind") == kind
        and check.get("url")
    ]


def first_url(*groups: list[str]) -> str:
    for group in groups:
        for url in group:
            if url:
                return url
    return ""


def infer_terms(text: str, mapping: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(label for label, terms in mapping.items() if any(term in lowered for term in terms))


def normalized_llm_list(llm_review: dict[str, Any], field: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if llm_review.get("status") != "ok":
        return ()
    values = llm_review.get(field) or []
    if not isinstance(values, list):
        return ()
    return tuple(value for value in allowed if value in values)


def derive_model_name(candidate: dict[str, Any], huggingface_url: str) -> str:
    llm_review = candidate.get("llm_review") or {}
    if llm_review.get("status") == "ok":
        llm_name = clean_text(llm_review.get("model_name"), limit=100)
        if llm_name:
            return llm_name

    title = clean_text(candidate.get("title"), limit=200)
    prefix, separator, _ = title.partition(":")
    if separator and 1 < len(prefix) <= 60:
        return prefix.strip()

    if huggingface_url:
        segments = [segment for segment in urlsplit(huggingface_url).path.split("/") if segment]
        if len(segments) >= 2:
            return segments[-1]

    return title[:100]


def derive_organization(candidate: dict[str, Any], *official_urls: str) -> str:
    llm_review = candidate.get("llm_review") or {}
    if llm_review.get("status") == "ok":
        organization = clean_text(llm_review.get("organization"), limit=120)
        if organization:
            return organization

    for url in official_urls:
        segments = [segment for segment in urlsplit(url).path.split("/") if segment]
        if segments and segments[0] not in {"datasets", "spaces"}:
            return segments[0]
        if len(segments) >= 2:
            return segments[1]
    return ""


def derive_description(candidate: dict[str, Any]) -> str:
    llm_review = candidate.get("llm_review") or {}
    if llm_review.get("status") == "ok":
        entry_summary = clean_text(llm_review.get("entry_summary"), limit=700)
        if entry_summary:
            return entry_summary

    summary = clean_text(candidate.get("summary"), limit=1200)
    sentences = re.split(r"(?<=[.!?])\s+", summary)
    return " ".join(sentences[:2])[:700].strip()


def build_model_submission(candidate: dict[str, Any]) -> tuple[ModelSubmission | None, list[str]]:
    reasons: list[str] = []
    availability = candidate.get("artifact_availability") or {}

    if candidate.get("recommendation") != "needs_review" or candidate.get("review_bucket") == "reject":
        reasons.append("candidate is not eligible for maintainer review")
    if candidate.get("duplicate_matches"):
        reasons.append("candidate matches an existing repository entry")
    if candidate.get("exclusion_hits"):
        reasons.append("candidate contains an exclusion keyword")
    if candidate.get("relevance") != "high":
        reasons.append("candidate relevance is not high")
    has_verified_release = bool(
        availability.get("has_verified_model_link") or availability.get("has_verified_code_link")
    )
    if not has_verified_release:
        reasons.append("rule-based verification found neither public model weights nor public code")

    verified_model_links = [clean_text(url) for url in availability.get("verified_model_links") or [] if url]
    huggingface_url = first_url(verified_model_links, available_checks(candidate, "hf_model"))
    github_url = first_url(available_checks(candidate, "github"))
    if not (huggingface_url or github_url):
        reasons.append("verified model/code URL is missing")
    project_page_url = first_url(
        [clean_text(url) for url in availability.get("verified_project_pages") or [] if url],
        available_checks(candidate, "project"),
    )
    paper_url = candidate_submission_key(candidate)
    if not paper_url:
        reasons.append("paper URL is missing")

    model_name = derive_model_name(candidate, huggingface_url)
    model_id = slugify(model_name)
    organization = derive_organization(candidate, huggingface_url, github_url)
    description_en = derive_description(candidate)
    if not model_name or not model_id:
        reasons.append("model name or slug could not be derived")
    if not organization:
        reasons.append("organization could not be derived from reviewed metadata or an official repository")
    if not description_en:
        reasons.append("English description could not be derived")

    published = clean_text(candidate.get("published"), limit=10)
    year = int(published[:4]) if re.match(r"^\d{4}", published) else 0
    if not 2015 <= year <= 2030:
        reasons.append("publication year is missing or outside the accepted range")

    evidence_text = " ".join(
        [
            clean_text(candidate.get("title"), limit=300),
            clean_text(candidate.get("summary"), limit=2000),
            " ".join(str(hit) for hit in candidate.get("keyword_hits") or []),
        ]
    )
    llm_review = candidate.get("llm_review") or {}
    categories = normalized_llm_list(llm_review, "categories", VALID_CATEGORIES) or infer_terms(
        evidence_text, CATEGORY_TERMS
    )
    hardware = normalized_llm_list(llm_review, "hardware_targets", VALID_HARDWARE) or infer_terms(
        evidence_text, HARDWARE_TERMS
    )
    learning = normalized_llm_list(llm_review, "learning_methods", VALID_LEARNING) or infer_terms(
        evidence_text, LEARNING_TERMS
    )
    framework = normalized_llm_list(llm_review, "framework", VALID_FRAMEWORK) or infer_terms(
        evidence_text, FRAMEWORK_TERMS
    )
    communication = normalized_llm_list(llm_review, "communication", VALID_COMMUNICATION) or infer_terms(
        evidence_text, COMMUNICATION_TERMS
    )
    if not categories:
        reasons.append("no supported model category could be derived")
    if not hardware:
        reasons.append("no supported hardware target could be derived")
    if not learning:
        reasons.append("no supported learning method could be derived")

    if reasons:
        return None, reasons

    tags = tuple(dict.fromkeys(clean_text(hit, limit=50) for hit in candidate.get("keyword_hits") or [] if hit))
    evidence = tuple(str(reason) for reason in candidate.get("reasons") or [])
    source_run = clean_text(candidate.get("source_run"), limit=500)
    if source_run:
        evidence += (f"workflow: {source_run}",)

    return ModelSubmission(
        key=paper_url,
        id=model_id,
        name=model_name,
        organization=organization,
        year=year,
        description_en=description_en,
        description_ko="",
        github_url=github_url,
        paper_url=paper_url,
        huggingface_url=huggingface_url,
        project_page_url=project_page_url,
        categories=categories,
        hardware_targets=hardware,
        learning_methods=learning,
        framework=framework,
        communication=communication,
        tags=tags[:12],
        evidence=evidence,
    ), []


def select_submissions(
    candidates: list[dict[str, Any]], *, limit: int
) -> tuple[list[ModelSubmission], list[tuple[str, list[str]]]]:
    submissions: list[ModelSubmission] = []
    skipped: list[tuple[str, list[str]]] = []
    for candidate in candidates:
        submission, reasons = build_model_submission(candidate)
        if submission is None:
            skipped.append((clean_text(candidate.get("title"), limit=150) or "Untitled", reasons))
            continue
        submissions.append(submission)
        if limit > 0 and len(submissions) >= limit:
            break
    return submissions, skipped


def render_checkboxes(options: tuple[str, ...], selected: tuple[str, ...]) -> list[str]:
    selected_set = set(selected)
    return [f"- [{'x' if option in selected_set else ' '}] {option}" for option in options]


def render_add_model_issue(submission: ModelSubmission, *, source_run: str | None = None) -> str:
    lines = [
        f"<!-- discovery-key: {submission.key} -->",
        "### ID (slug)",
        submission.id,
        "",
        "### Name",
        submission.name,
        "",
        "### Organization",
        submission.organization,
        "",
        "### Year",
        str(submission.year),
        "",
        "### Description (English)",
        submission.description_en,
        "",
        "### Description (Korean)",
        submission.description_ko or "_No response_",
        "",
        "### GitHub URL",
        submission.github_url or "_No response_",
        "",
        "### Paper URL (arXiv)",
        submission.paper_url,
        "",
        "### HuggingFace URL",
        submission.huggingface_url or "_No response_",
        "",
        "### Project Page URL",
        submission.project_page_url or "_No response_",
        "",
        "### Categories",
        *render_checkboxes(VALID_CATEGORIES, submission.categories),
        "",
        "### Hardware Targets",
        *render_checkboxes(VALID_HARDWARE, submission.hardware_targets),
        "",
        "### Learning Methods",
        *render_checkboxes(VALID_LEARNING, submission.learning_methods),
        "",
        "### Framework",
        *render_checkboxes(VALID_FRAMEWORK, submission.framework),
        "",
        "### Communication",
        *render_checkboxes(VALID_COMMUNICATION, submission.communication),
        "",
        "### Tags (optional)",
        ", ".join(submission.tags) or "_No response_",
        "",
        "### Checklist",
        "- [x] The model is open-source (code or weights publicly available)",
        "- [x] At least one URL (GitHub, paper, or HuggingFace) is provided",
        "- [x] I have read the contribution guidelines",
        "",
        "### Discovery automation evidence",
        f"- Rule-based verified release URL: {submission.huggingface_url or submission.github_url}",
        f"- Canonical paper: {submission.paper_url}",
    ]
    lines.extend(f"- {evidence}" for evidence in submission.evidence)
    if source_run:
        lines.append(f"- Workflow run: {source_run}")
    return "\n".join(lines).rstrip() + "\n"


def github_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def existing_discovery_keys(*, repo: str, token: str) -> set[str]:
    keys: set[str] = set()
    page = 1
    while True:
        response = requests.get(
            f"{GITHUB_API_URL}/repos/{repo}/issues",
            headers=github_headers(token),
            params={"state": "all", "labels": "add-model", "per_page": 100, "page": page},
            timeout=30,
        )
        response.raise_for_status()
        issues = response.json()
        if not isinstance(issues, list):
            raise ValueError("GitHub issues API returned a non-list response")
        for issue in issues:
            if not isinstance(issue, dict) or "pull_request" in issue:
                continue
            body = str(issue.get("body") or "")
            marker = DISCOVERY_MARKER_RE.search(body)
            if marker:
                keys.add(canonicalize_seen_url(marker.group(1).strip()))
        if len(issues) < 100:
            break
        page += 1
    return keys


def create_add_model_issue(*, repo: str, token: str, submission: ModelSubmission, body: str) -> str:
    response = requests.post(
        f"{GITHUB_API_URL}/repos/{repo}/issues",
        headers=github_headers(token),
        json={"title": f"[Model] {submission.name}", "body": body, "labels": ["add-model"]},
        timeout=30,
    )
    response.raise_for_status()
    return str(response.json().get("html_url") or "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit high-confidence discovery candidates through the Add a Model issue-to-PR flow."
    )
    parser.add_argument("--input", required=True, help="Path to discover_new JSON output.")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""), help="GitHub repo in owner/name form.")
    parser.add_argument(
        "--token",
        default=os.environ.get("DISCOVERY_BOT_TOKEN") or os.environ.get("GITHUB_TOKEN", ""),
        help="Token with issues:write. In Actions, use a bot/App token rather than GITHUB_TOKEN.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Maximum Add a Model submissions. Use 0 for no limit.")
    parser.add_argument("--source-run", help="Workflow run URL to include as provenance.")
    parser.add_argument("--seen-cache", help="Mark created or existing submissions as terminal in this cache.")
    parser.add_argument("--dry-run", action="store_true", help="Render submissions without creating issues.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = load_candidates(args.input)
    eligible, skipped = select_submissions(candidates, limit=0)

    for title, reasons in skipped:
        print(f"Skipped: {title}: {'; '.join(reasons)}", file=sys.stderr)

    if not eligible:
        print("No PR-ready model candidates selected for submission.")
        return 0

    if args.dry_run:
        submissions = eligible[:args.limit] if args.limit > 0 else eligible
        for submission in submissions:
            body = render_add_model_issue(submission, source_run=args.source_run)
            print(f"[dry-run] Would submit Add a Model issue: [Model] {submission.name}")
            print(body)
        return 0

    if not args.repo or not args.token:
        print(
            "error: --repo/GITHUB_REPOSITORY and DISCOVERY_BOT_TOKEN are required unless --dry-run is used.",
            file=sys.stderr,
        )
        return 2

    existing_keys = existing_discovery_keys(repo=args.repo, token=args.token)
    submitted_keys = {submission.key for submission in eligible if submission.key in existing_keys}
    submissions = [submission for submission in eligible if submission.key not in existing_keys]
    if args.limit > 0:
        submissions = submissions[:args.limit]

    for key in sorted(submitted_keys):
        print(f"Skipped existing Add a Model submission: {key}")

    for submission in submissions:
        body = render_add_model_issue(submission, source_run=args.source_run)
        url = create_add_model_issue(repo=args.repo, token=args.token, submission=submission, body=body)
        print(f"Created Add a Model issue (automatic PR trigger): {url}")
        submitted_keys.add(submission.key)

    if args.seen_cache and submitted_keys:
        seen_cache = load_seen_cache(args.seen_cache)
        mark_seen_submitted(seen_cache, submitted_keys)
        write_seen_cache(args.seen_cache, seen_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
