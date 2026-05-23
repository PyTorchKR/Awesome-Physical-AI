#!/usr/bin/env python3
"""
Discover recent Physical AI candidates from paper-first sources.

This script intentionally does not create GitHub issues. It collects recent
arXiv cs.RO papers, extracts official-looking links from paper metadata,
verifies those links, checks for duplicates against data/*.yaml, and emits a
maintainer review report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import requests
import yaml


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ARXIV_API_URL = "https://export.arxiv.org/api/query"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    GITHUB_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

URL_RE = re.compile(r"https?://[^\s<>)\]\}]+")
GITHUB_RE = re.compile(r"https?://github\.com/([^/\s]+/[^/\s#?]+)")
HF_RE = re.compile(r"https?://huggingface\.co/(datasets/)?([^/\s]+/[^/\s#?]+)")

PHYSICAL_AI_KEYWORDS = {
    "robot", "robotic", "robotics", "embodied", "embodiment", "manipulation",
    "manipulator", "humanoid", "quadruped", "locomotion", "dexterous",
    "teleoperation", "imitation learning", "vision language action", "vla",
    "policy learning", "sim to real", "reinforcement learning", "grasp",
    "world model", "diffusion policy", "mobile manipulation", "robot learning",
}
EXCLUSION_KEYWORDS = {
    "autonomous driving", "self driving", "traffic", "lane detection",
    "driver assistance", "adas", "vehicle trajectory", "driving dataset",
}
PLACEHOLDER_PATTERNS = {
    "coming soon", "code coming", "code will be released", "to be released",
    "release soon", "under construction", "placeholder", "todo",
}
UNOFFICIAL_PATTERNS = {
    "unofficial", "reimplementation", "re implementation", "replica",
    "fine tuned", "finetuned", "converted", "community",
}


@dataclass
class LinkCheck:
    url: str
    kind: str
    status: str
    reason: str = ""
    official_evidence: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    source: str
    kind: str
    title: str
    url: str
    published: str
    summary: str
    authors: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    checks: list[LinkCheck] = field(default_factory=list)
    duplicate_matches: list[str] = field(default_factory=list)
    llm_review: dict[str, Any] = field(default_factory=dict)
    relevance: str = "unknown"
    recommendation: str = "needs_review"
    reasons: list[str] = field(default_factory=list)


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def slugify(value: str) -> str:
    value = normalize_text(value).replace(" ", "-")
    return re.sub(r"-+", "-", value).strip("-")


def extract_urls(text: str) -> list[str]:
    urls = []
    for match in URL_RE.findall(text or ""):
        url = match.rstrip(".,;:")
        if url not in urls:
            urls.append(url)
    return urls


def classify_url(url: str) -> str:
    if "github.com/" in url:
        return "github"
    if "huggingface.co/datasets/" in url:
        return "hf_dataset"
    if "huggingface.co/" in url:
        return "hf_model"
    if "arxiv.org/" in url:
        return "paper"
    return "project"


def load_yaml_entries() -> list[dict]:
    entries: list[dict] = []
    for path in (DATA_DIR / "models.yaml", DATA_DIR / "datasets.yaml", DATA_DIR / "tools.yaml"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        for entry in data:
            entry["_file"] = path.name
            entries.append(entry)
    return entries


def find_duplicate_matches(candidate: Candidate, existing_entries: list[dict]) -> list[str]:
    matches: list[str] = []
    candidate_names = {normalize_text(candidate.title), slugify(candidate.title)}
    candidate_urls = set(candidate.links + [candidate.url])

    for entry in existing_entries:
        entry_urls = {
            entry.get("github_url", ""),
            entry.get("paper_url", ""),
            entry.get("hf_url", ""),
            entry.get("project_url", ""),
        }
        if candidate_urls & {u for u in entry_urls if u}:
            matches.append(f"{entry['_file']}:{entry.get('id')} URL match")
            continue

        entry_names = {normalize_text(entry.get("name", "")), slugify(entry.get("name", ""))}
        if candidate_names & entry_names:
            matches.append(f"{entry['_file']}:{entry.get('id')} name match")

    return matches


def assess_relevance(title: str, summary: str) -> tuple[str, list[str]]:
    text = normalize_text(f"{title} {summary}")
    reasons: list[str] = []

    exclusion_hits = [kw for kw in EXCLUSION_KEYWORDS if kw in text]
    if exclusion_hits:
        reasons.append(f"exclusion keywords: {', '.join(sorted(exclusion_hits))}")
        return "low", reasons

    hits = [kw for kw in PHYSICAL_AI_KEYWORDS if kw in text]
    if len(hits) >= 2:
        reasons.append(f"physical-ai keywords: {', '.join(sorted(hits)[:8])}")
        return "high", reasons
    if hits:
        reasons.append(f"physical-ai keyword: {hits[0]}")
        return "medium", reasons

    reasons.append("no strong Physical AI keyword evidence")
    return "low", reasons


def inspect_github_readme(slug: str) -> tuple[str, str, list[str]]:
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{slug}/readme",
            headers=GITHUB_HEADERS,
            timeout=15,
        )
        if resp.status_code == 404:
            return "placeholder", "README not found", []
        if resp.status_code in (401, 403):
            return "unknown", f"README unavailable ({resp.status_code})", []
        resp.raise_for_status()
        readme = resp.json()
        download_url = readme.get("download_url")
        if not download_url:
            return "unknown", "README download URL missing", []

        raw = requests.get(download_url, timeout=15)
        raw.raise_for_status()
        text = normalize_text(raw.text[:20000])
    except requests.RequestException as exc:
        return "unknown", f"README check failed: {exc}", []

    if any(pattern in text for pattern in PLACEHOLDER_PATTERNS):
        return "placeholder", "README suggests code is not yet released", []
    if any(pattern in text for pattern in UNOFFICIAL_PATTERNS):
        return "unofficial", "README suggests unofficial or derived release", []

    evidence = []
    if "arxiv" in text:
        evidence.append("README links back to arXiv")
    if "paper" in text:
        evidence.append("README mentions paper")
    return "available", "README present", evidence


def verify_github_url(url: str) -> LinkCheck:
    match = GITHUB_RE.match(url)
    if not match:
        return LinkCheck(url=url, kind="github", status="invalid", reason="not a GitHub repository URL")

    slug = match.group(1).removesuffix(".git")
    try:
        resp = requests.get(f"https://api.github.com/repos/{slug}", headers=GITHUB_HEADERS, timeout=15)
        if resp.status_code == 404:
            return LinkCheck(url=url, kind="github", status="not_found", reason="GitHub repository returned 404")
        if resp.status_code in (401, 403):
            return LinkCheck(url=url, kind="github", status="private_or_rate_limited", reason=f"GitHub API returned {resp.status_code}")
        resp.raise_for_status()
        repo = resp.json()
    except requests.RequestException as exc:
        return LinkCheck(url=url, kind="github", status="unknown", reason=f"GitHub API error: {exc}")

    if repo.get("archived"):
        status = "archived"
    elif repo.get("disabled"):
        status = "not_available"
    elif repo.get("private"):
        status = "private_or_gated"
    elif repo.get("size", 0) <= 1:
        status = "placeholder"
    else:
        status = "available"

    evidence = []
    description = normalize_text(repo.get("description") or "")
    if any(pattern in description for pattern in UNOFFICIAL_PATTERNS):
        status = "unofficial"
        evidence.append("repository description suggests unofficial/community release")

    readme_status, readme_reason, readme_evidence = inspect_github_readme(slug)
    evidence.extend(readme_evidence)
    if readme_status in {"placeholder", "unofficial"}:
        status = readme_status

    reason_parts = [
        f"stars={repo.get('stargazers_count', 0)}",
        f"size={repo.get('size', 0)}KB",
        f"pushed_at={repo.get('pushed_at', 'unknown')}",
    ]
    if readme_reason:
        reason_parts.append(readme_reason)

    return LinkCheck(
        url=url,
        kind="github",
        status=status,
        reason="; ".join(reason_parts),
        official_evidence=evidence,
    )


def verify_hf_url(url: str) -> LinkCheck:
    match = HF_RE.match(url)
    if not match:
        return LinkCheck(url=url, kind="huggingface", status="invalid", reason="not a HuggingFace model/dataset URL")

    kind = "dataset" if match.group(1) else "model"
    slug = match.group(2)
    api_kind = "datasets" if kind == "dataset" else "models"

    try:
        resp = requests.get(f"https://huggingface.co/api/{api_kind}/{slug}", timeout=15)
        if resp.status_code == 404:
            return LinkCheck(url=url, kind=f"hf_{kind}", status="not_found", reason="HuggingFace API returned 404")
        if resp.status_code in (401, 403):
            return LinkCheck(url=url, kind=f"hf_{kind}", status="private_or_gated", reason=f"HuggingFace API returned {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return LinkCheck(url=url, kind=f"hf_{kind}", status="unknown", reason=f"HuggingFace API error: {exc}")

    card_text = normalize_text(str(data.get("cardData") or "") + " " + str(data.get("description") or ""))
    siblings = data.get("siblings") or []
    files = [s.get("rfilename", "") for s in siblings if isinstance(s, dict)]
    gated = data.get("gated") not in (False, None, "false")

    status = "private_or_gated" if gated else "available"
    if len(files) <= 1:
        status = "placeholder"
    if any(pattern in card_text for pattern in UNOFFICIAL_PATTERNS):
        status = "unofficial"
    if any(pattern in card_text for pattern in PLACEHOLDER_PATTERNS):
        status = "placeholder"

    evidence = []
    if "arxiv" in card_text:
        evidence.append("model card links back to arXiv")
    if "paper" in card_text:
        evidence.append("model card mentions paper")

    downloads = data.get("downloads", 0) or data.get("downloadsAllTime", 0)
    reason = f"files={len(files)}; downloads={downloads}"
    if gated:
        reason += "; gated=true"

    return LinkCheck(url=url, kind=f"hf_{kind}", status=status, reason=reason, official_evidence=evidence)


def verify_generic_url(url: str) -> LinkCheck:
    try:
        resp = requests.get(url, timeout=15, allow_redirects=True)
        if resp.status_code == 404:
            return LinkCheck(url=url, kind=classify_url(url), status="not_found", reason="HTTP 404")
        if resp.status_code in (401, 403):
            return LinkCheck(url=url, kind=classify_url(url), status="private_or_gated", reason=f"HTTP {resp.status_code}")
        resp.raise_for_status()
    except requests.RequestException as exc:
        return LinkCheck(url=url, kind=classify_url(url), status="unknown", reason=f"HTTP error: {exc}")

    text = normalize_text(resp.text[:20000])
    status = "available"
    if any(pattern in text for pattern in PLACEHOLDER_PATTERNS):
        status = "placeholder"
    if any(pattern in text for pattern in UNOFFICIAL_PATTERNS):
        status = "unofficial"

    evidence = []
    if "github com" in text:
        evidence.append("page links to GitHub")
    if "huggingface co" in text:
        evidence.append("page links to HuggingFace")

    return LinkCheck(url=url, kind=classify_url(url), status=status, reason=f"HTTP {resp.status_code}", official_evidence=evidence)


def verify_link(url: str) -> LinkCheck:
    kind = classify_url(url)
    if kind == "github":
        return verify_github_url(url)
    if kind in {"hf_model", "hf_dataset"}:
        return verify_hf_url(url)
    if kind == "paper":
        return LinkCheck(url=url, kind=kind, status="available", reason="paper link")
    return verify_generic_url(url)


def decide_recommendation(candidate: Candidate) -> tuple[str, list[str]]:
    reasons: list[str] = []
    available_links = [c for c in candidate.checks if c.status == "available" and c.kind != "paper"]
    bad_links = [
        c for c in candidate.checks
        if c.status in {
            "placeholder",
            "not_found",
            "unofficial",
            "private_or_gated",
            "archived",
            "not_available",
        }
    ]

    if candidate.duplicate_matches:
        reasons.append("possible duplicate with existing data")
        return "reject", reasons
    if candidate.relevance == "low":
        reasons.append("low Physical AI relevance")
        return "reject", reasons
    if bad_links and not available_links:
        reasons.append("only non-available or unofficial public links found")
        return "needs_review", reasons
    if not available_links:
        reasons.append("paper found, but no verified public code/model/dataset link")
        return "needs_review", reasons
    if candidate.relevance == "high" and available_links:
        reasons.append("verified public link and high Physical AI relevance")
        return "needs_review", reasons

    reasons.append("requires maintainer review")
    return "needs_review", reasons


def candidate_review_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "title": candidate.title,
        "summary": candidate.summary,
        "authors": candidate.authors,
        "paper_url": candidate.url,
        "links": candidate.links,
        "duplicate_matches": candidate.duplicate_matches,
        "link_checks": [asdict(check) for check in candidate.checks],
        "question": (
            "Decide whether this is an official, open Physical AI / robotics "
            "model, dataset, or simulator candidate. Exclude autonomous-driving-only, "
            "unofficial reimplementations, fine-tunes, and paper-only entries."
        ),
        "expected_json": {
            "is_physical_ai": "boolean",
            "is_official": "boolean",
            "entry_type": "model|dataset|tool|paper_only|irrelevant|unclear",
            "decision": "accept|needs_review|reject",
            "reason": "short explanation",
        },
    }


def run_llm_review_command(candidate: Candidate, command: str) -> dict[str, Any]:
    """Run an external LLM reviewer command.

    The command receives candidate JSON on stdin and should return one JSON
    object on stdout. This keeps the script provider-agnostic and avoids
    hard-coding API keys or model choices into repository code.
    """
    payload = json.dumps(candidate_review_payload(candidate), ensure_ascii=False)
    try:
        result = subprocess.run(
            shlex.split(command),
            input=payload,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "reason": str(exc)}

    if result.returncode != 0:
        return {
            "status": "error",
            "reason": result.stderr.strip() or f"command exited with {result.returncode}",
        }

    try:
        review = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "error",
            "reason": f"invalid JSON from LLM command: {exc}",
            "raw": result.stdout.strip()[:1000],
        }

    if not isinstance(review, dict):
        return {"status": "error", "reason": "LLM command returned non-object JSON"}
    review.setdefault("status", "ok")
    return review


def fetch_arxiv_cs_ro(days: int, max_results: int) -> list[Candidate]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    params = {
        "search_query": "cat:cs.RO",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    candidates: list[Candidate] = []

    for entry in root.findall("atom:entry", ns):
        title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
        summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split())
        url = entry.findtext("atom:id", default="", namespaces=ns)
        published = entry.findtext("atom:published", default="", namespaces=ns)
        authors = [
            a.findtext("atom:name", default="", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]

        published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if published_dt < cutoff:
            continue

        links = [url]
        for link in entry.findall("atom:link", ns):
            href = link.attrib.get("href", "")
            if href and href not in links:
                links.append(href)
        for extracted in extract_urls(summary):
            if extracted not in links:
                links.append(extracted)

        candidates.append(Candidate(
            source="arxiv",
            kind="paper",
            title=title,
            url=url,
            published=published_dt.date().isoformat(),
            summary=summary[:800],
            authors=[a for a in authors if a],
            links=links,
        ))

    return candidates


def evaluate_candidates(
    candidates: list[Candidate],
    verify_links: bool = True,
    llm_review_command: str | None = None,
) -> list[Candidate]:
    existing_entries = load_yaml_entries()
    for candidate in candidates:
        candidate.relevance, relevance_reasons = assess_relevance(candidate.title, candidate.summary)
        candidate.reasons.extend(relevance_reasons)
        candidate.duplicate_matches = find_duplicate_matches(candidate, existing_entries)

        official_candidate_links = [
            url for url in candidate.links
            if classify_url(url) in {"github", "hf_model", "hf_dataset", "project"}
        ]
        if verify_links:
            candidate.checks = [verify_link(url) for url in official_candidate_links]
        else:
            candidate.checks = [
                LinkCheck(url=url, kind=classify_url(url), status="not_checked", reason="--no-verify")
                for url in official_candidate_links
            ]

        recommendation, decision_reasons = decide_recommendation(candidate)
        candidate.recommendation = recommendation
        candidate.reasons.extend(decision_reasons)

        if llm_review_command:
            candidate.llm_review = run_llm_review_command(candidate, llm_review_command)

    return candidates


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return asdict(candidate)


def render_markdown(candidates: list[Candidate]) -> str:
    lines = [
        "# Physical AI Discovery Report",
        "",
        "This report is for maintainer review. No GitHub issues were created.",
        "",
        "| Recommendation | Count |",
        "|---|---:|",
    ]
    for recommendation in ("needs_review", "reject"):
        count = sum(1 for c in candidates if c.recommendation == recommendation)
        lines.append(f"| `{recommendation}` | {count} |")

    lines.extend(["", "## Candidates", ""])
    for i, candidate in enumerate(candidates, start=1):
        lines.extend([
            f"### {i}. {candidate.title}",
            "",
            f"- Source: `{candidate.source}`",
            f"- Published: `{candidate.published}`",
            f"- Paper: {candidate.url}",
            f"- Relevance: `{candidate.relevance}`",
            f"- Recommendation: `{candidate.recommendation}`",
        ])
        if candidate.authors:
            lines.append(f"- Authors: {', '.join(candidate.authors[:8])}")
        if candidate.duplicate_matches:
            lines.append(f"- Duplicate matches: {', '.join(candidate.duplicate_matches)}")
        if candidate.reasons:
            lines.append(f"- Reasons: {'; '.join(candidate.reasons)}")
        if candidate.llm_review:
            lines.append(f"- LLM review: `{candidate.llm_review.get('status', 'unknown')}`")
            if candidate.llm_review.get("decision"):
                lines.append(f"- LLM decision: `{candidate.llm_review.get('decision')}`")
            if candidate.llm_review.get("reason"):
                lines.append(f"- LLM reason: {candidate.llm_review.get('reason')}")

        lines.extend(["", "**Verified links**", ""])
        if candidate.checks:
            for check in candidate.checks:
                evidence = f" Evidence: {', '.join(check.official_evidence)}." if check.official_evidence else ""
                lines.append(f"- `{check.status}` `{check.kind}`: {check.url} ({check.reason}).{evidence}")
        else:
            lines.append("- No official code/model/dataset/project links found in arXiv metadata.")

        lines.extend(["", "**Summary**", "", candidate.summary, ""])

    return "\n".join(lines).rstrip() + "\n"


def write_output(candidates: list[Candidate], output_format: str, output_path: str | None) -> None:
    if output_format == "json":
        content = json.dumps([candidate_to_dict(c) for c in candidates], ensure_ascii=False, indent=2)
    elif output_format == "jsonl":
        content = "\n".join(json.dumps(candidate_to_dict(c), ensure_ascii=False) for c in candidates) + "\n"
    else:
        content = render_markdown(candidates)

    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")
    else:
        print(content, end="")


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover recent Physical AI candidates from arXiv cs.RO.")
    parser.add_argument("--days", type=int, default=7, help="Look back this many days.")
    parser.add_argument("--max-arxiv", type=int, default=20, help="Maximum arXiv papers to fetch.")
    parser.add_argument("--format", choices=("markdown", "json", "jsonl"), default="markdown")
    parser.add_argument("--output", help="Write report to this file instead of stdout.")
    parser.add_argument("--no-verify", action="store_true", help="Skip network checks for extracted official links.")
    parser.add_argument(
        "--llm-review-command",
        help="Optional command that receives candidate JSON on stdin and returns one review JSON object.",
    )
    args = parser.parse_args()

    try:
        candidates = fetch_arxiv_cs_ro(args.days, args.max_arxiv)
        candidates = evaluate_candidates(
            candidates,
            verify_links=not args.no_verify,
            llm_review_command=args.llm_review_command,
        )
    except requests.RequestException as exc:
        print(f"error: discovery request failed: {exc}", file=sys.stderr)
        return 1

    write_output(candidates, args.format, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
