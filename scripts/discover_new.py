"""
Discover recent Physical AI candidates from paper-first sources.

This script intentionally does not create GitHub issues. It collects recent
arXiv cs.RO papers, extracts official-looking links from paper metadata,
verifies those links, checks for duplicates against data/*.yaml, and emits a
maintainer review report.

Hybrid review strategy:
1. Keep deterministic keyword/rule-based filtering as the default layer.
2. Cap borderline candidates with --max-ambiguous for human or optional LLM review.
3. Report LLM review and public-facing entry summaries side-by-side with rule-based
   results instead of silently replacing deterministic decisions.
4. Ask the LLM for both a public-facing entry_summary and a maintainer_summary
   for candidates that will be reviewed in weekly GitHub issues.
5. Track verified model/code/dataset/artifact links separately so that paper-only
   or project-page-only candidates are not confused with official model releases.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
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
HF_MODEL_RE = re.compile(r"https?://huggingface\.co/([^/\s]+/[^/\s#?]+)")
HF_DATASET_RE = re.compile(r"https?://huggingface\.co/datasets/([^/\s]+/[^/\s#?]+)")
ARXIV_ID_RE = re.compile(r"^/abs/([^/?#]+)")
DEFAULT_USER_AGENT = "Awesome-Physical-AI discover_new.py/1.0"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
HTTP_SESSION = requests.Session()


def http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
    allow_redirects: bool = True,
    retries: int = 2,
) -> requests.Response:
    request_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        request_headers.update(headers)

    last_response: requests.Response | None = None
    for attempt in range(retries + 1):
        response = HTTP_SESSION.get(
            url,
            params=params,
            headers=request_headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )
        last_response = response
        if response.status_code not in RETRY_STATUS_CODES or attempt == retries:
            return response

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = min(int(retry_after), 30)
        else:
            delay = min(2 ** attempt, 8)
        time.sleep(delay)

    return last_response  # type: ignore[return-value]

PHYSICAL_AI_KEYWORDS = {
    "robot", "robotic", "robotics", "embodied", "embodiment", "manipulation",
    "manipulator", "humanoid", "quadruped", "locomotion", "dexterous",
    "teleoperation", "imitation learning", "vision language action", "vla",
    "policy learning", "sim to real", "reinforcement learning", "grasp",
    "world model", "diffusion policy", "mobile manipulation", "robot learning",
    "navigation", "vision language navigation", "vln", "embodied navigation",
    "mobile robot", "motion planning", "path planning", "task planning",
    "action prediction", "action generation", "control policy", "robot policy",
    "simulator", "simulation", "real world interaction", "physical interaction",
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
BAD_LINK_STATUSES = {
    "placeholder", "not_found", "unofficial", "private_or_gated",
    "archived", "not_available",
}
AMBIGUOUS_LINK_STATUSES = {"unknown", "private_or_rate_limited"}


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
    llm_review_selected: bool = False
    relevance: str = "unknown"
    recommendation: str = "needs_review"
    review_bucket: str = "normal"  # normal | ambiguous | reject
    keyword_hits: list[str] = field(default_factory=list)
    exclusion_hits: list[str] = field(default_factory=list)
    artifact_availability: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def slugify(value: str) -> str:
    value = normalize_text(value).replace(" ", "-")
    return re.sub(r"-+", "-", value).strip("-")


def parsed_hostname(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def _canonical_arxiv_path(path: str) -> str:
    match = ARXIV_ID_RE.match(path)
    if not match:
        return path.rstrip("/")
    arxiv_id = re.sub(r"v\d+$", "", match.group(1))
    return f"/abs/{arxiv_id}"


def canonicalize_url(url: str) -> str:
    """Return a stable URL representation for classification and duplicate checks."""
    url = (url or "").strip().rstrip(".,;:")
    if not url:
        return ""

    try:
        parts = urlsplit(url)
    except ValueError:
        return url

    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if not host:
        return url

    path = re.sub(r"/+", "/", parts.path or "/")
    query = parts.query

    if host in {"github.com", "www.github.com"}:
        host = "github.com"
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 2:
            repo = segments[1].removesuffix(".git")
            path = f"/{segments[0]}/{repo}"
        else:
            path = "/" + "/".join(segments)
        query = ""
    elif host in {"huggingface.co", "www.huggingface.co"}:
        host = "huggingface.co"
        segments = [segment for segment in path.split("/") if segment]
        if segments[:1] in (["datasets"], ["spaces"]) and len(segments) >= 3:
            path = f"/{segments[0]}/{segments[1]}/{segments[2]}"
        elif len(segments) >= 2:
            path = f"/{segments[0]}/{segments[1]}"
        else:
            path = "/" + "/".join(segments)
        query = ""
    elif host in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        host = "arxiv.org"
        path = _canonical_arxiv_path(path)
        query = ""
    else:
        path = path.rstrip("/") or "/"
        query = urlencode(sorted(parse_qsl(query, keep_blank_values=True)))

    return urlunsplit((scheme, host, path, query, ""))


def extract_urls(text: str) -> list[str]:
    urls = []
    for match in URL_RE.findall(text or ""):
        url = canonicalize_url(match)
        if url not in urls:
            urls.append(url)
    return urls


def classify_url(url: str) -> str:
    url = canonicalize_url(url)
    host = parsed_hostname(url)
    path = urlsplit(url).path if host else ""

    if host == "github.com":
        return "github"
    if host == "huggingface.co" and path.startswith("/datasets/"):
        return "hf_dataset"
    if host == "huggingface.co" and path.startswith("/spaces/"):
        return "hf_space"
    if host == "huggingface.co" and path.startswith("/papers/"):
        return "paper"
    if host == "huggingface.co":
        return "hf_model"
    if host == "arxiv.org":
        return "paper"
    if host == "doi.org":
        return "publication"
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
    candidate_urls = {canonicalize_url(url) for url in candidate.links + [candidate.url] if canonicalize_url(url)}

    for entry in existing_entries:
        entry_urls = {
            entry.get("github_url", ""),
            entry.get("paper_url", ""),
            entry.get("hf_url", ""),
            entry.get("project_url", ""),
        }
        if candidate_urls & {canonicalize_url(u) for u in entry_urls if canonicalize_url(u)}:
            matches.append(f"{entry['_file']}:{entry.get('id')} URL match")
            continue

        entry_names = {normalize_text(entry.get("name", "")), slugify(entry.get("name", ""))}
        if candidate_names & entry_names:
            matches.append(f"{entry['_file']}:{entry.get('id')} name match")

    return matches


def assess_relevance_details(title: str, summary: str) -> tuple[str, list[str], list[str], list[str]]:
    text = normalize_text(f"{title} {summary}")
    reasons: list[str] = []

    exclusion_hits = sorted(kw for kw in EXCLUSION_KEYWORDS if kw in text)
    if exclusion_hits:
        reasons.append(f"exclusion keywords: {', '.join(exclusion_hits)}")
        return "low", reasons, [], exclusion_hits

    hits = sorted(kw for kw in PHYSICAL_AI_KEYWORDS if kw in text)
    if len(hits) >= 2:
        reasons.append(f"physical-ai keywords: {', '.join(hits[:8])}")
        return "high", reasons, hits, []
    if hits:
        reasons.append(f"physical-ai keyword: {hits[0]}")
        return "medium", reasons, hits, []

    reasons.append("no strong Physical AI keyword evidence")
    return "low", reasons, [], []

def assess_relevance(title: str, summary: str) -> tuple[str, list[str]]:
    relevance, reasons, _keyword_hits, _exclusion_hits = assess_relevance_details(title, summary)
    return relevance, reasons


def inspect_github_readme(slug: str) -> tuple[str, str, list[str]]:
    try:
        resp = http_get(
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

        raw = http_get(download_url, timeout=15)
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
    url = canonicalize_url(url)
    match = GITHUB_RE.match(url)
    if not match:
        return LinkCheck(url=url, kind="github", status="invalid", reason="not a GitHub repository URL")

    slug = match.group(1).removesuffix(".git")
    try:
        resp = http_get(f"https://api.github.com/repos/{slug}", headers=GITHUB_HEADERS, timeout=15)
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
    url = canonicalize_url(url)
    if classify_url(url) == "hf_dataset":
        match = HF_DATASET_RE.match(url)
        kind = "dataset"
        api_kind = "datasets"
    else:
        match = HF_MODEL_RE.match(url)
        kind = "model"
        api_kind = "models"

    if not match:
        return LinkCheck(url=url, kind="huggingface", status="invalid", reason="not a HuggingFace model/dataset URL")

    slug = match.group(1)
    try:
        resp = http_get(f"https://huggingface.co/api/{api_kind}/{slug}", timeout=15)
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
    if not gated and len(files) <= 1:
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
    url = canonicalize_url(url)
    try:
        resp = http_get(url, timeout=15, allow_redirects=True)
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



def available_checks(candidate: Candidate, kinds: set[str] | None = None) -> list[LinkCheck]:
    checks = [c for c in candidate.checks if c.status == "available"]
    if kinds is not None:
        checks = [c for c in checks if c.kind in kinds]
    return checks


def build_artifact_availability(candidate: Candidate) -> dict[str, Any]:
    """Summarize verified artifact availability for rule and LLM review.

    A generic project page is tracked separately from concrete artifacts. This
    prevents a project page from being treated as an official model release.
    """
    verified_model_links = available_checks(candidate, {"hf_model"})
    verified_dataset_links = available_checks(candidate, {"hf_dataset"})
    verified_code_links = available_checks(candidate, {"github"})
    verified_space_links = available_checks(candidate, {"hf_space"})
    verified_project_pages = available_checks(candidate, {"project"})

    verified_artifact_links = (
        verified_model_links
        + verified_dataset_links
        + verified_code_links
        + verified_space_links
    )

    return {
        "has_verified_model_link": bool(verified_model_links),
        "has_verified_dataset_link": bool(verified_dataset_links),
        "has_verified_code_link": bool(verified_code_links),
        "has_verified_space_link": bool(verified_space_links),
        "has_verified_artifact_link": bool(verified_artifact_links),
        "has_verified_project_page": bool(verified_project_pages),
        "verified_model_links": [c.url for c in verified_model_links],
        "verified_dataset_links": [c.url for c in verified_dataset_links],
        "verified_code_links": [c.url for c in verified_code_links],
        "verified_space_links": [c.url for c in verified_space_links],
        "verified_project_pages": [c.url for c in verified_project_pages],
    }

def decide_recommendation_details(candidate: Candidate) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    availability = candidate.artifact_availability or build_artifact_availability(candidate)
    available_links = [
        c for c in candidate.checks
        if c.status == "available" and c.kind not in {"paper", "publication"}
    ]
    bad_links = [c for c in candidate.checks if c.status in BAD_LINK_STATUSES]

    has_verified_model = availability.get("has_verified_model_link", False)
    has_verified_artifact = availability.get("has_verified_artifact_link", False)
    has_verified_project_page = availability.get("has_verified_project_page", False)

    if candidate.duplicate_matches:
        reasons.append("possible duplicate with existing data")
        return "reject", reasons, "reject"

    if candidate.exclusion_hits:
        reasons.append("excluded by domain-specific exclusion keywords")
        return "reject", reasons, "reject"

    if bad_links and not available_links:
        reasons.append("only unavailable, archived, placeholder, gated, or unofficial public links found")
        return "needs_review", reasons, "ambiguous"

    if not available_links:
        reasons.append("paper found, but no verified public code/model/dataset/project link")
        return "needs_review", reasons, "ambiguous"

    if not has_verified_artifact:
        if has_verified_project_page:
            reasons.append("verified project page found, but no verified concrete artifact link")
        else:
            reasons.append("no verified model/code/dataset/space artifact link")
        return "needs_review", reasons, "ambiguous"

    if not has_verified_model:
        reasons.append("verified artifact link found, but no verified model link")
        return "needs_review", reasons, "ambiguous"

    if candidate.relevance == "low":
        reasons.append("low keyword-based Physical AI relevance despite verified model link; kept for maintainer review")
        return "needs_review", reasons, "ambiguous"

    if candidate.relevance == "medium":
        reasons.append("verified model link with medium keyword-based Physical AI relevance; selected for possible human/LLM review")
        return "needs_review", reasons, "ambiguous"

    if any(c.status in AMBIGUOUS_LINK_STATUSES for c in candidate.checks):
        reasons.append("some link checks are inconclusive")
        return "needs_review", reasons, "ambiguous"

    if candidate.relevance == "high" and has_verified_model:
        reasons.append("verified model link and high Physical AI relevance")
        return "needs_review", reasons, "normal"

    raise ValueError(f"Unexpected relevance value: {candidate.relevance}")

def decide_recommendation(candidate: Candidate) -> tuple[str, list[str]]:
    recommendation, reasons, _review_bucket = decide_recommendation_details(candidate)
    return recommendation, reasons

def candidate_review_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "question": (
            "Review this candidate for Awesome-Physical-AI. Decide whether it is an official, open "
            "Physical AI / robotics / embodied AI model, dataset, benchmark, simulator, or tool. "
            "Exclude autonomous-driving-only entries, unofficial reimplementations, fine-tunes, "
            "closed/private artifacts, and paper-only entries unless they clearly provide a useful "
            "official artifact."
        ),
        "title": candidate.title,
        "abstract": candidate.summary,
        "authors": candidate.authors,
        "paper_url": candidate.url,
        "links": candidate.links,
        "duplicate_matches": candidate.duplicate_matches,
        "artifact_availability": candidate.artifact_availability,
        "keyword_based_review": {
            "relevance": candidate.relevance,
            "keyword_hits": candidate.keyword_hits,
            "exclusion_hits": candidate.exclusion_hits,
            "recommendation": candidate.recommendation,
            "review_bucket": candidate.review_bucket,
            "reasons": candidate.reasons,
        },
        "link_checks": [asdict(check) for check in candidate.checks],
        "review_policy": {
            "llm_direct_review": True,
            "needs_human_maintainer_summary": True,
            "intended_use": "weekly_github_issue_for_human_maintainer_review",
        },
        "task": (
            "Review this candidate for Awesome-Physical-AI. Decide whether it is an official, open "
            "Physical AI / robotics / embodied AI model, dataset, benchmark, simulator, or tool. "
            "Exclude autonomous-driving-only entries, unofficial reimplementations, fine-tunes, "
            "closed/private artifacts, and paper-only entries unless they clearly provide a useful "
            "official artifact. Treat a verified official model link as the strongest positive signal. "
            "Do not treat a generic project page as a verified model release unless the input explicitly "
            "contains a verified model/code/dataset/simulator/tool artifact link. Also generate a "
            "public-facing entry summary suitable for an Awesome-list item."
        ),
        "expected_json": {
            "has_verified_model_link": "boolean",
            "has_verified_artifact_link": "boolean",
            "entry_type": "model|dataset|tool|benchmark|simulator|paper_only|irrelevant|unclear",
            "decision": "accept|needs_review|reject",
            "entry_summary": "2-3 sentence public-facing Awesome-list description",
            "maintainer_summary": "2-3 sentence human maintainer review note for weekly GitHub issue triage",
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
    cmd = shlex.split(command)
    if cmd and cmd[0] in {"python", "python3"}:
        cmd[0] = sys.executable

    try:
        result = subprocess.run(
            cmd,
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


def arxiv_submitted_date_range(days: int) -> str:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return f"[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"


def fetch_arxiv_cs_ro(days: int, max_results: int) -> list[Candidate]:
    params = {
        "search_query": f"cat:cs.RO AND submittedDate:{arxiv_submitted_date_range(days)}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = http_get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    candidates: list[Candidate] = []

    for entry in root.findall("atom:entry", ns):
        title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
        summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split())
        url = canonicalize_url(entry.findtext("atom:id", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns)
        authors = [
            a.findtext("atom:name", default="", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]

        published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))

        links = [url]
        for link in entry.findall("atom:link", ns):
            href = canonicalize_url(link.attrib.get("href", ""))
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
            summary=summary[:1200],
            authors=[a for a in authors if a],
            links=links,
        ))

    return candidates


def llm_review_targets(
    candidates: list[Candidate],
    llm_review_mode: str,
    max_ambiguous: int,
) -> list[Candidate]:
    if llm_review_mode == "off":
        return []
    if llm_review_mode == "all":
        return [c for c in candidates if c.review_bucket != "reject"]

    ambiguous = [c for c in candidates if c.review_bucket == "ambiguous"]
    return ambiguous[:max_ambiguous]


def evaluate_candidates(
    candidates: list[Candidate],
    verify_links: bool = True,
    llm_review_command: str | None = None,
    llm_review_mode: str = "ambiguous",
    max_ambiguous: int = 10,
) -> list[Candidate]:
    existing_entries = load_yaml_entries()
    for candidate in candidates:
        (
            candidate.relevance,
            relevance_reasons,
            candidate.keyword_hits,
            candidate.exclusion_hits,
        ) = assess_relevance_details(candidate.title, candidate.summary)
        candidate.reasons.extend(relevance_reasons)
        candidate.duplicate_matches = find_duplicate_matches(candidate, existing_entries)

        official_candidate_links = [
            url for url in candidate.links
            if classify_url(url) in {"github", "hf_model", "hf_dataset", "hf_space", "project", "publication"}
        ]
        if verify_links:
            candidate.checks = [verify_link(url) for url in official_candidate_links]
        else:
            candidate.checks = [
                LinkCheck(url=url, kind=classify_url(url), status="not_checked", reason="--no-verify")
                for url in official_candidate_links
            ]

        candidate.artifact_availability = build_artifact_availability(candidate)
        recommendation, decision_reasons, review_bucket = decide_recommendation_details(candidate)
        candidate.recommendation = recommendation
        candidate.review_bucket = review_bucket
        candidate.reasons.extend(decision_reasons)

    targets = llm_review_targets(candidates, llm_review_mode, max_ambiguous)
    target_ids = {id(c) for c in targets}
    for candidate in candidates:
        candidate.llm_review_selected = id(candidate) in target_ids
        if llm_review_command and candidate.llm_review_selected:
            candidate.llm_review = run_llm_review_command(candidate, llm_review_command)

    return candidates


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return asdict(candidate)


def llm_entry_summary(candidate: Candidate) -> str | None:
    if candidate.llm_review.get("status") != "ok":
        return None
    summary = candidate.llm_review.get("entry_summary") or candidate.llm_review.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None


def llm_maintainer_summary(candidate: Candidate) -> str | None:
    if candidate.llm_review.get("status") != "ok":
        return None
    summary = candidate.llm_review.get("maintainer_summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None


def render_markdown(candidates: list[Candidate]) -> str:
    lines = [
        "# Physical AI Discovery Report",
        "",
        "This report is for maintainer review. No GitHub issues were created.",
        "",
        (
            "Rule-based link checks are the source of truth for artifact availability and verified links. "
            "LLM link assessments are auxiliary annotations for maintainer review only and do not override "
            "rule-based results."
        ),
        "",
        "| Recommendation | Count |",
        "|---|---:|",
    ]
    for recommendation in ("needs_review", "reject"):
        count = sum(1 for c in candidates if c.recommendation == recommendation)
        lines.append(f"| `{recommendation}` | {count} |")

    lines.extend([
        "",
        "| Review bucket | Count |",
        "|---|---:|",
    ])
    for review_bucket in ("normal", "ambiguous", "reject"):
        count = sum(1 for c in candidates if c.review_bucket == review_bucket)
        lines.append(f"| `{review_bucket}` | {count} |")

    llm_target_count = sum(1 for c in candidates if c.llm_review_selected)
    llm_completed_count = sum(1 for c in candidates if c.llm_review.get("status") == "ok")
    lines.extend([
        "",
        "| LLM review | Count |",
        "|---|---:|",
        f"| `targeted for optional review` | {llm_target_count} |",
        f"| `completed` | {llm_completed_count} |",
    ])

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
            f"- Review bucket: `{candidate.review_bucket}`",
            f"- Targeted for optional LLM review: `{candidate.llm_review_selected}`",
        ])
        if candidate.authors:
            lines.append(f"- Authors: {', '.join(candidate.authors[:8])}")
        if candidate.keyword_hits:
            lines.append(f"- Keyword hits: {', '.join(candidate.keyword_hits[:12])}")
        if candidate.exclusion_hits:
            lines.append(f"- Exclusion hits: {', '.join(candidate.exclusion_hits)}")
        if candidate.duplicate_matches:
            lines.append(f"- Duplicate matches: {', '.join(candidate.duplicate_matches)}")
        if candidate.reasons:
            lines.append(f"- Rule-based reasons: {'; '.join(candidate.reasons)}")
        if candidate.artifact_availability:
            availability = candidate.artifact_availability
            lines.extend([
                f"- Rule-based verified model link: `{availability.get('has_verified_model_link', False)}`",
                f"- Rule-based verified artifact link: `{availability.get('has_verified_artifact_link', False)}`",
                f"- Rule-based verified project page: `{availability.get('has_verified_project_page', False)}`",
            ])
        if candidate.llm_review:
            lines.append(f"- LLM review status: `{candidate.llm_review.get('status', 'unknown')}`")
            if candidate.llm_review.get("decision"):
                lines.append(f"- LLM decision: `{candidate.llm_review.get('decision')}`")
            if "has_verified_model_link" in candidate.llm_review:
                lines.append(f"- LLM model-link annotation (reference only): `{candidate.llm_review.get('has_verified_model_link')}`")
            if "has_verified_artifact_link" in candidate.llm_review:
                lines.append(f"- LLM artifact-link annotation (reference only): `{candidate.llm_review.get('has_verified_artifact_link')}`")
            if candidate.llm_review.get("entry_type"):
                lines.append(f"- LLM entry type: `{candidate.llm_review.get('entry_type')}`")
            if candidate.llm_review.get("reason"):
                lines.append(f"- LLM reason: {candidate.llm_review.get('reason')}")

        lines.extend(["", "**Verified links**", ""])
        if candidate.checks:
            for check in candidate.checks:
                evidence = f" Evidence: {', '.join(check.official_evidence)}." if check.official_evidence else ""
                lines.append(f"- `{check.status}` `{check.kind}`: {check.url} ({check.reason}).{evidence}")
        else:
            lines.append("- No official code/model/dataset/project links found in arXiv metadata.")

        generated_summary = llm_entry_summary(candidate)
        if generated_summary:
            lines.extend(["", "**LLM entry summary**", "", generated_summary, ""])

        maintainer_summary = llm_maintainer_summary(candidate)
        if maintainer_summary:
            lines.extend(["", "**LLM maintainer summary**", "", maintainer_summary, ""])

        lines.extend(["", "**Original abstract excerpt**", "", candidate.summary, ""])

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
        "--max-ambiguous",
        type=int,
        default=10,
        help="Maximum ambiguous candidates to select for human/LLM review.",
    )
    parser.add_argument(
        "--llm-review-mode",
        choices=("off", "ambiguous", "all"),
        default="ambiguous",
        help=(
            "Choose which candidates are sent to --llm-review-command. "
            "'ambiguous' limits LLM review to --max-ambiguous borderline candidates."
        ),
    )
    parser.add_argument(
        "--llm-review-command",
        help=(
            "Optional command that receives candidate JSON on stdin and returns one JSON object. "
            "The returned JSON may include has_verified_model_link, has_verified_artifact_link, "
            "entry_type, decision, entry_summary, maintainer_summary, and reason."
        ),
    )
    args = parser.parse_args()

    try:
        candidates = fetch_arxiv_cs_ro(args.days, args.max_arxiv)
        candidates = evaluate_candidates(
            candidates,
            verify_links=not args.no_verify,
            llm_review_command=args.llm_review_command,
            llm_review_mode=args.llm_review_mode,
            max_ambiguous=args.max_ambiguous,
        )
    except requests.RequestException as exc:
        print(f"error: discovery request failed: {exc}", file=sys.stderr)
        return 1

    write_output(candidates, args.format, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
