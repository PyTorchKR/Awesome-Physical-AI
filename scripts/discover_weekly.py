"""Run the discovery crawler with a persistent seen-paper cache.

This is the operational entry point used by the weekly GitHub Actions workflow.
The discovery and evaluation rules remain in ``discover_new.py``; this module
filters completed papers, schedules limited rechecks, and persists that state.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

import discover_new as discover


SEEN_CACHE_VERSION = 2
RETRY_DELAYS_DAYS = (14, 30, 60, 120)
ARXIV_ABS_RE = re.compile(r"^/abs/([^/?#]+)")


class SeenCandidate(Protocol):
    source: str
    title: str
    url: str
    published: str
    updated: str
    arxiv_version: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def empty_seen_cache() -> dict[str, Any]:
    return {"version": SEEN_CACHE_VERSION, "seen": {}}


def canonicalize_seen_url(url: str) -> str:
    """Normalize paper URLs so arXiv versions are treated as one paper."""
    url = (url or "").strip().rstrip(".,;:")
    if not url:
        return ""

    try:
        parts = urlsplit(url)
    except ValueError:
        return url.rstrip("/")

    host = (parts.hostname or "").lower()
    match = ARXIV_ABS_RE.match(parts.path or "")
    if host in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} and match:
        arxiv_id = re.sub(r"v\d+$", "", match.group(1))
        return f"https://arxiv.org/abs/{arxiv_id}"

    return url.rstrip("/")


def candidate_seen_key(candidate: SeenCandidate) -> str:
    return canonicalize_seen_url(candidate.url)


def load_seen_cache(path: str | None) -> dict[str, Any]:
    if not path:
        return empty_seen_cache()

    cache_path = Path(path)
    if not cache_path.exists():
        return empty_seen_cache()

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_seen_cache()

    if not isinstance(data, dict):
        return empty_seen_cache()

    seen = data.get("seen", {})
    if isinstance(seen, list):
        seen = {canonicalize_seen_url(str(url)): {} for url in seen if canonicalize_seen_url(str(url))}
    if not isinstance(seen, dict):
        seen = {}

    normalized_seen: dict[str, Any] = {}
    for url, raw_meta in seen.items():
        key = canonicalize_seen_url(str(url))
        if not key:
            continue
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        # Version 1 treated every evaluated paper as permanently seen. Recheck
        # those entries once so earlier dry-runs cannot suppress submissions.
        if "status" not in meta:
            meta.update({"status": "retry", "reason": "legacy_cache", "next_check_at": ""})
        normalized_seen[key] = meta

    return {
        "version": SEEN_CACHE_VERSION,
        "seen": normalized_seen,
    }


def filter_seen_candidates(
    candidates: list[SeenCandidate],
    seen_cache: dict[str, Any],
    *,
    now: str | None = None,
) -> tuple[list[SeenCandidate], list[SeenCandidate]]:
    seen = seen_cache.get("seen", {})
    if not isinstance(seen, dict) or not seen:
        return candidates, []

    now = now or utc_now_iso()
    fresh: list[SeenCandidate] = []
    skipped: list[SeenCandidate] = []
    for candidate in candidates:
        meta = seen.get(candidate_seen_key(candidate))
        if not isinstance(meta, dict):
            fresh.append(candidate)
            continue

        status = meta.get("status")
        revision_recheck = status == "retry" or meta.get("reason") in {"paper_only", "retry_exhausted"}
        revision_changed = bool(
            revision_recheck
            and getattr(candidate, "updated", "")
            and meta.get("arxiv_updated")
            and getattr(candidate, "updated", "") != meta.get("arxiv_updated")
        )
        retry_due = status == "retry" and str(meta.get("next_check_at") or "") <= now
        if revision_changed or retry_due:
            fresh.append(candidate)
        else:
            skipped.append(candidate)
    return fresh, skipped


def due_retry_keys(
    seen_cache: dict[str, Any],
    *,
    now: str | None = None,
    limit: int = 50,
) -> list[str]:
    """Return cached arXiv papers whose release status should be checked again."""
    now = now or utc_now_iso()
    seen = seen_cache.get("seen", {})
    if not isinstance(seen, dict):
        return []

    due = [
        (str(meta.get("next_check_at") or ""), key)
        for key, meta in seen.items()
        if isinstance(meta, dict)
        and meta.get("status") == "retry"
        and str(meta.get("next_check_at") or "") <= now
    ]
    due.sort()
    keys = [key for _next_check, key in due]
    return keys[:limit] if limit > 0 else keys


def arxiv_id_from_seen_key(key: str) -> str:
    try:
        match = ARXIV_ABS_RE.match(urlsplit(key).path)
    except ValueError:
        return ""
    return match.group(1) if match else ""


def candidate_cache_disposition(candidate: Any) -> tuple[str, str]:
    if getattr(candidate, "duplicate_matches", []):
        return "terminal", "repository_duplicate"
    if getattr(candidate, "exclusion_hits", []) or getattr(candidate, "review_bucket", "") == "reject":
        return "terminal", "excluded"
    if getattr(candidate, "relevance", "unknown") == "low":
        return "terminal", "low_relevance"

    availability = getattr(candidate, "artifact_availability", {}) or {}
    if getattr(candidate, "relevance", "unknown") == "high" and (
        availability.get("has_verified_model_link") or availability.get("has_verified_code_link")
    ):
        return "retry", "awaiting_submission"

    if availability.get("has_verified_project_page"):
        return "retry", "project_without_release"

    checks = getattr(candidate, "checks", []) or []
    retryable_statuses = {
        "not_found",
        "placeholder",
        "private_or_gated",
        "private_or_rate_limited",
        "unknown",
    }
    if any(check.status in retryable_statuses for check in checks):
        return "retry", "artifact_temporarily_unavailable"

    definitive_bad = {"archived", "not_available", "unofficial"}
    if checks and all(check.status in definitive_bad for check in checks):
        return "terminal", "unavailable_or_unofficial"
    return "terminal", "paper_only"


def retry_at(seen_at: str, attempt_count: int) -> str:
    delay_days = 7 if attempt_count == 0 else RETRY_DELAYS_DAYS[
        min(attempt_count - 1, len(RETRY_DELAYS_DAYS) - 1)
    ]
    current = datetime.fromisoformat(seen_at.replace("Z", "+00:00"))
    return (current + timedelta(days=delay_days)).isoformat().replace("+00:00", "Z")


def update_seen_cache(
    seen_cache: dict[str, Any],
    candidates: list[SeenCandidate],
    *,
    seen_at: str | None = None,
) -> dict[str, Any]:
    seen_at = seen_at or utc_now_iso()
    seen = seen_cache.setdefault("seen", {})
    if not isinstance(seen, dict):
        seen = {}
        seen_cache["seen"] = seen

    for candidate in candidates:
        key = candidate_seen_key(candidate)
        if not key:
            continue

        existing = seen.get(key, {})
        if not isinstance(existing, dict):
            existing = {}

        status, reason = candidate_cache_disposition(candidate)
        previous_updated = str(existing.get("arxiv_updated") or "")
        current_updated = str(getattr(candidate, "updated", "") or "")
        revision_changed = bool(previous_updated and current_updated and previous_updated != current_updated)
        attempt_count = 1 if revision_changed else int(existing.get("attempt_count", 0) or 0) + 1
        if (
            status == "retry"
            and reason != "awaiting_submission"
            and attempt_count > len(RETRY_DELAYS_DAYS)
        ):
            status, reason = "terminal", "retry_exhausted"
        retry_attempt = 0 if reason == "awaiting_submission" else attempt_count

        seen[key] = {
            "title": candidate.title,
            "source": candidate.source,
            "published": candidate.published,
            "url": candidate.url,
            "first_seen_at": existing.get("first_seen_at") or seen_at,
            "last_checked_at": seen_at,
            "arxiv_updated": getattr(candidate, "updated", ""),
            "arxiv_version": getattr(candidate, "arxiv_version", ""),
            "status": status,
            "reason": reason,
            "attempt_count": attempt_count,
            "next_check_at": retry_at(seen_at, retry_attempt) if status == "retry" else "",
        }

    seen_cache["version"] = SEEN_CACHE_VERSION
    return seen_cache


def mark_seen_submitted(
    seen_cache: dict[str, Any],
    keys: list[str] | set[str],
    *,
    seen_at: str | None = None,
) -> dict[str, Any]:
    seen_at = seen_at or utc_now_iso()
    seen = seen_cache.setdefault("seen", {})
    if not isinstance(seen, dict):
        seen = {}
        seen_cache["seen"] = seen

    for raw_key in keys:
        key = canonicalize_seen_url(raw_key)
        if not key:
            continue
        meta = seen.get(key, {})
        if not isinstance(meta, dict):
            meta = {}
        meta.update({
            "status": "submitted",
            "reason": "add_model_issue_exists",
            "last_checked_at": seen_at,
            "next_check_at": "",
        })
        seen[key] = meta

    seen_cache["version"] = SEEN_CACHE_VERSION
    return seen_cache


def write_seen_cache(path: str, seen_cache: dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(seen_cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run weekly Physical AI discovery with seen-paper filtering.")
    parser.add_argument("--days", type=int, default=7, help="Look back this many days.")
    parser.add_argument("--max-arxiv", type=int, default=500, help="Maximum arXiv papers to fetch.")
    parser.add_argument(
        "--max-rechecks",
        type=int,
        default=50,
        help="Maximum cached papers whose latest arXiv metadata is fetched again.",
    )
    parser.add_argument("--format", choices=("markdown", "json", "jsonl"), default="markdown")
    parser.add_argument("--output", help="Write report to this file instead of stdout.")
    parser.add_argument("--no-verify", action="store_true", help="Skip network checks for extracted official links.")
    parser.add_argument("--seen-cache", required=True, help="JSON cache path for papers already processed.")
    parser.add_argument(
        "--no-update-seen-cache",
        action="store_true",
        help="Filter with --seen-cache but do not persist newly processed papers.",
    )
    parser.add_argument(
        "--max-ambiguous",
        type=int,
        default=10,
        help="Maximum ambiguous candidates to send to optional LLM review.",
    )
    parser.add_argument(
        "--llm-review-mode",
        choices=("off", "ambiguous", "all"),
        default="off",
        help="Choose which candidates are sent to --llm-review-command.",
    )
    parser.add_argument(
        "--llm-review-command",
        help="Command that receives candidate JSON on stdin and returns one JSON review object.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        candidates = discover.fetch_arxiv_cs_ro(args.days, args.max_arxiv)
        time.sleep(3)
        revised_candidates = discover.fetch_recent_arxiv_updates(args.days, args.max_arxiv)
        candidates_by_key = {candidate_seen_key(candidate): candidate for candidate in candidates}
        candidates_by_key.update(
            {candidate_seen_key(candidate): candidate for candidate in revised_candidates}
        )
        candidates = list(candidates_by_key.values())

        seen_cache = load_seen_cache(args.seen_cache)
        recent_keys = {candidate_seen_key(candidate) for candidate in candidates}
        retry_keys = [
            key
            for key in due_retry_keys(seen_cache, limit=args.max_rechecks)
            if key not in recent_keys
        ]
        retry_ids = [arxiv_id_from_seen_key(key) for key in retry_keys]
        retry_ids = [arxiv_id for arxiv_id in retry_ids if arxiv_id]
        if retry_ids:
            # arXiv asks clients to leave three seconds between consecutive API calls.
            time.sleep(3)
            # Put deferred candidates first so a small submission limit cannot
            # starve them behind every week's newly published papers.
            candidates = discover.fetch_arxiv_by_ids(retry_ids) + candidates

        fresh_candidates, skipped_candidates = filter_seen_candidates(candidates, seen_cache)
        if skipped_candidates:
            print(
                f"seen-cache: skipped {len(skipped_candidates)} previously seen candidate(s)",
                file=sys.stderr,
            )
        if retry_ids:
            print(f"seen-cache: rechecking {len(retry_ids)} cached candidate(s)", file=sys.stderr)

        evaluated = discover.evaluate_candidates(
            fresh_candidates,
            verify_links=not args.no_verify,
            llm_review_command=args.llm_review_command,
            llm_review_mode=args.llm_review_mode,
            max_ambiguous=args.max_ambiguous,
        )

        if not args.no_update_seen_cache:
            update_seen_cache(seen_cache, evaluated)
            write_seen_cache(args.seen_cache, seen_cache)
    except (requests.RequestException, discover.ArxivResultLimitError) as exc:
        print(f"error: discovery request failed: {exc}", file=sys.stderr)
        return 1

    discover.write_output(evaluated, args.format, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
