#!/usr/bin/env python3
"""
update_stats.py — Weekly stats updater for Awesome Physical AI.

Fetches GitHub stars/forks and HuggingFace download counts for all entries
in data/models.yaml and data/datasets.yaml, then writes the updated stats back.

Requires:
  GITHUB_TOKEN env var (for higher API rate limits)

Usage:
  python scripts/update_stats.py
"""

import os
import re
import time
import logging
from datetime import date
from pathlib import Path

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "")
TODAY = date.today().isoformat()

GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    GITHUB_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

HF_HEADERS = {}
if HF_TOKEN:
    HF_HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

_broken_urls: list[tuple[str, str, str]] = []  # (entry_id, field, url)


def _parse_github_slug(url: str) -> str | None:
    if not url:
        return None
    m = re.match(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?(?:/.*)?$", url)
    return m.group(1) if m else None


def fetch_github_stats(github_url: str, entry_id: str = "") -> dict:
    slug = _parse_github_slug(github_url)
    if not slug:
        return {}
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{slug}",
            headers=GITHUB_HEADERS,
            timeout=15,
        )
        if resp.status_code == 404:
            log.warning("GitHub repo not found: %s", slug)
            _broken_urls.append((entry_id, "github_url", github_url))
            return {}
        resp.raise_for_status()
        data = resp.json()
        return {
            "github_stars": data.get("stargazers_count", 0),
            "github_forks": data.get("forks_count", 0),
        }
    except requests.RequestException as exc:
        log.error("GitHub API error for %s: %s", slug, exc)
        return {}


def _parse_hf_slug(url: str) -> tuple[str | None, str | None]:
    if not url:
        return None, None
    m = re.match(r"https?://huggingface\.co/(datasets/)?([^/]+/[^/?#]+)", url)
    if not m:
        return None, None
    kind = "datasets" if m.group(1) else "models"
    return kind, m.group(2)


def fetch_hf_downloads(hf_url: str) -> int:
    kind, slug = _parse_hf_slug(hf_url)
    if not slug:
        return 0
    try:
        resp = requests.get(
            f"https://huggingface.co/api/{kind}/{slug}",
            headers=HF_HEADERS,
            timeout=15,
        )
        if resp.status_code == 404:
            log.warning("HF resource not found: %s/%s", kind, slug)
            return 0
        resp.raise_for_status()
        data = resp.json()
        return data.get("downloads", 0) or data.get("downloadsAllTime", 0)
    except requests.RequestException as exc:
        log.error("HF API error for %s/%s: %s", kind, slug, exc)
        return 0


def load_yaml(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def save_yaml(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def create_broken_url_issue(broken: list[tuple[str, str, str]]) -> None:
    if not GITHUB_REPO or not GITHUB_TOKEN:
        log.warning("Cannot create issue: GITHUB_REPOSITORY or GITHUB_TOKEN not set")
        return
    rows = "\n".join(f"- `{eid}` · `{field}`: {url}" for eid, field, url in broken)
    body = (
        f"The weekly stats update on {TODAY} found **{len(broken)}** broken URL(s) "
        f"that returned 404.\n\nPlease update or remove these entries:\n\n{rows}"
    )
    resp = requests.post(
        f"https://api.github.com/repos/{GITHUB_REPO}/issues",
        headers=GITHUB_HEADERS,
        json={"title": f"chore: broken URLs detected ({TODAY})", "body": body, "labels": ["data-quality"]},
        timeout=15,
    )
    if resp.status_code == 201:
        log.info("Created issue: %s", resp.json().get("html_url"))
    else:
        log.error("Failed to create issue: %s %s", resp.status_code, resp.text)


def update_entry(entry: dict) -> dict:
    entry_id = entry.get("id", "?")
    log.info("Updating: %s", entry_id)
    stats = entry.setdefault("stats", {})

    gh_stats = fetch_github_stats(entry.get("github_url", ""), entry_id)
    if gh_stats:
        stats.update(gh_stats)
        log.info("  stars=%d forks=%d", stats.get("github_stars", 0), stats.get("github_forks", 0))
    time.sleep(0.5)

    hf_url = entry.get("hf_url", "")
    if hf_url:
        stats["hf_downloads"] = fetch_hf_downloads(hf_url)
        log.info("  hf_downloads=%d", stats["hf_downloads"])
    time.sleep(0.3)

    stats["last_updated"] = TODAY
    return entry


def update_tool_entry(entry: dict) -> dict:
    """Like update_entry but tools only track github_stars (no HF downloads)."""
    entry_id = entry.get("id", "?")
    log.info("Updating tool: %s", entry_id)
    stats = entry.setdefault("stats", {})

    gh_stats = fetch_github_stats(entry.get("github_url", ""), entry_id)
    if gh_stats:
        stats["github_stars"] = gh_stats.get("github_stars", 0)
        log.info("  stars=%d", stats["github_stars"])
    time.sleep(0.5)

    stats["last_updated"] = TODAY
    return entry


def update_file(yaml_path: Path) -> int:
    if not yaml_path.exists():
        log.error("File not found: %s", yaml_path)
        return 0
    entries = load_yaml(yaml_path)
    for i, entry in enumerate(entries):
        try:
            entries[i] = update_entry(entry)
        except Exception as exc:
            log.error("Failed to update '%s': %s", entry.get("id"), exc)
    save_yaml(yaml_path, entries)
    log.info("Saved %d entries → %s", len(entries), yaml_path)
    return len(entries)


def update_tools_file(yaml_path: Path) -> int:
    if not yaml_path.exists():
        log.error("File not found: %s", yaml_path)
        return 0
    entries = load_yaml(yaml_path)
    for i, entry in enumerate(entries):
        try:
            entries[i] = update_tool_entry(entry)
        except Exception as exc:
            log.error("Failed to update tool '%s': %s", entry.get("id"), exc)
    save_yaml(yaml_path, entries)
    log.info("Saved %d tools → %s", len(entries), yaml_path)
    return len(entries)


def main() -> None:
    if not GITHUB_TOKEN:
        log.warning("GITHUB_TOKEN not set — rate-limited to 60 requests/hour.")
    if not HF_TOKEN:
        log.warning("HF_TOKEN not set — gated HuggingFace repos will return 401.")
    update_file(DATA_DIR / "models.yaml")
    update_file(DATA_DIR / "datasets.yaml")
    update_tools_file(DATA_DIR / "tools.yaml")
    if _broken_urls:
        log.warning("%d broken URL(s) detected — creating GitHub issue", len(_broken_urls))
        create_broken_url_issue(_broken_urls)
    log.info("Done.")


if __name__ == "__main__":
    main()
