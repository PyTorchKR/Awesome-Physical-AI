#!/usr/bin/env python3
"""
process_issue.py — Parse a GitHub Issue form body and append a new entry to the YAML.

Called by .github/workflows/process-issue.yml when a new "Add Model", "Add Dataset",
or "Add Simulator" issue is opened. The workflow passes the issue body and metadata
via environment variables, and this script:
  1. Parses the structured issue form (handles both text inputs and checkboxes)
  2. Validates the entry
  3. Writes it to the appropriate YAML file
  4. The workflow then creates a PR for admin review

Environment variables (set by the GitHub Actions workflow):
  ISSUE_BODY        — raw issue body text
  ISSUE_TYPE        — "model", "dataset", or "tool"
  ISSUE_NUMBER      — GitHub issue number
  ISSUE_AUTHOR      — GitHub username of issue author
"""

import os
import re
import sys
from pathlib import Path
from datetime import date

import requests
import yaml

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def check_url(url: str) -> tuple[bool, str]:
    """HEAD request a URL. Returns (is_valid, warning_message).

    404 → invalid (broken link).
    401/403 → valid URL but access-restricted; return warning only.
    Other errors → treated as valid to avoid false positives.
    """
    if not url:
        return True, ""
    headers = {}
    if "huggingface.co" in url and HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    elif "github.com" in url and GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        resp = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if resp.status_code == 404:
            return False, f"URL returned 404 (not found): {url}"
        if resp.status_code == 401:
            return True, f"URL requires authentication (gated): {url}"
        if resp.status_code == 403:
            return True, f"URL is access-restricted: {url}"
    except requests.RequestException:
        pass  # network errors are not treated as broken links
    return True, ""

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
TODAY = date.today().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Form parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_form(body: str) -> dict[str, str]:
    """Parse GitHub issue form body into a key→value dict.

    GitHub Forms render as:
        ### Field Label
        value text   (for inputs / textareas)

        ### Checkboxes
        - [x] checked item
        - [ ] unchecked item
    """
    result: dict[str, str] = {}
    current_key = None
    current_lines: list[str] = []

    for line in body.splitlines():
        heading = re.match(r"^###\s+(.+)$", line)
        if heading:
            if current_key is not None:
                result[current_key] = "\n".join(current_lines).strip()
            current_key = heading.group(1).strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            current_lines = []
        elif current_key is not None:
            if line.strip() not in ("_No response_", ""):
                current_lines.append(line)

    if current_key is not None:
        result[current_key] = "\n".join(current_lines).strip()

    return result


def parse_checkboxes(value: str) -> list[str]:
    """Extract only the checked items from a markdown checkbox block.

    GitHub renders checkboxes as:
        - [x] checked value
        - [ ] unchecked value
    """
    checked = []
    for line in value.splitlines():
        m = re.match(r"^\s*-\s*\[x\]\s+(.+)$", line, re.IGNORECASE)
        if m:
            # Strip trailing parenthetical descriptions added to dropdown options
            # e.g. "physics_engine — 물리 엔진 (MuJoCo ...)" → "physics_engine"
            item = m.group(1).strip()
            item = re.split(r"\s+[—–-]\s+", item)[0].strip()
            checked.append(item)
    return checked


def parse_list(value: str) -> list[str]:
    """Parse a comma-separated or newline-separated string into a cleaned list."""
    return [v.strip() for v in re.split(r"[,\n]+", value) if v.strip()]


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(re.sub(r"[^\d]", "", value))
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Entry builders
# ─────────────────────────────────────────────────────────────────────────────

def build_model_entry(form: dict) -> dict:
    return {
        "id": re.sub(r"[^a-z0-9-]", "", form.get("id_slug", form.get("id", "")).lower().replace(" ", "-")),
        "name": form.get("name", ""),
        "org": form.get("organization", ""),
        "year": to_int(form.get("year", str(date.today().year))),
        "description_en": form.get("description_english", form.get("description_en", "")),
        "description_ko": form.get("description_korean", form.get("description_ko", "")),
        "github_url": form.get("github_url", ""),
        "paper_url": form.get("paper_url_arxiv", form.get("paper_url", "")),
        "hf_url": form.get("huggingface_url", ""),
        "project_url": form.get("project_page_url", form.get("project_/_docs_url", "")),
        "categories": parse_checkboxes(form.get("categories", "")),
        "hardware": parse_checkboxes(form.get("hardware_targets", "")),
        "learning": parse_checkboxes(form.get("learning_methods", "")),
        "framework": parse_checkboxes(form.get("framework", "")),
        "communication": parse_checkboxes(form.get("communication", "")),
        "stats": {
            "github_stars": 0,
            "github_forks": 0,
            "hf_downloads": 0,
            "last_updated": TODAY,
        },
        "added_date": TODAY,
        "tags": parse_list(form.get("tags_optional", form.get("tags", ""))),
    }


def build_dataset_entry(form: dict) -> dict:
    return {
        "id": re.sub(r"[^a-z0-9-]", "", form.get("id_slug", form.get("id", "")).lower().replace(" ", "-")),
        "name": form.get("name", ""),
        "org": form.get("organization", ""),
        "year": to_int(form.get("year", str(date.today().year))),
        "description_en": form.get("description_english", form.get("description_en", "")),
        "description_ko": form.get("description_korean", form.get("description_ko", "")),
        "github_url": form.get("github_url", ""),
        "paper_url": form.get("paper_url_arxiv", form.get("paper_url", "")),
        "hf_url": form.get("huggingface_url", ""),
        "project_url": form.get("project_page_url", ""),
        "categories": parse_checkboxes(form.get("categories", "")),
        "hardware": parse_checkboxes(form.get("hardware_targets", "")),
        "source": parse_checkboxes(form.get("data_source", "")),
        "modality": parse_checkboxes(form.get("modality", "")),
        "scale": {
            "trajectories": to_int(form.get("number_of_trajectories", "0")),
            "hours": to_int(form.get("total_hours", "0")),
            "environments": to_int(form.get("number_of_environments", "0")),
            "robots": to_int(form.get("number_of_robot_types", "0")),
        },
        "stats": {
            "github_stars": 0,
            "hf_downloads": 0,
            "last_updated": TODAY,
        },
        "added_date": TODAY,
        "tags": parse_list(form.get("tags_(optional)", form.get("tags", ""))),
    }


def build_tool_entry(form: dict) -> dict:
    # The dropdown "Type" field may include a description suffix — strip it
    raw_type = form.get("type", "")
    tool_type = re.split(r"\s+[—–-]\s+", raw_type.splitlines()[0] if raw_type else "")[0].strip()

    features = parse_checkboxes(form.get("features", ""))
    gpu = any("gpu" in f.lower() for f in features)
    ros = any("ros" in f.lower() for f in features)

    raw_lang = form.get("primary_languages", form.get("primary_language_s", ""))
    languages = parse_list(raw_lang)

    return {
        "id": re.sub(r"[^a-z0-9-]", "", form.get("id_slug", form.get("id", "")).lower().replace(" ", "-")),
        "name": form.get("name", ""),
        "org": form.get("organization", ""),
        "year": to_int(form.get("year", str(date.today().year))),
        "description_en": form.get("description_english", form.get("description_en", "")),
        "description_ko": form.get("description_korean", form.get("description_ko", "")),
        "github_url": form.get("github_url", ""),
        "paper_url": form.get("paper_url_arxiv", form.get("paper_url", "")),
        "project_url": form.get("project_/_docs_url", form.get("project_url", "")),
        "type": tool_type,
        "gpu_accelerated": gpu,
        "ros_support": ros,
        "language": languages,
        "stats": {
            "github_stars": 0,
            "last_updated": TODAY,
        },
        "added_date": TODAY,
        "tags": parse_list(form.get("tags_optional", form.get("tags", ""))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# YAML I/O
# ─────────────────────────────────────────────────────────────────────────────

def append_entry(yaml_path: Path, entry: dict) -> None:
    with open(yaml_path, encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []

    existing_ids = {e.get("id") for e in entries}
    if entry["id"] in existing_ids:
        print(f"::error::Entry with id '{entry['id']}' already exists in {yaml_path.name}")
        sys.exit(1)

    entries.append(entry)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(entries, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"✅ Appended '{entry['id']}' to {yaml_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    body = os.environ.get("ISSUE_BODY", "")
    issue_type = os.environ.get("ISSUE_TYPE", "").lower()
    issue_number = os.environ.get("ISSUE_NUMBER", "?")
    author = os.environ.get("ISSUE_AUTHOR", "unknown")

    if not body:
        print("::error::ISSUE_BODY is empty")
        sys.exit(1)

    if issue_type not in ("model", "dataset", "tool"):
        print(f"::error::ISSUE_TYPE must be 'model', 'dataset', or 'tool', got: '{issue_type}'")
        sys.exit(1)

    form = parse_form(body)
    print(f"Parsed form fields: {list(form.keys())}")

    if issue_type == "model":
        entry = build_model_entry(form)
        yaml_path = DATA_DIR / "models.yaml"
    elif issue_type == "dataset":
        entry = build_dataset_entry(form)
        yaml_path = DATA_DIR / "datasets.yaml"
    else:
        entry = build_tool_entry(form)
        yaml_path = DATA_DIR / "tools.yaml"

    if not entry["id"]:
        print("::error::Could not determine entry 'id' from form")
        sys.exit(1)

    if not entry["name"]:
        print("::error::Entry 'name' is required")
        sys.exit(1)

    url_fields = ["github_url", "paper_url", "hf_url", "project_url"]
    broken, warned = [], []
    for field in url_fields:
        url = entry.get(field, "")
        if not url:
            continue
        valid, msg = check_url(url)
        if not valid:
            broken.append(field)
        elif msg:
            warned.append(field)

    for field in warned:
        print(f"::warning::{field} is access-restricted (gated or private) — included anyway")
    if broken:
        field_list = ", ".join(broken)
        print(f"::error::The following URLs returned 404: {field_list}. Please fix the links and edit the issue to retry.")
        sys.exit(1)

    append_entry(yaml_path, entry)
    print(f"Entry '{entry['name']}' added by @{author} (issue #{issue_number})")


if __name__ == "__main__":
    main()
