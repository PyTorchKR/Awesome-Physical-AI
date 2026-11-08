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
  GEMINI_API_KEY    — enables benchmark score extraction for new models
  BENCHMARK_REPORT_PATH — where to write the extraction report for the PR body
"""

import os
import re
import sys
from pathlib import Path
from datetime import date

import requests
import yaml

from extract_benchmarks import extract_for_model, render_report

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


def attach_benchmarks(entry: dict) -> dict:
    """Add scores from the model's own paper right after paper_url.

    Extraction never blocks the entry: on any failure the entry is returned as is
    and the reason goes into the report that the workflow appends to the PR body.
    """
    try:
        result = extract_for_model(entry)
        report = render_report([result])
    except Exception as exc:
        print(f"::warning::Benchmark extraction failed: {exc}")
        return entry

    report_path = os.environ.get("BENCHMARK_REPORT_PATH", "")
    if report_path:
        Path(report_path).write_text(report, encoding="utf-8")
    if result.error:
        print(f"::warning::Benchmark extraction skipped: {result.error}")
    if not result.accepted:
        return entry

    ordered = {}
    for key, value in entry.items():
        ordered[key] = value
        if key == "paper_url":
            ordered["benchmarks"] = result.scores
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Partial-update builders (edit-* issues)
#
# An edit form only carries the fields the submitter wants to change, so these
# builders emit a sparse dict: a key is present only when the form actually
# supplied a value. Everything else — including 'stats', 'added_date' and 'id' —
# is left untouched on the existing entry.
# ─────────────────────────────────────────────────────────────────────────────

def _set_text(form: dict, updates: dict, entry_key: str, *form_keys: str) -> None:
    """Take the first non-empty form field. Blank input leaves the field alone."""
    for fk in form_keys:
        value = form.get(fk, "").strip()
        if value:
            updates[entry_key] = value
            return


def _set_int(form: dict, updates: dict, entry_key: str, *form_keys: str) -> None:
    for fk in form_keys:
        value = form.get(fk, "").strip()
        if value:
            updates[entry_key] = to_int(value)
            return


def _set_bool(form: dict, updates: dict, entry_key: str, form_key: str) -> None:
    """Read a yes/no dropdown. Leaving it unselected keeps the current value."""
    value = form.get(form_key, "").strip().lower()
    if value in ("yes", "true"):
        updates[entry_key] = True
    elif value in ("no", "false"):
        updates[entry_key] = False


def _set_group(form: dict, updates: dict, entry_key: str, form_key: str) -> None:
    """Checkbox group: any box checked replaces the whole list, none keeps it.

    Emptying a group is deliberately not offered — several of these are required
    when an entry is first submitted, so clearing one would break the entry.
    """
    checked = parse_checkboxes(form.get(form_key, ""))
    if checked:
        updates[entry_key] = checked


def _set_tags(form: dict, updates: dict) -> None:
    raw = form.get("tags_optional", form.get("tags", "")).strip()
    if raw:
        updates["tags"] = parse_list(raw)


def build_model_updates(form: dict) -> dict:
    updates: dict = {}
    _set_text(form, updates, "name", "name")
    _set_text(form, updates, "org", "organization")
    _set_text(form, updates, "description_en", "description_english", "description_en")
    _set_text(form, updates, "description_ko", "description_korean", "description_ko")
    _set_text(form, updates, "github_url", "github_url")
    _set_text(form, updates, "paper_url", "paper_url_arxiv", "paper_url")
    _set_text(form, updates, "hf_url", "huggingface_url")
    _set_text(form, updates, "project_url", "project_page_url")
    _set_int(form, updates, "year", "year")
    _set_group(form, updates, "categories", "categories")
    _set_group(form, updates, "hardware", "hardware_targets")
    _set_group(form, updates, "learning", "learning_methods")
    _set_group(form, updates, "framework", "framework")
    _set_group(form, updates, "communication", "communication")
    _set_tags(form, updates)
    return updates


def build_dataset_updates(form: dict) -> dict:
    updates: dict = {}
    _set_text(form, updates, "name", "name")
    _set_text(form, updates, "org", "organization")
    _set_text(form, updates, "description_en", "description_english", "description_en")
    _set_text(form, updates, "description_ko", "description_korean", "description_ko")
    _set_text(form, updates, "github_url", "github_url")
    _set_text(form, updates, "paper_url", "paper_url_arxiv", "paper_url")
    _set_text(form, updates, "hf_url", "huggingface_url")
    _set_text(form, updates, "project_url", "project_page_url")
    _set_int(form, updates, "year", "year")
    _set_group(form, updates, "categories", "categories")
    _set_group(form, updates, "hardware", "hardware_targets")
    _set_group(form, updates, "source", "data_source")
    _set_group(form, updates, "modality", "modality")

    # 'scale' is nested — collect only the numbers that were filled in, so
    # update_entry can merge them into the existing scale dict.
    scale: dict = {}
    _set_int(form, scale, "trajectories", "number_of_trajectories")
    _set_int(form, scale, "hours", "total_hours", "total_hours_of_data")
    _set_int(form, scale, "environments", "number_of_environments", "number_of_environments_/_tasks")
    _set_int(form, scale, "robots", "number_of_robot_types")
    if scale:
        updates["scale"] = scale

    _set_tags(form, updates)
    return updates


def build_tool_updates(form: dict) -> dict:
    updates: dict = {}
    _set_text(form, updates, "name", "name")
    _set_text(form, updates, "org", "organization")
    _set_text(form, updates, "description_en", "description_english", "description_en")
    _set_text(form, updates, "description_ko", "description_korean", "description_ko")
    _set_text(form, updates, "github_url", "github_url")
    _set_text(form, updates, "paper_url", "paper_url_arxiv", "paper_url")
    _set_text(form, updates, "project_url", "project_/_docs_url", "project_url")
    _set_int(form, updates, "year", "year")

    # The Type dropdown carries a description suffix — keep the value only
    raw_type = form.get("type", "").strip()
    if raw_type:
        updates["type"] = re.split(r"\s+[—–-]\s+", raw_type.splitlines()[0])[0].strip()

    _set_bool(form, updates, "gpu_accelerated", "gpu-accelerated")
    _set_bool(form, updates, "ros_support", "ros2_support")

    _set_text(form, updates, "_language_raw", "primary_languages", "primary_language_s")
    if "_language_raw" in updates:
        updates["language"] = parse_list(updates.pop("_language_raw"))

    _set_tags(form, updates)
    return updates


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


def update_entry(yaml_path: Path, entry_id: str, updates: dict) -> None:
    """Apply a partial update to the entry with the given id, preserving its other fields."""
    with open(yaml_path, encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []

    index = next((i for i, e in enumerate(entries) if e.get("id") == entry_id), None)
    if index is None:
        known = ", ".join(sorted(str(e.get("id", "")) for e in entries))
        print(f"::error::No entry with id '{entry_id}' in {yaml_path.name}. Known ids: {known}")
        sys.exit(1)

    if not updates:
        print("::error::The edit form did not contain any new values. "
              "Fill in at least one field and edit the issue to retry.")
        sys.exit(1)

    # Only re-check the URLs this edit actually touches
    broken, warned = [], []
    for field in ("github_url", "paper_url", "hf_url", "project_url"):
        url = updates.get(field, "")
        if not url:
            continue
        valid, msg = check_url(url)
        if not valid:
            broken.append(field)
        elif msg:
            warned.append(msg)

    for msg in warned:
        print(f"::warning::{msg}")
    if broken:
        field_list = ", ".join(broken)
        print(f"::error::The following URLs returned 404: {field_list}. Please fix the links and edit the issue to retry.")
        sys.exit(1)

    entry = entries[index]
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(entry.get(key), dict):
            entry[key].update(value)  # merge nested dicts (e.g. 'scale') rather than replace
        else:
            entry[key] = value

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(entries, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"✅ Updated '{entry_id}' in {yaml_path.name} — fields: {sorted(updates)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    body = os.environ.get("ISSUE_BODY", "")
    issue_type = os.environ.get("ISSUE_TYPE", "").lower()
    action = os.environ.get("ISSUE_ACTION", "add").lower()
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

    yaml_path = DATA_DIR / {
        "model": "models.yaml",
        "dataset": "datasets.yaml",
        "tool": "tools.yaml",
    }[issue_type]

    if action == "edit":
        # The dropdown value reads "<id> — <name>"; keep only the id
        raw_id = re.split(r"\s+[—–-]\s+", form.get("id_slug", form.get("id", "")))[0]
        entry_id = re.sub(r"[^a-z0-9-]", "", raw_id.strip().lower())
        if not entry_id:
            print("::error::Could not determine which entry to edit from the form")
            sys.exit(1)

        builder = {
            "model": build_model_updates,
            "dataset": build_dataset_updates,
            "tool": build_tool_updates,
        }[issue_type]
        update_entry(yaml_path, entry_id, builder(form))
        print(f"Entry '{entry_id}' edited by @{author} (issue #{issue_number})")
        return

    if issue_type == "model":
        entry = build_model_entry(form)
    elif issue_type == "dataset":
        entry = build_dataset_entry(form)
    else:
        entry = build_tool_entry(form)

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

    if issue_type == "model":
        entry = attach_benchmarks(entry)

    append_entry(yaml_path, entry)
    print(f"Entry '{entry['name']}' added by @{author} (issue #{issue_number})")


if __name__ == "__main__":
    main()
