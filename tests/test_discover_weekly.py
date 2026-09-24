"""Unit tests for the weekly discovery runner and its integrated seen cache."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import discover_weekly as weekly


@dataclass
class Candidate:
    source: str = "arxiv"
    title: str = "Open Robot Manipulation Policy"
    url: str = "https://arxiv.org/abs/2601.00001"
    published: str = "2026-01-01"
    updated: str = "2026-01-01T00:00:00Z"
    arxiv_version: str = "v1"


def test_canonicalize_seen_url_normalizes_arxiv_versions_and_scheme():
    assert weekly.canonicalize_seen_url("http://arxiv.org/abs/2601.00001v3") == "https://arxiv.org/abs/2601.00001"
    assert weekly.canonicalize_seen_url("https://arxiv.org/abs/2601.00001v2?x=1") == "https://arxiv.org/abs/2601.00001"


def test_seen_cache_filters_canonical_paper_url():
    seen_cache = {
        "version": 1,
        "seen": {"https://arxiv.org/abs/2601.00001": {"title": "Already Seen"}},
    }
    seen = Candidate(url="http://arxiv.org/abs/2601.00001v2")
    fresh = Candidate(title="New Robot Policy", url="https://arxiv.org/abs/2601.00002")

    remaining, skipped = weekly.filter_seen_candidates([seen, fresh], seen_cache)

    assert remaining == [fresh]
    assert skipped == [seen]


def test_seen_cache_update_preserves_first_seen_timestamp():
    candidate = Candidate()
    seen_cache = {
        "version": 1,
        "seen": {"https://arxiv.org/abs/2601.00001": {"first_seen_at": "2026-01-01T00:00:00Z"}},
    }

    updated = weekly.update_seen_cache(seen_cache, [candidate], seen_at="2026-01-02T00:00:00Z")
    entry = updated["seen"]["https://arxiv.org/abs/2601.00001"]

    assert entry["title"] == "Open Robot Manipulation Policy"
    assert entry["first_seen_at"] == "2026-01-01T00:00:00Z"
    assert entry["last_checked_at"] == "2026-01-02T00:00:00Z"
    assert entry["status"] == "terminal"
    assert entry["reason"] == "paper_only"
    assert entry["next_check_at"] == ""


def test_seen_cache_write_and_load_roundtrip(tmp_path):
    cache_path = tmp_path / "seen.json"
    candidate = Candidate(url="http://arxiv.org/abs/2601.00001v3")
    seen_cache = weekly.update_seen_cache(
        weekly.empty_seen_cache(), [candidate], seen_at="2026-01-02T00:00:00Z"
    )

    weekly.write_seen_cache(str(cache_path), seen_cache)
    loaded = weekly.load_seen_cache(str(cache_path))

    assert "https://arxiv.org/abs/2601.00001" in loaded["seen"]
    assert loaded["seen"]["https://arxiv.org/abs/2601.00001"]["last_checked_at"] == "2026-01-02T00:00:00Z"
    assert loaded["version"] == 2


def test_load_seen_cache_migrates_v1_entries_to_immediate_retry(tmp_path):
    cache_path = tmp_path / "seen.json"
    cache_path.write_text(
        '{"version": 1, "seen": {"https://arxiv.org/abs/2601.00001": {}}}',
        encoding="utf-8",
    )

    loaded = weekly.load_seen_cache(str(cache_path))
    entry = loaded["seen"]["https://arxiv.org/abs/2601.00001"]

    assert loaded["version"] == 2
    assert entry["status"] == "retry"
    assert entry["next_check_at"] == ""


def test_retry_candidate_is_rechecked_when_due_or_revision_changes():
    key = "https://arxiv.org/abs/2601.00001"
    seen_cache = {
        "version": 2,
        "seen": {
            key: {
                "status": "retry",
                "next_check_at": "2026-02-01T00:00:00Z",
                "arxiv_updated": "2026-01-01T00:00:00Z",
            }
        },
    }

    unchanged = Candidate()
    fresh, skipped = weekly.filter_seen_candidates(
        [unchanged], seen_cache, now="2026-01-15T00:00:00Z"
    )
    assert fresh == [] and skipped == [unchanged]

    revised = Candidate(updated="2026-01-10T00:00:00Z", arxiv_version="v2")
    fresh, skipped = weekly.filter_seen_candidates(
        [revised], seen_cache, now="2026-01-15T00:00:00Z"
    )
    assert fresh == [revised] and skipped == []

    fresh, skipped = weekly.filter_seen_candidates(
        [unchanged], seen_cache, now="2026-02-01T00:00:00Z"
    )
    assert fresh == [unchanged] and skipped == []

    seen_cache["seen"][key]["status"] = "submitted"
    fresh, skipped = weekly.filter_seen_candidates(
        [revised], seen_cache, now="2026-02-01T00:00:00Z"
    )
    assert fresh == [] and skipped == [revised]


def test_submission_ready_candidate_retries_next_week_until_submitted():
    candidate = SimpleNamespace(
        source="arxiv",
        title="RoboPolicy",
        url="https://arxiv.org/abs/2601.00001",
        published="2026-01-01",
        updated="2026-01-01T00:00:00Z",
        arxiv_version="v1",
        duplicate_matches=[],
        exclusion_hits=[],
        review_bucket="normal",
        relevance="high",
        checks=[],
        artifact_availability={"has_verified_code_link": True},
    )

    cache = weekly.update_seen_cache(
        weekly.empty_seen_cache(), [candidate], seen_at="2026-01-01T00:00:00Z"
    )
    entry = cache["seen"]["https://arxiv.org/abs/2601.00001"]

    assert entry["status"] == "retry"
    assert entry["reason"] == "awaiting_submission"
    assert entry["next_check_at"] == "2026-01-08T00:00:00Z"

    weekly.mark_seen_submitted(cache, {candidate.url}, seen_at="2026-01-02T00:00:00Z")
    assert entry["status"] == "submitted"
    assert entry["next_check_at"] == ""


def test_placeholder_candidate_is_scheduled_for_recheck():
    candidate = SimpleNamespace(
        source="arxiv",
        title="RoboPolicy",
        url="https://arxiv.org/abs/2601.00001",
        published="2026-01-01",
        updated="2026-01-01T00:00:00Z",
        arxiv_version="v1",
        duplicate_matches=[],
        exclusion_hits=[],
        review_bucket="ambiguous",
        relevance="high",
        checks=[SimpleNamespace(status="placeholder")],
        artifact_availability={},
    )

    cache = weekly.update_seen_cache(
        weekly.empty_seen_cache(), [candidate], seen_at="2026-01-01T00:00:00Z"
    )
    entry = cache["seen"][candidate.url]

    assert entry["status"] == "retry"
    assert entry["reason"] == "artifact_temporarily_unavailable"
    assert entry["next_check_at"] == "2026-01-15T00:00:00Z"


def test_weekly_cli_defaults_to_llm_off():
    args = weekly.build_parser().parse_args(["--seen-cache", "seen.json"])

    assert args.llm_review_mode == "off"
    assert args.max_arxiv == 500


def test_report_only_run_does_not_create_seen_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "seen.json"
    monkeypatch.setattr(weekly.discover, "fetch_arxiv_cs_ro", lambda *_args: [Candidate()])
    monkeypatch.setattr(weekly.discover, "fetch_recent_arxiv_updates", lambda *_args: [])
    monkeypatch.setattr(weekly.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        weekly.discover,
        "evaluate_candidates",
        lambda candidates, **_kwargs: candidates,
    )
    monkeypatch.setattr(weekly.discover, "write_output", lambda *_args: None)

    result = weekly.main([
        "--seen-cache", str(cache_path),
        "--no-update-seen-cache",
    ])

    assert result == 0
    assert not cache_path.exists()


def test_truncated_arxiv_run_does_not_create_seen_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "seen.json"

    def fail_fetch(*_args):
        raise weekly.discover.ArxivResultLimitError("too many papers")

    monkeypatch.setattr(weekly.discover, "fetch_arxiv_cs_ro", fail_fetch)

    result = weekly.main(["--seen-cache", str(cache_path)])

    assert result == 1
    assert not cache_path.exists()


def test_workflow_uses_integrated_runner_and_dedicated_submission_token():
    workflow = (Path(__file__).parent.parent / ".github" / "workflows" / "discover-weekly.yml").read_text(
        encoding="utf-8"
    )

    assert "python scripts/discover_weekly.py" in workflow
    assert "python scripts/discovery_model_submitter.py" in workflow
    assert "python scripts/llm_reviewer.py" in workflow
    assert "secrets.DISCOVERY_BOT_TOKEN" in workflow
    assert "ARGS+=(--no-update-seen-cache)" in workflow
    assert 'if [ "$SUBMIT_MODE" != "true" ]' in workflow
    assert "--max-arxiv 500" in workflow
    assert "--max-rechecks 50" in workflow
    assert "GITHUB_TOKEN: ${{ github.token }}" in workflow
    assert "--seen-cache .cache/discover_seen.json" in workflow
    assert "discover_new_cached.py" not in workflow
    assert "discovery_issue_creator.py" not in workflow
    assert "llm_reviewer_openrouter.py" not in workflow
    assert "llm_reviewer_gemini.py" not in workflow
