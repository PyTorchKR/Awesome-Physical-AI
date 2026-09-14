"""
Unit tests for scripts/discover_new.py

Run with:
  pytest tests/test_discover_new.py -v
"""

import discover_new as dn


def _candidate(**overrides):
    base = {
        "source": "arxiv",
        "kind": "paper",
        "title": "Open Robot Manipulation Policy",
        "url": "https://arxiv.org/abs/2601.00001",
        "published": "2026-01-01",
        "summary": "A robot learning method for manipulation with imitation learning.",
        "links": ["https://arxiv.org/abs/2601.00001"],
    }
    base.update(overrides)
    return dn.Candidate(**base)


def test_extract_urls_deduplicates_and_strips_punctuation():
    urls = dn.extract_urls("See https://example.com, and https://github.com/a/b.")
    assert urls == ["https://example.com/", "https://github.com/a/b"]


def test_canonicalize_url_normalizes_known_hosts():
    assert dn.canonicalize_url("https://github.com/org/repo.git?tab=readme#x") == "https://github.com/org/repo"
    assert dn.canonicalize_url("https://arxiv.org/abs/2601.00001v2") == "https://arxiv.org/abs/2601.00001"
    assert dn.canonicalize_url("https://huggingface.co/datasets/org/data/tree/main") == "https://huggingface.co/datasets/org/data"


def test_classify_url():
    assert dn.classify_url("https://github.com/org/repo") == "github"
    assert dn.classify_url("https://huggingface.co/datasets/org/data") == "hf_dataset"
    assert dn.classify_url("https://huggingface.co/org/model") == "hf_model"
    assert dn.classify_url("https://arxiv.org/abs/1234.56789") == "paper"
    assert dn.classify_url("https://doi.org/10.1109/example") == "publication"
    assert dn.classify_url("https://project.example") == "project"


def test_classify_url_uses_hostname_not_substring():
    assert dn.classify_url("https://example.com/redirect?target=https://github.com/org/repo") == "project"


def test_assess_relevance_high_for_robotics_terms():
    relevance, reasons = dn.assess_relevance(
        "Robot Manipulation with Vision Language Action Models",
        "We train a robot policy for dexterous manipulation.",
    )
    assert relevance == "high"
    assert any("physical-ai keywords" in reason for reason in reasons)


def test_assess_relevance_low_for_autonomous_driving_only():
    relevance, reasons = dn.assess_relevance(
        "End-to-End Autonomous Driving Dataset",
        "A traffic and lane detection benchmark for self driving.",
    )
    assert relevance == "low"
    assert any("exclusion keywords" in reason for reason in reasons)


def test_find_duplicate_matches_by_url():
    candidate = _candidate(links=["https://github.com/example/robot"])
    matches = dn.find_duplicate_matches(candidate, [
        {
            "_file": "models.yaml",
            "id": "robot",
            "name": "Other",
            "github_url": "https://github.com/example/robot",
            "paper_url": "",
            "hf_url": "",
            "project_url": "",
        }
    ])
    assert matches == ["models.yaml:robot URL match"]


def test_find_duplicate_matches_by_canonical_url():
    candidate = _candidate(links=["https://github.com/example/robot.git?tab=readme"])
    matches = dn.find_duplicate_matches(candidate, [
        {
            "_file": "models.yaml",
            "id": "robot",
            "name": "Other",
            "github_url": "https://github.com/example/robot/",
            "paper_url": "",
            "hf_url": "",
            "project_url": "",
        }
    ])
    assert matches == ["models.yaml:robot URL match"]


def test_find_duplicate_matches_by_arxiv_versionless_url():
    candidate = _candidate(url="https://arxiv.org/abs/2601.00001v2", links=["https://arxiv.org/abs/2601.00001v2"])
    matches = dn.find_duplicate_matches(candidate, [
        {
            "_file": "models.yaml",
            "id": "robot-paper",
            "name": "Other",
            "github_url": "",
            "paper_url": "https://arxiv.org/abs/2601.00001",
            "hf_url": "",
            "project_url": "",
        }
    ])
    assert matches == ["models.yaml:robot-paper URL match"]


def test_http_get_retries_transient_status(monkeypatch):
    calls = []

    class Response:
        headers = {}

        def __init__(self, status_code):
            self.status_code = status_code

    class Session:
        def get(self, *_args, **_kwargs):
            calls.append(1)
            return Response(429 if len(calls) == 1 else 200)

    monkeypatch.setattr(dn, "HTTP_SESSION", Session())
    monkeypatch.setattr(dn.time, "sleep", lambda _delay: None)

    response = dn.http_get("https://example.com/")

    assert response.status_code == 200
    assert len(calls) == 2


def test_find_duplicate_matches_by_name():
    candidate = _candidate(title="Open Robot Manipulation Policy")
    matches = dn.find_duplicate_matches(candidate, [
        {
            "_file": "models.yaml",
            "id": "open-robot-manipulation-policy",
            "name": "Open Robot Manipulation Policy",
            "github_url": "",
            "paper_url": "",
            "hf_url": "",
            "project_url": "",
        }
    ])
    assert matches == ["models.yaml:open-robot-manipulation-policy name match"]


def test_decide_recommendation_rejects_duplicates():
    candidate = _candidate()
    candidate.relevance = "high"
    candidate.duplicate_matches = ["models.yaml:x URL match"]
    candidate.checks = [
        dn.LinkCheck(
            url="https://github.com/example/robot",
            kind="github",
            status="available",
        )
    ]
    recommendation, reasons = dn.decide_recommendation(candidate)
    assert recommendation == "reject"
    assert "duplicate" in reasons[0]


def test_decide_recommendation_needs_review_without_public_links():
    candidate = _candidate()
    candidate.relevance = "high"
    recommendation, reasons = dn.decide_recommendation(candidate)
    assert recommendation == "needs_review"
    assert "no verified public" in reasons[0]


def test_publication_link_does_not_count_as_project_page():
    candidate = _candidate()
    candidate.relevance = "high"
    candidate.checks = [
        dn.LinkCheck(
            url="https://doi.org/10.1109/example",
            kind="publication",
            status="available",
        )
    ]

    availability = dn.build_artifact_availability(candidate)
    recommendation, reasons = dn.decide_recommendation(candidate)

    assert availability["has_verified_project_page"] is False
    assert availability["has_verified_artifact_link"] is False
    assert recommendation == "needs_review"
    assert "no verified public" in reasons[0]


def test_evaluate_candidates_no_verify_marks_links_not_checked(monkeypatch):
    monkeypatch.setattr(dn, "load_yaml_entries", lambda: [])
    candidate = _candidate(links=[
        "https://arxiv.org/abs/2601.00001",
        "https://github.com/example/robot",
    ])

    result = dn.evaluate_candidates([candidate], verify_links=False)

    assert result[0].checks[0].status == "not_checked"
    assert result[0].checks[0].kind == "github"


def test_candidate_review_payload_contains_expected_contract():
    candidate = _candidate()
    payload = dn.candidate_review_payload(candidate)
    assert payload["paper_url"] == "https://arxiv.org/abs/2601.00001"
    assert "expected_json" in payload
    assert "autonomous-driving-only" in payload["question"]


def test_run_llm_review_command_parses_json():
    candidate = _candidate()
    review = dn.run_llm_review_command(
        candidate,
        "python3 -c \"import json; print(json.dumps({'decision':'needs_review','reason':'ok'}))\"",
    )
    assert review["status"] == "ok"
    assert review["decision"] == "needs_review"


def test_run_llm_review_command_handles_invalid_json():
    candidate = _candidate()
    review = dn.run_llm_review_command(candidate, "python3 -c \"print('not json')\"")
    assert review["status"] == "error"
    assert "invalid JSON" in review["reason"]


def test_fetch_arxiv_cs_ro_uses_submitted_date_range_and_max_results(monkeypatch):
    calls = []

    def entry(arxiv_id):
        return f"""
        <entry>
          <id>https://arxiv.org/abs/{arxiv_id}v2</id>
          <published>2026-06-24T00:00:00Z</published>
          <updated>2026-06-25T00:00:00Z</updated>
          <title>Robot Policy {arxiv_id}</title>
          <summary>Robot manipulation paper. https://github.com/example/{arxiv_id}</summary>
          <author><name>Alice</name></author>
          <link href="https://arxiv.org/abs/{arxiv_id}v2" />
        </entry>
        """

    class Response:
        status_code = 200
        headers = {}
        text = f"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">{entry("2601.00001")}</feed>"""

        def raise_for_status(self):
            pass

    def fake_http_get(_url, *, params=None, **_kwargs):
        calls.append(params)
        return Response()

    monkeypatch.setattr(dn, "http_get", fake_http_get)

    candidates = dn.fetch_arxiv_cs_ro(days=7, max_results=20)

    assert len(calls) == 1
    assert calls[0]["start"] == 0
    assert calls[0]["max_results"] == 20
    assert calls[0]["search_query"].startswith("cat:cs.RO AND submittedDate:[")
    assert candidates[0].url == "https://arxiv.org/abs/2601.00001"
    assert candidates[0].arxiv_version == "v2"
    assert candidates[0].updated == "2026-06-25T00:00:00Z"


def test_fetch_arxiv_cs_ro_rejects_truncated_result_set(monkeypatch):
    class Response:
        status_code = 200
        headers = {}
        text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
          <opensearch:totalResults>21</opensearch:totalResults>
        </feed>"""

        def raise_for_status(self):
            pass

    monkeypatch.setattr(dn, "http_get", lambda *args, **kwargs: Response())

    try:
        dn.fetch_arxiv_cs_ro(days=7, max_results=20)
    except dn.ArxivResultLimitError as exc:
        assert "matched 21 papers" in str(exc)
    else:
        raise AssertionError("Expected ArxivResultLimitError")


def test_fetch_arxiv_by_ids_requests_latest_versions(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        headers = {}
        text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

        def raise_for_status(self):
            pass

    def fake_http_get(_url, *, params=None, **_kwargs):
        calls.append(params)
        return Response()

    monkeypatch.setattr(dn, "http_get", fake_http_get)

    assert dn.fetch_arxiv_by_ids(["2601.00001", "2601.00002"]) == []
    assert calls[0]["id_list"] == "2601.00001,2601.00002"
    assert calls[0]["max_results"] == 2


def test_fetch_recent_arxiv_updates_filters_by_updated_date(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        headers = {}
        text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>https://arxiv.org/abs/2601.00001v2</id>
            <published>2026-01-01T00:00:00Z</published>
            <updated>2026-02-09T00:00:00Z</updated>
            <title>Recent revision</title><summary>robot policy</summary>
          </entry>
          <entry>
            <id>https://arxiv.org/abs/2601.00002v1</id>
            <published>2026-01-01T00:00:00Z</published>
            <updated>2026-01-01T00:00:00Z</updated>
            <title>Old paper</title><summary>robot policy</summary>
          </entry>
        </feed>"""

        def raise_for_status(self):
            pass

    def fake_http_get(_url, *, params=None, **_kwargs):
        calls.append(params)
        return Response()

    monkeypatch.setattr(dn, "http_get", fake_http_get)
    recent = dn.fetch_recent_arxiv_updates(
        days=7,
        max_results=3,
        now=dn.datetime(2026, 2, 10, tzinfo=dn.timezone.utc),
    )

    assert [candidate.title for candidate in recent] == ["Recent revision"]
    assert calls[0]["sortBy"] == "lastUpdatedDate"


def test_fetch_recent_arxiv_updates_rejects_truncation(monkeypatch):
    class Response:
        status_code = 200
        headers = {}
        text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>https://arxiv.org/abs/2601.00001v2</id>
            <published>2026-01-01T00:00:00Z</published>
            <updated>2026-02-09T00:00:00Z</updated>
            <title>Recent revision</title><summary>robot policy</summary>
          </entry>
        </feed>"""

        def raise_for_status(self):
            pass

    monkeypatch.setattr(dn, "http_get", lambda *args, **kwargs: Response())

    try:
        dn.fetch_recent_arxiv_updates(
            days=7,
            max_results=1,
            now=dn.datetime(2026, 2, 10, tzinfo=dn.timezone.utc),
        )
    except dn.ArxivResultLimitError as exc:
        assert "at least 1 cs.RO papers" in str(exc)
    else:
        raise AssertionError("Expected ArxivResultLimitError")


def test_evaluate_candidates_runs_llm_review_command(monkeypatch):
    monkeypatch.setattr(dn, "load_yaml_entries", lambda: [])
    candidate = _candidate()
    result = dn.evaluate_candidates(
        [candidate],
        verify_links=False,
        llm_review_command="python3 -c \"import json; print(json.dumps({'decision':'reject','reason':'paper only'}))\"",
    )
    assert result[0].llm_review["status"] == "ok"
    assert result[0].llm_review["decision"] == "reject"


def test_render_markdown_states_no_issues_created():
    candidate = _candidate()
    candidate.relevance = "high"
    candidate.recommendation = "needs_review"
    candidate.llm_review_selected = True
    candidate.llm_review = {
        "status": "ok",
        "decision": "needs_review",
        "entry_summary": "A public-facing summary.",
        "maintainer_summary": "A maintainer-facing summary.",
        "reason": "official link unclear",
    }
    report = dn.render_markdown([candidate])
    assert "No GitHub issues were created" in report
    assert "Open Robot Manipulation Policy" in report
    assert "Targeted for optional LLM review" in report
    assert "LLM decision" in report
    assert "LLM entry summary" in report
    assert "A public-facing summary." in report
    assert "LLM maintainer summary" in report
    assert "A maintainer-facing summary." in report
