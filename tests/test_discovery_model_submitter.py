import json

import discovery_model_submitter as submitter
import process_issue


def candidate(**overrides):
    base = {
        "title": "RoboPolicy: Imitation Learning for Robot Manipulation",
        "url": "http://arxiv.org/abs/2601.00001v2",
        "published": "2026-01-02",
        "summary": "RoboPolicy learns robot manipulation from demonstrations using imitation learning.",
        "source": "arxiv",
        "relevance": "high",
        "recommendation": "needs_review",
        "review_bucket": "normal",
        "duplicate_matches": [],
        "exclusion_hits": [],
        "keyword_hits": ["robotics", "manipulation", "imitation learning"],
        "reasons": ["verified model link and high Physical AI relevance"],
        "artifact_availability": {
            "has_verified_model_link": True,
            "has_verified_artifact_link": True,
            "verified_model_links": ["https://huggingface.co/robot-lab/robopolicy"],
            "verified_project_pages": ["https://robot-lab.github.io/robopolicy"],
        },
        "checks": [
            {
                "status": "available",
                "kind": "hf_model",
                "url": "https://huggingface.co/robot-lab/robopolicy",
            },
            {
                "status": "available",
                "kind": "github",
                "url": "https://github.com/robot-lab/robopolicy",
            },
        ],
        "llm_review": {},
    }
    base.update(overrides)
    return base


def test_build_submission_requires_high_confidence_rule_based_model():
    submission, reasons = submitter.build_model_submission(candidate())

    assert reasons == []
    assert submission is not None
    assert submission.name == "RoboPolicy"
    assert submission.id == "robopolicy"
    assert submission.organization == "robot-lab"
    assert submission.paper_url == "https://arxiv.org/abs/2601.00001"
    assert submission.categories == ("manipulation",)
    assert submission.hardware_targets == ("manipulator",)
    assert submission.learning_methods == ("IL",)


def test_build_submission_rejects_candidate_without_verified_code_or_weights():
    submission, reasons = submitter.build_model_submission(
        candidate(
            review_bucket="ambiguous",
            artifact_availability={"has_verified_model_link": False, "verified_model_links": []},
        )
    )

    assert submission is None
    assert any("neither public model weights nor public code" in reason for reason in reasons)


def test_build_submission_accepts_high_relevance_verified_code_without_weights():
    code_only = candidate(
        review_bucket="ambiguous",
        artifact_availability={
            "has_verified_model_link": False,
            "has_verified_code_link": True,
            "has_verified_artifact_link": True,
            "verified_model_links": [],
            "verified_project_pages": [],
        },
        checks=[
            {
                "status": "available",
                "kind": "github",
                "url": "https://github.com/robot-lab/robopolicy",
            }
        ],
    )

    submission, reasons = submitter.build_model_submission(code_only)

    assert submission is not None and not reasons
    assert submission.github_url == "https://github.com/robot-lab/robopolicy"
    assert submission.huggingface_url == ""


def test_add_model_body_is_compatible_with_existing_issue_parser():
    submission, reasons = submitter.build_model_submission(candidate())
    assert submission is not None and not reasons

    body = submitter.render_add_model_issue(submission, source_run="https://github.com/org/repo/actions/runs/1")
    form = process_issue.parse_form(body)
    entry = process_issue.build_model_entry(form)

    assert entry["id"] == "robopolicy"
    assert entry["name"] == "RoboPolicy"
    assert entry["org"] == "robot-lab"
    assert entry["year"] == 2026
    assert entry["github_url"] == "https://github.com/robot-lab/robopolicy"
    assert entry["paper_url"] == "https://arxiv.org/abs/2601.00001"
    assert entry["hf_url"] == "https://huggingface.co/robot-lab/robopolicy"
    assert entry["categories"] == ["manipulation"]
    assert entry["hardware"] == ["manipulator"]
    assert entry["learning"] == ["IL"]
    assert "discovery-key: https://arxiv.org/abs/2601.00001" in body


def test_llm_metadata_is_used_only_after_enum_filtering():
    llm_review = {
        "status": "ok",
        "model_name": "Official RoboPolicy",
        "organization": "Robot Lab",
        "entry_summary": "An evidence-based model summary.",
        "categories": ["manipulation", "unsupported"],
        "hardware_targets": ["manipulator", "car"],
        "learning_methods": ["IL", "supervised"],
        "framework": ["pytorch", "numpy"],
        "communication": ["ros2", "mqtt"],
    }

    submission, reasons = submitter.build_model_submission(candidate(llm_review=llm_review))

    assert submission is not None and not reasons
    assert submission.name == "Official RoboPolicy"
    assert submission.organization == "Robot Lab"
    assert submission.categories == ("manipulation",)
    assert submission.hardware_targets == ("manipulator",)
    assert submission.learning_methods == ("IL",)
    assert submission.framework == ("pytorch",)
    assert submission.communication == ("ros2",)


def test_create_issue_uses_add_model_title_and_label(monkeypatch):
    submission, _ = submitter.build_model_submission(candidate())
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"html_url": "https://github.com/org/repo/issues/1"}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(submitter.requests, "post", fake_post)
    url = submitter.create_add_model_issue(
        repo="org/repo", token="token", submission=submission, body="body"
    )

    assert url.endswith("/issues/1")
    assert captured["json"]["title"] == "[Model] RoboPolicy"
    assert captured["json"]["labels"] == ["add-model"]


def test_existing_keys_read_discovery_marker_and_ignore_pull_requests(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"body": "<!-- discovery-key: http://arxiv.org/abs/2601.00001v3 -->"},
                {"body": "<!-- discovery-key: https://arxiv.org/abs/9999.99999 -->", "pull_request": {}},
            ]

    monkeypatch.setattr(submitter.requests, "get", lambda *args, **kwargs: Response())

    assert submitter.existing_discovery_keys(repo="org/repo", token="token") == {
        "https://arxiv.org/abs/2601.00001"
    }


def test_load_candidates_requires_json_list(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"items": []}), encoding="utf-8")

    try:
        submitter.load_candidates(str(path))
    except ValueError as exc:
        assert "must be a list" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-list discovery JSON")


def test_dry_run_cli_renders_add_model_submission(tmp_path, capsys):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([candidate()]), encoding="utf-8")

    result = submitter.main(["--input", str(path), "--dry-run", "--limit", "1"])
    output = capsys.readouterr().out

    assert result == 0
    assert "Would submit Add a Model issue: [Model] RoboPolicy" in output
    assert "### ID (slug)\nrobopolicy" in output


def test_existing_submission_does_not_consume_limit_and_updates_cache(tmp_path, monkeypatch):
    first = candidate(url="https://arxiv.org/abs/2601.00001")
    second = candidate(
        title="RoboPolicy Two: Imitation Learning for Robot Manipulation",
        url="https://arxiv.org/abs/2601.00002",
    )
    input_path = tmp_path / "candidates.json"
    input_path.write_text(json.dumps([first, second]), encoding="utf-8")
    cache_path = tmp_path / "seen.json"
    cache_path.write_text(
        json.dumps({
            "version": 2,
            "seen": {
                "https://arxiv.org/abs/2601.00001": {"status": "retry"},
                "https://arxiv.org/abs/2601.00002": {"status": "retry"},
            },
        }),
        encoding="utf-8",
    )
    created = []

    monkeypatch.setattr(
        submitter,
        "existing_discovery_keys",
        lambda **_kwargs: {"https://arxiv.org/abs/2601.00001"},
    )
    monkeypatch.setattr(
        submitter,
        "create_add_model_issue",
        lambda **kwargs: created.append(kwargs["submission"].key) or "https://github.com/org/repo/issues/2",
    )

    result = submitter.main([
        "--input", str(input_path),
        "--repo", "org/repo",
        "--token", "token",
        "--limit", "1",
        "--seen-cache", str(cache_path),
    ])
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    assert result == 0
    assert created == ["https://arxiv.org/abs/2601.00002"]
    assert cache["seen"]["https://arxiv.org/abs/2601.00001"]["status"] == "submitted"
    assert cache["seen"]["https://arxiv.org/abs/2601.00002"]["status"] == "submitted"


def test_candidates_beyond_submission_limit_remain_retryable(tmp_path, monkeypatch):
    candidates = [
        candidate(
            title=f"RoboPolicy {index}: Imitation Learning for Robot Manipulation",
            url=f"https://arxiv.org/abs/2601.0000{index}",
        )
        for index in range(1, 4)
    ]
    input_path = tmp_path / "candidates.json"
    input_path.write_text(json.dumps(candidates), encoding="utf-8")
    cache_path = tmp_path / "seen.json"
    cache_path.write_text(
        json.dumps({
            "version": 2,
            "seen": {
                item["url"]: {"status": "retry", "next_check_at": "2026-01-08T00:00:00Z"}
                for item in candidates
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(submitter, "existing_discovery_keys", lambda **_kwargs: set())
    monkeypatch.setattr(
        submitter,
        "create_add_model_issue",
        lambda **_kwargs: "https://github.com/org/repo/issues/1",
    )

    submitter.main([
        "--input", str(input_path),
        "--repo", "org/repo",
        "--token", "token",
        "--limit", "1",
        "--seen-cache", str(cache_path),
    ])
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    statuses = [cache["seen"][item["url"]]["status"] for item in candidates]

    assert statuses == ["submitted", "retry", "retry"]
