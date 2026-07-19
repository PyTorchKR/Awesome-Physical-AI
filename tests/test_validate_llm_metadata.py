"""
Unit tests for scripts/validate_llm_metadata.py
"""

import shutil
from pathlib import Path

import validate_llm_metadata as vlm


class FakeFetcher:
    def __init__(self, evidence_by_id):
        self.evidence_by_id = evidence_by_id

    def fetch(self, entry):
        return self.evidence_by_id[entry["id"]]


def _entry(**overrides):
    base = {
        "id": "test-entry",
        "name": "Test Entry",
        "description_en": "A robot policy for manipulation.",
        "tags": ["manipulation", "pytorch"],
        "github_url": "https://github.com/example/project",
        "paper_url": "https://arxiv.org/abs/2401.00001",
    }
    base.update(overrides)
    return base


def test_normalize_validation_result_accepts_expected_shape():
    result = vlm.normalize_validation_result(
        {
            "tag_score": 0.9,
            "summary_score": 0.8,
            "final_verdict": "pass",
            "reason": "Tags and summary are supported by the evidence.",
            "unsupported_tags": [],
            "unsupported_claims": [],
        }
    )

    assert result["tag_score"] == 0.9
    assert result["summary_score"] == 0.8
    assert result["final_verdict"] == "pass"


def test_normalize_validation_result_rejects_missing_keys():
    try:
        vlm.normalize_validation_result({"tag_score": 0.5})
    except ValueError as exc:
        assert "missing required keys" in str(exc)
    else:
        raise AssertionError("normalize_validation_result should reject incomplete JSON")


def test_validate_entry_pass_case():
    fetcher = FakeFetcher(
        {
            "test-entry": vlm.EvidenceBundle(
                readme_text="README says the project uses PyTorch for manipulation.",
                abstract_text="The paper studies manipulation with an imitation learning policy.",
            )
        }
    )
    validator = vlm.MockGeminiValidator(
        {
            "test-entry": {
                "tag_score": 0.95,
                "summary_score": 0.93,
                "final_verdict": "pass",
                "reason": "Both tags and summary align with README and abstract.",
                "unsupported_tags": [],
                "unsupported_claims": [],
            }
        }
    )

    result = vlm.validate_entry("model", _entry(), fetcher, validator)

    assert result.final_verdict == "pass"
    assert result.evidence_sources == ["README", "Abstract"]


def test_validate_entry_invalid_tag_case():
    fetcher = FakeFetcher(
        {
            "test-entry": vlm.EvidenceBundle(
                readme_text="README only mentions manipulation and PyTorch.",
                abstract_text="The abstract focuses on imitation learning for manipulation.",
            )
        }
    )
    validator = vlm.MockGeminiValidator(
        {
            "test-entry": {
                "tag_score": 0.3,
                "summary_score": 0.84,
                "final_verdict": "fail",
                "reason": "One tag is not supported by the evidence.",
                "unsupported_tags": ["humanoid"],
                "unsupported_claims": [],
            }
        }
    )

    result = vlm.validate_entry(
        "model",
        _entry(tags=["manipulation", "humanoid"]),
        fetcher,
        validator,
    )

    assert result.final_verdict == "fail"
    assert result.unsupported_tags == ["humanoid"]


def test_validate_entry_exaggerated_summary_case():
    fetcher = FakeFetcher(
        {
            "test-entry": vlm.EvidenceBundle(
                readme_text="README describes a research codebase for tabletop manipulation.",
                abstract_text="The abstract reports results on tabletop manipulation.",
            )
        }
    )
    validator = vlm.MockGeminiValidator(
        {
            "test-entry": {
                "tag_score": 0.88,
                "summary_score": 0.2,
                "final_verdict": "fail",
                "reason": "The summary claims whole-body control that is not supported.",
                "unsupported_tags": [],
                "unsupported_claims": ["supports whole-body humanoid control"],
            }
        }
    )

    result = vlm.validate_entry(
        "model",
        _entry(description_en="Supports whole-body humanoid control for many robots."),
        fetcher,
        validator,
    )

    assert result.final_verdict == "fail"
    assert "whole-body humanoid control" in result.unsupported_claims[0]


def test_validate_entry_insufficient_evidence_case():
    fetcher = FakeFetcher(
        {
            "test-entry": vlm.EvidenceBundle(
                readme_text="",
                abstract_text="",
                issues=["README evidence unavailable", "Abstract evidence unavailable"],
            )
        }
    )
    validator = vlm.MockGeminiValidator(
        {
            "test-entry": {
                "tag_score": 0.55,
                "summary_score": 0.52,
                "final_verdict": "warning",
                "reason": "Evidence is too limited for a confident pass.",
                "unsupported_tags": [],
                "unsupported_claims": [],
            }
        }
    )

    result = vlm.validate_entry("model", _entry(), fetcher, validator)

    assert result.final_verdict == "warning"
    assert result.evidence_issues == ["README evidence unavailable", "Abstract evidence unavailable"]


def test_validate_entry_retry_exhausted_becomes_warning():
    fetcher = FakeFetcher(
        {
            "test-entry": vlm.EvidenceBundle(
                readme_text="README mentions manipulation.",
                abstract_text="Abstract mentions manipulation.",
            )
        }
    )
    validator = vlm.MockGeminiValidator(
        {
            "test-entry": vlm.RetryableGeminiError(
                entry_id="test-entry",
                status_code=503,
                attempts=3,
                message="Gemini request failed after 3 attempts",
            )
        }
    )

    result = vlm.validate_entry("model", _entry(), fetcher, validator)

    assert result.final_verdict == "warning"
    assert result.tag_score == 0.0
    assert result.summary_score == 0.0
    assert "temporarily unavailable" in result.reason
    assert any("status 503" in issue for issue in result.evidence_issues)


def test_build_report_counts_and_exit_code():
    results = [
        vlm.ValidationResult(
            entry_id="a",
            entry_type="model",
            name="A",
            tags=[],
            summary="",
            summary_ko="",
            tag_score=0.9,
            summary_score=0.9,
            final_verdict="pass",
            reason="ok",
            unsupported_tags=[],
            unsupported_claims=[],
            evidence_sources=["README"],
            evidence_issues=[],
        ),
        vlm.ValidationResult(
            entry_id="b",
            entry_type="dataset",
            name="B",
            tags=[],
            summary="",
            summary_ko="",
            tag_score=0.4,
            summary_score=0.4,
            final_verdict="fail",
            reason="bad",
            unsupported_tags=["bad-tag"],
            unsupported_claims=[],
            evidence_sources=["README"],
            evidence_issues=[],
        ),
    ]

    report = vlm.build_report(results)

    assert report["counts"] == {"pass": 1, "warning": 0, "fail": 1}
    assert vlm.determine_exit_code(report) == 0


def test_build_report_counts_warning_from_retry_fallback():
    results = [
        vlm.ValidationResult(
            entry_id="retry-case",
            entry_type="model",
            name="Retry Case",
            tags=[],
            summary="",
            summary_ko="",
            tag_score=0.0,
            summary_score=0.0,
            final_verdict="warning",
            reason="Gemini validation temporarily unavailable after 3 attempts (status 503); marked as warning and continued.",
            unsupported_tags=[],
            unsupported_claims=[],
            evidence_sources=["README"],
            evidence_issues=["Gemini validation temporarily unavailable after 3 attempts (status 503); marked as warning and continued."],
        )
    ]

    report = vlm.build_report(results)

    assert report["counts"] == {"pass": 0, "warning": 1, "fail": 0}
    assert vlm.determine_exit_code(report) == 0


def test_main_skips_without_api_key(monkeypatch):
    tmp_dir = Path("reports") / "pytest_llm_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    output = tmp_dir / "llm.json"
    summary = tmp_dir / "llm.md"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    exit_code = vlm.main(["--output", str(output), "--summary-output", str(summary)])

    assert exit_code == 0
    assert output.exists()
    assert summary.exists()
    assert '"status": "skipped"' in output.read_text(encoding="utf-8")
    shutil.rmtree(tmp_dir)


def test_render_actions_summary_contains_table():
    report = {
        "status": "completed",
        "skipped_reason": None,
        "counts": {"pass": 1, "warning": 1, "fail": 0},
        "results": [
            {
                "entry_id": "foo",
                "entry_type": "model",
                "final_verdict": "warning",
                "tag_score": 0.7,
                "summary_score": 0.6,
                "unsupported_tags": ["rl"],
                "unsupported_claims": [],
                "evidence_issues": ["Abstract evidence unavailable"],
                "reason": "Need more evidence",
            }
        ],
    }

    summary = vlm.render_actions_summary(report)

    assert "| Entry | Type | Verdict |" in summary
    assert "foo" in summary
    assert "reason: Need more evidence" in summary
    assert "unsupported tags: rl" in summary


def test_build_prompt_contains_readable_korean_instructions():
    prompt = vlm.build_prompt(
        "model",
        _entry(description_ko="조작 정책 요약입니다."),
        vlm.EvidenceBundle(readme_text="README evidence", abstract_text="Abstract evidence"),
    )

    assert "메타데이터를 검증하는 엄격한 검수자입니다." in prompt
    assert "reason은 반드시 한국어로 작성하세요." in prompt
    assert "README evidence:" in prompt
    assert "Abstract evidence:" in prompt
    
