import extract_benchmarks as eb
import process_issue as pi
import pytest
import yaml

PAPER = """
Table 1: CALVIN Benchmark Results.
| Method | Experiment | 1 | 2 | 3 | 4 | 5 | Avg. Len.
| HULC | ABCD→D | 0.889 | 0.733 | 0.587 | 0.475 | 0.383 | 3.06
| GR-1 (Ours) | ABCD→D | 0.949 | 0.896 | 0.844 | 0.789 | 0.731 | 4.21
| GR-1 (Ours) | ABC→D | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06
Table 12: LIBERO simulation benchmark results.
| OpenVLA fine-tuned (ours) | 84.7 ± 0.9% | 1 | 88.4 ± 0.8% | 2
"""


def _score(**overrides) -> dict:
    base = {
        "benchmark": "CALVIN_ACL",
        "reported_value": "4.21",
        "table": "Table 1",
        "row_label": "GR-1 (Ours)",
        "setting": "ABCD→D",
        "is_own_model": True,
        "confident": True,
        "note": "",
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Paper text
# ─────────────────────────────────────────────────────────────────────────────

def test_arxiv_id_ignores_version_and_url_form():
    assert eb.arxiv_id("https://arxiv.org/abs/2312.13139v2") == "2312.13139"
    assert eb.arxiv_id("https://arxiv.org/pdf/2406.09246") == "2406.09246"
    assert eb.arxiv_id("https://openreview.net/forum?id=abc") is None


def test_html_to_text_keeps_a_table_row_on_one_line():
    html = """<table>
<tr><td><span>GR-1 (Ours)</span></td>
<td><p>ABCD<math><mo>→</mo><annotation>\\rightarrow</annotation></math>D</p></td>
<td><span>4.21</span></td></tr>
</table><p>After the table.</p>"""

    lines = eb.html_to_text(html).splitlines()

    assert lines == ["| GR-1 (Ours) | ABCD→D | 4.21", "After the table."]


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_to_stored_value_turns_percent_success_rates_into_fractions():
    assert eb.to_stored_value("LIBERO_Spatial", "84.7") == 0.847
    assert eb.to_stored_value("LIBERO_Spatial", "84.7%") == 0.847
    assert eb.to_stored_value("LIBERO_Spatial", "0.847") == 0.847
    assert eb.to_stored_value("CALVIN_ACL", "4.21") == 4.21
    assert eb.to_stored_value("CALVIN_ACL", "n/a") is None


def test_appears_in_requires_the_exact_number():
    assert eb.appears_in(PAPER, "4.21")
    assert eb.appears_in(PAPER, "84.7%")
    assert not eb.appears_in(PAPER, "4.2")    # prefix of 4.21
    assert not eb.appears_in(PAPER, "21")     # tail of 4.21
    assert not eb.appears_in(PAPER, "4.5")


@pytest.mark.parametrize("setting", ["ABCD→D", "ABCD -> D", "ABCD (Lang) → D", "train ABCD, test D"])
def test_normalize_setting_accepts_common_spellings(setting):
    assert eb.normalize_setting(setting) == "ABCD→D"


def test_classify_accepts_own_verified_score():
    accepted, uncertain = eb.classify([_score()], PAPER)

    assert [(f.benchmark, f.value) for f in accepted] == [("CALVIN_ACL", 4.21)]
    assert uncertain == []


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"is_own_model": False, "row_label": "HULC", "reported_value": "3.06"}, "다른 모델"),
        ({"reported_value": "4.35"}, "같은 숫자를 찾지 못함"),
        ({"setting": "ABC→D", "reported_value": "3.06"}, "평가 설정"),
        ({"confident": False, "note": "two variants"}, "확신하지 못함: two variants"),
        ({"reported_value": "n/a"}, "숫자로 읽을 수 없음"),
    ],
)
def test_classify_moves_doubtful_scores_to_uncertain(overrides, reason):
    accepted, uncertain = eb.classify([_score(**overrides)], PAPER)

    assert accepted == []
    assert len(uncertain) == 1
    assert reason in uncertain[0].reason


def test_classify_rejects_number_from_another_benchmarks_table():
    # Seen with RoboVLMs: SimplerEnv WidowX per-task rates filed under LIBERO
    paper = "Extended Table 3: SimplerEnv WidowX+Bridge.\n| RoboVLMs (Ours) | 0.458 | 0.417\n"
    raw = [_score(benchmark="LIBERO_Spatial", reported_value="0.458", setting="WidowX+Bridge")]

    accepted, uncertain = eb.classify(raw, paper)

    assert accepted == []
    assert "LIBERO 표기가 없음" in uncertain[0].reason


def test_near_benchmark_name_looks_only_around_the_row():
    far = "CALVIN results.\n" + "filler\n" * 40 + "| Ours | 4.21\n"

    assert eb.near_benchmark_name(PAPER, "4.21", "CALVIN_ACL")
    assert not eb.near_benchmark_name(far, "4.21", "CALVIN_ACL")


def test_classify_rejects_value_outside_scale():
    # 88.4 → 0.884 is fine for LIBERO, but CALVIN keeps raw values and 88.4 > 5
    accepted, uncertain = eb.classify([_score(reported_value="88.4")], PAPER)

    assert accepted == []
    assert "허용 범위" in uncertain[0].reason


def test_classify_drops_conflicting_values_for_one_benchmark():
    raw = [_score(), _score(reported_value="3.06", row_label="GR-1 (Ours) w/o pretraining")]

    accepted, uncertain = eb.classify(raw, PAPER)

    assert accepted == []
    assert {f.reason for f in uncertain} == {"같은 벤치마크에 서로 다른 값이 여러 개"}


def test_classify_ignores_unknown_benchmarks():
    accepted, uncertain = eb.classify([_score(benchmark="RoboCasa")], PAPER)

    assert accepted == [] and uncertain == []


def test_extract_for_model_reports_missing_key_without_raising(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = eb.extract_for_model({"id": "gr1", "paper_url": "https://arxiv.org/abs/2312.13139"})

    assert result.error == "GEMINI_API_KEY가 설정되지 않음"
    assert result.accepted == []


def test_extract_for_model_turns_fetch_errors_into_result(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")

    def broken_fetch(url):
        raise RuntimeError("timeout")

    monkeypatch.setattr(eb, "fetch_paper_text", broken_fetch)

    result = eb.extract_for_model({"id": "gr1", "paper_url": "https://arxiv.org/abs/2312.13139"})

    assert result.error == "추출 실패: timeout"


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

MODELS_YAML = """- id: openvla
  name: OpenVLA
  paper_url: https://arxiv.org/abs/2406.09246
  hf_url: https://huggingface.co/openvla/openvla-7b
  tags:
  - VLA
- id: gr1
  name: GR-1
  paper_url: https://arxiv.org/abs/2312.13139
  hf_url: ''
"""


def test_insert_benchmarks_only_adds_lines_after_paper_url():
    updated = eb.insert_benchmarks(MODELS_YAML, "openvla", {"LIBERO_Spatial": 0.847, "LIBERO_Long": 0.537})

    original_lines = MODELS_YAML.splitlines()
    new_lines = updated.splitlines()
    assert new_lines[3:6] == ["  benchmarks:", "    LIBERO_Spatial: 0.847", "    LIBERO_Long: 0.537"]
    assert new_lines[:3] + new_lines[6:] == original_lines
    assert yaml.safe_load(updated)[0]["benchmarks"] == {"LIBERO_Spatial": 0.847, "LIBERO_Long": 0.537}


def test_insert_benchmarks_appends_to_an_existing_block():
    once = eb.insert_benchmarks(MODELS_YAML, "openvla", {"LIBERO_Spatial": 0.847})

    twice = eb.insert_benchmarks(once, "openvla", {"LIBERO_Long": 0.537})

    assert twice.splitlines()[3:6] == ["  benchmarks:", "    LIBERO_Spatial: 0.847", "    LIBERO_Long: 0.537"]
    assert twice.splitlines()[6] == "  hf_url: https://huggingface.co/openvla/openvla-7b"


def test_insert_benchmarks_refuses_to_overwrite():
    once = eb.insert_benchmarks(MODELS_YAML, "gr1", {"CALVIN_ACL": 4.21})

    with pytest.raises(ValueError, match="already has a 'CALVIN_ACL' score"):
        eb.insert_benchmarks(once, "gr1", {"CALVIN_ACL": 4.09})
    with pytest.raises(ValueError, match="not found"):
        eb.insert_benchmarks(MODELS_YAML, "missing", {"CALVIN_ACL": 4.21})


def test_render_report_lists_sources_and_sanitizes_llm_text():
    result = eb.ExtractionResult(model_id="gr1", source_url="https://arxiv.org/html/2312.13139")
    result.accepted, result.uncertain = eb.classify(
        [_score(), _score(benchmark="LIBERO_Goal", reported_value="79.2", table="Tab|le @someone")], PAPER
    )

    report = eb.render_report([result])

    assert "| CALVIN_ACL | 4.21 | 4.21 ABCD→D | Table 1 · GR-1 (Ours) |" in report
    assert "Tab\\|le @\u200bsomeone" in report
    assert "같은 숫자를 찾지 못함" in report


def test_main_missing_fills_only_models_without_benchmarks(tmp_path, monkeypatch, capsys):
    path = tmp_path / "models.yaml"
    path.write_text(eb.insert_benchmarks(MODELS_YAML, "openvla", {"LIBERO_Goal": 0.792}), encoding="utf-8")
    monkeypatch.setattr(eb, "MODELS_PATH", path)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(eb, "fetch_paper_text", lambda url: (PAPER, "https://arxiv.org/html/x"))
    seen = []

    def fake_llm(name, text, **kwargs):
        seen.append(name)
        return {"scores": [_score()], "other_benchmarks": []}

    monkeypatch.setattr(eb, "ask_llm", fake_llm)

    assert eb.main(["--missing"]) == 0

    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert seen == ["GR-1"]
    assert entries[1]["benchmarks"] == {"CALVIN_ACL": 4.21}
    assert entries[0]["benchmarks"] == {"LIBERO_Goal": 0.792}
    assert "**gr1**" in capsys.readouterr().out


def test_main_benchmark_fills_one_new_benchmark_without_touching_existing_scores(tmp_path, monkeypatch):
    # openvla already has LIBERO_Goal; a newly defined benchmark should still be checked for it
    path = tmp_path / "models.yaml"
    path.write_text(eb.insert_benchmarks(MODELS_YAML, "openvla", {"LIBERO_Goal": 0.792}), encoding="utf-8")
    monkeypatch.setattr(eb, "MODELS_PATH", path)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(eb, "fetch_paper_text", lambda url: (PAPER, "https://arxiv.org/html/x"))
    llm_scores = {
        # CALVIN is found too, but only the requested benchmark may be written
        "OpenVLA": [_score(benchmark="LIBERO_Spatial", reported_value="84.7", setting=""),
                    _score(benchmark="CALVIN_ACL")],
        "GR-1": [_score()],
    }
    monkeypatch.setattr(
        eb, "ask_llm", lambda name, text, **kw: {"scores": llm_scores[name], "other_benchmarks": []}
    )

    assert eb.main(["--benchmark", "LIBERO_Spatial"]) == 0

    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert entries[0]["benchmarks"] == {"LIBERO_Goal": 0.792, "LIBERO_Spatial": 0.847}
    assert "benchmarks" not in entries[1]


def test_main_rejects_undefined_benchmark_option(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "test")

    assert eb.main(["--benchmark", "NEWBENCH"]) == 1
    assert "Unknown benchmark" in capsys.readouterr().err


def test_undefined_benchmarks_skips_defined_names_and_duplicates():
    raw = [
        {"name": "RoboCasa", "table": "Table 3", "is_public_benchmark": True},
        {"name": "robocasa ", "table": "Table 5", "is_public_benchmark": True},
        {"name": "LIBERO-90", "table": "Table 2", "is_public_benchmark": True},
        {"name": "CALVIN ABC→D", "table": "Table 1", "is_public_benchmark": True},
        {"name": "Franka Real-World  Evaluation", "table": "Table 6", "is_public_benchmark": False},
        {"name": "NEWBENCH", "table": "", "is_public_benchmark": True},
        {"name": "", "table": "Table 9", "is_public_benchmark": True},
    ]

    assert eb.undefined_benchmarks(raw) == [("RoboCasa", "Table 3"), ("NEWBENCH", "")]


def test_render_report_lists_undefined_benchmarks_even_without_scores():
    result = eb.ExtractionResult(model_id="groot", other_benchmarks=[("RoboCasa", "Table 3"), ("NEWBENCH", "")])

    report = eb.render_report([result])

    assert "- 정의된 벤치마크 결과를 찾지 못했습니다." in report
    assert "- 정의되지 않은 벤치마크: RoboCasa (Table 3), NEWBENCH" in report


# ─────────────────────────────────────────────────────────────────────────────
# process_issue integration
# ─────────────────────────────────────────────────────────────────────────────

def test_attach_benchmarks_places_scores_after_paper_url(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(report))
    accepted, _ = eb.classify([_score()], PAPER)
    monkeypatch.setattr(pi, "extract_for_model", lambda entry: eb.ExtractionResult("gr1", accepted=accepted))

    entry = pi.attach_benchmarks({"id": "gr1", "paper_url": "https://arxiv.org/abs/2312.13139", "hf_url": ""})

    assert list(entry) == ["id", "paper_url", "benchmarks", "hf_url"]
    assert entry["benchmarks"] == {"CALVIN_ACL": 4.21}
    assert "CALVIN_ACL" in report.read_text(encoding="utf-8")


def test_attach_benchmarks_keeps_entry_when_extraction_fails(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    monkeypatch.setenv("BENCHMARK_REPORT_PATH", str(report))
    monkeypatch.setattr(pi, "extract_for_model", lambda entry: eb.ExtractionResult("gr1", error="추출 실패: timeout"))
    entry = {"id": "gr1", "paper_url": "https://arxiv.org/abs/2312.13139"}

    assert pi.attach_benchmarks(entry) == entry
    assert "추출 실패: timeout" in report.read_text(encoding="utf-8")


def test_attach_benchmarks_survives_unexpected_errors(monkeypatch):
    def boom(entry):
        raise KeyError("bug")

    monkeypatch.setattr(pi, "extract_for_model", boom)
    entry = {"id": "gr1"}

    assert pi.attach_benchmarks(entry) == entry
