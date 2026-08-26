"""
Unit tests for scripts/generate_site.py

Run with:
  pytest tests/test_generate_site.py -v
"""

import json
import pytest
import yaml
import generate_site as gs
import validate_data as vd


# ─────────────────────────────────────────────────────────────────────────────
# Sample fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_model():
    return {
        "id": "test-model",
        "name": "Test Model",
        "org": "Test Org",
        "year": 2024,
        "categories": ["manipulation"],
        "hardware": ["manipulator"],
        "learning": ["IL"],
        "github_url": "https://github.com/test/repo",
        "paper_url": "https://arxiv.org/abs/0000.00000",
        "hf_url": "",
        "project_url": "",
        "tags": ["tag-a"],
        "stats": {"github_stars": 1234, "github_forks": 100, "hf_downloads": 0},
    }


@pytest.fixture
def sample_dataset():
    return {
        "id": "test-dataset",
        "name": "Test Dataset",
        "org": "Test Org",
        "year": 2023,
        "categories": ["manipulation"],
        "source": ["real"],
        "modality": ["rgb"],
        "github_url": "https://github.com/test/dataset",
        "paper_url": "https://arxiv.org/abs/1111.11111",
        "hf_url": "",
        "project_url": "",
        "scale": {"trajectories": 5000},
        "stats": {"github_stars": 500, "github_forks": 50, "hf_downloads": 0},
    }


@pytest.fixture
def sample_tool():
    return {
        "id": "test-sim",
        "name": "Test Sim",
        "org": "Test Org",
        "year": 2022,
        "type": "physics_engine",
        "gpu_accelerated": True,
        "ros_support": False,
        "language": ["python", "c++"],
        "github_url": "https://github.com/test/sim",
        "paper_url": "https://arxiv.org/abs/2222.22222",
        "project_url": "",
        "stats": {"github_stars": 8000},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def test_badge_format():
    result = gs.badge("Stars", "42", "yellow")
    assert "Stars" in result
    assert "42" in result
    assert "yellow" in result
    assert result.startswith("![")


def test_badge_hyphen_escaping():
    result = gs.badge("sim-to-real", "value", "blue")
    assert "sim--to--real" in result


def test_stars_badge_thousands():
    result = gs.stars_badge(1500)
    assert "1.5k" in result


def test_stars_badge_ten_thousand_plus():
    result = gs.stars_badge(12000)
    assert "12k%2B" in result


def test_stars_badge_small():
    result = gs.stars_badge(42)
    assert "42" in result


def test_link_with_url():
    result = gs.link("GitHub", "https://github.com/test")
    assert result == "[GitHub](https://github.com/test)"


def test_link_without_url():
    result = gs.link("GitHub", "")
    assert result == "GitHub"
    assert "[" not in result


def test_tag_str():
    result = gs.tag_str(["foo", "bar"])
    assert "`foo`" in result
    assert "`bar`" in result


# ─────────────────────────────────────────────────────────────────────────────
# model_row
# ─────────────────────────────────────────────────────────────────────────────

def test_model_row_contains_name(sample_model):
    row = gs.model_row(sample_model)
    assert "Test Model" in row


def test_model_row_contains_org(sample_model):
    row = gs.model_row(sample_model)
    assert "Test Org" in row


def test_model_row_contains_stars(sample_model):
    row = gs.model_row(sample_model)
    assert "1,234" in row


def test_model_row_contains_paper_link(sample_model):
    row = gs.model_row(sample_model)
    assert "📄" in row


def test_model_row_no_hf_when_empty(sample_model):
    row = gs.model_row(sample_model)
    assert "🤗" not in row


# ─────────────────────────────────────────────────────────────────────────────
# dataset_row
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_row_contains_name(sample_dataset):
    row = gs.dataset_row(sample_dataset)
    assert "Test Dataset" in row


def test_dataset_row_contains_trajectories(sample_dataset):
    row = gs.dataset_row(sample_dataset)
    assert "5,000" in row


def test_dataset_row_no_trajectories_when_zero(sample_dataset):
    sample_dataset["scale"] = {"trajectories": 0}
    row = gs.dataset_row(sample_dataset)
    assert "—" in row


# ─────────────────────────────────────────────────────────────────────────────
# tool_row
# ─────────────────────────────────────────────────────────────────────────────

def test_tool_row_contains_name(sample_tool):
    row = gs.tool_row(sample_tool)
    assert "Test Sim" in row


def test_tool_row_contains_type(sample_tool):
    row = gs.tool_row(sample_tool)
    assert "physics_engine" in row


def test_tool_row_gpu_checkmark(sample_tool):
    row = gs.tool_row(sample_tool)
    assert "✅" in row


def test_tool_row_no_ros(sample_tool):
    row = gs.tool_row(sample_tool)
    # gpu_accelerated=True → ✅, ros_support=False → —
    assert "—" in row


def test_tool_row_contains_stars(sample_tool):
    row = gs.tool_row(sample_tool)
    assert "8,000" in row


# ─────────────────────────────────────────────────────────────────────────────
# generate_readme
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_readme_has_model_section(sample_model, sample_dataset, sample_tool):
    readme = gs.generate_readme([sample_model], [sample_dataset], [sample_tool])
    assert "## 🤖 Models" in readme


def test_generate_readme_has_dataset_section(sample_model, sample_dataset, sample_tool):
    readme = gs.generate_readme([sample_model], [sample_dataset], [sample_tool])
    assert "## 📦 Datasets" in readme


def test_generate_readme_has_tools_section(sample_model, sample_dataset, sample_tool):
    readme = gs.generate_readme([sample_model], [sample_dataset], [sample_tool])
    assert "## 🔬 Simulators" in readme


def test_generate_readme_badge_counts(sample_model, sample_dataset, sample_tool):
    readme = gs.generate_readme([sample_model], [sample_dataset], [sample_tool])
    assert "Models-1-blue" in readme
    assert "Datasets-1-green" in readme
    assert "Simulators-1-purple" in readme


def test_generate_readme_sorted_by_stars():
    low = {"id": "low", "name": "Low", "org": "O", "year": 2023, "categories": [],
           "hardware": [], "learning": [], "github_url": "https://github.com/a/a",
           "paper_url": "", "hf_url": "", "project_url": "", "stats": {"github_stars": 100}}
    high = {"id": "high", "name": "High", "org": "O", "year": 2023, "categories": [],
            "hardware": [], "learning": [], "github_url": "https://github.com/b/b",
            "paper_url": "", "hf_url": "", "project_url": "", "stats": {"github_stars": 9000}}
    readme = gs.generate_readme([low, high], [], [])
    # High-star model should appear before low-star model in table
    assert readme.index("High") < readme.index("Low")


# ─────────────────────────────────────────────────────────────────────────────
# generate_data_json
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_data_json_has_required_keys(sample_model, sample_dataset, sample_tool):
    result = gs.generate_data_json([sample_model], [sample_dataset], [sample_tool])
    assert "metadata" in result
    assert "models" in result
    assert "datasets" in result
    assert "tools" in result


def test_generate_data_json_counts(sample_model, sample_dataset, sample_tool):
    result = gs.generate_data_json([sample_model], [sample_dataset], [sample_tool])
    assert result["metadata"]["total_models"] == 1
    assert result["metadata"]["total_datasets"] == 1
    assert result["metadata"]["total_tools"] == 1


def test_generate_data_json_is_serializable(sample_model, sample_dataset, sample_tool):
    result = gs.generate_data_json([sample_model], [sample_dataset], [sample_tool])
    # Should not raise
    json.dumps(result, ensure_ascii=False)


def test_generate_data_json_org_count(sample_model, sample_dataset, sample_tool):
    # All share "Test Org" → total_orgs should be 1
    result = gs.generate_data_json([sample_model], [sample_dataset], [sample_tool])
    assert result["metadata"]["total_orgs"] == 1


def test_generate_data_json_empty_inputs():
    result = gs.generate_data_json([], [], [])
    assert result["metadata"]["total_models"] == 0
    assert result["metadata"]["total_datasets"] == 0
    assert result["metadata"]["total_tools"] == 0
    assert result["models"] == []
    assert result["datasets"] == []
    assert result["tools"] == []


def test_generate_site_reuses_benchmark_meta_source():
    assert gs.BENCHMARK_META is vd.BENCHMARK_META


def test_generate_site_main_writes_benchmark_meta_at_root(
    tmp_path, monkeypatch, sample_model, sample_dataset, sample_tool
):
    data_dir = tmp_path / "data"
    docs_dir = tmp_path / "docs"
    readme_path = tmp_path / "README.md"
    data_dir.mkdir()

    for filename, entries in {
        "models.yaml": [sample_model],
        "datasets.yaml": [sample_dataset],
        "tools.yaml": [sample_tool],
    }.items():
        (data_dir / filename).write_text(
            yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8"
        )

    monkeypatch.setattr(gs, "DATA_DIR", data_dir)
    monkeypatch.setattr(gs, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(gs, "README_PATH", readme_path)

    gs.main()

    payload = json.loads((docs_dir / "data.json").read_text(encoding="utf-8"))
    assert payload["benchmark_meta"] == vd.BENCHMARK_META
    assert set(payload["benchmark_meta"]) == set(vd.BENCHMARK_META)
    assert "benchmark_meta" not in payload["metadata"]
