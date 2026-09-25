#!/usr/bin/env python3
"""
validate_data.py — Schema validation for Awesome Physical AI data files.

Run automatically on every PR via GitHub Actions (validate-pr.yml).
Also useful locally before submitting a PR.

Usage:
  python scripts/validate_data.py
"""

import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

VALID_TOOL_TYPES = {"physics_engine", "rl_framework", "benchmark", "full_stack"}

REQUIRED_TOOL_KEYS = {
    "id", "name", "org", "year", "description_en", "description_ko",
    "type", "added_date",
}

VALID_CATEGORIES = {
    "manipulation", "locomotion", "navigation", "dexterous",
    "whole-body", "aerial", "underwater",
}
VALID_HARDWARE = {
    "manipulator", "humanoid", "quadruped", "biped",
    "mobile", "drone", "underwater", "hand",
}
VALID_LEARNING = {
    "VLA", "IL", "RL", "diffusion", "world_model", "sim2real",
    "meta-learning", "self-supervised",
}
VALID_FRAMEWORK = {"pytorch", "jax", "tensorflow", "mujoco", "isaacgym", "other"}
VALID_COMMUNICATION = {"ros2", "grpc", "lcm", "zenoh", "other"}
VALID_SOURCE = {"real", "simulation", "teleoperation", "human_demo", "mocap"}
VALID_MODALITY = {
    "rgb", "rgbd", "depth", "lidar", "tactile",
    "proprioception", "audio", "force_torque",
}

BENCHMARK_META = {
    "LIBERO_Spatial": {"metric": "success_rate",     "scale": [0, 1]},
    "LIBERO_Object":  {"metric": "success_rate",     "scale": [0, 1]},
    "LIBERO_Goal":    {"metric": "success_rate",     "scale": [0, 1]},
    "LIBERO_Long":    {"metric": "success_rate",     "scale": [0, 1]},
    "SimplerEnv_VM":  {"metric": "success_rate",     "scale": [0, 1]},
    "CALVIN_ACL":     {"metric": "avg_chain_length", "scale": [0, 5]},
}

REQUIRED_MODEL_KEYS = {
    "id", "name", "org", "year", "description_en", "description_ko",
    "categories", "hardware", "learning", "framework", "added_date",
}
REQUIRED_DATASET_KEYS = {
    "id", "name", "org", "year", "description_en", "description_ko",
    "categories", "hardware", "source", "modality", "added_date",
}

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)
    print(f"  ✗ {msg}")


def check_list_values(entry_id: str, field: str, values: list, valid_set: set) -> None:
    for v in values:
        if v not in valid_set:
            err(f"[{entry_id}] '{field}' has unknown value '{v}' (valid: {sorted(valid_set)})")


def check_benchmarks(model: dict) -> None:
    if "benchmarks" not in model:
        return

    entry_id = model.get("id", "<no-id>")
    benchmarks = model["benchmarks"]
    if not isinstance(benchmarks, dict):
        err(f"[{entry_id}] 'benchmarks' must be a mapping")
        return

    for key, value in benchmarks.items():
        if key not in BENCHMARK_META:
            err(
                f"[{entry_id}] 'benchmarks' has unknown key '{key}' "
                f"(valid: {sorted(BENCHMARK_META)})"
            )
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            err(f"[{entry_id}] benchmark '{key}' must be a number")
            continue

        scale = BENCHMARK_META[key]["scale"]
        if not scale[0] <= value <= scale[1]:
            err(
                f"[{entry_id}] benchmark '{key}' value {value} is outside "
                f"allowed range {scale}"
            )


def validate_model(entry: dict) -> None:
    entry_id = entry.get("id", "<no-id>")
    print(f"  Checking model: {entry_id}")

    # Required keys
    for key in REQUIRED_MODEL_KEYS:
        if key not in entry:
            err(f"[{entry_id}] Missing required field: '{key}'")

    # Year range
    year = entry.get("year")
    if year and not (2015 <= int(year) <= 2030):
        err(f"[{entry_id}] 'year' {year} looks wrong")

    # Controlled vocabularies
    check_list_values(entry_id, "categories", entry.get("categories", []), VALID_CATEGORIES)
    check_list_values(entry_id, "hardware", entry.get("hardware", []), VALID_HARDWARE)
    check_list_values(entry_id, "learning", entry.get("learning", []), VALID_LEARNING)
    check_list_values(entry_id, "framework", entry.get("framework", []), VALID_FRAMEWORK)
    check_list_values(entry_id, "communication", entry.get("communication", []), VALID_COMMUNICATION)
    check_benchmarks(entry)

    # At least one URL
    urls = [entry.get("github_url"), entry.get("paper_url"), entry.get("hf_url")]
    if not any(urls):
        err(f"[{entry_id}] Must have at least one of: github_url, paper_url, hf_url")


def validate_dataset(entry: dict) -> None:
    entry_id = entry.get("id", "<no-id>")
    print(f"  Checking dataset: {entry_id}")

    for key in REQUIRED_DATASET_KEYS:
        if key not in entry:
            err(f"[{entry_id}] Missing required field: '{key}'")

    year = entry.get("year")
    if year and not (2015 <= int(year) <= 2030):
        err(f"[{entry_id}] 'year' {year} looks wrong")

    check_list_values(entry_id, "categories", entry.get("categories", []), VALID_CATEGORIES)
    check_list_values(entry_id, "hardware", entry.get("hardware", []), VALID_HARDWARE)
    check_list_values(entry_id, "source", entry.get("source", []), VALID_SOURCE)
    check_list_values(entry_id, "modality", entry.get("modality", []), VALID_MODALITY)

    urls = [entry.get("github_url"), entry.get("paper_url"), entry.get("hf_url")]
    if not any(urls):
        err(f"[{entry_id}] Must have at least one of: github_url, paper_url, hf_url")


def validate_tool(entry: dict) -> None:
    entry_id = entry.get("id", "<no-id>")
    print(f"  Checking tool: {entry_id}")

    for key in REQUIRED_TOOL_KEYS:
        if key not in entry:
            err(f"[{entry_id}] Missing required field: '{key}'")

    year = entry.get("year")
    if year and not (2010 <= int(year) <= 2030):
        err(f"[{entry_id}] 'year' {year} looks wrong")

    tool_type = entry.get("type")
    if tool_type and tool_type not in VALID_TOOL_TYPES:
        err(f"[{entry_id}] 'type' has unknown value '{tool_type}' (valid: {sorted(VALID_TOOL_TYPES)})")

    urls = [entry.get("github_url"), entry.get("paper_url"), entry.get("project_url")]
    if not any(urls):
        err(f"[{entry_id}] Must have at least one of: github_url, paper_url, project_url")


def check_unique_ids(entries: list[dict], kind: str) -> None:
    seen: set[str] = set()
    for e in entries:
        eid = e.get("id", "")
        if eid in seen:
            err(f"[{kind}] Duplicate id: '{eid}'")
        seen.add(eid)


def main() -> int:
    print("=== Validating data/models.yaml ===")
    models_path = DATA_DIR / "models.yaml"
    if models_path.exists():
        with open(models_path, encoding="utf-8") as f:
            models = yaml.safe_load(f) or []
        check_unique_ids(models, "models")
        for m in models:
            validate_model(m)
    else:
        err("data/models.yaml not found")

    print("\n=== Validating data/datasets.yaml ===")
    datasets_path = DATA_DIR / "datasets.yaml"
    if datasets_path.exists():
        with open(datasets_path, encoding="utf-8") as f:
            datasets = yaml.safe_load(f) or []
        check_unique_ids(datasets, "datasets")
        for d in datasets:
            validate_dataset(d)
    else:
        err("data/datasets.yaml not found")

    print("\n=== Validating data/tools.yaml ===")
    tools_path = DATA_DIR / "tools.yaml"
    tools: list[dict] = []
    if tools_path.exists():
        with open(tools_path, encoding="utf-8") as f:
            tools = yaml.safe_load(f) or []
        check_unique_ids(tools, "tools")
        for t in tools:
            validate_tool(t)
    else:
        print("  (skipping — data/tools.yaml not found)")

    print()
    if errors:
        print(f"❌ Validation failed with {len(errors)} error(s):")
        for e in errors:
            print(f"   • {e}")
        return 1
    else:
        print(f"✅ All entries valid ({len(models)} models, {len(datasets)} datasets, {len(tools)} tools)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
