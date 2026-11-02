import llm_reviewer as reviewer


def test_extract_json_object_handles_markdown_fence():
    data = reviewer.extract_json_object(
        """```json
{"decision": "needs_review", "entry_type": "paper_only"}
```"""
    )

    assert data["decision"] == "needs_review"
    assert data["entry_type"] == "paper_only"


def test_normalize_review_fills_required_fields():
    data = reviewer.normalize_review({"decision": "accept", "has_verified_model_link": True})

    assert data["status"] == "ok"
    assert data["decision"] == "accept"
    assert data["entry_type"] == "unclear"
    assert data["has_verified_model_link"] is True
    assert data["has_verified_artifact_link"] is False
    assert "maintainer_summary" in data


def test_normalize_review_clamps_unknown_values():
    data = reviewer.normalize_review(
        {
            "decision": "maybe",
            "entry_type": "other",
            "categories": ["manipulation", "invalid"],
            "hardware_targets": "manipulator",
        }
    )

    assert data["decision"] == "needs_review"
    assert data["entry_type"] == "unclear"
    assert data["categories"] == ["manipulation"]
    assert data["hardware_targets"] == []


def test_provider_model_can_change_without_code(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "provider/new-model")

    assert reviewer.provider_model("openrouter", None) == "provider/new-model"
    assert reviewer.provider_model("gemini", "gemini-explicit") == "gemini-explicit"


def test_provider_specific_model_is_fallback(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/provider-model")

    assert reviewer.provider_model("openrouter", None) == "openrouter/provider-model"


def test_review_candidate_dispatches_to_selected_provider(monkeypatch):
    monkeypatch.setattr(
        reviewer,
        "call_openrouter",
        lambda **kwargs: {"status": "ok", "provider": "openrouter", "model": kwargs["model"]},
    )
    monkeypatch.setattr(
        reviewer,
        "call_gemini",
        lambda **kwargs: {"status": "ok", "provider": "gemini", "model": kwargs["model"]},
    )

    assert reviewer.review_candidate(
        provider="openrouter", model="model-a", api_key="key", candidate={}
    )["provider"] == "openrouter"
    assert reviewer.review_candidate(
        provider="gemini", model="model-b", api_key="key", candidate={}
    )["provider"] == "gemini"


def test_openrouter_adapter_uses_requested_model(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"decision":"accept"}'}}]}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(reviewer.requests, "post", fake_post)
    result = reviewer.call_openrouter(api_key="key", model="org/new-model", candidate={"title": "Model"})

    assert captured["json"]["model"] == "org/new-model"
    assert result["decision"] == "accept"
