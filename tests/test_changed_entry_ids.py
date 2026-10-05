from scripts.changed_entry_ids import changed_entry_ids_from_diff


def test_detects_an_existing_entry_updated_without_changing_its_id():
    diff = """--- a/data/models.yaml
+++ b/data/models.yaml
@@ -3 +3 @@
-  tags: [old]
+  tags: [new]
"""
    head_files = {"data/models.yaml": "- id: model-a\n  name: A\n  tags: [new]\n"}

    assert changed_entry_ids_from_diff(diff, head_files) == ["model-a"]


def test_detects_added_entries_and_multiple_entries_in_one_hunk():
    diff = """--- a/data/models.yaml
+++ b/data/models.yaml
@@ -1,0 +1,4 @@
+- id: model-a
+  name: A
+- id: model-b
+  name: B
"""
    head_files = {
        "data/models.yaml": "- id: model-a\n  name: A\n- id: model-b\n  name: B\n"
    }

    assert changed_entry_ids_from_diff(diff, head_files) == ["model-a", "model-b"]
