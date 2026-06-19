"""Unit tests for the ResearchAgent->DocentAgent output-key handoff (genre routing).

Regression test for B0.2 / AION-77a6f4c3: DocentAgent read the legacy
'individual_results'/'consolidated_results' keys while ResearchAgent.unified_research
emits 'research_attempts'/'consolidated_findings'. The mismatch made every research
result look invalid, so DocentAgent fell back to empty metadata and the 'Default'
genre for every artwork, defeating genre-aware VisionAgent routing.

These tests assert the handoff helpers read the current contract (with legacy
fallback) and preserve a non-Default genre.
"""
from docent_agent import DocentAgent


def test_consolidated_reads_current_key_and_preserves_genre():
    research_output = {
        "research_attempts": {"grok_image": {"success": True, "result": {}}},
        "consolidated_findings": {"genre": "Landscape", "title": "Wheatfield", "author": "van Gogh"},
    }
    consolidated = DocentAgent._research_consolidated(research_output)
    assert consolidated is not None
    assert consolidated.get("genre") == "Landscape"
    # the routing key DocentAgent derives must NOT collapse to Default
    assert (consolidated.get("genre") or "Default") != "Default"


def test_consolidated_legacy_key_still_supported():
    research_output = {"consolidated_results": {"genre": "Portrait"}}
    consolidated = DocentAgent._research_consolidated(research_output)
    assert consolidated is not None and consolidated.get("genre") == "Portrait"


def test_individual_reads_current_and_legacy_keys():
    assert DocentAgent._research_individual({"research_attempts": {"a": 1}}) == {"a": 1}
    assert DocentAgent._research_individual({"individual_results": {"b": 2}}) == {"b": 2}


def test_missing_or_malformed_inputs_are_safe():
    assert DocentAgent._research_consolidated({}) is None
    assert DocentAgent._research_consolidated("nope") is None
    assert DocentAgent._research_individual({}) == {}
    assert DocentAgent._research_individual(None) == {}
