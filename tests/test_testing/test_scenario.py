"""Unit tests for Scenario validation and the YAML loader."""

import os
from typing import Any

import pytest

from pathlib import Path

from vision_agents.testing import Scenario, find_scenarios, load_scenario

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_assets")


@pytest.fixture
def scenario_data() -> dict[str, Any]:
    return {
        "name": "reschedule-appointment",
        "mode": "text",
        "persona": {"impatient": True, "interruptions": "frequent"},
        "context": {"name": "Alice", "appointment": "Tuesday 3pm"},
        "goal": "Move appointment to Friday morning.",
        "constraints": ["Reject anything after 11am."],
        "success": ["appointment_rescheduled", "correct_time_confirmed"],
        "variations": 2,
        "repeat": 3,
        "judge_target": "gemini-3-flash-preview",
    }


class TestScenario:
    def test_loads_yaml_file(self):
        scenario = load_scenario(
            os.path.join(ASSETS_DIR, "scenarios", "reschedule-appointment.yaml")
        )
        assert scenario.name == "reschedule-appointment"
        assert scenario.mode == "text"
        assert scenario.persona == {"impatient": True, "interruptions": "frequent"}
        assert scenario.context == {"name": "Alice", "appointment": "Tuesday 3pm"}
        assert scenario.goal == "Move appointment to Friday morning."
        assert scenario.constraints == [
            "Don't volunteer confirmation ID unless asked.",
            "Reject anything after 11am.",
        ]
        assert scenario.success == [
            "appointment_rescheduled",
            "correct_time_confirmed",
            "identity_verified",
        ]
        assert scenario.variations == 1
        assert scenario.repeat == 1
        assert scenario.judge_target == "gemini-3-flash-preview"

    def test_from_dict_reads_all_fields(self, scenario_data):
        scenario = Scenario.from_dict(scenario_data)
        assert scenario.variations == 2
        assert scenario.repeat == 3
        assert scenario.judge_target == "gemini-3-flash-preview"

    def test_optional_fields_default(self):
        scenario = Scenario.from_dict({"name": "s", "goal": "g", "success": ["done"]})
        assert scenario.mode == "text"
        assert scenario.persona == {}
        assert scenario.context == {}
        assert scenario.constraints == []
        assert scenario.variations == 1
        assert scenario.repeat == 1
        assert scenario.judge_target is None

    @pytest.mark.parametrize("missing", ["name", "goal", "success"])
    def test_missing_required_field_names_it(self, scenario_data, missing):
        del scenario_data[missing]
        with pytest.raises(ValueError, match=f"missing required field.*'{missing}'"):
            Scenario.from_dict(scenario_data)

    def test_unknown_field_names_it(self, scenario_data):
        scenario_data["personna"] = {"impatient": True}
        with pytest.raises(ValueError, match="Unknown scenario field.*'personna'"):
            Scenario.from_dict(scenario_data)

    def test_unknown_non_string_key_reported(self, scenario_data):
        scenario_data[3] = "x"
        scenario_data["extra"] = "y"
        with pytest.raises(
            ValueError, match=r"Unknown scenario field\(s\): 'extra', 3"
        ):
            Scenario.from_dict(scenario_data)

    def test_unknown_mode_rejected(self, scenario_data):
        scenario_data["mode"] = "voice"
        with pytest.raises(ValueError, match="'mode'.*'audio'.*'text'"):
            Scenario.from_dict(scenario_data)

    def test_audio_mode_picks_caller_providers(self, scenario_data):
        scenario_data["mode"] = "audio"
        scenario_data["caller_tts"] = "elevenlabs"
        scenario_data["caller_stt"] = "deepgram"
        scenario = Scenario.from_dict(scenario_data)
        assert scenario.mode == "audio"
        assert scenario.caller_tts == "elevenlabs"
        assert scenario.caller_stt == "deepgram"

    def test_caller_providers_default_to_none(self, scenario_data):
        scenario = Scenario.from_dict(scenario_data)
        assert scenario.caller_tts is None
        assert scenario.caller_stt is None

    @pytest.mark.parametrize("field_name", ["caller_tts", "caller_stt"])
    def test_caller_providers_must_be_strings(self, scenario_data, field_name):
        scenario_data[field_name] = ["elevenlabs"]
        with pytest.raises(ValueError, match=f"'{field_name}'"):
            Scenario.from_dict(scenario_data)

    def test_empty_success_rejected(self, scenario_data):
        scenario_data["success"] = []
        with pytest.raises(ValueError, match="'success'"):
            Scenario.from_dict(scenario_data)

    def test_success_must_be_string_list(self, scenario_data):
        scenario_data["success"] = {"appointment_rescheduled": True}
        with pytest.raises(ValueError, match="'success'"):
            Scenario.from_dict(scenario_data)

    @pytest.mark.parametrize("field_name", ["variations", "repeat"])
    @pytest.mark.parametrize("value", [0, -1, "2", True])
    def test_counts_must_be_positive_ints(self, scenario_data, field_name, value):
        scenario_data[field_name] = value
        with pytest.raises(ValueError, match=f"'{field_name}'"):
            Scenario.from_dict(scenario_data)

    def test_context_values_must_be_scalars(self, scenario_data):
        scenario_data["context"]["tags"] = ["a", "b"]
        with pytest.raises(ValueError, match="'context'.*'tags'"):
            Scenario.from_dict(scenario_data)

    def test_empty_values_count_as_not_provided(self, tmp_path):
        path = tmp_path / "sparse.yaml"
        path.write_text(
            "name: s\ngoal: g\nsuccess:\n  - done\nconstraints:\npersona:\n"
        )
        scenario = load_scenario(path)
        assert scenario.constraints == []
        assert scenario.persona == {}

    def test_empty_required_value_reported_as_missing(self, scenario_data):
        scenario_data["goal"] = None
        with pytest.raises(ValueError, match="missing required field.*'goal'"):
            Scenario.from_dict(scenario_data)

    def test_non_mapping_file_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a list\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_scenario(path)

    def test_brief_includes_goal_facts_and_constraints(self, scenario_data):
        brief = Scenario.from_dict(scenario_data).brief
        assert "Goal: Move appointment to Friday morning." in brief
        assert "- impatient: True" in brief
        assert "- appointment: Tuesday 3pm" in brief
        assert "- Reject anything after 11am." in brief


class TestFindScenarios:
    def test_file_is_returned_as_is(self, tmp_path: Path):
        path = tmp_path / "one.yaml"
        path.write_text("name: one\n")

        assert find_scenarios(path) == [path]

    def test_directory_lists_yaml_files_sorted(self, tmp_path: Path):
        for name in ("b.yaml", "a.yml", "notes.toml", "README.md"):
            (tmp_path / name).write_text("")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "c.yaml").write_text("")

        assert find_scenarios(tmp_path) == [tmp_path / "a.yml", tmp_path / "b.yaml"]

    def test_missing_target_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            find_scenarios(tmp_path / "nowhere")

    def test_load_error_names_the_file(self, tmp_path: Path):
        path = tmp_path / "broken.yaml"
        path.write_text("goal: x\n")

        with pytest.raises(ValueError, match="broken.yaml: .*missing required field"):
            load_scenario(path)

    def test_yaml_syntax_error_names_the_file(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text("name: [unclosed\n")

        with pytest.raises(ValueError, match="bad.yaml"):
            load_scenario(path)
