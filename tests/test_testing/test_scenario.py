"""Unit tests for Scenario validation and the YAML loader."""

import os

import pytest

from vision_agents.testing import Scenario, load_scenario

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_assets")


@pytest.fixture
def scenario_data() -> dict:
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

    def test_non_text_mode_rejected(self, scenario_data):
        scenario_data["mode"] = "voice"
        with pytest.raises(ValueError, match="'mode'"):
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
