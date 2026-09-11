from pathlib import Path

import pytest
from vision_agents.plugins.stream.folder import find, load, resolve


def write(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestFolder:
    def test_named_workers_and_skill_binding_round_trip(self, tmp_path):
        write(
            tmp_path,
            "agent.yaml",
            "name: vision\nsubagents:\n  default: llm-thinking\n  vision: vlm\n",
        )
        write(
            tmp_path,
            "skills/vision.md",
            "---\nname: vision\nsubagent: vision\ncapture_video: true\ndescription: inspect images\n---\nDescribe the evidence.",
        )
        folder = load(tmp_path)
        assert folder.settings.subagents == {"default": "llm-thinking", "vision": "vlm"}
        assert folder.skills[0].subagent == "vision"
        assert folder.skills[0].capture_video

    def test_a_directory_is_read_as_instructions_skills_and_knowledge(
        self, tmp_path: Path
    ):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(root, "instructions.md", "You are Jean.\n")
        write(
            root,
            "skills/think.md",
            "---\ndescription: Work something out before answering\ndeadline: 30s\n---\n"
            "Take your time and reason it through.\n",
        )
        write(root, "knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n")

        folder = load(root)

        assert folder.name == "jean"
        assert folder.instructions == "You are Jean."
        assert len(folder.skills) == 1
        assert folder.skills[0].name == "think"
        assert folder.skills[0].description == "Work something out before answering"
        assert folder.skills[0].deadline_seconds == 30
        assert folder.skills[0].instructions == "Take your time and reason it through."
        assert folder.knowledge[0].source == "pricing.md"
        assert folder.knowledge_namespace() == "jean"

    def test_the_same_files_hash_the_same(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(root, "instructions.md", "You are Jean.\n")
        write(root, "knowledge/pricing.md", "# Pricing\n")

        first = load(root).hash()
        second = load(root).hash()
        assert first == second

        write(root, "instructions.md", "You are someone else.\n")
        assert load(root).hash() != first

    def test_editing_the_declaration_changes_the_hash(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(root, "instructions.md", "You are Jean.\n")
        first = load(root).hash()

        write(root, "agent.yaml", "name: jean\ndescription: the receptionist\n")
        assert load(root).hash() != first

    def test_the_declaration_names_the_agent(self, tmp_path: Path):
        root = tmp_path / "jean-the-agent"
        write(root, "agent.yaml", "name: jean\n")

        assert load(root).name == "jean"

    def test_the_declaration_says_what_the_agent_runs_on(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(
            root,
            "agent.yaml",
            "name: jean\n"
            "mode: text\n"
            "llm: llm-fast\n"
            "subagent: llm-thinking\n"
            "sandbox: daytona\n"
            "greeting: Hello.\n"
            "keyterms:\n  - Vision Agents\n  - ''\n"
            "tags:\n  team: support\n",
        )

        settings = load(root).settings

        assert settings.mode == "text"
        assert settings.llm == "llm-fast"
        assert settings.subagent == "llm-thinking"
        assert settings.sandbox == "daytona"
        assert settings.greeting == "Hello."
        assert settings.keyterms == ["Vision Agents"]
        assert settings.tags == {"team": "support"}

    def test_a_declaration_that_names_no_model_decides_nothing(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\ndescription: the receptionist\n")

        settings = load(root).settings

        assert settings.llm == ""
        assert settings.sandbox == ""
        assert settings.plugins == []
        assert settings.tags == {}

    def test_a_key_nobody_knows_is_refused(self, tmp_path: Path):
        # A misspelled llm that went quietly would leave the agent running on a model the
        # file does not name.
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\nlmm: llm-fast\n")

        with pytest.raises(ValueError, match="lmm"):
            load(root)

    def test_video_selection_is_read_from_a_nested_block(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(
            root,
            "agent.yaml",
            "name: jean\nvideo:\n  source: roboflow\n  max_frames: 2\n",
        )

        assert load(root).settings.video_source == "roboflow"
        assert load(root).settings.video_max_frames == 2

    def test_a_list_setting_given_as_one_word_is_refused(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\nkeyterms: Vision Agents\n")

        with pytest.raises(ValueError, match="keyterms"):
            load(root)

    def test_a_directory_without_a_name_is_called_after_itself(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "description: the receptionist\n")

        assert load(root).name == "jean"

    def test_a_skill_without_a_description_is_refused(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(
            root,
            "skills/think.md",
            "Just a body, with nothing saying when to use it.\n",
        )

        with pytest.raises(ValueError, match="description"):
            load(root)

    def test_nested_knowledge_keeps_the_path_it_was_found_at(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "knowledge/reference/api.md", "# API\n\nthe endpoints\n")
        write(root, "knowledge/logo.png", "not a document")
        write(root, "knowledge/empty.md", "   \n")

        folder = load(root)

        assert [document.source for document in folder.knowledge] == [
            "reference/api.md"
        ]

    def test_resolve_finds_examples_voice_agents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = tmp_path / "examples" / "voice_agents" / "support"
        write(agent, "agent.yaml", "name: support\n")
        write(agent, "instructions.md", "Help.\n")
        monkeypatch.chdir(tmp_path)

        assert resolve("support") == agent.resolve()

    def test_resolve_finds_an_agent_filed_under_another_kind_of_example(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = tmp_path / "examples" / "text_agents" / "docs_agent"
        write(agent, "agent.yaml", "name: docs_agent\n")
        write(agent, "instructions.md", "Answer from the docs.\n")
        monkeypatch.chdir(tmp_path)

        assert resolve("docs_agent") == agent.resolve()

    def test_a_directory_without_a_declaration_is_not_an_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Instructions alone no longer make a directory an agent: agent.yaml is what says
        # so, and without it a name belongs to whatever is stored on the router.
        write(
            tmp_path / "examples" / "voice_agents" / "support",
            "instructions.md",
            "Help.\n",
        )
        monkeypatch.chdir(tmp_path)

        assert find("support") is None
        with pytest.raises(FileNotFoundError, match="agent.yaml"):
            resolve("support")
