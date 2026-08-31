from pathlib import Path

import pytest
from vision_agents.plugins.stream.folder import load, resolve


def write(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestFolder:
    def test_a_directory_is_read_as_instructions_skills_and_knowledge(
        self, tmp_path: Path
    ):
        root = tmp_path / "jean"
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
        write(root, "instructions.md", "You are Jean.\n")
        write(root, "knowledge/pricing.md", "# Pricing\n")

        first = load(root).hash()
        second = load(root).hash()
        assert first == second

        write(root, "instructions.md", "You are someone else.\n")
        assert load(root).hash() != first

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

    def test_resolve_finds_examples_agents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = tmp_path / "examples" / "agents" / "support"
        write(agent, "instructions.md", "Help.\n")
        monkeypatch.chdir(tmp_path)

        assert resolve("support") == agent.resolve()
