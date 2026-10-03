from pathlib import Path

import pytest
from vision_agents.plugins.stream.folder import SandboxSettings, find, load, resolve


def write(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestFolder:
    def test_a_skill_that_captures_video_round_trips(self, tmp_path):
        write(tmp_path, "agent.yaml", "name: vision\nsubagent: vlm\n")
        write(
            tmp_path,
            "skills/vision.md",
            "---\nname: vision\ncapture_video: true\ndescription: inspect images\n---\nDescribe the evidence.",
        )
        folder = load(tmp_path)
        assert folder.settings.subagent == "vlm"
        assert folder.skills[0].capture_video

    def test_named_subagents_are_refused(self, tmp_path):
        write(tmp_path, "agent.yaml", "name: vision\nsubagents:\n  vision: vlm\n")
        with pytest.raises(ValueError, match="subagents"):
            load(tmp_path)

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

    def test_the_declaration_says_who_connects_each_plugin(self, tmp_path: Path):
        root = tmp_path / "triage"
        write(
            root,
            "agent.yaml",
            "name: triage\nplugins:\n  - sentry\nuser_plugins:\n  - google_calendar\n",
        )

        settings = load(root).settings

        assert settings.plugins == ["sentry"]
        assert settings.user_plugins == ["google_calendar"]

    def test_a_declaration_that_names_no_model_decides_nothing(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\ndescription: the receptionist\n")

        settings = load(root).settings

        assert settings.llm == ""
        assert settings.sandbox == ""
        assert settings.plugins == []
        assert settings.user_plugins == []
        assert settings.tags == {}

    def test_a_key_nobody_knows_is_refused(self, tmp_path: Path):
        # A misspelled llm that went quietly would leave the agent running on a model the
        # file does not name.
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\nlmm: llm-fast\n")

        with pytest.raises(ValueError, match="lmm"):
            load(root)

    def test_the_applications_own_section_is_left_to_it(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\napp:\n  sandbox_profile: support\n")

        assert load(root).settings.app == {"sandbox_profile": "support"}

    def test_video_selection_is_read_from_a_nested_block(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(
            root,
            "agent.yaml",
            "name: jean\nvideo:\n  source: roboflow\n  max_frames: 2\n",
        )

        assert load(root).settings.video_source == "roboflow"
        assert load(root).settings.video_max_frames == 2

    def test_dispatch_is_read_from_a_nested_block(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\ndispatch:\n  text: enabled\n")

        assert load(root).settings.dispatch == {"text": "enabled"}

    def test_an_unknown_dispatch_setting_is_refused(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\ndispatch:\n  txet: enabled\n")

        with pytest.raises(ValueError, match="txet"):
            load(root)

    def test_how_the_sandbox_is_built_is_read_from_a_nested_block(self, tmp_path: Path):
        root = tmp_path / "artist"
        write(
            root,
            "agent.yaml",
            "name: artist\nsandbox: daytona\nsandbox_options:\n"
            "  image: python:3.13-slim-bookworm\n"
            "  setup:\n    - pip install bpy==5.2.2\n"
            "  timeout: 5m\n  cpu: 2\n  memory_gb: 4\n",
        )

        options = load(root).settings.sandbox_options

        assert options == SandboxSettings(
            image="python:3.13-slim-bookworm",
            setup=["pip install bpy==5.2.2"],
            timeout_seconds=300,
            cpu=2,
            memory_gb=4,
        )

    def test_a_declaration_without_sandbox_options_has_none(self, tmp_path: Path):
        root = tmp_path / "analyst"
        write(root, "agent.yaml", "name: analyst\nsandbox: daytona\n")

        assert load(root).settings.sandbox_options is None

    @pytest.mark.parametrize(
        "options",
        ["  timeout: 2h\n", "  timeout: soon\n", "  memory: 4\n", "  cpu: two\n"],
    )
    def test_sandbox_options_nobody_can_honour_are_refused(
        self, tmp_path: Path, options: str
    ):
        root = tmp_path / "artist"
        write(root, "agent.yaml", "name: artist\nsandbox_options:\n" + options)

        with pytest.raises(ValueError, match="sandbox_options"):
            load(root)

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
        write(root, "agent.yaml", "name: jean\n")
        write(
            root,
            "skills/think.md",
            "Just a body, with nothing saying when to use it.\n",
        )

        with pytest.raises(ValueError, match="description"):
            load(root)

    def test_nested_knowledge_keeps_the_path_it_was_found_at(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(root, "knowledge/reference/api.md", "# API\n\nthe endpoints\n")
        write(root, "knowledge/logo.png", "not a document")
        write(root, "knowledge/empty.md", "   \n")

        folder = load(root)

        assert [document.source for document in folder.knowledge] == [
            "reference/api.md"
        ]

    def test_declared_pages_are_read_without_being_ingested(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(
            root,
            "knowledge/urls.yaml",
            "- https://example.com/pricing\n"
            "- url: https://example.com/plans\n"
            "  title: Plans\n"
            "  description: What each plan includes.\n",
        )
        write(root, "knowledge/reference/urls.yaml", "the urls we used to have\n")

        folder = load(root)

        assert [document.source for document in folder.knowledge] == [
            "reference/urls.yaml"
        ]
        assert [(page.url, page.title) for page in folder.knowledge_urls] == [
            ("https://example.com/pricing", ""),
            ("https://example.com/plans", "Plans"),
        ]
        assert folder.knowledge_urls[1].description == "What each plan includes."
        assert folder.knowledge_namespace() == "jean"

    @pytest.mark.parametrize(
        "declaration",
        [
            "- example.com/pricing\n",
            "- url: https://example.com/plans\n  heading: Plans\n",
            "- [https://example.com/plans]\n",
        ],
    )
    def test_a_page_that_cannot_be_fetched_or_described_is_refused(
        self, tmp_path: Path, declaration: str
    ):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(root, "knowledge/urls.yaml", declaration)

        with pytest.raises(ValueError):
            load(root)

    def test_declaring_a_page_changes_the_hash(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        before = load(root).hash()

        write(root, "knowledge/urls.yaml", "- https://example.com/plans\n")

        assert load(root).hash() != before

    def test_speed_and_harness_are_read_from_the_declaration(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\nspeed: 1.1\nharness: default\n")

        settings = load(root).settings

        assert settings.speed == 1.1
        assert settings.harness == "default"

    def test_a_speed_that_is_not_a_number_is_refused(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\nspeed: fast\n")

        with pytest.raises(ValueError, match="speed"):
            load(root)

    def test_a_page_may_be_read_again_on_a_schedule(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(
            root,
            "knowledge/urls.yaml",
            "- url: https://example.com/plans\n  refresh_hours: 24\n"
            "- https://example.com/pricing\n",
        )

        pages = load(root).knowledge_urls

        assert [page.refresh_hours for page in pages] == [24, 0]

    @pytest.mark.parametrize("hours", ["0", "1.5", "daily"])
    def test_a_schedule_that_is_not_a_whole_number_of_hours_is_refused(
        self, tmp_path: Path, hours: str
    ):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(
            root,
            "knowledge/urls.yaml",
            f"- url: https://example.com/plans\n  refresh_hours: {hours}\n",
        )

        with pytest.raises(ValueError, match="refresh_hours"):
            load(root)

    def test_simulations_are_read_from_every_file_in_name_order(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(
            root,
            "simulations/b.yml",
            "- name: refund\n  scenario: Ask for a refund.\n  assertion: None is promised.\n"
            "  mode: audio\n  variations: 3\n  tags:\n    team: support\n",
        )
        write(
            root,
            "simulations/a.yaml",
            "- name: lunch\n  scenario: Order lunch.\n  assertion: One wrap.\n",
        )
        write(root, "simulations/notes.md", "not a simulation\n")

        simulations = load(root).simulations

        assert simulations is not None
        assert [simulation.name for simulation in simulations] == ["lunch", "refund"]
        assert simulations[1].mode == "audio"
        assert simulations[1].variations == 3
        assert simulations[1].tags == {"team": "support"}

    def test_no_simulations_directory_is_none_and_an_empty_one_is_empty(
        self, tmp_path: Path
    ):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        assert load(root).simulations is None

        (root / "simulations").mkdir()
        assert load(root).simulations == []

    @pytest.mark.parametrize(
        "declaration, refused",
        [
            ("- name: a\n  scenario: s\n  assertion: x\n  judge: me\n", "judge"),
            ("- name: a\n  assertion: x\n", "scenario"),
            ("- name: a\n  scenario: s\n", "assertion"),
            ("- name: a\n  scenario: s\n  assertion: x\n  mode: video\n", "video"),
            (
                "- name: a\n  scenario: s\n  assertion: x\n  variations: many\n",
                "variations",
            ),
            ("name: a\n", "list"),
        ],
    )
    def test_a_simulation_that_cannot_be_run_is_refused(
        self, tmp_path: Path, declaration: str, refused: str
    ):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(root, "simulations/a.yaml", declaration)

        with pytest.raises(ValueError, match=refused):
            load(root)

    def test_two_simulations_cannot_share_a_name(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        simulation = "- name: lunch\n  scenario: s\n  assertion: a\n"
        write(root, "simulations/a.yaml", simulation)
        write(root, "simulations/b.yaml", simulation)

        with pytest.raises(ValueError, match="also declared in a.yaml"):
            load(root)

    def test_a_directory_without_schedules_or_simulations_keeps_its_hash(
        self, tmp_path: Path
    ):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(root, "knowledge/urls.yaml", "- https://example.com/plans\n")

        # What every SDK took this directory to be before either was declared.
        assert load(root).hash() == "bb5804bc853eaac855a30ec02037106a"

    def test_schedules_and_simulations_hash_the_way_the_go_sdk_does(
        self, tmp_path: Path
    ):
        root = tmp_path / "jean"
        write(root, "agent.yaml", "name: jean\n")
        write(
            root,
            "knowledge/urls.yaml",
            "- url: https://example.com/plans\n  refresh_hours: 24\n",
        )
        write(
            root,
            "simulations/lunch.yaml",
            "- name: lunch <order> & change\n"
            "  scenario: |\n    Order a turkey club, then swap it.\n"
            "  assertion: One veggie wrap.\n"
            "  variations: 3\n"
            "  tags:\n    b: two\n    a: one\n",
        )

        # What the Go SDK's agents.Load(...).Hash() gives the same files.
        assert load(root).hash() == "c55bc13d9e0e146d7facbe1db9e67774"

    def test_a_directory_without_a_declaration_cannot_be_loaded(self, tmp_path: Path):
        root = tmp_path / "jean"
        write(root, "instructions.md", "You are Jean.\n")

        with pytest.raises(ValueError, match="agent.yaml"):
            load(root)

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
