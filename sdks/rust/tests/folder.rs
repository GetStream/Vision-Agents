use std::path::{Path, PathBuf};
use std::time::Duration;

use vision_agents::{Folder, types};

fn write(root: &Path, name: &str, content: &str) {
    let path = root.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

fn directory(name: &str) -> (tempfile::TempDir, PathBuf) {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join(name);
    std::fs::create_dir_all(&root).unwrap();
    (temporary, root)
}

#[test]
fn a_directory_is_read_as_instructions_skills_and_knowledge() {
    let (_temporary, root) = directory("jean");
    write(&root, "agent.yaml", "name: jean\n");
    write(&root, "instructions.md", "You are Jean.\n");
    write(
        &root,
        "skills/think.md",
        "---\ndescription: Work something out before answering\ndeadline: 30s\n---\nTake your time and reason it through.\n",
    );
    write(
        &root,
        "knowledge/pricing.md",
        "# Pricing\n\nA call costs a penny.\n",
    );

    let folder = Folder::load(&root).unwrap();

    assert_eq!(folder.name, "jean");
    assert_eq!(folder.instructions, "You are Jean.");
    assert_eq!(folder.skills.len(), 1);
    let skill = &folder.skills[0];
    assert_eq!(
        (skill.name.as_str(), skill.description.as_str()),
        ("think", "Work something out before answering")
    );
    assert_eq!(skill.deadline, Duration::from_secs(30));
    assert_eq!(skill.instructions, "Take your time and reason it through.");
    assert_eq!(folder.knowledge.len(), 1);
    assert_eq!(folder.knowledge[0].source, "pricing.md");
    assert_eq!(folder.knowledge_namespace(), "jean");
}

#[test]
fn a_directory_with_only_instructions_is_an_agent() {
    let (_temporary, root) = directory("quiet");
    write(&root, "agent.yaml", "");
    write(&root, "instructions.md", "Say little.\n");

    let folder = Folder::load(&root).unwrap();

    assert!(folder.skills.is_empty() && folder.knowledge.is_empty());
    assert_eq!(folder.knowledge_namespace(), "");
    assert_eq!(folder.name, "quiet");
}

#[test]
fn nested_knowledge_keeps_the_path_it_was_found_at() {
    let (_temporary, root) = directory("jean");
    write(&root, "agent.yaml", "name: jean\n");
    write(
        &root,
        "knowledge/reference/api.md",
        "# API\n\nthe endpoints\n",
    );
    write(&root, "knowledge/logo.png", "not a document");
    write(&root, "knowledge/empty.md", "   \n");

    let folder = Folder::load(&root).unwrap();

    assert_eq!(folder.knowledge.len(), 1);
    assert_eq!(folder.knowledge[0].source, "reference/api.md");
}

#[test]
fn declared_pages_are_read_without_being_ingested() {
    let (_temporary, root) = directory("jean");
    write(&root, "agent.yaml", "name: jean\n");
    write(
        &root,
        "knowledge/pricing.md",
        "# Pricing\n\nA call costs a penny.\n",
    );
    write(
        &root,
        "knowledge/urls.yaml",
        "- https://example.com/pricing\n- url: https://example.com/plans\n  title: Plans\n  description: What each plan includes.\n",
    );
    write(
        &root,
        "knowledge/reference/urls.yaml",
        "the urls we used to have\n",
    );

    let folder = Folder::load(&root).unwrap();

    let sources: Vec<_> = folder
        .knowledge
        .iter()
        .map(|document| document.source.as_str())
        .collect();
    assert_eq!(sources, ["pricing.md", "reference/urls.yaml"]);
    assert_eq!(folder.knowledge_urls.len(), 2);
    let (bare, described) = (&folder.knowledge_urls[0], &folder.knowledge_urls[1]);
    assert_eq!(
        (
            bare.url.as_str(),
            bare.title.as_str(),
            bare.description.as_str()
        ),
        ("https://example.com/pricing", "", "")
    );
    assert_eq!(
        (
            described.url.as_str(),
            described.title.as_str(),
            described.description.as_str()
        ),
        (
            "https://example.com/plans",
            "Plans",
            "What each plan includes."
        )
    );
}

#[test]
fn a_directory_with_only_pages_has_somewhere_to_look_things_up_in() {
    let (_temporary, root) = directory("jean");
    write(&root, "agent.yaml", "name: jean\n");
    write(
        &root,
        "knowledge/urls.yaml",
        "- https://example.com/pricing\n",
    );

    let folder = Folder::load(&root).unwrap();

    assert!(folder.knowledge.is_empty());
    assert_eq!(folder.knowledge_namespace(), "jean");
}

#[test]
fn a_page_that_cannot_be_fetched_or_described_is_refused() {
    for declaration in [
        "- example.com/pricing\n",
        "- url: https://example.com/plans\n  heading: Plans\n",
        "- [https://example.com/plans]\n",
    ] {
        let (_temporary, root) = directory("jean");
        write(&root, "agent.yaml", "name: jean\n");
        write(&root, "knowledge/urls.yaml", declaration);

        assert!(Folder::load(&root).is_err(), "accepted {declaration:?}");
    }
}

#[test]
fn a_skill_without_a_description_is_refused() {
    let (_temporary, root) = directory("jean");
    write(&root, "agent.yaml", "name: jean\n");
    write(
        &root,
        "skills/think.md",
        "Just a body, with nothing saying when to use it.\n",
    );

    assert!(Folder::load(&root).is_err());
}

#[test]
fn a_skill_can_be_named_something_other_than_its_file() {
    let (_temporary, root) = directory("jean");
    write(&root, "agent.yaml", "name: jean\n");
    write(
        &root,
        "skills/01-think.md",
        "---\nname: think\ndescription: Work something out\ncapture_video: true\n---\nReason it through.\n",
    );

    let folder = Folder::load(&root).unwrap();

    assert_eq!(folder.skills[0].name, "think");
    assert!(folder.skills[0].capture_video);
}

#[test]
fn a_deadline_is_a_go_duration_or_seconds() {
    for (written, wanted) in [
        ("1m30s", 90_000),
        ("250ms", 250),
        ("1.5", 1_500),
        ("2h", 7_200_000),
    ] {
        let (_temporary, root) = directory("jean");
        write(&root, "agent.yaml", "");
        write(
            &root,
            "skills/think.md",
            &format!("---\ndescription: d\ndeadline: {written}\n---\nbody\n"),
        );

        let folder = Folder::load(&root).unwrap();

        assert_eq!(folder.skills[0].deadline.as_millis(), wanted, "{written}");
    }
}

#[test]
fn a_directory_without_a_declaration_is_not_an_agent() {
    let (_temporary, root) = directory("jean");
    write(&root, "instructions.md", "You are Jean.\n");

    assert!(Folder::load(&root).is_err());
}

#[test]
fn the_declaration_says_what_the_agent_is_called_and_runs_on() {
    let (_temporary, root) = directory("jean");
    write(
        &root,
        "agent.yaml",
        "name: receptionist\nllm: openai/gpt-5.6\nsts: \"\"\nkeyterms: [Vision Agents]\nmode: voice\nsandbox: daytona\nvideo:\n  source: camera\n",
    );

    let folder = Folder::load(&root).unwrap();

    assert_eq!(folder.name, "receptionist");
    let settings = &folder.settings;
    assert_eq!(settings.llm, "openai/gpt-5.6");
    assert_eq!(settings.keyterms, ["Vision Agents"]);
    assert_eq!(settings.sts.as_deref(), Some(""));
    assert_eq!(settings.mode, Some(types::AgentMode::Voice));
    assert_eq!(settings.sandbox, Some(types::Sandbox::Daytona));
    let video = settings.video.as_ref().unwrap();
    assert_eq!((video.source.as_str(), video.max_frames), ("camera", 1));
}

#[test]
fn saying_nothing_about_speech_to_speech_is_not_turning_it_off() {
    let (_temporary, root) = directory("jean");
    write(&root, "agent.yaml", "name: jean\n");

    assert_eq!(Folder::load(&root).unwrap().settings.sts, None);
}

#[test]
fn a_declaration_key_nobody_knows_is_refused() {
    for declaration in [
        "name: jean\nlmm: openai/gpt-5.6\n",
        "video:\n  max_frames: 9\n",
        "keyterms: Vision Agents\n",
        "video:\n  frames: 2\n",
        "sandbox: docker\n",
    ] {
        let (_temporary, root) = directory("jean");
        write(&root, "agent.yaml", declaration);

        assert!(Folder::load(&root).is_err(), "accepted {declaration:?}");
    }
}

#[test]
fn a_directory_hashes_the_way_the_python_and_go_sdks_hash_it() {
    let (_temporary, root) = directory("jean");
    write(&root, "agent.yaml", "name: jean\nllm: openai/gpt-5.6\n");
    write(&root, "instructions.md", "You are Jean.\n");
    write(
        &root,
        "skills/think.md",
        "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n",
    );
    write(&root, "knowledge/pricing.md", "# Pricing\n\nA penny.\n");

    let folder = Folder::load(&root).unwrap();
    assert_eq!(folder.hash(), "02a7b2c8428f31e3a2b93ca2f5a6ec70");

    write(
        &root,
        "knowledge/urls.yaml",
        "- https://example.com/plans\n",
    );
    assert_ne!(Folder::load(&root).unwrap().hash(), folder.hash());
}

#[test]
fn something_that_is_not_a_directory_is_not_an_agent() {
    let temporary = tempfile::tempdir().unwrap();
    write(temporary.path(), "jean.md", "You are Jean.");

    assert!(Folder::load(temporary.path().join("jean.md")).is_err());
}

#[test]
fn an_agent_directory_is_found_from_anywhere_below_the_repository() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path();
    write(
        repository,
        "examples/voice_agents/jean/agent.yaml",
        "name: jean\n",
    );
    write(repository, "agents/ada/agent.yaml", "name: ada\n");
    std::fs::create_dir_all(repository.join("deep/down")).unwrap();

    let from = repository.join("deep/down");
    assert_eq!(
        Folder::find("jean", &from),
        Some(repository.join("examples/voice_agents/jean"))
    );
    assert_eq!(
        Folder::find("ada", &from),
        Some(repository.join("agents/ada"))
    );
    assert_eq!(Folder::find("nobody", &from), None);
    let inside = repository.join("examples/voice_agents/jean");
    assert_eq!(Folder::find("jean", &inside), Some(inside.clone()));
}

#[test]
fn a_stamp_records_what_was_synced() {
    let temporary = tempfile::tempdir().unwrap();

    assert_eq!(vision_agents::folder::read_stamp(temporary.path()), "");
    vision_agents::folder::write_stamp(temporary.path(), "abc").unwrap();

    assert_eq!(vision_agents::folder::read_stamp(temporary.path()), "abc");
    let written = std::fs::read_to_string(temporary.path().join(".agent_sync")).unwrap();
    assert!(written.ends_with('\n'));
    let stamp: serde_json::Value = serde_json::from_str(&written).unwrap();
    let synced_at = stamp["synced_at"].as_str().unwrap();
    assert_eq!(
        synced_at.len(),
        "2026-09-24T10:00:00+00:00".len(),
        "{synced_at}"
    );
}
