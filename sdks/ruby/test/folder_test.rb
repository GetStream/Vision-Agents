# frozen_string_literal: true

require_relative "test_helper"

class TestFolder < Minitest::Test
  def setup
    @root = File.join(Dir.mktmpdir, "jean")
  end

  def teardown
    FileUtils.rm_rf(File.dirname(@root))
  end

  def write(name, content)
    path = File.join(@root, name)
    FileUtils.mkdir_p(File.dirname(path))
    File.write(path, content)
  end

  def test_a_directory_is_read_as_instructions_skills_and_knowledge
    write("agent.yaml", "name: jean\n")
    write("instructions.md", "You are Jean.\n")
    write("skills/think.md", "---\ndescription: Work something out before answering\ndeadline: 30s\n---\n" \
                             "Take your time and reason it through.\n")
    write("knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n")

    folder = VA::Folder.load(@root)

    assert_equal "jean", folder.name
    assert_equal "You are Jean.", folder.instructions
    assert_equal [VA::Skill.new(name: "think", description: "Work something out before answering",
                                instructions: "Take your time and reason it through.", deadline: 30.0)],
                 folder.skills
    assert_equal ["pricing.md"], folder.knowledge.map(&:source)
  end

  def test_the_fingerprint_is_the_one_every_sdk_takes
    write("agent.yaml", "name: jean\nllm: openai/gpt-5.6\n")
    write("instructions.md", "You are Jean.\n")
    write("skills/think.md", "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n")
    write("knowledge/pricing.md", "# Pricing\n\nA penny.\n")

    folder = VA::Folder.load(@root)
    assert_equal "02a7b2c8428f31e3a2b93ca2f5a6ec70", folder.fingerprint

    write("knowledge/urls.yaml", "- https://example.com/plans\n")
    refute_equal folder.fingerprint, VA::Folder.load(@root).fingerprint
  end

  def test_nested_knowledge_keeps_its_path_and_skips_what_is_not_a_document
    write("agent.yaml", "name: jean\n")
    write("knowledge/reference/api.md", "# API\n\nthe endpoints\n")
    write("knowledge/logo.png", "not a document")
    write("knowledge/empty.md", "   \n")

    assert_equal ["reference/api.md"], VA::Folder.load(@root).knowledge.map(&:source)
  end

  def test_declared_pages_are_read_without_being_ingested
    write("agent.yaml", "name: jean\n")
    write("knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n")
    write("knowledge/urls.yaml", "- https://example.com/pricing\n- url: https://example.com/plans\n  title: Plans\n" \
                                 "  description: What each plan includes.\n")
    write("knowledge/reference/urls.yaml", "the urls we used to have\n")

    folder = VA::Folder.load(@root)

    assert_equal ["pricing.md", "reference/urls.yaml"], folder.knowledge.map(&:source)
    assert_equal [VA::KnowledgeURL.new(url: "https://example.com/pricing"),
                  VA::KnowledgeURL.new(url: "https://example.com/plans", title: "Plans",
                                       description: "What each plan includes.")], folder.knowledge_urls
  end

  def test_a_page_that_cannot_be_fetched_or_described_is_refused
    ["- example.com/pricing\n", "- url: https://example.com/plans\n  heading: Plans\n",
     "- [https://example.com/plans]\n"].each do |declaration|
      write("agent.yaml", "name: jean\n")
      write("knowledge/urls.yaml", declaration)

      assert_raises(VA::ConfigurationError, declaration) { VA::Folder.load(@root) }
    end
  end

  def test_a_skill_needs_a_description_and_can_be_renamed
    write("agent.yaml", "name: jean\n")
    write("skills/01-think.md", "---\nname: think\ndescription: Work something out\n---\nReason it through.\n")
    assert_equal "think", VA::Folder.load(@root).skills[0].name

    write("skills/think.md", "Just a body, with nothing saying when to use it.\n")
    assert_raises(VA::ConfigurationError) { VA::Folder.load(@root) }
  end

  def test_the_declaration_says_what_the_agent_is_called_and_runs_on
    write("agent.yaml", "name: receptionist\nllm: openai/gpt-5.6\nsts: \"\"\nkeyterms: [Vision Agents]\n" \
                        "video:\n  source: camera\n")

    folder = VA::Folder.load(@root)

    assert_equal "receptionist", folder.name
    assert_equal "openai/gpt-5.6", folder.settings["llm"]
    assert_equal ["Vision Agents"], folder.settings["keyterms"]
    assert_equal "", folder.settings["sts"]
    assert_equal({ "source" => "camera", "max_frames" => 1 }, folder.settings["video"])
  end

  def test_a_declaration_key_nobody_knows_is_refused
    ["name: jean\nlmm: openai/gpt-5.6\n", "video:\n  max_frames: 9\n", "keyterms: Vision Agents\n"].each do |yaml|
      write("agent.yaml", yaml)

      assert_raises(VA::ConfigurationError, yaml) { VA::Folder.load(@root) }
    end
  end

  def test_a_directory_without_a_declaration_is_not_an_agent
    write("instructions.md", "You are Jean.\n")

    assert_raises(VA::ConfigurationError) { VA::Folder.load(@root) }
  end

  def test_the_stamp_records_the_hash_and_when
    write("agent.yaml", "name: jean\n")
    folder = VA::Folder.load(@root)
    assert_nil folder.stamp

    folder.write_stamp("abc")

    recorded = JSON.parse(File.read(File.join(@root, ".agent_sync")))
    assert_equal "abc", recorded["hash"]
    assert_match(/\A\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00\z/, recorded["synced_at"])
    assert_equal "abc", folder.stamp
  end
end
