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

  def test_refresh_hours_and_simulations_fingerprint_the_way_go_does
    write("agent.yaml", "name: jean\nllm: openai/gpt-5.6\n")
    write("instructions.md", "You are Jean.\n")
    write("knowledge/urls.yaml", "- https://example.com/pricing\n- url: https://example.com/plans\n  title: Plans\n" \
                                 "  refresh_hours: 24\n")
    write("simulations/b.yaml", "- name: lunch <order> & \"change\"\n  scenario: \"Order a club,\\n\\tthen swap it\"\n" \
                                "  assertion: One veggie wrap.\n  mode: audio\n  variations: 3\n  tags:\n" \
                                "    team: voice\n    a: b\n- name: dinner\n  scenario: Book a table\n" \
                                "  assertion: Booked\n")
    write("simulations/a.yml", "- name: breakfast\n  scenario: Eggs\n  assertion: Eggs\n  max_turns: 4\n  tags: {}\n")
    write("simulations/notes.txt", "not a simulation")

    folder = VA::Folder.load(@root)

    assert_equal 24, folder.knowledge_urls[1].refresh_hours
    assert_equal ["breakfast", 'lunch <order> & "change"', "dinner"], folder.simulations.map(&:name)
    assert_equal "229a3c5c66835374f8aee179a67eedf2", folder.fingerprint
  end

  def test_a_refresh_that_is_not_a_whole_number_of_hours_is_refused
    write("agent.yaml", "name: jean\n")
    ["0", "1.5", "daily"].each do |hours|
      write("knowledge/urls.yaml", "- url: https://example.com/plans\n  refresh_hours: #{hours}\n")

      assert_raises(VA::ConfigurationError, hours) { VA::Folder.load(@root) }
    end
  end

  def test_an_empty_simulations_directory_is_not_the_same_as_none
    write("agent.yaml", "name: jean\n")
    without = VA::Folder.load(@root)
    FileUtils.mkdir_p(File.join(@root, "simulations"))
    empty = VA::Folder.load(@root)

    assert_nil without.simulations
    assert_equal [], empty.simulations
    refute_equal without.fingerprint, empty.fingerprint
  end

  def test_a_simulation_that_cannot_be_run_is_refused
    write("agent.yaml", "name: jean\n")
    ["- name: a\n  scenario: s\n", "- name: a\n  scenario: s\n  assertion: x\n  judge: llm\n",
     "- name: a\n  scenario: s\n  assertion: x\n  mode: video\n", "- name: a\n  scenario: s\n  assertion: x\n  variations: many\n",
     "name: a\n"].each do |yaml|
      write("simulations/one.yaml", yaml)

      assert_raises(VA::ConfigurationError, yaml) { VA::Folder.load(@root) }
    end
  end

  def test_a_simulation_name_two_files_share_is_refused
    write("agent.yaml", "name: jean\n")
    write("simulations/a.yaml", "- name: lunch\n  scenario: s\n  assertion: x\n")
    write("simulations/b.yaml", "- name: lunch\n  scenario: t\n  assertion: y\n")

    error = assert_raises(VA::ConfigurationError) { VA::Folder.load(@root) }
    assert_includes error.message, "a.yaml"
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
                        "video:\n  source: camera\nspeed: 1.1\nharness: default\n")

    folder = VA::Folder.load(@root)

    assert_equal "receptionist", folder.name
    assert_equal "openai/gpt-5.6", folder.settings["llm"]
    assert_equal 1.1, folder.settings["speed"]
    assert_equal "default", folder.settings["harness"]
    assert_equal ["Vision Agents"], folder.settings["keyterms"]
    assert_equal "", folder.settings["sts"]
    assert_equal({ "source" => "camera", "max_frames" => 1 }, folder.settings["video"])
  end

  def test_a_declaration_key_nobody_knows_is_refused
    ["name: jean\nlmm: openai/gpt-5.6\n", "video:\n  max_frames: 9\n", "keyterms: Vision Agents\n",
     "dispatch:\n  sms: enabled\n", "speed: fast\n"].each do |yaml|
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
