# frozen_string_literal: true

require_relative "../test_helper"

# The SDK against a router that is actually running. Opt in with VISION_AGENTS_URL; the
# customer defaults to "examples". Run with `bundle exec rake live`.
#
# What it proves is the part the local router in the unit tests cannot: that the requests
# this gem builds are ones a real router accepts.
class TestLiveRouter < Minitest::Test
  URL = ENV.fetch("VISION_AGENTS_URL", "")
  CUSTOMER = ENV.fetch("VISION_AGENTS_CUSTOMER_ID", "examples")

  def setup
    skip "set VISION_AGENTS_URL to run against a router" if URL.empty?
    @api = VA::Client.new(url: URL, customer_id: CUSTOMER)
    @opened = []
  end

  def teardown
    @opened&.each(&:close)
  end

  def agent
    name = @api.get("/v1/agents/configs").map { |config| config["name"] }.compact.first
    skip "this router has no agent configured to address by name" unless name
    @api.agent(name)
  end

  # A conversation model this deployment resolves, since a name is resolved against the
  # deployment's own catalogue.
  def model
    @model ||= %w[llm-fast llm-thinking llm-flow].find do |target|
      @api.get("/v1/{modality}/routes/{target}", path: { modality: "llm", target: target })
    rescue VA::RouterError => e
      raise unless e.status == 404
    end || flunk("this deployment resolves none of the conversation models")
  end

  def open(**options)
    agent.sessions.create(llm: model, **options).tap { |session| @opened << session }
  end

  def answered(session)
    session.events.each do |event|
      return if event.kind == "responded"

      skip "nothing left to answer with: #{event.error}" if event.kind == "error" && event.error.include?("quota")
      flunk "the session reported: #{event.error}" if event.kind == "error"
    end
  end

  def unique(what)
    "ruby-#{what}-#{Time.now.to_i}-#{rand(1_000_000)}"
  end

  def test_the_router_is_healthy
    health = @api.get("/health")

    assert_equal "ok", health.dig("dependencies", "llm")
    assert_equal "ok", health["status"], "the router is #{health["status"]}: #{health["dependencies"]}"
  end

  def test_a_conversation_is_opened_with_labels_and_found_again
    title = "Sendbird #{unique("search")}"
    session = open(title: title, project: "docs", custom: { suite: "ruby" }, persist_conversation: true)
    session.close

    assert_equal title, session.created["title"]
    assert_equal({ "suite" => "ruby" }, session.created["custom"])
    assert_match(/\Aagent:/, session.conversation_id)
    assert(agent.sessions.query(limit: 50).any? { |each| each["id"] == session.id })
    assert(agent.sessions.search("Sendbird", limit: 50).any? { |each| each["id"] == session.id })
  end

  def test_an_incognito_conversation_keeps_nothing
    title = unique("incognito")
    session = open(incognito: true, title: title, persist_conversation: true)
    session.close

    assert_equal "", session.conversation_id
    assert_empty agent.sessions.search(title, limit: 50)
  end

  def test_a_turn_is_named_and_written_down
    session = open(persist_conversation: true)

    turn = session.responses.create("Reply with the single word: pong.")
    answered(session)

    refute_empty turn.id
    items = turn.items.all
    assert_equal "said", items.first["kind"]
    assert(items.any? { |item| item["kind"] == "answer" })
  end

  def test_rewinding_takes_the_later_turns_out_and_a_fork_starts_from_a_turn
    session = open
    ["Reply with the single word: one.", "Reply with the single word: two."].each do |question|
      session.responses.create(question)
      answered(session)
    end

    kept = session.responses.list.first
    session.responses.rewind(kept)
    assert_equal [kept["id"]], session.responses.list.map { |response| response["id"] }

    forked = session.fork(response_id: kept).tap { |fork| @opened << fork }
    assert_equal session.id, forked.created["forked_from"]
  end

  def test_a_conversation_kept_in_stream_chat_is_forked_rather_than_rewound
    session = open(persist_conversation: true)

    error = assert_raises(VA::RouterError) { session.responses.rewind("anything") }
    assert_equal 400, error.status

    forked = session.fork(title: "asked again").tap { |fork| @opened << fork }
    assert_equal session.id, forked.created["forked_from"]
    refute_equal session.conversation_id, forked.conversation_id
  end

  def test_a_closed_conversation_is_no_longer_running
    session = open
    session.close

    refute(agent.sessions.query(state: "running", limit: 50).any? { |each| each["id"] == session.id })
  end

  def test_a_guest_is_minted
    guest = @api.guest_user(name: "Ruby guest")

    refute_empty guest["id"]
    refute_empty guest["token"]
  end

  def test_a_folder_is_synced_once_and_then_read_back
    root = File.join(Dir.mktmpdir, unique("agent"))
    FileUtils.mkdir_p(root)
    File.write(File.join(root, "agent.yaml"), "llm: #{model}\n")
    File.write(File.join(root, "instructions.md"), "You are a test agent. Answer in one word.\n")

    first = VA::Agent.new(folder: root, client: @api).sync
    second = VA::Agent.new(folder: root, client: @api).sync

    assert_equal File.basename(root), first.dig("config", "name")
    assert_equal true, second["unchanged"]
    assert_equal first.dig("config", "id"), second.dig("config", "id")
  ensure
    id = first&.dig("config", "id")
    @api.delete("/v1/agents/configs/{id}", path: { id: id }) if id
    FileUtils.rm_rf(File.dirname(root)) if root
  end

  def test_a_worker_is_told_its_id
    dispatch = VA::Dispatch.new(client: @api).wait_for_call { nil }
    runner = Thread.new { dispatch.run }

    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + 10
    sleep 0.05 until dispatch.worker_id || Process.clock_gettime(Process::CLOCK_MONOTONIC) > deadline
    dispatch.stop
    runner.join(5)

    refute_nil dispatch.worker_id
  end

  def test_a_question_is_answered_over_the_llm_stream
    complete = VA::Router.new(client: @api).llm.realtime(target: model) do |llm|
      llm.respond("Reply with the single word: pong.")
    end

    refute_empty complete["text"]
  end

  def test_search_answers
    answer = VA::Router.new(client: @api).search("Stream Vision Agents", results: 3)

    refute_empty answer["results"]
  end
end
