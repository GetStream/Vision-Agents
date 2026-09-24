# frozen_string_literal: true

require_relative "test_helper"

class TestAgent < LocalRouterTest
  CALLS = %r{\A/api/v2/video/call/}

  def setup
    super
    @router.on(:post, CALLS, body: { "duration" => "1ms" })
  end

  def edge
    VA::Edge.new(api_key: "key", api_secret: "secret", base_url: @router.url, monitor_url: "https://demo.example")
  end

  def agent(**options)
    VA::Agent.new(config: "support", client: client, edge: edge, **options)
  end

  def session_request
    @router.last(:post, "/v1/agents/sessions").json
  end

  def test_join_creates_the_call_and_starts_from_the_config
    serve_session

    agent(cost_tracking: { env: "production", team: 7 }, memory_filter: { user_id: 123, plan: "pro" })
      .join("call-1", participant_wait_timeout: 0, wait_for_end: false) { |session| assert session.live? }

    created = @router.last(:post, CALLS)
    assert_equal "/api/v2/video/call/agent/call-1", created.path
    assert_equal "support", created.json.dig("data", "created_by_id")
    assert_equal({ "agent" => "support", "user_id" => "support", "user_name" => "support", "agent_id" => "support",
                   "tags" => { "env" => "production", "team" => "7" },
                   "memory" => { "user_id" => "123", "filter" => { "plan" => "pro" } },
                   "call_id" => "call-1", "call_type" => "agent" }, session_request)
  end

  def test_leaving_the_block_closes_the_session_and_returns_its_value
    closed = Thread::Queue.new
    serve_session { |peer| closed << peer.receive_type("close", timeout: 10) }

    value = agent.join("call-1", participant_wait_timeout: 0, wait_for_end: false) { :done }

    assert_equal :done, value
    assert_equal({ "type" => "close" }, closed.pop(timeout: 5))
  end

  def test_the_block_waits_for_the_call_to_end
    serve_session do |peer|
      peer.send_frame(type: "participant_joined", participant: { id: "p1", user_id: "ada", name: "Ada" })
      sleep 0.2
      peer.send_frame(type: "left")
    end

    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    participants = agent.join("call-1") { |session| session.participants }

    assert_equal [VA::Participant.new("p1", "ada", "Ada")], participants
    assert_operator Process.clock_gettime(Process::CLOCK_MONOTONIC) - started, :>=, 0.2
  end

  def test_a_participant_that_is_the_agent_is_not_somebody_in_the_call
    serve_session do |peer|
      peer.send_frame(type: "participant_joined", participant: { id: "a", user_id: "agent-user" })
      peer.receive_type("close", timeout: 10)
    end

    agent.join("call-1", participant_wait_timeout: 0, wait_for_end: false) do |session|
      refute session.wait_for_participant(timeout: 0.3)
      assert_empty session.participants
    end
  end

  def test_an_inbound_call_is_joined_as_it_arrived
    serve_session do |peer|
      peer.send_frame(type: "heard", text: "hello")
      peer.receive_type("close", timeout: 10)
    end
    call = VA::InboundCall.from("call_id" => "pstn-1", "called_number" => "+15550100", "caller_number" => "+15550199")

    agent.join(call, wait_for_end: false) { assert call.wait_for_phone_participant(timeout: 5) }

    assert_empty @router.seen(:post, CALLS)
    assert_equal "pstn-1", session_request["call_id"]
    assert_equal "default", session_request["call_type"]
    assert_equal({ "number" => "+15550100" }, session_request["phone"])
  end

  def test_the_caller_cannot_be_waited_for_before_joining
    call = VA::InboundCall.new(call_id: "pstn-1")

    assert_raises(VA::ConfigurationError) { call.wait_for_phone_participant(timeout: 0) }
  end

  def test_responses_create_names_the_turn
    serve_session
    @router.on(:post, "/v1/agents/sessions/sess_1/responses") do |request|
      [202, { "id" => "resp_1", "session_id" => "sess_1", "status" => "running", "said" => request.json["text"] }]
    end
    support = agent

    response = support.join("call-1", participant_wait_timeout: 0, wait_for_end: false) do
      support.responses.create("greet the user")
    end

    assert_equal "resp_1", response.id
    assert_equal({ "text" => "greet the user" }, @router.last(:post, "/v1/agents/sessions/sess_1/responses").json)
  end

  def test_responses_need_a_conversation
    assert_raises(VA::ConfigurationError) { agent.responses }
  end

  def test_a_tool_call_is_answered_with_its_command_and_turn
    results = Thread::Queue.new
    serve_session do |peer|
      peer.send_frame(type: "tool_call", id: "t1", name: "weather", arguments: '{"city":"Paris"}',
                      command_id: "c1", turn_id: "turn_1")
      results << peer.receive_type("tool_result")
      peer.send_frame(type: "tool_call", id: "t2", name: "broken", arguments: "{}")
      results << peer.receive_type("tool_result")
      peer.receive_type("close", timeout: 10)
    end
    tools = VA::Tools.new
    tools.register("weather", description: "Weather for a city") { |args| { city: args["city"], sky: "clear" } }
    tools.register("broken", description: "Always fails") { raise ArgumentError, "no sky today" }

    agent(tools: tools).join("call-1", participant_wait_timeout: 0, wait_for_end: false) do
      assert_equal({ "type" => "tool_result", "tool_call_id" => "t1", "output" => '{"city":"Paris","sky":"clear"}',
                     "command_id" => "c1", "turn_id" => "turn_1" }, results.pop(timeout: 5))
      assert_equal({ "type" => "tool_result", "tool_call_id" => "t2", "error" => "no sky today" },
                   results.pop(timeout: 5))
    end
    assert_equal "weather", session_request["tools"][0]["name"]
  end

  def test_a_cancelled_tool_answers_nothing
    results = Thread::Queue.new
    release = Thread::Queue.new
    serve_session do |peer|
      peer.send_frame(type: "tool_call", id: "t1", name: "slow", arguments: "{}")
      peer.send_frame(type: "tool_cancel", id: "t1")
      sleep 0.05
      release << true
      results << peer.receive_type("tool_result", timeout: 0.5)
    end
    tools = VA::Tools.new.register("slow", description: "Takes a while") { release.pop(timeout: 5) && "late" }

    agent(tools: tools).join("call-1", participant_wait_timeout: 0, wait_for_end: false) do
      assert_nil results.pop(timeout: 5)
    end
  end

  def test_events_report_what_the_conversation_did
    serve_session do |peer|
      peer.send_frame(type: "heard", text: "hi", participant: { id: "p1", user_id: "ada" })
      peer.send_frame(type: "responded", text: "hello", pending_work: true)
      peer.send_frame(type: "left")
    end

    events = agent.join("call-1", participant_wait_timeout: 0) { |session| session.events.to_a }

    assert_equal %w[heard responded left], events.map(&:kind)
    assert_equal "ada", events[0].participant.user_id
    assert events[1].pending_work
  end

  def test_the_events_socket_asks_for_what_was_wanted
    serve_session

    agent.join("call-1", participant_wait_timeout: 0, wait_for_end: false, interim: true) { nil }

    assert_equal({ "interim" => "true", "decisions" => "false" },
                 @router.last(:get, "/v1/agents/sessions/sess_1/events").query)
  end

  def test_a_session_nothing_can_watch_is_closed
    @router.on(:post, "/v1/agents/sessions", body: { "id" => "sess_9" })
    @router.on(:delete, "/v1/agents/sessions/sess_9", status: 204)

    assert_raises(VA::RouterError) { agent.join("call-1", participant_wait_timeout: 0) }
    assert_equal 1, @router.seen(:delete, "/v1/agents/sessions/sess_9").size
  end

  def test_chat_holds_the_conversation_in_writing
    serve_session

    agent.chat(persist: true, conversation_id: "agent:room-1") { nil }
    assert_equal true, session_request["text"]
    assert_equal true, session_request["persist_conversation"]
    assert_equal "agent:room-1", session_request["conversation_id"]

    agent.chat(persist: true, incognito: true) { nil }
    refute session_request.key?("persist_conversation")
    assert_equal true, session_request["incognito"]
    assert_empty @router.seen(:post, CALLS)
  end

  def test_reply_answers_in_the_channel_the_message_came_from
    serve_session
    message = VA::InboundMessage.from("channel_id" => "room-1", "agent_id" => "support-1", "text" => "hi")

    agent.reply(message) { nil }

    assert_equal "agent:room-1", session_request["conversation_id"]
    assert_equal "support-1", session_request["agent_id"]
    assert_equal true, session_request["persist_conversation"]
  end

  def test_an_outbound_call_is_placed_before_the_agent_joins
    serve_session
    @router.on(:post, "/v1/phone/calls") do |request|
      [202, { "vendor_call_id" => "CA123", "status" => "queued", "call_id" => request.json["call_id"] }]
    end

    agent(cost_tracking: { env: "production" })
      .outbound_call(from: "+15550100", to: "+15550199", call_id: "out-1", ring_timeout: 30,
                     participant_wait_timeout: 0, wait_for_end: false) { nil }

    placed = @router.last(:post, "/v1/phone/calls").json
    assert_equal({ "from" => "+15550100", "to" => "+15550199", "call_id" => "out-1", "call_type" => "agent",
                   "ring_timeout_seconds" => 30, "tags" => { "env" => "production" } }, placed)
    assert_equal true, session_request["navigating"]
    assert_equal({ "number" => "+15550100", "vendor_call_id" => "CA123" }, session_request["phone"])
    order = @router.requests.map { |r| r.path.split("/")[2] }
    assert_equal %w[v2 phone agents], order.first(3)
  end

  def test_what_the_code_sets_is_sent_and_nothing_else
    serve_session
    skills = [VA::Skill.new(name: "research", description: "Looks things up", instructions: "Search first",
                            deadline: 30)]

    agent(name: "Ada", instructions: "Be brief", pipeline: { llm: "fast", language: "fr", max_tokens: 200 },
          skills: skills, sandbox: VA::Sandbox.daytona, tasks: 2)
      .chat { nil }

    request = session_request
    assert_equal "Ada", request["user_name"]
    assert_equal "ada", request["user_id"]
    assert_equal "Be brief", request["instructions"]
    assert_equal "fast", request["llm"]
    assert_equal ["fr"], request["languages"]
    assert_equal 200, request["max_tokens"]
    assert_equal "daytona", request["sandbox"]
    assert_equal 2, request["tasks"]
    assert_equal [{ "name" => "research", "description" => "Looks things up", "instructions" => "Search first",
                    "capture_video" => false, "deadline_ms" => 30_000 }], request["skills"]
    refute request.key?("stt")
  end

  def test_turning_the_built_in_skills_off_sends_an_empty_list
    serve_session

    agent(use_skills: false).chat { nil }

    assert_equal [], session_request["skills"]
  end

  def test_what_cannot_be_an_agent_is_refused
    assert_raises(VA::ConfigurationError) { VA::Agent.new(client: client) }
    assert_raises(VA::ConfigurationError) { agent(pipeline: { lmm: "fast" }) }
    assert_raises(VA::ConfigurationError) do
      agent(skills: [VA::Skill.new(name: "research", description: "", instructions: "x")])
    end
  end

  def test_one_agent_holds_one_conversation
    serve_session { |peer| peer.receive_type("close", timeout: 10) }
    support = agent

    support.chat
    assert_raises(VA::ConfigurationError) { support.chat }
    support.close
  end

  def test_the_monitor_url_joins_the_call_as_somebody_else
    serve_session
    support = agent

    url = support.join("call-1", participant_wait_timeout: 0, wait_for_end: false) { support.monitor_url }

    assert url.start_with?("https://demo.example/join/call-1?api_key=key&token=")
    assert_includes url, "user_name=Monitor"
  end

  def test_user_ids_are_made_from_names
    assert_equal "ada-lovelace", VA::Agent.user_id_of("  Ada Lovelace!")
    assert_equal "vision-agent", VA::Agent.user_id_of("!!!")
  end

  def folder
    root = File.join(Dir.mktmpdir, "jean")
    FileUtils.mkdir_p(File.join(root, "knowledge"))
    File.write(File.join(root, "agent.yaml"), "name: jean\nllm: openai/gpt-5.6\ntags:\n  team: voice\n")
    File.write(File.join(root, "instructions.md"), "You are Jean.\n")
    File.write(File.join(root, "guardrail.md"), "Never quote prices.\n")
    File.write(File.join(root, "knowledge/pricing.md"), "# Pricing\n\nA penny.\n")
    File.write(File.join(root, "knowledge/urls.yaml"), "- https://example.com/plans\n")
    root
  end

  def test_sync_stores_the_folder_in_one_request_and_stamps_it
    root = folder
    @router.on(:post, "/v1/agents/sync", body: { "unchanged" => false, "config" => { "name" => "jean" } })
    jean = VA::Agent.new(folder: root, client: client)

    jean.sync

    request = @router.last(:post, "/v1/agents/sync").json
    assert_equal VA::Folder.load(root).fingerprint, request["hash"]
    assert_equal({ "name" => "jean", "hash" => request["hash"], "instructions" => "You are Jean.",
                   "guardrail" => "Never quote prices.", "llm" => "openai/gpt-5.6",
                   "knowledge" => [{ "source" => "pricing.md", "text" => "# Pricing\n\nA penny.\n" }],
                   "knowledge_urls" => [{ "url" => "https://example.com/plans" }],
                   "tags" => { "team" => "voice" } }, request)
    assert_equal request["hash"], JSON.parse(File.read(File.join(root, ".agent_sync")))["hash"]
  ensure
    FileUtils.rm_rf(File.dirname(root))
  end

  def test_an_untouched_folder_is_read_back_rather_than_written
    root = folder
    @router.on(:post, "/v1/agents/sync", body: { "unchanged" => false, "config" => { "name" => "jean" } })
    @router.on(:get, "/v1/agents/configs", body: [{ "name" => "jean", "id" => "cfg_1" }])
    VA::Agent.new(folder: root, client: client).sync

    result = VA::Agent.new(folder: root, client: client).sync

    assert_equal({ "unchanged" => true, "config" => { "name" => "jean", "id" => "cfg_1" } }, result)
    assert_equal 1, @router.seen(:post, "/v1/agents/sync").size
    assert_equal({ "name" => "jean" }, @router.last(:get, "/v1/agents/configs").query)
  ensure
    FileUtils.rm_rf(File.dirname(root))
  end

  def test_a_config_deleted_since_the_stamp_is_synced_again
    root = folder
    @router.on(:post, "/v1/agents/sync", body: { "unchanged" => false, "config" => { "name" => "jean" } })
    @router.on(:get, "/v1/agents/configs", body: [])
    VA::Agent.new(folder: root, client: client).sync

    VA::Agent.new(folder: root, client: client).sync

    assert_equal 2, @router.seen(:post, "/v1/agents/sync").size
  ensure
    FileUtils.rm_rf(File.dirname(root))
  end

  def test_cost_labels_are_synced_and_fingerprinted
    root = folder
    @router.on(:post, "/v1/agents/sync", body: { "unchanged" => false })

    VA::Agent.new(folder: root, client: client, cost_tracking: { env: "production" }).sync

    request = @router.last(:post, "/v1/agents/sync").json
    assert_equal({ "team" => "voice", "env" => "production" }, request["tags"])
    expected = VA::Folder.fingerprint(VA::Folder.load(root).fingerprint, "", "map[env:production]", [], [], [])
    assert_equal expected, request["hash"]
  ensure
    FileUtils.rm_rf(File.dirname(root))
  end

  def test_knowledge_is_added_under_the_agents_name_and_waited_for
    reads = 0
    @router.on(:post, "/v1/agents/knowledge/urls", body: { "id" => "k1", "state" => "pending" })
    @router.on(:get, "/v1/agents/knowledge/urls/k1") do
      reads += 1
      { "id" => "k1", "state" => reads < 2 ? "pending" : "indexed", "passages" => 4 }
    end

    page = agent.knowledge.add_url("https://example.com/pricing", title: "Pricing")

    assert_equal "indexed", page["state"]
    assert_equal({ "namespace" => "support", "url" => "https://example.com/pricing", "title" => "Pricing" },
                 @router.last(:post, "/v1/agents/knowledge/urls").json)
  end
end

class TestSessions < LocalRouterTest
  def test_sessions_are_read_back_by_agent
    @router.on(:get, "/v1/agents/sessions", body: [{ "id" => "s1" }])
    @router.on(:get, "/v1/agents/sessions/search", body: [{ "id" => "s2" }])
    sessions = client.agent("docs").sessions

    assert_equal [{ "id" => "s1" }], sessions.query(state: "closed", custom: { tenant: "acme" })
    assert_equal [{ "id" => "s2" }], sessions.search("billing")

    assert_equal({ "state" => "closed", "custom" => '{"tenant":"acme"}', "agent" => "docs" },
                 @router.last(:get, "/v1/agents/sessions").query)
    assert_equal({ "agent" => "docs", "q" => "billing" }, @router.last(:get, "/v1/agents/sessions/search").query)
  end

  def test_create_opens_a_written_conversation
    serve_session

    session = client.agent("docs").sessions.create(title: "Is Stream better?", persist_conversation: true)
    session.close

    request = @router.last(:post, "/v1/agents/sessions").json
    assert_equal({ "agent" => "docs", "user_id" => "docs", "user_name" => "docs", "agent_id" => "docs",
                   "title" => "Is Stream better?", "persist_conversation" => true, "text" => true }, request)
  end
end

class TestResponses < LocalRouterTest
  def responses
    VA::Responses.new(client, "sess_1")
  end

  def test_rewind_goes_back_to_a_response
    @router.on(:post, "/v1/agents/sessions/sess_1/rewind", status: 204)

    assert_nil responses.rewind({ "response_id" => "resp_1", "type" => "message" })
    assert_equal({ "response_id" => "resp_1" }, @router.last(:post, "/v1/agents/sessions/sess_1/rewind").json)
  end

  def test_a_persisted_conversation_cannot_be_rewound
    @router.on(:post, "/v1/agents/sessions/sess_1/rewind", status: 400,
                                                           body: { "error" => "fork a persisted conversation instead" })

    error = assert_raises(VA::RouterError) { responses.rewind("resp_1") }
    assert_equal 400, error.status
  end

  def test_rewind_needs_an_id
    assert_raises(VA::ConfigurationError) { responses.rewind(VA::AgentResponse.new(client, { "id" => "" })) }
  end

  def test_items_are_read_a_page_at_a_time
    @router.on(:get, "/v1/agents/sessions/sess_1/responses/items") do |request|
      offset = request.query["offset"].to_i
      (offset...[offset + 2, 3].min).map { |i| { "id" => "item_#{i}" } }
    end

    items = responses.items.each(page: 2).to_a

    assert_equal %w[item_0 item_1 item_2], items.map { |item| item["id"] }
  end

  def test_one_turns_items
    @router.on(:get, "/v1/agents/sessions/sess_1/responses/items", body: [])
    turn = VA::AgentResponse.new(client, { "id" => "resp_1", "session_id" => "sess_1" })

    turn.items.list(limit: 10)

    assert_equal({ "response_id" => "resp_1", "limit" => "10" },
                 @router.last(:get, "/v1/agents/sessions/sess_1/responses/items").query)
  end

  def test_a_fork_takes_the_history_up_to_a_response
    serve_session("sess_1")
    @router.on(:post, "/v1/agents/sessions/sess_1/fork", body: { "id" => "sess_2" })
    @router.on_socket("/v1/agents/sessions/sess_2/events") do |peer|
      peer.receive_type("close", timeout: 10)
      peer.close
    end

    parent = client.agent("docs").sessions.create
    fork = parent.fork(response_id: VA::AgentResponse.new(client, { "id" => "resp_1" }), title: "Branch")
    fork.close
    parent.close

    assert_equal "sess_2", fork.id
    assert_equal({ "response_id" => "resp_1", "title" => "Branch" },
                 @router.last(:post, "/v1/agents/sessions/sess_1/fork").json)
  end
end
