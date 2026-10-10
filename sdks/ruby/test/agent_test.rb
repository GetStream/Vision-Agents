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

  def test_join_opens_the_session_with_voice_on_its_own_call
    serve_session

    call, started = agent(cost_tracking: { env: "production", team: 7 }, memory_filter: { user_id: 123, plan: "pro" })
                    .join(participant_wait_timeout: 0, wait_for_end: false) do |session|
                      [session.call, session.voice.started?]
                    end

    assert_empty @router.seen(:post, CALLS)
    assert_equal VA::Edge::Call.new(id: "sess_1", type: "agent"), call
    assert started
    assert_equal({ "agent" => "support", "user_id" => "support", "user_name" => "support", "agent_id" => "support",
                   "tags" => { "env" => "production", "team" => "7" },
                   "memory" => { "user_id" => "123", "filter" => { "plan" => "pro" } },
                   "start_voice" => true }, session_request)
  end

  def test_join_is_not_given_a_call_to_name
    assert_raises(VA::ConfigurationError) { agent.join("call-1") }
    assert_empty @router.requests
  end

  def test_leaving_the_block_closes_the_session_and_returns_its_value
    closed = Thread::Queue.new
    serve_session { |peer| closed << peer.receive_type("close", timeout: 10) }

    value = agent.join(participant_wait_timeout: 0, wait_for_end: false) { :done }

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
    participants = agent.join { |session| session.participants }

    assert_equal [VA::Participant.new("p1", "ada", "Ada")], participants
    assert_operator Process.clock_gettime(Process::CLOCK_MONOTONIC) - started, :>=, 0.2
  end

  def test_a_participant_that_is_the_agent_is_not_somebody_in_the_call
    serve_session do |peer|
      peer.send_frame(type: "participant_joined", participant: { id: "a", user_id: "agent-user" })
      peer.receive_type("close", timeout: 10)
    end

    agent.join(participant_wait_timeout: 0, wait_for_end: false) do |session|
      refute session.wait_for_participant(timeout: 0.3)
      assert_empty session.participants
    end
  end

  def test_an_inbound_call_joins_the_session_it_was_handed
    serve_session do |peer|
      peer.send_frame(type: "heard", text: "hello")
      peer.receive_type("close", timeout: 10)
    end
    call = VA::InboundCall.from("call_id" => "agent:inbound-1", "session_id" => "inbound-1",
                                "called_number" => "+15550100", "caller_number" => "+15550199")

    agent.join(call, wait_for_end: false) { assert call.wait_for_phone_participant(timeout: 5) }

    assert_empty @router.seen(:post, CALLS)
    assert_equal "inbound-1", session_request["id"]
    assert_equal true, session_request["start_voice"]
    refute session_request.key?("call_id")
    assert_equal({ "number" => "+15550100" }, session_request["phone"])
  end

  def test_an_inbound_call_that_names_no_session_is_refused
    call = VA::InboundCall.from("call_id" => "pstn-1", "called_number" => "+15550100")

    assert_raises(VA::ConfigurationError) { agent.join(call) }
    assert_empty @router.seen(:post, "/v1/agents/sessions")
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

    response = support.join(participant_wait_timeout: 0, wait_for_end: false) do
      support.responses.create("greet the user")
    end

    assert_equal "resp_1", response.id
    sent = @router.last(:post, "/v1/agents/sessions/sess_1/responses").json
    assert_equal "greet the user", sent["text"]
    assert_match(/\A[0-9a-f]{32}\z/, sent["request_id"])
  end

  def test_each_text_question_carries_a_fresh_request_id_and_one_with_an_image_none
    serve_session
    @router.on(:post, "/v1/agents/sessions/sess_1/responses",
               body: { "id" => "resp_1", "session_id" => "sess_1", "status" => "running" })
    support = agent

    support.join(participant_wait_timeout: 0, wait_for_end: false) do
      support.responses.create("first")
      support.responses.create("second")
      support.responses.create("look", images: [{ url: "https://example.com/a.png" }])
    end

    sent = @router.seen(:post, "/v1/agents/sessions/sess_1/responses").map(&:json)
    assert_equal 2, sent.first(2).map { |body| body["request_id"] }.uniq.size
    sent.first(2).each { |body| assert_match(/\A[0-9a-f]{32}\z/, body["request_id"]) }
    assert_equal({ "text" => "look", "images" => [{ "url" => "https://example.com/a.png" }] }, sent[2])
  end

  def test_responses_need_a_conversation
    assert_raises(VA::ConfigurationError) { agent.responses }
  end

  def test_a_tool_call_is_answered_with_its_request_and_turn
    results = Thread::Queue.new
    serve_session do |peer|
      peer.send_frame(type: "tool_call", id: "t1", name: "weather", arguments: '{"city":"Paris"}',
                      request_id: "c1", turn_id: "turn_1")
      results << peer.receive_type("tool_result")
      peer.send_frame(type: "tool_call", id: "t2", name: "broken", arguments: "{}")
      results << peer.receive_type("tool_result")
      peer.receive_type("close", timeout: 10)
    end
    tools = VA::Tools.new
    tools.register("weather", description: "Weather for a city", executor: "client",
                              display_title: "Checking the sky") { |args| { city: args["city"], sky: "clear" } }
    tools.register("broken", description: "Always fails") { raise ArgumentError, "no sky today" }

    agent(tools: tools).join(participant_wait_timeout: 0, wait_for_end: false) do
      assert_equal({ "type" => "tool_result", "tool_call_id" => "t1", "output" => '{"city":"Paris","sky":"clear"}',
                     "request_id" => "c1", "turn_id" => "turn_1" }, results.pop(timeout: 5))
      assert_equal({ "type" => "tool_result", "tool_call_id" => "t2", "error" => "no sky today" },
                   results.pop(timeout: 5))
    end
    assert_equal [{ "name" => "weather", "description" => "Weather for a city",
                    "parameters" => { "type" => "object", "properties" => {} }, "executor" => "client",
                    "display_title" => "Checking the sky" },
                  { "name" => "broken", "description" => "Always fails",
                    "parameters" => { "type" => "object", "properties" => {} } }], session_request["tools"]
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

    agent(tools: tools).join(participant_wait_timeout: 0, wait_for_end: false) do
      assert_nil results.pop(timeout: 5)
    end
  end

  def test_events_report_what_the_conversation_did
    serve_session do |peer|
      peer.send_frame(type: "heard", text: "hi", participant: { id: "p1", user_id: "ada" })
      peer.send_frame(type: "responded", text: "hello", pending_work: true)
      peer.send_frame(type: "left")
    end

    events = agent.join(participant_wait_timeout: 0) { |session| session.events.to_a }

    assert_equal %w[heard responded left], events.map(&:kind)
    assert_equal "ada", events[0].participant.user_id
    assert events[1].pending_work
  end

  def test_the_events_socket_asks_for_what_was_wanted
    serve_session

    agent.join(participant_wait_timeout: 0, wait_for_end: false, interim: true) { nil }

    assert_equal({ "interim" => "true", "decisions" => "false" },
                 @router.last(:get, "/v1/agents/sessions/sess_1/events").query)
  end

  def test_a_session_nothing_can_watch_is_stopped_rather_than_deleted
    @router.on(:post, "/v1/agents/sessions", body: { "id" => "sess_9" })
    @router.on(:post, "/v1/agents/sessions/sess_9/stop", status: 204)

    assert_raises(VA::RouterError) { agent.join(participant_wait_timeout: 0) }
    assert_equal 1, @router.seen(:post, "/v1/agents/sessions/sess_9/stop").size
    assert_empty @router.seen(:delete, "/v1/agents/sessions/sess_9")
  end

  def test_chat_holds_the_conversation_in_writing
    serve_session

    started = agent.chat { |session| session.voice.started? }
    refute started
    %w[start_voice text conversation_id incognito].each { |key| refute session_request.key?(key), key }

    agent.chat(incognito: true) { nil }
    assert_equal true, session_request["incognito"]
    assert_empty @router.seen(:post, CALLS)
  end

  def test_voice_is_started_and_stopped_on_a_chat
    serve_session
    @router.on(:post, "/v1/agents/sessions/sess_1/voice", body: { "id" => "sess_1", "call_id" => "sess_1" })
    @router.on(:delete, "/v1/agents/sessions/sess_1/voice", body: { "id" => "sess_1", "call_id" => "" })
    support = agent

    support.chat do |session|
      assert_raises(VA::ConfigurationError) { support.monitor_url }
      assert_equal "sess_1", session.voice.start["call_id"]
      assert session.voice.started?
      assert support.monitor_url.start_with?("https://demo.example/join/sess_1?")
      session.voice.stop
      refute session.voice.started?
    end

    assert_equal 1, @router.seen(:post, "/v1/agents/sessions/sess_1/voice").size
    assert_equal 1, @router.seen(:delete, "/v1/agents/sessions/sess_1/voice").size
  end

  def test_reply_answers_the_conversation_the_message_came_from
    serve_session
    message = VA::InboundMessage.from("channel_id" => "room-1", "agent_id" => "support-1", "text" => "hi")

    agent.reply(message) { nil }

    assert_equal "support-1", session_request["agent_id"]
    %w[conversation_id text incognito].each { |key| refute session_request.key?(key), key }
  end

  def test_an_outbound_call_is_placed_for_the_session_the_agent_joins
    serve_session
    @router.on(:post, "/v1/phone/calls",
               status: 202, body: { "vendor_call_id" => "CA123", "status" => "queued", "session_id" => "placed-1" })

    agent(cost_tracking: { env: "production" })
      .outbound_call(from: "+15550100", to: "+15550199", ring_timeout: 30,
                     participant_wait_timeout: 0, wait_for_end: false) { nil }

    placed = @router.last(:post, "/v1/phone/calls").json
    assert_equal({ "from" => "+15550100", "to" => "+15550199",
                   "ring_timeout_seconds" => 30, "tags" => { "env" => "production" } }, placed)
    assert_equal "placed-1", session_request["id"]
    assert_equal true, session_request["start_voice"]
    assert_equal true, session_request["navigating"]
    assert_equal({ "number" => "+15550100", "vendor_call_id" => "CA123" }, session_request["phone"])
    assert_equal %w[phone agents], @router.requests.map { |r| r.path.split("/")[2] }.first(2)
  end

  def test_a_call_placed_for_no_session_is_not_joined
    @router.on(:post, "/v1/phone/calls", status: 202, body: { "vendor_call_id" => "CA123", "status" => "queued" })

    assert_raises(VA::RouterError) { agent.outbound_call(from: "+15550100", to: "+15550199") }
    assert_empty @router.seen(:post, "/v1/agents/sessions")
  end

  def test_what_the_code_sets_is_sent_and_nothing_else
    serve_session

    agent(name: "Ada", instructions: "Be brief",
          pipeline: { llm: "fast", language: "fr", max_tokens: 200, greeting: "Hello" })
      .chat { nil }

    request = session_request
    assert_equal "Ada", request["user_name"]
    assert_equal "ada", request["user_id"]
    refute request.key?("instructions")
    assert_equal({ "text" => "Hello" }, request["greeting"])
    assert_equal "fast", request["llm"]
    assert_equal ["fr"], request["languages"]
    assert_equal 200, request["max_tokens"]
    refute request.key?("stt")
  end

  def test_the_harness_is_synced_onto_the_config_and_never_sent_with_a_session
    serve_session
    @router.on(:post, "/v1/agents/sync", body: { "unchanged" => false })
    skills = [VA::Skill.new(name: "research", description: "Looks things up", instructions: "Search first",
                            deadline: 30)]
    support = agent(harness: "default", pipeline: { subagent: "llm-thinking" }, skills: skills,
                    sandbox: VA::Sandbox.daytona)

    support.chat { nil }
    support.sync

    request = session_request
    %w[harness subagent thinking_llm sandbox skills skill_names tasks].each { |key| refute request.key?(key), key }
    synced = @router.last(:post, "/v1/agents/sync").json
    assert_equal "default", synced["harness"]
    assert_equal "llm-thinking", synced["subagent"]
    refute synced.key?("thinking_llm")
    assert_equal "daytona", synced["sandbox"]
    assert_equal [{ "name" => "research", "description" => "Looks things up", "instructions" => "Search first",
                    "capture_video" => false, "deadline_ms" => 30_000, "config_id" => "" }], synced["skills"]
    expected = VA::Folder.fingerprint(VA::Folder.fingerprint("", "", "", skills, [], []),
                                      "defaultllm-thinkingdaytona", "map[]", [], [], [])
    assert_equal expected, synced["hash"]
  end

  def test_update_config_patches_only_what_it_is_given
    @router.on(:get, "/v1/agents/configs", body: [{ "name" => "support", "id" => "cfg_1" }])
    @router.on(:patch, "/v1/agents/configs/cfg_1") { |request| request.json.merge("id" => "cfg_1") }

    config = agent.update_config(guardrail: "Never quote prices.", visible_tools: ["athena_*"], llm: "fast")

    assert_equal "cfg_1", config["id"]
    assert_equal({ "guardrail" => "Never quote prices.", "visible_tools" => ["athena_*"], "llm" => "fast" },
                 @router.last(:patch, "/v1/agents/configs/cfg_1").json)
    assert_equal({ "name" => "support" }, @router.last(:get, "/v1/agents/configs").query)
  end

  def test_update_config_needs_a_stored_config
    @router.on(:get, "/v1/agents/configs", body: [])

    assert_raises(VA::Error) { agent.update_config(guardrail: "x") }
    assert_empty @router.seen(:patch, %r{\A/v1/agents/configs/})
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

    url = support.join(participant_wait_timeout: 0, wait_for_end: false) { support.monitor_url }

    assert url.start_with?("https://demo.example/join/sess_1?api_key=key&token=")
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

  def test_sync_sends_what_the_folder_leaves_to_dispatch
    root = folder
    File.write(File.join(root, "agent.yaml"), "name: jean\ndispatch:\n  text: enabled\n")
    @router.on(:post, "/v1/agents/sync", body: { "unchanged" => false })

    VA::Agent.new(folder: root, client: client).sync

    assert_equal({ "text" => "enabled" }, @router.last(:post, "/v1/agents/sync").json["dispatch"])
  ensure
    FileUtils.rm_rf(File.dirname(root))
  end

  def test_sync_sends_the_greeting_harness_plugins_pages_and_simulations_the_folder_declares
    root = folder
    File.write(File.join(root, "agent.yaml"),
               "name: jean\ngreeting:\n  text: Bonjour\n  mode: exact\nharness: default\nplugins: [linear]\n")
    File.write(File.join(root, "knowledge/urls.yaml"), "- url: https://example.com/plans\n  refresh_hours: 24\n")
    FileUtils.mkdir_p(File.join(root, "simulations"))
    File.write(File.join(root, "simulations/lunch.yaml"),
               "- name: lunch\n  scenario: Order a club\n  assertion: One club\n  variations: 3\n")
    @router.on(:post, "/v1/agents/sync", body: { "unchanged" => false })

    VA::Agent.new(folder: root, client: client).sync

    request = @router.last(:post, "/v1/agents/sync").json
    assert_equal({ "text" => "Bonjour", "mode" => "exact" }, request["greeting"])
    assert_equal "default", request["harness"]
    assert_equal ["linear"], request["plugins"]
    %w[speed agent_plugins].each { |key| refute request.key?(key), key }
    assert_equal [{ "url" => "https://example.com/plans", "refresh_hours" => 24 }], request["knowledge_urls"]
    assert_equal [{ "name" => "lunch", "scenario" => "Order a club", "assertion" => "One club", "variations" => 3 }],
                 request["simulations"]
    assert_equal VA::Folder.load(root).fingerprint, request["hash"]
  ensure
    FileUtils.rm_rf(File.dirname(root))
  end

  def test_simulations_are_sent_only_when_the_folder_has_a_directory_for_them
    root = folder
    File.write(File.join(root, "agent.yaml"), "name: jean\n")
    @router.on(:post, "/v1/agents/sync", body: { "unchanged" => false })

    VA::Agent.new(folder: root, client: client).sync
    FileUtils.mkdir_p(File.join(root, "simulations"))
    VA::Agent.new(folder: root, client: client).sync

    without, empty = @router.seen(:post, "/v1/agents/sync").map(&:json)
    refute without.key?("simulations")
    assert_equal [], empty["simulations"]
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

    page = agent.knowledge.add_url("https://example.com/pricing", title: "Pricing", refresh_hours: 24)

    assert_equal "indexed", page["state"]
    assert_equal({ "namespace" => "support", "url" => "https://example.com/pricing", "title" => "Pricing",
                   "refresh_hours" => 24 }, @router.last(:post, "/v1/agents/knowledge/urls").json)
  end
end

class TestSessions < LocalRouterTest
  def test_sessions_are_queried_by_agent_a_page_at_a_time
    queries = []
    @router.on(:post, "/v1/agents/sessions/query") do |request|
      queries << request.json
      { "items" => [{ "id" => "s#{queries.size}" }], "has_more" => queries.size == 1, "next_cursor" => "c1" }
    end
    sessions = client.agent("docs").sessions

    first = sessions.query(user_id: "u1", state: "live", agent_id: "docs-1", limit: 10)
    sessions.query(user_id: "u1", state: "live", agent_id: "docs-1", limit: 10, cursor: first["next_cursor"])
    found = sessions.search("billing", modality: "text")

    assert_equal [{ "id" => "s1" }], first["items"]
    assert_equal [{ "id" => "s3" }], found["items"]
    filter = { "agent" => "docs", "user_id" => "u1", "state" => "live", "agent_id" => "docs-1" }
    assert_equal [{ "filter" => filter, "limit" => 10 }, { "filter" => filter, "limit" => 10, "cursor" => "c1" },
                  { "filter" => { "agent" => "docs", "modality" => "text", "text" => { "$q" => "billing" } } }],
                 queries
  end

  def test_create_opens_a_written_conversation
    serve_session

    session = client.agent("docs").sessions.create(id: "billing_ada-1", title: "Is Stream better?",
                                                   project_id: "pricing")
    session.close

    request = @router.last(:post, "/v1/agents/sessions").json
    assert_equal({ "agent" => "docs", "user_id" => "docs", "user_name" => "docs", "agent_id" => "docs",
                   "id" => "billing_ada-1", "title" => "Is Stream better?", "project_id" => "pricing" }, request)
    refute session.voice.started?
  end

  def test_a_conversation_is_resumed_by_its_session_id
    @router.on(:get, "/v1/agents/sessions/s1", body: { "id" => "s1", "user_id" => "docs", "call_id" => "" })
    @router.on_socket("/v1/agents/sessions/s1/events") do |peer|
      peer.send_frame(type: "responding")
      peer.receive_type("close", timeout: 10)
      peer.close
    end

    session = client.agent("docs").sessions.resume("s1")
    kind = session.events.first.kind
    session.close

    assert_equal "s1", session.id
    assert_equal "responding", kind
    assert_empty @router.seen(:post, "/v1/agents/sessions")
  end

  def test_update_changes_one_session
    serve_session
    @router.on(:patch, "/v1/agents/sessions/sess_1", body: { "id" => "sess_1", "llm" => "llm-thinking" })

    session = client.agent("docs").sessions.create
    updated = session.update(title: "Pricing", llm: "llm-thinking", thinking: "high", sts: "")
    session.close

    assert_equal "llm-thinking", updated["llm"]
    assert_equal({ "title" => "Pricing", "llm" => "llm-thinking", "thinking" => "high", "sts" => "" },
                 @router.last(:patch, "/v1/agents/sessions/sess_1").json)
    assert_raises(VA::ConfigurationError) { session.update(model: "llm-fast") }
    assert_raises(VA::ConfigurationError) { session.update(subagent: "llm-thinking") }
  end

  def test_an_ended_session_is_renamed_deleted_and_forgotten_by_id
    @router.on(:patch, "/v1/agents/sessions/s1") { |request| request.json.merge("id" => "s1") }
    @router.on(:delete, "/v1/agents/sessions/s1/memories", status: 204)
    @router.on(:delete, "/v1/agents/sessions/s1", status: 204)
    sessions = client.agent("docs").sessions

    assert_equal "Pricing", sessions.update("s1", title: "Pricing")["title"]
    assert_nil sessions.delete_memories("s1")
    assert_nil sessions.delete("s1")
    assert_equal 1, @router.seen(:delete, "/v1/agents/sessions/s1/memories").size
    assert_equal 1, @router.seen(:delete, "/v1/agents/sessions/s1").size
  end

  def test_closing_a_session_stops_it_and_only_delete_deletes_it
    closed = Thread::Queue.new
    serve_session { |peer| closed << peer.receive_type("close", timeout: 10) }
    @router.on(:delete, "/v1/agents/sessions/sess_1/memories", status: 204)
    @router.on(:delete, "/v1/agents/sessions/sess_1", status: 204)
    session = client.agent("docs").sessions.create

    session.delete_memories
    session.close
    assert_equal({ "type" => "close" }, closed.pop(timeout: 5))
    assert_empty @router.seen(:delete, "/v1/agents/sessions/sess_1")
    session.delete

    assert_equal 1, @router.seen(:delete, "/v1/agents/sessions/sess_1/memories").size
    assert_equal 1, @router.seen(:delete, "/v1/agents/sessions/sess_1").size
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
    refused = LocalRouter.failure("invalid_request", "invalid_request", "fork a persisted conversation instead")
    @router.on(:post, "/v1/agents/sessions/sess_1/rewind", status: 400, body: refused)

    error = assert_raises(VA::RouterError) { responses.rewind("resp_1") }
    assert_equal 400, error.status
    assert_equal "rewindSession: fork a persisted conversation instead", error.message
  end

  def test_rewind_needs_an_id
    assert_raises(VA::ConfigurationError) { responses.rewind(VA::AgentResponse.new(client, { "id" => "" })) }
  end

  def test_items_are_read_a_page_at_a_time_by_cursor
    @router.on(:get, "/v1/agents/sessions/sess_1/responses/items") do |request|
      start = request.query["cursor"].to_i
      ids = (start...[start + 2, 3].min).map { |i| { "id" => "item_#{i}" } }
      more = start + 2 < 3
      { "items" => ids, "has_more" => more, "next_cursor" => (more ? (start + 2).to_s : nil) }.compact
    end

    items = responses.items.each(page: 2).to_a

    assert_equal %w[item_0 item_1 item_2], items.map { |item| item["id"] }
    assert_equal [{ "limit" => "2" }, { "limit" => "2", "cursor" => "2" }],
                 @router.seen(:get, "/v1/agents/sessions/sess_1/responses/items").map(&:query)
  end

  def test_one_turns_items
    @router.on(:get, "/v1/agents/sessions/sess_1/responses/items", body: { "items" => [], "has_more" => false })
    turn = VA::AgentResponse.new(client, { "id" => "resp_1", "session_id" => "sess_1" })

    turn.items.list(limit: 10, cursor: "c1")

    assert_equal({ "response_id" => "resp_1", "limit" => "10", "cursor" => "c1" },
                 @router.last(:get, "/v1/agents/sessions/sess_1/responses/items").query)
  end

  def test_turns_are_listed_a_page_at_a_time
    @router.on(:get, "/v1/agents/sessions/sess_1/responses",
               body: { "items" => [{ "id" => "resp_1" }], "has_more" => true, "next_cursor" => "c1" })

    page = responses.list(limit: 1)

    assert_equal "c1", page["next_cursor"]
    assert_equal({ "limit" => "1" }, @router.last(:get, "/v1/agents/sessions/sess_1/responses").query)
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
