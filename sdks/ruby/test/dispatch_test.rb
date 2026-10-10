# frozen_string_literal: true

require_relative "test_helper"

class TestDispatch < LocalRouterTest
  def setup
    super
    # The connection stays open after the script returns; the test drives the peer.
    @router.on_socket("/v1/dispatch") { |_peer| nil }
  end

  def dispatch
    @dispatch ||= VA::Dispatch.new(capacity: 2, report_every: 0.05, client: client)
  end

  def run_until_closed
    runner = Thread.new { dispatch.run }
    peer = @router.peer
    yield peer
  ensure
    peer&.close
    runner&.join(5)
  end

  def test_a_call_handled_and_one_that_raises_are_each_reported_done
    handled = Thread::Queue.new
    dispatch.wait_for_call do |call|
      handled << call
      raise ArgumentError, "no agent for #{call.called_number}" if call.call_id == "c2"
    end

    run_until_closed do |peer|
      peer.send_frame(type: "ready", worker_id: "w1")
      peer.send_frame(type: "call", work_id: "wk1", call_id: "c1", called_number: "+15550100",
                      caller_number: "+15550199", custom: { "tier" => "gold", "n" => 1 })
      assert_equal({ "type" => "done", "work_id" => "wk1" }, peer.receive_type("done"))
      peer.send_frame(type: "call", work_id: "wk2", call_id: "c2", call_type: "sip", called_number: "+15550101")
      assert_equal({ "type" => "done", "work_id" => "wk2", "error" => "no agent for +15550101" },
                   peer.receive_type("done"))
    end

    first = handled.pop
    assert_equal "default", first.call_type
    assert_equal({ "tier" => "gold" }, first.custom)
    assert_equal "sip", handled.pop.call_type
    assert_equal "w1", dispatch.worker_id
  end

  def test_connecting_says_what_the_worker_holds_and_handles
    dispatch.wait_for_message { nil }

    run_until_closed do |peer|
      assert_equal({ "capacity" => "2", "active" => "0", "handles" => "message" }, peer.request.query)
    end
  end

  def test_a_worker_that_only_hosts_tools_handles_nothing_and_finishes_work_it_is_handed
    agent = client.agent("my-agent")
    agent.tools.register("weather_lookup", description: "Weather") { "sunny" }
    dispatch.host(agent)

    run_until_closed do |peer|
      assert_equal({ "capacity" => "2", "active" => "0", "handles" => "" }, peer.request.query)
      peer.send_frame(type: "message", work_id: "wk1", channel_id: "room-1", text: "hi")
      assert_equal({ "type" => "done", "work_id" => "wk1", "error" => "this worker answers no messages" },
                   peer.receive_type("done"))
      peer.send_frame(type: "call", work_id: "wk2", call_id: "c1")
      assert_equal({ "type" => "done", "work_id" => "wk2", "error" => "this worker answers no calls" },
                   peer.receive_type("done"))
    end
  end

  def test_a_message_carries_the_session_and_request_it_was_written_to
    handled = Thread::Queue.new
    dispatch.wait_for_message { |message| handled << message }

    run_until_closed do |peer|
      peer.send_frame(type: "message", work_id: "wk1", session_id: "sess_1", request_id: "req_1",
                      agent_id: "support", text: "hi", user_id: "ada")
      assert_equal({ "type" => "done", "work_id" => "wk1" }, peer.receive_type("done"))
    end

    message = handled.pop(timeout: 5)
    assert_equal "sess_1", message.session_id
    assert_equal "req_1", message.request_id
    assert_equal "", message.channel_id
  end

  def test_answer_has_the_model_answer_with_the_server_credential_acting_for_the_writer
    @router.on(:post, "/v1/agents/sessions/sess_1/responses", body: { "id" => "resp_1", "session_id" => "sess_1" })
    server = VA::Client.new(url: @router.url, api_key: "key", api_secret: "secret")
    worker = VA::Dispatch.new(client: server)
    message = VA::InboundMessage.from({ "session_id" => "sess_1", "request_id" => "req_1", "text" => "hi",
                                        "user_id" => "ada" })

    answered = worker.answer(message)

    assert_equal "resp_1", answered.id
    request = @router.last(:post, "/v1/agents/sessions/sess_1/responses")
    assert_equal({ "text" => "hi", "request_id" => "req_1" }, request.json)
    assert_equal "ada", request.headers["x-stream-user-id"]
    assert_equal "server", request.headers["stream-auth-type"]
    assert_raises(VA::ConfigurationError) { worker.answer(VA::InboundMessage.from({ "text" => "hi" })) }
  end

  def test_answer_behind_the_proxy_keeps_the_server_token
    @router.on(:post, "/v1/agents/sessions/sess_1/responses", body: { "id" => "resp_1", "session_id" => "sess_1" })
    server = VA::Client.new(url: @router.url, api_key: "key", api_secret: "secret", authenticate: true)
    message = VA::InboundMessage.from({ "session_id" => "sess_1", "text" => "hi", "user_id" => "ada" })

    VA::Dispatch.new(client: server).answer(message)

    headers = @router.last(:post, "/v1/agents/sessions/sess_1/responses").headers
    assert_equal "ada", headers["x-stream-user-id"]
    claims = JSON.parse(headers["authorization"].delete_prefix("Bearer ").split(".")[1].tr("-_", "+/").unpack1("m"))
    assert_equal true, claims["server"]
    refute claims.key?("user_id")
  end

  def test_a_message_a_session_is_holding_is_not_given_an_agent
    message = VA::InboundMessage.from({ "session_id" => "sess_1", "channel_id" => "room-1", "text" => "hi" })

    refused = assert_raises(VA::ConfigurationError) do
      dispatch.get_or_create_agent(message) { VA::Agent.new(config: "support", client: client) }
    end
    assert_match(/answer/, refused.message)
  end

  def test_the_worker_reports_its_load_and_times_the_round_trip
    dispatch.wait_for_call { nil }

    run_until_closed do |peer|
      assert_equal 0, peer.receive_type("load")["active_agents"]
      ping = peer.receive_type("ping")
      assert_kind_of Numeric, ping["at"]
      peer.send_frame(type: "pong", at: ping["at"])

      measured = nil
      5.times do
        measured = peer.receive_type("load")["latency_ms"]
        break if measured
      end
      assert_kind_of Integer, measured
    end
  end

  def test_a_message_is_answered_by_one_agent_per_channel
    agents = Thread::Queue.new
    dispatch.wait_for_message do |message|
      agents << dispatch.get_or_create_agent(message) { VA::Agent.new(config: "support", client: client) }
    end

    run_until_closed do |peer|
      peer.send_frame(type: "message", channel_id: "room-1", text: "hi", user_id: "ada")
      first = agents.pop(timeout: 5)
      peer.send_frame(type: "message", channel_id: "room-2", text: "hi")
      second = agents.pop(timeout: 5)
      refute_same first, second
    end
  end

  def test_a_hosted_tool_is_declared_on_every_ready_and_answered_without_blocking_the_socket
    release = Thread::Queue.new
    agent = client.agent("my-agent", name: "Max")
    tools = agent.tools
    tools.register("weather_lookup", description: "Weather for a place",
                                     parameters: { type: "object", properties: { location: { type: "string" } } }) do |args|
      { "location" => args["location"], "sky" => "sunny" }
    end
    tools.register("slow", description: "Waits to be released") { release.pop(timeout: 5) }
    tools.register("broken", description: "Always fails") { raise ArgumentError, "no sky today" }
    dispatch.host(agent, tool_timeout: 2.5)

    run_until_closed do |peer|
      peer.send_frame(type: "ready", worker_id: "w1")
      declared = peer.receive_type("host_tools")
      assert_equal "my-agent", declared["agent_id"]
      assert_equal 2500, declared["timeout_ms"]
      assert_equal %w[weather_lookup slow broken], declared["tools"].map { |tool| tool["name"] }
      assert_equal({ "type" => "object", "properties" => { "location" => { "type" => "string" } } },
                   declared["tools"].first["parameters"])

      peer.send_frame(type: "tool_call", id: "t1", session_id: "s1", name: "slow", arguments: "{}")
      peer.send_frame(type: "tool_call", id: "t2", session_id: "s1", name: "weather_lookup",
                      arguments: '{"location":"Colorado"}')
      assert_equal({ "type" => "tool_result", "id" => "t2", "output" => '{"location":"Colorado","sky":"sunny"}' },
                   peer.receive_type("tool_result"))
      eventually { dispatch.active == 1 }
      release << "done"
      assert_equal({ "type" => "tool_result", "id" => "t1", "output" => "done" }, peer.receive_type("tool_result"))

      peer.send_frame(type: "tool_call", id: "t3", name: "broken", arguments: "")
      assert_equal({ "type" => "tool_result", "id" => "t3", "error" => "no sky today" }, peer.receive_type("tool_result"))
      peer.send_frame(type: "tool_call", id: "t4", name: "missing", arguments: "{}")
      assert_equal({ "type" => "tool_result", "id" => "t4", "error" => "this worker does not run missing" },
                   peer.receive_type("tool_result"))

      peer.send_frame(type: "hosting", agent_id: "my-agent", tools: %w[weather_lookup slow broken])
      peer.send_frame(type: "ready", worker_id: "w2")
      assert_equal "my-agent", peer.receive_type("host_tools")["agent_id"]
    end
    assert_equal "w2", dispatch.worker_id
  end

  def test_a_worker_whose_tools_are_refused_stops_with_the_reason
    agent = client.agent("my-agent")
    agent.tools.register("weather_lookup", description: "Weather for a place") { "sunny" }
    dispatch.host(agent)
    runner = Thread.new { dispatch.run }
    runner.report_on_exception = false
    peer = @router.peer
    peer.send_frame(type: "ready", worker_id: "w1")
    assert_equal 0, peer.receive_type("host_tools")["timeout_ms"]
    peer.send_frame(type: "hosting_refused", agent_id: "my-agent", reason: "another customer's agent")

    refused = assert_raises(VA::Error) { runner.join(5) }
    assert_equal "the router refused to host tools for agent my-agent: another customer's agent", refused.message
  end

  def test_a_handler_or_hosted_tools_are_needed_before_running
    assert_raises(VA::ConfigurationError) { dispatch.run }
    assert_raises(VA::ConfigurationError) { dispatch.host(client.agent("my-agent")) }
  end
end
