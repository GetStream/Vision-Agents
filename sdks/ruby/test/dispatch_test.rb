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

  def test_a_call_handled_is_accepted_and_one_that_raises_is_rejected
    handled = Thread::Queue.new
    dispatch.wait_for_call do |call|
      handled << call
      raise ArgumentError, "no agent for #{call.called_number}" if call.call_id == "c2"
    end

    run_until_closed do |peer|
      assert_equal({ "capacity" => "2" }, peer.request.query)
      peer.send_frame(type: "ready", worker_id: "w1")
      peer.send_frame(type: "call", call_id: "c1", called_number: "+15550100", caller_number: "+15550199",
                      custom: { "tier" => "gold", "n" => 1 })
      assert_equal({ "type" => "accepted", "call_id" => "c1" }, peer.receive_type("accepted"))
      peer.send_frame(type: "call", call_id: "c2", call_type: "sip", called_number: "+15550101")
      assert_equal({ "type" => "rejected", "call_id" => "c2", "reason" => "no agent for +15550101" },
                   peer.receive_type("rejected"))
    end

    first = handled.pop
    assert_equal "default", first.call_type
    assert_equal({ "tier" => "gold" }, first.custom)
    assert_equal "sip", handled.pop.call_type
    assert_equal "w1", dispatch.worker_id
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
      assert_nil peer.receive_type("accepted", timeout: 0.2)
    end
  end

  def test_a_hosted_tool_is_declared_on_every_ready_and_answered_without_blocking_the_socket
    release = Thread::Queue.new
    tools = VA::Tools.new
    tools.register("weather_lookup", description: "Weather for a place",
                                     parameters: { type: "object", properties: { location: { type: "string" } } }) do |args|
      { "location" => args["location"], "sky" => "sunny" }
    end
    tools.register("slow", description: "Waits to be released") { release.pop(timeout: 5) }
    tools.register("broken", description: "Always fails") { raise ArgumentError, "no sky today" }
    dispatch.host("my-agent", tools, timeout: 2.5)

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
    tools = VA::Tools.new.register("weather_lookup", description: "Weather for a place") { "sunny" }
    dispatch.host("my-agent", tools)
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
    assert_raises(VA::ConfigurationError) { dispatch.host("my-agent", VA::Tools.new) }
  end
end
