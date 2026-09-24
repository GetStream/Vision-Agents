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

  def test_a_handler_is_needed_before_running
    assert_raises(VA::ConfigurationError) { dispatch.run }
  end
end
