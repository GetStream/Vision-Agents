# frozen_string_literal: true

require_relative "test_helper"

class TestToolCall < Minitest::Test
  BOOKING = { type: "object", properties: { flight_id: { type: "string" } } }.freeze

  def test_a_tools_policies_are_declared_and_the_routers_defaults_left_unsaid
    tools = VA::Tools.new
    tools.register("book_flight", description: "Book a flight", parameters: BOOKING, interruptible: true,
                                  timeout: 5, approval: " Book flight {flight_id}? ") { "booked" }
    tools.register("lookup", description: "Look it up") { "found" }

    booking, lookup = tools.declarations
    assert_equal true, booking[:interruptible]
    assert_equal 5000, booking[:timeout_ms]
    assert_equal({ title: "Book flight {flight_id}?" }, booking[:approval])
    %i[interruptible timeout_ms approval].each { |policy| refute lookup.key?(policy), policy }
  end

  def test_an_approval_naming_an_argument_the_tool_lacks_is_refused
    refused = assert_raises(VA::ConfigurationError) do
      VA::Tools.new.register("book_flight", description: "Book", parameters: BOOKING, approval: "Book {flight}?") { "" }
    end
    assert_match(/\{flight\}/, refused.message)
    assert_raises(VA::ConfigurationError) do
      VA::Tools.new.register("book_flight", description: "Book", approval: "   ") { "" }
    end
    assert_raises(VA::ConfigurationError) do
      VA::Tools.new.register("slow", description: "Slow", timeout: -1) { "" }
    end
  end

  def test_a_call_that_takes_longer_than_its_timeout_fails_and_is_stopped
    tool_call_context = VA::ToolCall.new(id: "t1")
    tools = VA::Tools.new.register("stuck", description: "Never finishes", timeout: 0.05) do
      sleep 1
      "never"
    end

    failed = assert_raises(VA::Error) { tools.call("stuck", "{}", tool_call_context) }
    assert_equal "stuck took longer than 0.05s", failed.message
    assert tool_call_context.cancelled?
  end

  def test_a_block_is_handed_its_call_and_a_one_argument_lambda_is_not
    seen = nil
    tools = VA::Tools.new
    tools.register("whoami", description: "Who asked") { |_args, tool_call_context| seen = tool_call_context.id }
    tools.register("plain", description: "Takes arguments only", &->(args) { args["x"] })

    tools.call("whoami", "{}", VA::ToolCall.new(id: "t1"))
    assert_equal "t1", seen
    assert_equal "7", tools.call("plain", '{"x":"7"}')
  end

  def test_a_call_without_a_socket_does_nothing
    tool_call_context = VA::ToolCall.new

    assert_nil tool_call_context.update("checking")
    assert_nil tool_call_context.say("One moment.")
    assert_nil tool_call_context.disallow_interruptions
    refute tool_call_context.cancelled?
  end

  def test_a_call_reports_how_it_is_going_and_a_call_stopped_first_is_not_held
    sent = []
    tool_call_context = VA::ToolCall.new(id: "t1", send: ->(frame) { sent << frame })

    tool_call_context.update("checking")
    tool_call_context.say("One moment.")
    tool_call_context.cancel!

    assert_equal [{ type: "tool_update", tool_call_id: "t1", text: "checking", verbatim: false },
                  { type: "tool_update", tool_call_id: "t1", text: "One moment.", verbatim: true }], sent
    assert_raises(VA::CallNotHeldError) { tool_call_context.disallow_interruptions }
  end
end
