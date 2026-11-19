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
    tool_call_context = VA::ToolCallContext.new(id: "t1")
    tools = VA::Tools.new.register("stuck", description: "Never finishes", timeout: 0.05) do
      sleep 1
      "never"
    end

    failed = assert_raises(VA::Error) { tools.call("stuck", "{}", tool_call_context) }
    assert_equal "stuck took longer than 0.05s", failed.message
    assert tool_call_context.cancelled?
  end

  def test_a_block_of_two_is_handed_its_call_first_and_a_block_of_one_only_the_arguments
    tools = VA::Tools.new
    tools.register("whoami", description: "Who asked") { |tool_call_context, args| "#{tool_call_context.id} #{args["x"]}" }
    tools.register("plain", description: "Takes arguments only") { |args| args["x"] }
    tools.register("lambda", description: "Takes arguments only", &->(args) { args["x"] })
    tools.register("none", description: "Takes nothing") { "done" }

    assert_equal "t1 7", tools.call("whoami", '{"x":"7"}', VA::ToolCallContext.new(id: "t1"))
    assert_equal "7", tools.call("plain", '{"x":"7"}', VA::ToolCallContext.new(id: "t1"))
    assert_equal "7", tools.call("lambda", '{"x":"7"}')
    assert_equal "done", tools.call("none", "{}")
  end

  def test_params_declares_each_argument_a_required_string
    tools = VA::Tools.new.register("weather_lookup", description: "Weather", params: { location: "a city or region" },
                                                     approval: "Weather in {location}?") { |location:| location }

    assert_equal({ type: "object", properties: { location: { type: "string", description: "a city or region" } },
                   required: ["location"] }, tools.declarations.first[:parameters])
  end

  def test_params_and_parameters_together_are_refused
    assert_raises(ArgumentError) do
      VA::Tools.new.register("weather_lookup", description: "Weather", params: { location: "a city" },
                                               parameters: BOOKING) { "" }
    end
  end

  def test_a_block_of_keywords_is_handed_the_arguments_it_names
    tools = VA::Tools.new
    tools.register("weather_lookup", description: "Weather", params: { location: "a city" }) do |location:, units: "C"|
      "#{location} #{units}"
    end
    tools.register("everything", description: "All of it") { |location:, **rest| [location, rest] }

    assert_equal "Paris C", tools.call("weather_lookup", '{"location":"Paris","mood":"sunny"}')
    assert_equal "Oslo F", tools.call("weather_lookup", '{"location":"Oslo","units":"F"}')
    assert_equal '["Paris",{"mood":"sunny"}]', tools.call("everything", '{"location":"Paris","mood":"sunny"}')
  end

  def test_a_block_of_keywords_missing_one_it_requires_fails_the_call
    tools = VA::Tools.new
    tools.register("weather_lookup", description: "Weather", params: { location: "a city" }) { |location:| location }

    missing = assert_raises(ArgumentError) { tools.call("weather_lookup", '{"city":"Paris"}') }
    assert_match(/location/, missing.message)
  end

  def test_a_block_of_keywords_with_a_positional_argument_is_handed_its_call_first
    tools = VA::Tools.new
    tools.register("weather_lookup", description: "Weather",
                                     params: { location: "a city" }) do |tool_call_context, location:|
      "#{tool_call_context.id} #{location}"
    end

    assert_equal "t1 Paris", tools.call("weather_lookup", '{"location":"Paris"}', VA::ToolCallContext.new(id: "t1"))
  end

  def test_a_call_without_a_socket_does_nothing
    tool_call_context = VA::ToolCallContext.new

    assert_nil tool_call_context.disallow_interruptions
    refute tool_call_context.cancelled?
  end

  def test_a_call_stopped_first_is_not_held
    sent = []
    tool_call_context = VA::ToolCallContext.new(id: "t1", send: ->(frame) { sent << frame })

    tool_call_context.cancel!

    assert_empty sent
    assert_raises(VA::CallNotHeldError) { tool_call_context.disallow_interruptions }
  end
end
