# frozen_string_literal: true

require "fileutils"
require "minitest/autorun"
require "tmpdir"
require "getstream/vision_agents"
require_relative "support/local_router"

VA = GetStream::VisionAgents

# A test against a LocalRouter, torn down afterwards. Nothing here reads the environment:
# every client is built from the router's url and a customer id.
class LocalRouterTest < Minitest::Test
  CUSTOMER = "acme"

  def setup
    @router = LocalRouter.new
  end

  def teardown
    @router.close
  end

  def client
    @client ||= VA::Client.new(url: @router.url, customer_id: CUSTOMER)
  end

  # Waits for a condition a background thread makes true.
  def eventually(timeout: 5)
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + timeout
    until yield
      flunk "the condition did not hold within #{timeout}s" if Process.clock_gettime(Process::CLOCK_MONOTONIC) > deadline
      sleep 0.01
    end
  end

  # A session the router creates, with the events socket scripted by the block.
  def serve_session(id = "sess_1", created: {}, &script)
    @router.on(:post, "/v1/agents/sessions") do |request|
      { "id" => id, "user_id" => "agent-user", "status" => "running" }
        .merge(created).merge("call_id" => request.json["call_id"].to_s)
    end
    @router.on_socket("/v1/agents/sessions/#{id}/events") do |peer|
      script ? script.call(peer) : peer.receive_type("close", timeout: 10)
      peer.close
    end
  end
end
