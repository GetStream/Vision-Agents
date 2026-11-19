# frozen_string_literal: true

require "test_helper"

# The user_id: keyword every request method takes, and what a request without it still sends.
class TestRequestUser < LocalRouterTest
  CONFIG = "/v1/agents/configs/cfg_1"

  def setup
    super
    @router.on(:get, "/v1/agents/connectors", body: { "items" => [] })
    @router.on(:post, "/v1/agents/guests", body: { "id" => "g" })
    @router.on(:delete, "/v1/agents/users/ada/memories", status: 204)
    @router.on(:put, CONFIG, body: {})
    @router.on(:patch, CONFIG, body: {})
  end

  # Each request method, called with the options given, and the request the router saw.
  def calls(client, **options)
    {
      get: -> { client.get("/v1/agents/connectors", **options) && @router.last(:get, "/v1/agents/connectors") },
      post: lambda {
        client.post("/v1/agents/guests", body: { name: "x" }, **options) && @router.last(:post, "/v1/agents/guests")
      },
      delete: lambda {
        client.delete("/v1/agents/users/{user_id}/memories", path: { user_id: "ada" }, **options)
        @router.last(:delete, "/v1/agents/users/ada/memories")
      },
      put: lambda {
        client.put("/v1/agents/configs/{id}", path: { id: "cfg_1" }, body: { name: "x" }, **options)
        @router.last(:put, CONFIG)
      },
      patch: lambda {
        client.patch("/v1/agents/configs/{id}", path: { id: "cfg_1" }, body: { name: "x" }, **options)
        @router.last(:patch, CONFIG)
      },
      socket: lambda {
        assert_raises(VA::RouterError) { client.socket("/v1/dispatch", **options) }
        @router.last(:get, "/v1/dispatch")
      }
    }
  end

  def clients
    server = { url: @router.url, api_key: "key", api_secret: "secret" }
    {
      customer: VA::Client.new(url: @router.url, customer_id: CUSTOMER),
      server: VA::Client.new(**server, authenticate: false),
      proxy: VA::Client.new(**server, authenticate: true),
      server_user: VA::Client.new(**server, user_id: "bob", authenticate: false),
      proxy_user: VA::Client.new(**server, user_id: "bob", authenticate: true),
      acting: VA::Client.new(backend: VA::Backend.new(**server, authenticate: true).acting_for("ada"))
    }
  end

  # The headers a test compares: everything but the token, which is signed per second, and the
  # handshake's random key.
  def shown(headers)
    kept = headers.to_h.transform_keys(&:downcase).reject do |name, _|
      %w[authorization host user-agent accept-encoding content-length connection sec-websocket-key].include?(name)
    end
    token = headers.find { |name, _| name.downcase == "authorization" }&.last.to_s.delete_prefix("Bearer ")
    claims = (JSON.parse(token.split(".")[1].tr("-_", "+/").unpack1("m")).except("iat", "exp") if token.count(".") == 2)
    [kept.sort.to_h, claims]
  end

  def test_a_user_given_is_named_with_the_client_s_own_credential
    clients.each do |kind, client|
      calls(client, user_id: "alice").each do |method, call|
        headers, claims = shown(call.call.headers)
        assert_equal "alice", headers["x-stream-user-id"], "#{kind} #{method}"
        # Behind the proxy the server's token, not bob's: the request is still the backend's.
        assert_equal({ "server" => true }, claims, "#{kind} #{method}") unless kind == :customer
      end
    end
  end

  def test_an_empty_user_names_nobody_even_on_a_client_acting_for_one
    calls(clients[:acting], user_id: "").each do |method, call|
      refute call.call.headers.key?("x-stream-user-id"), method
    end
  end

  # Expected values are the base sha's (fa2a6a66d), probed with the same calls before this change.
  AUTH = {
    customer: [{ "x-customer-id" => "acme" }, nil],
    server: [{ "stream-auth-type" => "server", "x-api-key" => "key" }, { "server" => true }],
    proxy: [{ "api_key" => "key", "stream-auth-type" => "jwt" }, { "server" => true }],
    server_user: [{ "stream-auth-type" => "server", "x-api-key" => "key", "x-stream-user-id" => "bob" },
                  { "server" => true }],
    proxy_user: [{ "api_key" => "key", "stream-auth-type" => "jwt" }, { "user_id" => "bob" }],
    acting: [{ "api_key" => "key", "stream-auth-type" => "jwt", "x-stream-user-id" => "ada" }, { "server" => true }]
  }.freeze
  EXTRA = {
    get: { "accept" => "application/json" },
    delete: { "accept" => "application/json" },
    post: { "accept" => "application/json", "content-type" => "application/json" },
    put: { "accept" => "application/json", "content-type" => "application/json" },
    patch: { "accept" => "application/json", "content-type" => "application/json" },
    socket: { "sec-websocket-version" => "13", "upgrade" => "websocket" }
  }.freeze

  def test_a_request_without_a_user_is_the_one_sent_before
    clients.each do |kind, client|
      auth, claims = AUTH.fetch(kind)
      assert_equal [auth.sort.to_h, claims], shown(client.backend.headers), "#{kind} headers"
      calls(client).each do |method, call|
        assert_equal [auth.merge(EXTRA.fetch(method)).sort.to_h, claims], shown(call.call.headers), "#{kind} #{method}"
      end
    end
  end

  def test_dispatch_answers_for_the_writer_and_for_nobody_as_before
    @router.on(:post, "/v1/agents/sessions/sess_1/responses", body: { "id" => "resp_1", "session_id" => "sess_1" })
    worker = VA::Dispatch.new(client: clients[:server])

    worker.answer(VA::InboundMessage.from({ "session_id" => "sess_1", "request_id" => "req_1", "text" => "hi" }))

    request = @router.last(:post, "/v1/agents/sessions/sess_1/responses")
    assert_equal [{ "accept" => "application/json", "content-type" => "application/json",
                    "stream-auth-type" => "server", "x-api-key" => "key" }, { "server" => true }],
                 shown(request.headers)
    assert_equal({ "text" => "hi", "request_id" => "req_1" }, request.json)
  end
end
