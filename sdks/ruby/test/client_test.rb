# frozen_string_literal: true

require_relative "test_helper"

class TestClient < LocalRouterTest
  def test_a_request_carries_the_customer_and_the_spec_path
    @router.on(:get, "/v1/agents/configs", body: [{ "name" => "docs" }])

    assert_equal [{ "name" => "docs" }], client.get("/v1/agents/configs", query: { name: "docs" })

    request = @router.last(:get, "/v1/agents/configs")
    assert_equal CUSTOMER, request.headers["x-customer-id"]
    assert_equal({ "name" => "docs" }, request.query)
  end

  def test_path_parameters_are_escaped
    @router.on(:get, %r{\A/v1/agents/sessions/}, body: { "id" => "a b/c" })

    client.get("/v1/agents/sessions/{id}", path: { id: "a b/c" })

    assert_equal "/v1/agents/sessions/a%20b%2Fc", @router.requests.last.path
  end

  def test_query_values_are_rendered_for_the_wire
    @router.on(:get, "/v1/agents/sessions", body: [])

    client.get("/v1/agents/sessions",
               query: { custom: { "tenant" => "acme" }, created_after: Time.utc(2026, 1, 2, 3, 4, 5), limit: 5,
                        project: nil })

    assert_equal({ "custom" => '{"tenant":"acme"}', "created_after" => "2026-01-02T03:04:05Z", "limit" => "5" },
                 @router.last(:get, "/v1/agents/sessions").query)
  end

  def test_nil_body_fields_are_left_out
    @router.on(:post, "/v1/agents/guests", body: { "id" => "guest_1" })

    client.post("/v1/agents/guests", body: { name: "Ada", id: nil })

    assert_equal({ "name" => "Ada" }, @router.last(:post, "/v1/agents/guests").json)
  end

  def test_what_the_spec_does_not_have_is_refused_before_sending
    assert_raises(VA::ConfigurationError) { client.get("/v1/nothing") }
    assert_raises(VA::ConfigurationError) { client.get("/v1/agents/configs", query: { nme: "docs" }) }
    assert_raises(VA::ConfigurationError) { client.post("/v1/agents/guests", body: { nmae: "Ada" }) }
    assert_raises(VA::ConfigurationError) { client.post("/v1/agents/sessions/{id}/rewind", path: { id: "s" }, body: {}) }
    assert_raises(VA::ConfigurationError) { client.get("/v1/agents/sessions/{id}") }
    assert_raises(VA::ConfigurationError) { client.get("/v1/dispatch") }
    assert_empty @router.requests
  end

  def test_a_no_content_answer_is_nil
    @router.on(:delete, "/v1/agents/sessions/s1", status: 204)

    assert_nil client.delete("/v1/agents/sessions/{id}", path: { id: "s1" })
  end

  def test_a_refusal_carries_the_status_the_operation_and_the_message
    @router.on(:post, "/v1/agents/sessions", status: 429, body: { "error" => "slow down" })

    error = assert_raises(VA::RouterError) { client.post("/v1/agents/sessions", body: { text: true }) }

    assert_equal 429, error.status
    assert_equal "createSession", error.operation
    assert_equal "createSession: slow down", error.message
    assert_equal({ "error" => "slow down" }, error.body)
  end

  def test_a_router_that_is_not_there_is_status_zero
    gone = VA::Client.new(url: "http://127.0.0.1:1", customer_id: CUSTOMER)

    error = assert_raises(VA::RouterError) { gone.get("/v1/agents/configs") }

    assert_equal 0, error.status
  end

  def test_a_socket_that_is_refused_is_a_router_error
    error = assert_raises(VA::RouterError) { client.socket("/v1/dispatch") }

    assert_equal 404, error.status
  end

  def test_a_guest_is_minted_and_acted_for
    @router.on(:post, "/v1/agents/guests", body: { "id" => "guest_1", "token" => "tok", "name" => "Guest" })
    @router.on(:get, "/v1/agents/sessions", body: [])
    keyed = VA::Client.new(url: @router.url, api_key: "key", api_secret: "secret")

    guest = keyed.guest_user(name: "Ada")
    keyed.as_guest(guest).get("/v1/agents/sessions")

    assert_equal({ "name" => "Ada" }, @router.last(:post, "/v1/agents/guests").json)
    headers = @router.last(:get, "/v1/agents/sessions").headers
    assert_equal "Bearer tok", headers["authorization"]
    assert_equal "jwt", headers["stream-auth-type"]
    assert_equal "key", headers["x-api-key"]
  end

  def test_claiming_a_guest_is_server_side_only
    @router.on(:post, "/v1/agents/guests/claim",
               body: { "guest_id" => "guest_1", "user_id" => "ada", "sessions_moved" => 2 })
    keyed = VA::Client.new(url: @router.url, api_key: "key", api_secret: "secret")

    assert_equal 2, keyed.claim_guest_user("guest_1", "ada")["sessions_moved"]
    assert_equal({ "guest_id" => "guest_1", "user_id" => "ada" }, @router.last(:post, "/v1/agents/guests/claim").json)

    device = VA::Client.new(url: @router.url, api_key: "key", token: "user-token", user_id: "ada")
    assert_raises(VA::ConfigurationError) { device.claim_guest_user("guest_1", "ada") }
  end
end

class TestBackend < Minitest::Test
  def test_a_server_credential_signs_a_server_token
    headers = VA::Backend.new(url: "http://router", api_key: "key", api_secret: "secret", user_id: "ada").headers

    assert_equal "server", headers["Stream-Auth-Type"]
    assert_equal "ada", headers["X-Stream-User-Id"]
    header, payload, signature = headers["Authorization"].delete_prefix("Bearer ").split(".")
    expected = VA::Backend.encode(OpenSSL::HMAC.digest("SHA256", "secret", "#{header}.#{payload}"))
    assert_equal expected, signature
    assert_equal true, JSON.parse(payload.tr("-_", "+/").unpack1("m"))["server"]
  end

  def test_behind_the_proxy_a_user_token_is_minted
    headers = VA::Backend.new(url: "http://router", api_key: "key", api_secret: "secret", user_id: "ada",
                              authenticate: true).headers

    assert_equal "jwt", headers["stream-auth-type"]
    payload = headers["Authorization"].split(".")[1]
    assert_equal "ada", JSON.parse(payload.tr("-_", "+/").unpack1("m"))["user_id"]
  end

  def test_a_key_without_a_secret_or_token_is_refused
    assert_raises(VA::ConfigurationError) { VA::Backend.new(url: "http://router", api_key: "key", api_secret: "") }
  end

  def test_sockets_use_the_websocket_scheme
    backend = VA::Backend.new(url: "https://router.example/", customer_id: "acme")

    assert_equal "wss://router.example/v1/dispatch", backend.socket_url("/v1/dispatch")
    assert backend.server_side?
  end
end
