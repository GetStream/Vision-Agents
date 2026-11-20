# frozen_string_literal: true

require "test_helper"

class TestConnections < LocalRouterTest
  CONNECTIONS = "/v1/agents/connections"

  def server
    @server ||= VA::Client.new(url: @router.url, api_key: "key", api_secret: "secret", authenticate: false)
  end

  # A client still built the deprecated way, acting for bob in every request.
  def acting_for_bob
    VA::Client.new(backend: VA::Backend.new(url: @router.url, api_key: "key", api_secret: "secret",
                                            authenticate: false).acting_for("bob"))
  end

  def test_a_user_s_connection_names_the_user_as_owner_and_header
    @router.on(:post, CONNECTIONS, status: 201, body: { "id" => "conn_1", "status" => "pending" })

    created = server.connections.create("linear", label: "work", user_id: "alice")

    assert_equal({ "id" => "conn_1", "status" => "pending" }, created)
    request = @router.last(:post, CONNECTIONS)
    assert_equal({ "connector_id" => "linear", "label" => "work",
                   "owner" => { "type" => "user", "user_id" => "alice" } }, request.json)
    assert_equal "alice", request.headers["x-stream-user-id"]
    assert_equal "server", request.headers["stream-auth-type"]
  end

  def test_no_user_is_the_app_s_connection_with_no_header
    @router.on(:post, CONNECTIONS, status: 201, body: { "id" => "conn_1" })

    server.connections.create("linear")
    server.connections.create("linear", label: "", user_id: "")

    @router.seen(:post, CONNECTIONS).each do |request|
      assert_equal({ "connector_id" => "linear", "owner" => { "type" => "app" } }, request.json)
      refute request.headers.key?("x-stream-user-id")
    end
  end

  def test_the_user_given_wins_over_whoever_the_client_acts_for
    @router.on(:post, CONNECTIONS, status: 201, body: { "id" => "conn_1" })
    @router.on(:get, "#{CONNECTIONS}/conn_1", body: { "id" => "conn_1" })
    client = acting_for_bob

    client.connections.create("linear", user_id: "alice")
    assert_equal "alice", @router.last(:post, CONNECTIONS).headers["x-stream-user-id"]
    client.connections.create("linear")
    refute @router.last(:post, CONNECTIONS).headers.key?("x-stream-user-id")
    client.connections.get("conn_1")
    refute @router.last(:get, "#{CONNECTIONS}/conn_1").headers.key?("x-stream-user-id")
  end

  def test_authorize_without_a_user_on_an_acting_client_names_nobody
    @router.on(:post, "#{CONNECTIONS}/conn_1/authorizations", status: 201, body: {})
    acting_for_bob.connections.authorize("conn_1")
    refute @router.last(:post, "#{CONNECTIONS}/conn_1/authorizations").headers.key?("x-stream-user-id")
  end

  def test_list_without_a_user_on_an_acting_client_names_nobody
    @router.on(:get, CONNECTIONS, body: { "items" => [], "has_more" => false })
    acting_for_bob.connections.list
    refute @router.last(:get, CONNECTIONS).headers.key?("x-stream-user-id")
  end

  def test_delete_without_a_user_on_an_acting_client_names_nobody
    @router.on(:delete, "#{CONNECTIONS}/conn_1", status: 204)
    acting_for_bob.connections.delete("conn_1")
    refute @router.last(:delete, "#{CONNECTIONS}/conn_1").headers.key?("x-stream-user-id")
  end

  def test_create_sends_the_connector_given
    @router.on(:post, CONNECTIONS, status: 201, body: {})
    server.connections.create("github")
    assert_equal "github", @router.last(:post, CONNECTIONS).json["connector_id"]
  end

  def test_authorize_sends_the_return_url_and_answers_the_consent
    consent = { "launch_url" => "https://router/consent", "handoff_token" => "tok",
                "expires_at" => "2026-11-19T12:00:00Z" }
    @router.on(:post, "#{CONNECTIONS}/conn_1/authorizations", status: 201, body: consent)

    answered = server.connections.authorize("conn_1", return_url: "https://app.example.com/settings", user_id: "alice")
    server.connections.authorize("conn_1")

    assert_equal consent, answered
    with_user, app = @router.seen(:post, "#{CONNECTIONS}/conn_1/authorizations")
    assert_equal({ "return_url" => "https://app.example.com/settings" }, with_user.json)
    assert_equal "alice", with_user.headers["x-stream-user-id"]
    assert_equal({}, app.json)
    refute app.headers.key?("x-stream-user-id")
  end

  def test_get_and_delete_name_the_user_only_when_given
    @router.on(:get, "#{CONNECTIONS}/conn_1", body: { "id" => "conn_1", "status" => "pending" })
    @router.on(:delete, "#{CONNECTIONS}/conn_1", status: 204)

    assert_equal "pending", server.connections.get("conn_1", user_id: "alice")["status"]
    server.connections.get("conn_1")
    assert_nil server.connections.delete("conn_1", user_id: "alice")
    server.connections.delete("conn_1")

    %i[get delete].each do |method|
      with_user, app = @router.seen(method, "#{CONNECTIONS}/conn_1")
      assert_equal "alice", with_user.headers["x-stream-user-id"], method
      refute app.headers.key?("x-stream-user-id"), method
      assert_equal({}, app.query, method)
    end
  end

  def test_list_reads_every_page_of_the_user_s_connections
    @router.on(:get, CONNECTIONS) do |request|
      if request.query["cursor"]
        { "items" => [{ "id" => "conn_1" }], "has_more" => false }
      else
        { "items" => [{ "id" => "conn_2" }], "has_more" => true, "next_cursor" => "c2" }
      end
    end

    listed = server.connections.list(connector: "linear", user_id: "alice")

    assert_equal %w[conn_2 conn_1], listed.map { |connection| connection["id"] }
    first, second = @router.seen(:get, CONNECTIONS)
    assert_equal({ "owner_type" => "user", "connector_id" => "linear", "limit" => "200" }, first.query)
    assert_equal({ "owner_type" => "user", "connector_id" => "linear", "limit" => "200", "cursor" => "c2" },
                 second.query)
    assert_equal %w[alice alice], [first, second].map { |request| request.headers["x-stream-user-id"] }
  end

  def test_list_without_a_user_is_the_app_s_and_names_nobody
    @router.on(:get, CONNECTIONS, body: { "items" => [], "has_more" => false })

    assert_equal [], server.connections.list
    request = @router.last(:get, CONNECTIONS)
    assert_equal({ "owner_type" => "app", "limit" => "200" }, request.query)
    refute request.headers.key?("x-stream-user-id")
  end

  def test_list_stops_on_a_page_that_has_no_more_or_no_cursor
    pages = [{ "items" => [{ "id" => "a" }], "has_more" => true, "next_cursor" => "" },
             { "items" => [{ "id" => "b" }], "has_more" => false, "next_cursor" => "c3" }]
    pages.each do |page|
      router = LocalRouter.new
      served = 0
      router.on(:get, CONNECTIONS) do
        served += 1
        served > 2 ? { "items" => [], "has_more" => false } : page
      end
      client = VA::Client.new(url: router.url, customer_id: CUSTOMER)

      assert_equal [page["items"].first], client.connections.list
      assert_equal 1, router.seen(:get, CONNECTIONS).size
    ensure
      router&.close
    end
  end

  def test_an_empty_id_is_refused_before_sending
    assert_raises(VA::ConfigurationError) { server.connections.get("") }
    assert_raises(VA::ConfigurationError) { server.connections.authorize(nil) }
    assert_raises(VA::ConfigurationError) { server.connections.delete("", user_id: "alice") }
    %i[get post delete].each { |method| assert_empty @router.seen(method, %r{\A/v1/agents/connections}) }
  end

  def test_another_user_s_connection_is_the_router_s_not_found
    @router.on(:get, "#{CONNECTIONS}/conn_1",
               status: 404, body: LocalRouter.failure("not_found", "not_found", "connection not found"))

    error = assert_raises(VA::RouterError) { server.connections.get("conn_1", user_id: "bob") }

    assert_equal 404, error.status
    assert_equal "getConnection", error.operation
  end

  # Api key and its user's token, no secret.
  def device
    VA::Client.new(url: @router.url, api_key: "key", token: "jean-token", authenticate: false)
  end

  def test_a_device_naming_a_user_is_refused_before_sending
    calls = {
      create: -> { device.connections.create("linear", user_id: "alice") },
      authorize: -> { device.connections.authorize("conn_1", user_id: "alice") },
      get: -> { device.connections.get("conn_1", user_id: "alice") },
      list: -> { device.connections.list(user_id: "alice") },
      delete: -> { device.connections.delete("conn_1", user_id: "alice") }
    }

    calls.each do |name, call|
      error = assert_raises(VA::ConfigurationError, name.to_s, &call)
      assert_match(/acts only as that user/, error.message)
    end
    %i[get post delete].each { |method| assert_empty @router.seen(method, %r{\A/v1/agents/connections}) }
  end

  def test_a_device_naming_no_user_sends_what_it_always_did
    @router.on(:get, "#{CONNECTIONS}/conn_1", body: { "id" => "conn_1" })
    @router.on(:get, CONNECTIONS, body: { "items" => [], "has_more" => false })

    [nil, ""].each do |user|
      device.connections.get("conn_1", user_id: user)
      device.connections.list(user_id: user)
    end

    sent = @router.seen(:get, %r{\A/v1/agents/connections})
    assert_equal 4, sent.size
    sent.each do |request|
      assert_equal "Bearer jean-token", request.headers["authorization"]
      assert_equal "jwt", request.headers["stream-auth-type"]
      refute request.headers.key?("x-stream-user-id")
    end
  end

  def test_a_customer_id_client_still_names_a_user
    @router.on(:post, CONNECTIONS, status: 201, body: { "id" => "conn_1" })
    page = VA::Client.new(url: @router.url, customer_id: CUSTOMER).as_user("jean", "jean-token")

    page.connections.create("linear", user_id: "alice")

    request = @router.last(:post, CONNECTIONS)
    assert_equal({ "type" => "user", "user_id" => "alice" }, request.json["owner"])
    assert_equal "alice", request.headers["x-stream-user-id"]
    assert_equal CUSTOMER, request.headers["x-customer-id"]
    refute request.headers.key?("authorization")
  end
end
