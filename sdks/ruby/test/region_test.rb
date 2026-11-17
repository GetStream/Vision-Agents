# frozen_string_literal: true

require_relative "test_helper"

# Sends what a client opens to the hosted router to a LocalRouter instead, leaving the
# request as the SDK built it: its URL, and so its Host header, stay the regional ones.
module RegionalRedirect
  class << self
    attr_accessor :router, :hosts
  end

  def start(address, *args, **options, &)
    router = RegionalRedirect.router
    return super unless router

    RegionalRedirect.hosts << address
    port = URI(router.url).port
    super("127.0.0.1", port, **options, use_ssl: false, &)
  end
end
Net::HTTP.singleton_class.prepend(RegionalRedirect)

class TestRegion < LocalRouterTest
  REGIONAL_HOST = "accelerate-us-east4.gcp.stream-io-api.com"
  CLEARED = [VA::Backend::URL_ENV, VA::Backend::REGION_ENV, VA::Backend::CUSTOMER_ENV,
             VA::Backend::AUTHENTICATE_ENV].freeze
  CREDENTIALS = { VA::Backend::API_KEY_ENV => "test-key", VA::Backend::API_SECRET_ENV => "test-secret" }.freeze

  def setup
    super
    @saved = (CLEARED + CREDENTIALS.keys).to_h { |name| [name, ENV.fetch(name, nil)] }
    CLEARED.each { |name| ENV.delete(name) }
    ENV.update(CREDENTIALS)
    @router.on(:get, "/v1/agents/simulations", body: [{ "id" => "sim_1" }])
    RegionalRedirect.hosts = []
    RegionalRedirect.router = @router
  end

  def teardown
    RegionalRedirect.router = nil
    @saved.each { |name, value| value.nil? ? ENV.delete(name) : ENV[name] = value }
    super
  end

  def assert_reads_simulations(api)
    assert_equal [{ "id" => "sim_1" }], api.simulations.list

    assert_equal [REGIONAL_HOST], RegionalRedirect.hosts
    request = @router.last(:get, "/v1/agents/simulations")
    assert_equal REGIONAL_HOST, request.headers["host"]
    assert_equal "test-key", request.headers["api_key"]
    assert_equal "jwt", request.headers["stream-auth-type"]
    refute_empty request.headers["authorization"].to_s
  end

  def test_the_default_client_reads_simulations
    assert_reads_simulations(VA::Client.new)
  end

  def test_an_explicit_region_reads_simulations
    assert_reads_simulations(VA::Client.new(region: "us-east4"))
  end

  def test_a_region_from_the_environment_reads_simulations
    ENV[VA::Backend::REGION_ENV] = "us-east4"

    assert_reads_simulations(VA::Client.new)
  end

  def test_a_conflicting_url_and_region_are_refused
    assert_raises(VA::ConfigurationError) { VA::Client.new(region: "europe-west4", url: VA::Backend::DEFAULT_URL) }
  end

  def test_an_invalid_region_with_a_url_is_refused
    assert_raises(VA::ConfigurationError) do
      VA::Client.new(region: "us-east4.evil.com/x", url: VA::Backend::DEFAULT_URL)
    end
  end

  def test_a_regional_client_can_act_for_a_user
    user = VA::Client.new(region: "us-east4").as_user("jean", "user-token")

    assert_equal VA::Backend::DEFAULT_URL, user.backend.url
    assert_equal "jean", user.backend.user_id
    assert_reads_simulations(user)
    assert_equal "Bearer user-token", @router.last(:get, "/v1/agents/simulations").headers["authorization"]
  end
end
