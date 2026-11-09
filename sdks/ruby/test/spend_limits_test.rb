# frozen_string_literal: true

require_relative "test_helper"

class TestUSD < Minitest::Test
  def test_dollars_are_held_exactly_in_micros
    assert_equal 20_000_000_000, VA.usd(20_000).micros
    assert_equal 12_500_000, VA.usd("12.50").micros
    assert_equal 1, VA.usd("0.000001").micros
    assert_equal 250_000, VA.usd(Rational(1, 4)).micros
    assert_equal VA.usd("0.30"), VA.usd("0.10") + VA.usd("0.20")
    assert_equal VA.usd(10), VA.usd(25) - 15
    assert_operator VA.usd(50), :>, VA.usd("49.999999")
    five = VA.usd(5)
    assert_same five, VA.usd(five)
  end

  def test_an_amount_prints_as_dollars
    assert_equal "$20000.00", VA.usd(20_000).to_s
    assert_equal "$0.000125", VA::USD.micros(125).to_s
    assert_equal "$12.50", VA.usd("12.5").to_s
    assert_equal "-$1.00", (VA.usd(0) - 1).to_s
  end

  def test_what_is_not_an_exact_amount_is_refused
    [0.1, nil, "", "12.5.0", "$12", "1.0000001", Rational(1, 3), "1e3"].each do |amount|
      assert_raises(VA::ConfigurationError, amount.inspect) { VA.usd(amount) }
    end
    assert_raises(VA::ConfigurationError) { VA::USD.micros(1.5) }
    assert_raises(NoMethodError) { VA::USD.new(5) }
  end
end

class TestSpendLimits < LocalRouterTest
  LIMIT = { "id" => 7, "subject" => "api_key", "interval" => "daily", "limit_micros" => 10_000_000 }.freeze

  def limits
    client.spend_limits
  end

  def test_an_organization_limit_is_created_in_micros
    @router.on(:post, "/v1/spend/limits", status: 201, body: LIMIT)

    assert_equal LIMIT, limits.create(:organization, interval: :monthly, amount: 20_000)

    assert_equal({ "subject" => "organization", "interval" => "monthly", "limit_micros" => 20_000_000_000 },
                 @router.last(:post, "/v1/spend/limits").json)
  end

  def test_a_key_s_limit_names_its_app_key_level_and_note
    @router.on(:post, "/v1/spend/limits", status: 201, body: LIMIT)

    limits.create("api_key", app_id: "1234", interval: "daily", amount: VA.usd("10"), level: :api_key,
                             api_key: "mz7kq2xa9b", note: "CI key")

    assert_equal({ "subject" => "api_key", "interval" => "daily", "limit_micros" => 10_000_000, "app_id" => "1234",
                   "api_key" => "mz7kq2xa9b", "level" => "api_key", "note" => "CI key" },
                 @router.last(:post, "/v1/spend/limits").json)
  end

  def test_a_team_or_category_within_an_app_is_named_for_the_router
    @router.on(:post, "/v1/spend/limits", status: 201, body: LIMIT)

    limits.create(:team, team: "finance", app_id: "1234", interval: :weekly, amount: 100)
    assert_equal "app_team", @router.last(:post, "/v1/spend/limits").json["subject"]
    limits.create(:category, category: "project:onboarding", interval: :weekly, amount: 100)
    assert_equal "category", @router.last(:post, "/v1/spend/limits").json["subject"]
  end

  def test_a_list_is_filtered_and_read_page_by_page
    @router.on(:get, "/v1/spend/limits") do |request|
      if request.query["cursor"]
        { "items" => [LIMIT.merge("id" => 8)], "has_more" => false }
      else
        { "items" => [LIMIT], "has_more" => true, "next_cursor" => "c2" }
      end
    end

    listed = limits.list(subject: :api_key, app_id: "1234", api_key: "mz7kq2xa9b", interval: :daily, level: "api_key")

    assert_equal([7, 8], listed.map { |limit| limit["id"] })
    first, second = @router.seen(:get, "/v1/spend/limits").map(&:query)
    assert_equal({ "subject" => "api_key", "app_id" => "1234", "api_key_id" => "mz7kq2xa9b", "level" => "api_key",
                   "interval" => "daily", "limit" => "200" }, first)
    assert_equal "c2", second["cursor"]
  end

  def test_a_limit_is_read_changed_and_deleted_by_id
    @router.on(:get, "/v1/spend/limits/7", body: LIMIT)
    @router.on(:patch, "/v1/spend/limits/7") { |request| LIMIT.merge(request.json) }
    @router.on(:delete, "/v1/spend/limits/7", status: 204)

    assert_equal LIMIT, limits.get(7)
    assert_equal 25_000_000, limits.set_amount(7, 25)["limit_micros"]
    assert_equal({ "limit_micros" => 25_000_000 }, @router.last(:patch, "/v1/spend/limits/7").json)
    limits.set_note(7, "")
    assert_equal({ "note" => "" }, @router.last(:patch, "/v1/spend/limits/7").json)
    assert_nil limits.delete(7)
    assert_equal 1, @router.seen(:delete, "/v1/spend/limits/7").size
  end

  def test_a_reset_names_the_subject_the_reason_and_the_intervals
    @router.on(:post, "/v1/spend/limits/usage/reset", body: { "subject" => "user", "windows" => [] })

    limits.reset(:user, app_id: "1234", user_id: "user-42", reason: "support ticket", intervals: [:weekly])

    assert_equal({ "subject" => "user", "app_id" => "1234", "user_id" => "user-42", "reason" => "support ticket",
                   "intervals" => ["weekly"] }, @router.last(:post, "/v1/spend/limits/usage/reset").json)
  end

  def test_usage_and_grants_name_the_subject
    @router.on(:get, "/v1/spend/limits/usage", body: [{ "subject" => "user", "windows" => [] }])
    @router.on(:post, "/v1/spend/limits/grants", status: 201, body: { "id" => 3 })

    assert_equal "user", limits.usage(:user, app_id: "1234", api_key: "k1", user_id: "user-42")["subject"]
    limits.grant(:user, user_id: "user-42", amount: "10", reason: "goodwill", expires_at: Time.utc(2026, 12, 1))

    assert_equal({ "subject" => "user", "app_id" => "1234", "api_key_id" => "k1", "user_id" => "user-42" },
                 @router.last(:get, "/v1/spend/limits/usage").query)
    assert_equal({ "subject" => "user", "user_id" => "user-42", "amount_micros" => 10_000_000, "reason" => "goodwill",
                   "expires_at" => "2026-12-01T00:00:00Z" }, @router.last(:post, "/v1/spend/limits/grants").json)
  end

  def test_what_the_router_would_refuse_is_refused_before_sending
    [
      -> { limits.create(nil, interval: :daily, amount: 1) },
      -> { limits.create(:planet, interval: :daily, amount: 1) },
      -> { limits.create(:organization, interval: :hourly, amount: 1) },
      -> { limits.create(:organization, interval: :daily, amount: -1) },
      -> { limits.create(:organization, interval: :daily, amount: 1.5) },
      -> { limits.create(:user, team: "finance", interval: :daily, amount: 1) },
      -> { limits.create(:team, interval: :daily, amount: 1) },
      -> { limits.create(:app_team, team: "finance", interval: :daily, amount: 1) },
      -> { limits.create(:team, team: "fin ance", interval: :daily, amount: 1) },
      -> { limits.create(:category, category: "Team:finance", interval: :daily, amount: 1) },
      -> { limits.create(:organization, interval: :daily, amount: 1, level: :galaxy) },
      -> { limits.list(category: "team:finance") },
      -> { limits.reset(:user, user_id: "user-42", reason: "") },
      -> { limits.set_amount(7, "-0.01") },
      -> { limits.grant(:user, user_id: "user-42", amount: 1, reason: nil) }
    ].each { |call| assert_raises(VA::ConfigurationError) { call.call } }
    assert_empty @router.requests
  end
end

class TestTeams < LocalRouterTest
  TEAM_LIMIT = { "id" => 9, "subject" => "team", "team" => "finance", "interval" => "monthly",
                 "limit_micros" => 1 }.freeze

  def test_teams_are_listed_within_an_app
    @router.on(:get, "/v1/teams", body: [{ "team" => "finance" }])

    assert_equal [{ "team" => "finance" }], client.teams.list(app_id: "1234")
    assert_equal({ "app_id" => "1234" }, @router.last(:get, "/v1/teams").query)
    client.teams.list
    assert_empty(@router.last(:get, "/v1/teams").query)
  end

  def test_a_team_with_no_limit_gets_one_created
    @router.on(:get, "/v1/spend/limits", body: { "items" => [], "has_more" => false })
    @router.on(:post, "/v1/spend/limits", status: 201, body: TEAM_LIMIT)

    client.teams.set_limit("finance", interval: :monthly, amount: 8_000, note: "Q4")

    assert_equal({ "subject" => "team", "team" => "finance", "interval" => "monthly" },
                 @router.last(:get, "/v1/spend/limits").query.except("limit"))
    assert_equal({ "subject" => "team", "team" => "finance", "interval" => "monthly",
                   "limit_micros" => 8_000_000_000, "note" => "Q4" }, @router.last(:post, "/v1/spend/limits").json)
  end

  def test_a_team_s_limit_within_an_app_is_changed_in_place
    within = TEAM_LIMIT.merge("subject" => "app_team", "app_id" => "1234")
    @router.on(:get, "/v1/spend/limits", body: { "items" => [TEAM_LIMIT, within], "has_more" => false })
    @router.on(:patch, "/v1/spend/limits/9") { |request| within.merge(request.json) }

    client.teams.set_limit("finance", app_id: "1234", interval: "monthly", amount: "8000")

    assert_equal "app_team", @router.last(:get, "/v1/spend/limits").query["subject"]
    assert_equal({ "limit_micros" => 8_000_000_000 }, @router.last(:patch, "/v1/spend/limits/9").json)
    assert_empty @router.seen(:post, "/v1/spend/limits")
  end

  def test_a_limit_created_meanwhile_is_changed_instead
    listed = 0
    @router.on(:get, "/v1/spend/limits") do
      listed += 1
      { "items" => listed == 1 ? [] : [TEAM_LIMIT], "has_more" => false }
    end
    @router.on(:post, "/v1/spend/limits", status: 409,
                                          body: LocalRouter.failure("conflict", "limit_exists", "it exists"))
    @router.on(:patch, "/v1/spend/limits/9", body: TEAM_LIMIT)

    client.teams.set_limit("finance", interval: :monthly, amount: 10)

    assert_equal({ "limit_micros" => 10_000_000 }, @router.last(:patch, "/v1/spend/limits/9").json)
  end

  def test_removing_a_team_s_limit_deletes_it_or_does_nothing
    @router.on(:get, "/v1/spend/limits", body: { "items" => [TEAM_LIMIT], "has_more" => false })
    @router.on(:delete, "/v1/spend/limits/9", status: 404, body: LocalRouter.failure("not_found", "not_found", "gone"))

    assert_nil client.teams.remove_limit("finance", interval: :monthly)
    assert_nil client.teams.remove_limit("finance", interval: :weekly)
    assert_equal 1, @router.seen(:delete, "/v1/spend/limits/9").size
    assert_raises(VA::ConfigurationError) { client.teams.set_limit("fin/ance", interval: :monthly, amount: 1) }
  end
end
