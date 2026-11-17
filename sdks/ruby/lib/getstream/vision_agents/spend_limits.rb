# frozen_string_literal: true

module GetStream
  module VisionAgents
    # The customer's spend limits: on the organization, its apps, keys, end users, teams and
    # usage categories. Server side only: the router refuses a user's token, 403
    # server_side_only.
    #
    # Every limit here is the customer's own. Stream's spend limit for the organization is
    # never among them, and nothing here lifts it.
    #
    # Amounts are anything GetStream::VisionAgents.usd takes: an Integer of dollars, a String
    # such as "12.50", a Rational or a USD. Answers are the router's JSON, in micros:
    # USD.micros(limit["limit_micros"]) reads one back.
    #
    # Work a limit refuses raises RouterError with status 402 and code spend_limit_reached, or
    # 429 user_limit_reached for one end user's limit. Stream's own limit refuses with 402
    # system_limit_reached.
    #
    #   limits = api.spend_limits
    #   limits.create(:organization, interval: :monthly, amount: 20_000)
    #   limits.create(:user, app_id: "1234", user_id: "user-42", interval: :daily, amount: 50)
    class SpendLimits
      SUBJECTS = %w[organization app api_key user team app_team category app_category].freeze
      LEVELS = %w[organization app api_key user].freeze
      INTERVALS = %w[daily weekly biweekly monthly].freeze
      GRANT_STATES = %w[live used expired revoked].freeze
      # The page size lists are read in, the most the router answers.
      PAGE = 200
      LIMITS = "/v1/spend/limits"
      LIMIT = "/v1/spend/limits/{limit_id}"
      GRANTS = "/v1/spend/limits/grants"

      def initialize(client)
        @client = client
      end

      # The millionths of a dollar the router is sent for an amount, or ConfigurationError
      # saying why it is not one a limit or a grant can be.
      def self.micros(amount)
        usd = USD.of(amount)
        raise ConfigurationError, "an amount of #{usd} is below zero" if usd.negative?

        usd.micros
      end

      # Every limit the filters match, read page by page.
      #
      # subject: :team lists the teams' limits, :category every other category's. With app_id
      # they are the ones within that app.
      #
      # @return [Array<Hash>] the Limits.
      def list(subject: nil, level: nil, interval: nil, app_id: nil, api_key: nil, user_id: nil, team: nil,
               category: nil)
        query = filters(subject, app_id: app_id, api_key: api_key, user_id: user_id, team: team, category: category)
        query[:level] = one_of(level, LEVELS, "level") if level
        query[:interval] = one_of(interval, INTERVALS, "interval") if interval
        paged(LIMITS, query)
      end

      # One limit by its id.
      def get(limit_id)
        @client.get(LIMIT, path: { limit_id: limit_id })
      end

      # Creates a limit: what one subject may spend in each window of the interval.
      #
      # The level is the most specific one the ids name, which the router works out: :user
      # with a user_id is that user's own limit, :user with only an app_id the default for
      # every user of the app, and :user alone the default for every user of the
      # organization. Keys go the same way. level: overrides it. :team with an app_id is the
      # team within that app.
      #
      # The router refuses with code limit_above_parent (the limit is above one of a subject
      # that contains it), limit_exists (the subject has one at that level and interval:
      # change it with #set_amount) or use_team.
      #
      # @param subject [Symbol, String] organization, app, api_key, user, team, app_team,
      #   category or app_category.
      # @param interval [Symbol, String] daily, weekly, biweekly or monthly.
      # @return [Hash] the Limit.
      def create(subject, interval:, amount:, app_id: nil, api_key: nil, user_id: nil, team: nil, category: nil,
                 level: nil, note: nil)
        kind, ids = target(subject, app_id: app_id, api_key: api_key, user_id: user_id, team: team, category: category)
        body = ids.merge(subject: kind, interval: one_of(interval, INTERVALS, "interval"),
                         limit_micros: SpendLimits.micros(amount))
        body[:level] = one_of(level, LEVELS, "level") if level
        body[:note] = note unless note.to_s.empty?
        @client.post(LIMITS, body: body)
      end

      # Changes what a limit allows. What it is on never changes.
      def set_amount(limit_id, amount)
        @client.patch(LIMIT, path: { limit_id: limit_id }, body: { limit_micros: SpendLimits.micros(amount) })
      end

      # Changes a limit's note. An empty one clears it.
      def set_note(limit_id, note)
        @client.patch(LIMIT, path: { limit_id: limit_id }, body: { note: note.to_s })
      end

      # Deletes a limit. Its subject's spend from then on is held only to its parents'.
      def delete(limit_id)
        @client.delete(LIMIT, path: { limit_id: limit_id })
      end

      # Counts a subject's limits from now until each window ends: every interval's, or only
      # those named. Nothing is refunded: what it spent stays spent. The router refuses with
      # code no_limit a subject with no limit of the customer's.
      #
      # @return [Hash] the SubjectUsage, as it is after the reset.
      def reset(subject, reason:, intervals: nil, app_id: nil, api_key: nil, user_id: nil, team: nil, category: nil)
        raise ConfigurationError, "a reset needs a reason" if reason.to_s.empty?

        kind, ids = target(subject, app_id: app_id, api_key: api_key, user_id: user_id, team: team, category: category)
        body = ids.merge(subject: kind, reason: reason)
        body[:intervals] = Array(intervals).map { |interval| one_of(interval, INTERVALS, "interval") } if intervals
        @client.post("/v1/spend/limits/usage/reset", body: body)
      end

      # What one subject spent in its current windows, against its limits and grants.
      #
      # @return [Hash] the SubjectUsage.
      def usage(subject, app_id: nil, api_key: nil, user_id: nil, team: nil, category: nil)
        kind, ids = target(subject, app_id: app_id, api_key: api_key, user_id: user_id, team: team, category: category)
        answered = @client.get("/v1/spend/limits/usage", query: query_names(ids).merge(subject: kind))
        unless answered.is_a?(Array) && answered.size == 1
          raise RouterError.new(0, "getLimitUsage", "reading the spend of one #{kind}: the router answered " \
                                                    "#{answered.is_a?(Array) ? answered.size : "no list"}")
        end

        answered.first
      end

      # A one-time amount one subject may spend past its own spent limits.
      #
      # It is drawn while any of the subject's limits is spent, across windows, until it is
      # used up, expires_at passes or it is revoked. It never lifts a parent's limit or
      # Stream's. The router refuses with code no_limit (404: the subject has no limit of the
      # customer's to grant past), invalid_expiry or too_many_grants.
      #
      # @param expires_at [Time] nil is the router's default.
      # @return [Hash] the LimitGrant.
      def grant(subject, amount:, reason:, expires_at: nil, app_id: nil, api_key: nil, user_id: nil, team: nil,
                category: nil)
        raise ConfigurationError, "a grant needs a reason" if reason.to_s.empty?

        kind, ids = target(subject, app_id: app_id, api_key: api_key, user_id: user_id, team: team, category: category)
        body = ids.merge(subject: kind, amount_micros: SpendLimits.micros(amount), reason: reason)
        body[:expires_at] = expires_at.utc.iso8601 if expires_at
        @client.post(GRANTS, body: body)
      end

      # Every grant the filters match, read page by page.
      #
      # @param state [Symbol, String] live, used, expired or revoked.
      # @return [Array<Hash>] the LimitGrants.
      def grants(subject: nil, state: nil, app_id: nil, api_key: nil, user_id: nil, team: nil, category: nil)
        query = filters(subject, app_id: app_id, api_key: api_key, user_id: user_id, team: team, category: category)
        query[:state] = one_of(state, GRANT_STATES, "state") if state
        paged(GRANTS, query)
      end

      # Ends a grant before it is used up or expires.
      def revoke_grant(grant_id, reason:)
        raise ConfigurationError, "revoking a grant needs a reason" if reason.to_s.empty?

        @client.post("#{GRANTS}/{grant_id}/revoke", path: { grant_id: grant_id }, body: { reason: reason })
      end

      private

      def paged(template, query)
        items = []
        cursor = nil
        loop do
          page = @client.get(template, query: query.merge(limit: PAGE, cursor: cursor))
          items.concat(page["items"] || [])
          cursor = page["next_cursor"]
          return items unless page["has_more"] && !cursor.to_s.empty?
        end
      end

      # The subject a write is about, as the router names it, and the ids it names.
      def target(subject, **ids)
        raise ConfigurationError, "name the subject: one of #{SUBJECTS.join(", ")}" if subject.to_s.empty?

        named = present(ids)
        [kind(one_of(subject, SUBJECTS, "subject"), named), named]
      end

      # A list's filters, with the subject as the router names it.
      def filters(subject, **ids)
        if subject
          kind, named = target(subject, **ids)
          return query_names(named).merge(subject: kind)
        end

        named = present(ids)
        Backend.check_team(named[:team]) if named[:team]
        Backend.check_category(named[:category]) if named[:category]
        query_names(named)
      end

      def present(ids)
        ids.reject { |_, value| value.to_s.empty? }.transform_values(&:to_s)
      end

      # The subject as the router names it: a team or category within an app is app_team or
      # app_category. It refuses what the router would, before anything is sent.
      def kind(subject, ids)
        Backend.check_category(ids[:category]) if ids[:category]
        Backend.check_team(ids[:team]) if ids[:team]
        in_app = ids.key?(:app_id)
        case subject
        when "team", "app_team"
          raise ConfigurationError, "a #{subject} subject names its team in team:" unless ids[:team]

          subject = "app_team" if in_app
        when "category", "app_category"
          raise ConfigurationError, "a #{subject} subject names its category in category:" unless ids[:category]

          subject = "app_category" if in_app
        else
          raise ConfigurationError, "team: is for the team subject, not #{subject}" if ids[:team]
          raise ConfigurationError, "category: is for the category subject, not #{subject}" if ids[:category]
        end
        if %w[app_team app_category].include?(subject) && !in_app
          raise ConfigurationError, "an #{subject} subject names its app in app_id:"
        end

        subject
      end

      # A key is api_key in a body and api_key_id in a query, where api_key is the credential
      # Stream's proxy requires on every call.
      def query_names(ids)
        ids.transform_keys { |name| name == :api_key ? :api_key_id : name }
      end

      def one_of(value, allowed, name)
        written = value.to_s
        return written if allowed.include?(written)

        raise ConfigurationError, "#{name} #{value.inspect} is not one of #{allowed.join(", ")}"
      end
    end

    # The organization's teams. Server side only.
    #
    # A team has no members list: a user is in the team their token names. A team's spend is
    # the spend of the work those tokens ran, and of the sessions opened with team:, across
    # the organization or within one app.
    #
    #   api.teams.set_limit("finance", interval: :monthly, amount: 8_000)
    class Teams
      def initialize(client)
        @client = client
        @limits = SpendLimits.new(client)
      end

      # Every team with a limit or with spend this month, each with its apps, its spend in
      # the current windows and its limits. With an app_id, only within that app.
      #
      # @return [Array<Hash>] the Teams.
      def list(app_id: nil)
        @client.get("/v1/teams", query: { app_id: app_id.to_s.empty? ? nil : app_id })
      end

      # Makes the team's limit for the interval the amount given: created when it has none,
      # changed when it has one. Across the organization, or within app_id.
      #
      # @return [Hash] the Limit.
      def set_limit(team, interval:, amount:, app_id: nil, note: nil)
        change = { limit_micros: SpendLimits.micros(amount) }
        change[:note] = note unless note.to_s.empty?
        existing = limit(team, interval, app_id)
        unless existing
          begin
            return @limits.create(:team, team: team, app_id: app_id, interval: interval, amount: amount, note: note)
          rescue RouterError => e
            # Somebody else created it since it was looked up.
            raise unless e.code == "limit_exists"
          end
          existing = limit(team, interval, app_id)
          raise Error, "the #{interval} limit on team #{team} exists and could not be found" unless existing
        end
        @client.patch(SpendLimits::LIMIT, path: { limit_id: existing.fetch("id") }, body: change)
      end

      # Deletes the team's limit for the interval, across the organization or within app_id.
      # Nothing happens when it has none.
      def remove_limit(team, interval:, app_id: nil)
        existing = limit(team, interval, app_id)
        return nil unless existing

        begin
          @limits.delete(existing.fetch("id"))
        rescue RouterError => e
          # Somebody else deleted it since it was looked up.
          raise unless e.status == 404
        end
        nil
      end

      # What the team spent in its current windows, against its limits and grants.
      def usage(team, app_id: nil)
        @limits.usage(:team, team: team, app_id: app_id)
      end

      private

      # The team's own limit in the interval, across the organization or within the app.
      def limit(team, interval, app_id)
        Backend.check_team(team)
        raise ConfigurationError, "a team's limit needs an interval" if interval.to_s.empty?

        @limits.list(subject: :team, team: team, app_id: app_id, interval: interval).find do |each|
          each["team"] == team && each["interval"] == interval.to_s && each["app_id"].to_s == app_id.to_s
        end
      end
    end
  end
end
