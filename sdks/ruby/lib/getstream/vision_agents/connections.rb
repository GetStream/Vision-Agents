# frozen_string_literal: true

module GetStream
  module VisionAgents
    # The accounts the app and its users hold at connectors, such as a user's Linear: what an
    # agent's tools reach that account through. Server side only: the router refuses a user's
    # token, and a client holding one is refused naming a user before anything is sent.
    #
    # Every method takes the user as user_id:, named once per call. It goes to the router as
    # the owner and in X-Stream-User-Id alike, so the two never differ, whoever the client acts
    # for otherwise. A user's connection is reached only naming that user: another user's is
    # not found, the same as one that does not exist. The app's are reached by every call, one
    # naming a user included. No user_id is the app's own connection.
    #
    #   connections = api.connections
    #   connection = connections.create("linear", user_id: "alice")
    #   consent = connections.authorize(connection["id"], return_url: "https://app.example.com/settings",
    #                                                     user_id: "alice")
    class Connections
      # The page size #list reads in, the most the router answers.
      PAGE = 200
      CONNECTIONS = "/v1/agents/connections"
      CONNECTION = "/v1/agents/connections/{id}"

      def initialize(client)
        @client = client
      end

      # Stores a pending connection to a connector, the user's or else the app's.
      #
      # @param connector [String] a built-in, such as linear, or one of the app's own.
      # @param label [String] tells the owner's connections to the same connector apart.
      # @return [Hash] the Connection, pending until an account is connected.
      def create(connector, label: nil, user_id: nil)
        user = named(user_id)
        owner = user.empty? ? { type: "app" } : { type: "user", user_id: user }
        @client.post(CONNECTIONS, body: { connector_id: connector, label: presence(label), owner: owner },
                                  user_id: user)
      end

      # Starts the provider's consent for a connection: a first one for a pending connection,
      # a reconnect for one connected before. Open the launch_url it answers in a popup from a
      # page of return_url's origin and post it the handoff_token; the attempt ends at
      # expires_at.
      #
      # @param return_url [String] the app's page the browser goes back to when the consent
      #   ends, with connection_id and status added to its query. nil goes back to the dashboard.
      # @return [Hash] the Authorization: launch_url, handoff_token and expires_at.
      def authorize(id, return_url: nil, user_id: nil)
        @client.post("#{CONNECTION}/authorizations", path: { id: id }, body: { return_url: presence(return_url) },
                                                     user_id: named(user_id))
      end

      # One connection, the user's or the app's.
      #
      # @return [Hash] the Connection.
      def get(id, user_id: nil)
        @client.get(CONNECTION, path: { id: id }, user_id: named(user_id))
      end

      # Every connection of the user's, or of the app's without one, newest first, read page by
      # page. A user's list holds only that user's connections, not the app's.
      #
      # @param connector [String] keeps one connector's. nil is every connector's.
      # @return [Array<Hash>] the Connections.
      def list(connector: nil, user_id: nil)
        user = named(user_id)
        query = { owner_type: user.empty? ? "app" : "user", connector_id: presence(connector), limit: PAGE }
        items = []
        loop do
          page = @client.get(CONNECTIONS, query: query, user_id: user)
          items.concat(page["items"] || [])
          query[:cursor] = page["next_cursor"]
          return items unless page["has_more"] && !query[:cursor].to_s.empty?
        end
      end

      # Disconnects a connection, the user's or the app's, and drops its credentials. One an
      # agent config binds as its fixed connection is refused with a 409.
      def delete(id, user_id: nil)
        @client.delete(CONNECTION, path: { id: id }, user_id: named(user_id))
      end

      private

      # The user a call names, or ConfigurationError for a device naming one: it holds its own
      # user's token, which the router refuses here, so the call is refused before it is sent.
      def named(user_id)
        user = user_id.to_s
        if !user.empty? && !@client.server_side?
          raise ConfigurationError,
                "a user's connections keep the app's credential, which only a backend holds; a device " \
                "holds its user's token (as_user) and acts only as that user, so leave user_id off"
        end

        user
      end

      def presence(value)
        value.to_s.empty? ? nil : value
      end
    end
  end
end
