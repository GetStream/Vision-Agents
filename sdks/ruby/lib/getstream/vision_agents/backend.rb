# frozen_string_literal: true

require "json"
require "openssl"

module GetStream
  module VisionAgents
    # Where the acceleration router is, and who is calling it.
    #
    # There are three ways to say who that is, and which one a deployment takes is a property
    # of the deployment rather than a choice: a customer id for a router with nothing in front
    # of it, a key and secret for a process the customer runs, and a key and a user token for
    # anything acting on one person's behalf.
    class Backend
      URL_ENV = "STREAM_ACCELERATION_URL"
      CUSTOMER_ENV = "STREAM_ACCELERATION_CUSTOMER_ID"
      API_KEY_ENV = "STREAM_API_KEY"
      API_SECRET_ENV = "STREAM_API_SECRET"
      AUTHENTICATE_ENV = "STREAM_ACCELERATION_AUTHENTICATE"
      DEFAULT_URL = "http://localhost:8080"
      CUSTOMER_HEADER = "X-Customer-Id"

      # How long a token minted here lasts. Short, because it is minted per request and a
      # stolen one should stop working sooner than the credential behind it.
      TOKEN_VALIDITY = 60 * 60

      attr_reader :url, :customer_id, :api_key, :user_id, :user

      # Every argument falls back to the environment, so a process deployed next to a router
      # needs none of them.
      #
      # @param url [String] the router's base URL, then STREAM_ACCELERATION_URL, then localhost.
      # @param customer_id [String] who the work is billed to, for a router that trusts the header.
      # @param api_key [String] the public half of a Stream credential.
      # @param api_secret [String] its secret, which is what makes this a backend.
      # @param token [String] a token minted for user_id to hold, in place of the secret.
      # @param user_id [String] the end user this acts for, if any.
      # @param authenticate [Boolean] whether the router sits behind Stream's proxy.
      def initialize(url: nil, customer_id: nil, api_key: nil, api_secret: nil, token: nil,
                     user_id: nil, authenticate: nil, user: nil)
        @url = present(url) || present(ENV.fetch(URL_ENV, nil)) || DEFAULT_URL
        @url = @url.chomp("/")
        @customer_id = customer_id || ENV.fetch(CUSTOMER_ENV, "")
        # Naming a customer is choosing how a router with nothing in front of it is reached,
        # and a key that happens to be in the environment does not overrule the choice.
        @api_key = api_key || (@customer_id.empty? ? ENV.fetch(API_KEY_ENV, "") : "")
        @token = token.to_s
        # A token handed in is the caller's answer to who they are, so an ambient secret does
        # not turn a client built for a user into a backend.
        @api_secret = api_secret || (@token.empty? ? ENV.fetch(API_SECRET_ENV, "") : "")
        @authenticate = authenticate.nil? ? flag(ENV.fetch(AUTHENTICATE_ENV, nil)) : authenticate
        @user = user || {}
        @user_id = (user_id || @user["id"] || @user[:id]).to_s

        validate
      end

      # Whether this speaks for a process the customer runs rather than for a device.
      #
      # Only a server-side caller reaches the operations the spec does not mark client
      # accessible. Worth asking before a call rather than reading a 403 afterwards.
      def server_side?
        !@api_secret.empty? || (@api_key.empty? && !@customer_id.empty?)
      end

      def authenticate?
        @authenticate
      end

      # A backend acting for one end user, holding the token that proves it.
      #
      # A new backend rather than a change to this one, because a process usually holds both.
      def as_user(user, token)
        named = user.is_a?(Hash) ? user.transform_keys(&:to_s) : { "id" => user.to_s }
        raise ConfigurationError, "a user needs an id" if named["id"].to_s.empty?
        raise ConfigurationError, "there is no token for #{named["id"]} to hold" if token.to_s.empty?

        Backend.new(url: @url, customer_id: @customer_id, api_key: @api_key, api_secret: "",
                    token: token, user_id: named["id"], authenticate: @authenticate, user: named)
      end

      # What every request and socket handshake carries. Minted per read, so a client left
      # idle longer than a token lasts does not wake up holding an expired one.
      def headers
        return { CUSTOMER_HEADER => @customer_id } if @api_key.empty?

        if @authenticate
          # jwt whoever the token is for: the proxy works out the caller from the token and
          # refuses a request claiming to be the server.
          return { "api_key" => @api_key, "stream-auth-type" => "jwt",
                   "Authorization" => "Bearer #{proxy_token}" }
        end

        headers = { "X-Api-Key" => @api_key }
        if @api_secret.empty?
          headers["Authorization"] = "Bearer #{@token}"
          headers["Stream-Auth-Type"] = "jwt"
        else
          headers["Authorization"] = "Bearer #{Backend.sign({ server: true }, @api_secret)}"
          headers["Stream-Auth-Type"] = "server"
          headers["X-Stream-User-Id"] = @user_id unless @user_id.empty?
        end
        headers
      end

      # The WebSocket URL for a path on the router. Credentials travel as headers.
      def socket_url(path)
        @url.sub(/\Ahttp/, "ws") + path
      end

      # Signs a Stream token: HS256 out of OpenSSL rather than a JWT gem, since issuing one
      # fixed kind of token is all this does.
      def self.sign(claims, secret, validity: TOKEN_VALIDITY)
        raise ConfigurationError, "a token cannot be signed without a secret" if secret.to_s.empty?

        issued = Time.now.to_i
        header = encode(JSON.generate({ alg: "HS256", typ: "JWT" }))
        payload = encode(JSON.generate({ iat: issued, exp: issued + validity }.merge(claims)))
        signing = "#{header}.#{payload}"
        "#{signing}.#{encode(OpenSSL::HMAC.digest("SHA256", secret, signing))}"
      end

      def self.encode(raw)
        [raw].pack("m0").tr("+/", "-_").delete("=")
      end

      private

      def validate
        if @authenticate && @api_key.empty?
          raise ConfigurationError,
                "a router behind the proxy is reached with a credential; pass api_key or set #{API_KEY_ENV}"
        end
        if @api_key.empty? && @customer_id.empty?
          raise ConfigurationError,
                "who is calling is not set; pass customer_id or set #{CUSTOMER_ENV} for a router " \
                "that trusts one, or api_key with either api_secret or token"
        end
        return unless !@api_key.empty? && @api_secret.empty? && @token.empty?

        raise ConfigurationError, "api_key needs the secret it belongs to, or a token minted with it"
      end

      # The token the proxy is given, which names a user where there is one: the proxy has no
      # header to read a backend's choice of user from.
      def proxy_token
        return @token unless @token.empty?
        return Backend.sign({ user_id: @user_id }, @api_secret) unless @user_id.empty?

        Backend.sign({ server: true }, @api_secret)
      end

      def present(value)
        value.nil? || value.empty? ? nil : value
      end

      def flag(value)
        %w[1 true yes on].include?(value.to_s.downcase)
      end
    end
  end
end
