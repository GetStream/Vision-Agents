# frozen_string_literal: true

require "json"
require "net/http"
require "time"
require "uri"

module GetStream
  module VisionAgents
    # The router held once: where it is, who is calling it, and every endpoint it has.
    #
    # One method per HTTP method rather than one per endpoint. The path is the spec's own
    # template, and it is looked up in the table generated from the spec, so a path, a query
    # parameter or a body key the spec does not have is refused before anything is sent.
    #
    #   api = GetStream::VisionAgents::Client.new(customer_id: "examples")
    #   api.get("/v1/agents/configs", query: { name: "docs" })
    #   api.delete("/v1/agents/sessions/{id}", path: { id: "sess_1" })
    #
    # Answers are the router's JSON as string-keyed hashes, whole: a field added to the spec
    # after this gem shipped still reaches the caller. A 204 answers nil.
    class Client
      OPEN_TIMEOUT = 10
      # Creating a session returns once the agent is in the call, which can take seconds.
      READ_TIMEOUT = 60

      attr_reader :backend

      # @param backend [Backend] one already built, instead of building one from the options.
      # @param options the Backend options: url, customer_id, api_key, api_secret, token,
      #   user_id and authenticate, each falling back to the environment.
      def initialize(backend: nil, **options)
        @backend = backend || Backend.new(**options)
      end

      def get(template, path: {}, query: {})
        request(:get, template, path: path, query: query)
      end

      def post(template, path: {}, query: {}, body: nil)
        request(:post, template, path: path, query: query, body: body)
      end

      def put(template, path: {}, query: {}, body: nil)
        request(:put, template, path: path, query: query, body: body)
      end

      def patch(template, path: {}, query: {}, body: nil)
        request(:patch, template, path: path, query: query, body: body)
      end

      def delete(template, path: {}, query: {})
        request(:delete, template, path: path, query: query)
      end

      # Opens one of the router's sockets. OpenAPI stops at the upgrade, so what is said on it
      # is the caller's to read; see Socket.
      def socket(template, path: {}, query: {})
        operation = operation_for(:get, template)
        raise ConfigurationError, "GET #{template} is not a socket" unless operation[:socket]

        url = @backend.socket_url(expand(operation, template, path, query))
        Socket.open(url, headers: @backend.headers)
      end

      # Whether this speaks for a process the customer runs rather than for a device.
      def server_side?
        @backend.server_side?
      end

      # A client acting for one end user, holding the token that proves it.
      #
      # A new client rather than a change to this one: a process usually holds both its own
      # credential and one per user, and switching the user on a shared client would make
      # which user a request was for depend on when it happened to run.
      def as_user(user, token)
        Client.new(backend: @backend.as_user(user, token))
      end

      # A client acting for a guest, which is what their conversations belong to.
      def as_guest(guest)
        as_user({ "id" => guest.fetch("id"), "name" => guest["name"] || "Guest" }, guest.fetch("token"))
      end

      # Mints a guest so somebody can talk to an agent before they sign up.
      #
      # Nothing is remembered here: a server handling two visitors that remembered a guest
      # would hand them each other's conversations. Keeping which visitor is which guest is
      # the caller's job, because only the caller knows what a visitor is.
      #
      # @param id [String] a guest to reuse, for somebody coming back. Empty mints a new one.
      # @return [Hash] the GuestUser: id, token, name, custom and expires_at.
      def guest_user(id: nil, name: nil, custom: nil)
        post("/v1/agents/guests", body: { id: presence(id), name: presence(name), custom: custom })
      end

      # Moves a guest's conversations onto the account they turned out to be.
      #
      # Server side only: only the app's own backend knows that a given guest is a given
      # account, because it is the thing that just authenticated them.
      #
      # @return [Hash] the ClaimGuestResult: guest_id, user_id and sessions_moved.
      def claim_guest_user(guest_id, user_id)
        unless server_side?
          raise ConfigurationError,
                "claiming a guest is server side only: it is the backend that just authenticated " \
                "the account that knows which guest it was"
        end
        if guest_id.to_s.empty? || user_id.to_s.empty?
          raise ConfigurationError, "claiming a guest needs the guest and the account"
        end

        post("/v1/agents/guests/claim", body: { guest_id: guest_id, user_id: user_id })
      end

      # An agent addressed by the name it is configured under. No request is made.
      def agent(name, **options)
        Agent.new(config: name, client: self, **options)
      end

      private

      def request(method, template, path:, query:, body: nil)
        operation = operation_for(method, template)
        raise ConfigurationError, "#{template} is a socket; open it with #socket" if operation[:socket]

        target = URI(@backend.url + expand(operation, template, path, query))
        payload = encode_body(operation, body)
        started = Net::HTTP.const_get(method.to_s.capitalize).new(target)
        @backend.headers.each { |name, value| started[name] = value }
        started["Accept"] = "application/json"
        if payload
          started["Content-Type"] = "application/json"
          started.body = payload
        end

        answer(operation, send_request(operation, target, started))
      end

      def send_request(operation, target, request)
        Net::HTTP.start(target.host, target.port, use_ssl: target.scheme == "https",
                                                  open_timeout: OPEN_TIMEOUT, read_timeout: READ_TIMEOUT) do |http|
          http.request(request)
        end
      rescue SystemCallError, IOError, SocketError, Timeout::Error, OpenSSL::SSL::SSLError => e
        raise RouterError.new(0, operation[:id], "the request never arrived: #{e.message}")
      end

      def answer(operation, response)
        status = response.code.to_i
        text = response.body.to_s
        if status.between?(200, 299)
          return nil if text.strip.empty?

          return JSON.parse(text)
        end

        said = begin
          JSON.parse(text)
        rescue JSON::ParserError
          nil
        end
        message = said.is_a?(Hash) && said["error"] ? said["error"] : text.strip
        message = "the router answered #{status}" if message.empty?
        retry_after = response["Retry-After"]&.to_i
        raise RouterError.new(status, operation[:id], message, body: said, retry_after: retry_after)
      end

      def operation_for(method, template)
        operation = Client.operations[[method, template]]
        return operation if operation

        raise ConfigurationError, "#{method.to_s.upcase} #{template} is not an operation the spec has"
      end

      def expand(operation, template, path, query)
        given = stringify(path)
        expanded = template.gsub(/\{([^}]+)\}/) do
          value = given[::Regexp.last_match(1)].to_s
          raise ConfigurationError, "#{template} needs #{::Regexp.last_match(1)}" if value.empty?

          escape(value)
        end
        unknown = given.keys - operation[:path_params]
        raise ConfigurationError, "#{template} has no #{unknown.join(", ")} in its path" unless unknown.empty?

        encoded = encode_query(operation, template, query)
        encoded.empty? ? expanded : "#{expanded}?#{encoded}"
      end

      def encode_query(operation, template, query)
        pairs = stringify(query).reject { |_, value| value.nil? }
        unknown = pairs.keys - operation[:query]
        unless unknown.empty?
          raise ConfigurationError,
                "#{template} takes no #{unknown.join(", ")}; it takes #{operation[:query].join(", ")}"
        end

        URI.encode_www_form(pairs.transform_values { |value| query_value(value) })
      end

      def query_value(value)
        case value
        when Time then value.utc.iso8601
        when Hash, Array then JSON.generate(value)
        else value.to_s
        end
      end

      # nil means omit: a key left nil is left out, so the config or the router decides.
      def encode_body(operation, body)
        if body.nil?
          raise ConfigurationError, "#{operation[:path]} needs a body" if operation[:body_required]

          return nil
        end
        raise ConfigurationError, "#{operation[:path]} takes no body" if operation[:body].nil?

        fields = stringify(body).reject { |_, value| value.nil? }
        schema = Generated::SCHEMAS[operation[:body]]
        if schema
          unknown = schema[:open] ? [] : fields.keys - schema[:properties]
          unless unknown.empty?
            raise ConfigurationError, "#{operation[:body]} has no #{unknown.join(", ")}"
          end

          missing = schema[:required] - fields.keys
          raise ConfigurationError, "#{operation[:body]} needs #{missing.join(", ")}" unless missing.empty?
        end
        JSON.generate(fields)
      end

      def stringify(hash)
        (hash || {}).to_h.transform_keys(&:to_s)
      end

      def escape(value)
        value.b.gsub(/[^A-Za-z0-9\-._~]/) { |char| format("%%%02X", char.ord) }
      end

      def presence(value)
        value.nil? || value.to_s.empty? ? nil : value
      end

      class << self
        # The generated table, indexed by method and path template.
        def operations
          @operations ||= Generated::OPERATIONS.each_with_object({}) do |(id, operation), index|
            index[[operation[:method], operation[:path]]] = operation.merge(id: id)
          end.freeze
        end
      end
    end
  end
end
