# frozen_string_literal: true

require "getstream_ruby"
require "securerandom"
require "uri"

module GetStream
  module VisionAgents
    # The Stream call an agent is asked to join.
    #
    # The acceleration backend joins a call that already exists, so what is needed here is
    # creating one, through Stream's own server-side gem, and a link a person can open to be
    # on the other end of it. No media crosses this class.
    #
    # Server side only, because it holds the app secret.
    class Edge
      Call = Data.define(:id, :type)

      DEFAULT_CALL_TYPE = "agent"
      DEFAULT_MONITOR_URL = "https://getstream.io/video/demos"
      MONITOR_ENV = "EXAMPLE_BASE_URL"
      # How long a monitoring link's token lasts. A call outliving it is a call nobody is
      # still on.
      MONITOR_TOKEN_VALIDITY = 60 * 60

      attr_reader :api_key

      # @param base_url [String] Stream's API, for pointing at another deployment.
      def initialize(api_key: nil, api_secret: nil, base_url: nil, monitor_url: nil)
        @api_key = api_key || ENV.fetch(Backend::API_KEY_ENV, "")
        @api_secret = api_secret || ENV.fetch(Backend::API_SECRET_ENV, "")
        if @api_key.empty? || @api_secret.empty?
          raise ConfigurationError, "#{Backend::API_KEY_ENV} and #{Backend::API_SECRET_ENV} are required to create a call"
        end

        @monitor_url = (monitor_url || ENV.fetch(MONITOR_ENV, nil) || DEFAULT_MONITOR_URL).chomp("/")
        options = base_url ? { base_url: base_url } : {}
        @stream = GetStreamRuby.manual(api_key: @api_key, api_secret: @api_secret, **options)
      end

      # Creates the call the backend will join, or returns the one already under that id.
      #
      # An empty id names a new call after a random one, which is what a one-off conversation
      # wants.
      def create_call(id: nil, type: nil, created_by:)
        raise ConfigurationError, "a call needs somebody to have created it" if created_by.to_s.empty?

        call = Call.new(id: id.to_s.empty? ? SecureRandom.hex(8) : id, type: type.to_s.empty? ? DEFAULT_CALL_TYPE : type)
        request = GetStream::Generated::Models::GetOrCreateCallRequest.new(
          data: GetStream::Generated::Models::CallRequest.new(created_by_id: created_by)
        )
        @stream.video.get_or_create_call(call.type, call.id, request)
        call
      rescue GetStreamRuby::ApiError => e
        raise RouterError.new(e.status_code.to_i, "POST /video/call/#{call.type}/#{call.id}", e.message)
      rescue GetStreamRuby::StreamError => e
        raise RouterError.new(0, "POST /video/call/#{call.type}/#{call.id}", e.message)
      end

      # A token for somebody to join a call as, signed here so no request is made.
      def token(user_id, validity: MONITOR_TOKEN_VALIDITY)
        raise ConfigurationError, "a token needs a user to name" if user_id.to_s.empty?

        Backend.sign({ user_id: user_id }, @api_secret, validity: validity)
      end

      # A link a person can open to join a call from a browser and hear the agent.
      def monitor_url(call, user_id:, name: nil)
        raise ConfigurationError, "there is no call to watch" if call.id.to_s.empty?

        query = URI.encode_www_form(api_key: @api_key, token: token(user_id), skip_lobby: "true",
                                    user_name: name || user_id)
        "#{@monitor_url}/join/#{URI.encode_uri_component(call.id)}?#{query}"
      end
    end
  end
end
