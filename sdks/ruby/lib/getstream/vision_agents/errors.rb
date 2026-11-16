# frozen_string_literal: true

require "json"

module GetStream
  module VisionAgents
    # Everything this gem raises on purpose.
    class Error < StandardError; end

    # Something was asked of the SDK that cannot be sent: refused before any request is made.
    class ConfigurationError < Error; end

    # The router refused a request, or it never arrived.
    #
    # A request that never arrived is status 0, because a caller retrying a network failure
    # and one retrying a 500 are doing different things.
    #
    # type, code and doc_url are the router's error envelope; branch on code, which may be
    # one this gem has never heard of. A 500 only says "something went wrong", so request_id
    # (the X-Request-Id header) is what to quote to support.
    class RouterError < Error
      attr_reader :status, :operation, :body, :retry_after, :type, :code, :doc_url, :request_id

      def initialize(status, operation, message, body: nil, retry_after: nil, type: nil, code: nil, doc_url: nil,
                     request_id: nil)
        super("#{operation}: #{message}")
        @status = status
        @operation = operation
        @body = body
        @retry_after = retry_after
        @type = type
        @code = code
        @doc_url = doc_url
        @request_id = request_id
      end

      # Reads an answer that was not a success, a refused socket upgrade included.
      #
      # The router answers {"error": {"message", "type", "code", "doc_url"}}. Anything else,
      # a proxy's page, an empty body or the old {"error": "..."}, gives its text as the
      # message and no type, code or doc_url.
      #
      # @param headers [#[]] the answer's headers, looked up by name in any case.
      def self.answered(status, operation, text, headers)
        text = text.to_s.dup.force_encoding(Encoding::UTF_8).scrub
        said = begin
          JSON.parse(text)
        rescue JSON::ParserError
          nil
        end
        detail = said["error"] if said.is_a?(Hash)
        envelope = detail.is_a?(Hash) ? detail : {}
        message = envelope["message"] || (detail if detail.is_a?(String)) || text.strip
        message = "the router answered #{status}" if message.to_s.empty?
        new(status, operation, message, body: said, retry_after: headers["Retry-After"]&.to_i,
                                        type: envelope["type"], code: envelope["code"],
                                        doc_url: envelope["doc_url"], request_id: headers["X-Request-Id"])
      end
    end

    # A socket closed underneath something that needed it open.
    class SocketClosedError < Error
      attr_reader :code

      def initialize(message, code: nil)
        super(message)
        @code = code
      end
    end
  end
end
