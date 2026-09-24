# frozen_string_literal: true

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
    class RouterError < Error
      attr_reader :status, :operation, :body, :retry_after

      def initialize(status, operation, message, body: nil, retry_after: nil)
        super("#{operation}: #{message}")
        @status = status
        @operation = operation
        @body = body
        @retry_after = retry_after
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
