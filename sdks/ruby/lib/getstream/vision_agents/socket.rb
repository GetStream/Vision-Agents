# frozen_string_literal: true

require "json"
require "openssl"
require "socket"
require "uri"
require "websocket/driver"

module GetStream
  module VisionAgents
    # One of the router's sockets: the session events, dispatch, or a modality stream.
    #
    # websocket-driver speaks the protocol; the TCP and TLS connection, the reader thread and
    # the lifecycle are here. Text frames arrive as string-keyed hashes, whole, so an event
    # the router learned after this gem shipped still reaches the caller; binary frames
    # arrive as binary strings. A text frame that is not JSON is skipped rather than fatal.
    #
    # There is one reader. Two loops over one socket would take half the frames each.
    #
    # There is no reconnection, on purpose: respond and tool_result are not idempotent and
    # the protocol has no sequence number to resume from, so replaying would duplicate turns.
    class Socket
      OPEN_TIMEOUT = 10
      # The router takes a message of up to 5 MB; this leaves room for what it sends back.
      MAX_MESSAGE = 8 * 1024 * 1024
      # How long a close waits for the router to answer before the connection is dropped.
      CLOSE_TIMEOUT = 2

      attr_reader :url, :close_code

      def self.open(url, headers: {})
        new(url, headers).tap(&:connect)
      end

      def initialize(url, headers)
        @url = url
        @headers = headers
        @inbox = Thread::Queue.new
        @driver_lock = Mutex.new
        @io_lock = Mutex.new
        @state = Mutex.new
        @changed = ConditionVariable.new
        @open = false
        @finished = false
      end

      # Opens the connection and waits for the upgrade.
      #
      # @raise [RouterError] with the handshake's status when the router refused it, or 0
      #   when it could not be reached.
      def connect
        uri = URI(@url)
        @io = dial(uri)
        @driver = WebSocket::Driver.client(self, max_length: MAX_MESSAGE, binary_data_format: :string)
        @headers.each { |name, value| @driver.set_header(name, value) }
        @driver.on(:open) { update { @open = true } }
        @driver.on(:message) { |event| received(event.data) }
        @driver.on(:error) { |event| update { @failure = event.message } }
        # The connection is dropped once the close handshake is done, or the reader would wait
        # on a connection the router has finished with.
        @driver.on(:close) { |event| drop(event.code) }
        @driver_lock.synchronize { @driver.start }
        @reader = Thread.new { read }
        @reader.report_on_exception = false

        opened = wait_until(OPEN_TIMEOUT) { @open || @finished }
        return self if opened && @open

        status = @driver.respond_to?(:status) ? @driver.status.to_i : 0
        drop
        raise RouterError.new(status, "GET #{uri.path}", @failure || "the socket did not open")
      rescue SystemCallError, IOError, SocketError, OpenSSL::SSL::SSLError => e
        raise RouterError.new(0, "GET #{uri.path}", "the socket never opened: #{e.message}")
      end

      def open?
        @state.synchronize { @open && !@finished }
      end

      # Sends one JSON frame.
      def send_frame(frame)
        transmit { @driver.text(JSON.generate(frame)) }
      end

      # Sends raw bytes, which is how audio goes.
      def send_binary(bytes)
        transmit { @driver.binary(bytes.b) }
      end

      # The next frame, or nil once the socket has closed or the timeout passes.
      def receive(timeout: nil)
        message = timeout ? @inbox.pop(timeout: timeout) : @inbox.pop
        # The end is put back, so a second reader after the close is told too rather than
        # left waiting.
        @inbox << nil if message.nil? && @state.synchronize { @finished }
        message
      end

      # Yields every frame until the socket closes.
      def each_message
        return enum_for(:each_message) unless block_given?

        while (message = receive)
          yield message
        end
      end

      # Closes the socket. Safe to call twice, and from any thread.
      def close
        if open?
          @driver_lock.synchronize { @driver.close }
          wait_until(CLOSE_TIMEOUT) { @finished }
        end
        drop
      end

      # The adapter half of websocket-driver: where it writes, and what it upgrades.
      def write(data)
        @io_lock.synchronize { @io.write(data) }
      rescue SystemCallError, IOError, OpenSSL::SSL::SSLError
        finish(nil)
      end

      private

      def dial(uri)
        io = ::Socket.tcp(uri.host, uri.port || (uri.scheme == "wss" ? 443 : 80), connect_timeout: OPEN_TIMEOUT)
        return io unless uri.scheme == "wss"

        context = OpenSSL::SSL::SSLContext.new
        context.set_params(verify_mode: OpenSSL::SSL::VERIFY_PEER)
        tls = OpenSSL::SSL::SSLSocket.new(io, context)
        tls.hostname = uri.host
        tls.sync_close = true
        tls.connect
        tls
      end

      def read
        loop do
          data = @io.readpartial(16_384)
          @driver_lock.synchronize { @driver.parse(data) }
        end
      rescue EOFError, IOError, SystemCallError, OpenSSL::SSL::SSLError
        finish(nil)
      end

      def received(data)
        if data.encoding == Encoding::BINARY
          @inbox << data
          return
        end
        frame = JSON.parse(data)
        @inbox << frame if frame.is_a?(Hash)
      rescue JSON::ParserError
        nil
      end

      def transmit
        raise SocketClosedError.new("the socket to #{@url} is closed", code: @close_code) unless open?

        sent = @driver_lock.synchronize { yield }
        raise SocketClosedError.new("the socket to #{@url} did not take the frame", code: @close_code) unless sent
      end

      def finish(code)
        first = update do
          was = @finished
          @finished = true
          @close_code ||= code
          !was
        end
        @inbox << nil if first
      end

      def drop(code = nil)
        finish(code)
        @io_lock.synchronize { @io.close unless @io.closed? } if @io
      rescue IOError, SystemCallError
        nil
      end

      def update
        @state.synchronize do
          result = yield
          @changed.broadcast
          result
        end
      end

      def wait_until(seconds)
        deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + seconds
        @state.synchronize do
          until yield
            remaining = deadline - Process.clock_gettime(Process::CLOCK_MONOTONIC)
            return false if remaining <= 0

            @changed.wait(@state, remaining)
          end
          true
        end
      end
    end
  end
end
