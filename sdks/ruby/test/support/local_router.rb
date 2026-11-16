# frozen_string_literal: true

require "json"
require "socket"
require "uri"
require "websocket/driver"

# A router on a real port: HTTP answered from scripted routes, and websockets upgraded
# with websocket-driver's server half. Every request is recorded as it arrived.
class LocalRouter
  Request = Data.define(:method, :path, :query, :headers, :body) do
    def json
      body.to_s.empty? ? nil : JSON.parse(body)
    end
  end

  # The server end of one websocket, for a test to script.
  class Peer
    attr_reader :request

    def initialize(io, request, head)
      @io = io
      @request = request
      @inbox = Thread::Queue.new
      @lock = Mutex.new
      @env = nil
      @driver = WebSocket::Driver.server(self, binary_data_format: :string)
      @driver.on(:connect) { @driver.start }
      @driver.on(:message) { |event| @inbox << decode(event.data) }
      @driver.on(:close) do
        @inbox << nil
        @io.close
      end
      @lock.synchronize { @driver.parse(head) }
    end

    def send_frame(frame)
      @lock.synchronize { @driver.text(JSON.generate(frame)) }
    end

    def send_binary(bytes)
      @lock.synchronize { @driver.binary(bytes.b) }
    end

    # The next thing the client sent, or nil once it closed or the timeout passed.
    def receive(timeout: 5)
      @inbox.pop(timeout: timeout)
    end

    # The next frame of this type, skipping the others.
    def receive_type(type, timeout: 5)
      deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + timeout
      loop do
        remaining = deadline - Process.clock_gettime(Process::CLOCK_MONOTONIC)
        return nil if remaining <= 0

        frame = receive(timeout: remaining)
        return frame if frame.nil? || (frame.is_a?(Hash) && frame["type"] == type)
      end
    end

    def close
      @lock.synchronize { @driver.close }
    end

    def run
      loop do
        data = @io.readpartial(16_384)
        @lock.synchronize { @driver.parse(data) }
      end
    rescue EOFError, IOError, SystemCallError
      @inbox << nil
    end

    def write(data)
      @io.write(data)
    rescue IOError, SystemCallError
      nil
    end

    private

    def decode(data)
      data.encoding == Encoding::BINARY ? data : JSON.parse(data)
    end
  end

  attr_reader :requests, :peers

  def initialize
    @server = TCPServer.new("127.0.0.1", 0)
    @routes = []
    @sockets = []
    @requests = []
    @peers = Thread::Queue.new
    @lock = Mutex.new
    @threads = []
    @acceptor = Thread.new { accept }
  end

  def url
    "http://127.0.0.1:#{@server.addr[1]}"
  end

  # The router's error envelope, which every failure it answers is.
  def self.failure(type, code, message)
    { "error" => { "message" => message, "type" => type, "code" => code,
                   "doc_url" => "https://getstream.io/agents/docs/api/errors/##{code}" } }
  end

  # Answers method and path (a String, or a Regexp matched against the path) with the
  # block's return: a body, [status, body] or [status, body, headers]. A String body is
  # sent as it is, anything else as JSON. A GET also answers an upgrade no socket is
  # scripted for, which is how a refused upgrade is scripted.
  def on(method, path, status: 200, body: nil, headers: {}, &handler)
    handler ||= ->(_request) { [status, body, headers] }
    @lock.synchronize { @routes.unshift([method.to_s.upcase, path, handler]) }
    self
  end

  # Upgrades a path, and runs the block with the Peer on the connection's own thread.
  def on_socket(path, &script)
    @lock.synchronize { @sockets.unshift([path, script]) }
    self
  end

  # The requests seen for a method and path, in order.
  def seen(method, path)
    @lock.synchronize { @requests.select { |r| r.method == method.to_s.upcase && match?(path, r.path) } }
  end

  def last(method, path)
    seen(method, path).last
  end

  # The next socket a client opened.
  def peer(timeout: 5)
    @peers.pop(timeout: timeout)
  end

  def close
    @server.close
    @threads.each { |thread| thread.join(1) }
  end

  private

  def accept
    loop do
      io = @server.accept
      thread = Thread.new { serve(io) }
      thread.report_on_exception = false
      @lock.synchronize { @threads << thread }
    end
  rescue IOError, SystemCallError
    nil
  end

  def serve(io)
    head = +""
    head << io.readpartial(16_384) until head.include?("\r\n\r\n")
    header, rest = head.split("\r\n\r\n", 2)
    line, *fields = header.split("\r\n")
    method, target = line.split(" ")
    headers = fields.to_h { |field| field.split(": ", 2).then { |name, value| [name.downcase, value] } }
    uri = URI(target)
    query = URI.decode_www_form(uri.query.to_s).to_h

    if headers["upgrade"].to_s.downcase == "websocket"
      upgrade(io, Request.new(method, uri.path, query, headers, nil), head)
      return
    end

    length = headers["content-length"].to_i
    rest << io.read(length - rest.bytesize) while rest.bytesize < length
    request = Request.new(method, uri.path, query, headers, rest)
    @lock.synchronize { @requests << request }
    respond(io, *answer(request))
  rescue IOError, SystemCallError
    nil
  ensure
    io.close unless io.closed?
  end

  def answer(request)
    route = @lock.synchronize { @routes.find { |m, path, _| m == request.method && match?(path, request.path) } }
    unless route
      return [404, LocalRouter.failure("not_found", "not_found", "no route for #{request.method} #{request.path}"), {}]
    end

    reply = route[2].call(request)
    reply = [200, reply] unless reply.is_a?(Array) && reply.size.between?(2, 3) && reply[0].is_a?(Integer)
    status, body, headers = reply
    [status, body, headers || {}]
  end

  def respond(io, status, body, headers)
    text = case body
           when nil then ""
           when String then body
           else JSON.generate(body)
           end
    type = body.is_a?(String) ? "text/html" : "application/json"
    fields = headers.map { |name, value| "#{name}: #{value}\r\n" }.join
    io.write("HTTP/1.1 #{status} X\r\nContent-Type: #{type}\r\nContent-Length: #{text.bytesize}\r\n#{fields}" \
             "Connection: close\r\n\r\n#{text}")
  end

  def upgrade(io, request, head)
    @lock.synchronize { @requests << request }
    script = @lock.synchronize { @sockets.find { |path, _| match?(path, request.path) }&.last }
    unless script
      respond(io, *answer(request))
      return
    end

    peer = Peer.new(io, request, head)
    reader = Thread.new { peer.run }
    reader.report_on_exception = false
    @peers << peer
    script.call(peer)
    reader.join
  ensure
    io.close unless io.closed?
  end

  def match?(pattern, path)
    pattern.is_a?(Regexp) ? pattern.match?(path) : pattern == path
  end
end
