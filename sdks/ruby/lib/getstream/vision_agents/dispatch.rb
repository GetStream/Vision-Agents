# frozen_string_literal: true

module GetStream
  module VisionAgents
    # Answers calls and messages the router hands this worker over /v1/dispatch.
    #
    #   dispatch = GetStream::VisionAgents::Dispatch.new
    #   dispatch.wait_for_call do |call|
    #     GetStream::VisionAgents::Agent.new(config: "support").join(call)
    #   end
    #   dispatch.wait_for_message do |message|
    #     dispatch.get_or_create_agent(message) { GetStream::VisionAgents::Agent.new(config: "support") }
    #             .reply(message)
    #   end
    #   dispatch.run
    #
    # Each call and message runs its handler on its own thread. A call handler that returns
    # accepts the call; one that raises rejects it, so the router offers it to another worker.
    # A message is not accepted or rejected: nothing waits on it the way a ringing caller does.
    #
    # The socket is not reopened when it drops: #run returns and the process decides.
    class Dispatch
      # How many conversations this worker offers to hold at once.
      CAPACITY = 4
      # How often, in seconds, the worker reports its load, which is also what keeps the
      # socket from going idle.
      REPORT_EVERY = 15

      attr_reader :worker_id

      def initialize(capacity: CAPACITY, report_every: REPORT_EVERY, client: nil)
        raise ConfigurationError, "a worker needs room for at least one conversation" if capacity < 1

        @capacity = capacity
        @report_every = report_every
        @client = client.is_a?(Client) ? client : Client.new(**(client || {}))
        @lock = Mutex.new
        @handlers = {}
        @agents = {}
        @threads = []
        @latency = nil
      end

      # Handles every call the router hands this worker.
      def wait_for_call(&handler)
        raise ArgumentError, "wait_for_call needs a block" unless handler

        @lock.synchronize { @handlers["call"] = handler }
        self
      end

      # Handles every message written to an agent that is not running.
      def wait_for_message(&handler)
        raise ArgumentError, "wait_for_message needs a block" unless handler

        @lock.synchronize { @handlers["message"] = handler }
        self
      end

      # The agent answering a channel: the one already holding it, or a new one from the block.
      #
      # A channel written to twice while its first answer is still being written would
      # otherwise get two agents talking over each other.
      def get_or_create_agent(message)
        @lock.synchronize do
          agent = @agents[message.cid]
          return agent if agent&.live?

          @agents[message.cid] = yield
        end
      end

      # Connects and hands out work until the socket closes or #stop is called, then waits for
      # every handler to finish and closes the agents it made.
      def run
        raise ConfigurationError, "register wait_for_call or wait_for_message before running" if @handlers.empty?

        @socket = @client.socket("/v1/dispatch", query: { capacity: @capacity })
        reporter = Thread.new { report }
        @socket.each_message { |frame| received(frame) if frame.is_a?(Hash) }
      ensure
        reporter&.kill
        @socket&.close
        @lock.synchronize { @threads.dup }.each(&:join)
        @lock.synchronize { @agents.values }.each(&:close)
      end

      # Stops taking work. #run returns once the handlers already running finish.
      def stop
        @socket&.close
      end

      # How many handlers are running.
      def active
        @lock.synchronize { @threads.count(&:alive?) }
      end

      private

      def received(frame)
        case frame["type"]
        when "ready" then @worker_id = frame["worker_id"]
        when "call" then handle("call", InboundCall.from(frame))
        when "message" then handle("message", InboundMessage.from(frame))
        when "pong" then pong(frame)
        end
      end

      def handle(kind, work)
        handler = @lock.synchronize { @handlers[kind] }
        return reject(work, "this worker does not answer calls") if handler.nil? && kind == "call"
        return if handler.nil?

        thread = Thread.new do
          handler.call(work)
          @socket.send_frame(type: "accepted", call_id: work.call_id) if kind == "call"
        rescue StandardError => e
          raise unless kind == "call"

          reject(work, e.message)
        ensure
          @lock.synchronize { @threads.delete(Thread.current) }
        end
        thread.report_on_exception = kind == "call" ? false : true
        @lock.synchronize { @threads << thread }
      end

      def reject(call, reason)
        @socket.send_frame(type: "rejected", call_id: call.call_id, reason: reason)
      rescue SocketClosedError
        nil
      end

      def report
        loop do
          frame = { type: "load", active_agents: active }
          frame[:latency_ms] = @latency if @latency
          @socket.send_frame(frame)
          @socket.send_frame(type: "ping", at: now)
          sleep @report_every
        end
      rescue SocketClosedError
        nil
      end

      # The router sends the ping's number straight back, so this side's clock is the only
      # one that matters.
      def pong(frame)
        @latency = ((now - frame["at"]) * 1000).round if frame["at"].is_a?(Numeric)
      end

      def now
        Process.clock_gettime(Process::CLOCK_MONOTONIC)
      end
    end
  end
end
