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
    #     next dispatch.answer(message) unless message.session_id.empty?
    #
    #     dispatch.get_or_create_agent(message) { GetStream::VisionAgents::Agent.new(config: "support") }
    #             .reply(message)
    #   end
    #   dispatch.host(support) # an Agent whose tools are registered
    #   dispatch.run
    #
    # Each call, message and hosted tool call runs on its own thread. When a call or message
    # handler finishes the router is told it is done, with the error when it raised, which is
    # what gives this worker its room back.
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
        @hosted = []
        @threads = []
        # The calls and messages being handled, which is what the router counts against
        # capacity. Hosted tool calls are not.
        @handling = 0
        @latency = nil
      end

      # Runs an agent's tools for every session opened under it, whoever opened it.
      #
      # A session's own tools run in the process that opened it. Hosting is the other
      # direction: the router offers agent.tools to each session naming the agent's config
      # (its name when it has none, as #sync stores it) and sends every call to this worker.
      # tool_timeout is how many seconds the router waits for one tool call to be answered
      # before telling the model it failed, not how long the worker runs; nil takes the
      # router's default of two minutes. Call before #run.
      def host(agent, tool_timeout: nil)
        agent_id = (agent.config || agent.name).to_s
        raise ConfigurationError, "hosting needs an agent with a name" if agent_id.empty?
        raise ConfigurationError, "#{agent_id} needs at least one tool to host" if agent.tools.empty?

        @lock.synchronize { @hosted << { agent_id: agent_id, tools: agent.tools, tool_timeout: tool_timeout } }
        self
      end

      # Handles every call the router hands this worker.
      def wait_for_call(&handler)
        raise ArgumentError, "wait_for_call needs a block" unless handler

        @lock.synchronize { @handlers["call"] = handler }
        self
      end

      # Handles every message written to an agent that is not running, or to a running session
      # whose agent leaves text to dispatch.
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
        unless message.session_id.empty?
          raise ConfigurationError, "a session is already holding this conversation; answer it there with answer"
        end

        @lock.synchronize do
          agent = @agents[message.cid]
          return agent if agent&.live?

          @agents[message.cid] = yield
        end
      end

      # Has the model answer a message written to a running session whose agent leaves text to
      # dispatch.
      #
      # The response is created with this worker's own credential acting for whoever wrote the
      # message, so it goes to the model rather than back to a worker, and it carries the
      # message's request id so the answer lands on it.
      #
      # @return [AgentResponse]
      def answer(message)
        if message.session_id.empty?
          raise ConfigurationError, "no session is holding this message; open one with get_or_create_agent"
        end

        acting = Client.new(backend: @client.backend.acting_for(message.user_id))
        Responses.new(acting, message.session_id).answer(message.text, message.request_id)
      end

      # Connects and hands out work until the socket closes or #stop is called, then waits for
      # every handler to finish and closes the agents it made.
      #
      # @raise [Error] when the router refuses the tools this worker hosts.
      def run
        if @handlers.empty? && @hosted.empty?
          raise ConfigurationError, "register wait_for_call or wait_for_message, or host tools, before running"
        end

        @socket = @client.socket("/v1/dispatch", query: waiting)
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
        when "ready"
          @worker_id = frame["worker_id"]
          offer_hosted
        when "call" then handle("call", frame["work_id"].to_s, InboundCall.from(frame))
        when "message" then handle("message", frame["work_id"].to_s, InboundMessage.from(frame))
        when "tool_call" then run_hosted(frame)
        when "hosting_refused"
          # A worker whose tools were refused is one nobody will call; saying so beats sitting
          # connected looking healthy.
          raise Error, "the router refused to host tools for agent #{frame['agent_id']}: #{frame['reason']}"
        when "pong" then pong(frame)
        end
      end

      def offer_hosted
        @lock.synchronize { @hosted.dup }.each do |offer|
          timeout_ms = offer[:tool_timeout] ? (offer[:tool_timeout] * 1000).round : 0
          @socket.send_frame(type: "host_tools", agent_id: offer[:agent_id],
                             tools: offer[:tools].declarations, timeout_ms: timeout_ms)
        end
      end

      # On its own thread: the socket a tool call arrived on is also what delivers the next.
      def run_hosted(frame)
        id = frame["id"].to_s
        name = frame["name"].to_s
        tools = @lock.synchronize { @hosted.map { |offer| offer[:tools] }.find { |set| set.include?(name) } }
        return tell(type: "tool_result", id: id, error: "this worker does not run #{name}") unless tools

        thread = Thread.new do
          result = begin
            { output: tools.call(name, frame["arguments"]) }
          rescue StandardError => e
            { error: e.message }
          end
          tell({ type: "tool_result", id: id }.merge(result))
        ensure
          @lock.synchronize { @threads.delete(Thread.current) }
        end
        @lock.synchronize { @threads << thread }
      end

      def tell(frame)
        @socket.send_frame(frame)
      rescue SocketClosedError
        nil
      end

      # What this worker says about itself on the handshake: how much it holds, how much it is
      # already holding, and which kinds of work it answers. handles is sent even when empty.
      def waiting
        @lock.synchronize do
          { capacity: @capacity, active: @handling,
            handles: %w[call message].select { |kind| @handlers.key?(kind) }.join(",") }
        end
      end

      # Work with no handler is still reported done: the router holds its room until it is.
      def handle(kind, work_id, work)
        handler = @lock.synchronize { @handlers[kind] }
        return finished(work_id, "this worker answers no #{kind}s") unless handler

        @lock.synchronize { @handling += 1 }
        thread = Thread.new do
          failure = begin
            handler.call(work)
            nil
          rescue StandardError => e
            e.message
          end
          finished(work_id, failure)
        ensure
          @lock.synchronize do
            @threads.delete(Thread.current)
            @handling -= 1
          end
        end
        @lock.synchronize { @threads << thread }
      end

      def finished(work_id, failure)
        frame = { type: "done", work_id: work_id }
        frame[:error] = failure if failure
        tell(frame)
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
