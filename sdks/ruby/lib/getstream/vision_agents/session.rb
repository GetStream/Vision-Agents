# frozen_string_literal: true

module GetStream
  module VisionAgents
    # Somebody on the call, as the backend reports them.
    Participant = Data.define(:id, :user_id, :name) do
      def self.from(frame)
        who = frame["participant"]
        return nil unless who.is_a?(Hash) && !who.empty?

        new(who["id"].to_s, who["user_id"].to_s, who["name"].to_s)
      end
    end

    # One thing the conversation did.
    #
    # kind is the backend's own name for it: joined, participant_joined, heard, responding,
    # response_delta, responded, spoke, turn, delegated, tool_ran, error, left and the rest.
    # The fields are filled from whichever frames carry them, and frame is the whole thing for
    # anything they do not cover, including kinds this gem has never heard of.
    Event = Data.define(:kind, :text, :participant, :interrupted, :pending_work, :error, :frame) do
      def self.from(frame)
        new(frame["type"].to_s, frame["text"].to_s, Participant.from(frame), frame["interrupted"] == true,
            frame["pending_work"] == true, frame["error"].to_s, frame)
      end
    end

    # One conversation, held in the acceleration backend.
    #
    # Nothing here does inference or touches media. The backend hears the caller, answers and
    # speaks, and what arrives here are the events saying so. What stays here is function
    # calling, because the functions are here.
    #
    # The socket is read from the moment the session opens, whether or not anybody iterates
    # #events: a tool call the model is waiting on cannot depend on the caller having started
    # a loop.
    class Session
      # Events held for a reader before the oldest are dropped rather than stalling the
      # socket. Dropping an event never drops a turn, because tool calls are answered by the
      # reader thread, not by whoever iterates.
      EVENT_BUFFER = 256
      # Tool calls running at once before the model is told to carry on without one.
      RUNNING_TOOLS = 16
      EVENTS_PATH = "/v1/agents/sessions/{id}/events"

      attr_reader :created, :responses

      # Starts watching a session the router has already created.
      #
      # @param interim [Boolean] also report what the caller is part way through saying.
      # @param decisions [Boolean] report the router's own routing decisions, which are
      #   several a second and off unless asked for.
      def self.watching(client, created, tools: Tools.new, interim: false, decisions: false)
        query = { interim: (true if interim), decisions: (false unless decisions) }
        socket = begin
          client.socket(EVENTS_PATH, path: { id: created.fetch("id") }, query: query)
        rescue Error
          # The session is live in the backend even though nothing here can watch it, so it
          # is closed rather than left holding a call nobody is listening to.
          begin
            client.delete("/v1/agents/sessions/{id}", path: { id: created["id"] })
          rescue Error
            nil
          end
          raise
        end
        new(client, created, tools, socket, interim: interim, decisions: decisions)
      end

      def initialize(client, created, tools, socket, interim: false, decisions: false)
        @client = client
        @created = created
        @tools = tools
        @socket = socket
        @options = { interim: interim, decisions: decisions }
        @responses = Responses.new(client, id)
        @lock = Mutex.new
        @changed = ConditionVariable.new
        @events = []
        @participants = {}
        @heard = false
        @ended = false
        @running = {}
        @reader = Thread.new { watch }
        @reader.report_on_exception = false
      end

      # The backend's id for the conversation.
      def id
        @created.fetch("id")
      end

      # The channel replies are written into, empty for a conversation that keeps none.
      def conversation_id
        @created["conversation_id"].to_s
      end

      # The Stream call the conversation is on, empty for one held in writing.
      def call_id
        @created["call_id"].to_s
      end

      def call_type
        @created["call_type"].to_s
      end

      def live?
        @lock.synchronize { !@ended }
      end

      # Yields what the backend did until the conversation ends.
      #
      # There is one stream: two loops over it take half the events each.
      def events
        return enum_for(:events) unless block_given?

        while (event = next_event)
          yield event
        end
      end

      # Speaks text without going through the model, for when you already know what to say.
      def say(text)
        command(type: "say", text: text)
      end

      # Answers text through the model, as though it had been said on the call.
      # Responses#create is the same with an id back.
      def respond(text)
        command(type: "respond", text: text)
      end

      # Abandons the reply being spoken.
      def interrupt
        command(type: "interrupt")
      end

      # Changes what the agent is told to be, from the next turn.
      def set_instructions(instructions)
        command(type: "instructions", instructions: instructions)
      end

      # Continues this conversation as a new one.
      #
      # The parent is untouched. The fork inherits this session's tools, since they are here
      # in this process and a conversation continued without them would offer the model tools
      # nothing can run.
      #
      # @param response_id [String] carry the history only up to the end of this response,
      #   so the fork branches from there. A persistent conversation cannot be rewound; this
      #   is how it is taken back instead.
      # @param options any other ForkSessionRequest field: agent, config_id, title,
      #   description, project, custom, model_overwrites, instructions, incognito, messages,
      #   call_id.
      def fork(response_id: nil, **options)
        body = options.merge(response_id: response_id && Responses.id_of(response_id))
        forked = @client.post("/v1/agents/sessions/{id}/fork", path: { id: id }, body: body)
        Session.watching(@client, forked, tools: @tools, **@options)
      end

      # Blocks until the conversation ends. Returns false if the timeout passed first.
      def wait(timeout: nil)
        until_true(timeout) { @ended }
      end

      # Blocks until somebody other than the agent is in the call, or has been heard.
      # Returns false if the timeout passed first.
      def wait_for_participant(timeout: nil)
        until_true(timeout) { @heard || !@participants.empty? || @ended }
        @lock.synchronize { @heard || !@participants.empty? }
      end

      # Who is in the call besides the agent, as far as the backend has reported.
      def participants
        @lock.synchronize { @participants.values }
      end

      # Ends the conversation. Safe to call after it has already ended.
      def close
        if @socket.open?
          begin
            @socket.send_frame(type: "close")
          rescue SocketClosedError
            nil
          end
        elsif live?
          begin
            @client.delete("/v1/agents/sessions/{id}", path: { id: id })
          rescue RouterError
            nil
          end
        end
        @socket.close
        @reader.join(Socket::CLOSE_TIMEOUT)
        ended
        nil
      end

      private

      def command(frame)
        raise SocketClosedError, "the session #{id} is not being held" unless @socket.open?

        @socket.send_frame(frame)
        nil
      end

      def watch
        @socket.each_message do |frame|
          received(frame) if frame.is_a?(Hash)
        end
      ensure
        ended
      end

      def received(frame)
        case frame["type"]
        when "tool_call" then run_tool(frame)
        when "tool_cancel"
          # The model has moved on. A Ruby thread cannot be stopped safely from outside, so the
          # tool is left to finish and its answer is dropped rather than sent.
          @lock.synchronize { @running.delete(frame["id"].to_s) }
        else
          seen(frame)
          push(Event.from(frame))
        end
      end

      def seen(frame)
        participant = Participant.from(frame)
        @lock.synchronize do
          case frame["type"]
          when "participant_joined"
            @participants[participant.user_id] = participant if participant && participant.user_id != agent_user
          when "participant_left"
            @participants.delete(participant.user_id) if participant
          when "heard"
            @heard = true
          end
          @changed.broadcast
        end
      end

      def run_tool(frame)
        call_id = frame["id"].to_s
        answer = { type: "tool_result", tool_call_id: call_id }
        answer[:command_id] = frame["command_id"] unless frame["command_id"].to_s.empty?
        answer[:turn_id] = frame["turn_id"] unless frame["turn_id"].to_s.empty?

        busy = @lock.synchronize do
          full = @running.size >= RUNNING_TOOLS
          # Held before the thread starts, so a tool quicker than this line still finds itself
          # wanted.
          @running[call_id] = true unless full
          full
        end
        return reply(answer.merge(error: "too many tools are already running")) if busy

        Thread.new do
          # A tool that raises is reported to the model rather than dropped: it is
          # mid-sentence waiting, and can only say something useful if it is told.
          result = begin
            { output: @tools.call(frame["name"].to_s, frame["arguments"]) }
          rescue StandardError => e
            { error: e.message }
          end
          wanted = @lock.synchronize { @running.delete(call_id) }
          reply(answer.merge(result)) if wanted
        end
      end

      def reply(frame)
        @socket.send_frame(frame) if @socket.open?
      rescue SocketClosedError
        nil
      end

      def push(event)
        @lock.synchronize do
          @events.shift if @events.size >= EVENT_BUFFER
          @events << event
          @changed.broadcast
        end
      end

      def next_event
        @lock.synchronize do
          @changed.wait(@lock) while @events.empty? && !@ended
          @events.shift
        end
      end

      def ended
        @lock.synchronize do
          @ended = true
          @running.clear
          @changed.broadcast
        end
      end

      def until_true(timeout)
        deadline = timeout && (Process.clock_gettime(Process::CLOCK_MONOTONIC) + timeout)
        @lock.synchronize do
          until yield
            remaining = deadline && (deadline - Process.clock_gettime(Process::CLOCK_MONOTONIC))
            return false if remaining && remaining <= 0

            @changed.wait(@lock, remaining)
          end
          true
        end
      end

      def agent_user
        @created["user_id"].to_s
      end
    end
  end
end
