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

      attr_reader :created, :responses, :voice

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
          # is stopped rather than left holding a call nobody is listening to.
          begin
            client.post("/v1/agents/sessions/{id}/stop", path: { id: created["id"] })
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
        @voice = Voice.new(client, id, !created["call_id"].to_s.empty?)
        @lock = Mutex.new
        @changed = ConditionVariable.new
        @events = []
        @participants = {}
        @heard = false
        @ended = false
        @running = {}
        # added are the functions add_tools gave the session after it opened, copied from the
        # caller's into a Tools of the session's own; they run here before the agent's own of
        # the same name. withdrawn are the names remove_tool took from the session, which the
        # agent's own no longer run, and unsettled the ones whose adding or withdrawing was never
        # answered, which the router may or may not offer.
        @added = Tools.new
        @withdrawn = {}
        @unsettled = {}
        # One change of the session's tools at a time, across its request.
        @changing = Mutex.new
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

      # The Stream call the session's voice is on, agent:<session id>, whether or not voice
      # is started.
      def call
        Edge::Call.new(id: id, type: Edge::DEFAULT_CALL_TYPE)
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

      # Abandons the reply being spoken.
      def interrupt
        command(type: "interrupt")
      end

      # Continues this conversation as a new one, held in writing.
      #
      # The parent is untouched. The fork runs this session's tools as they are now, the ones
      # it opened with less the ones remove_tool took and with the ones add_tools gave it, as
      # its own: the router gives the fork the same tools, and a conversation continued without
      # their functions would offer the model tools nothing can run.
      #
      # @param response_id [String] carry the history only up to the end of this response,
      #   so the fork branches from there. A persistent conversation cannot be rewound; this
      #   is how it is taken back instead.
      # @param options any other ForkSessionRequest field: agent, config_id, title,
      #   description, project_id, custom, model_overwrites, incognito, messages.
      def fork(response_id: nil, **options)
        body = options.merge(response_id: response_id && Responses.id_of(response_id))
        # Copied, and the fork asked for, between changes of the tools, so the router copies
        # the same ones.
        tools, forked = @changing.synchronize do
          [@lock.synchronize { @tools.except(*@withdrawn.keys).merge(@added) },
           @client.post("/v1/agents/sessions/{id}/fork", path: { id: id }, body: body)]
        end
        Session.watching(@client, forked, tools: tools, **@options)
      end

      # Changes this session only: its title, description, custom labels, models or voice,
      # from the next turn. The config it started from is untouched, and a target that does
      # not route is refused before anything changes. Server side only.
      #
      # @param fields any UpdateSessionRequest field: title, description, custom,
      #   llm, stt, tts, sts, voice, thinking, temperature, max_output_tokens,
      #   verbosity, pinned. A field left out is left as it is; an empty sts makes the session a
      #   cascade again, and an empty voice returns to the provider's default.
      # @return [Hash] the session as the router now has it.
      def update(**fields)
        @client.patch("/v1/agents/sessions/{id}", path: { id: id }, body: fields)
      end

      # Gives this session the functions in more while it runs, for it alone, and runs their
      # calls here. The agent's own tools are left as they are, and no other session gets
      # these. One named as a tool the session already has is refused: to change one,
      # remove_tool it first, so that no call of it runs half under each.
      #
      # The router waits for the turn in progress, and every call in flight, to finish, and the
      # model is offered them from the next turn, so do not call this from inside one of the
      # session's tools. Only the kind of caller that opened the session may.
      #
      # @param more [Tools] the functions to add, as they are now: one registered in it later
      #   is not run in their place.
      # @raise [RouterError] when the router refused them, or with status 0 when its answer was
      #   lost, in which case their calls are still run here until the name is changed again.
      def add_tools(more)
        more = Tools.new.merge(more)
        declared = more.declarations
        raise ConfigurationError, "there are no tools to add" if declared.empty?

        names = declared.map { |tool| tool[:name] }
        @changing.synchronize do
          # A name whose adding was never answered may or may not be the router's: it is
          # withdrawn first, so the router and this process agree on which function it is.
          names.each { |name| withdraw(name, absent_is_fine: true) if unsettled?(name) }

          # Run from before the router answers: the next turn can call one before the answer is
          # here. A name already run here keeps its function, since the router refuses to add
          # it again.
          fresh = @lock.synchronize do
            names.reject { |name| @added.include?(name) || (!@withdrawn[name] && @tools&.include?(name)) }
                 .tap { |picked| @added = @added.merge(more.slice(*picked)) }
          end
          answered = false
          begin
            @client.post("/v1/agents/sessions/{id}/tools", path: { id: id }, body: { tools: declared })
            answered = true
          rescue RouterError => e
            # Refused, the router has none of them.
            answered = e.status.positive?
            @lock.synchronize { @added = @added.except(*fresh) } if answered
            raise
          ensure
            # Without an answer, lost or given up on, the router may have them all, so their
            # calls are still run here until a later change of the name settles it.
            @lock.synchronize { fresh.each { |name| @unsettled[name] = true } } unless answered
          end
        end
        nil
      end

      # Withdraws one of this session's tools, whether the agent declared it when the session
      # opened or add_tools added it since. The router waits as it does for add_tools, so once
      # this returns no call of the tool is in flight and none can be made: do not call it from
      # inside one of the session's tools.
      #
      # @raise [RouterError] 404 when the session has no tool by that name, or with status 0 when
      #   the answer was lost, in which case its calls are still run here until the name is
      #   changed again.
      def remove_tool(name)
        name = name.to_s
        @changing.synchronize { withdraw(name, absent_is_fine: unsettled?(name)) }
      end

      # Pins this conversation, so a query sorted by pinned lists it first. Its place in
      # last_message_at order is unchanged.
      #
      # @return [Hash] the session as the router now has it, with pinned_at.
      def pin
        update(pinned: true)
      end

      # Unpins this conversation. The session answered has no pinned_at.
      def unpin
        update(pinned: false)
      end

      # Deletes this conversation: it is stopped, and its turns and what it remembered are
      # deleted with it. The user's other memories are kept.
      def delete
        @client.delete("/v1/agents/sessions/{id}", path: { id: id })
      end

      # Deletes what this conversation remembered, and leaves the rest of the user's
      # memories alone. Server side only.
      def delete_memories
        @client.delete("/v1/agents/sessions/{id}/memories", path: { id: id })
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

      # Stops the conversation. Safe to call after it has already ended. What it recorded and
      # remembered is kept; #delete takes it away.
      def close
        if @socket.open?
          begin
            @socket.send_frame(type: "close")
          rescue SocketClosedError
            nil
          end
        elsif live?
          begin
            @client.post("/v1/agents/sessions/{id}/stop", path: { id: id })
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
        when "tool_call"
          # Every watcher is sent every call, and the router says which are to run it. One
          # from before it said holds the same rule: answering a tool held elsewhere would
          # beat whoever runs it, with an error.
          tools = tools_for(frame["name"].to_s)
          run_tool(frame, tools) if frame["run"] != false && tools&.include?(frame["name"].to_s)
        when "tool_cancel"
          # The model has moved on. A Ruby thread cannot be stopped safely from outside, so the
          # tool is left to finish and its answer is dropped rather than sent, unless it asked
          # to be held.
          @lock.synchronize do
            tool_call_context = @running[frame["id"].to_s]
            @running.delete(frame["id"].to_s) if tool_call_context&.cancel!
          end
        else
          @lock.synchronize { @running[frame["id"].to_s]&.held! } if frame["type"] == "tool_held"
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

      def run_tool(frame, tools)
        call_id = frame["id"].to_s
        answer = { type: "tool_result", tool_call_id: call_id }
        answer[:request_id] = frame["request_id"] unless frame["request_id"].to_s.empty?
        answer[:turn_id] = frame["turn_id"] unless frame["turn_id"].to_s.empty?

        tool_call_context = ToolCallContext.from(frame, session_id: id, send: ->(sent) { reply(sent) })
        busy = @lock.synchronize do
          full = @running.size >= RUNNING_TOOLS
          # Held before the thread starts, so a tool quicker than this line still finds itself
          # wanted.
          @running[call_id] = tool_call_context unless full
          full
        end
        return reply(answer.merge(error: "too many tools are already running")) if busy

        Thread.new do
          # A tool that raises is reported to the model rather than dropped: it is
          # mid-sentence waiting, and can only say something useful if it is told.
          result = begin
            { output: tools.call(frame["name"].to_s, frame["arguments"], tool_call_context) }
          rescue StandardError => e
            { error: e.message }
          end
          wanted = @lock.synchronize { @running.delete(call_id) }
          reply(answer.merge(result)) if wanted
        end
      end

      # Takes the tool called name from the session and from what runs here. absent_is_fine
      # settles a name whose last change was never answered: the router having no tool by it is
      # as good as withdrawing it.
      def withdraw(name, absent_is_fine:)
        answered = false
        begin
          # Sent once: one sent again would be answered 404 for the tool the first withdrew.
          @client.delete("/v1/agents/sessions/{id}/tools/{name}", path: { id: id, name: name }, resend: false)
          answered = true
        rescue RouterError => e
          answered = e.status.positive?
          raise unless answered && absent_is_fine && e.status == 404
        ensure
          # Without an answer, lost or given up on, the router may or may not have withdrawn it:
          # its function still runs here, and the name is withdrawn again, a 404 accepted,
          # before it changes next.
          @lock.synchronize { @unsettled[name] = true } unless answered
        end
        @lock.synchronize do
          @added = @added.except(name)
          @unsettled.delete(name)
          @withdrawn[name] = true
        end
        nil
      end

      def unsettled?(name)
        @lock.synchronize { @unsettled.key?(name) }
      end

      # The Tools that run the tool called name: the session's own once add_tools added it, none
      # once remove_tool withdrew it, else the agent's.
      def tools_for(name)
        @lock.synchronize { @added.include?(name) ? @added : (@tools unless @withdrawn[name]) }
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
          @running.each_value(&:stop!)
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

    # A session's voice: the agent on the call agent:<session id>. Stopping it carries the
    # conversation on in writing, and what is said and what is typed are one conversation.
    class Voice
      PATH = "/v1/agents/sessions/{id}/voice"

      def initialize(client, session_id, started)
        @client = client
        @session_id = session_id
        @started = started
      end

      # Has the agent join the call. Starting voice that is already on does nothing.
      #
      # @return [Hash] the session as the router now has it.
      def start
        seen(@client.post(PATH, path: { id: @session_id }))
      end

      # Takes the agent off the call. Stopping voice that is off does nothing.
      #
      # @return [Hash] the session as the router now has it.
      def stop
        seen(@client.delete(PATH, path: { id: @session_id }))
      end

      # Whether the agent is on the call, as this process last saw it.
      def started?
        @started
      end

      private

      def seen(session)
        @started = !session["call_id"].to_s.empty?
        session
      end
    end
  end
end
