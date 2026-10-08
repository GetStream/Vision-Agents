# frozen_string_literal: true

module GetStream
  module VisionAgents
    # Where code the agent writes gets run.
    #
    # Code execution never happens on the live speech path: a sandbox is offered to the slower
    # model doing delegated work, not to the one holding the conversation.
    Sandbox = Data.define(:provider) do
      # A Daytona sandbox. The backend needs DAYTONA_API_KEY for it to do anything.
      def self.daytona
        new("daytona")
      end
    end

    # A configured agent, before and between the calls it holds.
    #
    # An agent here is configuration and function calling. The conversation itself (joining
    # the call, hearing the caller, answering and speaking) happens in the backend, and what
    # arrives here are the events saying so.
    #
    #   agent = GetStream::VisionAgents::Agent.new(config: "simple_voice_ai",
    #                                              cost_tracking: { env: "production" },
    #                                              memory_filter: { user_id: 123 })
    #   agent.join("call-1") do
    #     agent.responses.create("greet the user in one short sentence")
    #   end
    #
    # An agent holds one conversation at a time, which is what #responses answers for.
    class Agent
      # The memory filter key naming who the memories are about. Everything else narrows
      # recall.
      USER_KEY = "user_id"

      PIPELINE = %i[llm stt tts sts subagent search voice greeting language backchannel max_tokens
                    tool_timeout_ms keyterms video].freeze

      # How long a block form waits for somebody to be in a fresh call before running.
      PARTICIPANT_WAIT = 10
      # How long an outbound call's block waits for the person to answer.
      ANSWER_WAIT = 60

      attr_reader :name, :config, :folder, :user_id, :tools, :client, :session, :cost_tracking,
                  :memory_filter

      # @param config [String] a stored agent config to start from, by name.
      # @param folder [String, Folder] an agent directory. #sync stores it as a config named
      #   after it, which is then what a session starts from.
      # @param name [String] who the agent appears as. Defaults to the folder's or the
      #   config's name.
      # @param instructions [String] the system prompt, over whatever the config says.
      # @param guardrail [String] a guardrail.md, enforced in the backend.
      # @param pipeline [Hash] models over the config's: llm, stt, tts, sts, search, voice,
      #   greeting, language, backchannel, max_tokens, tool_timeout_ms, keyterms, video, and
      #   subagent, which is harness and so only reaches the config through #sync, as
      #   thinking_llm.
      # @param harness [String] which harness the config runs; nil is "default". Like skills
      #   and sandbox it is agent config, written by #sync, never sent with a session.
      # @param skills [Array<Skill>] skills of your own, stored by #sync in place of the
      #   built-in set.
      # @param sandbox [Sandbox] where delegated code runs, stored by #sync.
      # @param cost_tracking [Hash] labels on every request the session makes.
      # @param memory_filter [Hash] who the memories are about, under user_id, and what
      #   narrows recall.
      # @param client [Client, Hash] the router, or the options to build one from.
      # @param edge [Edge] creates the Stream calls the backend joins; built from the
      #   environment the first time a call is needed.
      def initialize(config: nil, folder: nil, name: nil, instructions: nil, guardrail: nil, pipeline: {},
                     harness: nil, skills: nil, sandbox: nil, cost_tracking: nil,
                     memory_filter: nil, user_id: nil, tools: nil, client: nil, edge: nil)
        @folder = folder.is_a?(String) ? Folder.load(folder) : folder
        @config = config || @folder&.name
        @name = name || @config
        raise ConfigurationError, "an agent needs a config, a folder or a name" if @name.to_s.empty?

        unknown = pipeline.keys.map(&:to_sym) - PIPELINE
        raise ConfigurationError, "the pipeline has no #{unknown.join(", ")}" unless unknown.empty?

        @instructions = instructions
        @guardrail = guardrail
        @pipeline = pipeline.transform_keys(&:to_sym)
        @harness = harness
        @skills = skills
        @sandbox = sandbox.is_a?(String) ? Sandbox.new(sandbox) : sandbox
        @cost_tracking = cost_tracking&.to_h { |key, value| [key.to_s, value.to_s] }
        @memory_filter = memory_filter&.to_h { |key, value| [key.to_s, value.to_s] }
        @user_id = user_id || Agent.user_id_of(@name)
        @tools = tools || Tools.new
        @client = client.is_a?(Client) ? client : Client.new(**(client || {}))
        @edge = edge
        @lock = Mutex.new
        validate_skills
      end

      # Creates the Stream calls the backend joins, built from the environment when needed,
      # so an agent that only ever chats needs no Stream credentials.
      def edge
        @lock.synchronize { @edge ||= Edge.new }
      end

      # One conversation's turns, for the conversation this agent is holding.
      def responses
        current.responses
      end

      # This agent's conversations: the ones being held and the ones that were.
      def sessions
        Sessions.new(self)
      end

      # The agent's knowledge base, named after it.
      def knowledge
        Knowledge.new(@config, @client)
      end

      # Has the backend join a call and hold a conversation on it.
      #
      # call is a call id, nil for a new call named after a random string, or an InboundCall
      # from dispatch, which is joined as it is because the caller is already in it. It
      # returns once the backend is in the call.
      #
      # With a block the session is yielded and closed afterwards, the way File.open closes
      # a file: leaving the block waits for the call to end unless wait_for_end is false.
      #
      # @param participant_wait_timeout [Numeric, nil] how long to wait for somebody else to
      #   be in the call before returning or yielding. 0 does not wait, nil waits for ever.
      #   An InboundCall does not wait; call wait_for_phone_participant on it instead.
      # @param options any CreateSessionRequest field (title, custom, greeting, ...), plus
      #   interim and decisions for what the events report.
      def join(call = nil, call_type: nil, wait_for_end: true, participant_wait_timeout: :default, **options, &block)
        if call.is_a?(InboundCall)
          request = { call_id: call.call_id, call_type: call.call_type }
          request[:phone] = { number: call.called_number } unless call.called_number.empty?
          session = open(request, options)
          call.attach(session)
          wait = participant_wait_timeout == :default ? 0 : participant_wait_timeout
        else
          created = edge.create_call(id: call, type: call_type, created_by: @user_id)
          session = open({ call_id: created.id, call_type: created.type }, options)
          wait = participant_wait_timeout == :default ? PARTICIPANT_WAIT : participant_wait_timeout
        end
        await_participant(session, wait)
        hold(session, wait_for_end, &block)
      end

      # Holds the conversation in writing rather than on a call.
      #
      # No call is joined, nothing is transcribed and nothing is spoken; the instructions,
      # skills and knowledge are the same. The conversation is kept in Stream Chat unless
      # incognito: true is given. With a block the session is closed afterwards.
      #
      # @param conversation_id [String] the channel an earlier session was held in, to resume.
      # @param agent_id [String] the conversation being answered, which names the channel
      #   replies are written into. A worker answering several conversations has to set it.
      def chat(conversation_id: nil, agent_id: nil, **options, &block)
        request = { text: true, conversation_id: conversation_id, agent_id: agent_id }
        hold(open(request, options), false, &block)
      end

      # Answers a message written to an agent that is not running, in the channel it came
      # from, so whoever wrote it is already reading the answer as it is generated.
      def reply(message, **options, &block)
        chat(conversation_id: message.cid,
             agent_id: message.agent_id.empty? ? nil : message.agent_id, **options, &block)
      end

      # Rings somebody and holds the conversation when they answer.
      #
      # The call is placed before the agent joins it: placing makes the routing rule pinned
      # to this call, and the leg's vendor id is what lets the agent press digits. The agent
      # is told it is navigating, so recordings are let finish and menus are answered rather
      # than talked over.
      #
      # @param from [String] one of your own numbers, which is what the person sees.
      # @param to [String] who to ring.
      # @param participant_wait_timeout [Numeric, nil] how long to wait for them to answer
      #   before returning or yielding.
      def outbound_call(from:, to:, call_id: nil, call_type: nil, ring_timeout: nil, initial_digits: nil,
                        headers: nil, custom: nil, wait_for_end: true, participant_wait_timeout: ANSWER_WAIT,
                        **options, &block)
        if from.to_s.empty? || to.to_s.empty?
          raise ConfigurationError, "a call needs a number to ring from and one to ring"
        end

        created = edge.create_call(id: call_id, type: call_type, created_by: @user_id)
        placed = @client.post("/v1/phone/calls", body: {
                                from: from, to: to, call_id: created.id, call_type: created.type,
                                ring_timeout_seconds: ring_timeout, initial_digits: initial_digits,
                                headers: headers, custom: custom, tags: @cost_tracking
                              })
        session = open({ call_id: created.id, call_type: created.type, navigating: true,
                         phone: { number: from, vendor_call_id: placed["vendor_call_id"] }.compact }, options)
        await_participant(session, participant_wait_timeout)
        hold(session, wait_for_end, &block)
      end

      # A link a person can open to join the call from a browser and hear the agent.
      def monitor_url(session = current)
        if session.call_id.empty?
          raise ConfigurationError, "a conversation held in writing has no call to watch"
        end

        edge.monitor_url(Edge::Call.new(id: session.call_id, type: session.call_type),
                         user_id: "monitor-#{session.id}", name: "Monitor")
      end

      # Blocks until the conversation this agent is holding ends.
      def finish(timeout: nil)
        session = @lock.synchronize { @session }
        session ? session.wait(timeout: timeout) : true
      end

      # Ends the conversation this agent is holding, if it is holding one.
      def close
        session = @lock.synchronize { @session }
        session&.close
      end

      def live?
        session = @lock.synchronize { @session }
        session ? session.live? : false
      end

      # Stores the agent in the backend: its instructions, guardrail, skills, knowledge and
      # what agent.yaml declares, in one POST /v1/agents/sync.
      #
      # The request carries the folder's fingerprint, taken the way the Go and Python SDKs
      # take it, and .agent_sync records it. A folder nothing has touched since is only read
      # back, not written. A setting left out leaves whatever is stored, so a model chosen in
      # the dashboard survives a sync that says nothing about it. What the code set wins over
      # what the folder says, and is part of the fingerprint.
      #
      # Server side only: how an agent is configured is not a device's to rewrite.
      #
      # @return [Hash] the SyncAgentResult: unchanged, and the config as stored.
      def sync
        hash = fingerprint
        if @folder&.stamp == hash
          stored = @client.get("/v1/agents/configs", query: { name: @config }).find { |each| each["name"] == @config }
          # A config deleted since the stamp was written is synced again rather than trusted.
          return { "unchanged" => true, "config" => stored } if stored
        end

        result = @client.post("/v1/agents/sync", body: sync_request.merge(hash: hash))
        @folder&.write_stamp(hash)
        result
      end

      # Changes some of how the agent is configured. A field left out keeps what is stored,
      # so setting a guardrail leaves the instructions, skills and models alone.
      #
      # Server side only: how an agent is configured is not a device's to change.
      #
      # @param patch any AgentConfigPatch field: instructions, guardrail, llm, voice, speed,
      #   harness, visible_tools, dispatch and the rest.
      # @return [Hash] the AgentConfig as it now is.
      def update_config(**patch)
        stored = @client.get("/v1/agents/configs", query: { name: @config }).find { |each| each["name"] == @config }
        raise Error, "there is no agent called #{@config} to update" unless stored

        @client.patch("/v1/agents/configs/{id}", path: { id: stored.fetch("id") }, body: patch)
      end

      # Turns a name into something a call can be joined under.
      def self.user_id_of(name)
        id = name.to_s.downcase.gsub(/[^a-z0-9_-]/, "-").gsub(/\A-+|-+\z/, "")
        id.empty? ? "vision-agent" : id
      end

      # Renders the agent into a session request. Only what the code set is sent: the config
      # the session starts from decides the rest, and a schema default copied in here would
      # silently overrule it. The harness is the config's, so none of it is sent here.
      def session_request(call, options)
        request = {
          agent: @config, user_id: @user_id, user_name: @name, agent_id: @user_id,
          instructions: @instructions, tags: @cost_tracking, memory: memory
        }
        request.merge!(@pipeline.except(:language, :subagent))
        request[:languages] = [@pipeline[:language]] if @pipeline[:language]
        request[:tools] = @tools.declarations unless @tools.empty?
        request.merge!(call.compact).merge!(options).compact
      end

      private

      def open(call, options)
        watch = { interim: options.delete(:interim) || false, decisions: options.delete(:decisions) || false }
        @lock.synchronize do
          raise ConfigurationError, "#{@name} is already holding a conversation" if @session&.live?
        end
        created = @client.post("/v1/agents/sessions", body: session_request(call, options))
        session = Session.watching(@client, created, tools: @tools, **watch)
        @lock.synchronize { @session = session }
      end

      def hold(session, wait_for_end)
        return session unless block_given?

        begin
          result = yield session
          session.wait if wait_for_end
          result
        ensure
          session.close
        end
      end

      def await_participant(session, timeout)
        return if timeout&.zero?

        session.wait_for_participant(timeout: timeout)
      end

      def current
        session = @lock.synchronize { @session }
        return session if session

        raise ConfigurationError, "#{@name} is not holding a conversation; join a call or open a chat first"
      end

      def memory
        return nil if @memory_filter.nil? || @memory_filter.empty?

        narrowing = @memory_filter.except(USER_KEY)
        { user_id: @memory_filter[USER_KEY], filter: (narrowing unless narrowing.empty?) }.compact
      end

      def synced_skills
        @skills && !@skills.empty? ? @skills : (@folder&.skills || [])
      end

      def instructions_to_sync
        @instructions || @folder&.instructions || ""
      end

      def guardrail_to_sync
        @guardrail || @folder&.guardrail || ""
      end

      # Go's syncFolder fingerprint: the folder with the code's overrides, then the code's
      # harness, subagent and sandbox and the cost labels folded in when any is set.
      def fingerprint
        hash = Folder.fingerprint(@folder&.declaration.to_s, instructions_to_sync, guardrail_to_sync,
                                  synced_skills, @folder&.knowledge || [], @folder&.knowledge_urls || [],
                                  @folder&.simulations)
        harness = "#{@harness}#{@pipeline[:subagent]}#{@sandbox&.provider}"
        return hash if harness.empty? && (@cost_tracking.nil? || @cost_tracking.empty?)

        Folder.fingerprint(hash, harness, go_map(@cost_tracking || {}), [], [], [])
      end

      # How Go's fmt.Sprint writes a map[string]string, which is what the fingerprint hashes.
      def go_map(map)
        "map[#{map.sort.map { |key, value| "#{key}:#{value}" }.join(" ")}]"
      end

      def sync_request
        settings = @folder&.settings || {}
        tags = (settings["tags"] || {}).merge(@cost_tracking || {})
        video = settings["video"]
        {
          name: @config || @name,
          instructions: presence(instructions_to_sync),
          guardrail: presence(guardrail_to_sync),
          skills: synced_skills.map { |skill| skill.request.merge(config_id: "") }.then { |s| s unless s.empty? },
          knowledge: @folder&.knowledge&.map { |document| { source: document.source, text: document.text } }
                            &.then { |d| d unless d.empty? },
          knowledge_urls: @folder&.knowledge_urls&.map(&:declaration)&.then { |p| p unless p.empty? },
          # nil when the folder has no simulations/, which leaves the stored ones alone; empty
          # when it has none, which deletes them.
          simulations: @folder&.simulations&.map(&:declaration),
          mode: presence(settings["mode"]), stt: presence(settings["stt"]), tts: presence(settings["tts"]),
          sts: settings["sts"], voice: presence(settings["voice"]),
          speed: (settings["speed"] unless settings["speed"].to_f.zero?), llm: presence(settings["llm"]),
          harness: presence(@harness) || presence(settings["harness"]),
          search: presence(settings["search"]), greeting: presence(settings["greeting"]),
          agent_plugins: settings["plugins"]&.then { |p| p unless p.empty? },
          keyterms: settings["keyterms"]&.then { |k| k unless k.empty? },
          video: video && { source: video["source"], max_frames: video["max_frames"] }.compact,
          dispatch: settings["dispatch"],
          thinking_llm: presence(@pipeline[:subagent].to_s) || presence(settings["subagent"]),
          sandbox: @sandbox&.provider || presence(settings["sandbox"]),
          tags: (tags unless tags.empty?)
        }
      end

      def presence(value)
        value.nil? || value.to_s.empty? ? nil : value
      end

      def validate_skills
        (@skills || []).each do |skill|
          raise ConfigurationError, "a skill needs a name" if skill.name.to_s.empty?
          if skill.description.to_s.empty?
            raise ConfigurationError, "#{skill.name} needs a description, since it is all the fast model sees"
          end
          if skill.instructions.to_s.empty?
            raise ConfigurationError, "#{skill.name} needs instructions, since they are what the subagent answers under"
          end
        end
      end
    end

    # One agent's conversations, addressed by the name it is configured under.
    #
    #   docs = api.agent("docs")
    #   session = docs.sessions.create(title: "Is Stream better?")
    #   docs.sessions.search("billing")["items"]
    class Sessions
      def initialize(agent)
        @agent = agent
        @client = agent.client
      end

      # Opens a conversation and starts watching it. Held in writing unless a call_id is given.
      #
      # @param options any CreateSessionRequest field (id, a UUID to hold the session by;
      #   title, description, project_id, custom, incognito, ...), plus interim and decisions.
      def create(**options)
        watch = { interim: options.delete(:interim) || false, decisions: options.delete(:decisions) || false }
        options[:text] = true unless options[:call_id]
        created = @client.post("/v1/agents/sessions", body: @agent.session_request({}, options))
        Session.watching(@client, created, tools: @agent.tools, **watch)
      end

      # A page of the agent's conversations, most recently updated first, the ones that ended
      # included. Rows rather than live handles: reading a conversation back is not holding
      # one. Pass the page's next_cursor as cursor, with the same filters, for the next.
      #
      # @param project_id [String] one project's. A search covers every project, so it
      #   cannot be combined with text.
      # @param user_id [String] one user's, which only a server-side caller may ask for.
      # @param modality [String] text, voice or video.
      # @param state [String] live or ended.
      # @param agent_id [String] the sessions created with this agent id.
      # @param limit [Integer] up to 200; nil is 25.
      # @return [Hash] the SessionPage: items, has_more and next_cursor.
      def query(project_id: nil, user_id: nil, modality: nil, state: nil, agent_id: nil, limit: nil, cursor: nil)
        filter = { agent: @agent.config, project_id: project_id, user_id: user_id, modality: modality, state: state,
                   agent_id: agent_id }
        @client.post("/v1/agents/sessions/query", body: query_body(filter, limit, cursor))
      end

      # Finds a conversation by what it was called: title, description, project and agent
      # name, best match first. Takes the filters of #query but project_id, since a search
      # covers every project, and pages the same way.
      def search(text, user_id: nil, modality: nil, state: nil, agent_id: nil, limit: nil, cursor: nil)
        filter = { agent: @agent.config, user_id: user_id, modality: modality, state: state, agent_id: agent_id,
                   text: { "$q" => text } }
        @client.post("/v1/agents/sessions/query", body: query_body(filter, limit, cursor))
      end

      # One conversation, whether or not it is still being held.
      def get(id)
        @client.get("/v1/agents/sessions/{id}", path: { id: id })
      end

      # Changes one conversation, whether or not it is still being held. One that ended can
      # still be renamed and relabelled; instructions, models and voice need it running.
      # Server side only.
      #
      # @param fields any UpdateSessionRequest field; see Session#update.
      # @return [Hash] the session as the router now has it.
      def update(id, **fields)
        @client.patch("/v1/agents/sessions/{id}", path: { id: id }, body: fields)
      end

      # Deletes a conversation, running or ended: it is stopped, and its turns and what it
      # remembered are deleted with it. The user's other memories are kept.
      def delete(id)
        @client.delete("/v1/agents/sessions/{id}", path: { id: id })
      end

      # Deletes what one conversation remembered, running or ended. Server side only.
      def delete_memories(id)
        @client.delete("/v1/agents/sessions/{id}/memories", path: { id: id })
      end

      # A session's turns, read back without holding the conversation.
      def responses(id)
        Responses.new(@client, id)
      end

      private

      # Each filter field is a bare value, the short form of {"$eq": value}; nil leaves it out.
      def query_body(filter, limit, cursor)
        { filter: filter.compact, limit: limit, cursor: cursor }
      end
    end
  end
end
