# frozen_string_literal: true

require "digest"
require "psych"
require "securerandom"

module GetStream
  module VisionAgents
    # Everything the acceleration backend routes, configured once.
    #
    #   router = GetStream::VisionAgents::Router.new("healthcare")
    #   router.stt.realtime(languages: ["en"]) do |stt|
    #     stt.send_audio(pcm)
    #     stt.each { |frame| puts frame["text"] if frame["type"] == "transcript" }
    #   end
    #   transcript = router.stt.recording("https://example.com/call.mp3", diarize: true)
    #   hits = router.search("perioperative antibiotic guidance", results: 5)
    #
    # Everything in the named config is a default, and every keyword on a call overrides one
    # field of it.
    class Router
      ROUTER_FILE = "router.yaml"
      ROUTER_STAMP = ".router_sync"
      ROUTER_KEYS = %w[name description stt tts llm sts search].freeze
      # How often a recording job is asked about.
      POLL_INTERVAL = 0.5

      attr_reader :config, :tags, :client, :stt, :tts, :llm, :sts

      # @param config [String] a stored router config, by name or id. Without one every call
      #   says what it wants for itself.
      # @param tags [Hash] cost labels carried onto everything routed here.
      def initialize(config = "", tags: {}, client: nil)
        @config = config.to_s
        @tags = tags.to_h { |key, value| [key.to_s, value.to_s] }
        @client = client.is_a?(Client) ? client : Client.new(**(client || {}))
        @stt = SpeechToText.new(self)
        @tts = TextToSpeech.new(self)
        @llm = Completions.new(self)
        @sts = SpeechToSpeech.new(self)
      end

      # Answers a query out of what is true now.
      #
      # @param options any field of the search block: providers, depth, results,
      #   include_domains, category, max_age_hours, location, contents.
      # @return [Hash] the SearchAnswer: results, and the provider's own answer where it
      #   wrote one.
      def search(query, **options)
        @client.post("/v1/search", body: labelled(query: query, options: Router.block("SearchOptions", options)))
      end

      # Stores how this router transcribes, leaving the other modalities as stored.
      def configure_stt(**options)
        configure("stt", "SttOptions", options)
      end

      # Stores how this router speaks, leaving the other modalities as stored.
      def configure_tts(**options)
        configure("tts", "TtsOptions", options)
      end

      # Stores how this router holds a conversation with one native audio model.
      def configure_sts(**options)
        configure("sts", "StsOptions", options)
      end

      # The request with the router's config and labels on it.
      def labelled(body)
        body[:config_id] = @config unless @config.empty?
        body[:tags] = @tags unless @tags.empty?
        body
      end

      # Opens a modality socket and waits for the router to say the model took the options.
      def open(modality, block, kind: Realtime, **start)
        frame = labelled({ type: "start", modality.to_sym => block }.merge(start))
        # The router refuses a start frame with neither a config nor a top-level target, and
        # does not look for one in the block.
        frame[:target] = block["target"] if block&.key?("target")
        socket = @client.socket("/v1/{modality}/stream", path: { modality: modality })
        kind.open(socket, modality, frame)
      end

      # Waits for a recording job to stop being one.
      def until_done(job, template)
        while %w[queued running].include?(job["status"])
          sleep POLL_INTERVAL
          job = @client.get(template, path: { id: job.fetch("id") })
        end
        raise Error, job["error"].to_s.empty? ? "the recording failed" : job["error"] if job["status"] == "failed"

        job
      end

      # Stores a router config a Router can then be named after. Found by name first, so
      # defining one twice edits it rather than storing a copy.
      def self.define(name, client: nil, stt: nil, tts: nil, llm: nil, sts: nil, search: nil)
        client = client.is_a?(Client) ? client : Client.new(**(client || {}))
        request = { name: name, stt: block("SttOptions", stt), tts: block("TtsOptions", tts),
                    llm: block("LlmOptions", llm), sts: block("StsOptions", sts),
                    search: block("SearchOptions", search) }
        store(client, request, find(client, name))
      end

      # Stores every {name}/router.yaml under a directory, each named by its name key or its
      # folder, and writes .router_sync next to each with the md5 of what was stored.
      #
      # @return [Array<Hash>] the stored configs, in the order the folders were read.
      def self.sync(directory, client: nil)
        folders = Dir.children(directory).sort.map { |each| File.join(directory, each) }
                     .select { |each| File.file?(File.join(each, ROUTER_FILE)) }
        raise ConfigurationError, "#{directory} holds no {name}/#{ROUTER_FILE} router configs" if folders.empty?

        folders.map do |folder|
          text = File.read(File.join(folder, ROUTER_FILE))
          described = read(text, File.join(folder, ROUTER_FILE))
          described.delete("description")
          name = described.delete("name") || File.basename(folder)
          stored = define(name, client: client, **described.transform_keys(&:to_sym))
          Folder.write_stamp(folder, ROUTER_STAMP, Digest::MD5.hexdigest(text.strip))
          stored
        end
      end

      # Checks options against the block the spec gives them, so a misspelt option is refused
      # here rather than dropped somewhere along the way.
      def self.block(schema, options)
        return nil if options.nil?

        options = options.to_h { |key, value| [key.to_s, value] }.compact
        unknown = options.keys - Generated::SCHEMAS.fetch(schema)[:properties]
        raise ConfigurationError, "#{schema} has no #{unknown.join(", ")}" unless unknown.empty?

        options
      end

      # Yields a stream and closes it afterwards, or returns it open without a block.
      def self.hold(stream)
        return stream unless block_given?

        begin
          yield stream
        ensure
          stream.close
        end
      end

      # What to transcribe: a URL is passed on for the provider to fetch, which makes a long
      # recording somebody else's bandwidth. A path is read and sent inline, and so are bytes.
      def self.source(source)
        raise ConfigurationError, "a recording needs a source" unless source.is_a?(String)
        return { audio: [source].pack("m0") } if source.encoding == Encoding::BINARY
        return { url: source } if source.match?(%r{\Ahttps?://})
        return { audio: [File.binread(source)].pack("m0") } if File.file?(source)

        raise ConfigurationError, "#{source[0, 80].inspect} is neither a URL nor a file that exists"
      end

      def self.read(text, file)
        described = Psych.safe_load(text, aliases: false) || {}
        raise ConfigurationError, "#{file} must be a mapping" unless described.is_a?(Hash)

        unknown = described.keys - ROUTER_KEYS
        unless unknown.empty?
          raise ConfigurationError, "#{file} has no #{unknown.join(", ")}; it takes #{ROUTER_KEYS.join(", ")}"
        end

        described
      end

      def self.find(client, name)
        client.get("/v1/router/configs").find { |each| each["name"] == name }
      end

      def self.store(client, request, stored)
        request = request.compact
        return client.post("/v1/router/configs", body: request) unless stored

        client.put("/v1/router/configs/{id}", path: { id: stored.fetch("id") }, body: request)
      end

      private

      def configure(modality, schema, options)
        if @config.empty?
          raise ConfigurationError, "configure_#{modality} writes a named config, so the router needs a name"
        end

        stored = Router.find(@client, @config)
        # Carried forward rather than restated: a config is one row, and writing the speech
        # half of it should not silently drop the voice half.
        request = { name: @config }
        %w[stt tts llm sts search].each { |key| request[key.to_sym] = stored[key] } if stored
        request[modality.to_sym] = Router.block(schema, options)
        Router.store(@client, request, stored)
      end
    end

    # One modality socket, open. Frames arrive as Hashes, audio as Audio.
    class Realtime
      include Enumerable

      # A piece of the model's voice: 16-bit little-endian PCM and how to play it. The
      # generation is speech-to-speech's, and lets the tail of a reply that was cut off be
      # dropped.
      Audio = Data.define(:pcm, :sample_rate, :channels, :generation)

      # How long the router may take to say the model took the options.
      READY_TIMEOUT = 30
      # rate, channels and a reserved field; speech-to-speech adds a generation and an index.
      HEADER = 8
      STS_HEADER = 16

      attr_reader :provider, :model

      def self.open(socket, modality, start)
        realtime = new(socket, modality)
        realtime.start(start)
        realtime
      rescue StandardError
        socket.close
        raise
      end

      def initialize(socket, modality)
        @socket = socket
        @header = modality == "sts" ? STS_HEADER : HEADER
      end

      # Sends the start frame and waits for started, raising the router's refusal otherwise.
      def start(frame)
        @socket.send_frame(frame)
        loop do
          reply = @socket.receive(timeout: READY_TIMEOUT)
          raise Error, "the router did not start the stream in #{READY_TIMEOUT}s" if reply.nil?
          next unless reply.is_a?(Hash)
          raise Error, reply["error"].to_s if reply["type"] == "error"
          next unless reply["type"] == "started"

          @provider = reply["provider"]
          @model = reply["model"]
          return
        end
      end

      # Everything the router sends until the stream closes.
      def each
        return enum_for(:each) unless block_given?

        while (message = next_message)
          yield message
        end
      end

      # The next thing the router sends, or nil once the stream has closed.
      def next_message(timeout: nil)
        loop do
          message = @socket.receive(timeout: timeout)
          return nil if message.nil? || (message.is_a?(Hash) && message["type"] == "closed")
          return message unless message.is_a?(String)

          # A frame too short to hold its header is nothing that can be played.
          audio = audio(message)
          return audio if audio
        end
      end

      def send_frame(frame)
        @socket.send_frame(frame)
      end

      # Sends 16-bit little-endian mono PCM, at the rate the stream was opened at. The
      # router's audio does not stop until the socket closes, which is how it is told the
      # audio is over.
      def send_audio(pcm)
        @socket.send_binary(pcm)
      end

      # Abandons what is being generated.
      def interrupt
        @socket.send_frame(type: "interrupt")
      end

      def open?
        @socket.open?
      end

      def close
        @socket.close
      end

      private

      def audio(bytes)
        return nil if bytes.bytesize < @header

        rate, channels = bytes.unpack("Vv")
        generation = @header == STS_HEADER ? bytes.byteslice(8, 4).unpack1("V") : 0
        Audio.new(pcm: bytes.byteslice(@header..), sample_rate: rate, channels: channels, generation: generation)
      end
    end

    # A text-to-speech stream, one utterance at a time.
    class Speaking < Realtime
      # Speaks text, yielding each piece of audio as it arrives.
      #
      # @return [String] all of it, as 16-bit PCM.
      def speak(text, voice: nil, language: nil)
        send_frame({ type: "speak", id: SecureRandom.uuid, text: text, voice: voice, language: language,
                     final: true }.compact)
        spoken = +"".b
        while (message = next_message)
          if message.is_a?(Audio)
            spoken << message.pcm
            yield message if block_given?
            next
          end
          raise Error, message["error"].to_s if message["type"] == "error"
          return spoken if message["type"] == "synthesis_complete"
        end
        raise SocketClosedError, "the stream closed before the speech was complete"
      end
    end

    # An LLM stream, one completion at a time.
    class Answering < Realtime
      # Answers messages, yielding each delta of text as it is written.
      #
      # @param messages [String, Array<Hash>] a question, or the conversation as role and
      #   content pairs.
      # @return [Hash] the complete frame: text, input_tokens, output_tokens, model.
      def respond(messages, instructions: nil, max_tokens: nil)
        messages = [{ role: "user", content: messages }] if messages.is_a?(String)
        send_frame({ type: "respond", id: SecureRandom.uuid, instructions: instructions, messages: messages,
                     max_tokens: max_tokens }.compact)
        while (message = next_message)
          next unless message.is_a?(Hash)

          case message["type"]
          when "delta" then yield message["text"].to_s if block_given?
          when "complete" then return message
          when "error" then raise Error, message["error"].to_s
          end
        end
        raise SocketClosedError, "the stream closed before the answer was complete"
      end
    end

    # Transcription, live or from a recording.
    class SpeechToText
      SAMPLE_RATE = 16_000

      def initialize(router)
        @router = router
      end

      # A transcription stream: send 16 kHz mono PCM with send_audio and read transcript
      # frames off it.
      #
      # @param options any field of the stt block: target, languages, interim, endpointing,
      #   diarize, keyterms, format, redact.
      def realtime(**options, &block)
        Router.hold(@router.open("stt", Router.block("SttOptions", options), sample_rate: SAMPLE_RATE), &block)
      end

      # Transcribes a whole recording, with the batch half of a vendor rather than the
      # streaming one. Waits for the job unless a callback URL is given.
      #
      # @param source [String] a URL for the provider to fetch, a path, or the audio itself
      #   as a binary String.
      def recording(source, callback: nil, **options)
        body = @router.labelled(source: Router.source(source), options: Router.block("SttOptions", options),
                                callback: callback)
        job = @router.client.post("/v1/stt/recordings", body: body)
        callback ? job : @router.until_done(job, "/v1/stt/recordings/{id}")
      end
    end

    # A voice, live or recorded.
    class TextToSpeech
      def initialize(router)
        @router = router
      end

      # A speaking stream.
      #
      # @param options any field of the tts block: target, providers, voice, languages,
      #   speed, emotion, stability, format.
      def realtime(**options, &block)
        Router.hold(@router.open("tts", Router.block("TtsOptions", options), kind: Speaking), &block)
      end

      # Speaks a whole text into one file. Waits for the job unless a callback URL is given.
      def recording(text, callback: nil, **options)
        body = @router.labelled(text: text, options: Router.block("TtsOptions", options), callback: callback)
        job = @router.client.post("/v1/tts/recordings", body: body)
        callback ? job : @router.until_done(job, "/v1/tts/recordings/{id}")
      end
    end

    # The model that answers, with the answer arriving as it is written. There is no
    # recording: a completion is already whole by the time it is returned.
    class Completions
      def initialize(router)
        @router = router
      end

      # An answering stream.
      #
      # @param options any field of the llm block: target, providers, max_output_tokens,
      #   temperature, reasoning_effort, format, verbosity, tool_choice.
      def realtime(**options, &block)
        Router.hold(@router.open("llm", Router.block("LlmOptions", options), kind: Answering), &block)
      end
    end

    # A conversation with one native audio model. Live only: there is no recording of one.
    class SpeechToSpeech
      SAMPLE_RATE = 16_000

      def initialize(router)
        @router = router
      end

      # A speech-to-speech stream: send 16 kHz mono PCM with send_audio, text with
      # send_frame(type: "text", text:), and read audio and transcript frames off it.
      #
      # @param options any field of the sts block: target, instructions, voice, languages,
      #   turn_detection, silence_ms, interrupt_response, input_transcript,
      #   output_transcript, tools, text, images.
      def realtime(**options, &block)
        Router.hold(@router.open("sts", Router.block("StsOptions", options), sample_rate: SAMPLE_RATE), &block)
      end
    end
  end
end
