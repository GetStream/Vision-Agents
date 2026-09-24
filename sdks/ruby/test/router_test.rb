# frozen_string_literal: true

require_relative "test_helper"

class TestRouter < LocalRouterTest
  def router(config = "healthcare", **options)
    VA::Router.new(config, client: client, **options)
  end

  def test_search_carries_the_config_and_the_labels
    @router.on(:post, "/v1/search", body: { "results" => [{ "url" => "https://example.com" }] })

    hits = router(tags: { env: "production" }).search("antibiotic guidance", results: 5)

    assert_equal "https://example.com", hits["results"][0]["url"]
    assert_equal({ "query" => "antibiotic guidance", "options" => { "results" => 5 }, "config_id" => "healthcare",
                   "tags" => { "env" => "production" } }, @router.last(:post, "/v1/search").json)
  end

  def test_an_option_the_block_does_not_have_is_refused
    assert_raises(VA::ConfigurationError) { router.search("x", reslts: 5) }
    assert_raises(VA::ConfigurationError) { router.stt.realtime(diarise: true) }
    assert_empty @router.requests
  end

  def test_a_transcription_stream_sends_audio_and_reads_transcripts
    heard = Thread::Queue.new
    @router.on_socket("/v1/stt/stream") do |peer|
      heard << peer.receive
      peer.send_frame(type: "started", provider: "deepgram", model: "nova-3")
      heard << peer.receive
      peer.send_frame(type: "transcript", text: "hello", final: true)
      peer.send_frame(type: "closed")
    end

    frames = router.stt.realtime(languages: ["en"]) do |stt|
      assert_equal "deepgram", stt.provider
      stt.send_audio("\x01\x02".b)
      stt.to_a
    end

    assert_equal({ "type" => "start", "stt" => { "languages" => ["en"] }, "sample_rate" => 16_000,
                   "config_id" => "healthcare" }, heard.pop)
    assert_equal "\x01\x02".b, heard.pop
    assert_equal [{ "type" => "transcript", "text" => "hello", "final" => true }], frames
  end

  def test_a_refused_start_is_raised
    @router.on_socket("/v1/llm/stream") do |peer|
      peer.receive
      peer.send_frame(type: "error", error: "no provider serves that")
      peer.send_frame(type: "closed")
    end

    error = assert_raises(VA::Error) { router.llm.realtime(target: "nobody/nothing") }
    assert_equal "no provider serves that", error.message
  end

  def test_speech_arrives_as_audio_behind_its_header
    @router.on_socket("/v1/tts/stream") do |peer|
      peer.receive
      peer.send_frame(type: "started")
      speak = peer.receive_type("speak")
      peer.send_binary([24_000, 1, 0].pack("Vvv") + "\x05\x06".b)
      peer.send_binary([24_000, 1, 0].pack("Vvv") + "\x07".b)
      peer.send_frame(type: "synthesis_complete", id: speak["id"])
      peer.receive
    end

    pieces = []
    spoken = router.tts.realtime(voice: "Kore") { |tts| tts.speak("hello") { |audio| pieces << audio } }

    assert_equal "\x05\x06\x07".b, spoken
    assert_equal [24_000, 1], [pieces[0].sample_rate, pieces[0].channels]
  end

  def test_an_answer_streams_its_deltas
    started = Thread::Queue.new
    @router.on_socket("/v1/llm/stream") do |peer|
      started << peer.receive
      peer.send_frame(type: "started")
      respond = peer.receive_type("respond")
      peer.send_frame(type: "delta", id: respond["id"], text: "Hel")
      peer.send_frame(type: "delta", id: respond["id"], text: "lo")
      peer.send_frame(type: "complete", id: respond["id"], text: "Hello", output_tokens: 2)
      peer.receive
    end

    deltas = []
    complete = router("").llm.realtime(target: "llm-fast", temperature: 0) do |llm|
      llm.respond("say hello", instructions: "Be brief") { |delta| deltas << delta }
    end

    assert_equal({ "type" => "start", "llm" => { "target" => "llm-fast", "temperature" => 0 },
                   "target" => "llm-fast" }, started.pop)
    assert_equal %w[Hel lo], deltas
    assert_equal "Hello", complete["text"]
  end

  def test_speech_to_speech_audio_carries_its_generation
    @router.on_socket("/v1/sts/stream") do |peer|
      peer.receive
      peer.send_frame(type: "started")
      peer.send_binary([24_000, 1, 1, 3, 0].pack("VvvVV") + "\x09".b)
      peer.send_frame(type: "closed")
    end

    audio = router.sts.realtime(voice: "Kore") { |sts| sts.next_message }

    assert_equal 3, audio.generation
    assert_equal "\x09".b, audio.pcm
  end

  def test_a_recording_is_waited_for
    polls = 0
    @router.on(:post, "/v1/stt/recordings", body: { "id" => "job_1", "status" => "queued" })
    @router.on(:get, "/v1/stt/recordings/job_1") do
      polls += 1
      { "id" => "job_1", "status" => polls < 2 ? "running" : "completed", "text" => "hello" }
    end

    transcript = router.stt.recording("https://example.com/call.mp3", diarize: true)

    assert_equal "hello", transcript["text"]
    assert_equal({ "source" => { "url" => "https://example.com/call.mp3" }, "options" => { "diarize" => true },
                   "config_id" => "healthcare" }, @router.last(:post, "/v1/stt/recordings").json)
  end

  def test_a_recording_with_a_callback_is_not_waited_for
    @router.on(:post, "/v1/tts/recordings", body: { "id" => "job_1", "status" => "queued" })

    job = router.tts.recording("A long chapter.", callback: "https://example.com/done", voice: "Kore")

    assert_equal "queued", job["status"]
    assert_empty @router.seen(:get, "/v1/tts/recordings/job_1")
  end

  def test_a_failed_recording_is_raised
    @router.on(:post, "/v1/stt/recordings", body: { "id" => "job_1", "status" => "failed", "error" => "bad audio" })

    error = assert_raises(VA::Error) { router.stt.recording("\x00\x01".b) }
    assert_equal "bad audio", error.message
    assert_equal({ "audio" => "AAE=" }, @router.last(:post, "/v1/stt/recordings").json["source"])
  end

  def test_configuring_one_modality_keeps_the_others
    @router.on(:get, "/v1/router/configs",
               body: [{ "id" => "cfg_1", "name" => "healthcare", "tts" => { "voice" => "Kore" } }])
    @router.on(:put, "/v1/router/configs/cfg_1") { |request| request.json.merge("id" => "cfg_1") }

    router.configure_stt(providers: %w[deepgram], profanity_filter: true)

    assert_equal({ "name" => "healthcare", "tts" => { "voice" => "Kore" },
                   "stt" => { "providers" => ["deepgram"], "profanity_filter" => true } },
                 @router.last(:put, "/v1/router/configs/cfg_1").json)
  end

  def test_configuring_needs_a_named_router
    assert_raises(VA::ConfigurationError) { router("").configure_tts(voice: "Kore") }
  end

  def test_router_folders_are_stored_by_name_and_stamped
    root = Dir.mktmpdir
    FileUtils.mkdir_p(File.join(root, "healthcare"))
    text = "stt:\n  providers: [deepgram]\n"
    File.write(File.join(root, "healthcare", "router.yaml"), text)
    @router.on(:get, "/v1/router/configs", body: [])
    @router.on(:post, "/v1/router/configs") { |request| [201, request.json.merge("id" => "cfg_1")] }

    stored = VA::Router.sync(root, client: client)

    assert_equal ["healthcare"], stored.map { |config| config["name"] }
    assert_equal({ "name" => "healthcare", "stt" => { "providers" => ["deepgram"] } },
                 @router.last(:post, "/v1/router/configs").json)
    stamp = JSON.parse(File.read(File.join(root, "healthcare", ".router_sync")))
    assert_equal Digest::MD5.hexdigest(text.strip), stamp["hash"]
  ensure
    FileUtils.rm_rf(root)
  end

  def test_a_router_file_with_an_unknown_key_is_refused
    root = Dir.mktmpdir
    FileUtils.mkdir_p(File.join(root, "healthcare"))
    File.write(File.join(root, "healthcare", "router.yaml"), "sst:\n  providers: [deepgram]\n")

    assert_raises(VA::ConfigurationError) { VA::Router.sync(root, client: client) }
  ensure
    FileUtils.rm_rf(root)
  end
end
