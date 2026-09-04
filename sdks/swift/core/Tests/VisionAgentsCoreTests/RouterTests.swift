import Foundation
import Testing

@testable import VisionAgentsCore

/// The modality frames and the option blocks, checked against what the router writes and
/// reads. Frames are quoted as `streamws.go` writes them, so a rename there fails here.
@Suite struct RouterTests {
    private let router = Router(
        url: URL(string: "http://localhost:8080")!, customerID: "acme", config: "healthcare",
        tags: ["team": "clinical"])

    @Test func aTranscriptCarriesWhoSaidItAndWhetherItIsFinal() throws {
        let frame = try JSONDecoder().decode(
            RoutedFrame.self,
            from: Data(
                #"{"type":"transcript","text":"take two","final":true,"speaker":"1","language":"en"}"#
                    .utf8))

        #expect(frame.kind == .transcript)
        let transcript = Transcript(frame)
        #expect(transcript.text == "take two")
        #expect(transcript.isFinal)
        #expect(transcript.speaker == "1")
        #expect(transcript.language == "en")
    }

    @Test func anUnknownFrameKeepsItsNameAndItsFields() throws {
        let frame = try JSONDecoder().decode(
            RoutedFrame.self, from: Data(#"{"type":"warmed","provider":"deepgram"}"#.utf8))

        #expect(frame.type == "warmed")
        #expect(frame.kind == nil)
        #expect(frame["provider"].stringValue == "deepgram")
    }

    @Test func aDeltaAndTheFrameThatFinishesItAreToldApart() throws {
        let delta = Answer(RoutedFrame(type: "delta", fields: ["text": .string("two ")]))
        let complete = Answer(
            RoutedFrame(type: "complete", fields: ["text": .string("two grams")]))

        #expect(delta.delta == "two " && delta.text.isEmpty && !delta.isComplete)
        #expect(complete.text == "two grams" && complete.delta.isEmpty && complete.isComplete)
    }

    /// The header is a little-endian uint32 rate, a uint16 channel count, and two bytes held
    /// back so the samples that follow stay aligned.
    @Test func anAudioFrameSaysHowToPlayWhatFollows() throws {
        var payload = Data([0x80, 0x3E, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00])
        payload.append(contentsOf: [0x01, 0x00, 0xFF, 0x7F])

        let audio = try #require(SpokenAudio(payload))
        #expect(audio.sampleRate == 16000)
        #expect(audio.channels == 1)
        #expect(audio.samples.count == 4)
    }

    @Test func somethingTooShortToHaveAHeaderIsNotAudio() {
        #expect(SpokenAudio(Data([0x00, 0x01])) == nil)
    }

    @Test func aStartFrameCarriesTheConfigTheTagsAndTheModalityBlock() throws {
        var wanted = TranscriptionOptions()
        wanted.diarize = true
        wanted.keyterms = ["metformin"]

        let frame = router.startFrame(modality: "stt", block: wanted.frame)

        #expect(frame["type"]?.stringValue == "start")
        #expect(frame["config_id"]?.stringValue == "healthcare")
        #expect(frame["tags"]?.objectValue["team"]?.stringValue == "clinical")
        let block = try #require(frame["stt"]?.objectValue)
        #expect(block["diarize"]?.boolValue == true)
        #expect(block["keyterms"]?.arrayValue.map(\.stringValue) == ["metformin"])
    }

    /// A field nobody set is left out rather than sent as a copy of the router's own default,
    /// which is how a caller would otherwise lose what their config named.
    @Test func anOptionNobodySetIsNotSent() {
        var wanted = VoiceOptions()
        wanted.speed = 1.2

        let block = wanted.frame

        #expect(block["speed"]?.doubleValue == 1.2)
        #expect(block["voice"] == nil)
        #expect(block["stability"] == nil)
        #expect(block["languages"] == nil)
    }

    @Test func theBlockUsesTheNamesTheRouterReads() {
        var wanted = ModelOptions()
        wanted.maxOutputTokens = 200
        wanted.reasoningEffort = .low

        let block = wanted.frame

        #expect(block["max_output_tokens"]?.doubleValue == 200)
        #expect(block["reasoning_effort"]?.stringValue == "low")
    }

    @Test func aQuestionEncodesToWhatTheRouterReads() {
        var question = Question("summarise the call")
        question.instructions = "be brief"
        question.temperature = 0.2

        let frame = question.frame

        #expect(frame["type"]?.stringValue == "respond")
        #expect(frame["instructions"]?.stringValue == "be brief")
        #expect(frame["temperature"]?.doubleValue == 0.2)
        let said = try? #require(frame["messages"]?.arrayValue.first?.objectValue)
        #expect(said?["role"]?.stringValue == "user")
        #expect(said?["content"]?.stringValue == "summarise the call")
    }

    /// Routing has to be told where to go, so a socket with neither a config nor a target is
    /// refused here rather than opened and closed again by the router.
    @Test func aSocketWithNothingToRouteOnIsRefused() {
        let bare = Router(url: URL(string: "http://localhost:8080")!, customerID: "acme")

        #expect(throws: AgentsError.self) { try bare.stt.realtime() }
        #expect(throws: Never.self) { try bare.stt.realtime(targeted()) }
        #expect(throws: Never.self) { try router.stt.realtime() }
    }

    @Test func aRecordingIsEitherAUrlOrTheAudioItself() {
        let hosted = Recording.url(URL(string: "https://example.com/call.mp3")!)
        let inline = Recording.audio(Data([0x01, 0x02]))

        #expect(hosted.schema.url == "https://example.com/call.mp3")
        #expect(hosted.schema.audio == nil)
        #expect(inline.schema.url == nil)
        #expect(inline.schema.audio?.data == ArraySlice([0x01, 0x02]))
    }

    private func targeted() -> TranscriptionOptions {
        var wanted = TranscriptionOptions()
        wanted.target = "en-low-latency"
        return wanted
    }
}
