//go:build integration

package api

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// StreamsSuite covers the four modality sockets, which a caller holding its own pipeline
// uses instead of a session: the start frame that says where to route, and what crosses in
// each direction afterwards.
type StreamsSuite struct {
	RouterSuite
}

func TestStreamsSuite(t *testing.T) {
	runSuite(t, new(StreamsSuite))
}

func (s *StreamsSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *StreamsSuite) TestAModalityThisDeploymentDoesNotRouteIsNotFound() {
	_, status := s.serverClient.watch("/v1/nonsense/stream")

	s.Equal(http.StatusNotFound, status)
}

func (s *StreamsSuite) TestAudioSentUpComesBackTranscribed() {
	listening := s.start("/v1/stt/stream", frame{"target": "en-low-latency"})

	started := s.nextFrame(listening)
	s.Equal("started", started["type"])
	s.Equal("stub", started["provider"])

	pcm := audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1}
	s.Require().NoError(listening.WriteMessage(websocket.BinaryMessage, pcm.Bytes()))
	s.ears.emitter.Send(stt.Transcript{
		Text: "a call costs a penny", Mode: stt.ModeFinal, Confidence: 0.9,
		Language: "en", Provider: "stub", Model: "stub-stt",
	})

	transcript := s.nextFrame(listening)
	s.Equal("transcript", transcript["type"])
	s.Equal("a call costs a penny", transcript["text"])
	s.Equal(true, transcript["final"])
	s.Equal("en", transcript["language"])
}

func (s *StreamsSuite) TestAVendorThatRefusesTheAccountSaysSoOnce() {
	listening := s.refused()

	silence := audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1}
	for range 20 {
		s.Require().NoError(listening.WriteMessage(websocket.BinaryMessage, silence.Bytes()))
	}
	// Frames are read in order, so by the time the close is answered every frame of audio
	// before it has been handled.
	s.Require().NoError(listening.WriteMessage(websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, "")))
	s.Require().NoError(listening.SetReadDeadline(time.Now().Add(5 * time.Second)))
	var repeated int
	for {
		_, payload, err := listening.ReadMessage()
		if err != nil {
			break
		}
		var decoded frame
		if json.Unmarshal(payload, &decoded) == nil && decoded["type"] == "error" {
			repeated++
		}
	}
	s.Zero(repeated, "audio sent after the refusal is not answered with an error per frame")
}

func (s *StreamsSuite) TestAModelThatRefusedTheAccountIsListedUnavailable() {
	// The turns it served before the refusal, which an error rate would weigh it against.
	for range 3 {
		s.Require().NoError(s.live.RecordRequest(context.Background(), live.Usage{
			Modality: "stt", CustomerID: s.customerID(), Provider: "refusing", Model: refusingModel,
			LatencyMs: 150, AudioMs: 1000, Success: true,
		}))
	}
	s.Require().NoError(s.refused().Close())

	s.Require().Eventually(func() bool {
		var providers []Provider
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/stt/providers", nil, &providers))
		for _, provider := range providers {
			if provider.Model == refusingModel {
				return !provider.Health.Available
			}
		}
		return false
	}, 5*time.Second, 20*time.Millisecond)
	var recorded store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&recorded).
		Where("model = ? AND NOT success", refusingModel).Limit(1).Scan(context.Background()))
	s.Equal(termsRefusal, recorded.ErrorMessage, "the row says why")
}

func (s *StreamsSuite) TestATargetInTheOptionBlockIsEnoughToRoute() {
	answering := s.start("/v1/llm/stream", frame{"llm": frame{"target": "echo/echo-model"}})

	started := s.nextFrame(answering)
	s.Equal("started", started["type"], "the options sent outright name the target, as the SDKs send them")
}

func (s *StreamsSuite) TestAFrameNamingNothingToRouteToIsRefused() {
	// A target for another modality says nothing about where this socket's audio goes.
	listening := s.start("/v1/stt/stream", frame{"tts": frame{"target": "en-low-latency"}})

	refused := s.nextFrame(listening)
	s.Equal("error", refused["type"])
}

func (s *StreamsSuite) TestAnOptionNothingRecognisesIsRefusedRatherThanIgnored() {
	listening := s.start("/v1/stt/stream", frame{
		"target": "en-low-latency", "stt": frame{"mode": "whichever"}})

	refused := s.nextFrame(listening)
	s.Equal("error", refused["type"])
	s.Equal("closed", s.nextFrame(listening)["type"], "the socket says it is done")
}

func (s *StreamsSuite) TestMoreKeytermsThanASessionMayNameAreRefusedAtTheStart() {
	keyterms := make([]string, stt.MaxKeyterms+1)
	for i := range keyterms {
		keyterms[i] = fmt.Sprintf("term %d", i)
	}
	listening := s.start("/v1/stt/stream", frame{
		"target": "stub/stub-model", "stt": frame{"keyterms": keyterms}})

	refused := s.nextFrame(listening)
	s.Equal("error", refused["type"])
	s.Equal("stt: at most 100 keyterms may be named, and this asks for 101", refused["error"])
	s.Equal("closed", s.nextFrame(listening)["type"])
}

func (s *StreamsSuite) TestARefusalReachesACallerStillSendingAudio() {
	// A refusal larger than the caller's receive buffer, 128 KiB on a Mac
	// (net.inet.tcp.recvspace), but not than that and the router's send buffer together,
	// is still on its way when the router ends the socket, the way a small one is on a
	// slow network.
	target := "nobody/" + strings.Repeat("x", 200<<10)
	listening := s.start("/v1/stt/stream", frame{"target": target})
	speech := audio.PcmData{Samples: make([]int16, 1600), SampleRate: 16_000, Channels: 1}
	for range 20 {
		if listening.WriteMessage(websocket.BinaryMessage, speech.Bytes()) != nil {
			break
		}
	}
	// A caller busy elsewhere, which leaves the rest of the refusal in the router's buffer
	// while the router ends the socket.
	time.Sleep(300 * time.Millisecond)

	read, ended := s.ending(listening)

	s.Require().Len(read, 2, "the refusal and the end both arrive, then: %v", ended)
	s.Equal(fmt.Sprintf("routing: unknown target %q", target), read[0]["error"])
	s.Equal("closed", read[1]["type"])
	var closed *websocket.CloseError
	s.Require().ErrorAs(ended, &closed)
	s.Equal(websocket.ClosePolicyViolation, closed.Code)
}

func (s *StreamsSuite) TestAStartPinnedToABatchModelIsRefusedAndLeavesItsHealthAlone() {
	pinned := "batch/" + batchModel
	read, _ := s.ending(s.start("/v1/stt/stream", frame{"target": pinned}))

	s.Require().Len(read, 2)
	s.Equal("routing: "+pinned+" is a batch model for recordings, not live calls; for a live call name en-low-latency",
		read[0]["error"])
	s.Never(func() bool {
		count, err := s.store.DB().NewSelect().Model((*store.Request)(nil)).
			Where("model = ?", batchModel).Count(context.Background())
		return err != nil || count > 0
	}, time.Second, 50*time.Millisecond, "no start_failed row counts against the model")
}

func (s *StreamsSuite) TestAStartPinnedToABatchModelEndsAsAPolicyViolation() {
	read, ended := s.ending(s.start("/v1/stt/stream", frame{"target": "batch/" + batchModel}))

	s.Require().NotEmpty(read)
	var closed *websocket.CloseError
	s.Require().ErrorAs(ended, &closed)
	s.Equal(websocket.ClosePolicyViolation, closed.Code, "the caller's to change, not the router's failure")
	s.True(strings.HasPrefix(read[0]["error"].(string), closed.Text), "the refusal is the reason, cut to fit")
}

func (s *StreamsSuite) TestAVendorThatFailsToStartEndsAsAnInternalError() {
	read, ended := s.ending(s.start("/v1/stt/stream", frame{"target": "down/" + downModel}))

	s.Require().NotEmpty(read)
	s.Contains(read[0]["error"], "batch: connect failed")
	var closed *websocket.CloseError
	s.Require().ErrorAs(ended, &closed)
	s.Equal(websocket.CloseInternalServerErr, closed.Code)
}

func (s *StreamsSuite) TestAStartWithNoTargetEndsWithTheSameClose() {
	s.refusedStartEndsWithItsMessage(frame{"tts": frame{"target": "en-low-latency"}})
}

func (s *StreamsSuite) TestAStartNamingAnUnknownConfigEndsWithTheSameClose() {
	s.refusedStartEndsWithItsMessage(frame{"config_id": "no such config " + s.utils.uuid()})
}

func (s *StreamsSuite) TestAStartWithABadTagEndsWithTheSameClose() {
	s.refusedStartEndsWithItsMessage(frame{"target": "en-low-latency", "tags": frame{"Not A Key!": "x"}})
}

func (s *StreamsSuite) refusedStartEndsWithItsMessage(opening frame) {
	read, ended := s.ending(s.start("/v1/stt/stream", opening))

	s.Require().Len(read, 1)
	s.Equal("error", read[0]["type"])
	var closed *websocket.CloseError
	s.Require().ErrorAs(ended, &closed)
	s.Equal(websocket.CloseInternalServerErr, closed.Code)
	s.Equal(read[0]["error"], closed.Text)
}

func (s *StreamsSuite) TestAFailureThatIsNotUTF8IsSentAsTextThatIs() {
	reason := closeReason("provider said \xff\xfe and stopped")

	s.True(utf8.ValidString(reason))
	s.Equal("provider said \uFFFD and stopped", reason, "a run of bad bytes is one replacement")
}

func (s *StreamsSuite) TestACallerThatNeverAnswersTheCloseIsHungUpOn() {
	listening := s.start("/v1/stt/stream", frame{"target": "nobody/x"})
	// Never answers the close: the router's wait for it has to end on its own, in about
	// closeWait. The read deadline keeps a router that never lets go from hanging the suite.
	began := time.Now()
	raw := listening.UnderlyingConn()
	s.Require().NoError(raw.SetReadDeadline(began.Add(5 * time.Second)))
	buffer := make([]byte, 4096)
	for {
		if _, err := raw.Read(buffer); err != nil {
			var timeout net.Error
			s.False(errors.As(err, &timeout) && timeout.Timeout(), "the router still holds the socket: %v", err)
			s.ErrorIs(err, io.EOF)
			s.Less(time.Since(began), 1500*time.Millisecond, "the router waits for the close for about a second")
			return
		}
	}
}

func (s *StreamsSuite) TestARefusalTooLongForACloseFrameIsCutBetweenCharacters() {
	listening := s.start("/v1/stt/stream", frame{"target": "nobody/" + strings.Repeat("é", 100)})

	read, ended := s.ending(listening)

	s.Require().NotEmpty(read)
	var closed *websocket.CloseError
	s.Require().ErrorAs(ended, &closed)
	s.Equal(websocket.ClosePolicyViolation, closed.Code)
	s.LessOrEqual(len(closed.Text), 123, "RFC 6455 5.5: a control frame's payload is 125 bytes")
	s.True(utf8.ValidString(closed.Text))
	s.True(strings.HasPrefix(read[0]["error"].(string), closed.Text))
	s.Greater(len(closed.Text), 120, "only what does not fit is cut")
}

func (s *StreamsSuite) TestASocketTheRouterEndsWithoutAFailureClosesNormally() {
	// The echo model is new each session: the suite's one transcriber and one voice end
	// with the first session that closes them.
	answering := s.start("/v1/llm/stream", frame{"target": "echo/echo-model"})
	s.Require().Equal("started", s.nextFrame(answering)["type"])
	// A frame the socket cannot read ends it, the one end the router chooses itself.
	s.Require().NoError(answering.WriteMessage(websocket.TextMessage, []byte("not json")))

	read, ended := s.ending(answering)

	s.Require().Len(read, 1)
	s.Equal("closed", read[0]["type"])
	var closed *websocket.CloseError
	s.Require().ErrorAs(ended, &closed)
	s.Equal(websocket.CloseNormalClosure, closed.Code)
}

func (s *StreamsSuite) TestACallerThatHangsUpIsAnsweredWithItsOwnClose() {
	answering := s.start("/v1/llm/stream", frame{"target": "echo/echo-model"})
	s.Require().Equal("started", s.nextFrame(answering)["type"])

	s.Require().NoError(answering.WriteMessage(websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseGoingAway, "")))
	read, ended := s.ending(answering)

	s.Empty(read)
	var closed *websocket.CloseError
	s.Require().ErrorAs(ended, &closed)
	s.Equal(websocket.CloseGoingAway, closed.Code)
}

func (s *StreamsSuite) TestAsManyKeytermsAsASessionMayNameGetPastTheCap() {
	keyterms := make([]string, stt.MaxKeyterms)
	for i := range keyterms {
		keyterms[i] = fmt.Sprintf("term %d", i)
	}
	// A blank one is dropped before counting, as a session's are.
	keyterms = append(keyterms, "   ")
	// A target nothing serves gets past the cap and no further, so no session is opened on
	// the suite's one transcriber.
	listening := s.start("/v1/stt/stream", frame{
		"target": "nobody/nothing", "stt": frame{"keyterms": keyterms}})

	refused := s.nextFrame(listening)
	s.Equal("error", refused["type"])
	s.Equal(`routing: unknown target "nobody/nothing"`, refused["error"])
}

func (s *StreamsSuite) TestWhatIsSentToBeSpokenReachesTheVoice() {
	speaking := s.start("/v1/tts/stream", frame{"target": "en-low-latency"})
	s.Equal("started", s.nextFrame(speaking)["type"])

	line := "your call is important, " + s.utils.uuid()
	s.Require().NoError(speaking.WriteJSON(frame{"type": "speak", "id": "s1", "text": line}))

	s.Require().Eventually(func() bool {
		for _, said := range s.voice.spoken() {
			if said == line {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond)
}

func (s *StreamsSuite) TestTheStartedFrameNamesTheVoiceSpeaking() {
	speaking := s.start("/v1/tts/stream", frame{"target": "en-low-latency", "voice": "library-voice"})

	started := s.nextFrame(speaking)
	s.Equal("started", started["type"])
	s.Equal("library-voice", started["voice"])
}

// The name order matters: tests run alphabetically and every TTS socket in the suite shares
// one stub voice, so this one must sort after TestSpeechComesBackAsAudioThatDescribesItself,
// or its still-open socket reads the audio that test sends.
func (s *StreamsSuite) TestTheStartedFrameOfAVoiceAskedForNoneNamesNone() {
	speaking := s.start("/v1/tts/stream", frame{"target": "en-low-latency"})

	started := s.nextFrame(speaking)
	s.Equal("started", started["type"])
	s.NotContains(started, "voice")
}

func (s *StreamsSuite) TestSpeechComesBackAsAudioThatDescribesItself() {
	speaking := s.start("/v1/tts/stream", frame{"target": "en-low-latency"})
	s.Equal("started", s.nextFrame(speaking)["type"])

	s.voice.emitter.Send(tts.AudioChunk{
		SynthesisID: "s1",
		Audio:       audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24_000, Channels: 1},
	})

	_, payload := s.next(speaking)
	s.Require().Len(payload, audioHeader+6)
	s.EqualValues(24_000, binary.LittleEndian.Uint32(payload[0:4]))
	s.EqualValues(1, binary.LittleEndian.Uint16(payload[4:6]))
	s.Equal([]int16{1, 2, 3}, audio.FromBytes(payload[audioHeader:], 24_000, 1).Samples)
}

func (s *StreamsSuite) TestAReplyStreamsBackAndThenSaysItIsComplete() {
	answering := s.start("/v1/llm/stream", frame{"target": "echo/echo-model"})

	started := s.nextFrame(answering)
	s.Equal("started", started["type"])
	s.Equal(true, started["tool_history"], "a caller may send back what the model called")

	// The model answers with the instructions it was given, so what comes back is this
	// test's own words rather than whatever the last one left behind.
	said := s.utils.uuid()
	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1",
		"instructions": said,
		"messages":     []frame{{"role": "user", "content": "hello"}}}))

	s.Equal(said, s.until(answering, "delta")["text"])

	complete := s.until(answering, "complete")
	s.Equal("r1", complete["id"], "a caller with several answers in flight tells them apart")
	s.Equal(said, complete["text"], "the whole answer, for a caller that did not keep the pieces")
}

func (s *StreamsSuite) TestAToolResultWithoutTheCallItAnswersIsRefused() {
	answering := s.start("/v1/llm/stream", frame{"target": "echo/echo-model"})
	s.Equal("started", s.nextFrame(answering)["type"])

	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1",
		"messages": []frame{{"role": "tool", "content": "result"}}}))

	s.Equal("error", s.nextFrame(answering)["type"])
}

func (s *StreamsSuite) TestAHistoryOfWhatTheModelCalledIsTakenBack() {
	answering := s.start("/v1/llm/stream", frame{"target": "echo/echo-model"})
	s.Equal("started", s.nextFrame(answering)["type"])

	said := s.utils.uuid()
	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1",
		"instructions": said,
		"messages": []frame{
			{"role": "user", "content": "what is the weather"},
			{"role": "assistant", "tool_calls": []frame{
				{"id": "call-1", "name": "get_weather", "arguments": "{}"}}},
			{"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
		}}))

	// A history the socket refused would come back as an error instead of an answer.
	s.Equal(said, s.until(answering, "delta")["text"])
}

func (s *StreamsSuite) TestAModelOfTheCustomersOwnAnswersAtItsEndpointWithItsKey() {
	name := "own-" + s.utils.uuid()
	sealed, err := s.sealer.SealWithAAD("sk-own", []byte(s.customerID()))
	s.Require().NoError(err)
	s.Require().NoError(s.configs.CreateCustomModel(context.Background(), &store.CustomModel{
		CustomerID: s.customerID(), Name: name, BaseURL: "https://8.8.8.8/v1", Model: "acme/tuned-27b",
		APIKeySealed: sealed, KEKVersion: s.sealer.CurrentVersion(),
	}))

	answering := s.start("/v1/llm/stream", frame{"target": "custom/" + name})
	started := s.nextFrame(answering)
	s.Equal("started", started["type"])
	s.Equal("custom", started["provider"])

	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1",
		"messages": []frame{{"role": "user", "content": "hello"}}}))

	s.Equal("https://8.8.8.8/v1 acme/tuned-27b sk-own", s.until(answering, "complete")["text"])
}

func (s *StreamsSuite) TestAnotherCustomersModelIsNotOneThisCustomerCanName() {
	name := "theirs-" + s.utils.uuid()
	s.Require().NoError(s.configs.CreateCustomModel(context.Background(), &store.CustomModel{
		CustomerID: s.utils.uuid(), Name: name, BaseURL: "https://8.8.8.8/v1", Model: "acme/tuned-27b",
	}))

	answering := s.start("/v1/llm/stream", frame{"target": "custom/" + name})

	refused := s.nextFrame(answering)
	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "not one of this customer's models")
}

func (s *StreamsSuite) TestAPictureAModelCannotSeeIsRefusedRatherThanSentAnyway() {
	answering := s.start("/v1/llm/stream", frame{"target": "llm-flow"})
	s.Equal("started", s.nextFrame(answering)["type"])

	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1",
		"messages": []frame{{"role": "user", "content": []frame{
			{"type": "text", "text": "what is this"},
			{"type": "image_url", "image_url": frame{"url": roses}},
		}}}}))

	refused := s.nextFrame(answering)
	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "does not accept image")
}

func (s *StreamsSuite) TestAPictureReachesTheModelThatCanSeeIt() {
	answering := s.start("/v1/llm/stream", frame{"target": "vlm"})
	s.Equal("started", s.nextFrame(answering)["type"])

	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1",
		"messages": []frame{{"role": "user", "content": []frame{
			{"type": "text", "text": "what is this"},
			{"type": "image_url", "image_url": frame{"url": roses, "detail": "low"}},
		}}}}))

	s.Equal("Two roses.", s.until(answering, "complete")["text"])
}

func (s *StreamsSuite) TestAPictureSomewhereElseIsFetchedRatherThanRefused() {
	// A caller that holds a URL rather than the bytes is the ordinary case from a browser.
	answering := s.start("/v1/llm/stream", frame{"target": "vlm"})
	s.Equal("started", s.nextFrame(answering)["type"])

	s.Require().NoError(answering.WriteJSON(frame{"type": "respond", "id": "r1",
		"messages": []frame{{"role": "user", "content": []frame{
			{"type": "image_url", "image_url": frame{"url": "https://example.test/cat.jpg"}},
		}}}}))

	s.Equal("Two roses.", s.until(answering, "complete")["text"])
}

func (s *StreamsSuite) TestAStartedConversationSaysWhatIsAnsweringAndAtWhatRate() {
	conversation := s.start("/v1/sts/stream", frame{
		"target": "stub/stub-sts", "sts": frame{"instructions": "Be brief."}})

	started := s.nextConversationFrame(conversation)
	s.Equal("started", started["type"])
	s.Equal("stub", started["provider"])
	s.Equal("stub-sts", started["model"])
	s.EqualValues(24_000, started["sample_rate"],
		"the rate the model speaks at, so playback can be readied")
}

func (s *StreamsSuite) TestTheCallersAudioReachesTheModelAsItWasSent() {
	conversation, model := s.converse(frame{"target": "stub/stub-sts"})

	pcm := audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1}
	s.Require().NoError(conversation.WriteMessage(websocket.BinaryMessage, pcm.Bytes()))

	select {
	case heard := <-model.heard:
		s.Equal(16_000, heard.SampleRate)
		s.Len(heard.Samples, 160)
	case <-time.After(settleFor):
		s.Fail("the model never heard the audio")
	}
}

func (s *StreamsSuite) TestTheModelsVoiceComesBackUnderAHeaderNamingTheReply() {
	conversation, model := s.converse(frame{"target": "stub/stub-sts"})

	model.emitter.Send(sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()})
	model.emitter.Send(sts.AudioChunk{
		ResponseID: "r1", Generation: 1, Index: 0,
		Audio: audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24_000, Channels: 1},
	})

	begun := s.nextConversationFrame(conversation)
	s.Equal("response_started", begun["type"])
	s.Equal("r1", begun["id"])
	s.EqualValues(1, begun["generation"])

	_, payload := s.next(conversation)
	s.Require().Len(payload, stsAudioHeader+6)
	s.EqualValues(24_000, binary.LittleEndian.Uint32(payload[0:4]))
	s.EqualValues(1, binary.LittleEndian.Uint16(payload[4:6]))
	s.EqualValues(stsAudioVersion, binary.LittleEndian.Uint16(payload[6:8]))
	s.EqualValues(1, binary.LittleEndian.Uint32(payload[8:12]),
		"the generation is what lets a client drop a cut-off reply's tail")
	s.EqualValues(0, binary.LittleEndian.Uint32(payload[12:16]))
	s.Equal([]int16{1, 2, 3}, audio.FromBytes(payload[stsAudioHeader:], 24_000, 1).Samples)
}

func (s *StreamsSuite) TestATypedTurnReachesTheModel() {
	conversation, model := s.converse(frame{"target": "stub/stub-sts"})

	s.Require().NoError(conversation.WriteJSON(frame{"type": "text", "text": "hello"}))

	s.Equal("hello", s.told(model.typed))
}

func (s *StreamsSuite) TestWhatATooRanReachesTheModelAsItsResult() {
	conversation, model := s.converse(frame{"target": "stub/stub-sts"})

	s.Require().NoError(conversation.WriteJSON(frame{
		"type": "tool_result", "tool_call_id": "c1", "output": "sunny"}))

	s.Equal("c1=sunny", s.told(model.answers))
}

func (s *StreamsSuite) TestAnInterruptionSaysHowMuchOfTheReplyWasHeard() {
	conversation, model := s.converse(frame{"target": "stub/stub-sts"})

	s.Require().NoError(conversation.WriteJSON(frame{"type": "interrupt", "played_ms": 120}))

	s.Equal(120, s.told(model.interrupts))
}

func (s *StreamsSuite) TestAudioFromAnInterruptedReplyDoesNotFollowItsCompletion() {
	conversation, model := s.converse(frame{"target": "stub/stub-sts"})

	model.emitter.Send(sts.ResponseComplete{
		ResponseID: "r1", Generation: 1, Interrupted: true, AudioDurationMs: 0.125})
	model.emitter.Send(sts.AudioChunk{ResponseID: "r1", Generation: 1, Index: 1,
		Audio: audio.PcmData{Samples: []int16{9}, SampleRate: 24_000, Channels: 1}})
	model.emitter.Send(sts.ToolCall{
		ResponseID: "r1", CallID: "c2", Name: "get_weather", Arguments: "{}"})

	complete := s.nextConversationFrame(conversation)
	s.Equal("response_complete", complete["type"])
	s.Equal(true, complete["interrupted"])

	decoded, payload := s.next(conversation)
	s.Nil(payload, "audio from the interrupted reply must not follow its completion")
	s.Equal("tool_call", decoded["type"])
	s.Equal("get_weather", decoded["name"])
}

func (s *StreamsSuite) TestAPictureAConversationCannotSeeIsRefusedRatherThanDropped() {
	conversation, _ := s.converse(frame{"target": "stub/stub-sts"})

	s.Require().NoError(conversation.WriteJSON(frame{"type": "frame", "image_url": roses}))

	refused := s.nextConversationFrame(conversation)
	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "does not accept images")
}

func (s *StreamsSuite) TestHangingUpClosesTheModelsSession() {
	conversation, model := s.converse(frame{"target": "stub/stub-sts"})

	s.Require().NoError(conversation.Close())

	select {
	case <-model.closed:
	case <-time.After(settleFor):
		s.Fail("hanging up should close the model's session")
	}
}

func (s *StreamsSuite) TestATermNothingCanServeIsRefusedBeforeAnythingIsBilled() {
	conversation := s.start("/v1/sts/stream", frame{
		"target": "sts-fast", "sts": frame{"turn_detection": "semantic"}})

	refused := s.nextConversationFrame(conversation)
	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "semantic_turns")
	s.Equal("closed", s.nextConversationFrame(conversation)["type"])
}

func (s *StreamsSuite) TestToolsAModelWillNeverCallAreRefused() {
	// Handing a model tools it cannot call is the thing terms exist to prevent.
	conversation := s.start("/v1/sts/stream", frame{
		"target": "sts-fast",
		"tools":  []frame{{"name": "get_weather", "description": "Weather."}}})

	refused := s.nextConversationFrame(conversation)
	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "tools")
}

// ending reads a socket to its end: every JSON frame left on it, and how it ended.
func (s *StreamsSuite) ending(connection *websocket.Conn) ([]frame, error) {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	var read []frame
	for {
		kind, payload, err := connection.ReadMessage()
		if err != nil {
			return read, err
		}
		var decoded frame
		if kind == websocket.TextMessage && json.Unmarshal(payload, &decoded) == nil {
			read = append(read, decoded)
		}
	}
}

// refused opens a socket on the model that refuses the account, speaks into it and returns
// it once the refusal has come back.
func (s *StreamsSuite) refused() *websocket.Conn {
	listening := s.start("/v1/stt/stream", frame{"target": "refusing/" + refusingModel})
	s.Require().Equal("started", s.nextFrame(listening)["type"])
	silence := audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1}
	s.Require().NoError(listening.WriteMessage(websocket.BinaryMessage, silence.Bytes()))

	refused := s.nextFrame(listening)
	s.Require().Equal("error", refused["type"])
	s.Equal(termsRefusal, refused["error"])
	s.Equal(true, refused["fatal"])
	return listening
}

// start opens a modality socket and sends its start frame.
func (s *StreamsSuite) start(path string, opening frame) *websocket.Conn {
	connection := s.serverClient.opens(path)
	opening["type"] = "start"
	s.Require().NoError(connection.WriteJSON(opening))
	return connection
}

// converse opens a speech-to-speech socket and returns the model it landed on, so a test
// can drive the conversation from both ends. The models the registry has already made are
// forgotten first, since the one this socket starts is the next one.
func (s *StreamsSuite) converse(opening frame) (*websocket.Conn, *stubConversation) {
	for {
		select {
		case <-conversing:
			continue
		default:
		}
		break
	}

	connection := s.start("/v1/sts/stream", opening)
	s.Require().Equal("started", s.nextConversationFrame(connection)["type"])
	select {
	case model := <-conversing:
		return connection, model
	case <-time.After(settleFor):
		s.FailNow("no model was started")
		return nil, nil
	}
}

// told is what the model was sent, which crosses a socket and a channel to reach it.
func (s *StreamsSuite) told[T any](sent <-chan T) T {
	select {
	case value := <-sent:
		return value
	case <-time.After(settleFor):
		s.FailNow("the model was never sent anything")
		var nothing T
		return nothing
	}
}

// next reads the next message: a JSON frame decoded, or a binary payload.
func (s *StreamsSuite) next(connection *websocket.Conn) (frame, []byte) {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	kind, payload, err := connection.ReadMessage()
	s.Require().NoError(err)
	if kind == websocket.BinaryMessage {
		return nil, payload
	}
	var decoded frame
	s.Require().NoError(json.Unmarshal(payload, &decoded))
	return decoded, nil
}

// nextFrame reads the next JSON frame.
func (s *StreamsSuite) nextFrame(connection *websocket.Conn) frame {
	decoded, _ := s.next(connection)
	s.Require().NotNil(decoded, "a binary frame arrived where a JSON one was expected")
	return decoded
}

// nextConversationFrame reads the next JSON frame, skipping the audio between them.
func (s *StreamsSuite) nextConversationFrame(connection *websocket.Conn) frame {
	for {
		decoded, payload := s.next(connection)
		if payload == nil {
			return decoded
		}
	}
}

// until reads frames until one of a kind arrives, since a reply comes in pieces.
func (s *StreamsSuite) until(connection *websocket.Conn, kind string) frame {
	for range 32 {
		decoded := s.nextFrame(connection)
		if decoded["type"] == kind {
			return decoded
		}
	}
	s.FailNow("no " + kind + " frame arrived")
	return nil
}
