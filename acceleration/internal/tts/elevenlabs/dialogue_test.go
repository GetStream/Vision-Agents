package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

type DialogueSuite struct {
	suite.Suite
}

func TestDialogueSuite(t *testing.T) {
	suite.Run(t, new(DialogueSuite))
}

// newDialogue returns a provider that is wired up but never connected.
func (s *DialogueSuite) newDialogue(options Options) *Dialogue {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := NewDialogue(options)
	s.Require().NoError(err)
	return provider
}

// connect returns a started provider and the server side of its connection.
func (s *DialogueSuite) connect(fake *fakeElevenLabs, options Options) (*Dialogue, *websocket.Conn) {
	options.BaseURL = fake.baseURL()
	// The fake never answers close_socket, and no test is about that wait.
	if options.CloseTimeout == 0 {
		options.CloseTimeout = 50 * time.Millisecond
	}
	provider := s.newDialogue(options)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))

	return provider, s.accepts(fake)
}

// accepts returns the server side of the next connection, taking the url the fake recorded
// for it. Left there, it would block the handler for the connection barge-in opens.
func (s *DialogueSuite) accepts(fake *fakeElevenLabs) *websocket.Conn {
	conn := fake.accept()
	s.Require().NotNil(conn, "the provider should have connected")
	select {
	case <-fake.url:
	default:
	}
	return conn
}

// frames reads the messages the provider sent upstream.
func (s *DialogueSuite) frames(conn *websocket.Conn, want int) []dialogueClientMessage {
	var messages []dialogueClientMessage
	for len(messages) < want {
		s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
		_, raw, err := conn.ReadMessage()
		s.Require().NoError(err)

		var message dialogueClientMessage
		s.Require().NoError(json.Unmarshal(raw, &message))
		messages = append(messages, message)
	}
	return messages
}

// collect reads events until the predicate is satisfied or the wait runs out.
func (s *DialogueSuite) collect(provider *Dialogue, until func(tts.Event) bool) []tts.Event {
	var events []tts.Event
	deadline := time.After(5 * time.Second)

	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
			if until(event) {
				return events
			}
		case <-deadline:
			s.FailNow("timed out waiting for events")
			return events
		}
	}
}

// says sends a frame of PCM audio the way the dialogue server does, which carries no
// utterance id.
func (s *DialogueSuite) says(conn *websocket.Conn, samples []int16) {
	pcm := audio.PcmData{Samples: samples, SampleRate: DefaultSampleRate, Channels: 1}
	payload, err := json.Marshal(map[string]any{
		"audio": base64.StdEncoding.EncodeToString(pcm.Bytes()),
	})
	s.Require().NoError(err)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, payload))
}

// endsTurn tells the provider the audio for the turn in flight is complete.
func (s *DialogueSuite) endsTurn(conn *websocket.Conn) {
	payload, err := json.Marshal(map[string]any{"is_final_audio_for_turn": true})
	s.Require().NoError(err)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, payload))
}

func (s *DialogueSuite) TestOnlyADialogueModelIsServedHere() {
	// The two endpoints speak different protocols, so opening one against the other's
	// model would fail on the connection rather than here.
	_, err := NewDialogue(Options{APIKey: "k", Model: DefaultModel})
	s.ErrorContains(err, "not a dialogue model")

	_, err = New(Options{APIKey: "k", Model: DefaultDialogueModel})
	s.ErrorContains(err, "is a dialogue model")
}

func (s *DialogueSuite) TestNewDefaultsToTheConversationalModel() {
	provider := s.newDialogue(Options{})

	s.Equal(DefaultDialogueModel, provider.Model())
	s.Equal(ProviderName, provider.Provider())
	s.True(provider.Streaming())
	s.True(provider.Performs(), "acting audio tags is what this endpoint is for")
	s.Contains(provider.Prompt(), "[laughs]")
}

func (s *DialogueSuite) TestTheEndpointCarriesTheModelAndFormat() {
	url := s.newDialogue(Options{SampleRate: 24_000}).url()

	s.Contains(url, "/v1/text-to-dialogue/stream-input")
	s.Contains(url, "model_id="+DefaultDialogueModel)
	s.Contains(url, "output_format=pcm_24000")
}

func (s *DialogueSuite) TestTheSessionRegistersItsVoiceBeforeSayingAnything() {
	// The server rejects a line for a voice it was never told to load.
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	opening := s.frames(conn, 1)[0]

	s.Equal([]string{"v1"}, opening.Voices)
	s.Require().NotNil(opening.VoiceSettings)
	s.Empty(opening.Inputs, "the first frame is registration, not a line")
}

func (s *DialogueSuite) TestOnlyASecondUtteranceEndsTheTurnBeforeIt() {
	// Without a turn boundary two replies run together as one breath; asking for one on
	// the first would end a turn that was never started.
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "a", Text: "Hello"}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "a", Text: "there.", Final: true}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "b", Text: "Bye.", Final: true}))

	sent := s.frames(conn, 6)

	s.False(sent[1].Inputs[0].NewTurn, "the first line of the call opens the first turn")
	s.False(sent[2].Inputs[0].NewTurn, "a delta continues the utterance it belongs to")
	s.True(sent[3].Flush, "the end of an utterance asks for the audio it is owed")
	s.True(sent[4].Inputs[0].NewTurn, "a new utterance is a new turn")
}

func (s *DialogueSuite) TestAudioIsAttributedToTheUtteranceInFlight() {
	// The protocol has no contexts, so nothing on an audio frame says what it belongs to.
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "a", Text: "Hello.", Final: true}))
	s.says(conn, make([]int16, 160))
	s.endsTurn(conn)

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var chunks int
	for _, event := range events {
		if chunk, ok := event.(tts.AudioChunk); ok {
			chunks++
			s.Equal("a", chunk.SynthesisID)
		}
	}
	s.Equal(1, chunks)

	completed := events[len(events)-1].(tts.SynthesisComplete)
	s.Equal("a", completed.SynthesisID)
	s.False(completed.Interrupted)
	s.Positive(completed.AudioDurationMs)
}

func (s *DialogueSuite) TestTurnsAreSettledInTheOrderTheyWereStarted() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "a", Text: "Hello.", Final: true}))
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "b", Text: "Goodbye.", Final: true}))

	s.says(conn, make([]int16, 160))
	s.endsTurn(conn)
	s.says(conn, make([]int16, 160))
	s.endsTurn(conn)

	var settled []string
	s.collect(provider, func(event tts.Event) bool {
		if done, ok := event.(tts.SynthesisComplete); ok {
			settled = append(settled, done.SynthesisID)
		}
		return len(settled) == 2
	})

	s.Equal([]string{"a", "b"}, settled)
}

func (s *DialogueSuite) TestBargeInAbandonsWhatWasBeingSaidAndOpensTheVoiceAgain() {
	// There is no cancel frame on this protocol, so the only way to stop the server
	// generating is to stop being connected to it. The call is still going, so a voice has
	// to be there for the next thing the agent says.
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "a", Text: "Hello.", Final: true}))
	s.frames(conn, 3)

	s.Require().NoError(provider.Interrupt())

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	completed := events[len(events)-1].(tts.SynthesisComplete)
	s.Equal("a", completed.SynthesisID)
	s.True(completed.Interrupted)

	reopened := s.accepts(fake)
	s.Equal([]string{"v1"}, s.frames(reopened, 1)[0].Voices)
}

func (s *DialogueSuite) TestTheFirstUtteranceAfterBargeInOpensTheTurn() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	s.Require().NoError(provider.Synthesize(tts.Request{ID: "a", Text: "Hello.", Final: true}))
	s.frames(conn, 3)
	s.Require().NoError(provider.Interrupt())

	reopened := s.accepts(fake)
	s.Require().NoError(provider.Synthesize(tts.Request{ID: "b", Text: "Sorry.", Final: true}))

	sent := s.frames(reopened, 2)

	s.False(sent[1].Inputs[0].NewTurn,
		"a fresh connection has no turn to end, and the server would reject one")
}

func (s *DialogueSuite) TestSynthesizeRejectsAVoiceTheConnectionCannotSpeak() {
	// The session registered one voice, and the server only knows that one.
	provider := s.newDialogue(Options{VoiceID: "bound-voice"})

	err := provider.Synthesize(tts.Request{Text: "hello", Voice: "other", Final: true})

	s.ErrorContains(err, "bound to voice bound-voice")
}

func (s *DialogueSuite) TestSynthesizeFailsBeforeStart() {
	err := s.newDialogue(Options{}).Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "not started")
}

func (s *DialogueSuite) TestAServerErrorIsReported() {
	fake := newFakeElevenLabs()
	defer fake.close()
	provider, conn := s.connect(fake, Options{VoiceID: "v1"})
	defer provider.Close()

	payload, err := json.Marshal(map[string]any{
		"message": "voice not found", "error": "voice_not_found", "code": 1008,
	})
	s.Require().NoError(err)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, payload))

	events := s.collect(provider, func(event tts.Event) bool {
		_, failed := event.(tts.Error)
		return failed
	})

	failure := events[len(events)-1].(tts.Error)
	s.ErrorContains(failure.Err, "voice not found")
	s.True(failure.Fatal, "the server named the close code it is about to send")
}
