package openairealtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

// RealtimeSocketSuite exercises the provider against a server that is answering, which is
// where the handshake, the audio encoding and the frames sent upstream live.
type RealtimeSocketSuite struct {
	suite.Suite
}

func TestRealtimeSocketSuite(t *testing.T) {
	suite.Run(t, new(RealtimeSocketSuite))
}

// fakeRealtime is a realtime server that accepts one session: it announces the session,
// takes the configuration, acknowledges it, and hands the connection to the test.
type fakeRealtime struct {
	server *httptest.Server
	conns  chan *websocket.Conn
	setups chan clientEvent
	url    chan string
	header chan http.Header
	// reject makes the server refuse the configuration instead of taking it.
	reject bool
	done   chan struct{}
}

func newFakeRealtime(reject bool) *fakeRealtime {
	fake := &fakeRealtime{
		conns:  make(chan *websocket.Conn, 1),
		setups: make(chan clientEvent, 1),
		url:    make(chan string, 1),
		header: make(chan http.Header, 1),
		reject: reject,
		done:   make(chan struct{}),
	}

	upgrader := websocket.Upgrader{}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fake.url <- r.URL.String()
		fake.header <- r.Header.Clone()
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"session.created","session":{}}`)); err != nil {
			return
		}

		_, raw, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var frame clientEvent
		if err := json.Unmarshal(raw, &frame); err != nil || frame.Type != eventSessionUpdate {
			return
		}
		fake.setups <- frame

		reply := `{"type":"session.updated","session":{}}`
		if fake.reject {
			reply = `{"type":"error","error":{"type":"invalid_request_error","code":"unknown_parameter","message":"no such voice"}}`
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(reply)); err != nil {
			return
		}
		fake.conns <- conn
		<-fake.done
	}))
	return fake
}

func (f *fakeRealtime) endpoint() string {
	return "ws://" + strings.TrimPrefix(f.server.URL, "http://")
}

func (f *fakeRealtime) close() {
	close(f.done)
	f.server.Close()
}

// connect returns a started provider and the server side of its connection.
func (s *RealtimeSocketSuite) connect(fake *fakeRealtime, settings Options) (*STS, *websocket.Conn) {
	settings.URL = fake.endpoint()
	if settings.APIKey == "" {
		settings.APIKey = "test-key"
	}
	provider, err := New(settings)
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))

	select {
	case conn := <-fake.conns:
		return provider, conn
	case <-time.After(5 * time.Second):
		s.FailNow("the provider never connected")
		return nil, nil
	}
}

// nextFrame reads the next message the provider sent upstream.
func (s *RealtimeSocketSuite) nextFrame(conn *websocket.Conn) clientEvent {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)

	var frame clientEvent
	s.Require().NoError(json.Unmarshal(raw, &frame))
	return frame
}

// serve sends one event from the server.
func (s *RealtimeSocketSuite) serve(conn *websocket.Conn, event serverEvent) {
	raw, err := json.Marshal(event)
	s.Require().NoError(err)
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, raw))
}

// speak sends ten milliseconds of caller audio.
func (s *RealtimeSocketSuite) speak(provider *STS) {
	pcm := sts.PcmData{Samples: make([]int16, 160), SampleRate: sts.InputSampleRate, Channels: 1}
	s.Require().NoError(provider.ProcessAudio(pcm, sts.Participant{ID: "alice"}))
}

func (s *RealtimeSocketSuite) TestTheSessionIsConfiguredBeforeItIsReportedReady() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, _ := s.connect(fake, Options{Vendor: OpenAI, Voice: "marin", Instructions: "Be brief."})
	defer func() { _ = provider.Close() }()

	s.Contains(<-fake.url, "model=gpt-realtime-2.1")
	s.Equal("Bearer test-key", (<-fake.header).Get("Authorization"))

	opened := <-fake.setups
	s.Require().NotNil(opened.Session)
	s.Equal("Be brief.", opened.Session.Instructions)
	s.Equal("marin", opened.Session.Audio.Output.Voice)

	select {
	case event := <-provider.Events():
		_, ok := event.(sts.Connected)
		s.True(ok, "the first event is the session being ready, and only after the server took the configuration")
	case <-time.After(time.Second):
		s.FailNow("no connected event")
	}
}

func (s *RealtimeSocketSuite) TestASessionTheServerRefusesFailsToStart() {
	fake := newFakeRealtime(true)
	defer fake.close()

	provider, err := New(Options{Vendor: OpenAI, APIKey: "k", URL: fake.endpoint()})
	s.Require().NoError(err)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = provider.Start(ctx)
	s.ErrorContains(err, "session rejected")
	s.ErrorContains(err, "no such voice", "the server's own reason is what the caller hears")
}

func (s *RealtimeSocketSuite) TestAudioIsResampledToTheRateTheVendorWants() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.speak(provider)

	frame := s.nextFrame(conn)
	s.Equal(eventAudioAppend, frame.Type)
	decoded, err := base64.StdEncoding.DecodeString(frame.Audio)
	s.Require().NoError(err)
	s.Len(decoded, 240*2, "ten milliseconds at 24 kHz, up from 16")
}

func (s *RealtimeSocketSuite) TestAudioForQwenStaysAtSixteenKilohertz() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: Qwen})
	defer func() { _ = provider.Close() }()

	s.speak(provider)

	frame := s.nextFrame(conn)
	decoded, err := base64.StdEncoding.DecodeString(frame.Audio)
	s.Require().NoError(err)
	s.Len(decoded, 160*2)
}

func (s *RealtimeSocketSuite) TestInterruptCancelsAndTruncatesWhatWasHeard() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.serve(conn, serverEvent{Type: eventOutputItemAdded, Item: &serverItem{ID: "item_9", Type: "message"}})
	s.serve(conn, serverEvent{
		Type: eventAudioDelta, ResponseID: "r1", ItemID: "item_9",
		Delta: base64.StdEncoding.EncodeToString(make([]byte, 2400*2)),
	})
	// Wait for the chunk to have been counted before interrupting.
	deadline := time.After(5 * time.Second)
	for counted := false; !counted; {
		select {
		case event := <-provider.Events():
			_, counted = event.(sts.AudioChunk)
		case <-deadline:
			s.FailNow("the audio never arrived")
		}
	}

	s.Require().NoError(provider.Interrupt(0))

	cancel := s.nextFrame(conn)
	s.Equal(eventResponseCancel, cancel.Type)
	truncate := s.nextFrame(conn)
	s.Equal(eventItemTruncate, truncate.Type)
	s.Equal("item_9", truncate.ItemID)
	s.Equal(0, *truncate.ContentIndex)
	s.Equal(100, *truncate.AudioEndMs, "nobody said how much was heard, so what was sent stands in")

	s.Require().NoError(provider.Interrupt(40))
	s.nextFrame(conn)
	truncate = s.nextFrame(conn)
	s.Equal(40, *truncate.AudioEndMs, "the listener's own count wins when there is one")
}

func (s *RealtimeSocketSuite) TestAnAnswerCarriesTheResultAndAsksForAReply() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.Answer("call_1", `{"forecast":"sunny"}`, nil))

	result := s.nextFrame(conn)
	s.Equal(eventItemCreate, result.Type)
	s.Equal("function_call_output", result.Item.Type)
	s.Equal("call_1", result.Item.CallID)
	s.Equal(`{"forecast":"sunny"}`, result.Item.Output)
	carry := s.nextFrame(conn)
	s.Equal(eventResponseCreate, carry.Type, "this protocol does not carry on by itself")
	s.Equal(clientResponse, carry.EventID)

	s.Require().NoError(provider.Answer("call_2", "", context.DeadlineExceeded))
	failed := s.nextFrame(conn)
	s.Equal("Error: context deadline exceeded", failed.Item.Output)
}

func (s *RealtimeSocketSuite) TestATypedTurnAndAPromptAskForReplies() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.SendText("hello", sts.Participant{ID: "alice"}))
	turn := s.nextFrame(conn)
	s.Equal(eventItemCreate, turn.Type)
	s.Equal("user", turn.Item.Role)
	s.Equal("input_text", turn.Item.Content[0].Type)
	s.Equal("hello", turn.Item.Content[0].Text)
	typed := s.nextFrame(conn)
	s.Equal(eventResponseCreate, typed.Type)
	s.Equal(clientResponse, typed.EventID)

	s.Require().NoError(provider.Prompt("Greet the caller."))
	prompt := s.nextFrame(conn)
	s.Equal(eventResponseCreate, prompt.Type)
	s.Equal("Greet the caller.", prompt.Response.Instructions, "a prompt guides the reply rather than adding a turn")
}

func (s *RealtimeSocketSuite) TestQwenAudioIsPromptedWithASystemMessageAndOnlyCancelled() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: Qwen, Model: "qwen-audio-3.1-realtime-plus"})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.Prompt("Greet the caller."))
	guidance := s.nextFrame(conn)
	s.Equal(eventItemCreate, guidance.Type)
	s.Equal("system", guidance.Item.Role)
	s.Equal("Greet the caller.", guidance.Item.Content[0].Text)
	reply := s.nextFrame(conn)
	s.Equal(eventResponseCreate, reply.Type)
	s.Nil(reply.Response, "Qwen-Audio's response.create takes no instructions")
	s.Equal(clientResponse, reply.EventID)

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.serve(conn, serverEvent{
		Type: eventAudioDeltaOld, ResponseID: "r1", ItemID: "item_9",
		Delta: base64.StdEncoding.EncodeToString(make([]byte, 2400*2)),
	})
	deadline := time.After(5 * time.Second)
	for counted := false; !counted; {
		select {
		case event := <-provider.Events():
			_, counted = event.(sts.AudioChunk)
		case <-deadline:
			s.FailNow("the audio never arrived")
		}
	}

	s.Require().NoError(provider.Interrupt(40))
	s.Equal(eventResponseCancel, s.nextFrame(conn).Type)
	s.speak(provider)
	s.Equal(eventAudioAppend, s.nextFrame(conn).Type, "no truncate follows the cancel")
}

func (s *RealtimeSocketSuite) TestInstructionsChangeThroughASessionUpdate() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.SetInstructions("Now be verbose."))

	update := s.nextFrame(conn)
	s.Equal(eventSessionUpdate, update.Type)
	s.Equal("realtime", update.Session.Type, "the GA session object names its type on every update")
	s.Equal("Now be verbose.", update.Session.Instructions)
}

// refused is the server refusing a reply while another is open, as gpt-realtime-2.1 sent it
// on 2026-10-20. A refusal of a client request names it; one of the server's own replies
// names nothing.
func (s *RealtimeSocketSuite) refused(conn *websocket.Conn, eventID string) {
	named := "null"
	if eventID != "" {
		named = `"` + eventID + `"`
	}
	raw := `{"type":"error","event_id":"event_1","error":{"type":"invalid_request_error",` +
		`"code":"conversation_already_has_active_response","message":"Conversation already has an active ` +
		`response in progress: r1. Wait until the response is finished before creating a new one.",` +
		`"param":null,"event_id":` + named + `}}`
	s.Require().NoError(conn.WriteMessage(websocket.TextMessage, []byte(raw)))
}

// until reads events until one passes the check, and returns every event read.
func (s *RealtimeSocketSuite) until(provider *STS, check func(sts.Event) bool) []sts.Event {
	var seen []sts.Event
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event := <-provider.Events():
			seen = append(seen, event)
			if check(event) {
				return seen
			}
		case <-deadline:
			s.FailNow("the event never arrived", "seen: %v", seen)
			return nil
		}
	}
}

// settled waits for the server's frames so far to have been read, by sending a transcript
// and waiting for it.
func (s *RealtimeSocketSuite) settled(provider *STS, conn *websocket.Conn) []sts.Event {
	s.serve(conn, serverEvent{Type: eventInputCompleted, Transcript: "marker"})
	return s.until(provider, func(event sts.Event) bool {
		heard, ok := event.(sts.InputTranscript)
		return ok && heard.Text == "marker"
	})
}

// nothingAsked proves no frame went upstream: caller audio sent now is the next frame.
func (s *RealtimeSocketSuite) nothingAsked(provider *STS, conn *websocket.Conn) {
	s.speak(provider)
	s.Equal(eventAudioAppend, s.nextFrame(conn).Type, "nothing was asked for")
}

func errorsIn(events []sts.Event) []sts.Error {
	var failures []sts.Error
	for _, event := range events {
		if failure, ok := event.(sts.Error); ok {
			failures = append(failures, failure)
		}
	}
	return failures
}

func (s *RealtimeSocketSuite) TestACallerTurnRefusedDuringAReplyIsAnsweredAfterIt() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.serve(conn, serverEvent{Type: eventSpeechStopped})
	s.refused(conn, "")
	events := s.settled(provider, conn)
	s.Empty(errorsIn(events), "a turn that will be answered is not a failure")
	s.nothingAsked(provider, conn)

	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})
	asked := s.nextFrame(conn)
	s.Equal(eventResponseCreate, asked.Type, "the caller's turn is asked for once the reply is done")
	s.Nil(asked.Response, "a plain reply answers everything the caller said")
	s.Equal(clientResponse, asked.EventID, "a refusal of this one is reported, not owed again")

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r2"}})
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r2", Status: "completed"}})
	s.settled(provider, conn)
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestARefusalThatArrivesAfterTheReplyIsAnsweredAtOnce() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})
	s.refused(conn, "")

	s.Equal(eventResponseCreate, s.nextFrame(conn).Type)
	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r2"}})
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r2", Status: "completed"}})
	s.settled(provider, conn)
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestALateRefusalWhileTheCallerSpeaksAgainAsksForNothing() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})
	s.serve(conn, serverEvent{Type: eventSpeechStarted})
	s.refused(conn, "")
	events := s.settled(provider, conn)

	s.Empty(errorsIn(events), "the server answers the caller's next stop itself")
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestRefusalsDuringOneReplyAreAnsweredOnce() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.refused(conn, "")
	s.refused(conn, "")
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})

	s.Equal(eventResponseCreate, s.nextFrame(conn).Type)
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestACallerWhoSpeaksAgainIsNotAnsweredForTheEarlierTurn() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.refused(conn, "")
	s.serve(conn, serverEvent{Type: eventSpeechStarted})
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})
	s.settled(provider, conn)

	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestABargeInDropsTheOwedReply() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.refused(conn, "")
	s.settled(provider, conn)

	s.Require().NoError(provider.Interrupt(0))
	s.Equal(eventResponseCancel, s.nextFrame(conn).Type)
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "cancelled"}})
	s.settled(provider, conn)

	s.nothingAsked(provider, conn)
}

// refusedOurs serves the refusal of the reply request of ours just sent, and returns it.
func (s *RealtimeSocketSuite) refusedOurs(conn *websocket.Conn) clientEvent {
	s.Require().NoError(conn.SetReadDeadline(time.Now().Add(5 * time.Second)))
	_, raw, err := conn.ReadMessage()
	s.Require().NoError(err)
	s.Contains(string(raw), `"event_id":"client_response"`, "the request is named so a refusal of it is told apart")
	var request clientEvent
	s.Require().NoError(json.Unmarshal(raw, &request))
	s.Require().Equal(eventResponseCreate, request.Type)
	s.refused(conn, request.EventID)
	return request
}

func (s *RealtimeSocketSuite) TestARefusedPromptIsAskedForAfterTheReplyWithItsInstructions() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.Require().NoError(provider.Prompt("Say: one moment."))
	s.refusedOurs(conn)
	events := s.settled(provider, conn)
	s.Empty(errorsIn(events), "a prompt that will be answered is not a failure")
	s.nothingAsked(provider, conn)

	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})
	asked := s.nextFrame(conn)
	s.Equal(eventResponseCreate, asked.Type, "the prompt is asked for once the reply is done")
	s.Require().NotNil(asked.Response)
	s.Equal("Say: one moment.", asked.Response.Instructions, "the prompt keeps what it asked for")
	s.Equal(clientResponse, asked.EventID)

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r2"}})
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r2", Status: "completed"}})
	s.settled(provider, conn)
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestARefusedToolResultIsAskedForAfterTheReply() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.Require().NoError(provider.Answer("call_1", `{"forecast":"sunny"}`, nil))
	s.Equal(eventItemCreate, s.nextFrame(conn).Type)
	s.refusedOurs(conn)
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})

	asked := s.nextFrame(conn)
	s.Equal(eventResponseCreate, asked.Type, "the model carries on from the result once the reply is done")
	s.Nil(asked.Response)
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestOnlyTheLastRefusedRequestOfOursIsAskedFor() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.Require().NoError(provider.Prompt("Say: first."))
	s.refusedOurs(conn)
	s.settled(provider, conn)
	s.Require().NoError(provider.Prompt("Say: second."))
	s.refusedOurs(conn)
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})

	asked := s.nextFrame(conn)
	s.Require().NotNil(asked.Response)
	s.Equal("Say: second.", asked.Response.Instructions, "the last request replaces the earlier one")
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestACallerTurnRefusedAfterARefusedPromptKeepsThePrompt() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.Require().NoError(provider.Prompt("Say: one moment."))
	s.refusedOurs(conn)
	s.refused(conn, "")
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r1", Status: "completed"}})

	asked := s.nextFrame(conn)
	s.Require().NotNil(asked.Response, "one reply answers both, guided by the prompt")
	s.Equal("Say: one moment.", asked.Response.Instructions)
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestARefusedRequestOfOursWithNoReplyOpenIsAnError() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	s.Require().NoError(provider.Prompt("Say: one moment."))
	s.refusedOurs(conn)
	events := s.settled(provider, conn)

	failures := errorsIn(events)
	s.Require().Len(failures, 1, "with no reply to wait for, asking again would be refused again")
	s.ErrorContains(failures[0].Err, activeResponse)
	s.Equal("server", failures[0].Context)
	s.nothingAsked(provider, conn)
}

func (s *RealtimeSocketSuite) TestAnOwedReplyThatCannotBeAskedForIsAnError() {
	fake := newFakeRealtime(false)
	defer fake.close()
	provider, conn := s.connect(fake, Options{Vendor: OpenAI})
	defer func() { _ = provider.Close() }()

	// A write deadline in the past fails every write and leaves reads alone.
	s.Require().NoError(provider.conn.SetWriteDeadline(time.Now().Add(-time.Second)))
	s.refused(conn, "")
	events := s.settled(provider, conn)

	failures := errorsIn(events)
	s.Require().Len(failures, 1)
	s.ErrorContains(failures[0].Err, "write response.create")
	s.Equal("response", failures[0].Context)
	s.False(failures[0].Fatal)
}

// TestABargeInThatInterruptsTheReplyAsksForNothing is the default path, interrupt_response
// true: the server cuts its reply off and answers the stop itself. Expected events come from
// a run on the base, before refusals were answered.
func (s *RealtimeSocketSuite) TestABargeInThatInterruptsTheReplyAsksForNothing() {
	fake := newFakeRealtime(false)
	defer fake.close()
	interrupts := true
	provider, conn := s.connect(fake, Options{Vendor: OpenAI, InterruptResponse: &interrupts})
	defer func() { _ = provider.Close() }()
	s.until(provider, func(event sts.Event) bool { _, ok := event.(sts.Connected); return ok })

	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r1"}})
	s.serve(conn, serverEvent{Type: eventSpeechStarted})
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{
		ID: "r1", Status: "cancelled", StatusDetails: &statusDetails{Type: "cancelled", Reason: "turn_detected"},
	}})
	s.serve(conn, serverEvent{Type: eventSpeechStopped})
	s.serve(conn, serverEvent{Type: eventResponseCreated, Response: &response{ID: "r2"}})
	s.serve(conn, serverEvent{Type: eventResponseDone, Response: &response{ID: "r2", Status: "completed"}})
	events := s.settled(provider, conn)

	var kinds []string
	for _, event := range events {
		kinds = append(kinds, fmt.Sprintf("%T", event))
	}
	s.Equal([]string{
		"sts.ResponseStarted", "sts.SpeechStarted", "sts.ResponseComplete", "sts.SpeechStopped",
		"sts.ResponseStarted", "sts.ResponseComplete", "sts.InputTranscript",
	}, kinds)
	s.True(events[2].(sts.ResponseComplete).Interrupted)
	s.False(events[5].(sts.ResponseComplete).Interrupted)
	s.nothingAsked(provider, conn)
}
