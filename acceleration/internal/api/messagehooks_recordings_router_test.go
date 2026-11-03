//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A voice recording sent alone in a plain agent channel nothing runs on reaches the worker
// handed the message as what it says, heard once for the channel's agent and the person who
// sent it. A delivery made again is ChatInboundSuite's: this suite's deployment acts on every
// delivery of a message, text too, as it always has (recorded).
func (s *MessageHooksSuite) TestAVoiceRecordingSentAloneReachesTheWorkerAsWhatItSays() {
	worker := s.workerOf(s.customerID())
	said := "is the walk-in cooler fixed " + s.utils.uuid()
	since := time.Now().UTC()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.recordedMessage(s.namesConfig(), "", s.recordingAt(said))))

	s.Equal(said, s.handed(worker).Text)
	rows := s.meteredRows(routing.STT, since, 1)
	s.Equal(s.channelID, rows[0].AgentID)
	s.Equal(routing.OriginHookChat, rows[0].APIKey)
	s.Equal("sam", rows[0].UserID)
	s.Empty(rows[0].SessionID, "no session runs yet")
}

// A voice recording sent with text reaches the worker as the text, then what it says.
func (s *MessageHooksSuite) TestAVoiceRecordingSentWithTextReachesTheWorkerAfterTheText() {
	worker := s.workerOf(s.customerID())
	said := "the fryer is leaking " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(s.namesConfig(), "About line two", s.recordingAt(said))))

	s.Equal("About line two\n\n"+said, s.handed(worker).Text)
}

// A recording that cannot be fetched is not transcribed: the worker is handed the note that
// says it could not be heard, and no transcription is billed.
func (s *MessageHooksSuite) TestAVoiceRecordingThatCannotBeFetchedIsSaidSoAndNotBilled() {
	worker := s.workerOf(s.customerID())
	since := time.Now().UTC()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(s.namesConfig(), "", s.recordingFrom(s.images.URL+"/gone.aac"))))

	s.Equal(unheardRecording, s.handed(worker).Text)
	s.Never(func() bool { return len(s.sttRowsSince(since)) > 0 }, dropped, 10*time.Millisecond, "a recording never fetched was billed")
}

// A recording nothing would answer, with no worker connected, is not heard.
func (s *MessageHooksSuite) TestAVoiceRecordingNoWorkerWouldAnswerIsNotHeard() {
	since := time.Now().UTC()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(s.namesConfig(), "", s.recordingAt("hello "+s.utils.uuid()))))

	s.Require().Eventually(func() bool {
		return strings.Contains(s.logged.String(), `msg="nobody could answer an arriving message" channel=`+s.channelID)
	}, settleFor, 10*time.Millisecond, "the message was never offered to a worker")
	s.Never(func() bool { return len(s.sttRowsSince(since)) > 0 }, time.Second, 10*time.Millisecond, "a recording nobody answers was heard")
}

// A voice recording sent to a running session whose agent leaves text to dispatch reaches its
// worker with the session as what it says, billed to that session.
func (s *MessageHooksSuite) TestAVoiceRecordingToADispatchSessionReachesItsWorkerAsWhatItSays() {
	session := s.dispatching()
	worker := s.workerOf(s.customerID())
	said := "we ran out of sweet tea " + s.utils.uuid()
	since := time.Now().UTC()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(s.ownedBy(s.customerID()), "", s.recordingAt(said))))

	handed := s.handed(worker)
	s.Equal(session, handed.SessionID)
	s.Equal(said, handed.Text)
	s.Equal(session, s.meteredRows(routing.STT, since, 1)[0].SessionID)
}

// A voice recording that cannot be heard, sent alone to a running session whose agent leaves
// text to dispatch, reaches its worker as the note saying so: never as an empty text, which a
// worker's session refuses to answer (nothingToAnswer).
func (s *MessageHooksSuite) TestAnUnheardVoiceRecordingToADispatchSessionReachesItsWorkerAsANote() {
	session := s.dispatching()
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(s.ownedBy(s.customerID()), "", s.recordingFrom(s.images.URL+"/gone.aac"))))

	handed := s.handed(worker)
	s.Equal(session, handed.SessionID)
	s.Equal(unheardRecording, handed.Text)
}

// A voice recording sent to a running session the router answers reaches its model as what it
// says, after the text sent with it, and is billed to that session.
func (s *MessageHooksSuite) TestAVoiceRecordingToARunningSessionReachesTheModelAsWhatItSays() {
	opened := s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	said := "the walk-in door does not close " + s.utils.uuid()
	since := time.Now().UTC()
	asked := len(s.vision.requests())

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.recordedMessage("", "", s.recordingAt(said))))
	alone := s.askedOf(s.vision, asked)
	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage("", "About my shift", s.recordingAt(said), s.photoAt("/photo.png"))))
	withMore := s.askedOf(s.vision, asked+1)

	s.Equal(said, alone.Content)
	s.Equal("About my shift\n\n"+said, withMore.Parts[0].Text)
	s.Len(imagesIn(withMore), 1, "the photo goes with what was said")
	rows := s.meteredRows(routing.STT, since, 2)
	s.Equal(opened.Id, rows[0].SessionID)
	s.Equal(s.channelID, rows[0].AgentID)
}

// A running session told of a recording it could not hear says so, after answering the text
// or the photo sent with it.
func (s *MessageHooksSuite) TestARecordingARunningSessionCannotHearIsSaidSo() {
	s.running("vision/vision-model")
	asked := len(s.vision.requests())

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage("", "is it fixed?", s.recordingFrom(s.images.URL+"/gone.aac"))))

	s.Equal("is it fixed?\n\n"+unheardRecordingWithMore, s.askedOf(s.vision, asked).Content)
	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage("", "", s.recordingFrom(s.images.URL+"/gone.aac"), s.photoAt("/photo.png"))))
	withPhoto := s.askedOf(s.vision, asked+1)
	s.Equal(unheardRecordingWithMore, withPhoto.Parts[0].Text, "the photo is the rest of the message")
	s.Len(imagesIn(withPhoto), 1)
}

// The voice recording the agent shows on a person's message it wrote itself carries its source,
// and is never heard back: not by a running session, nor by a worker.
func (s *MessageHooksSuite) TestARecordingOnAMessageTheAgentWroteIsNotHeard() {
	worker := s.workerOf(s.customerID())
	s.running("vision/vision-model")
	asked := len(s.vision.requests())
	since := time.Now().UTC()
	var message map[string]any
	s.Require().NoError(json.Unmarshal([]byte(s.recordedMessage(s.namesConfig(), "", s.recordingAt("hello "+s.utils.uuid()))), &message))
	message["message"].(map[string]any)["source"] = "agent"
	payload, err := json.Marshal(message)
	s.Require().NoError(err)

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", string(payload)))

	s.nothingReaches(worker, "the agent's own recording")
	s.Never(func() bool { return len(s.vision.requests()) > asked || len(s.sttRowsSince(since)) > 0 },
		dropped, 10*time.Millisecond, "the agent's own recording was heard")
}

// Control, as on base: a voice recording alone in a channel named as a thread channel that no
// thread is linked to reaches no worker and is not heard.
func (s *MessageHooksSuite) TestAVoiceRecordingAloneInAnUnlinkedThreadChannelIsNotHeard() {
	worker := s.workerOf(s.customerID())
	s.channelID = "thread-" + s.utils.uuid()
	since := time.Now().UTC()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(s.namesConfig(), "", s.recordingAt("hello "+s.utils.uuid()))))

	s.nothingReaches(worker, "a recording alone in an unlinked thread channel")
	s.Never(func() bool { return len(s.sttRowsSince(since)) > 0 }, dropped, 10*time.Millisecond, "the recording was heard")
}

// A channel whose last call names no config, or a config nobody holds any more, still hears a
// recording, for the channel's customer.
func (s *MessageHooksSuite) TestAVoiceRecordingInAChannelWithNoConfigIsHeardForItsCustomer() {
	worker := s.workerOf(s.customerID())
	for _, config := range []string{"", "gone-" + s.utils.uuid()} {
		s.channelID = "chat-" + s.utils.uuid()
		s.ran(config)
		said := "hello " + s.utils.uuid()

		s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.recordedMessage("", "", s.recordingAt(said))))

		s.Equal(said, s.handed(worker).Text, config)
	}
}

// A voice recording written in the channel a worker's session answers reaches that session's
// model as what it says, billed to it, and is not handed out again.
func (s *MessageHooksSuite) TestAVoiceRecordingToAWorkersSessionReachesItsModelAsWhatItSays() {
	handed := s.handedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{
		AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model"),
	})
	s.asked(opened.Id, handed.Text)
	asked := len(s.vision.requests())
	said := "and what about Spain " + s.utils.uuid()
	since := time.Now().UTC()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.recordedMessage("", "", s.recordingAt(said))))

	s.Equal(said, s.askedOf(s.vision, asked).Content)
	s.Equal(opened.Id, s.meteredRows(routing.STT, since, 1)[0].SessionID)
	s.nothingReaches(s.worker, "a recording to a running session was dispatched again")
}

// A hook is taken before its voice recording is heard: hearing takes up to inlineDeadline, and
// Stream waits on the delivery. The worker has it once it is heard.
func (s *MessageHooksSuite) TestAHookIsTakenBeforeItsVoiceRecordingIsHeard() {
	worker := s.workerOf(s.customerID())
	said := "the ice machine is out " + s.utils.uuid()
	served := make(chan struct{})
	recording := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-served
		_, _ = w.Write(wavOf(said))
	}))
	s.T().Cleanup(recording.Close)
	body := s.recordedMessage(s.namesConfig(), "", s.recordingFrom(recording.URL+"/recording.aac"))
	answered := make(chan int, 1)
	go func() {
		response, err := s.server.Client().Do(s.request("/v1/chat/hooks/stream", body, sign(body, suiteStreamSecret)))
		if err != nil {
			answered <- 0
			return
		}
		_ = response.Body.Close()
		answered <- response.StatusCode
	}()

	select {
	case status := <-answered:
		s.Equal(http.StatusOK, status)
	case <-time.After(3 * settleFor):
		s.Fail("the hook waited for its recording to be heard")
	}
	close(served)
	s.Equal(said, s.handed(worker).Text)
}

// A recording sent where no session runs is heard for the config answering the channel: one
// whose region has no batch transcriber does not hear it, and the worker is told so.
func (s *MessageHooksSuite) TestAVoiceRecordingForAConfigInARegionThatHearsNothingIsSaidSo() {
	worker := s.workerOf(s.customerID())
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(), Mode: "text", Region: pricing.RegionEU}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	var fetched atomic.Int32
	recording := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fetched.Add(1)
		_, _ = w.Write(wavOf("hello " + s.utils.uuid()))
	}))
	s.T().Cleanup(recording.Close)
	since := time.Now().UTC()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(fmt.Sprintf(`{%q: %q}`, ConfigField, config.ID), "", s.recordingFrom(recording.URL+"/recording.aac"))))

	s.Equal(unheardRecording, s.handed(worker).Text)
	s.Zero(fetched.Load(), "the recording was fetched")
	s.Never(func() bool { return len(s.sttRowsSince(since)) > 0 }, time.Second, 10*time.Millisecond, "a recording its region cannot hear was heard")
}

// Control, as on base: a deployment that hears nothing hands a worker a message with text and a
// voice recording as the text alone, with no note about the recording.
func (s *MessageHooksSuite) TestAVoiceRecordingSentWithTextToADeploymentThatHearsNothingIsTheTextAlone() {
	streams := s.router.streams
	deaf := *streams
	deaf.Transcriptions = nil
	s.router.streams = &deaf
	defer func() { s.router.streams = streams }()
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(s.namesConfig(), "About line two", s.recordingAt("the fryer is leaking"))))

	s.Equal("About line two", s.handed(worker).Text)
}

// A speech-to-speech session answers only on its call, so a recording written to it is not
// heard: nothing is billed for a message it does not answer in writing.
func (s *MessageHooksSuite) TestAVoiceRecordingToASpeechToSpeechSessionIsNotHeard() {
	opened := s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, StartVoice: pointerTo(true), Sts: pointerTo("sts-fast"), Incognito: pointerTo(true),
	})
	s.Require().Equal("native", string(value(s.serverClient.getSession(opened.Id).Mode)))
	since := time.Now().UTC()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage("", "is it fixed?", s.recordingAt("hello "+s.utils.uuid()))))

	s.Require().Eventually(func() bool {
		return strings.Contains(s.logged.String(), `msg="could not answer a message" channel=`+s.channelID)
	}, settleFor, 10*time.Millisecond, "the session was never asked")
	s.Never(func() bool { return len(s.sttRowsSince(since)) > 0 }, time.Second, 10*time.Millisecond, "a recording to a speech-to-speech session was heard")
}

// recordedMessage is a signed message.new for the suite's channel with text and attachments, as
// Stream would send it. The custom data is whatever the channel was created with.
func (s *MessageHooksSuite) recordedMessage(custom, text string, attachments ...map[string]any) string {
	if custom == "" {
		custom = "{}"
	}
	payload, err := json.Marshal(map[string]any{
		"type": "message.new", "cid": "agent:" + s.channelID, "channel_id": s.channelID, "channel_type": "agent",
		"channel_custom": json.RawMessage(custom),
		"message": map[string]any{
			"id": s.utils.uuid(), "text": text, "user": map[string]any{"id": "sam", "name": "Sam"}, "attachments": attachments,
		},
	})
	s.Require().NoError(err)
	return string(payload)
}

// recordingAt is a voice recording that says said, served by a server of the test's own.
func (s *MessageHooksSuite) recordingAt(said string) map[string]any {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wavOf(said)) }))
	s.T().Cleanup(server.Close)
	return s.recordingFrom(server.URL + "/recording.aac")
}

// recordingFrom is a voiceRecording attachment at url, as a Stream Chat client sends one.
func (s *MessageHooksSuite) recordingFrom(url string) map[string]any {
	return map[string]any{"type": "voiceRecording", "asset_url": url, "mime_type": "audio/aac", "duration": 2.5}
}

// sttRowsSince are the customer's transcription rows since then, as they stand now.
func (s *RouterSuite) sttRowsSince(since time.Time) []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).
		Where("customer_id = ? AND modality = ? AND started_at >= ?", s.customerID(), string(routing.STT), since).
		Scan(context.Background()))
	return rows
}
