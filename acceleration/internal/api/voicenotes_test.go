//go:build integration

package api

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// VoiceNotesSuite covers a voice note sent with a question: what the person said in it is
// what the agent answers, and a note that cannot be heard is refused before any turn starts.
type VoiceNotesSuite struct {
	RouterSuite

	// notes serves voice notes by path, as a CDN a chat client uploaded them to would.
	notes  *httptest.Server
	served sync.Map
	// named is the User-Agent each path was fetched with.
	named sync.Map
}

func TestVoiceNotesSuite(t *testing.T) {
	runSuite(t, new(VoiceNotesSuite))
}

func (s *VoiceNotesSuite) SetupSuite() {
	// The router fetches over its public-egress client, which refuses loopback; the notes
	// are served on loopback, so the suite's router fetches with a plain client.
	s.imageHTTP = &http.Client{}
	s.notes = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.named.Store(r.URL.Path, r.UserAgent())
		body, ok := s.served.Load(r.URL.Path)
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(body.([]byte))
	}))
	s.T().Cleanup(s.notes.Close)
	s.RouterSuite.SetupSuite()
}

func (s *VoiceNotesSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.forgetNoted()
}

func (s *VoiceNotesSuite) TestWhatAVoiceNoteSaysIsTheQuestionTheAgentAnswers() {
	opened := s.noting()
	said := "the fryer on line two is leaking oil " + s.utils.uuid()

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Audio: &AudioSource{Url: s.serve(wavOf(said))}}, &answering))

	s.Equal(said, value(answering.Said), "what was said is what was asked")
	asked := s.lastAsked(said)
	s.Equal(said, asked, "the model is asked what was said")
}

func (s *VoiceNotesSuite) TestAVoiceNoteIsTranscribedByTheBatchModelForTheSessionsCustomer() {
	opened := s.noting()
	since := time.Now().UTC()

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Audio: &AudioSource{Url: s.serve(wavOf("hello " + s.utils.uuid()))}}, nil))

	rows := s.meteredRows(routing.STT, since, 1)
	s.Equal("stub-batch", rows[0].Model, "an English session's note goes to the English batch model")
}

func (s *VoiceNotesSuite) TestTypedTextComesBeforeWhatTheVoiceNoteSays() {
	opened := s.noting()
	typed := "About my shift today " + s.utils.uuid()
	said := "the walk-in cooler door does not close"
	note := "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wavOf(said))

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Text: typed, Audio: &AudioSource{Url: note}}, nil))

	s.Equal(typed+"\n\n"+said, s.lastAsked(said))
}

func (s *VoiceNotesSuite) TestTheVoiceNoteIsShownOnThePersonsMessageWithWhatItSays() {
	opened := s.noting()
	command, said := s.utils.uuid(), "we ran out of sweet tea at noon "+s.utils.uuid()
	audio := wavOf(said)
	before := len(s.chat.Uploads())

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: &command, Audio: &AudioSource{Url: s.serve(audio)}}, nil))

	channel := strings.TrimPrefix(value(opened.ConversationId), "agent:")
	var message map[string]any
	s.Require().Eventually(func() bool {
		for _, stored := range s.chat.Stored(channel) {
			if stored["id"] == command {
				message = stored
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the person's message was never written")
	s.Equal(said, message["text"], "the message carries what was said")
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"type":"voiceRecording"`, "the voice note is played where it is shown")
	s.Contains(string(attachments), `"title":"voice-note.wave"`)
	uploads := s.chat.Uploads()[before:]
	s.Require().Len(uploads, 1, "the voice note is uploaded once")
	s.Equal(audio, uploads[0].Data, "what is shown is the voice note as sent")
	s.Contains(string(attachments), `"asset_url":"`, "the message points at the upload")
}

func (s *VoiceNotesSuite) TestAVoiceNoteThatIsNotAudioIsRefusedAndNothingIsAsked() {
	opened := s.noting()

	status, failure := s.refused(opened, s.serve([]byte("<html><body>Not found</body></html>")))

	s.Equal(http.StatusBadRequest, status)
	s.Equal(string(ErrorTypeInvalidRequest), failure.Code)
	s.Contains(failure.Message, "not audio")
	s.nothingWasAsked(opened)
}

func (s *VoiceNotesSuite) TestAVoiceNoteOverTheInlineLimitIsRefusedAndNothingIsAsked() {
	opened := s.noting()
	tooBig := append(wavOf(""), make([]byte, maxInlineAudio)...)

	status, failure := s.refused(opened, s.serve(tooBig))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, "over 8 MB")
	s.nothingWasAsked(opened)
}

func (s *VoiceNotesSuite) TestAVoiceNoteAtTheInlineLimitIsHeard() {
	opened := s.noting()
	said := "padded " + s.utils.uuid()
	atLimit := wavOf(said + strings.Repeat(" ", maxInlineAudio-len(wavOf(said))))

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Audio: &AudioSource{Url: s.serve(atLimit)}}, nil))

	s.Equal(said, s.lastAsked(said), "what was heard is trimmed of the silence around it")
}

func (s *VoiceNotesSuite) TestAVoiceNoteWithNoWordsInItIsRefusedAndNothingIsAsked() {
	opened := s.noting()

	status, failure := s.refused(opened, s.serve(wavOf("  ")))

	s.Equal(http.StatusBadRequest, status)
	s.Equal(errNothingHeard.Message, failure.Message)
	s.nothingWasAsked(opened)
}

func (s *VoiceNotesSuite) TestAVoiceNoteRefusedForMoneyIsAnsweredAsMoneyAndNothingIsAsked() {
	opened := s.noting()

	status, failure := s.refused(opened, s.serve(wavOf(refuseForMoney)))

	s.Equal(http.StatusPaymentRequired, status)
	s.Equal(codeSpendLimitReached, failure.Code)
	s.nothingWasAsked(opened)
}

func (s *VoiceNotesSuite) TestAVoiceNoteTheProviderFailsOnIsAModelFailureAndNothingIsAsked() {
	opened := s.noting()

	status, failure := s.refused(opened, s.serve(wavOf(failToTranscribe)))

	s.Equal(http.StatusServiceUnavailable, status)
	s.Equal(codeModelFailed, failure.Code)
	s.nothingWasAsked(opened)
}

func (s *VoiceNotesSuite) TestAVoiceNoteThatCannotBeFetchedIsRefusedAndNothingIsAsked() {
	opened := s.noting()

	status, failure := s.refused(opened, s.notes.URL+"/missing-"+s.utils.uuid())

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, "404")
	s.nothingWasAsked(opened)
}

func (s *VoiceNotesSuite) TestAQuestionWithNeitherTextNorAVoiceNoteIsRefused() {
	opened := s.noting()

	status, _ := s.serverClient.call(http.MethodPost, s.responses(opened), CreateResponseRequest{})

	s.Equal(http.StatusBadRequest, status)
}

func (s *VoiceNotesSuite) TestARetryOfAVoiceNoteIsHeardOnceAndMatchedOnWhatWasSent() {
	opened := s.noting()
	since := time.Now().UTC()
	command, typed, said := s.utils.uuid(), "From the morning shift", "the ice machine is empty "+s.utils.uuid()
	note := s.serve(wavOf(said))

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: &command, Text: typed, Audio: &AudioSource{Url: note}}, &answering))
	s.Require().Equal(typed+"\n\n"+said, value(answering.Said))
	var retried AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: &command, Text: typed, Audio: &AudioSource{Url: note}}, &retried))
	s.Empty(retried.Id, "a retry starts no second turn")

	for name, other := range map[string]CreateResponseRequest{
		"another voice note": {RequestId: &command, Text: typed, Audio: &AudioSource{Url: s.serve(wavOf(said + " and the fryer"))}},
		"other typed text":   {RequestId: &command, Text: "From the evening shift", Audio: &AudioSource{Url: note}},
	} {
		status, _ := s.serverClient.call(http.MethodPost, s.responses(opened), other)
		s.Equal(http.StatusConflict, status, name)
	}
	s.heard(since, 1)
}

func (s *VoiceNotesSuite) TestAVoiceNoteRetriedOnAQuestionSentWithoutOneIsAnotherQuestionAndNotHeard() {
	opened := s.noting()
	since := time.Now().UTC()
	command := s.utils.uuid()
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: &command, Text: "Is the fryer fixed?"}, nil))

	status, _ := s.serverClient.call(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: &command, Text: "Is the fryer fixed?", Audio: &AudioSource{Url: s.serve(wavOf("it is not"))}})

	s.Equal(http.StatusConflict, status)
	s.heard(since, 0)
}

func (s *VoiceNotesSuite) TestAVoiceNoteWithAnEffortTheModelDoesNotTakeIsRefusedBeforeItIsHeard() {
	opened := s.noting()
	since := time.Now().UTC()
	effort := "bogus-effort"

	status, _ := s.serverClient.call(http.MethodPost, s.responses(opened),
		CreateResponseRequest{ReasoningEffort: &effort, Audio: &AudioSource{Url: s.serve(wavOf("hi " + s.utils.uuid()))}})

	s.Equal(http.StatusBadRequest, status)
	s.heard(since, 0)
}

func (s *VoiceNotesSuite) TestAVoiceNoteWithAPhotoForAnAgentThatCannotSeeIsRefusedBeforeItIsHeard() {
	opened := s.noting()
	since := time.Now().UTC()
	photo := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))

	status, payload := s.serverClient.call(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Images: &[]ImageSource{{Url: photo}}, Audio: &AudioSource{Url: s.serve(wavOf("look " + s.utils.uuid()))}})

	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeNotConfigured, answered.Error.Code)
	s.heard(since, 0)
}

func (s *VoiceNotesSuite) TestAVoiceNoteWithTextTooLongForACommandIsRefusedBeforeItIsHeard() {
	opened := s.noting()
	since := time.Now().UTC()
	command := s.utils.uuid()

	status, _ := s.serverClient.call(http.MethodPost, s.responses(opened),
		CreateResponseRequest{RequestId: &command, Text: strings.Repeat("a", 1024*1024+1), Audio: &AudioSource{Url: s.serve(wavOf("long " + s.utils.uuid()))}})

	s.Equal(http.StatusBadRequest, status)
	s.heard(since, 0)
}

func (s *VoiceNotesSuite) TestAVoiceNoteIsBilledUnderTheSessionsTags() {
	tags := map[string]string{"team": "kitchen-" + s.utils.uuid()}
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("noted/noted-model"), Tags: &tags})
	since := time.Now().UTC()

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Audio: &AudioSource{Url: s.serve(wavOf("tagged " + s.utils.uuid()))}}, nil))

	rows := s.meteredRows(routing.STT, since, 1)
	s.Equal(tags["team"], rows[0].Tags["team"])
}

func (s *VoiceNotesSuite) TestAVoiceNoteIsBilledToItsSessionAndAgentAsTheAnswerIs() {
	opened := s.noting()
	s.Require().NotEmpty(opened.AgentId)
	since := time.Now().UTC()

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Audio: &AudioSource{Url: s.serve(wavOf("billed " + s.utils.uuid()))}}, nil))

	heard := s.meteredRows(routing.STT, since, 1)[0]
	s.Equal(opened.Id, heard.SessionID, "the transcription names the session that heard the note")
	s.Equal(opened.AgentId, heard.AgentID, "the transcription names the session's agent")
	answered := s.meteredRows(routing.LLM, since, 1)[0]
	s.Equal([]string{answered.SessionID, answered.AgentID, answered.APIKey, answered.UserID},
		[]string{heard.SessionID, heard.AgentID, heard.APIKey, heard.UserID},
		"the note is billed to whom its answer is")
}

func (s *VoiceNotesSuite) TestAVoiceNoteIsFetchedUnderTheRoutersName() {
	opened := s.noting()
	note := s.serve(wavOf("named " + s.utils.uuid()))

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Audio: &AudioSource{Url: note}}, nil))

	named, fetched := s.named.Load(strings.TrimPrefix(note, s.notes.URL))
	s.Require().True(fetched, "the note was never fetched")
	s.Equal(egress.UserAgent, named, "a host that refuses Go's own User-Agent still serves the note")
}

func (s *VoiceNotesSuite) TestAVoiceNoteSentAsADataURIOfTheInlineLimitIsHeard() {
	opened := s.noting()
	said := "big " + s.utils.uuid()
	atLimit := wavOf(said + strings.Repeat(" ", maxInlineAudio-len(wavOf(said))))

	status, payload := s.serverClient.call(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Audio: &AudioSource{Url: "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(atLimit)}})

	s.Require().Equal(http.StatusAccepted, status, "%.300s", payload)
	s.Equal(said, s.lastAsked(said))
}

// heard checks that n voice notes were transcribed for the suite's app since since. It hears
// one more on a session tagged its own, and counts once that one's row is written: one writer
// writes the rows in the order they were recorded, so theirs are written by then.
func (s *VoiceNotesSuite) heard(since time.Time, n int) {
	tags := map[string]string{"counted": s.utils.uuid()}
	counted := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("noted/noted-model"), Tags: &tags})
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(counted),
		CreateResponseRequest{Audio: &AudioSource{Url: s.serve(wavOf("counted " + s.utils.uuid()))}}, nil))
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.ShardDB().NewSelect().Model(&rows).
			Where("customer_id = ? AND modality = ? AND started_at >= ?", s.customerID(), string(routing.STT), since).
			Scan(context.Background())
		return err == nil && slices.ContainsFunc(rows, func(row store.Request) bool { return row.Tags["counted"] == tags["counted"] })
	}, settleFor, 20*time.Millisecond, "the counting note's row was never written")
	s.Len(rows, n+1, "voice notes transcribed besides the counting one")
}

// noting opens a text session on a noted model, which a test reads what it was asked from.
func (s *VoiceNotesSuite) noting() Session {
	return s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("noted/noted-model")})
}

func (s *VoiceNotesSuite) responses(opened Session) string {
	return "/v1/agents/sessions/" + opened.Id + "/responses"
}

// serve has the suite's CDN serve body, and returns where.
func (s *VoiceNotesSuite) serve(body []byte) string {
	path := "/" + s.utils.uuid() + ".wav"
	s.served.Store(path, body)
	return s.notes.URL + path
}

// refused sends a voice note the router refuses, and returns how.
func (s *VoiceNotesSuite) refused(opened Session, note string) (int, ErrorDetail) {
	status, payload := s.serverClient.call(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Audio: &AudioSource{Url: note}})
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error
}

// nothingWasAsked asks a question after a refused one and checks it is the first turn the
// session's model was asked: a turn the refusal had started would have been asked first.
func (s *VoiceNotesSuite) nothingWasAsked(opened Session) {
	next := "And the next question " + s.utils.uuid()
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, s.responses(opened),
		CreateResponseRequest{Text: next}, nil))
	s.lastAsked(next)
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	var replies int
	for _, model := range s.noted {
		replies += len(model.replies())
	}
	s.Equal(1, replies, "the refused voice note started a turn")
}
