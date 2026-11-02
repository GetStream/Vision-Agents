//go:build integration

package api

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// VoiceNotesSuite covers a voice note sent with a question: what the person said in it is
// what the agent answers, and a note that cannot be heard is refused before any turn starts.
type VoiceNotesSuite struct {
	RouterSuite

	// notes serves voice notes by path, as a CDN a chat client uploaded them to would.
	notes  *httptest.Server
	served sync.Map
}

func TestVoiceNotesSuite(t *testing.T) {
	runSuite(t, new(VoiceNotesSuite))
}

func (s *VoiceNotesSuite) SetupSuite() {
	// The router fetches over its public-egress client, which refuses loopback; the notes
	// are served on loopback, so the suite's router fetches with a plain client.
	s.imageHTTP = &http.Client{}
	s.notes = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	s.Contains(string(attachments), `"type":"file"`, "the voice note is shown as a file")
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

// lastAsked is the person's last message in the one turn a noted model was asked that ends
// with said.
func (s *VoiceNotesSuite) lastAsked(said string) string {
	var asked string
	s.Require().Eventually(func() bool {
		s.notedMu.Lock()
		defer s.notedMu.Unlock()
		for _, model := range s.noted {
			for _, request := range model.replies() {
				turn := harness.WithoutNotes(request.Input)
				if len(turn) == 0 {
					continue
				}
				if last := turn[len(turn)-1]; last.Role == llm.User && strings.HasSuffix(last.Content, said) {
					asked = last.Content
					return true
				}
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "no turn asked %q", said)
	return asked
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
