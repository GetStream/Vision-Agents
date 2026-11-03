//go:build integration

package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ResponsesAskedSuite covers what a question sent with text, and with photos, puts in front of
// the model and on the person's message in Stream Chat.
type ResponsesAskedSuite struct {
	RouterSuite
}

func TestResponsesAskedSuite(t *testing.T) {
	runSuite(t, new(ResponsesAskedSuite))
}

func (s *ResponsesAskedSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ResponsesAskedSuite) TestATextQuestionIsAskedAndShownAsTyped() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("noted/noted-model")})
	command, text := s.utils.uuid(), "Is the drive-thru open late "+s.utils.uuid()+"?"

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Text: text}, &answering))

	s.Equal(text, value(answering.Said))
	s.Equal(text, s.lastAsked(text), "the model is asked the text as typed")
	message := s.personsMessage(opened, command)
	s.Equal(text, message["text"])
	s.Empty(message["attachments"], "a text question carries no files")
}

func (s *ResponsesAskedSuite) TestAPhotoQuestionIsShownToTheModelAndOnThePersonsMessage() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	command, text := s.utils.uuid(), "What is wrong with this fryer "+s.utils.uuid()+"?"
	photo := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))
	before := len(s.chat.Uploads())

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Text: text, Images: &[]ImageSource{{Url: photo}}}, &answering))

	s.Equal(text, value(answering.Said))
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests() {
			if request.HasImage() && strings.Contains(fmt.Sprint(request.Input), text) {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the photo with the question")
	message := s.personsMessage(opened, command)
	s.Equal(text, message["text"])
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"type":"image"`)
	s.Contains(string(attachments), `"title":"photo-1.png"`)
	s.Len(s.chat.Uploads()[before:], 1)
}

func (s *ResponsesAskedSuite) TestAPhotoSentWithoutTextIsAQuestionTheModelIsShown() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	command := s.utils.uuid()
	photo := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"+command))
	before := len(s.vision.requests())

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Images: &[]ImageSource{{Url: photo}}}, &answering))

	s.NotEmpty(answering.Id, "a turn started")
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests()[before:] {
			if request.HasImage() {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the photo")
	message := s.personsMessage(opened, command)
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"type":"image"`)
}

// A photo with no text is a question without a request id too, and the person's message
// shows it as it shows a command's (AI-2550).
func (s *ResponsesAskedSuite) TestAPhotoSentWithoutTextOrRequestIDIsAQuestionShownOnThePersonsMessage() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	data := []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + s.utils.uuid())
	photo := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
	before, asked := len(s.chat.Uploads()), len(s.vision.requests())

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Images: &[]ImageSource{{Url: photo}}}, &answering))

	s.NotEmpty(answering.Id, "a turn started")
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests()[asked:] {
			if request.HasImage() {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the photo")
	message := s.onlyPersonsMessage(opened)
	s.Empty(message["text"])
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"type":"image"`)
	s.Contains(string(attachments), `"title":"photo-1.png"`)
	uploads := s.chat.Uploads()[before:]
	s.Require().Len(uploads, 1)
	s.Equal(data, uploads[0].Data, "the photo uploaded is the one sent")
}

// What a photo-only question without a request id is held to is what a command's photos are:
// a link is not taken from elsewhere, and a message shows at most 8.
func (s *ResponsesAskedSuite) TestAPhotoSentWithoutTextOrRequestIDIsHeldToACommandsRules() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	photo := ImageSource{Url: "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))}
	nine := []ImageSource{photo, photo, photo, photo, photo, photo, photo, photo, photo}
	before := len(s.chat.Uploads())

	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Images: &[]ImageSource{{Url: "https://example.com/a.png"}}})
	s.Equal(http.StatusBadRequest, status, "a photo linked from elsewhere")
	s.Contains(string(body), "takes photos as data URIs")
	status, body = s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Images: &nine})
	s.Equal(http.StatusBadRequest, status, "more photos than a message shows")
	s.Contains(string(body), "at most 8 photos")
	s.Len(s.chat.Uploads()[before:], 0, "nothing refused is uploaded")
}

// Controls, each answered the same on the base (49fa88bde): a question without a request id
// that has text is shown as typed, its photos going to the model alone, and one with nothing
// in it is refused.
func (s *ResponsesAskedSuite) TestAQuestionWithTextAndNoRequestIDIsShownAsTyped() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	text := "Is the drive-thru open late " + s.utils.uuid() + "?"
	before := len(s.chat.Uploads())

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses", CreateResponseRequest{Text: text}, nil))
	message := s.onlyPersonsMessage(opened)
	s.Equal(text, message["text"])
	s.Empty(message["attachments"], "a text question carries no files")

	other := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	linked := "What is in this photo " + s.utils.uuid() + "?"
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+other.Id+"/responses",
		CreateResponseRequest{Text: linked, Images: &[]ImageSource{{Url: "https://example.com/a.png"}}}, nil))
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests() {
			if request.HasImage() && strings.Contains(fmt.Sprint(request.Input), linked) {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the linked photo")
	s.Len(s.chat.Uploads()[before:], 0, "a question with text uploads nothing without a request id")

	empty := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	status, _ := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+empty.Id+"/responses",
		CreateResponseRequest{})
	s.Equal(http.StatusBadRequest, status, "a question with nothing in it")
}

// onlyPersonsMessage is the first message the person wrote in the session's channel.
func (s *ResponsesAskedSuite) onlyPersonsMessage(opened Session) map[string]any {
	channel := strings.TrimPrefix(value(opened.ConversationId), "agent:")
	var message map[string]any
	s.Require().Eventually(func() bool {
		for _, stored := range s.chat.Stored(channel) {
			custom, _ := stored["custom"].(map[string]any)
			if written, _ := custom["support_message"].(map[string]any); written["role"] == "user" {
				message = stored
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the person's message was never written")
	return message
}

// personsMessage is the message a command was written as in the session's channel.
func (s *ResponsesAskedSuite) personsMessage(opened Session, command string) map[string]any {
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
	return message
}

// lastAsked is the person's last message in a turn a noted model was asked that ends
// with said.
func (s *RouterSuite) lastAsked(said string) string {
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
