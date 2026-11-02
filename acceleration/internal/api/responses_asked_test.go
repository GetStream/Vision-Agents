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
	asked, _ := s.askedWith(text)
	turn := harness.WithoutNotes(asked.Input)
	s.Equal(text, turn[len(turn)-1].Content, "the model is asked the text as typed")
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
