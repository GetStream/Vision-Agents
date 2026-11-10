//go:build integration

package api

import (
	"context"
	"net/http"
	"slices"
	"strings"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// TestATypeNamedWithNoAnswerAnswersEveryMessage is the control for answer: a type named with no
// answer answers a message that mentions nobody, and the model answering it is not handed Ben's
// message written after the last reply, which no hook delivered. The expected values, one reply
// by the config's user and Ben's words in no request, are what base 4c2f7b835 gives (the probe
// in the PR runs this file there unchanged).
func (s *AppHooksSuite) TestATypeNamedWithNoAnswerAnswersEveryMessage() {
	s.forgetNoted()
	s.holdsType("callouts", memberWrites)
	config := s.answeringIn("callouts")
	channel := s.groupChannel("callouts", nil)
	ben, text := "ben", "feel better! "+s.utils.uuid()
	_, err := s.chat.Client.Chat().SendMessage(context.Background(), "callouts", channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &ben}})
	s.Require().NoError(err)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), s.groupMessage("callouts", channel, nil)))

	reply := s.replyIn("callouts:" + channel)
	s.Require().NotNil(reply, "a message mentioning nobody was not answered")
	s.Equal(config.ID, reply["user_id"])
	request, _ := s.askedWith("I'm sick, I can't come at 6")
	s.False(slices.ContainsFunc(request.Input, func(m llm.Message) bool { return strings.Contains(m.Content, text) }),
		"a message after the last reply was handed to the model: %+v", request.Input)
}
