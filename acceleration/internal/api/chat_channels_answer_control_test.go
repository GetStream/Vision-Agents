//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// TestATypeNamedWithNoAnswerAnswersEveryMessage is the control for answer: a type named with no
// answer answers a message that mentions nobody, and the model answering Anya is not handed Ben's
// message written before hers and after the last reply, which no hook delivered. The expected
// values, one reply by the config's user and Ben's words in no request, are what base e418743c6
// gives (the probe runs this file there unchanged).
func (s *AppHooksSuite) TestATypeNamedWithNoAnswerAnswersEveryMessage() {
	s.forgetNoted()
	s.holdsType("callouts", memberWrites)
	config := s.answeringIn("callouts")
	ctx, channel, anya, ben := context.Background(), "team-"+s.utils.uuid(), "anya", "ben"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(ctx, "callouts", channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &anya}})
	s.Require().NoError(err)
	unheard := "feel better! " + s.utils.uuid()
	_, err = s.chat.Client.Chat().SendMessage(ctx, "callouts", channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &unheard, UserID: &ben}})
	s.Require().NoError(err)
	// Anya's message is stored under the id its hook names, so a history read for it finds it.
	id, asked := "message-"+s.utils.uuid(), "I'm sick, I can't come at 6 "+s.utils.uuid()
	_, err = s.chat.Client.Chat().SendMessage(ctx, "callouts", channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{ID: &id, Text: &asked, UserID: &anya}})
	s.Require().NoError(err)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), fmt.Sprintf(`{
  "type": "message.new", "cid": "callouts:%s", "channel_id": %q, "channel_type": "callouts", "created_at": %q,
  "channel_custom": {}, "message": {"id": %q, "text": %q, "user": {"id": "anya", "name": "Anya"}}
}`, channel, channel, time.Now().UTC().Format(time.RFC3339Nano), id, asked)))

	reply := s.replyIn("callouts:" + channel)
	s.Require().NotNil(reply, "a message mentioning nobody was not answered")
	s.Equal(config.ID, reply["user_id"])
	request, _ := s.askedWith(asked)
	s.Require().NotEmpty(request.Input, "the model answering Anya was not found")
	s.False(slices.ContainsFunc(request.Input, func(m llm.Message) bool { return strings.Contains(m.Content, unheard) }),
		"a message no hook delivered was handed to the model: %+v", request.Input)
}
