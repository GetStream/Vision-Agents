//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// TestAMessageInAChannelOfATypeNoConfigNamesIsIgnored is the control: with no config naming a
// channel type, a message in a channel of a type other than agent is answered by nobody and
// handed to no worker, though the channel names one of the app's configs. The expected values,
// no hand-out and the person's message alone in the channel, are what base ed2c854b3 gives (the
// probe in the PR, which reads the channel by its id, as the fake held channels there).
func (s *AppHooksSuite) TestAMessageInAChannelOfATypeNoConfigNamesIsIgnored() {
	configID := s.config(s.customerID())
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	ctx, person, text, channel := context.Background(), "anya", "I'm sick, I can't come at 6", "team-"+s.utils.uuid()
	_, err := s.chat.Client.Chat().GetOrCreateChannel(ctx, "callouts", channel, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &person, Custom: map[string]any{ConfigField: configID}}})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().SendMessage(ctx, "callouts", channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &person}})
	s.Require().NoError(err)

	s.Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), fmt.Sprintf(`{
  "type": "message.new", "cid": "callouts:%s", "channel_id": %q, "channel_type": "callouts", "created_at": %q,
  "channel_custom": {%q: %q},
  "message": {"id": %q, "text": %q, "user": {"id": "anya", "name": "Anya"}}
}`, channel, channel, time.Now().UTC().Format(time.RFC3339Nano), ConfigField, configID, "message-"+s.utils.uuid(), text)))

	s.False(s.reaches(worker.Messages()), "the message was handed to a worker")
	s.Len(s.chat.Stored("callouts:"+channel), 1, "something answered in the channel")
}
