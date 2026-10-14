//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// EpisodeCardsSuite is the call path's episode card and the contact map behind it (T41 and
// T43): a session that joins a phone call writes one card into the caller's omni-channel,
// found by the caller's number. The caller is the call's SIP participant, which the suite's
// Stream holds as the routing rule names it, sip-<number>. The bridge's cards are in
// SlackChannelSuite. A config writes call cards only once it turns episode_cards on; the
// suite's agent does, and TestACallUnderAConfigThatDidNotTurnCardsOnIsAsBefore's does not.
type EpisodeCardsSuite struct {
	RouterSuite

	// config is the agent the test's calls run under, number the caller's.
	config AgentConfig
	number string
}

func TestEpisodeCardsSuite(t *testing.T) {
	runSuite(t, new(EpisodeCardsSuite))
}

func (s *EpisodeCardsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.config = s.agentConfig(true)
	s.number = s.utils.number()
}

// Before this PR a call under an agent config read nothing of its call and wrote no contact
// and no card. A config that did not turn the cards on keeps it so.
func (s *EpisodeCardsSuite) TestACallUnderAConfigThatDidNotTurnCardsOnIsAsBefore() {
	s.config = s.agentConfig(false)

	call := s.called("sip-" + s.number)

	s.Never(func() bool { return s.readCall(call) || s.contacts() > 0 }, dropped, 20*time.Millisecond,
		"the call was read, or a contact was made")
}

func (s *EpisodeCardsSuite) TestACallMakesOneCardWithSourceCallLinkedToItsCallChannel() {
	call := s.called("sip-" + s.number)

	omni := s.omniChannel(s.number)
	card := s.cards(omni, 1)[0]
	s.Never(func() bool { return len(s.chat.Stored(omni)) > 1 }, dropped, 20*time.Millisecond)
	custom, _ := card["custom"].(map[string]any)
	s.Equal("call", custom["source"])
	s.Equal(call, custom["call_id"])
	s.Equal("agent:"+call, custom["thread_channel"], "the call channel the transcript is in")
	s.Equal("in_progress", custom["status"])
	s.Equal(1, s.contacts(), "a new number makes one row")
	s.True(s.readCall(call), "the caller is read off the call, which is what a config that did not opt in never does")
}

// Channel ids are seen by every member and every client of the app, so the omni-channel's
// is never the number.
func (s *EpisodeCardsSuite) TestTheOmniChannelIdHoldsNoNumber() {
	s.called("sip-" + s.number)

	omni := s.omniChannel(s.number)

	s.NotContains(omni, strings.TrimPrefix(s.number, "+"))
	s.Regexp(`^omni-[0-9a-f-]{36}$`, omni)
}

// An SMS and then a call from one number are one person: the call's card goes beside the
// SMS thread's. SMS is not on the channel bridge until T53, so the SMS side opens its episode
// as the bridge will, with the number as the provider writes it.
func (s *EpisodeCardsSuite) TestTheSameNumberFromAnSMSAndACallIsOneOmniChannel() {
	cards, err := omnichannel.New(omnichannel.Options{Store: s.store, Stream: s.stream})
	s.Require().NoError(err)
	person, err := omnichannel.Phone(" " + s.number[:2] + " " + s.number[2:5] + "-" + s.number[5:])
	s.Require().NoError(err)
	ctx := context.Background()
	opened, err := cards.Open(ctx, omnichannel.Episode{
		CustomerID: s.customerID(), AgentConfigID: s.config.Id, AgentName: s.config.Name, Person: person,
		Source: "sms", ThreadChannel: "agent:thread-" + s.utils.uuid(),
	})
	s.Require().NoError(err)
	s.Require().NoError(cards.Write(ctx, opened))

	s.called("sip-" + s.number)

	omni := s.omniChannel(s.number)
	sources := map[any]bool{}
	for _, card := range s.cards(omni, 2) {
		custom, _ := card["custom"].(map[string]any)
		sources[custom["source"]] = true
	}
	s.Equal(map[any]bool{"sms": true, "call": true}, sources)
	s.Equal(1, s.contacts(), "one person, one row")
}

// A caller number that does not say it is international is not guessed at: no row, no card,
// and the call goes on as the session it is.
func (s *EpisodeCardsSuite) TestACallFromANumberWithoutItsPlusMakesNoCard() {
	call := s.utils.uuid()
	s.chat.PutCall("agent", call, "sip-5550100100")
	created := s.serverClient.createSession(CreateSessionRequest{Id: &call, StartVoice: pointerTo(true), ConfigId: &s.config.Id})

	s.Eventually(func() bool { return s.readCall(call) }, settleFor, 10*time.Millisecond)
	s.Never(func() bool { return s.contacts() > 0 }, dropped, 20*time.Millisecond)
	var running Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+created.Id, nil, &running))
	s.Equal(Live, running.State, "the call's session runs on")
}

// A call no phone is on, such as one from a browser, has nobody the contact map keys.
func (s *EpisodeCardsSuite) TestACallWithNoPhoneOnItMakesNoCard() {
	s.called("someone-in-a-browser")

	s.Never(func() bool { return s.contacts() > 0 }, dropped, 20*time.Millisecond)
}

// A call that started with the person's cards is refused a move onto a speech-to-speech
// model by both operations that move one, with the one named failure; a call that read no
// cards moves as before.
func (s *EpisodeCardsSuite) TestACallWithCardsIsRefusedAMoveOntoSpeechToSpeech() {
	s.smsWithALine(s.number, "is the clinic open on Sunday?")
	carded := s.joined("sip-" + s.number)
	plain := s.joined("sip-" + s.utils.number())

	for _, move := range []struct {
		path string
		body any
	}{
		{"/v1/agents/sessions/" + carded + "/settings", SessionSettingsRequest{Sts: pointerTo("sts-fast")}},
		{"/v1/agents/sessions/" + carded, UpdateSessionRequest{Sts: pointerTo("sts-fast")}},
	} {
		status, payload := s.serverClient.call(http.MethodPatch, move.path, move.body)
		s.Equal(http.StatusBadRequest, status, move.path)
		var failure ErrorResponse
		s.Require().NoError(json.Unmarshal(payload, &failure))
		s.Equal("carded_session_to_native", failure.Error.Code, move.path)
		s.NotContains(failure.Error.Message, "session:", "a clean message, not the session package's")
	}
	// The suite's sessions have no speech-to-speech models, so the move is refused by the agent,
	// as at base: the call without cards gets that refusal, not the cards' one.
	_, payload := s.serverClient.call(http.MethodPatch, "/v1/agents/sessions/"+plain+"/settings",
		SessionSettingsRequest{Sts: pointerTo("sts-fast")})
	var failure ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Equal("invalid_request", failure.Error.Code, "a call without cards moves, or fails to, as before")
	s.Contains(failure.Error.Message, "no speech-to-speech models")
}

// smsWithALine opens an SMS episode of number, as the bridge will (T53), with one line in its
// thread channel, so a call from the number starts with its card.
func (s *EpisodeCardsSuite) smsWithALine(number, text string) {
	ctx := context.Background()
	cards, err := omnichannel.New(omnichannel.Options{Store: s.store, Stream: s.stream})
	s.Require().NoError(err)
	person, err := omnichannel.Phone(number)
	s.Require().NoError(err)
	bound, err := s.stream.For(ctx, s.customerID())
	s.Require().NoError(err)
	thread, author := "thread-"+s.utils.uuid(), "sms-author"
	_, err = bound.Client.Chat().GetOrCreateChannel(ctx, "agent", thread, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &author, Custom: map[string]any{"support_customer_id": s.customerID()}},
	})
	s.Require().NoError(err)
	_, err = bound.Client.Chat().SendMessage(ctx, "agent", thread, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &author},
	})
	s.Require().NoError(err)
	opened, err := cards.Open(ctx, omnichannel.Episode{
		CustomerID: s.customerID(), AgentConfigID: s.config.Id, AgentName: s.config.Name, Person: person,
		Source: "sms", ThreadChannel: "agent:" + thread,
	})
	s.Require().NoError(err)
	s.Require().NoError(cards.Write(ctx, opened))
}

// joined is the id of a session under the test's agent on a call the participant is on.
func (s *EpisodeCardsSuite) joined(participant string) string {
	call := s.utils.uuid()
	s.chat.PutCall("agent", call, participant)
	return s.serverClient.createSession(CreateSessionRequest{Id: &call, StartVoice: pointerTo(true), ConfigId: &s.config.Id}).Id
}

// called opens a session under the test's agent on a call whose session holds the
// participants named, and returns the call's id.
func (s *EpisodeCardsSuite) called(participants ...string) string {
	call := s.utils.uuid()
	s.chat.PutCall("agent", call, participants...)
	s.serverClient.createSession(CreateSessionRequest{Id: &call, StartVoice: pointerTo(true), ConfigId: &s.config.Id})
	return call
}

// agentConfig is an agent of the test's app, with episode cards on or off.
func (s *EpisodeCardsSuite) agentConfig(cards bool) AgentConfig {
	return s.agentConfigOn(cards, "llm-flow")
}

// agentConfigOn is an agent of the test's app on the LLM named, with episode cards on or off.
func (s *EpisodeCardsSuite) agentConfigOn(cards bool, model string) AgentConfig {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo(model), Instructions: pointerTo("be brief"),
		EpisodeCards: pointerTo(cards),
	}, &created))
	return created
}

// The call hook ends the call's episode (T55): ended, then summarized by the agent config's
// own LLM, recites, which answers with what it was handed, so the summary holds the call's
// lines. The card is updated in place: the omni-channel still holds one message.
func (s *EpisodeCardsSuite) TestACallsCardIsSummarizedWhenTheCallEnds() {
	s.config = s.agentConfigOn(true, "recites/recites-model")
	call := s.called("sip-" + s.number)
	omni := s.omniChannel(s.number)
	s.cards(omni, 1)
	episode := s.callEpisode(call)
	s.say(call, "sip-"+s.number, "I need to move my appointment to Tuesday")

	s.Equal(http.StatusOK, s.signedly(phone.CallHookPath, s.sessionEnded(call)))

	s.Require().Eventually(func() bool { return s.episodeStatus(episode) == store.EpisodeSummarized }, settleFor, 10*time.Millisecond,
		"the episode is %s", s.episodeStatus(episode))
	stored := s.chat.Stored(omni)
	s.Require().Len(stored, 1, "the card is updated in place, never sent again")
	s.Contains(stored[0]["text"], "I need to move my appointment to Tuesday")
	custom, _ := stored[0]["custom"].(map[string]any)
	s.Equal(store.EpisodeSummarized, custom["status"])
}

// The summary is written off the hook: Stream has its 200 while the model is still writing,
// and the episode is ended until the summary lands.
func (s *EpisodeCardsSuite) TestTheCallHookAnswersBeforeTheSummaryIsWritten() {
	s.config = s.agentConfigOn(true, "holding/holding-model")
	call := s.called("sip-" + s.number)
	omni := s.omniChannel(s.number)
	s.cards(omni, 1)
	episode := s.callEpisode(call)
	s.say(call, "sip-"+s.number, "I need to move my appointment to Tuesday")
	s.holding.holds()
	s.T().Cleanup(s.holding.answers)
	request := s.request(phone.CallHookPath, s.sessionEnded(call), sign(s.sessionEnded(call), suiteStreamSecret))
	answered := make(chan int, 1)
	go func() {
		response, err := s.server.Client().Do(request)
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
	case <-time.After(settleFor):
		s.Fail("the hook waited for the summary")
	}
	s.Require().Eventually(func() bool {
		for _, asked := range s.holding.requests() {
			if asked.ID == "summary-"+episode {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the summary was asked for")
	s.Equal(store.EpisodeEnded, s.episodeStatus(episode), "the model has not answered yet")

	s.holding.answers()

	s.Require().Eventually(func() bool { return s.episodeStatus(episode) == store.EpisodeSummarized }, settleFor, 10*time.Millisecond)
}

// Before T55 the call.session_ended hook released the call's trunks and answered 200 with no
// body. A call under a config that did not turn the cards on has no episode, so the hook
// answers the same, writes no row and asks nothing of Stream.
func (s *EpisodeCardsSuite) TestACallUnderAConfigWithoutCardsEndsAsBefore() {
	s.config = s.agentConfig(false)
	call := s.called("sip-" + s.number)
	// What the session's own start asks of Stream, once it has stopped asking.
	before := -1
	s.Require().Eventually(func() bool {
		now := len(s.chat.Requests(suiteStreamKey))
		settled := now == before
		before = now
		return settled
	}, settleFor, 100*time.Millisecond)

	status, body := s.deliver(phone.CallHookPath, s.sessionEnded(call), sign(s.sessionEnded(call), suiteStreamSecret))

	s.Equal(http.StatusOK, status)
	s.Empty(body)
	s.Never(func() bool { return len(s.chat.Requests(suiteStreamKey)) > before }, dropped, 20*time.Millisecond,
		"the hook asked Stream something")
	var episodes int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM episodes WHERE customer_id = ?", s.customerID()).Scan(&episodes))
	s.Zero(episodes)
}

// sessionEnded is the call.session_ended event Stream sends for an agent call.
func (s *EpisodeCardsSuite) sessionEnded(call string) string {
	return fmt.Sprintf(`{"type":"call.session_ended","call_cid":"agent:%s","session_id":"session-1",`+
		`"call":{"cid":"agent:%s","id":"%s","type":"agent","custom":{}}}`, call, call, call)
}

// callEpisode is the id of the call's episode, once its card has opened it.
func (s *EpisodeCardsSuite) callEpisode(call string) string {
	var id string
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			"SELECT id FROM episodes WHERE customer_id = ? AND call_id = ?", s.customerID(), call).Scan(&id) == nil
	}, settleFor, 10*time.Millisecond)
	return id
}

// episodeStatus is an episode's status as stored.
func (s *EpisodeCardsSuite) episodeStatus(id string) string {
	var status string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(), "SELECT status FROM episodes WHERE id = ?", id).Scan(&status))
	return status
}

// say writes a line said on the call into its call channel, as the transcript writes it: by
// the participant, with source speech.
func (s *EpisodeCardsSuite) say(call, participant, text string) {
	ctx := context.Background()
	bound, err := s.stream.For(ctx, s.customerID())
	s.Require().NoError(err)
	_, err = bound.Client.Chat().GetOrCreateChannel(ctx, "agent", call, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &participant, Custom: map[string]any{"support_customer_id": s.customerID()}},
	})
	s.Require().NoError(err)
	_, err = bound.Client.Chat().SendMessage(ctx, "agent", call, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &participant, Custom: map[string]any{"source": "speech"}},
	})
	s.Require().NoError(err)
}

// readCall is whether the router asked Stream for a call, as a card's caller lookup does.
func (s *EpisodeCardsSuite) readCall(call string) bool {
	for _, request := range s.chat.Requests(suiteStreamKey) {
		if request.Method == http.MethodGet && strings.HasSuffix(request.Path, "/video/call/agent/"+call) {
			return true
		}
	}
	return false
}

// omniChannel is the id of the omni-channel the contact map gives a number for the test's
// agent, once the call's card made it.
func (s *EpisodeCardsSuite) omniChannel(number string) string {
	var cid string
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			"SELECT conversation_id FROM contact_map WHERE customer_id = ? AND agent_config_id = ? AND kind = ? AND address = ?",
			s.customerID(), s.config.Id, store.ContactPhone, number).Scan(&cid) == nil
	}, settleFor, 10*time.Millisecond, "no contact for %s", number)
	return strings.TrimPrefix(cid, "agent:")
}

// cards waits until an omni-channel holds count cards, written off the session's start.
func (s *EpisodeCardsSuite) cards(omni string, count int) []map[string]any {
	s.Require().Eventually(func() bool { return len(s.chat.Stored(omni)) >= count }, settleFor, 10*time.Millisecond,
		"%s holds %d cards, not %d", omni, len(s.chat.Stored(omni)), count)
	return s.chat.Stored(omni)
}

// contacts is how many contact map rows the test's agent has.
func (s *EpisodeCardsSuite) contacts() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM contact_map WHERE customer_id = ? AND agent_config_id = ?", s.customerID(), s.config.Id).Scan(&count))
	return count
}
