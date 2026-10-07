//go:build integration

package api

import (
	"context"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
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
	call := s.utils.callID()
	s.chat.PutCall("agent", call, "sip-5550100100")
	created := s.serverClient.createSession(CreateSessionRequest{CallId: &call, CallType: pointerTo("agent"), ConfigId: &s.config.Id})

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

// called opens a session under the test's agent on a call whose session holds the
// participants named, and returns the call's id.
func (s *EpisodeCardsSuite) called(participants ...string) string {
	call := s.utils.callID()
	s.chat.PutCall("agent", call, participants...)
	s.serverClient.createSession(CreateSessionRequest{CallId: &call, CallType: pointerTo("agent"), ConfigId: &s.config.Id})
	return call
}

// agentConfig is an agent of the test's app, with episode cards on or off.
func (s *EpisodeCardsSuite) agentConfig(cards bool) AgentConfig {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo("llm-flow"), Instructions: pointerTo("be brief"),
		EpisodeCards: pointerTo(cards),
	}, &created))
	return created
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
