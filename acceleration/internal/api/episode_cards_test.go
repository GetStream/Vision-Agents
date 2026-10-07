//go:build integration

package api

import (
	"context"
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
// SlackChannelSuite.
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
	s.config = s.data.createAgentConfig()
	s.number = s.utils.number()
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
