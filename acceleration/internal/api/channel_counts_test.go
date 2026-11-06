//go:build integration

package api

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
)

// TelnyxCountsSuite is a Telnyx line whose customer's organization is read from a real usage
// service, so the gate counts each reply the bridge sends against the organization's channel
// limits as a router does.
type TelnyxCountsSuite struct {
	telnyxLine
}

func TestTelnyxCountsSuite(t *testing.T) {
	runSuite(t, new(TelnyxCountsSuite))
}

// SetupSuite gives the suite a database of its own, as its tests start a scheduled-task worker.
func (s *TelnyxCountsSuite) SetupSuite() {
	s.database = "telnyx_counts"
	s.limited = true
	s.telnyxLine.SetupSuite()
}

// reached is who the app's organization reached by SMS today, as its channel counters say.
func (s *TelnyxCountsSuite) reached() []string {
	redis := s.live.Redis()
	key := "channel:" + time.Now().UTC().Format("20060102") + ":{" + s.RouterSuite.app.organization.ID + "}:sms:to"
	members, err := redis.Do(context.Background(), redis.B().Smembers().Key(key).Build()).AsStrSlice()
	s.Require().NoError(err)
	slices.Sort(members)
	return members
}

// A reply the bridge sent counts against the person it went to (channelbridge Bridge.sent):
// that number is reached today, one that was never written to is not, and neither is the empty
// recipient a reply to nobody would count against.
func (s *TelnyxCountsSuite) TestAReplyIsCountedAgainstThePersonItWentTo() {
	first, second := "+13125550001", "+13125550002"
	s.usage.Credit(s.T(), s.RouterSuite.app.organization.ID, 10_000_000)
	s.Require().Eventually(func() bool {
		return s.gate.Allow(context.Background(), s.customerID(), dlc.SMS, first) == nil
	}, 10*time.Second, 50*time.Millisecond, "the credit is read")
	s.Require().Empty(s.reached(), "nothing is reached before a reply")

	s.smsTurn(first, "hi")

	s.Require().Eventually(func() bool { return len(s.reached()) > 0 }, settleFor, 10*time.Millisecond, "the reply is counted")
	s.Equal([]string{first}, s.reached())

	s.smsTurn(second, "hello")

	s.Require().Eventually(func() bool { return len(s.reached()) > 1 }, settleFor, 10*time.Millisecond, "the second reply is counted")
	s.Equal([]string{first, second}, s.reached())
}
