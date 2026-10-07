package channelbridge

import (
	"regexp"
	"testing"

	"github.com/stretchr/testify/suite"
)

// BridgeSuite is what the bridge decides before it reaches a store or Stream Chat. Its whole
// flow, from a Slack event to a reply, runs in internal/api's SlackChannelSuite against
// Postgres, the suite's Stream Chat and the fake provider.
type BridgeSuite struct {
	suite.Suite
}

func TestBridgeSuite(t *testing.T) {
	suite.Run(t, new(BridgeSuite))
}

func (s *BridgeSuite) TestAnAuthorIsTheSameUserOnEveryMessage() {
	s.Equal(authorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"), authorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"))
}

// One Slack workspace can install two customers' apps; its person is a user of each, never
// one user both customers write as.
func (s *BridgeSuite) TestTheSamePersonForAnotherCustomerIsAnotherUser() {
	s.NotEqual(authorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"), authorUserID("globex", "slack_bot", "T0000TEAM", "U0000ALICE"))
}

func (s *BridgeSuite) TestPartsThatJoinAlikeAreStillTwoUsers() {
	s.NotEqual(authorUserID("acme", "slack_bot", "T0", "0U"), authorUserID("acme", "slack_bot", "T00", "U"))
}

// A phone number's "+" or any provider's id stays out of the user id.
func (s *BridgeSuite) TestAnAuthorUserIDKeepsToLettersDigitsUnderscoreAndHyphen() {
	s.Regexp(regexp.MustCompile(`^[A-Za-z0-9_-]+$`), authorUserID("acme", "linq", "+12025550100", "+12025550199"))
}

func (s *BridgeSuite) TestABridgeNeedsAStoreStreamTransportsAndAResolver() {
	_, err := New(Options{})
	s.ErrorContains(err, "a store, Stream clients, transports and a resolver are required")
}
