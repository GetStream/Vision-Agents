package omnichannel

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PersonSuite is how a person is keyed in the contact map, decided before any store or
// Stream Chat is reached. Cards are written end to end in internal/api's EpisodeCardsSuite
// and SlackChannelSuite.
type PersonSuite struct {
	suite.Suite
}

func TestPersonSuite(t *testing.T) {
	suite.Run(t, new(PersonSuite))
}

func (s *PersonSuite) TestANumberWrittenAnyWayIsOneE164Address() {
	for _, written := range []string{"+1 (555) 010-0100", "+1.555.010.0100", "15550100100", " +15550100100 "} {
		person, err := Phone(written)
		s.Require().NoError(err, written)
		s.Equal(Person{Kind: store.ContactPhone, Address: "+15550100100"}, person, written)
	}
}

// A national number starts with a trunk prefix, 0, which says nothing of the country.
func (s *PersonSuite) TestANationalNumberIsRefused() {
	_, err := Phone("020 7946 0018")
	s.ErrorContains(err, "not a phone number in E.164")
}

// E.164 allows 15 digits at most.
func (s *PersonSuite) TestSixteenDigitsAreRefused() {
	_, err := Phone("+1234567890123456")
	s.ErrorContains(err, "not a phone number in E.164")
}

func (s *PersonSuite) TestASlackUserIdIsNotANumber() {
	_, err := Phone("U0000ALICE")
	s.ErrorContains(err, "not a phone number in E.164")
}

func (s *PersonSuite) TestASlackUserIsKeyedByTheirWorkspace() {
	person, err := SlackUser("T0000TEAM", "U0000ALICE")

	s.Require().NoError(err)
	s.Equal(Person{Kind: store.ContactSlack, Address: "T0000TEAM:U0000ALICE"}, person)
}

func (s *PersonSuite) TestASlackUserNeedsAWorkspace() {
	_, err := SlackUser("", "U0000ALICE")
	s.ErrorContains(err, "a Slack team and a Slack user are required")
}

func (s *PersonSuite) TestCardsNeedAStoreAndStreamClients() {
	_, err := New(Options{})
	s.ErrorContains(err, "a store and Stream clients are required")
}
