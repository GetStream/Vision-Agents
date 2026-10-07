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

// An international number is read however it is written: with + or 00, spaced, hyphenated,
// dotted, bracketed, or with the (0) trunk prefix after its country code.
func (s *PersonSuite) TestAnInternationalNumberWrittenAnyWayIsOneE164Address() {
	for written, e164 := range map[string]string{
		"+15550100100":         "+15550100100",
		" +15550100100 ":       "+15550100100",
		"+1 (555) 010-0100":    "+15550100100",
		"+1.555.010.0100":      "+15550100100",
		"0015550100100":        "+15550100100",
		"+44 (0) 20 7946 0018": "+442079460018",
		"0044 (0)20 7946 0018": "+442079460018",
		"+44 20 7946 0018":     "+442079460018",
		"+123456789012345":     "+123456789012345",
		"+12345678":            "+12345678",
		"+55 11 91234-5678":    "+5511912345678",
	} {
		person, err := Phone(written)
		s.Require().NoError(err, written)
		s.Equal(Person{Kind: store.ContactPhone, Address: e164}, person, written)
	}
}

// What does not say it is international is refused, never guessed at: without its plus
// 5550100100 could be national anywhere, or Brazil's +55.
func (s *PersonSuite) TestAnythingThatIsNotPlainlyInternationalIsRefused() {
	for _, written := range []string{
		"5550100100",            // no plus: national, or +55?
		"15550100100",           // no plus either
		"1001",                  // a short code
		"911",                   // an emergency number
		"12",                    // too short to be a number
		"+1001",                 // under 8 digits
		"+1234567",              // 7 digits
		"+1234567890123456",     // over the 15 E.164 allows
		"020 7946 0018",         // national, with its trunk zero
		"+44 (020) 7946 0018",   // a trunk zero inside the area code
		"+0 555 010 0100",       // no country code starts with 0
		"U0000ALICE",            // a Slack user id
		"sip-+15550100100",      // a participant id, not a number
		"+1 555 010 0100 ext 2", // letters
		"",
	} {
		_, err := Phone(written)
		s.ErrorContains(err, "not a phone number in E.164", "%q", written)
	}
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
