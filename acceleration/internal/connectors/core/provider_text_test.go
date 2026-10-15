package core_test

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// LoggedDescriptionSuite is what a log line keeps of a provider's error_description (AI-1245):
// what the request carried redacted, URLs kept to their host, at most 256 bytes.
type LoggedDescriptionSuite struct {
	suite.Suite
}

func TestLoggedDescriptionSuite(t *testing.T) {
	suite.Run(t, new(LoggedDescriptionSuite))
}

func (s *LoggedDescriptionSuite) TestAnEchoedSecretIsRedactedAndAURLKeepsItsHost() {
	text := "refresh token rt-secret-1 is revoked, see https://idp.example/errors/rt-secret-1?client_secret=cs-2 now"

	s.Equal("refresh token [redacted] is revoked, see https://idp.example now",
		core.LoggedDescription(text, "rt-secret-1", "cs-2"))
}

// TestASecretOfOneByteIsLeftAlone: a byte alone names nothing, and taking it out would take
// out every such byte, as the Exchange's redaction leaves it.
func (s *LoggedDescriptionSuite) TestASecretOfOneByteIsLeftAlone() {
	s.Equal("a denied consent", core.LoggedDescription("a denied consent", "a", ""))
}

func (s *LoggedDescriptionSuite) TestALongDescriptionIsCutTo256Bytes() {
	described := core.LoggedDescription(strings.Repeat("a", 300))

	s.Equal(strings.Repeat("a", 256), described)
	s.LessOrEqual(len(described), 256)
}

// TestACutInsideARuneLeavesThatRuneOut: 255 bytes and a two-byte rune are 257 bytes, so the cut
// at 256 falls inside the rune.
func (s *LoggedDescriptionSuite) TestACutInsideARuneLeavesThatRuneOut() {
	described := core.LoggedDescription(strings.Repeat("a", 255) + "é")

	s.Equal(strings.Repeat("a", 255), described)
	s.True(utf8.ValidString(described))
}

func (s *LoggedDescriptionSuite) TestADescriptionOf256BytesIsKeptWhole() {
	text := strings.Repeat("a", 254) + "é"

	s.Equal(text, core.LoggedDescription(text))
}
