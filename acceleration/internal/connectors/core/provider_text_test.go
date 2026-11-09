package core_test

import (
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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

// TestAURLEscapedEchoOfASecretIsRedacted (AI-1649): a token request's form body and basic auth
// carry a secret QueryEscaped, and a provider may echo it so, or PathEscaped.
func (s *LoggedDescriptionSuite) TestAURLEscapedEchoOfASecretIsRedacted() {
	text := "bad s3cr%2Fet%2Btok%3Den%26x, path s3cr%2Fet+tok=en&x, raw s3cr/et+tok=en&x"

	s.Equal("bad [redacted], path [redacted], raw [redacted]", core.LoggedDescription(text, "s3cr/et+tok=en&x"))
}

// TestAJSONEscapedEchoOfASecretIsRedacted: a JSON body echoes a secret escaped, its & as \u0026
// by Go's encoding/json, its / as \/ by PHP's json_encode, or with only its " and \ escaped.
func (s *LoggedDescriptionSuite) TestAJSONEscapedEchoOfASecretIsRedacted() {
	secret := `tok/a&b<c"d\e`
	text := `go tok/a\u0026b\u003cc\"d\\e, php tok\/a\u0026b\u003cc\"d\\e, plain tok/a&b<c\"d\\e, plain php tok\/a&b<c\"d\\e`

	s.Equal("go [redacted], php [redacted], plain [redacted], plain php [redacted]", core.LoggedDescription(text, secret))
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

// LoggedMessageSuite is what a log line says of the error a call returned (LoggedMessage,
// AI-1242): the provider's or the refusal's words, with no URL's path or query, cut to 1 KiB.
type LoggedMessageSuite struct {
	suite.Suite
}

func TestLoggedMessageSuite(t *testing.T) {
	suite.Run(t, new(LoggedMessageSuite))
}

func (s *LoggedMessageSuite) TestAToolErrorIsTheToolsOwnWords() {
	err := stack.Wrap(&core.ToolError{Message: "the record is locked"})

	s.Equal("the record is locked", core.LoggedMessage(err, nil))
}

func (s *LoggedMessageSuite) TestNoErrorIsNoMessage() {
	s.Empty(core.LoggedMessage(nil, nil))
}

// TestARedactedURLStillKeepsItsHost: the URLs are cut before redact runs, so a URL whose path
// redact takes out still keeps its host, and redact takes the token out of the rest.
func (s *LoggedMessageSuite) TestARedactedURLStillKeepsItsHost() {
	redact := strings.NewReplacer("/primary/mcp", "[redacted]", queryToken, "[redacted]").Replace
	err := errors.New(`Post "https://crm.example/primary/mcp": invalid token ` + queryToken)

	s.Equal(`Post "https://crm.example": invalid token [redacted]`, core.LoggedMessage(err, redact))
}

func (s *LoggedMessageSuite) TestAURLErrorKeepsOnlyItsHost() {
	sending := &url.Error{Op: "Post", URL: "https://user:pass@crm.example/mcp/v1?access_token=" + queryToken + "#frag", Err: io.EOF}
	err := stack.Wrap(fmt.Errorf("mcp: crm__whoami: %w", sending))

	s.Equal(`mcp: crm__whoami: Post "https://crm.example": EOF`, core.LoggedMessage(err, nil))
}

// TestAURLTheSDKFlattenedIntoTextKeepsOnlyItsHost: the MCP SDK wraps a failed reconnect with
// %v (go-sdk v1.8.0 mcp/streamable.go:2174 and :2560), so no *url.Error is left to find.
func (s *LoggedMessageSuite) TestAURLTheSDKFlattenedIntoTextKeepsOnlyItsHost() {
	sending := &url.Error{Op: "Get", URL: "https://crm.example/mcp?access_token=" + queryToken, Err: io.EOF}
	err := fmt.Errorf("standalone SSE request failed (session ID: s1): %v", sending)

	s.Equal(`standalone SSE request failed (session ID: s1): Get "https://crm.example": EOF`, core.LoggedMessage(err, nil))
}

// An unparseable URL (the %zz) is replaced whole, its query token with it.
func (s *LoggedMessageSuite) TestAURLThatDoesNotParseIsReplacedWhole() {
	err := errors.New(`Get "https://crm.example/%zz?access_token=` + queryToken + `": EOF`)

	s.Equal(`Get "<url>": EOF`, core.LoggedMessage(err, nil))
}

func (s *LoggedMessageSuite) TestAnUppercaseSchemeKeepsOnlyItsHost() {
	err := errors.New(`Get "HTTPS://crm.example/mcp?access_token=` + queryToken + `": EOF`)

	s.Equal(`Get "https://crm.example": EOF`, core.LoggedMessage(err, nil))
}

// The one KiB is 1024 bytes, not whatever core.MessageBytes holds.
func (s *LoggedMessageSuite) TestAMessageIsCutTo1024Bytes() {
	message := core.LoggedMessage(&core.ToolError{Message: strings.Repeat("a", 5000)}, nil)

	s.Len(message, 1024)
}

// Text that is not UTF-8 (nothing but continuation bytes) is cut at most UTFMax-1 bytes
// before the limit, not back to nothing.
func (s *LoggedMessageSuite) TestTextThatIsNotUTF8IsStillCutNearTheLimit() {
	message := core.LoggedMessage(&core.ToolError{Message: strings.Repeat("\x80", 5000)}, nil)

	s.Len(message, 1024-(utf8.UTFMax-1))
}

func (s *LoggedMessageSuite) TestAMessageOverOneKiBIsCutAtARuneBoundary() {
	// 1023 bytes, then a 3-byte rune across the limit: the cut leaves the rune out whole.
	err := &core.ToolError{Message: strings.Repeat("a", core.MessageBytes-1) + "€ and more"}

	message := core.LoggedMessage(err, nil)

	s.Equal(strings.Repeat("a", core.MessageBytes-1), message)
	s.True(utf8.ValidString(message))
}

func (s *LoggedMessageSuite) TestAMessageOfOneKiBIsKeptWhole() {
	text := strings.Repeat("a", core.MessageBytes-3) + "€"

	s.Equal(text, core.LoggedMessage(&core.ToolError{Message: text}, nil))
}
