package session

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

// ProviderMessageSuite is what a "connector call failed" line says of the error a call
// returned (providerMessage, AI-1242): the provider's or the refusal's words, with no URL's
// path or query, cut to 1 KiB.
type ProviderMessageSuite struct {
	suite.Suite
}

func TestProviderMessageSuite(t *testing.T) {
	suite.Run(t, new(ProviderMessageSuite))
}

// The synthetic token a URL in these tests carries.
const queryToken = "query-token-must-not-print"

func (s *ProviderMessageSuite) TestAToolErrorIsTheToolsOwnWords() {
	err := stack.Wrap(&core.ToolError{Message: "the record is locked"})

	s.Equal("the record is locked", providerMessage(err, nil))
}

func (s *ProviderMessageSuite) TestNoErrorIsNoMessage() {
	s.Empty(providerMessage(nil, nil))
}

// TestARedactedURLStillKeepsItsHost: the URLs are cut before redact runs, so a URL whose path
// redact takes out still keeps its host, and redact takes the token out of the rest.
func (s *ProviderMessageSuite) TestARedactedURLStillKeepsItsHost() {
	redact := strings.NewReplacer("/primary/mcp", "[redacted]", queryToken, "[redacted]").Replace
	err := errors.New(`Post "https://crm.example/primary/mcp": invalid token ` + queryToken)

	s.Equal(`Post "https://crm.example": invalid token [redacted]`, providerMessage(err, redact))
}

func (s *ProviderMessageSuite) TestAURLErrorKeepsOnlyItsHost() {
	sending := &url.Error{Op: "Post", URL: "https://user:pass@crm.example/mcp/v1?access_token=" + queryToken + "#frag", Err: io.EOF}
	err := stack.Wrap(fmt.Errorf("mcp: crm__whoami: %w", sending))

	s.Equal(`mcp: crm__whoami: Post "https://crm.example": EOF`, providerMessage(err, nil))
}

// TestAURLTheSDKFlattenedIntoTextKeepsOnlyItsHost: the MCP SDK wraps a failed reconnect with
// %v (go-sdk v1.8.0 mcp/streamable.go:2174 and :2560), so no *url.Error is left to find.
func (s *ProviderMessageSuite) TestAURLTheSDKFlattenedIntoTextKeepsOnlyItsHost() {
	sending := &url.Error{Op: "Get", URL: "https://crm.example/mcp?access_token=" + queryToken, Err: io.EOF}
	err := fmt.Errorf("standalone SSE request failed (session ID: s1): %v", sending)

	s.Equal(`standalone SSE request failed (session ID: s1): Get "https://crm.example": EOF`, providerMessage(err, nil))
}

// An unparseable URL (the %zz) is replaced whole, its query token with it.
func (s *ProviderMessageSuite) TestAURLThatDoesNotParseIsReplacedWhole() {
	err := errors.New(`Get "https://crm.example/%zz?access_token=` + queryToken + `": EOF`)

	s.Equal(`Get "<url>": EOF`, providerMessage(err, nil))
}

func (s *ProviderMessageSuite) TestAnUppercaseSchemeKeepsOnlyItsHost() {
	err := errors.New(`Get "HTTPS://crm.example/mcp?access_token=` + queryToken + `": EOF`)

	s.Equal(`Get "https://crm.example": EOF`, providerMessage(err, nil))
}

// The one KiB is 1024 bytes, not whatever providerMessageBytes holds.
func (s *ProviderMessageSuite) TestAMessageIsCutTo1024Bytes() {
	message := providerMessage(&core.ToolError{Message: strings.Repeat("a", 5000)}, nil)

	s.Len(message, 1024)
}

// Text that is not UTF-8 (nothing but continuation bytes) is cut at most UTFMax-1 bytes
// before the limit, not back to nothing.
func (s *ProviderMessageSuite) TestTextThatIsNotUTF8IsStillCutNearTheLimit() {
	message := providerMessage(&core.ToolError{Message: strings.Repeat("\x80", 5000)}, nil)

	s.Len(message, 1024-(utf8.UTFMax-1))
}

func (s *ProviderMessageSuite) TestAMessageOverOneKiBIsCutAtARuneBoundary() {
	// 1023 bytes, then a 3-byte rune across the limit: the cut leaves the rune out whole.
	err := &core.ToolError{Message: strings.Repeat("a", providerMessageBytes-1) + "€ and more"}

	message := providerMessage(err, nil)

	s.Equal(strings.Repeat("a", providerMessageBytes-1), message)
	s.True(utf8.ValidString(message))
}

func (s *ProviderMessageSuite) TestAMessageOfOneKiBIsKeptWhole() {
	text := strings.Repeat("a", providerMessageBytes-3) + "€"

	s.Equal(text, providerMessage(&core.ToolError{Message: text}, nil))
}

// TestABodyThatEndsInsideARuneLosesThatRune: the transport keeps 1 KiB of a body
// (core.Exchange.ErrorBody), which can end on the first two bytes of a 3-byte rune.
func (s *ProviderMessageSuite) TestABodyThatEndsInsideARuneLosesThatRune() {
	kept := strings.Repeat("a", providerMessageBytes-2) + "€"[:2]

	s.Equal(strings.Repeat("a", providerMessageBytes-2), providerBody(kept))
}

func (s *ProviderMessageSuite) TestABodyThatEndsOnAWholeRuneIsKept() {
	kept := strings.Repeat("a", providerMessageBytes-3) + "€"

	s.Equal(kept, providerBody(kept))
}

func (s *ProviderMessageSuite) TestABodysURLsKeepOnlyTheirHost() {
	kept := `{"error":"see https://crm.example/errors/42?access_token=` + queryToken + `"}`

	s.Equal(`{"error":"see https://crm.example"}`, providerBody(kept))
}
