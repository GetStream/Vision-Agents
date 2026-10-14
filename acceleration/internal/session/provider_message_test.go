package session

import (
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

	s.Equal("the record is locked", providerMessage(err))
}

func (s *ProviderMessageSuite) TestNoErrorIsNoMessage() {
	s.Empty(providerMessage(nil))
}

func (s *ProviderMessageSuite) TestAURLErrorKeepsOnlyItsHost() {
	sending := &url.Error{Op: "Post", URL: "https://user:pass@crm.example/mcp/v1?access_token=" + queryToken + "#frag", Err: io.EOF}
	err := stack.Wrap(fmt.Errorf("mcp: crm__whoami: %w", sending))

	s.Equal(`mcp: crm__whoami: Post "https://crm.example": EOF`, providerMessage(err))
}

// TestAURLTheSDKFlattenedIntoTextKeepsOnlyItsHost: the MCP SDK wraps a failed reconnect with
// %v (go-sdk v1.8.0 mcp/streamable.go:2174 and :2560), so no *url.Error is left to find.
func (s *ProviderMessageSuite) TestAURLTheSDKFlattenedIntoTextKeepsOnlyItsHost() {
	sending := &url.Error{Op: "Get", URL: "https://crm.example/mcp?access_token=" + queryToken, Err: io.EOF}
	err := fmt.Errorf("standalone SSE request failed (session ID: s1): %v", sending)

	s.Equal(`standalone SSE request failed (session ID: s1): Get "https://crm.example": EOF`, providerMessage(err))
}

func (s *ProviderMessageSuite) TestAMessageOverOneKiBIsCutAtARuneBoundary() {
	// 1023 bytes, then a 3-byte rune across the limit: the cut leaves the rune out whole.
	err := &core.ToolError{Message: strings.Repeat("a", providerMessageBytes-1) + "€ and more"}

	message := providerMessage(err)

	s.Equal(strings.Repeat("a", providerMessageBytes-1), message)
	s.True(utf8.ValidString(message))
}

func (s *ProviderMessageSuite) TestAMessageOfOneKiBIsKeptWhole() {
	text := strings.Repeat("a", providerMessageBytes-3) + "€"

	s.Equal(text, providerMessage(&core.ToolError{Message: text}))
}
