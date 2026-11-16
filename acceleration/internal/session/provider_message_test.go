package session

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

// ProviderMessageSuite is what a "connector call failed" line says of the body a call's 4xx or
// 5xx answered (providerBody): its start without a cut rune, with no URL's path or query.
type ProviderMessageSuite struct {
	suite.Suite
}

func TestProviderMessageSuite(t *testing.T) {
	suite.Run(t, new(ProviderMessageSuite))
}

// The synthetic token a URL in these tests carries.
const queryToken = "query-token-must-not-print"

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
