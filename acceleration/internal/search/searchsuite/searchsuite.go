//go:build integration

// Package searchsuite is what every search provider is held to against its real API: a
// question comes back answered out of sources that say where they came from, as many as
// were asked for, and a question with nothing in it is never sent.
//
// A provider suite embeds Suite, says how to build its provider and which of the query's
// narrowing options it honours, and inherits those tests. A provider that also reads
// pages, by implementing search.Reader, is held to that too. Anything only one provider
// does stays in that provider's own file.
package searchsuite

import (
	"context"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// defaultTimeout bounds one search or read. It is looser than the providers' own, which
// are set for somebody waiting on the phone, so a slow answer fails on the provider's
// terms rather than the suite's.
const defaultTimeout = 30 * time.Second

// Suite is the shared behaviour. The fields are set where the suite is constructed rather
// than in a SetupSuite of the provider's own, which would shadow this one.
type Suite struct {
	suite.Suite

	// New builds a provider configured for an ordinary search.
	New func() search.Provider
	// Requires are the environment variables without which the provider cannot be
	// reached, and whose absence skips rather than fails.
	Requires []string

	// Answers marks a provider that writes a summary of what it found, rather than
	// leaving the sources to speak for themselves.
	Answers bool
	// NarrowsByDomain marks a provider that honours IncludeDomains and ExcludeDomains.
	NarrowsByDomain bool
	// Timeout bounds one call.
	Timeout time.Duration

	// Provider is what every test in the suite searches with. It is built once and never
	// closed: a search provider holds no connection of its own.
	Provider search.Provider
}

func (s *Suite) SetupSuite() {
	s.Require().NotNil(s.New, "a provider suite has to say how to build its provider")
	for _, name := range s.Requires {
		if os.Getenv(name) == "" {
			s.T().Skipf("%s not set", name)
		}
	}
	if s.Timeout == 0 {
		s.Timeout = defaultTimeout
	}
	s.Provider = s.New()
}

// Search runs one query on the suite's provider and fails the test unless it succeeds.
func (s *Suite) Search(query search.Query) search.Result {
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	defer cancel()

	found, err := s.Provider.Search(ctx, query)
	s.Require().NoError(err)
	return found
}

// Said is everything a result gives the model to read, in lower case: the answer, and
// each source's title and text.
func (s *Suite) Said(found search.Result) string {
	parts := []string{found.Answer}
	for _, document := range found.Documents {
		parts = append(parts, document.Title, document.Text)
	}
	return strings.ToLower(strings.Join(parts, "\n"))
}

// Hosts are where the result's sources came from, and fails the test on a source whose
// URL does not say.
func (s *Suite) Hosts(found search.Result) []string {
	var hosts []string
	for _, document := range found.Documents {
		parsed, err := url.Parse(document.URL)
		s.Require().NoError(err)
		s.Require().NotEmptyf(parsed.Host, "%q names no host", document.URL)
		hosts = append(hosts, parsed.Host)
	}
	return hosts
}
