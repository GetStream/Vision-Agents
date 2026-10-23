//go:build integration

package searchsuite

import (
	"context"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
)

// TestAQuestionIsAnsweredOutOfWhatIsOnTheWeb is the one that says the provider works at
// all: what comes back is something the model can say the answer from.
func (s *Suite) TestAQuestionIsAnsweredOutOfWhatIsOnTheWeb() {
	found := s.Search(austen)

	s.NotEmpty(found.Documents, "the answer should come with the sources behind it")
	s.Contains(s.Said(found), austenAnswer)
}

// TestEverySourceSaysWhereItCameFrom is what lets the agent say where it read something,
// and a source with nothing to read is a result the model cannot use.
func (s *Suite) TestEverySourceSaysWhereItCameFrom() {
	found := s.Search(austen)

	s.Require().NotEmpty(found.Documents)
	s.Len(s.Hosts(found), len(found.Documents))
	s.NotEmpty(strings.TrimSpace(found.Documents[0].Text), "the most relevant source has something to read")
}

// TestALimitCapsHowManySourcesComeBack is what keeps a voice agent from being handed a
// page of results to read out.
func (s *Suite) TestALimitCapsHowManySourcesComeBack() {
	query := austen
	query.Limit = 2

	found := s.Search(query)

	s.NotEmpty(found.Documents)
	s.LessOrEqual(len(found.Documents), 2)
}

// TestAQuestionWithNothingInItIsRefused is checked against the real provider because a
// search for nothing is billed like any other.
func (s *Suite) TestAQuestionWithNothingInItIsRefused() {
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	defer cancel()

	_, err := s.Provider.Search(ctx, search.Query{Text: "  "})

	s.ErrorContains(err, "nothing to look for")
}

// TestTheProviderWritesAnAnswerOfItsOwn is the sentence a voice agent wants: something to
// say, rather than sources to read.
func (s *Suite) TestTheProviderWritesAnAnswerOfItsOwn() {
	if !s.Answers {
		s.T().Skip("this provider leaves the sources to speak for themselves")
	}

	found := s.Search(austen)

	s.Contains(strings.ToLower(found.Answer), austenAnswer)
}

// TestIncludeDomainsKeepsTheSourcesToThoseDomains is what a business that trusts three
// sources and no others relies on.
func (s *Suite) TestIncludeDomainsKeepsTheSourcesToThoseDomains() {
	if !s.NarrowsByDomain {
		s.T().Skip("this provider does not narrow by domain")
	}
	query := austen
	query.IncludeDomains = []string{wikipedia}

	found := s.Search(query)

	s.Require().NotEmpty(found.Documents)
	for _, host := range s.Hosts(found) {
		s.Truef(strings.HasSuffix(host, wikipedia), "%s is not one of the domains asked for", host)
	}
}

// TestExcludeDomainsKeepsThoseDomainsOut is the other half: a source the business has
// ruled out never comes back.
func (s *Suite) TestExcludeDomainsKeepsThoseDomainsOut() {
	if !s.NarrowsByDomain {
		s.T().Skip("this provider does not narrow by domain")
	}
	query := austen
	query.ExcludeDomains = []string{wikipedia}

	found := s.Search(query)

	s.Require().NotEmpty(found.Documents)
	for _, host := range s.Hosts(found) {
		s.Falsef(strings.HasSuffix(host, wikipedia), "%s was ruled out", host)
	}
}

// TestAPageIsReadWhole is what fills a knowledge base from a URL somebody named.
func (s *Suite) TestAPageIsReadWhole() {
	reader, ok := s.Provider.(search.Reader)
	if !ok {
		s.T().Skip("this provider does not read pages")
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	defer cancel()

	page, err := reader.Read(ctx, examplePage)
	s.Require().NoError(err)

	s.Equal(examplePage, page.URL)
	s.Contains(strings.ToLower(page.Title+"\n"+page.Text), examplePageSays)
}
