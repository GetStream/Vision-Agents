package voices

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// ResolverSuite covers a voice that is not the customer's own, which needs no database:
// the resolver hands it to the provider's library to check. A customer's own voices are
// in ServiceSuite.
type ResolverSuite struct {
	suite.Suite
	ctx      context.Context
	library  *findingLister
	resolver *Resolver
}

func TestResolverSuite(t *testing.T) {
	suite.Run(t, new(ResolverSuite))
}

func (s *ResolverSuite) SetupTest() {
	s.ctx = context.Background()
	s.library = &findingLister{countingLister: countingLister{voices: []Library{sarah}}, exists: map[string]bool{}}
	catalogue := NewCatalogue()
	catalogue.Register("elevenlabs", s.library)
	s.resolver = NewResolver(nil, catalogue)
}

func (s *ResolverSuite) TestAVoiceTheProviderDoesNotHaveIsRefusedAsTheCallersMistake() {
	_, err := s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", "Sara")

	s.Require().ErrorIs(err, routing.ErrRequest, "the router moves on rather than marking the provider down")
	s.Contains(err.Error(), `"Sarah - Mature, Reassuring, Confident" (EXAVITQu4vr4xnSDxMaL)`)
}

func (s *ResolverSuite) TestAVoiceTheProviderHasIsPassedThroughAsAsked() {
	resolved, err := s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", sarah.ID)

	s.Require().NoError(err)
	s.Equal(sarah.ID, resolved)
}

func (s *ResolverSuite) TestAVoiceAskedForAsTheirOwnIsNeverLookedUpInALibrary() {
	_, err := s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", options.OwnVoicePrefix+"Rachel")

	// What the resolver said before it had a library.
	s.EqualError(err, `voices: "Rachel" is not one of this customer's voices`)
	s.Equal(0, s.library.lookups)
}
