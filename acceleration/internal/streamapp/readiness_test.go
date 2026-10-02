package streamapp

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

// ReadinessSuite reads what an app holds from a Stream app in memory.
type ReadinessSuite struct {
	suite.Suite
	ctx    context.Context
	stream *chattest.Server
	clock  *clock
	source *Deployment
	cache  *Clients
}

func TestReadinessSuite(t *testing.T) {
	suite.Run(t, new(ReadinessSuite))
}

func (s *ReadinessSuite) SetupTest() {
	s.ctx = context.Background()
	s.stream = chattest.NewServer(s.T())
	s.clock = &clock{now: time.Unix(1_700_000_000, 0)}
	s.source = NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: s.stream.URL})
	s.cache = NewClients(s.source, ClientsOptions{Now: s.clock.Now})
}

func (s *ReadinessSuite) read() Readiness {
	bound, err := s.cache.For(s.ctx, "acme")
	s.Require().NoError(err)
	readiness, err := s.cache.Readiness(s.ctx, bound)
	s.Require().NoError(err)
	return readiness
}

func (s *ReadinessSuite) TestAnAppSetUpForTheRouterIsReady() {
	s.stream.SetApp(chattest.App{
		ID:           4242,
		ChannelTypes: map[string]map[string][]string{AgentChannelType: {"channel_member": {"read-channel", "create-message"}}},
		CallTypes:    []string{AgentCallType},
	})

	readiness := s.read()

	s.Equal(int64(4242), readiness.App)
	s.Equal(TypePresent, readiness.ChannelType)
	s.Equal(TypePresent, readiness.CallType)
	s.Equal(s.clock.Now(), readiness.CheckedAt)
}

func (s *ReadinessSuite) TestAnAppWithoutTheAgentTypesSaysWhatIsMissing() {
	s.stream.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{"messaging": {}}})

	readiness := s.read()

	s.Equal(TypeMissing, readiness.ChannelType)
	s.Equal(TypeMissing, readiness.CallType)
}

func (s *ReadinessSuite) TestAChannelTypeAClientCanForgeIsUnsafe() {
	for name, grants := range map[string]map[string][]string{
		"a user may make one":             {"user": {"create-channel"}},
		"an owner may change one":         {"user": {"update-channel-owner"}},
		"a member may change its data":    {"channel_member": {"update-channel"}},
		"anybody may join one":            {"user": {"add-own-channel-membership"}},
		"a guest may read one":            {"guest": {"read-channel"}},
		"a moderator may change one":      {"channel_moderator": {"update-channel-members"}},
		"an anonymous caller may make it": {"anonymous": {"create-distinct-channel-for-others"}},
	} {
		s.Run(name, func() {
			s.stream.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{AgentChannelType: grants}})
			s.cache = NewClients(s.source, ClientsOptions{Now: s.clock.Now})

			s.Equal(TypeUnsafe, s.read().ChannelType)
		})
	}
}

func (s *ReadinessSuite) TestAnAdminMayDoAnything() {
	s.stream.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{
		AgentChannelType: {"admin": {"create-channel", "update-channel", "add-own-channel-membership"}},
	}})

	s.Equal(TypePresent, s.read().ChannelType)
}

func (s *ReadinessSuite) TestReadinessIsAskedOfStreamAtMostOnceAMinute() {
	s.read()
	s.read()
	s.Equal(1, s.stream.AppReads())

	s.clock.Advance(readinessTTL)
	s.read()

	s.Equal(2, s.stream.AppReads())
}

func (s *ReadinessSuite) TestTheDeploymentLearnsItsOwnAppFromStream() {
	s.stream.SetApp(chattest.App{ID: 1234})

	learned, err := s.cache.LearnDeploymentApp(s.ctx)

	s.Require().NoError(err)
	s.Equal(int64(1234), learned)
	s.Equal(int64(1234), s.cache.DeploymentApp())
}

func (s *ReadinessSuite) TestAConfiguredDeploymentAppIsCheckedOnce() {
	s.stream.SetApp(chattest.App{ID: 99})
	source := NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: s.stream.URL, App: 99})
	cache := NewClients(source, ClientsOptions{})

	learned, err := cache.LearnDeploymentApp(s.ctx)
	s.Require().NoError(err)
	_, err = cache.LearnDeploymentApp(s.ctx)
	s.Require().NoError(err)

	s.Equal(int64(99), learned)
	s.Equal(1, s.stream.AppReads())
}

func (s *ReadinessSuite) TestAMismatchedDeploymentAppStopsNamingPins() {
	// In deployment mode new work carries on, and a pin naming the configured id waits
	// rather than being finished with a key that belongs to another app.
	s.stream.SetApp(chattest.App{ID: 1})
	source := NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: s.stream.URL, App: 99})
	cache := NewClients(source, ClientsOptions{})

	_, err := cache.LearnDeploymentApp(s.ctx)

	s.ErrorIs(err, ErrDeploymentAppMismatch)
	s.Zero(cache.DeploymentApp())
	_, err = source.ForApp(s.ctx, "acme", 99)
	s.ErrorIs(err, ErrDeploymentAppUnknown)
	_, err = source.For(s.ctx, "acme")
	s.NoError(err)
}

func (s *ReadinessSuite) TestAStrictDeploymentStopsAnsweringOnAMismatch() {
	s.stream.SetApp(chattest.App{ID: 1})
	source := NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: s.stream.URL, App: 99, Strict: true})

	_, err := NewClients(source, ClientsOptions{}).LearnDeploymentApp(s.ctx)

	s.ErrorIs(err, ErrDeploymentAppMismatch)
	_, err = source.For(s.ctx, "acme")
	s.ErrorIs(err, ErrDeploymentAppMismatch)
}

func (s *ReadinessSuite) TestAnUnreachableStreamLeavesTheDeploymentAppUnknown() {
	closed := httptest.NewServer(nil)
	closed.Close()
	source := NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: closed.URL})
	cache := NewClients(source, ClientsOptions{})

	_, err := cache.LearnDeploymentApp(s.ctx)

	s.Require().Error(err)
	s.Zero(cache.DeploymentApp())
	bound, err := cache.For(s.ctx, "acme")
	s.Require().NoError(err, "new work still goes to the deployment's app")
	s.Equal("deploy-key", bound.Identity.APIKey)
}

func (s *ReadinessSuite) TestASourceWithoutADeploymentAppHasNothingToLearn() {
	_, err := NewClients(NewStatic(map[string]Identity{"acme": {StreamApp: 7}}, nil), ClientsOptions{}).LearnDeploymentApp(s.ctx)

	s.ErrorIs(err, ErrNoIdentity)
}
