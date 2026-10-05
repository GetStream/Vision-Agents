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
		"an anonymous caller may make it": {"anonymous": {"create-distinct-channel-for-others"}},
	} {
		s.Run(name, func() {
			s.stream.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{AgentChannelType: grants}})
			s.cache = NewClients(s.source, ClientsOptions{Now: s.clock.Now})

			s.Equal(TypeUnsafe, s.read().ChannelType)
		})
	}
}

func (s *ReadinessSuite) TestTheAppsOwnStaffMayDoAnything() {
	// A moderator or an admin is a role only the app's backend gives, which is how Stream's
	// own default grants have them.
	s.stream.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{AgentChannelType: {
		"admin":             {"create-channel", "update-channel", "add-own-channel-membership"},
		"moderator":         {"create-channel", "update-channel", "update-channel-members"},
		"channel_moderator": {"update-channel", "update-channel-members"},
		"global_admin":      {"create-channel-any-team", "update-channel-any-team"},
		"global_moderator":  {"update-channel-any-team"},
	}}})

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

func (s *ReadinessSuite) TestARefusalFromStreamIsKeptForTheMinute() {
	s.stream.SetApp(chattest.App{ID: 4242, Refuses: true})
	bound, err := s.cache.For(s.ctx, "acme")
	s.Require().NoError(err)
	_, err = s.cache.Readiness(s.ctx, bound)
	s.Require().Error(err)

	s.stream.SetApp(chattest.App{ID: 4242})
	_, err = s.cache.Readiness(s.ctx, bound)

	s.Error(err, "Stream's own answer is kept until the minute is up")
	s.Equal(1, s.stream.AppReads())
}

func (s *ReadinessSuite) TestAReadItsCallerAbandonedIsNotKept() {
	bound, err := s.cache.For(s.ctx, "acme")
	s.Require().NoError(err)
	ended, cancel := context.WithCancel(s.ctx)
	cancel()
	_, err = s.cache.Readiness(ended, bound)
	s.Require().ErrorIs(err, context.Canceled)

	readiness, err := s.cache.Readiness(s.ctx, bound)

	s.Require().NoError(err, "the next caller asks Stream rather than getting the cancellation")
	s.Equal(TypePresent, readiness.ChannelType)
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
	// In deployment mode new work carries on, and a pin naming the configured id is parked
	// rather than finished with a key that belongs to another app. Waiting would be waiting
	// for an id that can no longer be learned.
	s.stream.SetApp(chattest.App{ID: 1})
	source := NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: s.stream.URL, App: 99})
	cache := NewClients(source, ClientsOptions{})

	_, err := cache.LearnDeploymentApp(s.ctx)

	s.ErrorIs(err, ErrDeploymentAppMismatch)
	s.Zero(cache.DeploymentApp())
	_, err = source.ForApp(s.ctx, "acme", 99)
	s.ErrorIs(err, ErrStreamAppMoved)
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
