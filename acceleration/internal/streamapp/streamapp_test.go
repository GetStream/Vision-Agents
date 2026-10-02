package streamapp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"
)

type StreamAppSuite struct {
	suite.Suite
}

func TestStreamAppSuite(t *testing.T) {
	suite.Run(t, new(StreamAppSuite))
}

// counting is a Source that says how often it was asked.
type counting struct {
	Source
	asked atomic.Int32
}

func (c *counting) For(ctx context.Context, customer string) (Identity, error) {
	c.asked.Add(1)
	return c.Source.For(ctx, customer)
}

// clock is a settable time.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func deployment() *Deployment {
	return NewDeployment(DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret"})
}

func (s *StreamAppSuite) TestEveryCustomerResolvesToTheDeploymentAppInDeploymentMode() {
	source := deployment()

	for _, customer := range []string{"acme", "globex"} {
		identity, err := source.For(context.Background(), customer)
		s.Require().NoError(err)
		s.Equal("deploy-key", identity.APIKey)
		s.Equal(customer, identity.CustomerID)
	}
}

func (s *StreamAppSuite) TestDeploymentModeWritesNoPin() {
	// Nothing written before apps had identities records one, so what deployment mode
	// writes must read the same way.
	identity, err := deployment().For(context.Background(), "acme")

	s.Require().NoError(err)
	s.Zero(identity.StreamApp)
}

func (s *StreamAppSuite) TestTheDeploymentIdentityCarriesTheUserToken() {
	source := NewDeployment(DeploymentOptions{APIKey: "k", Secret: "s", UserToken: "user-token"})

	identity, err := source.For(context.Background(), "acme")

	s.Require().NoError(err)
	s.Equal("user-token", identity.UserToken)
}

func (s *StreamAppSuite) TestADeploymentWithoutCredentialsHasNoIdentity() {
	_, err := NewDeployment(DeploymentOptions{}).For(context.Background(), "acme")

	s.ErrorIs(err, ErrNoIdentity)
}

func (s *StreamAppSuite) TestWorkPinnedToTheDeploymentAppIsFinishedThere() {
	source := NewDeployment(DeploymentOptions{APIKey: "k", Secret: "s", App: 42})

	unpinned, err := source.ForApp(context.Background(), "acme", 0)
	s.Require().NoError(err)
	s.Equal("k", unpinned.APIKey)
	pinned, err := source.ForApp(context.Background(), "acme", 42)
	s.Require().NoError(err)
	s.Equal("k", pinned.APIKey)

	_, err = source.ForApp(context.Background(), "acme", 7)
	s.ErrorIs(err, ErrStreamAppMoved, "work written in another app is never finished in this one")
}

func (s *StreamAppSuite) TestAPinWaitsWhileTheDeploymentAppIsUnknown() {
	source := deployment()

	_, err := source.ForApp(context.Background(), "acme", 42)
	s.ErrorIs(err, ErrDeploymentAppUnknown)

	source.SetApp(42)
	_, err = source.ForApp(context.Background(), "acme", 42)
	s.NoError(err)
}

func (s *StreamAppSuite) TestAStaticSourceFinishesWorkOnlyInItsPinnedApp() {
	source := NewStatic(map[string]Identity{
		"acme": {StreamApp: 7, APIKey: "acme-key", Secret: NewSecret("acme-secret")},
	}, deployment())

	acme, err := source.For(context.Background(), "acme")
	s.Require().NoError(err)
	s.Equal("acme-key", acme.APIKey)
	s.Equal(int64(7), acme.StreamApp)
	legacy, err := source.ForApp(context.Background(), "acme", 0)
	s.Require().NoError(err)
	s.Equal("deploy-key", legacy.APIKey, "work written before acme had an app stays in the deployment's")
	globex, err := source.For(context.Background(), "globex")
	s.Require().NoError(err)
	s.Equal("deploy-key", globex.APIKey)
}

func (s *StreamAppSuite) TestOneClientIsBuiltPerAppAndSecret() {
	clients := NewClients(deployment(), ClientsOptions{})

	acme, err := clients.For(context.Background(), "acme")
	s.Require().NoError(err)
	globex, err := clients.For(context.Background(), "globex")
	s.Require().NoError(err)

	s.Same(acme.Client, globex.Client, "two customers in one app share its client")
	s.Equal("globex", globex.Identity.CustomerID)
}

func (s *StreamAppSuite) TestANewSecretBuildsANewClient() {
	clients := NewClients(NewStatic(map[string]Identity{
		"acme":   {StreamApp: 7, APIKey: "key", Secret: NewSecret("old")},
		"globex": {StreamApp: 7, APIKey: "key", Secret: NewSecret("new")},
	}, nil), ClientsOptions{})

	old, err := clients.For(context.Background(), "acme")
	s.Require().NoError(err)
	rotated, err := clients.For(context.Background(), "globex")
	s.Require().NoError(err)

	s.NotSame(old.Client, rotated.Client)
}

func (s *StreamAppSuite) TestAnIdentityIsReusedUntilItExpires() {
	source := &counting{Source: deployment()}
	at := &clock{now: time.Date(2026, 10, 1, 9, 0, 0, 0, time.UTC)}
	clients := NewClients(source, ClientsOptions{Now: at.Now})

	for range 3 {
		_, err := clients.For(context.Background(), "acme")
		s.Require().NoError(err)
	}
	s.Equal(int32(1), source.asked.Load())

	at.Advance(identityTTL + time.Second)
	_, err := clients.For(context.Background(), "acme")
	s.Require().NoError(err)
	s.Equal(int32(2), source.asked.Load(), "a rotated or revoked credential bites within the TTL")
}

func (s *StreamAppSuite) TestInvalidateMakesTheNextResolutionReadTheSource() {
	source := &counting{Source: deployment()}
	clients := NewClients(source, ClientsOptions{})
	_, err := clients.For(context.Background(), "acme")
	s.Require().NoError(err)

	clients.Invalidate("acme")
	_, err = clients.For(context.Background(), "acme")
	s.Require().NoError(err)

	s.Equal(int32(2), source.asked.Load())
}

func (s *StreamAppSuite) TestConcurrentFirstUseSharesOneClient() {
	clients := NewClients(deployment(), ClientsOptions{})
	got := make([]*getstream.Stream, 32)
	var wg sync.WaitGroup
	for i := range got {
		wg.Go(func() {
			bound, err := clients.For(context.Background(), fmt.Sprintf("customer-%d", i%4))
			s.NoError(err)
			got[i] = bound.Client
		})
	}
	wg.Wait()

	for _, client := range got {
		s.Same(got[0], client)
	}
}

func (s *StreamAppSuite) TestARefreshAfterA401IsRateLimited() {
	// A credential refused a moment ago and refused again is wrong, not stale, and asking
	// the source over and over does not make it right.
	at := &clock{now: time.Date(2026, 10, 1, 9, 0, 0, 0, time.UTC)}
	clients := NewClients(deployment(), ClientsOptions{Now: at.Now})
	bound, err := clients.For(context.Background(), "acme")
	s.Require().NoError(err)

	s.True(clients.Rejected(bound.Identity))
	s.False(clients.Rejected(bound.Identity))

	at.Advance(refreshEvery + time.Second)
	s.True(clients.Rejected(bound.Identity))
}

func (s *StreamAppSuite) TestAClientIgnoresStreamBaseURLInTheEnvironment() {
	// The SDK reads STREAM_BASE_URL itself. Left to it, every app's client would talk to
	// wherever the deployment's environment points.
	var deploymentHits, appHits atomic.Int32
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		deploymentHits.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	s.T().Cleanup(elsewhere.Close)
	own := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		appHits.Add(1)
		_, _ = w.Write([]byte(`{}`))
	}))
	s.T().Cleanup(own.Close)
	s.T().Setenv("STREAM_BASE_URL", elsewhere.URL)
	clients := NewClients(NewStatic(map[string]Identity{
		"acme": {StreamApp: 7, APIKey: "key", Secret: NewSecret("secret"), BaseURL: own.URL},
	}, nil), ClientsOptions{})

	bound, err := clients.For(context.Background(), "acme")
	s.Require().NoError(err)
	_, err = bound.Client.GetApp(context.Background(), &getstream.GetAppRequest{})
	s.Require().NoError(err)

	s.Equal(int32(1), appHits.Load())
	s.Zero(deploymentHits.Load())
}

func (s *StreamAppSuite) TestAnIdentityNeverPrintsItsSecret() {
	identity := Identity{CustomerID: "acme", APIKey: "key", Secret: NewSecret("hunter2")}

	for _, printed := range []string{
		fmt.Sprintf("%v", identity), fmt.Sprintf("%+v", identity), fmt.Sprintf("%#v", identity),
		fmt.Sprintf("%s", identity.Secret), fmt.Sprintf("%q", identity.Secret), fmt.Sprintf("%x", identity.Secret),
	} {
		s.NotContains(printed, "hunter2")
	}
	encoded, err := json.Marshal(identity)
	s.Require().NoError(err)
	s.NotContains(string(encoded), "hunter2")
	var logged bytes.Buffer
	slog.New(slog.NewTextHandler(&logged, nil)).Info("resolved", "identity", identity, "secret", identity.Secret)
	s.NotContains(logged.String(), "hunter2")
	s.Equal("hunter2", identity.Secret.Reveal())
}
