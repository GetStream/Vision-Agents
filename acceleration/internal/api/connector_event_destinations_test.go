//go:build integration

package api

import (
	"context"
	"crypto/tls"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorEventDestinationsSuite is the endpoints a backend manages a connector's event
// destinations with: create, list, delete and rotate a secret.
type ConnectorEventDestinationsSuite struct {
	RouterSuite
}

func TestConnectorEventDestinationsSuite(t *testing.T) {
	runSuite(t, new(ConnectorEventDestinationsSuite))
}

func (s *ConnectorEventDestinationsSuite) SetupSuite() {
	s.forwardHTTP = destinationClient()
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *ConnectorEventDestinationsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConnectorEventDestinationsSuite) TestOnlyTheAppsBackendManagesEventDestinations() {
	target := newDestination(s.T())

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, destinationsPath("slack_bot"), ConnectorEventDestinationRequest{URL: target.URL, Forward: store.ForwardAll}, nil)
	})
}

func (s *ConnectorEventDestinationsSuite) TestTheSecretIsShownOnCreateAndNeverListed() {
	target := newDestination(s.T())

	created := s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)

	s.True(strings.HasPrefix(created.Secret, "whsec_"), "a Standard Webhooks secret")
	s.Equal(target.URL, created.Destination.URL)
	s.Equal(ConnectorEventForward(store.ForwardUnhandled), created.Destination.Forward)
	status, listed := s.serverClient.call(http.MethodGet, destinationsPath("slack_bot"), nil)
	s.Require().Equal(http.StatusOK, status)
	s.Contains(string(listed), created.Destination.ID)
	s.NotContains(string(listed), created.Secret)
}

// The Standard Webhooks spec warns that webhook URLs are a server-side request forgery
// surface («Server side request forgery (SSRF)», https://www.standardwebhooks.com/); egress
// refuses every address IANA marks not globally reachable.
func (s *ConnectorEventDestinationsSuite) TestADestinationOnAPrivateAddressIsRefusedAtCreate() {
	for _, url := range []string{"https://10.0.0.7/hooks/slack", "https://192.168.1.20/hooks/slack", "https://169.254.169.254/latest"} {
		status, answer := s.serverClient.call(http.MethodPost, destinationsPath("slack_bot"),
			ConnectorEventDestinationRequest{URL: url, Forward: store.ForwardAll})

		s.Equal(http.StatusBadRequest, status, url)
		s.Contains(string(answer), "public https URL", url)
	}
	s.Empty(s.listDestinations("slack_bot", "").Items)
}

func (s *ConnectorEventDestinationsSuite) TestAPlainHTTPDestinationIsRefused() {
	status, _ := s.serverClient.call(http.MethodPost, destinationsPath("slack_bot"),
		ConnectorEventDestinationRequest{URL: "http://hooks.example.com/slack", Forward: store.ForwardAll})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ConnectorEventDestinationsSuite) TestAConnectorWithoutProviderEventsTakesNoDestination() {
	status, answer := s.serverClient.call(http.MethodPost, destinationsPath("github"),
		ConnectorEventDestinationRequest{URL: newDestination(s.T()).URL, Forward: store.ForwardAll})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(answer), "takes no provider events")
}

func (s *ConnectorEventDestinationsSuite) TestAFourthDestinationIsAConflict() {
	for range store.MaxEventDestinations {
		s.createDestination("slack_bot", newDestination(s.T()).URL, store.ForwardAll)
	}

	status, _ := s.serverClient.call(http.MethodPost, destinationsPath("slack_bot"),
		ConnectorEventDestinationRequest{URL: newDestination(s.T()).URL, Forward: store.ForwardAll})

	s.Equal(http.StatusConflict, status)
}

func (s *ConnectorEventDestinationsSuite) TestDestinationsArePagedNewestFirst() {
	first := s.createDestination("slack_bot", newDestination(s.T()).URL, store.ForwardAll)
	second := s.createDestination("slack_bot", newDestination(s.T()).URL, store.ForwardAll)

	page := s.listDestinations("slack_bot", "?limit=1")
	s.Require().Len(page.Items, 1)
	s.Equal(second.Destination.ID, page.Items[0].ID)
	s.True(page.HasMore)
	next := s.listDestinations("slack_bot", "?limit=1&cursor="+*page.NextCursor)
	s.Require().Len(next.Items, 1)
	s.Equal(first.Destination.ID, next.Items[0].ID)
	s.False(next.HasMore)
}

func (s *ConnectorEventDestinationsSuite) TestACursorNotHandedOutIsRefused() {
	status, _ := s.serverClient.call(http.MethodGet, destinationsPath("slack_bot")+"?cursor=not-a-cursor", nil)

	s.Equal(http.StatusBadRequest, status)
}

func (s *ConnectorEventDestinationsSuite) TestADeletedDestinationIsGone() {
	created := s.createDestination("slack_bot", newDestination(s.T()).URL, store.ForwardAll)

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, destinationPath("slack_bot", created.Destination.ID), nil, nil))

	s.Empty(s.listDestinations("slack_bot", "").Items)
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodDelete, destinationPath("slack_bot", created.Destination.ID), nil, nil))
}

func (s *ConnectorEventDestinationsSuite) TestAnotherAppsDestinationIsNotFound() {
	created := s.createDestination("slack_bot", newDestination(s.T()).URL, store.ForwardAll)

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, destinationPath("slack_bot", created.Destination.ID), nil, nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, destinationPath("slack_bot", created.Destination.ID)+"/rotate-secret", nil, nil)
	})
	s.Len(s.listDestinations("slack_bot", "").Items, 1)
}

func (s *ConnectorEventDestinationsSuite) TestARotationGivesANewSecretAndKeepsTheOldOneSigningForADay() {
	created := s.createDestination("slack_bot", newDestination(s.T()).URL, store.ForwardAll)

	var rotated ConnectorEventDestinationSecret
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		destinationPath("slack_bot", created.Destination.ID)+"/rotate-secret", nil, &rotated))

	s.True(strings.HasPrefix(rotated.Secret, "whsec_"))
	s.NotEqual(created.Secret, rotated.Secret)
	s.Require().NotNil(rotated.Destination.PreviousSecretUntil)
	s.WithinDuration(created.Destination.CreatedAt.Add(24*time.Hour), *rotated.Destination.PreviousSecretUntil, time.Minute)
}

// createDestination makes a destination of the test's app through the API.
func (s *RouterSuite) createDestination(connector, url, forward string) ConnectorEventDestinationSecret {
	var created ConnectorEventDestinationSecret
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, destinationsPath(connector),
		ConnectorEventDestinationRequest{URL: url, Forward: ConnectorEventForward(forward)}, &created))
	return created
}

func (s *RouterSuite) listDestinations(connector, query string) ConnectorEventDestinationPage {
	var page ConnectorEventDestinationPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, destinationsPath(connector)+query, nil, &page))
	return page
}

func destinationsPath(connector string) string {
	return "/v1/agents/connectors/" + connector + "/event-destinations"
}

func destinationPath(connector, id string) string {
	return destinationsPath(connector) + "/" + id
}

// destinationClient is what the suite's forwarder sends through: it trusts the test
// destinations' own certificates, which are made up for each one.
func destinationClient() *http.Client {
	return &http.Client{Transport: &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //nolint:gosec // the test's own destinations
	}}
}

// destination is a customer's URL: a TLS server on loopback that keeps what reached it and
// answers with the statuses it is told, then 200.
type destination struct {
	*httptest.Server

	mu       sync.Mutex
	received []forwarded
	answers  []int
	// hold, when set, keeps every request waiting until it is closed.
	hold chan struct{}
}

// forwarded is one request a destination took.
type forwarded struct {
	header http.Header
	body   []byte
}

func newDestination(t *testing.T) *destination {
	d := &destination{}
	d.Server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		d.mu.Lock()
		d.received = append(d.received, forwarded{header: r.Header.Clone(), body: body})
		status := http.StatusOK
		if len(d.answers) > 0 {
			status, d.answers = d.answers[0], d.answers[1:]
		}
		hold := d.hold
		d.mu.Unlock()
		if hold != nil {
			<-hold
		}
		w.WriteHeader(status)
	}))
	d.URL += "/hooks/slack"
	t.Cleanup(d.Close)
	return d
}

// answer has the next requests answered with statuses, in order.
func (d *destination) answer(statuses ...int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.answers = append(d.answers, statuses...)
}

// holding keeps every request waiting until the returned function is called.
func (d *destination) holding() func() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.hold = make(chan struct{})
	release := d.hold
	return func() { close(release) }
}

// requests are what reached the destination, oldest first.
func (d *destination) requests() []forwarded {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]forwarded(nil), d.received...)
}
