package session

import (
	"context"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// callableSuite tests the callable skip conditions in isolation: each skip must prevent the
// check from running, so that a voice session outside the check's scope is not refused.
// These are not integration tests: they call callable directly to avoid the full Create
// path and to set spec.CallType and spec.Edge without going through the API.

// withCallableManager builds a manager that only has Stream configured, the minimum
// callable needs to reach its Readiness call.
func (s *SessionSuite) withCallableManager(server *chattest.Server, key string) *Manager {
	source := streamapp.NewStatic(map[string]streamapp.Identity{
		"acme": {
			CustomerID: "acme", StreamApp: 1, APIKey: key,
			Secret: streamapp.NewSecret(key + "-secret"), BaseURL: server.URL,
			Registered: true,
		},
	}, streamapp.NewDeployment(streamapp.DeploymentOptions{APIKey: "deploy-key", Secret: "deploy-secret"}))
	clients := streamapp.NewClients(source, streamapp.ClientsOptions{})
	return &Manager{options: ManagerOptions{Stream: clients}, logger: slog.New(slog.DiscardHandler)}
}

// callableBound returns a Bound for "acme" via the manager's Clients, confirming the
// identity is Registered so the Readiness path is reachable.
func (s *SessionSuite) callableBound(m *Manager) streamapp.Bound {
	bound, err := m.options.Stream.For(context.Background(), "acme")
	s.Require().NoError(err)
	s.True(bound.Identity.Registered, "test precondition: the bound must be Registered so the check would run without the skip")
	return bound
}

func (s *SessionSuite) TestCallable_SkipsForNonAgentCallType() {
	// MD: spec.CallType != AgentCallType skips callable, so a voice session joining a call
	// of another type (phone, default) is not refused even if the app has no agent call type.
	server := chattest.NewServer(s.T())
	server.SetApp(chattest.App{ID: 1, ChannelTypes: map[string]map[string][]string{"agent": {}}})
	m := s.withCallableManager(server, "acme-key")
	bound := s.callableBound(m)

	err := m.callable(context.Background(), Spec{CallType: "default"}, bound)

	s.NoError(err, "a non-agent call type must not be refused for missing agent call type")
}

func (s *SessionSuite) TestCallable_SkipsForUnregisteredIdentity() {
	// MC: !Registered skips callable, so a session acting in a customer's own app but
	// without a forwarded registration (deployment mode) is not refused.
	server := chattest.NewServer(s.T())
	server.SetApp(chattest.App{ID: 1, ChannelTypes: map[string]map[string][]string{"agent": {}}})
	m := s.withCallableManager(server, "acme-key")
	bound, err := m.options.Stream.For(context.Background(), "acme")
	s.Require().NoError(err)
	// Override Registered to false to model the unregistered case.
	bound.Identity.Registered = false
	s.Require().NotNil(bound.Client, "test precondition: Client must be non-nil so the skip is on Registered alone")

	err = m.callable(context.Background(), Spec{CallType: streamapp.AgentCallType}, bound)

	s.NoError(err, "an unregistered identity must not be refused for missing agent call type")
}

func (s *SessionSuite) TestCallable_SkipsForSessionWithEdge() {
	// ME: spec.Edge != nil skips callable, so socket sessions and audio simulations that
	// supply their own edge are not refused even if the app has no agent call type.
	server := chattest.NewServer(s.T())
	server.SetApp(chattest.App{ID: 1, ChannelTypes: map[string]map[string][]string{"agent": {}}})
	m := s.withCallableManager(server, "acme-key")
	bound := s.callableBound(m)

	err := m.callable(context.Background(), Spec{CallType: streamapp.AgentCallType, Edge: newQuietEdge()}, bound)

	s.NoError(err, "a session with its own edge must not be refused for missing agent call type")
}
