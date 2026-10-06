//go:build integration

package store

import "strings"

// boundConfig stores a config of the customer's carrying bindings.
func (s *StoreSuite) boundConfig(customerID string, bindings ...ConnectorBinding) AgentConfig {
	config := &AgentConfig{CustomerID: customerID, Name: "agent-" + newID(), Connectors: bindings}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))
	return *config
}

func fixed(connectionID string) ConnectorBinding {
	return ConnectorBinding{
		Name: "crm", ConnectorID: "acme",
		Connection: ConnectionBinding{Type: "fixed", ConnectionID: connectionID},
		Tools:      []ToolGrant{{Name: "search", SchemaDigest: strings.Repeat("a", 64)}},
		Required:   true, TimeoutMs: 5000,
	}
}

func (s *StoreSuite) TestAConfigsBindingsAreReadBackAsTheyWereStored() {
	created := s.boundConfig("acme-app", fixed(s.connection("acme-app", nil).ID),
		ConnectorBinding{Name: "inbox", ConnectorID: "acme", Connection: ConnectionBinding{Type: "session"}, Tools: []ToolGrant{}})

	read, err := s.store.AgentConfig(s.ctx, "acme-app", created.ID)
	s.Require().NoError(err)
	s.Equal(created.Connectors, read.Connectors)
}

func (s *StoreSuite) TestAConfigCannotBindAConnectionThatIsNotLive() {
	deleted := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", deleted.ID))
	others := s.connection("other-app", nil)

	for _, id := range []string{deleted.ID, others.ID, newID()} {
		err := s.store.CreateAgentConfig(s.ctx, &AgentConfig{CustomerID: "acme-app", Name: "bound",
			Connectors: []ConnectorBinding{fixed(id)}})
		s.ErrorIs(err, ErrNoConnectorConnection, "deleted, another customer's, never made")
	}
	configs, err := s.store.CustomerAgentConfigs(s.ctx, "acme-app")
	s.Require().NoError(err)
	s.Empty(configs)
}

func (s *StoreSuite) TestAnUpdateCannotNewlyBindADeletedConnection() {
	deleted := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", deleted.ID))
	config := s.boundConfig("acme-app")

	config.Connectors = []ConnectorBinding{fixed(deleted.ID)}
	s.ErrorIs(s.store.UpdateAgentConfig(s.ctx, &config), ErrNoConnectorConnection)

	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Empty(read.Connectors)
}

// A forced delete leaves the binding behind on purpose, and saving the config for anything
// else keeps it rather than failing until it is removed.
func (s *StoreSuite) TestAnUpdateKeepsABindingAForcedDeleteLeftBehind() {
	connection := s.connection("acme-app", nil)
	config := s.boundConfig("acme-app", fixed(connection.ID))
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	config.Instructions = "be brief"
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, &config))

	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Equal("be brief", read.Instructions)
	s.Equal([]ConnectorBinding{fixed(connection.ID)}, read.Connectors)
}

func (s *StoreSuite) TestAConfigWithoutBindingsStoresAnEmptyList() {
	created := s.boundConfig("acme-app")

	var stored string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT connectors::text FROM agent_configs WHERE id = ?", created.ID).Scan(&stored))
	s.Equal("[]", stored)
}

func (s *StoreSuite) TestUpdatingAConfigReplacesItsBindings() {
	config := s.boundConfig("acme-app", fixed(s.connection("acme-app", nil).ID))
	replacement := s.connection("acme-app", nil)

	config.Connectors = []ConnectorBinding{fixed(replacement.ID)}
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, &config))

	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Require().Len(read.Connectors, 1)
	s.Equal(replacement.ID, read.Connectors[0].Connection.ConnectionID)
}

// The containment query in ConnectorConnectionReferenced reads the JSON this model writes,
// so a change to its tags shows here rather than as a connection deleted out from under an
// agent.
func (s *StoreSuite) TestAFixedBindingAConfigStoresReferencesItsConnection() {
	connection := s.connection("acme-app", nil)
	s.boundConfig("acme-app", fixed(connection.ID))

	referenced, err := s.store.ConnectorConnectionReferenced(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.True(referenced)
}
