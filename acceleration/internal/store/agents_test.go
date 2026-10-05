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
	created := s.boundConfig("acme-app", fixed("connection-1"),
		ConnectorBinding{Name: "inbox", ConnectorID: "acme", Connection: ConnectionBinding{Type: "session"}, Tools: []ToolGrant{}})

	read, err := s.store.AgentConfig(s.ctx, "acme-app", created.ID)
	s.Require().NoError(err)
	s.Equal(created.Connectors, read.Connectors)
}

func (s *StoreSuite) TestAConfigWithoutBindingsStoresAnEmptyList() {
	created := s.boundConfig("acme-app")

	var stored string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT connectors::text FROM agent_configs WHERE id = ?", created.ID).Scan(&stored))
	s.Equal("[]", stored)
}

func (s *StoreSuite) TestUpdatingAConfigReplacesItsBindings() {
	config := s.boundConfig("acme-app", fixed("connection-1"))

	config.Connectors = []ConnectorBinding{fixed("connection-2")}
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, &config))

	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Require().Len(read.Connectors, 1)
	s.Equal("connection-2", read.Connectors[0].Connection.ConnectionID)
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
