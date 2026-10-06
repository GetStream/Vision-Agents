//go:build integration

package store

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// oauthClient puts a customer client for the connector, with a sealed secret unless change
// says otherwise, and returns it as stored.
func (s *StoreSuite) oauthClient(customerID, connectorID string, change func(*ConnectorOAuthClient)) ConnectorOAuthClient {
	client := &ConnectorOAuthClient{
		CustomerID:   customerID,
		ConnectorID:  connectorID,
		Registration: core.ClientCustomer,
		ClientID:     "client-" + newID(),
		AuthMethod:   core.AuthClientSecretBasic,
		SecretSealed: []byte("sealed secret"),
		KEKVersion:   1,
	}
	if change != nil {
		change(client)
	}
	_, err := s.store.PutConnectorOAuthClient(s.ctx, client)
	s.Require().NoError(err)
	return *client
}

func (s *StoreSuite) TestPuttingAnOAuthClientAgainReplacesItsOneRow() {
	first := s.oauthClient("acme-app", "github", nil)

	rotated := &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "github", Registration: core.ClientCustomer,
		ClientID: first.ClientID, AuthMethod: core.AuthClientSecretPost,
		SecretSealed: []byte("rotated secret"), KEKVersion: 2,
	}
	created, err := s.store.PutConnectorOAuthClient(s.ctx, rotated)

	s.Require().NoError(err)
	s.False(created)
	found, err := s.store.ConnectorOAuthClient(s.ctx, "acme-app", "github")
	s.Require().NoError(err)
	s.Equal([]byte("rotated secret"), found.SecretSealed)
	s.Equal(2, found.KEKVersion)
	s.Equal(core.AuthClientSecretPost, found.AuthMethod)
	s.True(first.CreatedAt.Equal(found.CreatedAt), "created once")
	s.True(found.UpdatedAt.After(first.UpdatedAt))
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connector_oauth_clients").Scan(&rows))
	s.Equal(1, rows)
}

func (s *StoreSuite) TestTheFirstPutOfAnOAuthClientCreatesIt() {
	client := &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "github", Registration: core.ClientCustomer, ClientID: "public",
		AuthMethod: core.AuthNone,
	}

	created, err := s.store.PutConnectorOAuthClient(s.ctx, client)

	s.Require().NoError(err)
	s.True(created)
	found, err := s.store.ConnectorOAuthClient(s.ctx, "acme-app", "github")
	s.Require().NoError(err)
	s.Equal("public", found.ClientID)
	s.Empty(found.SecretSealed, "a public client has no secret")
	s.Zero(found.KEKVersion)
}

func (s *StoreSuite) TestAnOAuthClientIsOnlyItsOwnCustomers() {
	s.oauthClient("acme-app", "github", nil)

	_, err := s.store.ConnectorOAuthClient(s.ctx, "other-app", "github")
	s.ErrorIs(err, ErrNoConnectorOAuthClient)
	s.ErrorIs(s.store.DeleteConnectorOAuthClient(s.ctx, "other-app", "github", core.ClientCustomer), ErrNoConnectorOAuthClient)
	_, err = s.store.ConnectorOAuthClient(s.ctx, "acme-app", "github")
	s.NoError(err, "the other customer's delete left it")
}

func (s *StoreSuite) TestAnOAuthClientIsPerConnector() {
	s.oauthClient("acme-app", "github", nil)

	_, err := s.store.ConnectorOAuthClient(s.ctx, "acme-app", "salesforce")

	s.ErrorIs(err, ErrNoConnectorOAuthClient)
}

func (s *StoreSuite) TestDeletingAnOAuthClientRemovesIt() {
	s.oauthClient("acme-app", "github", nil)

	s.Require().NoError(s.store.DeleteConnectorOAuthClient(s.ctx, "acme-app", "github", core.ClientCustomer))

	_, err := s.store.ConnectorOAuthClient(s.ctx, "acme-app", "github")
	s.ErrorIs(err, ErrNoConnectorOAuthClient)
	s.ErrorIs(s.store.DeleteConnectorOAuthClient(s.ctx, "acme-app", "github", core.ClientCustomer), ErrNoConnectorOAuthClient)
}

func (s *StoreSuite) TestACustomerClientNeitherReplacesNorDeletesTheOperatorsOne() {
	operator := s.oauthClient("acme-app", "slack", func(c *ConnectorOAuthClient) { c.Registration = core.ClientOperator })

	_, err := s.store.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "slack", Registration: core.ClientCustomer, ClientID: "the-apps-own",
	})
	s.ErrorIs(err, ErrOAuthClientRegistration)
	s.ErrorIs(s.store.DeleteConnectorOAuthClient(s.ctx, "acme-app", "slack", core.ClientCustomer), ErrNoConnectorOAuthClient)

	found, err := s.store.ConnectorOAuthClient(s.ctx, "acme-app", "slack")
	s.Require().NoError(err)
	s.Equal(core.ClientOperator, found.Registration)
	s.Equal(operator.ClientID, found.ClientID)
}
