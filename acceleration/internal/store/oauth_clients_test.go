//go:build integration

package store

import (
	"errors"
	"fmt"
	"sync"
	"testing/fstest"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// everyPreregistration is a client.registration listing every registration a record can have.
const everyPreregistration = "customer, managed, operator"

// oauthConnector seeds a built-in connector id whose client.registration is registrations,
// as a router start with that file would. Seeding it again with the same list changes nothing.
func (s *StoreSuite) oauthConnector(id, registrations string) {
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, fstest.MapFS{id + ".yaml": {Data: []byte(fmt.Sprintf(`
id: %s
revision: 1
name: Acme
category: Testing
endpoints:
  mcp: https://mcp.acme.example/mcp
schemes: [oauth2_code]
client:
  registration: [%s]
sources:
  - kind: mcp
    endpoint: mcp
`, id, registrations))}}))
}

// oauthClient puts a customer client for the connector, with a sealed secret unless change
// says otherwise, and returns it as stored. The connector is seeded taking every registration.
func (s *StoreSuite) oauthClient(customerID, connectorID string, change func(*ConnectorOAuthClient)) ConnectorOAuthClient {
	s.oauthConnector(connectorID, everyPreregistration)
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
	s.oauthConnector("github", everyPreregistration)
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

// managedClient puts the app the router created for the customer at the connector, with its
// provider app and a sealed signing secret, and returns it as stored.
func (s *StoreSuite) managedClient(customerID, connectorID, providerAppID string) ConnectorOAuthClient {
	return s.oauthClient(customerID, connectorID, func(c *ConnectorOAuthClient) {
		c.Registration = core.ClientManaged
		c.ProviderAppID = providerAppID
		c.SigningSecretSealed = []byte("sealed signing secret of " + providerAppID)
		c.SigningKEKVersion = 1
	})
}

func (s *StoreSuite) TestAManagedClientKeepsItsProviderAppAndSigningSecret() {
	put := s.managedClient("acme-app", "acme_chat", "A012ABCD0A0")

	found, err := s.store.ConnectorOAuthClient(s.ctx, "acme-app", "acme_chat")

	s.Require().NoError(err)
	s.Equal(core.ClientManaged, found.Registration)
	s.Equal("A012ABCD0A0", found.ProviderAppID)
	s.Equal([]byte("sealed signing secret of A012ABCD0A0"), found.SigningSecretSealed)
	s.Equal(1, found.SigningKEKVersion)
	s.Equal(put.ClientID, found.ClientID)
}

func (s *StoreSuite) TestAConnectorListingOnlyTheOperatorRefusesAManagedClient() {
	s.oauthConnector("acme_operator_only", "operator")

	_, err := s.store.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "acme_operator_only", Registration: core.ClientManaged,
		ClientID: "created", ProviderAppID: "A012ABCD0A0",
	})

	s.ErrorIs(err, ErrOAuthClientRegistrationNotListed)
	_, err = s.store.ConnectorOAuthClient(s.ctx, "acme-app", "acme_operator_only")
	s.ErrorIs(err, ErrNoConnectorOAuthClient, "nothing was stored")
}

func (s *StoreSuite) TestAnUnknownConnectorTakesNoOAuthClient() {
	_, err := s.store.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "acme_nowhere", Registration: core.ClientCustomer, ClientID: "id",
	})

	s.ErrorIs(err, ErrNoConnectorDefinition)
}

func (s *StoreSuite) TestASecondRecordForTheSameAppAndConnectorIsRefused() {
	managed := s.managedClient("acme-app", "acme_chat", "A012ABCD0A0")

	_, err := s.store.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "acme_chat", Registration: core.ClientCustomer, ClientID: "the-apps-own",
	})

	s.ErrorIs(err, ErrOAuthClientRegistration)
	found, err := s.store.ConnectorOAuthClient(s.ctx, "acme-app", "acme_chat")
	s.Require().NoError(err)
	s.Equal(managed.ClientID, found.ClientID)
	s.Equal("A012ABCD0A0", found.ProviderAppID)
}

// Each writer is a router of its own with a pool of its own, so the inserts overlap in
// Postgres: one record is stored, and the other writer is told whose registration it is.
func (s *StoreSuite) TestTwoRegistrationsRacingForOneAppAndConnectorStoreOneRecord() {
	s.oauthConnector("acme_chat", everyPreregistration)
	registrations := []core.ClientRegistrationMethod{core.ClientManaged, core.ClientCustomer, core.ClientOperator}
	routers := make([]*Store, len(registrations))
	for i := range routers {
		routers[i] = s.router()
	}

	errs := make([]error, len(routers))
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, router := range routers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, errs[i] = router.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{
				CustomerID: "acme-app", ConnectorID: "acme_chat", Registration: registrations[i],
				ClientID: "client-" + string(registrations[i]), ProviderAppID: "A0" + string(registrations[i]),
			})
		}()
	}
	close(start)
	wg.Wait()

	won := 0
	for _, err := range errs {
		if err == nil {
			won++
			continue
		}
		s.True(errors.Is(err, ErrOAuthClientRegistration), "a losing writer is told the record is another registration's: %v", err)
	}
	s.Equal(1, won)
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connector_oauth_clients").Scan(&rows))
	s.Equal(1, rows)
}

func (s *StoreSuite) TestAProviderAppIsFoundOnlyForTheCustomerWhoseAppItIs() {
	s.managedClient("acme-app", "acme_chat", "A0ACME")
	s.managedClient("globex-app", "acme_chat", "A0GLOBEX")

	acme, err := s.store.ConnectorOAuthClientByProviderApp(s.ctx, "acme_chat", "A0ACME")
	s.Require().NoError(err)
	globex, err := s.store.ConnectorOAuthClientByProviderApp(s.ctx, "acme_chat", "A0GLOBEX")
	s.Require().NoError(err)

	s.Equal("acme-app", acme.CustomerID)
	s.Equal([]byte("sealed signing secret of A0ACME"), acme.SigningSecretSealed)
	s.Equal("globex-app", globex.CustomerID)
	s.Equal([]byte("sealed signing secret of A0GLOBEX"), globex.SigningSecretSealed)
	_, err = s.store.ConnectorOAuthClientByProviderApp(s.ctx, "acme_chat", "A0INITECH")
	s.ErrorIs(err, ErrNoConnectorOAuthClient)
	_, err = s.store.ConnectorOAuthClientByProviderApp(s.ctx, "acme_other", "A0ACME")
	s.ErrorIs(err, ErrNoConnectorOAuthClient, "an app id is the connector's")
}

func (s *StoreSuite) TestAProviderAppBelongsToOneCustomer() {
	s.managedClient("acme-app", "acme_chat", "A012ABCD0A0")

	_, err := s.store.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{
		CustomerID: "globex-app", ConnectorID: "acme_chat", Registration: core.ClientManaged,
		ClientID: "globex-client", ProviderAppID: "A012ABCD0A0",
	})

	s.ErrorIs(err, ErrProviderAppTaken)
	found, err := s.store.ConnectorOAuthClientByProviderApp(s.ctx, "acme_chat", "A012ABCD0A0")
	s.Require().NoError(err)
	s.Equal("acme-app", found.CustomerID)
}

// As TestTwoRegistrationsRacingForOneAppAndConnectorStoreOneRecord: a router and a pool per
// customer, all naming one app at once, and the unique index lets one of them have it.
func (s *StoreSuite) TestCustomersRacingForOneProviderAppStoreOneRecord() {
	s.oauthConnector("acme_chat", everyPreregistration)
	const customers = 8
	routers := make([]*Store, customers)
	for i := range routers {
		routers[i] = s.router()
	}

	errs := make([]error, customers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, router := range routers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, errs[i] = router.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{
				CustomerID: fmt.Sprintf("app-%d", i), ConnectorID: "acme_chat", Registration: core.ClientManaged,
				ClientID: fmt.Sprintf("client-%d", i), ProviderAppID: "A012ABCD0A0",
			})
		}()
	}
	close(start)
	wg.Wait()

	won := 0
	for _, err := range errs {
		if err == nil {
			won++
			continue
		}
		s.True(errors.Is(err, ErrProviderAppTaken), "a losing customer is told the app is taken: %v", err)
	}
	s.Equal(1, won)
}

func (s *StoreSuite) TestOneProviderAppIdAtTwoConnectorsIsTwoApps() {
	s.managedClient("acme-app", "acme_chat", "A012ABCD0A0")

	s.managedClient("globex-app", "acme_other", "A012ABCD0A0")

	found, err := s.store.ConnectorOAuthClientByProviderApp(s.ctx, "acme_other", "A012ABCD0A0")
	s.Require().NoError(err)
	s.Equal("globex-app", found.CustomerID)
}

func (s *StoreSuite) TestRecordsWithoutAProviderAppDoNotCollide() {
	s.oauthClient("acme-app", "acme_chat", nil)
	s.oauthClient("globex-app", "acme_chat", nil)

	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connector_oauth_clients WHERE provider_app_id = ''").Scan(&rows))
	s.Equal(2, rows)
}

// Acme was in Stream app 4242 when the router created its Slack app, then registered app 5555:
// the provider app's work still goes to 4242, where its threads were made.
func (s *StoreSuite) TestARecordKeepsTheStreamAppItWasCreatedIn() {
	s.oauthClient("acme-app", "acme_chat", func(c *ConnectorOAuthClient) {
		c.Registration, c.ProviderAppID, c.StreamAppPK = core.ClientManaged, "A012ABCD0A0", 4242
	})

	rotated := s.oauthClient("acme-app", "acme_chat", func(c *ConnectorOAuthClient) {
		c.Registration, c.ProviderAppID, c.StreamAppPK = core.ClientManaged, "A012ABCD0A0", 5555
	})

	s.Equal(int64(4242), rotated.StreamAppPK, "the put hands back the pin the record has")
	found, err := s.store.ConnectorOAuthClientByProviderApp(s.ctx, "acme_chat", "A012ABCD0A0")
	s.Require().NoError(err)
	s.Equal(int64(4242), found.StreamAppPK)
}

func (s *StoreSuite) TestARecordMadeInTheDeploymentsAppHasNoPin() {
	s.oauthClient("acme-app", "acme_chat", nil)

	var pinned bool
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT stream_app_pk IS NOT NULL FROM connector_oauth_clients WHERE customer_id = 'acme-app'").Scan(&pinned))
	s.False(pinned, "NULL is the deployment's own app, as on every other pinned table")
}

func (s *StoreSuite) TestARewrapOfAnOAuthClientAppliesOnlyToTheSecretItRead() {
	put := s.managedClient("acme-app", "acme_chat", "A012ABCD0A0")

	applied, err := s.store.RewrapConnectorOAuthClientSigningSecret(s.ctx, "acme-app", "acme_chat", put.SigningSecretSealed, []byte("signing under v2"), 2)
	s.Require().NoError(err)
	s.True(applied)
	stale, err := s.store.RewrapConnectorOAuthClientSigningSecret(s.ctx, "acme-app", "acme_chat", put.SigningSecretSealed, []byte("from a stale read"), 3)
	s.Require().NoError(err)
	s.False(stale, "the secret it read was replaced meanwhile")
	applied, err = s.store.RewrapConnectorOAuthClientSecret(s.ctx, "acme-app", "acme_chat", put.SecretSealed, []byte("client under v2"), 2)
	s.Require().NoError(err)
	s.True(applied)

	found, err := s.store.ConnectorOAuthClient(s.ctx, "acme-app", "acme_chat")
	s.Require().NoError(err)
	s.Equal([]byte("signing under v2"), found.SigningSecretSealed)
	s.Equal(2, found.SigningKEKVersion)
	s.Equal([]byte("client under v2"), found.SecretSealed)
	s.Equal(2, found.KEKVersion)
}
