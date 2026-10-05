//go:build integration

package store

import (
	"errors"
	"strings"
	"sync"
	"testing/fstest"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// seed runs the seeder over one built-in file, as a router start with that file would.
func (s *StoreSuite) seed(raw string) {
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, fstest.MapFS{"acme.yaml": {Data: []byte(raw)}}))
}

// revisions is every stored revision of a built-in, oldest first.
func (s *StoreSuite) revisions(id string) []ConnectorDefinition {
	var definitions []ConnectorDefinition
	s.Require().NoError(s.store.DB().NewSelect().Model(&definitions).
		Where("customer_id = ?", BuiltinCustomer).Where("id = ?", id).
		Order("revision").Scan(s.ctx))
	return definitions
}

func (s *StoreSuite) TestSeedingAnEmptyTableStoresEachShippedBuiltInAtTheRevisionItNames() {
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, providers.FS))
	shipped, err := builtinManifests(providers.FS)
	s.Require().NoError(err)

	listed, err := s.store.ListConnectorDefinitions(s.ctx, "acme", ConnectorDefinitionFilter{})
	s.Require().NoError(err)
	s.Require().Len(listed, 7)
	for i, id := range []string{"calcom", "calendly", "github", "gong", "linear", "salesforce", "slack"} {
		s.Equal(id, listed[i].ID)
		s.Equal(BuiltinCustomer, listed[i].CustomerID)
		s.Equal(shipped[i].Revision, listed[i].Revision, "stored at the revision its file names")
		s.Equal(shipped[i].Revision, listed[i].Manifest.Revision, "the stored manifest names the revision it is stored as")
		s.Equal(listed[i].Manifest.Name, listed[i].Name)
		s.NotEmpty(listed[i].Category)
		s.NotEmpty(listed[i].Description)
	}
}

func (s *StoreSuite) TestSeedingAnUnchangedManifestAddsNoRevision() {
	s.seed(acmeManifest)
	s.seed(acmeManifest)
	s.seed(acmeReformatted)

	stored := s.revisions("acme")
	s.Require().Len(stored, 1, "neither a restart nor a reformatted file is a change")
	s.Equal(1, stored[0].Revision)
}

func (s *StoreSuite) TestAChangedManifestAtANewRevisionIsTheLatestAndTheLastOneStays() {
	s.seed(acmeManifest)
	s.seed(acmeChanged)

	latest, err := s.store.LatestConnectorDefinition(s.ctx, "anyone", "acme")
	s.Require().NoError(err)
	s.Equal(2, latest.Revision)
	s.Equal(2, latest.Manifest.Revision)
	s.Equal([]string{"read"}, latest.Manifest.Scopes.List)

	first, err := s.store.ConnectorDefinition(s.ctx, "anyone", "acme", 1)
	s.Require().NoError(err)
	s.Equal([]string{"read", "write"}, first.Manifest.Scopes.List, "a connection pinned to revision 1 still reads what it was made from")
}

func (s *StoreSuite) TestTwoBuildsSeedingTheirOwnFilesDoNotFlipTheLatest() {
	// A rolling deploy, a rollback, or a branch build sharing staging's database with an
	// accelerate build: each build restarts with its own file, in any order.
	s.seed(acmeManifest)
	s.seed(acmeChanged)
	s.seed(acmeManifest)
	s.seed(acmeChanged)
	s.seed(acmeManifest)

	stored := s.revisions("acme")
	s.Require().Len(stored, 2, "each build finds its own revision stored")
	latest, err := s.store.LatestConnectorDefinition(s.ctx, "anyone", "acme")
	s.Require().NoError(err)
	s.Equal(2, latest.Revision, "an older build starting last does not make its manifest latest")
	s.Equal([]string{"read"}, latest.Manifest.Scopes.List)
}

func (s *StoreSuite) TestAnOlderBuildStartingAfterANewerOneLeavesTheNewerLatest() {
	// A fresh database where the new build's pods start first.
	s.seed(acmeChanged)
	s.seed(acmeManifest)

	latest, err := s.store.LatestConnectorDefinition(s.ctx, "anyone", "acme")
	s.Require().NoError(err)
	s.Equal(2, latest.Revision)
	first, err := s.store.ConnectorDefinition(s.ctx, "anyone", "acme", 1)
	s.Require().NoError(err, "the older build's revision is stored for the connections it makes")
	s.Equal([]string{"read", "write"}, first.Manifest.Scopes.List)
}

func (s *StoreSuite) TestAnEditWithoutANewRevisionIsRefused() {
	s.seed(acmeManifest)

	err := s.store.SeedConnectorDefinitions(s.ctx, fstest.MapFS{"acme.yaml": {Data: []byte(acmeEditedInPlace)}})

	s.ErrorContains(err, "acme.yaml says revision 1, which is already stored with other content")
	stored := s.revisions("acme")
	s.Require().Len(stored, 1)
	s.Equal([]string{"read", "write"}, stored[0].Manifest.Scopes.List, "a connection pinned to revision 1 keeps reading what it was made from")
}

func (s *StoreSuite) TestRevertingAManifestIsANewRevision() {
	s.seed(acmeManifest)
	s.seed(acmeChanged)
	s.seed(acmeReverted)

	latest, err := s.store.LatestConnectorDefinition(s.ctx, "anyone", "acme")
	s.Require().NoError(err)
	s.Equal(3, latest.Revision)
	s.Equal([]string{"read", "write"}, latest.Manifest.Scopes.List)
	s.Len(s.revisions("acme"), 3)
}

func (s *StoreSuite) TestTwoRoutersSeedingAChangeAtOnceStoreOneRevision() {
	s.seed(acmeManifest)

	// Each router has a pool of its own, connected before they all start at once, so the
	// seeders overlap rather than queue behind each other's dial.
	const routers = 8
	stores := make([]*Store, routers)
	for i := range stores {
		router, err := Open(s.dsn)
		s.Require().NoError(err)
		s.T().Cleanup(func() { router.Close() })
		s.Require().NoError(router.Ping(s.ctx))
		stores[i] = router
	}
	errs := make([]error, routers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, router := range stores {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errs[i] = router.SeedConnectorDefinitions(s.ctx, fstest.MapFS{"acme.yaml": {Data: []byte(acmeChanged)}})
		}()
	}
	close(start)
	wg.Wait()

	for _, err := range errs {
		s.NoError(err, "a router that loses the race waits and finds the change already stored")
	}
	stored := s.revisions("acme")
	s.Require().Len(stored, 2)
	s.Equal(2, stored[1].Revision)
}

func (s *StoreSuite) TestAnInvalidBuiltInStoresNoneOfTheBuiltIns() {
	broken := strings.Replace(acmeManifest, "id: acme", "id: broken", 1)
	broken = strings.Replace(broken, "schemes: [oauth2_code, test_key, test_mtls]", "schemes: []", 1)

	err := s.store.SeedConnectorDefinitions(s.ctx, fstest.MapFS{
		"acme.yaml":   {Data: []byte(acmeManifest)},
		"broken.yaml": {Data: []byte(broken)},
	})

	s.ErrorContains(err, "broken.yaml")
	s.ErrorContains(err, "schemes: is empty")
	s.Empty(s.revisions("acme"), "a valid file beside an invalid one is not stored either")
}

func (s *StoreSuite) TestACustomDefinitionIdStartsWithTheCustomPrefix() {
	_, err := s.store.CreateConnectorDefinition(s.ctx, "acme", parsed(s.T(), acmeManifest))

	s.ErrorContains(err, "a custom connector id starts with custom_")
}

func (s *StoreSuite) TestACustomDefinitionCannotShadowABuiltIn() {
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, providers.FS))
	shadow := parsed(s.T(), strings.Replace(acmeManifest, "id: acme", "id: slack", 1))

	_, err := s.store.CreateConnectorDefinition(s.ctx, "acme", shadow)
	s.Error(err)

	latest, err := s.store.LatestConnectorDefinition(s.ctx, "acme", "slack")
	s.Require().NoError(err)
	s.Equal(BuiltinCustomer, latest.CustomerID, "slack is still the built-in for this customer")
	s.Equal("Slack", latest.Name)
}

func (s *StoreSuite) TestNoCustomerCanCreateABuiltIn() {
	custom := parsed(s.T(), strings.Replace(acmeManifest, "id: acme", "id: custom_crm", 1))

	_, err := s.store.CreateConnectorDefinition(s.ctx, BuiltinCustomer, custom)

	s.ErrorContains(err, "customer id is required")
}

func (s *StoreSuite) TestCreatingACustomDefinitionAgainIsTheNextRevisionOnlyWhenItChanged() {
	custom := strings.Replace(acmeManifest, "id: acme", "id: custom_crm", 1)

	created, err := s.store.CreateConnectorDefinition(s.ctx, "acme", parsed(s.T(), custom))
	s.Require().NoError(err)
	s.Equal(1, created.Revision)
	s.Equal("acme", created.CustomerID)

	again, err := s.store.CreateConnectorDefinition(s.ctx, "acme", parsed(s.T(), custom))
	s.Require().NoError(err)
	s.Equal(1, again.Revision, "the same manifest again is the revision already stored")

	changed, err := s.store.CreateConnectorDefinition(s.ctx, "acme",
		parsed(s.T(), strings.Replace(acmeChanged, "id: acme", "id: custom_crm", 1)))
	s.Require().NoError(err)
	s.Equal(2, changed.Revision)
}

func (s *StoreSuite) TestACustomDefinitionIsOnlyItsOwnCustomers() {
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, providers.FS))
	_, err := s.store.CreateConnectorDefinition(s.ctx, "acme",
		parsed(s.T(), strings.Replace(acmeManifest, "id: acme", "id: custom_crm", 1)))
	s.Require().NoError(err)

	_, err = s.store.ConnectorDefinition(s.ctx, "globex", "custom_crm", 1)
	s.True(errors.Is(err, ErrNoConnectorDefinition), "fetched by id and revision: %v", err)
	_, err = s.store.LatestConnectorDefinition(s.ctx, "globex", "custom_crm")
	s.True(errors.Is(err, ErrNoConnectorDefinition), "fetched as the latest: %v", err)

	theirs, err := s.store.ListConnectorDefinitions(s.ctx, "globex", ConnectorDefinitionFilter{})
	s.Require().NoError(err)
	s.Equal([]string{"calcom", "calendly", "github", "gong", "linear", "salesforce", "slack"}, definitionIDs(theirs), "another customer sees the built-ins alone")

	ours, err := s.store.ListConnectorDefinitions(s.ctx, "acme", ConnectorDefinitionFilter{})
	s.Require().NoError(err)
	s.Equal([]string{"calcom", "calendly", "github", "gong", "linear", "salesforce", "slack", "custom_crm"}, definitionIDs(ours), "built-ins first, then the customer's own")
}

func (s *StoreSuite) TestAnUnknownRevisionIsNoDefinition() {
	s.seed(acmeManifest)

	_, err := s.store.ConnectorDefinition(s.ctx, "acme", "acme", 2)

	s.True(errors.Is(err, ErrNoConnectorDefinition), "got %v", err)
}

func definitionIDs(definitions []ConnectorDefinition) []string {
	ids := make([]string, 0, len(definitions))
	for _, definition := range definitions {
		ids = append(ids, definition.ID)
	}
	return ids
}
