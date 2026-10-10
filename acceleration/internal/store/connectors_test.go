//go:build integration

package store

import (
	"errors"
	"io/fs"
	"strings"
	"sync"
	"testing/fstest"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
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

// AI-863: linq.yaml is a new file, so a router that ships it, starting on a database the
// build before it seeded, adds linq at revision 1 and leaves every other built-in's latest
// revision as it was.
func (s *StoreSuite) TestSeedingLinqLeavesEveryOtherBuiltInsLatestRevisionAsItWas() {
	s.seedingANewBuiltInLeavesTheOthersAsTheyWere("linq")
}

// AI-881: telnyx.yaml is a new file too.
func (s *StoreSuite) TestSeedingTelnyxLeavesEveryOtherBuiltInsLatestRevisionAsItWas() {
	s.seedingANewBuiltInLeavesTheOthersAsTheyWere("telnyx")
}

// AI-879: whatsapp.yaml is a new file too.
func (s *StoreSuite) TestSeedingWhatsAppLeavesEveryOtherBuiltInsLatestRevisionAsItWas() {
	s.seedingANewBuiltInLeavesTheOthersAsTheyWere("whatsapp")
}

// seedingANewBuiltInLeavesTheOthersAsTheyWere seeds every shipped built-in but id, as the
// build before id's file did, then every one, and checks id lands at revision 1 and no other
// built-in's latest revision moves.
func (s *StoreSuite) seedingANewBuiltInLeavesTheOthersAsTheyWere(id string) {
	before := fstest.MapFS{}
	files, err := fs.ReadDir(providers.FS, ".")
	s.Require().NoError(err)
	for _, file := range files {
		if file.Name() == id+".yaml" {
			continue
		}
		raw, err := fs.ReadFile(providers.FS, file.Name())
		s.Require().NoError(err)
		before[file.Name()] = &fstest.MapFile{Data: raw}
	}
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, before))
	latest := func() map[string]int {
		revisions := map[string]int{}
		for name := range before {
			definition, err := s.store.LatestBuiltinConnectorDefinition(s.ctx, strings.TrimSuffix(name, ".yaml"))
			s.Require().NoError(err)
			revisions[definition.ID] = definition.Revision
		}
		return revisions
	}
	was := latest()

	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, providers.FS))

	s.Equal(was, latest())
	added, err := s.store.LatestBuiltinConnectorDefinition(s.ctx, id)
	s.Require().NoError(err)
	s.Equal(1, added.Revision)
}

// AI-816: slack.yaml at revision 5 reads the user from $.user_id. A router that ships it,
// starting on a database the build before it seeded with revision 4, stores 5 as the latest,
// keeps 4 for the connections pinned to it, and moves no other built-in.
// TestSlackRevisionSixSeedsOverTheRevisionFiveTheBuildBeforeStored: revision 6 says what 5
// says and marks 1 to 4 broken, which is a change, so it is a new revision rather than an
// edit the seeder refuses. The marks are stored with it, and every other built-in stays.
func (s *StoreSuite) TestSlackRevisionSixSeedsOverTheRevisionFiveTheBuildBeforeStored() {
	before := fstest.MapFS{}
	files, err := fs.ReadDir(providers.FS, ".")
	s.Require().NoError(err)
	for _, file := range files {
		raw, err := fs.ReadFile(providers.FS, file.Name())
		s.Require().NoError(err)
		before[file.Name()] = &fstest.MapFile{Data: raw}
	}
	// Revision 5 is this file with its revision put back and without the marks.
	five, marks, found := strings.Cut(string(before["slack.yaml"].Data), "\nbroken_revisions:\n")
	s.Require().True(found)
	_, rest, found := strings.Cut(marks, "\n# [proto]:3-5\n")
	s.Require().True(found)
	five = strings.Replace(five, "\nrevision: 6\n", "\nrevision: 5\n", 1) + "\n# [proto]:3-5\n" + rest
	before["slack.yaml"] = &fstest.MapFile{Data: []byte(five)}
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, before))
	latest := func() map[string]int {
		revisions := map[string]int{}
		for name := range before {
			definition, err := s.store.LatestBuiltinConnectorDefinition(s.ctx, strings.TrimSuffix(name, ".yaml"))
			s.Require().NoError(err)
			revisions[definition.ID] = definition.Revision
		}
		return revisions
	}
	was := latest()
	s.Require().Equal(5, was["slack"])
	_, broken, err := s.store.BrokenConnectorRevision(s.ctx, "slack", 4)
	s.Require().NoError(err)
	s.Require().False(broken, "revision 5 marks nothing")

	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, providers.FS), "the marks are a new revision, not refused")

	now := latest()
	s.Equal(6, now["slack"])
	delete(was, "slack")
	delete(now, "slack")
	s.Equal(was, now)
	for revision, want := range map[int]bool{1: true, 2: true, 3: true, 4: true, 5: false, 6: false} {
		reason, broken, err := s.store.BrokenConnectorRevision(s.ctx, "slack", revision)
		s.Require().NoError(err)
		s.Equal(want, broken, "revision %d", revision)
		if want {
			s.Equal("user_id read from $.authed_user.id; live Slack sends top-level user_id", reason)
		}
	}
}

// acmeMarking is acme at revision 3, marking revisions 1 and 2 broken.
var acmeMarking = strings.Replace(acmeManifest, "revision: 1", "revision: 3", 1) +
	"broken_revisions:\n  - revisions: [1, 2]\n    reason: the first reason\n"

func (s *StoreSuite) TestSeedingABuiltInStoresTheRevisionsItMarksBroken() {
	s.seed(acmeManifest)
	s.seed(acmeMarking)

	for revision, want := range map[int]bool{1: true, 2: true, 3: false} {
		reason, broken, err := s.store.BrokenConnectorRevision(s.ctx, "acme", revision)
		s.Require().NoError(err)
		s.Equal(want, broken, "revision %d", revision)
		if want {
			s.Equal("the first reason", reason)
		}
	}
	var stored []ConnectorBrokenRevision
	s.Require().NoError(s.store.DB().NewSelect().Model(&stored).Order("revision").Scan(s.ctx))
	s.Require().Len(stored, 2)
	s.Equal(3, stored[0].MarkedBy)
	s.seed(acmeMarking)
	marks, err := s.store.DB().NewSelect().Model((*ConnectorBrokenRevision)(nil)).Count(s.ctx)
	s.Require().NoError(err)
	s.Equal(2, marks, "a restart stores no mark twice")
}

// TestAMarkStaysWhenALaterRevisionDropsItOrGivesAnotherReason: marks only accumulate, so a
// manifest reverted to one without them is a new revision that unmarks nothing.
func (s *StoreSuite) TestAMarkStaysWhenALaterRevisionDropsItOrGivesAnotherReason() {
	s.seed(acmeMarking)
	s.seed(strings.Replace(acmeManifest, "revision: 1", "revision: 4", 1))
	s.seed(strings.Replace(acmeManifest, "revision: 1", "revision: 5", 1) +
		"broken_revisions:\n  - revisions: [1]\n    reason: another reason\n")

	reason, broken, err := s.store.BrokenConnectorRevision(s.ctx, "acme", 1)
	s.Require().NoError(err)
	s.True(broken)
	s.Equal("the first reason", reason)
	_, broken, err = s.store.BrokenConnectorRevision(s.ctx, "acme", 2)
	s.Require().NoError(err)
	s.True(broken)
}

// TestARestartRestoresAMarkThatWasMissingFromAnAlreadyStoredRevision: the revision is stored
// already, as when an earlier build stored it, so the seeder finds it unchanged and still
// records the marks it names.
func (s *StoreSuite) TestARestartRestoresAMarkThatWasMissingFromAnAlreadyStoredRevision() {
	s.seed(acmeManifest)
	s.seed(acmeMarking)
	_, err := s.store.DB().NewDelete().Model((*ConnectorBrokenRevision)(nil)).Where("connector_id = ?", "acme").Exec(s.ctx)
	s.Require().NoError(err)
	_, broken, err := s.store.BrokenConnectorRevision(s.ctx, "acme", 1)
	s.Require().NoError(err)
	s.Require().False(broken)

	s.seed(acmeMarking)

	for _, revision := range []int{1, 2} {
		reason, broken, err := s.store.BrokenConnectorRevision(s.ctx, "acme", revision)
		s.Require().NoError(err)
		s.True(broken, "revision %d", revision)
		s.Equal("the first reason", reason)
	}
}

// TestAnotherConnectorsMarkDoesNotBreakAConnectionOnTheSameRevisionNumber: marks are per
// connector, so acme marking its revision 1 broken leaves a connection on another connector's
// revision 1 current, in the same listing as acme's own broken one.
func (s *StoreSuite) TestAnotherConnectorsMarkDoesNotBreakAConnectionOnTheSameRevisionNumber() {
	s.seed(acmeManifest)
	s.seed(acmeMarking)
	other := strings.Replace(acmeManifest, "id: acme", "id: other", 1)
	s.Require().Contains(other, "id: other")
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, fstest.MapFS{"other.yaml": {Data: []byte(other)}}))
	mine, theirs := appConnection(), appConnection()
	theirs.ConnectorID = "other"
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, testSchemes, mine))
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, testSchemes, theirs))

	statuses, err := s.store.ConnectorDefinitionStatuses(s.ctx, "acme-app", []ConnectorConnection{*mine, *theirs})
	s.Require().NoError(err)

	s.Equal(map[string]DefinitionStatus{
		mine.ID:   {Status: DefinitionBroken, Reason: "the first reason"},
		theirs.ID: {Status: DefinitionCurrent},
	}, statuses)
}

func (s *StoreSuite) TestAConnectionsDefinitionIsCurrentOutdatedOrBroken() {
	s.seed(acmeManifest)
	s.seed(acmeChanged)
	at := func(revision int) ConnectorConnection {
		connection := appConnection()
		connection.DefinitionRevision = revision
		s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, testSchemes, connection))
		return *connection
	}
	one, two := at(1), at(2)

	statuses, err := s.store.ConnectorDefinitionStatuses(s.ctx, "acme-app", []ConnectorConnection{one, two})
	s.Require().NoError(err)
	s.Equal(map[string]DefinitionStatus{one.ID: {Status: DefinitionOutdated}, two.ID: {Status: DefinitionCurrent}}, statuses)

	s.seed(acmeMarking)
	three := at(3)
	statuses, err = s.store.ConnectorDefinitionStatuses(s.ctx, "acme-app", []ConnectorConnection{one, two, three})
	s.Require().NoError(err)
	s.Equal(map[string]DefinitionStatus{
		one.ID:   {Status: DefinitionBroken, Reason: "the first reason"},
		two.ID:   {Status: DefinitionBroken, Reason: "the first reason"},
		three.ID: {Status: DefinitionCurrent},
	}, statuses)
}

func (s *StoreSuite) TestSeedingAnEmptyTableStoresEachShippedBuiltInAtTheRevisionItNames() {
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx, providers.FS))
	shipped, err := builtinManifests(providers.FS)
	s.Require().NoError(err)

	listed, err := s.store.ListConnectorDefinitions(s.ctx, "acme", ConnectorDefinitionFilter{})
	s.Require().NoError(err)
	s.Require().Len(listed, 18)
	for i, id := range []string{"calcom", "calendly", "github", "gmail", "gong", "google_calendar", "google_docs", "google_drive", "hubspot", "linear", "linq", "salesforce", "sentry", "shopify", "slack", "slack_bot", "telnyx", "whatsapp"} {
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
	s.Equal([]string{"calcom", "calendly", "github", "gmail", "gong", "google_calendar", "google_docs", "google_drive", "hubspot", "linear", "linq", "salesforce", "sentry", "shopify", "slack", "slack_bot", "telnyx", "whatsapp"}, definitionIDs(theirs), "another customer sees the built-ins alone")

	ours, err := s.store.ListConnectorDefinitions(s.ctx, "acme", ConnectorDefinitionFilter{})
	s.Require().NoError(err)
	s.Equal([]string{"calcom", "calendly", "github", "gmail", "gong", "google_calendar", "google_docs", "google_drive", "hubspot", "linear", "linq", "salesforce", "sentry", "shopify", "slack", "slack_bot", "telnyx", "whatsapp", "custom_crm"}, definitionIDs(ours), "built-ins first, then the customer's own")
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

// customCRM is a custom definition that takes the customer's own OAuth client and the
// test_key scheme testSchemes registers.
const customCRM = `
id: custom_crm
revision: 1
name: CRM
endpoints:
  mcp: https://mcp.crm.example/mcp
schemes: [oauth2_code, test_key]
client:
  registration: [customer]
sources:
  - kind: mcp
    endpoint: mcp
`

// crm stores custom_crm as the customer's own and returns its id.
func (s *StoreSuite) crm(customerID string) string {
	_, err := s.store.CreateConnectorDefinition(s.ctx, customerID, parsed(s.T(), customCRM))
	s.Require().NoError(err)
	return "custom_crm"
}

// crmConnection is an app-owned connection of the customer's to custom_crm.
func (s *StoreSuite) crmConnection(customerID string) ConnectorConnection {
	connection := &ConnectorConnection{CustomerID: customerID, ConnectorID: "custom_crm", DefinitionRevision: 1,
		OwnerType: OwnerApp, AuthScheme: "test_key"}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, testSchemes, connection))
	return *connection
}

// crmBinding is a binding of custom_crm under alias, fixed to connectionID or, when it is
// empty, chosen per session.
func crmBinding(alias, connectionID string) ConnectorBinding {
	binding := ConnectorBinding{Name: alias, ConnectorID: "custom_crm", Connection: ConnectionBinding{Type: "session"}, Tools: []ToolGrant{}}
	if connectionID != "" {
		binding.Connection = ConnectionBinding{Type: "fixed", ConnectionID: connectionID}
	}
	return binding
}

// secretsOf counts the customer's rows naming the connector in each table that holds a sealed
// secret for it.
func (s *StoreSuite) secretsOf(customerID, connectorID string) map[string]int {
	counts := map[string]int{}
	for _, table := range []string{"connector_oauth_clients", "connector_config_tokens", "connector_event_destinations"} {
		var count int
		s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
			"SELECT count(*) FROM "+table+" WHERE customer_id = ? AND connector_id = ?", customerID, connectorID).Scan(&count))
		counts[table] = count
	}
	return counts
}

func (s *StoreSuite) TestAnUnusedCustomDefinitionGoesWithEveryRevisionAndMayBeMadeAgain() {
	id := s.crm("acme-app")
	_, err := s.store.CreateConnectorDefinition(s.ctx, "acme-app",
		parsed(s.T(), strings.Replace(customCRM, "name: CRM", "name: Our CRM", 1)))
	s.Require().NoError(err)

	deleted, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", id, false)

	s.Require().NoError(err)
	s.Empty(deleted.Connections)
	for _, revision := range []int{1, 2} {
		_, err = s.store.ConnectorDefinition(s.ctx, "acme-app", id, revision)
		s.ErrorIs(err, ErrNoConnectorDefinition, "revision %d", revision)
	}
	again, err := s.store.CreateConnectorDefinition(s.ctx, "acme-app", parsed(s.T(), customCRM))
	s.Require().NoError(err)
	s.Equal(1, again.Revision, "a new connector, numbered from the start")
}

func (s *StoreSuite) TestABuiltInOrAnotherCustomersDefinitionIsNotDeleted() {
	s.seed(acmeManifest)
	s.crm("other-app")

	_, builtin := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", "acme", true)
	_, others := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", "custom_crm", true)

	s.ErrorIs(builtin, ErrNoConnectorDefinition)
	s.ErrorIs(others, ErrNoConnectorDefinition)
	s.Len(s.revisions("acme"), 1)
	_, err := s.store.LatestConnectorDefinition(s.ctx, "other-app", "custom_crm")
	s.NoError(err, "the other customer's is still there")
}

func (s *StoreSuite) TestAnUnforcedDeleteOfAUsedDefinitionNamesItsUsersAndChangesNothing() {
	id := s.crm("acme-app")
	connection := s.crmConnection("acme-app")
	user := s.crmConnection("acme-app")
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE connector_connections SET owner_type = 'user', owner_id = 'u1' WHERE id = ?", user.ID)
	s.Require().NoError(err)
	config := s.boundConfig("acme-app", crmBinding("inbox", ""), crmBinding("crm", connection.ID))

	deleted, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", id, false)

	s.ErrorIs(err, ErrConnectorDefinitionInUse)
	s.Equal([]string{connection.ID, user.ID}, deleted.Uses.Connections, "the oldest first")
	s.Equal([]ConnectorBindingUse{
		{ConfigID: config.ID, ConfigName: config.Name, Binding: "crm"},
		{ConfigID: config.ID, ConfigName: config.Name, Binding: "inbox"},
	}, deleted.Uses.Bindings)
	s.Empty(deleted.Connections)
	_, err = s.store.LatestConnectorDefinition(s.ctx, "acme-app", id)
	s.NoError(err)
	_, err = s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.NoError(err)
}

func (s *StoreSuite) TestADeleteOfOneAppsConnectorLeavesAnotherAppsOfTheSameIdAlone() {
	id := s.crm("acme-app")
	s.crm("other-app")
	theirs := s.crmConnection("other-app")
	their := s.boundConfig("other-app", crmBinding("inbox", ""))

	unforced, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", id, false)
	s.Require().NoError(err, "the other app's uses do not block it")
	s.Empty(unforced.Uses.Connections)
	s.Empty(unforced.Uses.Bindings)
	s.crm("acme-app")
	mine := s.crmConnection("acme-app")
	forced, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", id, true)

	s.Require().NoError(err)
	s.Equal([]string{mine.ID}, forced.Uses.Connections)
	s.Empty(forced.Uses.Bindings)
	s.Equal([]DeletedConnection{{ID: mine.ID, ConnectorID: id, OwnerType: OwnerApp}}, forced.Connections)
	_, err = s.store.LatestConnectorDefinition(s.ctx, "other-app", id)
	s.NoError(err, "the other app's definition is still there")
	_, err = s.store.ConnectorConnection(s.ctx, "other-app", theirs.ID)
	s.NoError(err, "and its connection")
	read, err := s.store.AgentConfig(s.ctx, "other-app", their.ID)
	s.Require().NoError(err)
	s.Equal([]ConnectorBinding{crmBinding("inbox", "")}, read.Connectors, "and its binding")
}

func (s *StoreSuite) TestAGoneConnectionOrConfigDoesNotKeepAnUnforcedDeleteOff() {
	id := s.crm("acme-app")
	connection := s.crmConnection("acme-app")
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))
	config := s.boundConfig("acme-app", crmBinding("inbox", ""))
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, "acme-app", config.ID))

	deleted, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", id, false)

	s.Require().NoError(err)
	s.Empty(deleted.Uses.Connections)
	s.Empty(deleted.Uses.Bindings)
}

func (s *StoreSuite) TestABindingAloneKeepsAnUnforcedDeleteOff() {
	id := s.crm("acme-app")
	s.boundConfig("acme-app", crmBinding("inbox", ""))

	_, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", id, false)

	s.ErrorIs(err, ErrConnectorDefinitionInUse)
}

func (s *StoreSuite) TestAForcedDeleteDeletesItsConnectionsAndLeavesTheBindings() {
	id := s.crm("acme-app")
	granted := s.crmConnection("acme-app")
	_, err := s.store.DB().ExecContext(s.ctx,
		"UPDATE connector_connections SET credentials_sealed = 'sealed', credentials_kek_version = 1, status = 'connected' WHERE id = ?", granted.ID)
	s.Require().NoError(err)
	pending := s.crmConnection("acme-app")
	user := s.crmConnection("acme-app")
	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE connector_connections SET owner_type = 'user', owner_id = 'u1' WHERE id = ?", user.ID)
	s.Require().NoError(err)
	config := s.boundConfig("acme-app", crmBinding("crm", granted.ID))

	deleted, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", id, true)

	s.Require().NoError(err)
	s.Equal([]DeletedConnection{
		{ID: granted.ID, ConnectorID: id, OwnerType: OwnerApp, HadGrant: true},
		{ID: pending.ID, ConnectorID: id, OwnerType: OwnerApp, HadGrant: false},
		{ID: user.ID, ConnectorID: id, OwnerType: OwnerUser, HadGrant: false},
	}, deleted.Connections)
	for _, gone := range []string{granted.ID, pending.ID, user.ID} {
		_, err = s.store.ConnectorConnection(s.ctx, "acme-app", gone)
		s.ErrorIs(err, ErrNoConnectorConnection)
	}
	var sealed []byte
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT credentials_sealed FROM connector_connections WHERE id = ?", granted.ID).Scan(&sealed))
	s.Empty(sealed, "the credentials went with the connection")
	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Equal([]ConnectorBinding{crmBinding("crm", granted.ID)}, read.Connectors, "left in place, as a forced connection delete leaves it")
}

func (s *StoreSuite) TestADeleteLeavesNoSecretOfTheConnectorBehind() {
	for _, customer := range []string{"acme-app", "other-app"} {
		s.crm(customer)
		_, err := s.store.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{CustomerID: customer, ConnectorID: "custom_crm",
			Registration: core.ClientCustomer, ClientID: "client", SecretSealed: []byte("sealed secret"), KEKVersion: 1})
		s.Require().NoError(err)
		s.configToken(customer, "custom_crm", "sealed", s.base)
		s.eventDestination(customer, "custom_crm", newID(), "all")
	}

	_, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", "custom_crm", false)

	s.Require().NoError(err)
	none := map[string]int{"connector_oauth_clients": 0, "connector_config_tokens": 0, "connector_event_destinations": 0}
	s.Equal(none, s.secretsOf("acme-app", "custom_crm"))
	s.Equal(map[string]int{"connector_oauth_clients": 1, "connector_config_tokens": 1, "connector_event_destinations": 1},
		s.secretsOf("other-app", "custom_crm"), "another customer's are its own")
}

// The connection's create has locked the definition and is held at its INSERT by a SHARE lock
// on connector_connections (table 13.2,
// https://www.postgresql.org/docs/current/explicit-locking.html). The delete waits for it,
// then sees the connection.
func (s *StoreSuite) TestADeleteWaitsForAConnectionInProgressAndIsRefused() {
	id := s.crm("acme-app")
	held := s.begin()
	_, err := held.ExecContext(s.ctx, "LOCK TABLE connector_connections IN SHARE MODE")
	s.Require().NoError(err)

	creator := s.router()
	created := make(chan error, 1)
	connection := &ConnectorConnection{CustomerID: "acme-app", ConnectorID: id, DefinitionRevision: 1, OwnerType: OwnerApp, AuthScheme: "test_key"}
	go func() { created <- creator.CreateConnectorConnection(s.ctx, testSchemes, connection) }()
	// The two seconds and the ten milliseconds are assertTheWaitEnded's (credentials_test.go).
	s.Require().Eventually(func() bool { return s.waitingForTable("connector_connections") == 1 },
		2*time.Second, 10*time.Millisecond, "the create waits at its INSERT")
	deleter := s.router()
	deleted := make(chan error, 1)
	go func() {
		_, err := deleter.DeleteConnectorDefinition(s.ctx, "acme-app", id, false)
		deleted <- err
	}()
	s.Eventually(func() bool { return s.waitingForALock() == 2 },
		2*time.Second, 10*time.Millisecond, "the delete waits for the create's lock on the definition")
	s.Require().NoError(held.Commit())

	s.Require().NoError(<-created)
	s.ErrorIs(<-deleted, ErrConnectorDefinitionInUse)
	_, err = s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.NoError(err)
}

// deleteHeld starts an unforced delete of the customer's custom_crm and holds it, past its lock
// on the definition, at its DELETE of the OAuth clients, by a SHARE lock on that table. held
// lets it go once committed.
func (s *StoreSuite) deleteHeld(customerID string) (held bun.Tx, deleted chan error) {
	held = s.begin()
	_, err := held.ExecContext(s.ctx, "LOCK TABLE connector_oauth_clients IN SHARE MODE")
	s.Require().NoError(err)
	deleter := s.router()
	deleted = make(chan error, 1)
	go func() {
		_, err := deleter.DeleteConnectorDefinition(s.ctx, customerID, "custom_crm", false)
		deleted <- err
	}()
	s.Require().Eventually(func() bool { return s.waitingForTable("connector_oauth_clients") == 1 },
		2*time.Second, 10*time.Millisecond, "the delete waits at its DELETE")
	return held, deleted
}

func (s *StoreSuite) TestAConnectionWaitsForADeleteInProgressAndIsRefused() {
	id := s.crm("acme-app")
	held, deleted := s.deleteHeld("acme-app")

	creator := s.router()
	created := make(chan error, 1)
	go func() {
		created <- creator.CreateConnectorConnection(s.ctx, testSchemes, &ConnectorConnection{
			CustomerID: "acme-app", ConnectorID: id, DefinitionRevision: 1, OwnerType: OwnerApp, AuthScheme: "test_key"})
	}()
	s.Eventually(func() bool { return s.waitingForALock() == 2 },
		2*time.Second, 10*time.Millisecond, "the create waits for the delete's lock on the definition")
	s.Require().NoError(held.Commit())

	s.NoError(<-deleted)
	s.ErrorIs(<-created, ErrNoConnectorDefinition)
	connections, err := s.store.ConnectorConnectionsByOwner(s.ctx, "acme-app", ConnectionFilter{OwnerType: OwnerApp})
	s.Require().NoError(err)
	s.Empty(connections, "no connection to a deleted connector is stored")
}

// The config save has locked the definition and is held at its INSERT by a SHARE lock on
// agent_configs, which lets the delete's read of the table through (table 13.2).
func (s *StoreSuite) TestADeleteWaitsForABindingInProgressAndIsRefused() {
	id := s.crm("acme-app")
	held := s.begin()
	_, err := held.ExecContext(s.ctx, "LOCK TABLE agent_configs IN SHARE MODE")
	s.Require().NoError(err)

	saver := s.router()
	saved := make(chan error, 1)
	go func() {
		saved <- saver.CreateAgentConfig(s.ctx, &AgentConfig{CustomerID: "acme-app", Name: "bound",
			Connectors: []ConnectorBinding{crmBinding("crm", "")}})
	}()
	s.Require().Eventually(func() bool { return s.waitingForTable("agent_configs") == 1 },
		2*time.Second, 10*time.Millisecond, "the save waits at its INSERT")
	deleter := s.router()
	deleted := make(chan error, 1)
	go func() {
		_, err := deleter.DeleteConnectorDefinition(s.ctx, "acme-app", id, false)
		deleted <- err
	}()
	s.Eventually(func() bool { return s.waitingForALock() == 2 },
		2*time.Second, 10*time.Millisecond, "the delete waits for the save's lock on the definition")
	s.Require().NoError(held.Commit())

	s.Require().NoError(<-saved)
	s.ErrorIs(<-deleted, ErrConnectorDefinitionInUse)
}

func (s *StoreSuite) TestABindingWaitsForADeleteInProgressAndIsRefused() {
	s.crm("acme-app")
	held, deleted := s.deleteHeld("acme-app")

	saver := s.router()
	saved := make(chan error, 1)
	go func() {
		saved <- saver.CreateAgentConfig(s.ctx, &AgentConfig{CustomerID: "acme-app", Name: "bound",
			Connectors: []ConnectorBinding{crmBinding("crm", "")}})
	}()
	s.Eventually(func() bool { return s.waitingForALock() == 2 },
		2*time.Second, 10*time.Millisecond, "the save waits for the delete's lock on the definition")
	s.Require().NoError(held.Commit())

	s.NoError(<-deleted)
	s.ErrorIs(<-saved, ErrNoConnectorDefinition)
	configs, err := s.store.CustomerAgentConfigs(s.ctx, "acme-app")
	s.Require().NoError(err)
	s.Empty(configs, "no binding to the deleted connector is stored")
}

// A forced delete leaves its bindings behind on purpose, and saving the config for anything
// else keeps the binding rather than failing until it is removed.
func (s *StoreSuite) TestAnUpdateKeepsABindingAForcedConnectorDeleteLeftBehind() {
	id := s.crm("acme-app")
	config := s.boundConfig("acme-app", crmBinding("inbox", ""))
	_, err := s.store.DeleteConnectorDefinition(s.ctx, "acme-app", id, true)
	s.Require().NoError(err)

	config.Instructions = "be brief"
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, &config))
	other := config
	other.Connectors = append(other.Connectors, ConnectorBinding{Name: "erp", ConnectorID: "custom_erp", Connection: ConnectionBinding{Type: "session"}, Tools: []ToolGrant{}})
	s.ErrorIs(s.store.UpdateAgentConfig(s.ctx, &other), ErrNoConnectorDefinition, "a binding to a connector the customer has none of is refused")

	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Equal("be brief", read.Instructions)
	s.Equal([]ConnectorBinding{crmBinding("inbox", "")}, read.Connectors)
}

// The put has locked the definition and is held at its INSERT by the SHARE lock on
// connector_oauth_clients. The delete waits for it, then deletes the client it stored.
func (s *StoreSuite) TestAnOAuthClientPutInProgressGoesWithTheConnector() {
	id := s.crm("acme-app")
	held := s.begin()
	_, err := held.ExecContext(s.ctx, "LOCK TABLE connector_oauth_clients IN SHARE MODE")
	s.Require().NoError(err)

	putter := s.router()
	put := make(chan error, 1)
	go func() {
		_, err := putter.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{CustomerID: "acme-app", ConnectorID: id,
			Registration: core.ClientCustomer, ClientID: "client", SecretSealed: []byte("sealed secret"), KEKVersion: 1})
		put <- err
	}()
	s.Require().Eventually(func() bool { return s.waitingForTable("connector_oauth_clients") == 1 },
		2*time.Second, 10*time.Millisecond, "the put waits at its INSERT")
	deleter := s.router()
	deleted := make(chan error, 1)
	go func() {
		_, err := deleter.DeleteConnectorDefinition(s.ctx, "acme-app", id, false)
		deleted <- err
	}()
	s.Eventually(func() bool { return s.waitingForALock() == 2 },
		2*time.Second, 10*time.Millisecond, "the delete waits for the put's lock on the definition")
	s.Require().NoError(held.Commit())

	s.Require().NoError(<-put)
	s.Require().NoError(<-deleted)
	s.Equal(0, s.secretsOf("acme-app", id)["connector_oauth_clients"], "no client secret outlives its connector")
}

func (s *StoreSuite) TestAnOAuthClientPutWaitsForADeleteInProgressAndIsRefused() {
	id := s.crm("acme-app")
	held, deleted := s.deleteHeld("acme-app")

	putter := s.router()
	put := make(chan error, 1)
	go func() {
		_, err := putter.PutConnectorOAuthClient(s.ctx, &ConnectorOAuthClient{CustomerID: "acme-app", ConnectorID: id,
			Registration: core.ClientCustomer, ClientID: "client", SecretSealed: []byte("sealed secret"), KEKVersion: 1})
		put <- err
	}()
	s.Eventually(func() bool { return s.waitingForALock() == 2 },
		2*time.Second, 10*time.Millisecond, "the put waits for the delete's lock on the definition")
	s.Require().NoError(held.Commit())

	s.NoError(<-deleted)
	s.ErrorIs(<-put, ErrNoConnectorDefinition)
	s.Equal(0, s.secretsOf("acme-app", id)["connector_oauth_clients"])
}

// A config save binding a connection locks it FOR KEY SHARE before it locks the connector's
// definition, and a forced delete locks the definition before the connection. The delete
// takes the connection FOR NO KEY UPDATE, which does not wait for FOR KEY SHARE (table 13.3,
// https://www.postgresql.org/docs/current/explicit-locking.html#LOCKING-ROWS), so the two do
// not wait on each other in a cycle. Another transaction holds the connection FOR NO KEY
// UPDATE so the delete is caught between its two locks while the save takes the connection.
func (s *StoreSuite) TestAForcedDeleteAndASaveBindingItsConnectionDoNotDeadlock() {
	id := s.crm("acme-app")
	connection := s.crmConnection("acme-app")
	held := s.begin()
	_, err := held.ExecContext(s.ctx, "SELECT id FROM connector_connections WHERE id = ? FOR NO KEY UPDATE", connection.ID)
	s.Require().NoError(err)

	deleter := s.router()
	deleted := make(chan error, 1)
	go func() {
		_, err := deleter.DeleteConnectorDefinition(s.ctx, "acme-app", id, true)
		deleted <- err
	}()
	s.Require().Eventually(func() bool { return s.waitingForALock() == 1 },
		2*time.Second, 10*time.Millisecond, "the delete holds the definition and waits for the connection")
	saver := s.router()
	saved := make(chan error, 1)
	go func() {
		saved <- saver.CreateAgentConfig(s.ctx, &AgentConfig{CustomerID: "acme-app", Name: "bound",
			Connectors: []ConnectorBinding{crmBinding("crm", connection.ID)}})
	}()
	s.Eventually(func() bool { return s.waitingForALock() == 2 },
		2*time.Second, 10*time.Millisecond, "the save holds the connection and waits for the definition")
	s.Require().NoError(held.Commit())

	s.NoError(<-deleted)
	s.ErrorIs(<-saved, ErrNoConnectorDefinition, "the connector is gone")
}
