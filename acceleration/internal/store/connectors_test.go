//go:build integration

package store

import (
	"errors"
	"io/fs"
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
