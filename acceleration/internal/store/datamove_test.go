//go:build integration

package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"math/rand/v2"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// DataMoveSuite has two databases rather than one, because a move is between two
// deployments and the interesting mistakes are the ones a single database hides: rows
// keep their primary keys, so an import into the same tables would overwrite the very
// rows it was reading rather than copying them anywhere.
type DataMoveSuite struct {
	suite.Suite
	source      *Store
	destination *Store
	ctx         context.Context
}

func TestDataMoveSuite(t *testing.T) {
	suite.Run(t, new(DataMoveSuite))
}

func (s *DataMoveSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.ctx = context.Background()
	s.source = s.open(elsewhere(dsn, "leaving"))
	s.destination = s.open(elsewhere(dsn, "moved"))
}

// open connects to a database, creating it if this is the first run, and brings the
// schema up to date.
func (s *DataMoveSuite) open(dsn string) *Store {
	parsed, err := url.Parse(dsn)
	s.Require().NoError(err)
	name := strings.TrimPrefix(parsed.Path, "/")
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to run against %s", name)

	server := *parsed
	server.Path = "/postgres"
	admin := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(server.String())))
	defer admin.Close()
	var exists bool
	s.Require().NoError(admin.QueryRowContext(s.ctx,
		"SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)", name).Scan(&exists))
	if !exists {
		_, err = admin.ExecContext(s.ctx, `CREATE DATABASE "`+name+`"`)
		s.Require().NoError(err)
	}

	store, err := Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(store.Migrate(s.ctx))
	return store
}

// elsewhere is a deployment's database, which is the configured name with a word in front
// of the suffix that keeps it a test database. Both ends belong to this suite, which
// empties them between tests.
func elsewhere(dsn, deployment string) string {
	return testenv.Database(dsn, deployment)
}

func (s *DataMoveSuite) TearDownSuite() {
	for _, store := range []*Store{s.source, s.destination} {
		if store != nil {
			s.Require().NoError(store.Close())
		}
	}
}

func (s *DataMoveSuite) SetupTest() {
	tables := strings.Join(append(DataTables(), "data_changes", "data_change_capture"), ", ")
	for _, store := range []*Store{s.source, s.destination} {
		_, err := store.DB().ExecContext(s.ctx, "TRUNCATE "+tables+" CASCADE")
		s.Require().NoError(err)
		store.SetStreamPins(StreamPins{})
	}
}

// move copies everything a customer has to the other deployment, the way the replicate
// command does, and returns the cursor their changes carry on from.
func (s *DataMoveSuite) move(from, to string) int64 {
	export := s.export(from)
	for _, row := range export.rows {
		encoded, err := json.Marshal(row.row)
		s.Require().NoError(err)
		s.Require().NoError(s.destination.ImportRow(s.ctx, to, row.table, encoded))
	}
	return export.cursor
}

type exported struct {
	cursor int64
	rows   []exportedRow
}

type exportedRow struct {
	table string
	row   map[string]any
}

func (e exported) of(table string) []map[string]any {
	var rows []map[string]any
	for _, row := range e.rows {
		if row.table == table {
			rows = append(rows, row.row)
		}
	}
	return rows
}

func (s *DataMoveSuite) export(customerID string) exported {
	return s.exportFrom(s.source, customerID)
}

func (s *DataMoveSuite) exportFrom(store *Store, customerID string) exported {
	var collected exported
	cursor, err := store.ExportCustomer(s.ctx, customerID, func(table string, row json.RawMessage) error {
		var decoded map[string]any
		if err := json.Unmarshal(row, &decoded); err != nil {
			return err
		}
		collected.rows = append(collected.rows, exportedRow{table: table, row: decoded})
		return nil
	})
	s.Require().NoError(err)
	collected.cursor = cursor
	return collected
}

// capture turns on change recording for a customer, which is what an export does.
func (s *DataMoveSuite) capture(customerID string) {
	s.Require().NoError(s.source.StartDataCapture(s.ctx, customerID, time.Hour))
}

func (s *DataMoveSuite) TestExportCarriesOnlyTheCallersRows() {
	s.seedAgentConfig(s.source, "acme", "acme-agent")
	s.seedAgentConfig(s.source, "other", "other-agent")

	configs := s.export("acme").of("agent_configs")

	s.Require().Len(configs, 1)
	s.Equal("acme-agent", configs[0]["name"])
}

func (s *DataMoveSuite) TestExportLeavesCredentialsBehind() {
	s.seedPluginConnection(s.source, "acme", "xoxb-a-real-token")

	connections := s.export("acme").of("agent_plugin_connections")

	s.Require().Len(connections, 1)
	for _, column := range secretColumns {
		s.NotContains(connections[0], column, "a credential is not part of a customer's data")
	}
}

func (s *DataMoveSuite) TestAnImportedConnectionHasToBeAuthorizedAgain() {
	s.seedPluginConnection(s.source, "acme", "xoxb-a-real-token")

	s.move("acme", "acme")

	var token string
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT access_token FROM agent_plugin_connections").Scan(&token))
	s.Empty(token)
}

func (s *DataMoveSuite) TestImportWritesTheRowsUnderTheImportingCustomer() {
	s.seedAgentConfig(s.source, "acme", "acme-agent")

	// The same rows arriving at a deployment where the app goes by another id.
	s.move("acme", "moved")

	configs := s.exportFrom(s.destination, "moved").of("agent_configs")
	s.Require().Len(configs, 1)
	s.Equal("acme-agent", configs[0]["name"])
	s.Equal("moved", configs[0]["customer_id"], "whose row it is comes from the credential, not the file")
}

func (s *DataMoveSuite) TestImportingTwiceIsImportingOnce() {
	s.seedAgentConfig(s.source, "acme", "acme-agent")

	s.move("acme", "acme")
	s.move("acme", "acme")

	s.Len(s.exportFrom(s.destination, "acme").of("agent_configs"), 1)
}

func (s *DataMoveSuite) TestAnImportBringsLaterEditsAcross() {
	id := s.seedAgentConfig(s.source, "acme", "first-draft")
	s.move("acme", "acme")

	_, err := s.source.DB().ExecContext(s.ctx,
		"UPDATE agent_configs SET name = ? WHERE id = ?", "second-draft", id)
	s.Require().NoError(err)
	s.move("acme", "acme")

	configs := s.exportFrom(s.destination, "acme").of("agent_configs")
	s.Require().Len(configs, 1)
	s.Equal("second-draft", configs[0]["name"])
}

func (s *DataMoveSuite) TestImportRefusesATableAnExportDoesNotCarry() {
	s.ErrorContains(s.destination.ImportRow(s.ctx, "acme", "api_keys", json.RawMessage(`{"id":"vak_live_x"}`)),
		"not a table an export carries")
}

func (s *DataMoveSuite) TestChangesFollowWhatHappenedAfterTheExport() {
	s.capture("acme")
	s.seedAgentConfig(s.source, "acme", "before")
	cursor := s.move("acme", "acme")

	s.seedAgentConfig(s.source, "acme", "after")

	changes, next, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().Len(changes, 1)
	s.Equal("agent_configs", changes[0].Table)
	s.Equal(ChangeInsert, changes[0].Op)
	s.Greater(next, cursor)

	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))
	s.Len(s.exportFrom(s.destination, "acme").of("agent_configs"), 2)
}

func (s *DataMoveSuite) TestNothingIsRecordedForACustomerThatIsNotMoving() {
	s.seedAgentConfig(s.source, "acme", "quiet")

	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)

	s.Require().NoError(err)
	s.Empty(changes, "a deployment nobody is leaving records nothing")
}

func (s *DataMoveSuite) TestChangesStopAtTheCustomerAskingForThem() {
	s.capture("acme")
	s.capture("other")
	s.seedAgentConfig(s.source, "other", "not-yours")

	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)

	s.Require().NoError(err)
	s.Empty(changes)
}

func (s *DataMoveSuite) TestADeleteIsReplayedToo() {
	s.capture("acme")
	id := s.seedAgentConfig(s.source, "acme", "doomed")
	cursor := s.move("acme", "acme")

	_, err := s.source.DB().ExecContext(s.ctx, "DELETE FROM agent_configs WHERE id = ?", id)
	s.Require().NoError(err)

	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().Len(changes, 1)
	s.Equal(ChangeDelete, changes[0].Op)

	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))
	s.Empty(s.exportFrom(s.destination, "acme").of("agent_configs"))
}

// The columns a move copies are read from the catalog, so the bindings come with the
// config without being named anywhere.
func (s *DataMoveSuite) TestAConfigsBindingsMoveWithIt() {
	config := &AgentConfig{CustomerID: "acme", Name: "bound", Connectors: []ConnectorBinding{{
		// Per session: a fixed binding needs a live connection of the source's to be stored.
		Name: "crm", ConnectorID: "slack", Connection: ConnectionBinding{Type: "session"},
		Tools: []ToolGrant{{Name: "search", SchemaDigest: strings.Repeat("a", 64)}}, Required: true,
	}}}
	s.Require().NoError(s.source.CreateAgentConfig(s.ctx, config))

	s.move("acme", "moved")

	moved, err := s.destination.AgentConfig(s.ctx, "moved", config.ID)
	s.Require().NoError(err)
	s.Equal(config.Connectors, moved.Connectors)
}

func (s *DataMoveSuite) TestAnImportCannotOverwriteAnotherCustomersRow() {
	id := s.seedAgentConfig(s.source, "acme", "mine")
	// The destination already holds a row with that id, belonging to somebody else.
	// Primary keys come across as they are, so an import naming an id it has no business
	// with is the shape of the attack.
	_, err := s.destination.DB().ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES (?, ?, ?, ?)",
		id, "resident", "theirs", "be brief")
	s.Require().NoError(err)

	s.move("acme", "intruder")

	var name, owner string
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT name, customer_id FROM agent_configs WHERE id = ?", id).Scan(&name, &owner))
	s.Equal("theirs", name)
	s.Equal("resident", owner)
}

func (s *DataMoveSuite) TestAReplayedChangeCannotDeleteAnotherCustomersRow() {
	s.capture("acme")
	id := s.seedAgentConfig(s.source, "acme", "mine")
	_, err := s.source.DB().ExecContext(s.ctx, "DELETE FROM agent_configs WHERE id = ?", id)
	s.Require().NoError(err)
	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)
	s.Require().NoError(err)

	_, err = s.destination.DB().ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES (?, ?, ?, ?)",
		id, "resident", "theirs", "be brief")
	s.Require().NoError(err)

	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "intruder", changes))

	var count int
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM agent_configs WHERE id = ?", id).Scan(&count))
	s.Equal(1, count)
}

func (s *DataMoveSuite) TestAnExpiredCursorSaysToExportAgain() {
	s.capture("acme")
	s.seedAgentConfig(s.source, "acme", "one")
	_, cursor, err := s.source.Changes(s.ctx, "acme", 0, 100)
	s.Require().NoError(err)

	_, err = s.source.DB().ExecContext(s.ctx, "DELETE FROM data_changes")
	s.Require().NoError(err)
	s.seedAgentConfig(s.source, "acme", "two")

	_, _, err = s.source.Changes(s.ctx, "acme", cursor-1, 100)
	s.ErrorIs(err, ErrChangesExpired)
}

func (s *DataMoveSuite) TestPruningDropsWhatNobodyCanResumeFrom() {
	s.capture("acme")
	s.seedAgentConfig(s.source, "acme", "old")
	_, err := s.source.DB().ExecContext(s.ctx, "UPDATE data_changes SET at = now() - interval '30 days'")
	s.Require().NoError(err)

	removed, err := s.source.PruneDataChanges(s.ctx, 7*24*time.Hour)

	s.Require().NoError(err)
	s.Positive(removed)
}

func (s *DataMoveSuite) TestChildRowsFollowTheirParent() {
	voice := s.seedVoice(s.source, "acme", "narrator")
	_, err := s.source.DB().ExecContext(s.ctx,
		"INSERT INTO voice_samples (id, voice_id, object_key) VALUES (?, ?, ?)",
		newID(), voice, "voices/acme/1.wav")
	s.Require().NoError(err)

	s.move("acme", "acme")

	s.Len(s.exportFrom(s.destination, "acme").of("voice_samples"), 1)
}

func (s *DataMoveSuite) TestAChildRowCannotBeImportedOntoSomebodyElsesParent() {
	voice := s.seedVoice(s.destination, "resident", "narrator")
	sample, err := json.Marshal(map[string]any{
		"id": newID(), "voice_id": voice, "object_key": "voices/theirs/1.wav",
	})
	s.Require().NoError(err)

	s.Require().NoError(s.destination.ImportRow(s.ctx, "intruder", "voice_samples", sample))

	var count int
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM voice_samples WHERE voice_id = ?", voice).Scan(&count))
	s.Zero(count, "a row is only written where its parent belongs to the importing customer")
}

func (s *DataMoveSuite) TestOnlyTheAppsOwnPolicyMoves() {
	_, err := s.source.DB().ExecContext(s.ctx,
		"INSERT INTO policies (scope, scope_id, document, updated_at)"+
			" VALUES ('app', ?, ?, now()), ('organization', ?, ?, now())",
		"acme", `{"prompt_injection": true}`, "acme", `{"prompt_injection": false}`)
	s.Require().NoError(err)

	policies := s.export("acme").of("policies")

	s.Require().Len(policies, 1)
	s.Equal("app", policies[0]["scope"])
}

func (s *DataMoveSuite) seedAgentConfig(store *Store, customerID, name string) string {
	id := newID()
	_, err := store.DB().ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES (?, ?, ?, ?)",
		id, customerID, name, "be brief")
	s.Require().NoError(err)
	return id
}

func (s *DataMoveSuite) seedVoice(store *Store, customerID, name string) string {
	id := newID()
	_, err := store.DB().ExecContext(s.ctx,
		"INSERT INTO voices (id, customer_id, name) VALUES (?, ?, ?)", id, customerID, name)
	s.Require().NoError(err)
	return id
}

func (s *DataMoveSuite) seedPluginConnection(store *Store, customerID, token string) string {
	id := newID()
	_, err := store.DB().ExecContext(s.ctx,
		"INSERT INTO agent_plugin_connections (id, customer_id, config_id, plugin_id, access_token)"+
			" VALUES (?, ?, ?, ?, ?)",
		id, customerID, newID(), "slack", token)
	s.Require().NoError(err)
	return id
}

// pinnedSession is a session row on the source, pinned to the app given; zero leaves it
// unpinned, as everything written before pins was.
func (s *DataMoveSuite) pinnedSession(customerID string, app int64) string {
	id := uuid.NewString()
	s.Require().NoError(s.source.SaveSession(s.ctx, &AgentSession{
		ID: id, CustomerID: customerID, StreamAppPK: app, AgentID: "agent", UserID: "user",
	}))
	return id
}

// pinOnDestination is the pin a moved session arrived with, nil for none.
func (s *DataMoveSuite) pinOnDestination(id string) *int64 {
	var pin *int64
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT stream_app_pk FROM agent_sessions WHERE id = ?", id).Scan(&pin))
	return pin
}

func (s *DataMoveSuite) TestAnImportedRowCannotChooseItsStreamApp() {
	// A pin decides which app a row's work is finished in, so an import may name only the
	// app its customer acts in on this side.
	s.destination.SetStreamPins(StreamPins{For: func(context.Context, string) (int64, error) { return 4242, nil }})
	own, other, unpinned := s.pinnedSession("acme", 4242), s.pinnedSession("acme", 999), s.pinnedSession("acme", 0)

	s.move("acme", "acme")

	s.Equal(int64(4242), *s.pinOnDestination(own))
	s.Equal(ForeignStreamApp, *s.pinOnDestination(other))
	s.Equal(ForeignStreamApp, *s.pinOnDestination(unpinned), "an unpinned row was in somebody's deployment app, not this customer's")
}

func (s *DataMoveSuite) TestAnExportNamesTheAppANullPinMeant() {
	// NULL is the app of whichever deployment wrote it, which means nothing elsewhere.
	s.source.SetStreamPins(StreamPins{Deployment: func() int64 { return 1 }})
	id := s.pinnedSession("acme", 0)

	rows := s.export("acme").of("agent_sessions")

	s.Require().Len(rows, 1)
	s.Equal(id, rows[0]["id"])
	s.EqualValues(1, rows[0]["stream_app_pk"])
}

func (s *DataMoveSuite) TestAMoveWithinOneDeploymentAppKeepsItsPins() {
	// Both sides act in the same deployment app, so what it wrote stays its.
	s.source.SetStreamPins(StreamPins{Deployment: func() int64 { return 1 }})
	s.destination.SetStreamPins(StreamPins{Deployment: func() int64 { return 1 }})
	id := s.pinnedSession("acme", 0)

	s.move("acme", "acme")

	s.Require().NotNil(s.pinOnDestination(id))
	s.Equal(int64(1), *s.pinOnDestination(id))
}

func (s *DataMoveSuite) TestAMoveBetweenDeploymentsThatNeverPinnedLeavesRowsUnpinned() {
	id := s.pinnedSession("acme", 0)

	s.move("acme", "acme")

	s.Nil(s.pinOnDestination(id), "every move before pins existed worked this way")
}

func (s *DataMoveSuite) TestARowPinnedWhileThisDeploymentsAppIsUnknownWaits() {
	// The source knows its app and says so on every row. Until this side knows its own,
	// it cannot tell that row's app from somebody else's, and calling it foreign would
	// park it for good.
	s.source.SetStreamPins(StreamPins{Deployment: func() int64 { return 1 }})
	id := s.pinnedSession("acme", 0)
	rows := s.export("acme").of("agent_sessions")
	s.Require().Len(rows, 1)
	encoded, err := json.Marshal(rows[0])
	s.Require().NoError(err)

	s.destination.SetStreamPins(StreamPins{Knowable: func() bool { return true }})
	err = s.destination.ImportRow(s.ctx, "acme", "agent_sessions", encoded)
	s.Require().ErrorIs(err, ErrStreamAppUnknown)

	s.destination.SetStreamPins(StreamPins{Deployment: func() int64 { return 1 }})
	s.Require().NoError(s.destination.ImportRow(s.ctx, "acme", "agent_sessions", encoded))
	s.Require().NotNil(s.pinOnDestination(id))
	s.Equal(int64(1), *s.pinOnDestination(id))
}

func (s *DataMoveSuite) TestBackfillPinsTheDeploymentAppsRows() {
	unpinned, own := s.pinnedSession("acme", 0), s.pinnedSession("acme", 4242)

	pinned, err := s.source.BackfillStreamPins(s.ctx, 1)

	s.Require().NoError(err)
	s.GreaterOrEqual(pinned["agent_sessions"], int64(1))
	for id, want := range map[string]int64{unpinned: 1, own: 4242} {
		var pin int64
		s.Require().NoError(s.source.DB().QueryRowContext(s.ctx,
			"SELECT stream_app_pk FROM agent_sessions WHERE id = ?", id).Scan(&pin))
		s.Equal(want, pin, "a row already pinned keeps its app")
	}
	_, err = s.source.BackfillStreamPins(s.ctx, 0)
	s.Error(err, "an unknown deployment app pins nothing")
}

func (s *DataMoveSuite) TestADeploymentThatCanNeverKnowItsAppParksAPinnedRow() {
	// With no Stream key to ask with, waiting would be waiting for ever.
	s.source.SetStreamPins(StreamPins{Deployment: func() int64 { return 1 }})
	id := s.pinnedSession("acme", 0)
	rows := s.export("acme").of("agent_sessions")
	s.Require().Len(rows, 1)
	encoded, err := json.Marshal(rows[0])
	s.Require().NoError(err)

	s.Require().NoError(s.destination.ImportRow(s.ctx, "acme", "agent_sessions", encoded))

	s.Equal(ForeignStreamApp, *s.pinOnDestination(id))
}

func (s *DataMoveSuite) TestAMovedConnectorConnectionArrivesNeedingReauthorization() {
	id := s.seedConnectorConnection(s.source, "acme")

	exported := s.export("acme").of("connector_connections")
	s.Require().Len(exported, 1)
	s.NotContains(exported[0], "credentials_sealed", "sealed credentials are not part of a customer's data")

	s.move("acme", "moved")

	s.assertArrivedWithoutCredentials(id, "moved")
}

func (s *DataMoveSuite) TestAMovedConnectorConnectionLeavesItsProviderUnitBehind() {
	s.capture("acme")
	id := s.seedConnectorConnection(s.source, "acme")
	_, err := s.source.DB().ExecContext(s.ctx, "UPDATE connector_connections SET provider_unit_id = 'unit-1' WHERE id = ?", id)
	s.Require().NoError(err)

	s.move("acme", "moved")
	var imported sql.NullString
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT provider_unit_id FROM connector_connections WHERE id = ?", id).Scan(&imported))
	s.False(imported.Valid, "an import proves no unit is the importer's")

	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "moved", changes))
	var applied sql.NullString
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT provider_unit_id FROM connector_connections WHERE id = ?", id).Scan(&applied))
	s.False(applied.Valid, "nor does a change applied after it")
}

func (s *DataMoveSuite) TestACustomConnectorDefinitionMovesAndABuiltInDoesNot() {
	s.seedConnectorConnection(s.source, "acme")
	_, err := s.source.DB().ExecContext(s.ctx,
		"INSERT INTO connector_definitions (customer_id, id, revision, name, manifest) VALUES ('', 'acme', 1, 'Acme', '{}')")
	s.Require().NoError(err)

	s.move("acme", "moved")

	definitions := s.exportFrom(s.destination, "moved").of("connector_definitions")
	s.Require().Len(definitions, 1)
	s.Equal("custom_acme", definitions[0]["id"])
	s.Equal("moved", definitions[0]["customer_id"])
}

func (s *DataMoveSuite) TestAChangeToAConnectorConnectionLeavesItsCredentialsBehind() {
	s.capture("acme")
	id := s.seedConnectorConnection(s.source, "acme")

	_, err := s.source.DB().ExecContext(s.ctx,
		"UPDATE connector_connections SET credentials_sealed = 'rotated credentials', revision = 2 WHERE id = ?", id)
	s.Require().NoError(err)

	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)
	s.Require().NoError(err)
	recorded := 0
	for _, change := range changes {
		if change.Table != "connector_connections" {
			continue
		}
		recorded++
		s.NotContains(string(change.Payload), "credentials_sealed")
	}
	s.Equal(2, recorded, "the insert and the update")

	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))
	s.assertArrivedWithoutCredentials(id, "acme")
}

// assertArrivedWithoutCredentials checks a connection the destination received holds what a
// connection with no credentials holds: none, key version 0, no expiry, and a status asking
// for a reconnect.
func (s *DataMoveSuite) assertArrivedWithoutCredentials(id, owner string) {
	var sealed []byte
	var kekVersion int
	var expiresAt sql.NullTime
	var status, customer string
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT credentials_sealed, credentials_kek_version, expires_at, status, customer_id FROM connector_connections WHERE id = ?",
		id).Scan(&sealed, &kekVersion, &expiresAt, &status, &customer))
	s.Empty(sealed)
	s.Zero(kekVersion, "the source credentials' key version does not come along")
	s.False(expiresAt.Valid, "the source credentials' expiry does not come along")
	s.Equal(ConnectionNeedsReauthorization, status)
	s.Equal(owner, customer)
}

// seedConnectorConnection stores a custom definition and a connected connection to it with
// sealed credentials under key version 1 and an expiry, the state a credentials write leaves.
func (s *DataMoveSuite) seedConnectorConnection(store *Store, customerID string) string {
	_, err := store.DB().ExecContext(s.ctx,
		"INSERT INTO connector_definitions (customer_id, id, revision, name, manifest) VALUES (?, 'custom_acme', 1, 'Acme', '{}')",
		customerID)
	s.Require().NoError(err)
	id := newID()
	_, err = store.DB().ExecContext(s.ctx,
		"INSERT INTO connector_connections (id, customer_id, connector_id, definition_revision, owner_type, auth_scheme,"+
			" status, credentials_sealed, credentials_kek_version, expires_at)"+
			" VALUES (?, ?, 'custom_acme', 1, 'app', 'test_key', ?, 'sealed credentials', 1, now() + interval '1 hour')",
		id, customerID, ConnectionConnected)
	s.Require().NoError(err)
	return id
}

func (s *DataMoveSuite) TestAMovedSIPTrunkArrivesWithoutItsPassword() {
	trunk := &SIPTrunk{
		ID: newID(), CustomerID: "acme", Name: "main", Host: "trunk.example.com", Port: 5060,
		Transport: "tcp", Username: "agent", PasswordSealed: []byte("sealed here"),
		PasswordKEKVersion: 1, Codecs: []string{"PCMU"},
	}
	s.Require().NoError(s.source.CreateSIPTrunk(s.ctx, trunk))

	trunks := s.export("acme").of("sip_trunks")
	s.Require().Len(trunks, 1)
	s.NotContains(trunks[0], "password_sealed", "a sealed password is not part of a customer's data")

	s.move("acme", "acme")

	moved, err := s.destination.SIPTrunk(s.ctx, "acme", trunk.ID)
	s.Require().NoError(err)
	s.Empty(moved.PasswordSealed)
	s.Equal(0, moved.PasswordKEKVersion)
	s.Equal("trunk.example.com", moved.Host)
}

func (s *DataMoveSuite) TestAMovedNumberLeavesItsBridgePasswordBehind() {
	s.capture("acme")
	s.Require().NoError(s.source.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000110", Vendor: "telnyx", Country: "US", CustomerID: "acme",
	}))
	s.Require().NoError(s.source.AttachNumber(s.ctx, "acme", "+15550000110", NumberAttachment{
		TrunkID: "trunk-1", RouteID: "route-1", CallType: "default", CallID: "call-1",
		BridgeURI: "sip:trunk.example.com", BridgeUsername: "agent",
		BridgePasswordSealed: []byte("sealed bridge"), BridgePasswordKEKVersion: 1,
	}))

	numbers := s.export("acme").of("phone_numbers")
	s.Require().Len(numbers, 1)
	s.NotContains(numbers[0], "bridge_password_sealed", "a sealed password is not part of a customer's data")

	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)
	s.Require().NoError(err)
	recorded := 0
	for _, change := range changes {
		if change.Table != "phone_numbers" {
			continue
		}
		recorded++
		s.NotContains(string(change.Payload), "bridge_password_sealed")
	}
	s.Equal(2, recorded, "the insert and the attach")

	s.move("acme", "acme")
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))

	var sealed []byte
	var version sql.NullInt64
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT bridge_password_sealed, bridge_password_kek_version FROM phone_numbers WHERE e164 = ?",
		"+15550000110").Scan(&sealed, &version))
	s.Empty(sealed)
	s.False(version.Valid, "the source password's key version does not come along")

	// Nor does where its calls went, which a hook finds the number by.
	routing := map[string]*sql.NullString{"stream_trunk_id": {}, "stream_route_id": {}, "stream_call_id": {},
		"stream_call_type": {}, "bridge_uri": {}, "bridge_username": {}}
	for column, value := range routing {
		s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
			"SELECT "+column+" FROM phone_numbers WHERE e164 = ?", "+15550000110").Scan(value))
		s.False(value.Valid, "%s came along as %q", column, value.String)
	}
	_, err = s.destination.AttachedBridge(s.ctx, "telnyx", "+15550000110")
	s.ErrorIs(err, ErrNoNumber, "it arrives needing to be attached again")
}

func (s *DataMoveSuite) TestAMovedNumberIsHeldWhereItArrives() {
	s.Require().NoError(s.source.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000111", Vendor: "plivo", Country: "US", CustomerID: "acme",
	}))

	s.move("acme", "acme")
	s.move("acme", "acme")

	held, err := s.destination.Number(s.ctx, "acme", "+15550000111")
	s.Require().NoError(err)
	s.Equal("plivo", held.Vendor)
}

func (s *DataMoveSuite) TestANumberInAnotherFormIsNotImported() {
	s.Require().NoError(s.destination.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000112", Vendor: "plivo", Country: "US", CustomerID: "acme",
	}))

	// Each is a form a vendor reads as +15550000112.
	for _, form := range []struct{ vendor, e164 string }{
		{"plivo", "15550000112"}, {"bandwidth", "15550000112"}, {"bandwidth", "5550000112"},
		{"didww", "15550000112"}, {"vonage", "15550000112"},
	} {
		err := s.destination.ImportRow(s.ctx, "other", "phone_numbers", s.importedNumber(form.e164, form.vendor))
		s.ErrorIs(err, ErrNotE164, "%s %s", form.vendor, form.e164)
	}
	err := s.destination.ApplyChanges(s.ctx, "other", []DataChange{{
		Table: "phone_numbers", Op: ChangeInsert, Payload: s.importedNumber("15550000112", "plivo"),
	}})
	s.ErrorIs(err, ErrNotE164, "nor replayed as a change")

	held, err := s.destination.CustomerNumbers(s.ctx, "other", true)
	s.Require().NoError(err)
	s.Empty(held)
	_, err = s.destination.Number(s.ctx, "acme", "+15550000112")
	s.NoError(err, "the number stays with who held it")
}

func (s *DataMoveSuite) TestANumberAnotherCustomerHoldsIsNotImported() {
	s.Require().NoError(s.destination.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000113", Vendor: "plivo", Country: "US", CustomerID: "acme",
	}))
	_, err := s.destination.DB().ExecContext(s.ctx, "INSERT INTO phone_numbers (e164, vendor, country, customer_id)"+
		" VALUES ('15550000114', 'plivo', 'US', 'acme')")
	s.Require().NoError(err, "held in another form, before every write checked the form")
	s.Require().NoError(s.destination.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+5550000120", Vendor: "bandwidth", Country: "US", CustomerID: "acme",
	}))

	// Bandwidth reads +15550000113 and +5550000113 as one number.
	for _, number := range []struct{ vendor, e164 string }{
		{"plivo", "+15550000113"}, {"bandwidth", "+15550000113"}, {"didww", "+15550000114"},
		{"bandwidth", "+5550000113"}, {"bandwidth", "+15550000120"},
	} {
		err := s.destination.ImportRow(s.ctx, "other", "phone_numbers", s.importedNumber(number.e164, number.vendor))
		s.ErrorIs(err, ErrNumberHeldElsewhere, "%s %s", number.vendor, number.e164)
	}
}

func (s *DataMoveSuite) TestANumberIsReadByItsColumnsAsTheyAreNamed() {
	s.Require().NoError(s.destination.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000119", Vendor: "plivo", Country: "US", CustomerID: "acme",
	}))

	// A key in another case names no column, so it gives nothing back.
	for _, number := range []struct {
		vendor, e164 string
		refused      error
	}{
		{"bandwidth", "+15550000119", ErrNumberHeldElsewhere}, {"plivo", "15550000119", ErrNotE164},
	} {
		var row map[string]any
		s.Require().NoError(json.Unmarshal(s.importedNumber(number.e164, number.vendor), &row))
		delete(row, "released_at")
		row["Released_At"] = time.Now().UTC()
		encoded, err := json.Marshal(row)
		s.Require().NoError(err)

		err = s.destination.ImportRow(s.ctx, "other", "phone_numbers", encoded)
		s.ErrorIs(err, number.refused, "%s %s", number.vendor, number.e164)
	}
	held, err := s.destination.CustomerNumbers(s.ctx, "other", true)
	s.Require().NoError(err)
	s.Empty(held)
}

func (s *DataMoveSuite) TestANumberNobodyElseHoldsAtAVendorIsImported() {
	s.Require().NoError(s.destination.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000115", Vendor: "plivo", Country: "US", CustomerID: "acme",
	}))
	s.Require().NoError(s.destination.ReleaseNumber(s.ctx, "acme", "+15550000115", time.Time{}))
	s.Require().NoError(s.destination.AddTrunkNumber(s.ctx, &PhoneNumber{
		E164: "+15550000116", Country: "US", CustomerID: "acme", SIPTrunkID: s.trunkOf(s.destination, "acme").ID,
	}))
	s.Require().NoError(s.destination.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000117", Vendor: "plivo", Country: "US", CustomerID: "acme",
	}))

	for name, row := range map[string]json.RawMessage{
		"one acme gave back":                       s.importedNumber("+15550000115", "plivo"),
		"one acme has on its trunk":                s.importedNumber("+15550000116", "plivo"),
		"another one at a vendor that keeps its 1": s.importedNumber("+5550000117", "plivo"),
		"one given back, in any form":              s.importedNumber("15550000118", "plivo", "released_at", time.Now().UTC()),
		"one acme holds, on a trunk": s.importedNumber("+15550000117", "sip_trunk",
			"sip_trunk_id", s.trunkOf(s.destination, "other").ID),
	} {
		s.NoError(s.destination.ImportRow(s.ctx, "other", "phone_numbers", row), name)
	}
}

// importedNumber is a number as an import file can carry it, under an id nothing here has,
// with any further columns given as name and value pairs.
func (s *DataMoveSuite) importedNumber(e164, vendor string, columns ...any) json.RawMessage {
	row := map[string]any{
		"id": rand.Int64N(1 << 50), "e164": e164, "vendor": vendor, "country": "US",
		"capabilities": []string{"voice"}, "monthly_cost_micros": 0, "customer_id": "whoever",
		"tags": map[string]string{}, "purchased_at": time.Now().UTC(), "released_at": nil,
	}
	for i := 0; i+1 < len(columns); i += 2 {
		row[columns[i].(string)] = columns[i+1]
	}
	encoded, err := json.Marshal(row)
	s.Require().NoError(err)
	return encoded
}

func (s *DataMoveSuite) TestAnImportedNumberCannotPointAtAnotherCustomersTrunk() {
	trunk := &SIPTrunk{
		ID: newID(), CustomerID: "other", Name: "main", Host: "trunk.example.com", Port: 5060,
		Transport: "tcp", Username: "agent", PasswordSealed: []byte{}, Codecs: []string{"PCMU"},
	}
	s.Require().NoError(s.destination.CreateSIPTrunk(s.ctx, trunk))

	own := s.trunkOf(s.source, "acme")
	s.Require().NoError(s.source.AddTrunkNumber(s.ctx, &PhoneNumber{
		E164: "+15550000109", Country: "US", CustomerID: "acme", SIPTrunkID: own.ID,
	}))
	numbers := s.export("acme").of("phone_numbers")
	s.Require().Len(numbers, 1)
	numbers[0]["sip_trunk_id"] = trunk.ID
	row, err := json.Marshal(numbers[0])
	s.Require().NoError(err)

	err = s.destination.ImportRow(s.ctx, "acme", "phone_numbers", row)
	s.True(isViolation(err, foreignKeyViolation), "a number on somebody else's trunk is refused: %v", err)
}

func (s *DataMoveSuite) trunkOf(store *Store, customerID string) *SIPTrunk {
	trunk := &SIPTrunk{
		ID: newID(), CustomerID: customerID, Name: "main", Host: "trunk.example.com", Port: 5060,
		Transport: "tcp", Username: "agent", PasswordSealed: []byte{},
	}
	s.Require().NoError(store.CreateSIPTrunk(s.ctx, trunk))
	return trunk
}

func (s *DataMoveSuite) TestScheduledTasksMoveWithTheirRuns() {
	next := time.Now().UTC().Add(time.Hour).Truncate(time.Second)
	task := &ScheduledTask{
		ID: "stretch", CustomerID: "acme", ConfigID: "config-1", UserID: "sam", CallerKind: "authenticated",
		Title: "Stretch", Instructions: "Remind Sam to stretch.", StartLocal: next, Timezone: "Europe/Amsterdam",
		NextRunAt: &next, State: ScheduledTaskActive, CreatedBy: "api",
	}
	s.Require().NoError(s.source.CreateScheduledTask(s.ctx, task, 25))
	s.Require().NoError(s.source.InsertScheduledTaskRun(s.ctx, &ScheduledTaskRun{
		TaskID: "stretch", CustomerID: "acme", ScheduledFor: next.Add(-24 * time.Hour),
		State: ScheduledTaskRunDelivered, DeliveredTo: ScheduledTaskDeliveredConversation,
	}))

	s.move("acme", "moved")

	moved, err := s.destination.ScheduledTask(s.ctx, "moved", "stretch")
	s.Require().NoError(err)
	s.Equal("Remind Sam to stretch.", moved.Instructions)
	s.Equal("config-1", moved.ConfigID)
	s.Require().NotNil(moved.NextRunAt)
	s.Equal(next, moved.NextRunAt.UTC())
	runs, err := s.destination.ScheduledTaskRuns(s.ctx, "moved", "stretch", 25, nil)
	s.Require().NoError(err)
	s.Require().Len(runs, 1, "the task's history moves with it")
	s.Equal(ScheduledTaskRunDelivered, runs[0].State)
}
