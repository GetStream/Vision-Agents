//go:build integration

package store

import (
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"math/rand/v2"
	"net/url"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun"
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
	reservation, err := s.destination.ReserveImportedIDs(s.ctx)
	s.Require().NoError(err)
	for _, row := range export.rows {
		encoded, err := json.Marshal(row.row)
		s.Require().NoError(err)
		s.Require().NoError(reservation.Cover(s.ctx, row.table, encoded))
		s.Require().NoError(s.destination.ImportRow(s.ctx, to, row.table, encoded))
	}
	return export.cursor
}

// moveEdited is move with the rows of one table edited first, as a file the importer wrote
// itself would carry them.
func (s *DataMoveSuite) moveEdited(from, to, table string, edit func(map[string]any)) {
	reservation, err := s.destination.ReserveImportedIDs(s.ctx)
	s.Require().NoError(err)
	for _, row := range s.export(from).rows {
		if row.table == table {
			edit(row.row)
		}
		encoded, err := json.Marshal(row.row)
		s.Require().NoError(err)
		s.Require().NoError(reservation.Cover(s.ctx, row.table, encoded))
		s.Require().NoError(s.destination.ImportRow(s.ctx, to, row.table, encoded))
	}
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
	s.Require().NoError(s.source.StartDataCapture(s.ctx, customerID, time.Hour, time.Second))
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

func (s *DataMoveSuite) TestAGuestsRefreshSecretDoesNotLeaveWithThem() {
	// It is what lets a guest come back as themselves; a moved guest is handed a new one.
	_, hash, err := NewGuestSecret()
	s.Require().NoError(err)
	s.Require().NoError(s.source.RecordGuest(s.ctx, &User{ID: "guest-1", CustomerID: "acme", Name: "Alice", RefreshSecretHash: hash}))

	users := s.export("acme").of("users")
	s.Require().Len(users, 1)
	s.NotContains(users[0], "refresh_secret_hash", "a credential is not part of a customer's data")
}

func (s *DataMoveSuite) TestAChangeToAGuestLeavesTheirRefreshSecretBehind() {
	// The change feed strips columns in its own trigger, apart from secretColumns.
	s.capture("acme")
	_, hash, err := NewGuestSecret()
	s.Require().NoError(err)
	s.Require().NoError(s.source.RecordGuest(s.ctx, &User{ID: "guest-1", CustomerID: "acme", Name: "Alice", RefreshSecretHash: hash}))
	_, next, err := NewGuestSecret()
	s.Require().NoError(err)
	s.Require().NoError(s.source.SetGuestSecret(s.ctx, "acme", "guest-1", next))

	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)
	s.Require().NoError(err)
	recorded := 0
	for _, change := range changes {
		if change.Table != "users" {
			continue
		}
		recorded++
		s.NotContains(string(change.Payload), "refresh_secret_hash")
	}
	s.Equal(2, recorded, "the insert and the new secret")
}

func (s *DataMoveSuite) TestAGuestsRefreshSecretInAFileIsNotImported() {
	// A file the importer wrote could otherwise plant a secret it knows for a guest.
	s.Require().NoError(s.source.RecordGuest(s.ctx, &User{ID: "guest-1", CustomerID: "acme", Name: "Alice"}))

	s.moveEdited("acme", "moved", "users", func(row map[string]any) {
		row["refresh_secret_hash"] = "\\x" + hex.EncodeToString(GuestSecretHash("planted"))
	})

	var held []byte
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT refresh_secret_hash FROM users WHERE customer_id = 'moved' AND id = 'guest-1'").Scan(&held))
	s.Empty(held)
}

func (s *DataMoveSuite) TestAnImportedConnectionHasToBeAuthorizedAgain() {
	s.seedPluginConnection(s.source, "acme", "xoxb-a-real-token")

	s.move("acme", "acme")

	var token string
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT access_token FROM agent_plugin_connections").Scan(&token))
	s.Empty(token)
}

func (s *DataMoveSuite) TestAMoveKeepsWhoItsUsersAre() {
	// What the router recorded of a user is the customer's data: an account arrives an account,
	// and a claimed guest arrives claimed by the same account, so nobody else can claim it.
	s.Require().NoError(s.source.RecordAccount(s.ctx, "acme", "alice"))
	s.Require().NoError(s.source.RecordGuest(s.ctx, &User{ID: "guest-1", CustomerID: "acme", Name: "Guest"}))
	_, err := s.source.ClaimGuest(s.ctx, "acme", "guest-1", "alice")
	s.Require().NoError(err)
	claimed, err := s.source.Guest(s.ctx, "acme", "guest-1")
	s.Require().NoError(err)

	s.move("acme", "moved")

	_, err = s.destination.Guest(s.ctx, "moved", "alice")
	s.Error(err, "an account is nobody's guest to claim")
	guest, err := s.destination.Guest(s.ctx, "moved", "guest-1")
	s.Require().NoError(err)
	s.Equal("alice", guest.ClaimedBy)
	s.Require().NotNil(guest.ClaimedAt)
	s.True(claimed.ClaimedAt.Equal(*guest.ClaimedAt))
	_, err = s.destination.ClaimGuest(s.ctx, "moved", "guest-1", "mallory")
	s.ErrorContains(err, "already claimed")
}

func (s *DataMoveSuite) TestAMovedConnectionKeepsItsOwnerAndWhatItsConsentGranted() {
	// The owner is who reaches it, and the account is what a reconnect has to match.
	id := s.seedConnectorConnection(s.source, "acme")
	_, err := s.source.DB().ExecContext(s.ctx,
		"UPDATE connector_connections SET owner_type = ?, owner_id = 'bob', account_id = 'bob-at-acme',"+
			" granted_scopes = '[\"read\"]' WHERE id = ?", OwnerUser, id)
	s.Require().NoError(err)

	s.move("acme", "moved")

	listed, err := s.destination.ConnectorConnectionsByOwner(s.ctx, "moved",
		ConnectionFilter{OwnerType: OwnerUser, OwnerID: "bob"})
	s.Require().NoError(err)
	s.Require().Len(listed, 1)
	s.Equal(id, listed[0].ID)
	s.Equal("bob-at-acme", listed[0].AccountID)
	s.Equal([]string{"read"}, listed[0].GrantedScopes)
}

func (s *DataMoveSuite) TestAMovedModeratorIsStillTheAppsModerator() {
	id := s.seedAgentConfig(s.source, "acme", "nora")
	_, err := s.source.DB().ExecContext(s.ctx, "UPDATE agent_configs SET role = ? WHERE id = ?", RoleAgentModerator, id)
	s.Require().NoError(err)

	s.move("acme", "moved")

	moderator, found, err := s.destination.AgentConfigByRole(s.ctx, "moved", RoleAgentModerator)
	s.Require().NoError(err)
	s.Require().True(found, "the chat worker finds the app's moderator by its role")
	s.Equal(id, moderator.ID)
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

// The numbers are shared by every customer, so the ones between an export's cursor and the
// customer's first change after it can be another customer's.
func (s *DataMoveSuite) TestTheFirstChangeAfterAnExportIsAnsweredWhenOthersChangedInBetween() {
	s.capture("other")
	s.seedAgentConfig(s.source, "other", "before")
	s.capture("acme")
	cursor := s.export("acme").cursor
	s.seedAgentConfig(s.source, "other", "between")
	first := s.seedAgentConfig(s.source, "acme", "first")

	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)

	s.Require().NoError(err)
	s.Require().Len(changes, 1)
	s.Contains(string(changes[0].Key), first)
}

func (s *DataMoveSuite) TestACursorWhoseNextChangeWasPrunedSaysToExportAgain() {
	s.capture("acme")
	s.seedAgentConfig(s.source, "acme", "one")
	cursor := s.export("acme").cursor
	s.seedAgentConfig(s.source, "acme", "lost")
	s.age()
	s.seedAgentConfig(s.source, "acme", "kept")
	s.prune()

	_, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)

	s.ErrorIs(err, ErrChangesExpired)
}

func (s *DataMoveSuite) TestACursorRightBeforeTheOldestChangeKeptIsAnswered() {
	s.capture("acme")
	s.seedAgentConfig(s.source, "acme", "one")
	cursor := s.export("acme").cursor
	s.age()
	next := s.seedAgentConfig(s.source, "acme", "next")
	s.prune()

	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)

	s.Require().NoError(err)
	s.Require().Len(changes, 1)
	s.Contains(string(changes[0].Key), next)
}

// lapse lets a customer's capture expire, as it does when nobody reads their changes for longer
// than the retention.
func (s *DataMoveSuite) lapse(customerID string) {
	_, err := s.source.DB().ExecContext(s.ctx,
		"UPDATE data_change_capture SET expires_at = now() - interval '1 minute' WHERE customer_id = ?", customerID)
	s.Require().NoError(err)
}

// Nothing is recorded while a capture is off, so a cursor from before it lapsed is refused once
// the recording starts again, whatever was recorded after the start.
func (s *DataMoveSuite) TestACursorFromBeforeTheCaptureLapsedSaysToExportAgain() {
	cases := []struct {
		name string
		// between runs while the capture is off, after acme's write there.
		between func()
		// since runs after the capture is started again.
		since func()
	}{
		{name: "nothing recorded since", between: func() {}, since: func() {}},
		{name: "a change recorded since", between: func() {},
			since: func() { s.seedAgentConfig(s.source, "acme", "after") }},
		{name: "a prune while it was off", since: func() {}, between: func() {
			_, err := s.source.PruneDataChanges(s.ctx, 7*24*time.Hour)
			s.Require().NoError(err)
		}},
	}
	for _, c := range cases {
		s.Run(c.name, func() {
			s.SetupTest()
			s.capture("acme")
			s.seedAgentConfig(s.source, "acme", "followed")
			_, cursor, err := s.source.Changes(s.ctx, "acme", s.export("acme").cursor, 100)
			s.Require().NoError(err)
			s.lapse("acme")
			s.seedAgentConfig(s.source, "acme", "written while nothing was recorded")
			c.between()
			s.capture("acme")
			c.since()

			for _, after := range []int64{0, cursor} {
				_, _, err = s.source.Changes(s.ctx, "acme", after, 100)
				s.ErrorIs(err, ErrChangesExpired, "after %d", after)
			}
		})
	}
}

// A follower reading within the retention pushes the capture out, which is no lapse: it is
// answered every change since its cursor, as before.
func (s *DataMoveSuite) TestACursorFollowedWithinTheRetentionIsAnsweredEveryChange() {
	s.capture("acme")
	cursor := s.export("acme").cursor
	first := s.seedAgentConfig(s.source, "acme", "first")
	s.capture("acme")
	second := s.seedAgentConfig(s.source, "acme", "second")

	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)

	s.Require().NoError(err)
	s.Require().Len(changes, 2)
	s.Contains(string(changes[0].Key), first)
	s.Contains(string(changes[1].Key), second)
}

// An export after a lapse carries on from where the recording started again, even with no change
// kept at that number and the number after it rolled back.
func (s *DataMoveSuite) TestAnExportAfterALapseIsFollowedFromItsCursor() {
	s.capture("other")
	s.capture("acme")
	stale := s.export("acme").cursor
	s.seedAgentConfig(s.source, "acme", "followed")
	s.lapse("acme")
	lost := s.seedAgentConfig(s.source, "acme", "written while nothing was recorded")
	s.age()
	s.prune()
	s.capture("acme")
	_, _, err := s.source.Changes(s.ctx, "acme", stale, 100)
	s.Require().ErrorIs(err, ErrChangesExpired)

	exported := s.export("acme")
	rolledBack := s.begin()
	s.writeIn(rolledBack, "other", "rolled back")
	s.Require().NoError(rolledBack.Rollback())
	next := s.seedAgentConfig(s.source, "acme", "next")
	changes, _, err := s.source.Changes(s.ctx, "acme", exported.cursor, 100)

	s.Require().NoError(err)
	s.Require().Len(changes, 1)
	s.Contains(string(changes[0].Key), next)
	var ids []any
	for _, row := range exported.of("agent_configs") {
		ids = append(ids, row["id"])
	}
	s.Contains(ids, lost, "the write made while nothing was recorded is in the new export")
}

// afterQuery runs a function once, after the first query mentioning a phrase.
type afterQuery struct {
	mu     sync.Mutex
	phrase string
	run    func()
}

func (*afterQuery) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context { return ctx }

func (a *afterQuery) AfterQuery(_ context.Context, event *bun.QueryEvent) {
	a.mu.Lock()
	run := a.run
	if run != nil && strings.Contains(event.Query, a.phrase) {
		a.run = nil
	} else {
		run = nil
	}
	a.mu.Unlock()
	if run != nil {
		run()
	}
}

// disarm stops the hook from running again, so it cannot fire in a later test.
func (a *afterQuery) disarm() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.run = nil
}

// A prune between the read of what is kept and the read of the rows drops the rows right
// after a cursor. Read apart, the rows start above them and the follower jumps over them
// with no refusal; read in one snapshot, it is handed them.
func (s *DataMoveSuite) TestAPruneBetweenTheTwoReadsOfAFeedDoesNotLoseRowsAfterTheCursor() {
	s.capture("acme")
	s.seedAgentConfig(s.source, "acme", "one")
	cursor := s.export("acme").cursor
	first := s.seedAgentConfig(s.source, "acme", "first after the cursor")
	second := s.seedAgentConfig(s.source, "acme", "second after the cursor")
	s.age()
	young := s.seedAgentConfig(s.source, "acme", "young")
	hook := &afterQuery{phrase: "MIN(seq) FROM data_changes", run: func() {
		removed, err := s.source.PruneDataChanges(s.ctx, 7*24*time.Hour)
		s.NoError(err)
		s.Positive(removed)
	}}
	s.source.DB().AddQueryHook(hook)
	s.T().Cleanup(hook.disarm)

	changes, next, err := s.source.Changes(s.ctx, "acme", cursor, 100)

	s.Require().NoError(err)
	s.Require().Len(changes, 3)
	s.Contains(string(changes[0].Key), first)
	s.Contains(string(changes[1].Key), second)
	s.Contains(string(changes[2].Key), young)
	s.Equal(changes[2].Seq, next)
}

// A customer with nothing kept may have had nothing to record: a follower waiting out a quiet
// spell longer than the retention gets an empty page, as it did before the numbers were read
// across customers.
func (s *DataMoveSuite) TestAQuietCustomerWhoseCursorAgedOutIsNotRefused() {
	s.capture("acme")
	s.capture("other")
	s.seedAgentConfig(s.source, "acme", "one")
	cursor := s.export("acme").cursor
	s.seedAgentConfig(s.source, "other", "aged out too")
	s.age()
	s.prune()
	s.seedAgentConfig(s.source, "other", "later")

	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)

	s.Require().NoError(err)
	s.Empty(changes)
}

// age makes every change recorded so far older than the retention prune keeps.
func (s *DataMoveSuite) age() {
	_, err := s.source.DB().ExecContext(s.ctx, "UPDATE data_changes SET at = now() - interval '30 days'")
	s.Require().NoError(err)
}

func (s *DataMoveSuite) prune() {
	removed, err := s.source.PruneDataChanges(s.ctx, 7*24*time.Hour)
	s.Require().NoError(err)
	s.Require().Positive(removed)
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

// A change is numbered when it is written and seen when its transaction commits, so a
// transaction that started later can hold a lower number than one that already committed.
func (s *DataMoveSuite) TestAChangeOfATransactionThatCommitsLateIsNotSteppedOver() {
	s.capture("acme")
	cursor := s.export("acme").cursor
	older := s.begin()
	first := s.writeIn(older, "acme", "older, first")
	newer := s.begin()
	late := s.writeIn(newer, "acme", "newer, in flight")
	second := s.writeIn(older, "acme", "older, second")
	s.Require().NoError(older.Commit())

	handed, cursor := s.follow("acme", cursor)
	s.Equal([]string{first}, handed, "nothing past the change still in flight")

	s.Require().NoError(newer.Commit())
	rest, _ := s.follow("acme", cursor)
	s.Equal([]string{late, second}, rest)
}

// With every change up to the cursor pruned, nothing older is kept, so the number in flight
// right after the cursor is the only gap.
func (s *DataMoveSuite) TestAChangeInFlightRightAfterTheCursorDoesNotExpireIt() {
	s.capture("acme")
	s.ageChange(s.seedAgentConfig(s.source, "acme", "aged out"))
	_, cursor := s.follow("acme", 0)
	s.prune()
	inFlight := s.begin()
	late := s.writeIn(inFlight, "acme", "in flight")
	committed := s.seedAgentConfig(s.source, "acme", "committed")

	handed, next := s.follow("acme", cursor)
	s.Empty(handed)
	s.Require().NoError(inFlight.Commit())
	rest, _ := s.follow("acme", next)

	s.Equal([]string{late, committed}, rest)
}

// A number handed out and rolled back is never a change. Right after an export onto a
// table with no change kept, it is not one pruned either.
func (s *DataMoveSuite) TestANumberRolledBackRightAfterAnExportCursorIsNotAPrunedChange() {
	s.capture("acme")
	_, err := s.source.DB().ExecContext(s.ctx, "SELECT setval('data_changes_seq_seq', 1000, true)")
	s.Require().NoError(err)
	cursor := s.export("acme").cursor
	rolledBack := s.begin()
	s.writeIn(rolledBack, "acme", "rolled back")
	s.Require().NoError(rolledBack.Rollback())
	committed := s.seedAgentConfig(s.source, "acme", "committed")

	handed, _ := s.follow("acme", cursor)

	s.Equal([]string{committed}, handed)
}

// The lock naming a number is keyed by its two 32-bit halves, so a number whose low half
// has the sign bit set, or whose high half is not 0, is held back as well as a small one.
func (s *DataMoveSuite) TestAChangeInFlightIsHeldBackWhateverTheHalvesOfItsNumber() {
	s.capture("acme")
	for _, start := range []int64{1<<31 + 5, 1<<32 + 3, 1<<32 + 1<<31 + 7} {
		_, err := s.source.DB().ExecContext(s.ctx, "SELECT setval('data_changes_seq_seq', ?, true)", start)
		s.Require().NoError(err)
		cursor := s.export("acme").cursor
		older := s.begin()
		first := s.writeIn(older, "acme", fmt.Sprintf("older, first, from %d", start))
		newer := s.begin()
		late := s.writeIn(newer, "acme", fmt.Sprintf("newer, in flight, from %d", start))
		second := s.writeIn(older, "acme", fmt.Sprintf("older, second, from %d", start))
		s.Require().NoError(older.Commit())

		handed, next := s.follow("acme", cursor)
		s.Equal([]string{first}, handed, "numbers from %d", start)

		s.Require().NoError(newer.Commit())
		rest, _ := s.follow("acme", next)
		s.Equal([]string{late, second}, rest, "numbers from %d", start)
	}
}

func (s *DataMoveSuite) TestAQuietFeedHandsOutEveryChangeUpToTheNewest() {
	s.capture("acme")
	cursor := s.export("acme").cursor
	want := []string{
		s.seedAgentConfig(s.source, "acme", "one"),
		s.seedAgentConfig(s.source, "acme", "two"),
		s.seedAgentConfig(s.source, "acme", "three"),
	}
	var newest int64
	s.Require().NoError(s.source.DB().QueryRowContext(s.ctx, "SELECT MAX(seq) FROM data_changes").Scan(&newest))

	handed, next := s.follow("acme", cursor)

	s.Equal(want, handed)
	s.Equal(newest, next)
	s.Equal(newest, s.export("acme").cursor)
}

// The numbers are shared by every customer, so another customer's change in flight holds
// this customer's feed back too, whichever of them commits first.
func (s *DataMoveSuite) TestChangesInFlightForSeveralCustomersHoldEveryFeedBack() {
	s.capture("acme")
	s.capture("other")
	start := s.export("acme").cursor
	older := s.begin()
	first := s.writeIn(older, "acme", "older, first")
	acme := s.begin()
	acmeLate := s.writeIn(acme, "acme", "acme, in flight")
	second := s.writeIn(older, "acme", "older, second")
	other := s.begin()
	otherLate := s.writeIn(other, "other", "other, in flight")
	third := s.writeIn(older, "acme", "older, third")
	s.Require().NoError(older.Commit())

	handed, acmeCursor := s.follow("acme", start)
	s.Equal([]string{first}, handed)
	theirs, otherCursor := s.follow("other", start)
	s.Empty(theirs)

	s.Require().NoError(other.Commit())
	handed, acmeCursor = s.follow("acme", acmeCursor)
	s.Empty(handed, "acme's own change is still in flight")
	theirs, otherCursor = s.follow("other", otherCursor)
	s.Empty(theirs, "acme's change in flight holds a lower number")

	s.Require().NoError(acme.Commit())
	handed, _ = s.follow("acme", acmeCursor)
	s.Equal([]string{acmeLate, second, third}, handed)
	theirs, _ = s.follow("other", otherCursor)
	s.Equal([]string{otherLate}, theirs)
}

func (s *DataMoveSuite) TestAnOpenTransactionThatRecordsNothingDoesNotHoldTheFeedBack() {
	s.capture("acme")
	cursor := s.export("acme").cursor
	s.writeIn(s.begin(), "not-moving", "open")
	committed := s.seedAgentConfig(s.source, "acme", "committed")

	handed, _ := s.follow("acme", cursor)

	s.Equal([]string{committed}, handed)
}

func (s *DataMoveSuite) TestAChangeInFlightInAnotherDatabaseDoesNotHoldTheFeedBack() {
	s.capture("acme")
	cursor := s.export("acme").cursor
	s.Require().NoError(s.destination.StartDataCapture(s.ctx, "acme", time.Hour, time.Second))
	elsewhere, err := s.destination.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = elsewhere.Rollback() })
	_, err = elsewhere.ExecContext(s.ctx, "SELECT setval('data_changes_seq_seq', 1, false)")
	s.Require().NoError(err)
	s.writeIn(elsewhere, "acme", "in flight in the other database")
	committed := s.seedAgentConfig(s.source, "acme", "committed")

	handed, _ := s.follow("acme", cursor)

	s.Equal([]string{committed}, handed)
}

func (s *DataMoveSuite) TestAnExportCursorStaysBelowAChangeInFlight() {
	s.capture("acme")
	// A sequence that has handed out nothing yet holds the first number it will hand out.
	_, err := s.source.DB().ExecContext(s.ctx, "SELECT setval('data_changes_seq_seq', 1, false)")
	s.Require().NoError(err)
	inFlight := s.begin()
	late := s.writeIn(inFlight, "acme", "in flight")
	s.seedAgentConfig(s.source, "acme", "committed")

	cursor := s.export("acme").cursor
	s.Require().NoError(inFlight.Commit())
	handed, _ := s.follow("acme", cursor)

	s.Contains(handed, late)
}

// A transaction that started earlier and wrote later records an older time under a higher
// number: pruning by time alone drops it and keeps the lower one, and a cursor between them
// would carry on past the gap.
func (s *DataMoveSuite) TestPruningKeepsAChangeNumberedAboveOneStillKept() {
	s.capture("acme")
	old := s.seedAgentConfig(s.source, "acme", "aged out")
	s.seedAgentConfig(s.source, "acme", "young")
	late := s.seedAgentConfig(s.source, "acme", "started earlier")
	_, cursor, err := s.source.Changes(s.ctx, "acme", 0, 2)
	s.Require().NoError(err)
	s.ageChange(old)
	s.ageChange(late)
	s.prune()

	handed, _ := s.follow("acme", cursor)

	s.Equal([]string{late}, handed)
}

func (s *DataMoveSuite) TestPruningKeepsWhatIsNumberedAboveAChangeInFlight() {
	s.capture("acme")
	s.ageChange(s.seedAgentConfig(s.source, "acme", "aged out"))
	_, cursor := s.follow("acme", 0)
	inFlight := s.begin()
	held := s.writeIn(inFlight, "acme", "in flight")
	after := s.seedAgentConfig(s.source, "acme", "committed above it")
	s.ageChange(after)
	s.prune()
	s.Require().NoError(inFlight.Commit())

	handed, _ := s.follow("acme", cursor)

	s.Equal([]string{held, after}, handed)
}

// A write whose trigger looked before the capture committed recorded nothing, and an export
// read while it is open misses it too, so the wait holds it until it ends.
func (s *DataMoveSuite) TestWaitingForOpenWritesHoldsAWriteFromBeforeTheCaptureStarted() {
	before := s.begin()
	id := s.writeIn(before, "acme", "straddles the capture start")
	s.capture("acme")

	started := time.Now()
	s.ErrorIs(s.source.WaitForOpenWrites(s.ctx, 50*time.Millisecond), ErrWritesOpen)
	s.Less(time.Since(started), 5*time.Second, "it gives up once its bound is past")

	waited := make(chan error, 1)
	go func() { waited <- s.source.WaitForOpenWrites(s.ctx, time.Minute) }()
	s.Never(func() bool { return len(waited) > 0 }, 300*time.Millisecond, 10*time.Millisecond,
		"the wait returned while the write was open")
	s.Require().NoError(before.Commit())
	s.Require().NoError(<-waited)
	exported := s.export("acme")
	handed, _ := s.follow("acme", exported.cursor)
	s.True(slices.ContainsFunc(exported.of("agent_configs"), func(row map[string]any) bool { return row["id"] == id }) ||
		slices.Contains(handed, id), "the write is in neither the export nor the changes")
}

// The wait is for the writes open when it started. One that opens later sees the capture, so it
// records its own change, and waiting for it would make a busy database refuse every export.
func (s *DataMoveSuite) TestWaitingForOpenWritesIgnoresAWriteOpenedAfterItStarted() {
	first := s.begin()
	s.writeIn(first, "acme", "open before the wait")
	s.capture("acme")

	waited := make(chan error, 1)
	go func() { waited <- s.source.WaitForOpenWrites(s.ctx, time.Minute) }()
	s.Never(func() bool { return len(waited) > 0 }, 300*time.Millisecond, 10*time.Millisecond,
		"the wait returned while the first write was open")

	later := s.begin()
	s.writeIn(later, "acme", "opened after the wait started")
	s.Require().NoError(first.Commit())

	select {
	case err := <-waited:
		s.NoError(err)
	case <-time.After(10 * time.Second):
		s.Fail("the wait held on a write that opened after it started")
	}
}

// With nothing open there is nothing to wait for, so even no time at all is enough.
func (s *DataMoveSuite) TestWaitingForOpenWritesReturnsAtOnceWithNoneOpen() {
	s.NoError(s.source.WaitForOpenWrites(s.ctx, 0))
}

// A transaction that only reads holds no transaction id, and can hold no change.
func (s *DataMoveSuite) TestWaitingForOpenWritesIgnoresATransactionThatOnlyReads() {
	reading := s.begin()
	s.Require().NoError(reading.QueryRowContext(s.ctx, "SELECT count(*) FROM agent_configs").Scan(new(int)))

	s.NoError(s.source.WaitForOpenWrites(s.ctx, 0))
}

// Transaction ids are counted across the whole Postgres server; another database's writes are
// none of this one's.
func (s *DataMoveSuite) TestWaitingForOpenWritesIgnoresAnotherDatabase() {
	elsewhere, err := s.destination.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = elsewhere.Rollback() })
	s.writeIn(elsewhere, "acme", "in the other database")

	s.NoError(s.source.WaitForOpenWrites(s.ctx, 0))
}

// A repeatable read or serializable transaction reads with the snapshot of its first query, so
// its trigger sees no capture that started after it, and it has no transaction id for the wait
// to hold. Its write is refused instead, which those levels are retried on, and the retry is in
// the changes.
func (s *DataMoveSuite) TestARepeatableReadWriteOnASnapshotFromBeforeTheCaptureIsNotLost() {
	s.writeOnASnapshotFromBeforeTheCaptureIsNotLost(sql.LevelRepeatableRead)
}

func (s *DataMoveSuite) TestASerializableWriteOnASnapshotFromBeforeTheCaptureIsNotLost() {
	s.writeOnASnapshotFromBeforeTheCaptureIsNotLost(sql.LevelSerializable)
}

func (s *DataMoveSuite) writeOnASnapshotFromBeforeTheCaptureIsNotLost(level sql.IsolationLevel) {
	before := s.beginAt(level)
	s.capture("acme")
	s.Require().NoError(s.source.WaitForOpenWrites(s.ctx, time.Second))
	exported := s.export("acme")

	lost, err := s.tryWriteIn(before, "acme", "first write after the capture")
	if err == nil {
		err = before.Commit()
	}
	s.True(isViolation(err, serializationFailure), "the write on the old snapshot was not refused: %v", err)
	retry := s.beginAt(level)
	again := s.writeIn(retry, "acme", "the same write again")
	s.Require().NoError(retry.Commit())

	handed, _ := s.follow("acme", exported.cursor)
	s.NotContains(handed, lost)
	s.Equal([]string{again}, handed)
}

// A repeatable read write that looked for a capture before the start recorded nothing, so the
// start waits for it to end, and gives up rather than hangs, even when given no time at all.
func (s *DataMoveSuite) TestACaptureStartWaitsForARepeatableReadWriteThatLookedBeforeIt() {
	before := s.beginAt(sql.LevelRepeatableRead)
	id := s.writeIn(before, "acme", "looked before the start")

	bounded, cancel := context.WithTimeout(s.ctx, 5*time.Second)
	defer cancel()
	s.ErrorIs(s.source.StartDataCapture(bounded, "acme", time.Hour, 0), ErrWritesOpen)

	started := make(chan error, 1)
	go func() { started <- s.source.StartDataCapture(s.ctx, "acme", time.Hour, time.Minute) }()
	s.Never(func() bool { return len(started) > 0 }, 300*time.Millisecond, 10*time.Millisecond,
		"the start did not wait for the write")
	s.Require().NoError(before.Commit())
	s.Require().NoError(<-started)
	s.True(slices.ContainsFunc(s.export("acme").of("agent_configs"), func(row map[string]any) bool { return row["id"] == id }))
}

// Following the changes pushes out a capture still on, which is no start: it waits for nobody.
func (s *DataMoveSuite) TestPushingOutACaptureStillOnWaitsForNobody() {
	s.capture("acme")
	writing := s.beginAt(sql.LevelRepeatableRead)
	s.writeIn(writing, "not-moving", "holds the starts")

	s.NoError(s.source.StartDataCapture(s.ctx, "acme", time.Hour, 50*time.Millisecond))
}

// A capture that expired is off, so turning it on again is a start.
func (s *DataMoveSuite) TestRenewingAnExpiredCaptureIsAStart() {
	s.capture("acme")
	_, err := s.source.DB().ExecContext(s.ctx,
		"UPDATE data_change_capture SET expires_at = now() - interval '1 minute' WHERE customer_id = 'acme'")
	s.Require().NoError(err)
	before := s.beginAt(sql.LevelRepeatableRead)
	s.capture("acme")

	_, err = s.tryWriteIn(before, "acme", "on a snapshot of the expired capture")

	s.True(isViolation(err, serializationFailure), "the write on the old snapshot was not refused: %v", err)
}

// With nobody moving, a repeatable read write commits and records nothing, as before.
func (s *DataMoveSuite) TestARepeatableReadWriteWithNobodyMovingRecordsNothing() {
	writing := s.beginAt(sql.LevelRepeatableRead)
	s.writeIn(writing, "acme", "nobody moving")
	s.Require().NoError(writing.Commit())

	var recorded int
	s.Require().NoError(s.source.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM data_changes").Scan(&recorded))
	s.Zero(recorded)
}

// A transaction that only reads writes nothing a capture could miss, however old its snapshot,
// so an export neither waits for it nor is refused for it.
func (s *DataMoveSuite) TestALongReadDoesNotHoldAnExport() {
	reading, err := s.source.DB().BeginTx(s.ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = reading.Rollback() })
	s.Require().NoError(reading.QueryRowContext(s.ctx,
		"SELECT count(*) FROM agent_configs, data_change_capture").Scan(new(int)))

	s.capture("acme")
	s.NoError(s.source.WaitForOpenWrites(s.ctx, 0))
}

// serializationFailure is the SQLSTATE of a transaction Postgres refused to let go on at its
// isolation level, the one such a transaction is retried on.
const serializationFailure = "40001"

// beginAt opens a transaction at an isolation level and takes its snapshot.
func (s *DataMoveSuite) beginAt(level sql.IsolationLevel) bun.Tx {
	tx, err := s.source.DB().BeginTx(s.ctx, &sql.TxOptions{Isolation: level})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = tx.Rollback() })
	s.Require().NoError(tx.QueryRowContext(s.ctx, "SELECT count(*) FROM agent_configs").Scan(new(int)))
	return tx
}

// begin opens a transaction on the source that the test commits when it says so.
func (s *DataMoveSuite) begin() bun.Tx {
	tx, err := s.source.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = tx.Rollback() })
	return tx
}

func (s *DataMoveSuite) writeIn(tx bun.Tx, customerID, name string) string {
	id, err := s.tryWriteIn(tx, customerID, name)
	s.Require().NoError(err)
	return id
}

// tryWriteIn is writeIn for a write the test expects may be refused.
func (s *DataMoveSuite) tryWriteIn(tx bun.Tx, customerID, name string) (string, error) {
	id := newID()
	_, err := tx.ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES (?, ?, ?, ?)",
		id, customerID, name, "be brief")
	return id, err
}

// follow reads a customer's changes after a cursor and returns the ids of the rows they name,
// oldest first, and the cursor to read on from.
func (s *DataMoveSuite) follow(customerID string, cursor int64) ([]string, int64) {
	changes, next, err := s.source.Changes(s.ctx, customerID, cursor, 100)
	s.Require().NoError(err)
	var ids []string
	for _, change := range changes {
		var key struct {
			ID string `json:"id"`
		}
		s.Require().NoError(json.Unmarshal(change.Key, &key))
		ids = append(ids, key.ID)
	}
	return ids, next
}

// ageChange makes the change of one row older than the retention prune keeps.
func (s *DataMoveSuite) ageChange(id string) {
	_, err := s.source.DB().ExecContext(s.ctx,
		"UPDATE data_changes SET at = now() - interval '30 days' WHERE key ->> 'id' = ?", id)
	s.Require().NoError(err)
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

func (s *DataMoveSuite) TestCopyingAgainKeepsAConnectionAuthorizedAgainOnTheDestination() {
	id := s.seedConnectorConnection(s.source, "acme")
	s.move("acme", "moved")
	authorized := s.authorizeAgain(s.destination, "moved", id)
	// The source is still live during a move, and a refresh there rewrites its row after the
	// owner authorized the connection here.
	s.authorizeAgain(s.source, "acme", id)

	s.move("acme", "moved")

	s.assertCredentialsKept(authorized)
}

func (s *DataMoveSuite) TestAChangeAppliedLaterKeepsAConnectionAuthorizedAgainOnTheDestination() {
	s.capture("acme")
	id := s.seedConnectorConnection(s.source, "acme")
	s.move("acme", "moved")
	authorized := s.authorizeAgain(s.destination, "moved", id)
	s.authorizeAgain(s.source, "acme", id)

	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "moved", changes))

	s.assertCredentialsKept(authorized)
}

func (s *DataMoveSuite) TestCopyingAgainStillUpdatesAConnectionNotAuthorizedOnTheDestination() {
	id := s.seedConnectorConnection(s.source, "acme")
	s.move("acme", "moved")
	_, err := s.source.DB().ExecContext(s.ctx, "UPDATE connector_connections SET label = 'renamed' WHERE id = ?", id)
	s.Require().NoError(err)

	s.move("acme", "moved")

	s.assertArrivedWithoutCredentials(id, "moved")
	moved, err := s.destination.ConnectorConnection(s.ctx, "moved", id)
	s.Require().NoError(err)
	s.Equal("renamed", moved.Label, "the source's edit arrives")
}

func (s *DataMoveSuite) TestCopyingAgainDuringARefreshOnTheDestinationKeepsItsCredentials() {
	id := s.seedConnectorConnection(s.source, "acme")
	s.move("acme", "moved")
	s.authorizeAgain(s.destination, "moved", id)

	err := s.destination.WithLockedConnectorConnection(s.ctx, "moved", id,
		func(connection *ConnectorConnection, checkpoint func() error) (bool, error) {
			// The resolver checkpoints the row before the refresh token leaves.
			connection.Status = ConnectionNeedsReauthorization
			s.Require().NoError(checkpoint())
			// A repeat copy lands while the provider answers.
			s.move("acme", "moved")
			connection.Status = ConnectionConnected
			connection.Revision++
			connection.CredentialsSealed = []byte("rotated on moved")
			connection.CredentialsKEKVersion = 1
			return true, nil
		})

	s.Require().NoError(err, "the rotated credentials are saved")
	held, err := s.destination.ConnectorConnection(s.ctx, "moved", id)
	s.Require().NoError(err)
	s.Equal(ConnectionConnected, held.Status)
	s.Equal([]byte("rotated on moved"), held.CredentialsSealed)
}

// authorizeAgain saves new credentials onto a connection, as a consent or a refresh does, and
// returns the connection as it was saved.
func (s *DataMoveSuite) authorizeAgain(store *Store, customerID, id string) ConnectorConnection {
	connection, err := store.ConnectorConnection(s.ctx, customerID, id)
	s.Require().NoError(err)
	expires := time.Now().Add(2 * time.Hour).UTC().Truncate(time.Microsecond)
	connection.Status = ConnectionConnected
	connection.Revision++
	connection.CredentialsSealed = []byte("sealed on " + customerID)
	connection.CredentialsKEKVersion = 1
	connection.ExpiresAt = &expires
	s.Require().NoError(store.SaveConnectorConnectionAtRevision(s.ctx, &connection, connection.Revision-1))
	return connection
}

// assertCredentialsKept checks the destination still holds the credential state a save left:
// the sealed credentials open only at the revision and key version they were sealed under.
func (s *DataMoveSuite) assertCredentialsKept(saved ConnectorConnection) {
	held, err := s.destination.ConnectorConnection(s.ctx, saved.CustomerID, saved.ID)
	s.Require().NoError(err)
	s.Equal(ConnectionConnected, held.Status)
	s.Equal(saved.Revision, held.Revision)
	s.Equal(saved.CredentialsSealed, held.CredentialsSealed)
	s.Equal(saved.CredentialsKEKVersion, held.CredentialsKEKVersion)
	s.Require().NotNil(held.ExpiresAt)
	s.True(saved.ExpiresAt.Equal(*held.ExpiresAt))
}

// seedScheduledTask stores a task on the source in a state, claimed by a worker there, with a
// finished run.
// seedScheduledTask stores a task on the source with its run, and the agent config it runs
// on, config-1, unless the source already holds that id.
func (s *DataMoveSuite) seedScheduledTask(id, customerID, state string) {
	_, err := s.source.DB().ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES ('config-1', ?, 'a', 'be brief') ON CONFLICT (id) DO NOTHING",
		customerID)
	s.Require().NoError(err)
	next := time.Date(2026, 12, 1, 9, 0, 0, 0, time.UTC)
	task := &ScheduledTask{
		ID: id, CustomerID: customerID, ConfigID: "config-1", UserID: "sam", CallerKind: "authenticated",
		Title: "Stretch", Instructions: "Remind Sam to stretch.", StartLocal: next, Timezone: "UTC",
		State: state, NextRunAt: &next, CreatedBy: "api",
	}
	s.Require().NoError(s.source.CreateScheduledTask(s.ctx, task, 25))
	_, err = s.source.DB().ExecContext(s.ctx,
		"UPDATE scheduled_tasks SET claim_id = 'claim-1', lease_until = now() + interval '1 hour' WHERE id = ?", id)
	s.Require().NoError(err)
	_, err = s.source.DB().ExecContext(s.ctx, `
INSERT INTO scheduled_task_runs (id, task_id, customer_id, scheduled_for, state, finished_at)
VALUES (?, ?, ?, ?, 'delivered', now())`, id+"-run", id, customerID, next.Add(-day))
	s.Require().NoError(err)
}

// scheduledTaskOn reads a task's state, claim and next run from a deployment.
func (s *DataMoveSuite) scheduledTaskOn(store *Store, id string) (state, claim string, leased bool, next time.Time) {
	var lease sql.NullTime
	s.Require().NoError(store.DB().QueryRowContext(s.ctx,
		"SELECT state, claim_id, lease_until, next_run_at FROM scheduled_tasks WHERE id = ?", id).
		Scan(&state, &claim, &lease, &next))
	return state, claim, lease.Valid, next
}

func (s *DataMoveSuite) TestAnActiveScheduledTaskArrivesPausedAndUnclaimed() {
	s.seedScheduledTask("st-active", "acme", ScheduledTaskActive)

	s.move("acme", "moved")

	state, claim, leased, next := s.scheduledTaskOn(s.destination, "st-active")
	s.Equal(ScheduledTaskPaused, state)
	s.Empty(claim)
	s.False(leased)
	s.Equal(time.Date(2026, 12, 1, 9, 0, 0, 0, time.UTC), next.UTC(), "its schedule is the one it had")

	state, claim, leased, _ = s.scheduledTaskOn(s.source, "st-active")
	s.Equal(ScheduledTaskActive, state, "the deployment it left goes on delivering it")
	s.Equal("claim-1", claim)
	s.True(leased)
}

func (s *DataMoveSuite) TestAScheduledTaskThatIsNotActiveKeepsItsState() {
	for _, state := range []string{ScheduledTaskPaused, ScheduledTaskCompleted, ScheduledTaskFailed} {
		s.seedScheduledTask("st-"+state, "acme", state)
	}

	s.move("acme", "moved")

	for _, want := range []string{ScheduledTaskPaused, ScheduledTaskCompleted, ScheduledTaskFailed} {
		state, claim, leased, _ := s.scheduledTaskOn(s.destination, "st-"+want)
		s.Equal(want, state)
		s.Empty(claim, "a worker's claim there is nothing here")
		s.False(leased)
	}
}

func (s *DataMoveSuite) TestAScheduledTaskChangeAppliedLaterArrivesPausedToo() {
	s.capture("acme")
	s.seedScheduledTask("st-first", "acme", ScheduledTaskActive)
	cursor := s.move("acme", "moved")

	_, err := s.source.DB().ExecContext(s.ctx, "UPDATE scheduled_tasks SET title = 'Stretch more' WHERE id = 'st-first'")
	s.Require().NoError(err)
	s.seedScheduledTask("st-second", "acme", ScheduledTaskActive)

	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "moved", changes))

	for _, id := range []string{"st-first", "st-second"} {
		state, claim, leased, _ := s.scheduledTaskOn(s.destination, id)
		s.Equal(ScheduledTaskPaused, state, id)
		s.Empty(claim, id)
		s.False(leased, id)
	}
	var title string
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT title FROM scheduled_tasks WHERE id = 'st-first'").Scan(&title))
	s.Equal("Stretch more", title, "the rest of the change is applied")
}

// resume is the cut-over on the destination: what it resumed, and the campaigns left to start.
func (s *DataMoveSuite) resume(customerID string) (int64, []string) {
	resumed, _, _, campaigns, err := s.destination.ResumeMovedRows(s.ctx, customerID)
	s.Require().NoError(err)
	return resumed, campaigns
}

// movePausedOn reads whether a task on the destination carries the move's mark.
func (s *DataMoveSuite) movePausedOn(id string) bool {
	var marked bool
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
		"SELECT move_paused FROM scheduled_tasks WHERE id = ?", id).Scan(&marked))
	return marked
}

func (s *DataMoveSuite) TestTheCutOverResumesOnlyTheTasksTheMovePaused() {
	s.seedScheduledTask("st-active", "acme", ScheduledTaskActive)
	s.seedScheduledTask("st-paused", "acme", ScheduledTaskPaused)
	s.seedScheduledTask("st-completed", "acme", ScheduledTaskCompleted)
	s.move("acme", "moved")

	resumed, _ := s.resume("moved")

	s.Equal(int64(1), resumed)
	state, claim, leased, next := s.scheduledTaskOn(s.destination, "st-active")
	s.Equal(ScheduledTaskActive, state)
	s.Empty(claim)
	s.False(leased)
	s.Equal(time.Date(2026, 12, 1, 9, 0, 0, 0, time.UTC), next.UTC(), "it goes on from the run the source had not delivered")
	state, _, _, _ = s.scheduledTaskOn(s.destination, "st-paused")
	s.Equal(ScheduledTaskPaused, state, "the user paused it")
	state, _, _, _ = s.scheduledTaskOn(s.destination, "st-completed")
	s.Equal(ScheduledTaskCompleted, state)

	// The user pauses it here once it is theirs; a second cut-over leaves it to them.
	task, err := s.destination.ScheduledTask(s.ctx, "moved", "st-active")
	s.Require().NoError(err)
	task.State = ScheduledTaskPaused
	s.Require().NoError(s.destination.SaveScheduledTask(s.ctx, &task))
	again, _ := s.resume("moved")
	s.Zero(again)
	state, _, _, _ = s.scheduledTaskOn(s.destination, "st-active")
	s.Equal(ScheduledTaskPaused, state)
}

func (s *DataMoveSuite) TestATaskTheUserPausedAtTheSourceDuringTheMoveStaysPaused() {
	s.capture("acme")
	s.seedScheduledTask("st-first", "acme", ScheduledTaskActive)
	cursor := s.move("acme", "moved")

	_, err := s.source.DB().ExecContext(s.ctx, "UPDATE scheduled_tasks SET state = 'paused' WHERE id = 'st-first'")
	s.Require().NoError(err)
	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "moved", changes))

	resumed, _ := s.resume("moved")

	s.Zero(resumed)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-first")
	s.Equal(ScheduledTaskPaused, state)
}

func (s *DataMoveSuite) TestATaskTheUserResumedAtTheSourceDuringTheMoveIsResumedAtTheCutOver() {
	s.capture("acme")
	s.seedScheduledTask("st-first", "acme", ScheduledTaskPaused)
	cursor := s.move("acme", "moved")

	_, err := s.source.DB().ExecContext(s.ctx, "UPDATE scheduled_tasks SET state = 'active' WHERE id = 'st-first'")
	s.Require().NoError(err)
	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "moved", changes))
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-first")
	s.Require().Equal(ScheduledTaskPaused, state, "the source still delivers it")

	resumed, _ := s.resume("moved")

	s.Equal(int64(1), resumed)
	state, _, _, _ = s.scheduledTaskOn(s.destination, "st-first")
	s.Equal(ScheduledTaskActive, state)
}

func (s *DataMoveSuite) TestATaskPausedAtTheSourceIsNotResumedWhateverMarkItCarries() {
	// A source that is itself a move's destination, not cut over yet, holds a paused task
	// marked by that move. The user did not pause it, but what this move sees is paused.
	s.seedScheduledTask("st-chained", "acme", ScheduledTaskPaused)
	s.moveEdited("acme", "moved", "scheduled_tasks", func(row map[string]any) { row["move_paused"] = true })

	resumed, _ := s.resume("moved")

	s.Zero(resumed)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-chained")
	s.Equal(ScheduledTaskPaused, state)
}

// limitHere sets how many active tasks a user may hold on config-1 on the destination.
func (s *DataMoveSuite) limitHere(limit int) {
	_, err := s.destination.DB().ExecContext(s.ctx,
		"UPDATE agent_configs SET scheduling = jsonb_build_object('max_tasks_per_user', ?::int) WHERE id = 'config-1'", limit)
	s.Require().NoError(err)
}

// madeHere is an active task a user made on the destination before the cut-over.
func (s *DataMoveSuite) madeHere(id, user, config string) {
	next := time.Date(2026, 12, 2, 9, 0, 0, 0, time.UTC)
	s.Require().NoError(s.destination.CreateScheduledTask(s.ctx, &ScheduledTask{
		ID: id, CustomerID: "moved", ConfigID: config, UserID: user, CallerKind: "authenticated",
		Title: "Hydrate", Instructions: "Remind them to drink.", StartLocal: next, Timezone: "UTC",
		State: ScheduledTaskActive, NextRunAt: &next, CreatedBy: "api",
	}, 100))
}

func (s *DataMoveSuite) TestTheCutOverResumesAUserOnlyUpToTheRoomTheirLimitLeaves() {
	for _, id := range []string{"st-1", "st-2", "st-3"} {
		s.seedScheduledTask(id, "acme", ScheduledTaskActive)
	}
	s.move("acme", "moved")
	s.limitHere(2)
	s.madeHere("st-here", "sam", "config-1")

	resumed, skipped, overLimit, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Equal(int64(1), resumed, "one place is left under the limit of 2")
	s.Zero(skipped)
	s.Equal(int64(2), overLimit)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-1")
	s.Equal(ScheduledTaskActive, state, "the oldest takes the place")
	s.False(s.movePausedOn("st-1"))
	for _, id := range []string{"st-2", "st-3"} {
		state, _, _, _ := s.scheduledTaskOn(s.destination, id)
		s.Equal(ScheduledTaskPaused, state, id)
		s.True(s.movePausedOn(id), "%s is still owed its resume", id)
	}
	held, err := s.destination.ActiveScheduledTaskCount(s.ctx, "moved", "sam", "config-1")
	s.Require().NoError(err)
	s.Equal(2, held)
}

// A client chooses the id of its task, so id order is not age order.
func (s *DataMoveSuite) TestTheCutOverResumesTheOldestTaskWhateverItsId() {
	for _, id := range []string{"st-c", "st-b", "st-a"} {
		s.seedScheduledTask(id, "acme", ScheduledTaskActive)
	}
	s.move("acme", "moved")
	s.limitHere(2)
	s.madeHere("st-here", "sam", "config-1")

	resumed, _, overLimit, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Equal(int64(1), resumed)
	s.Equal(int64(2), overLimit)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-c")
	s.Equal(ScheduledTaskActive, state, "st-c was made first, though its id sorts last")
	for _, id := range []string{"st-b", "st-a"} {
		state, _, _, _ := s.scheduledTaskOn(s.destination, id)
		s.Equal(ScheduledTaskPaused, state, id)
	}
}

// The mark kept on a task left over the limit is what lets a second cut-over resume it.
func (s *DataMoveSuite) TestTheCutOverRunAgainResumesWhatItLeftOnceTheUserHasRoom() {
	for _, id := range []string{"st-1", "st-2", "st-3"} {
		s.seedScheduledTask(id, "acme", ScheduledTaskActive)
	}
	s.move("acme", "moved")
	s.limitHere(2)
	s.madeHere("st-here", "sam", "config-1")
	_, _, _, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")
	s.Require().NoError(err)
	_, err = s.destination.DB().ExecContext(s.ctx, "UPDATE scheduled_tasks SET state = 'completed' WHERE id = 'st-here'")
	s.Require().NoError(err)

	resumed, _, overLimit, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Equal(int64(1), resumed)
	s.Equal(int64(1), overLimit)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-2")
	s.Equal(ScheduledTaskActive, state, "the next oldest")
	state, _, _, _ = s.scheduledTaskOn(s.destination, "st-3")
	s.Equal(ScheduledTaskPaused, state)
}

// Control: the limit is a user's on one config, and a task they paused here uses no place.
func (s *DataMoveSuite) TestTheCutOverResumesEveryTaskThatFitsUnderTheLimit() {
	s.seedScheduledTask("st-sam-1", "acme", ScheduledTaskActive)
	s.seedScheduledTask("st-sam-other-config", "acme", ScheduledTaskActive)
	s.seedScheduledTask("st-kim", "acme", ScheduledTaskActive)
	_, err := s.source.DB().ExecContext(s.ctx, `
INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES ('config-2', 'acme', 'b', 'be brief');
UPDATE scheduled_tasks SET config_id = 'config-2' WHERE id = 'st-sam-other-config';
UPDATE scheduled_tasks SET user_id = 'kim' WHERE id = 'st-kim'`)
	s.Require().NoError(err)
	s.move("acme", "moved")
	s.limitHere(2)
	_, err = s.destination.DB().ExecContext(s.ctx,
		`UPDATE agent_configs SET scheduling = '{"max_tasks_per_user": 2}' WHERE id = 'config-2'`)
	s.Require().NoError(err)
	s.madeHere("st-here", "sam", "config-1")
	s.madeHere("st-here-other-config", "sam", "config-2")
	s.madeHere("st-here-paused", "sam", "config-1")
	_, err = s.destination.DB().ExecContext(s.ctx, "UPDATE scheduled_tasks SET state = 'paused' WHERE id = 'st-here-paused'")
	s.Require().NoError(err)

	resumed, _, overLimit, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Equal(int64(3), resumed, "sam's second place on config-1, their task on the other config, and kim's")
	s.Zero(overLimit)
}

// queriesSent records the queries a pool sends while on.
type queriesSent struct {
	mu      sync.Mutex
	on      bool
	queries []string
}

func (*queriesSent) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context { return ctx }

func (q *queriesSent) AfterQuery(_ context.Context, event *bun.QueryEvent) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.on {
		q.queries = append(q.queries, event.Query)
	}
}

// matching is how many of the queries sent mention every one of words.
func (q *queriesSent) matching(words ...string) int {
	q.mu.Lock()
	defer q.mu.Unlock()
	matched := 0
	for _, query := range q.queries {
		all := true
		for _, word := range words {
			all = all && strings.Contains(query, word)
		}
		if all {
			matched++
		}
	}
	return matched
}

// Each user on each config has the room their own limit leaves them, read for all of them at
// once: one query locks the configs and one counts what every user holds on each, however
// many users and configs the move brought.
func (s *DataMoveSuite) TestTheCutOverReadsEveryUsersRoomOnEveryConfigInOneQueryEach() {
	tasks := map[string][2]string{
		"st-sam-1-a": {"sam", "config-1"}, "st-sam-1-b": {"sam", "config-1"}, "st-sam-1-c": {"sam", "config-1"},
		"st-sam-2-a": {"sam", "config-2"}, "st-sam-2-b": {"sam", "config-2"},
		"st-kim-1-a": {"kim", "config-1"}, "st-kim-1-b": {"kim", "config-1"},
		"st-kim-2-a": {"kim", "config-2"},
	}
	// Made in id order, so each holder's "-a" task is its oldest.
	for _, id := range slices.Sorted(maps.Keys(tasks)) {
		s.seedScheduledTask(id, "acme", ScheduledTaskActive)
	}
	_, err := s.source.DB().ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES ('config-2', 'acme', 'b', 'be brief')")
	s.Require().NoError(err)
	for id, holder := range tasks {
		_, err := s.source.DB().ExecContext(s.ctx,
			"UPDATE scheduled_tasks SET user_id = ?, config_id = ? WHERE id = ?", holder[0], holder[1], id)
		s.Require().NoError(err)
	}
	s.move("acme", "moved")
	s.limitHere(2)
	_, err = s.destination.DB().ExecContext(s.ctx,
		`UPDATE agent_configs SET scheduling = '{"max_tasks_per_user": 1}' WHERE id = 'config-2'`)
	s.Require().NoError(err)
	s.madeHere("st-here-sam-1", "sam", "config-1")
	s.madeHere("st-here-kim-2", "kim", "config-2")
	sent := &queriesSent{on: true}
	s.destination.DB().AddQueryHook(sent)

	resumed, skipped, overLimit, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	sent.mu.Lock()
	sent.on = false
	sent.mu.Unlock()
	s.Require().NoError(err)
	s.Equal(int64(4), resumed)
	s.Zero(skipped)
	s.Equal(int64(4), overLimit)
	want := map[string]string{
		// sam has one place left of 2 on config-1, and the 1 of config-2.
		"st-sam-1-a": ScheduledTaskActive, "st-sam-1-b": ScheduledTaskPaused, "st-sam-1-c": ScheduledTaskPaused,
		"st-sam-2-a": ScheduledTaskActive, "st-sam-2-b": ScheduledTaskPaused,
		// kim has both places of config-1, and none of config-2.
		"st-kim-1-a": ScheduledTaskActive, "st-kim-1-b": ScheduledTaskActive,
		"st-kim-2-a": ScheduledTaskPaused,
	}
	for id, state := range want {
		got, _, _, _ := s.scheduledTaskOn(s.destination, id)
		s.Equal(state, got, id)
	}
	s.Equal(1, sent.matching(`"agent_configs"`, "FOR SHARE"), "one query locks every config")
	s.Equal(1, sent.matching(`"scheduled_tasks"`, "count("), "one query counts every user on every config")

	sent.mu.Lock()
	sent.on, sent.queries = true, nil
	sent.mu.Unlock()
	_, _, _, _, err = s.destination.ResumeMovedRows(s.ctx, "nobody")
	s.Require().NoError(err)
	s.Zero(sent.matching(`"agent_configs"`), "a cut-over with nothing to resume reads no config")
	s.Zero(sent.matching("count("), "nor counts anybody's tasks")
}

// Another customer's tasks take none of this customer's room, though they name the same user
// and config ids: no key ties a task's config to its customer.
func (s *DataMoveSuite) TestTheCutOverCountsOnlyTheCustomersOwnTasks() {
	s.seedScheduledTask("st-1", "acme", ScheduledTaskActive)
	s.move("acme", "moved")
	s.limitHere(1)
	_, err := s.destination.DB().ExecContext(s.ctx, `
INSERT INTO scheduled_tasks (id, customer_id, config_id, user_id, caller_kind, title, instructions, start_local,
	timezone, state, next_run_at, created_by, created_at, updated_at)
VALUES ('st-theirs', 'other', 'config-1', 'sam', 'authenticated', 'Hydrate', 'Remind Sam to drink.', now(),
	'UTC', 'active', now() + interval '1 day', 'api', now(), now())`)
	s.Require().NoError(err)

	resumed, _, overLimit, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Equal(int64(1), resumed)
	s.Zero(overLimit)
}

func (s *DataMoveSuite) TestTheCutOverHoldsAConfigWithoutALimitToTheDefault() {
	s.seedScheduledTask("st-1", "acme", ScheduledTaskActive)
	s.move("acme", "moved")
	for i := range 25 {
		s.madeHere(fmt.Sprintf("st-here-%d", i), "sam", "config-1")
	}

	resumed, _, overLimit, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Zero(resumed)
	s.Equal(int64(1), overLimit)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-1")
	s.Equal(ScheduledTaskPaused, state)
}

func (s *DataMoveSuite) TestTheCutOverResumesNothingForAUserAlreadyAboveALoweredLimit() {
	s.seedScheduledTask("st-1", "acme", ScheduledTaskActive)
	s.move("acme", "moved")
	s.madeHere("st-here-1", "sam", "config-1")
	s.madeHere("st-here-2", "sam", "config-1")
	s.limitHere(1)

	resumed, _, overLimit, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Zero(resumed)
	s.Equal(int64(1), overLimit)
}

func (s *DataMoveSuite) TestTheCutOverLeavesATaskWhoseConfigWasDeletedHerePaused() {
	s.seedScheduledTask("st-active", "acme", ScheduledTaskActive)
	s.move("acme", "moved")
	// What deleting the config over the API does here.
	s.Require().NoError(s.destination.DeleteAgentConfig(s.ctx, "moved", "config-1"))
	s.Require().NoError(s.destination.PauseScheduledTasksForConfig(s.ctx, "moved", "config-1"))

	resumed, skipped, _, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Zero(resumed, "every occurrence would fail on the deleted agent")
	s.Equal(int64(1), skipped)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-active")
	s.Equal(ScheduledTaskPaused, state)
	s.False(s.movePausedOn("st-active"), "a second cut-over has nothing of it to resume either")
}

// A repeating task the source delivers changes there each time, so the last --once imports
// it, and marks it, again after the config was deleted here.
func (s *DataMoveSuite) TestTheCutOverLeavesATaskWhoseConfigWasDeletedHereBeforeTheLastCatchUpPaused() {
	s.capture("acme")
	s.seedScheduledTask("st-active", "acme", ScheduledTaskActive)
	cursor := s.move("acme", "moved")
	s.Require().NoError(s.destination.DeleteAgentConfig(s.ctx, "moved", "config-1"))
	s.Require().NoError(s.destination.PauseScheduledTasksForConfig(s.ctx, "moved", "config-1"))
	_, err := s.source.DB().ExecContext(s.ctx,
		"UPDATE scheduled_tasks SET next_run_at = next_run_at + interval '1 day', updated_at = now() WHERE id = 'st-active'")
	s.Require().NoError(err)
	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "moved", changes))
	s.Require().True(s.movePausedOn("st-active"), "the catch-up marked it again")

	resumed, skipped, _, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Zero(resumed, "its config is deleted here")
	s.Equal(int64(1), skipped)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-active")
	s.Equal(ScheduledTaskPaused, state)
	s.False(s.movePausedOn("st-active"))
}

func (s *DataMoveSuite) TestTheCutOverLeavesATaskWhoseConfigIsAnotherCustomersHerePaused() {
	s.seedScheduledTask("st-active", "acme", ScheduledTaskActive)
	// The destination holds config-1 for somebody else, so the move cannot bring it.
	_, err := s.destination.DB().ExecContext(s.ctx,
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES ('config-1', 'resident', 'theirs', 'be brief')")
	s.Require().NoError(err)
	s.move("acme", "moved")

	resumed, skipped, _, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Zero(resumed, "the customer has no config-1 here to run it on")
	s.Equal(int64(1), skipped)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-active")
	s.Equal(ScheduledTaskPaused, state)
}

// A delete in flight when the cut-over reads the config is waited for, not read past.
func (s *DataMoveSuite) TestTheCutOverWaitsForADeleteOfTheConfigInFlight() {
	s.seedScheduledTask("st-active", "acme", ScheduledTaskActive)
	s.move("acme", "moved")
	deleting, err := s.destination.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = deleting.Rollback() }()
	_, err = deleting.ExecContext(s.ctx, "UPDATE agent_configs SET deleted_at = now() WHERE id = 'config-1'")
	s.Require().NoError(err)

	type outcome struct {
		resumed int64
		err     error
	}
	done := make(chan outcome, 1)
	go func() {
		resumed, _, _, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")
		done <- outcome{resumed, err}
	}()
	// Commit the delete once the cut-over waits on it; a cut-over that does not wait has
	// finished by then.
	deadline := time.Now().Add(30 * time.Second)
	var result *outcome
	for result == nil && time.Now().Before(deadline) {
		select {
		case got := <-done:
			result = &got
		case <-time.After(50 * time.Millisecond):
			var waiting int
			s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
				"SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'").Scan(&waiting))
			if waiting > 0 {
				s.Require().NoError(deleting.Commit())
				got := <-done
				result = &got
			}
		}
	}
	s.Require().NotNil(result, "the cut-over neither finished nor waited")
	s.Require().NoError(result.err)
	s.Zero(result.resumed, "the config was deleted before the cut-over could resume its task")
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-active")
	s.Equal(ScheduledTaskPaused, state)
}

// A task the user or its delivery changed after the cut-over read it is not resumed by what it read.
func (s *DataMoveSuite) TestTheCutOverDoesNotResumeATaskThatChangedWhileItReadTheRest() {
	s.seedScheduledTask("st-ended", "acme", ScheduledTaskActive)
	s.seedScheduledTask("st-paused", "acme", ScheduledTaskActive)
	// Two tasks on a config deleted here, which the cut-over leaves paused and unmarks
	// unless they change meanwhile.
	s.seedScheduledTask("st-gone-ended", "acme", ScheduledTaskActive)
	s.seedScheduledTask("st-gone-paused", "acme", ScheduledTaskActive)
	_, err := s.source.DB().ExecContext(s.ctx, `
INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES ('config-2', 'acme', 'b', 'be brief');
UPDATE scheduled_tasks SET config_id = 'config-2' WHERE id IN ('st-gone-ended', 'st-gone-paused')`)
	s.Require().NoError(err)
	s.move("acme", "moved")
	s.Require().NoError(s.destination.DeleteAgentConfig(s.ctx, "moved", "config-2"))
	changing, err := s.destination.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = changing.Rollback() }()
	_, err = changing.ExecContext(s.ctx, `
UPDATE scheduled_tasks SET state = 'completed' WHERE id IN ('st-ended', 'st-gone-ended');
UPDATE scheduled_tasks SET move_paused = false WHERE id IN ('st-paused', 'st-gone-paused')`)
	s.Require().NoError(err)

	type outcome struct {
		resumed, skipped int64
		err              error
	}
	done := make(chan outcome, 1)
	go func() {
		resumed, skipped, _, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")
		done <- outcome{resumed, skipped, err}
	}()
	// Commit the changes once the cut-over waits on them: it read the tasks as they were.
	deadline := time.Now().Add(30 * time.Second)
	var result *outcome
	for result == nil && time.Now().Before(deadline) {
		select {
		case got := <-done:
			result = &got
		case <-time.After(50 * time.Millisecond):
			var waiting int
			s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx,
				"SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'").Scan(&waiting))
			if waiting > 0 {
				s.Require().NoError(changing.Commit())
				got := <-done
				result = &got
			}
		}
	}
	s.Require().NotNil(result, "the cut-over neither finished nor waited")
	s.Require().NoError(result.err)
	s.Zero(result.resumed)
	s.Zero(result.skipped, "a task that changed meanwhile was not left paused by the cut-over")
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-ended")
	s.Equal(ScheduledTaskCompleted, state)
	state, _, _, _ = s.scheduledTaskOn(s.destination, "st-paused")
	s.Equal(ScheduledTaskPaused, state, "the user's pause, which cleared the mark, stands")
	state, _, _, _ = s.scheduledTaskOn(s.destination, "st-gone-ended")
	s.Equal(ScheduledTaskCompleted, state)
}

func (s *DataMoveSuite) TestTheCutOverLeavesATaskThatEndedHereAsItIs() {
	s.seedScheduledTask("st-active", "acme", ScheduledTaskActive)
	s.move("acme", "moved")
	// Resumed here by hand and delivered: the mark is what the move left, the state is not.
	_, err := s.destination.DB().ExecContext(s.ctx,
		"UPDATE scheduled_tasks SET state = 'completed' WHERE id = 'st-active'")
	s.Require().NoError(err)

	resumed, skipped, _, _, err := s.destination.ResumeMovedRows(s.ctx, "moved")

	s.Require().NoError(err)
	s.Zero(resumed)
	s.Zero(skipped, "it was not left paused")
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-active")
	s.Equal(ScheduledTaskCompleted, state)
}

func (s *DataMoveSuite) TestTheCutOverIsOnlyTheCustomersOwn() {
	s.seedScheduledTask("st-acme", "acme", ScheduledTaskActive)
	s.seedScheduledTask("st-other", "other", ScheduledTaskActive)
	acmeCampaign := s.seedCampaign("acme", Running)
	otherCampaign := s.seedCampaign("other", Running)
	s.move("acme", "moved")
	s.move("other", "other")

	resumed, campaigns := s.resume("moved")

	s.Equal(int64(1), resumed)
	s.Equal([]string{acmeCampaign}, campaigns)
	state, _, _, _ := s.scheduledTaskOn(s.destination, "st-other")
	s.Equal(ScheduledTaskPaused, state, "another customer's move is cut over on its own")
	s.True(s.movePausedOn("st-other"), "and keeps its mark for it")
	_, campaigns = s.resume("other")
	s.Equal([]string{otherCampaign}, campaigns)
}

// seedCampaign stores a campaign on the source in a state and returns its id.
func (s *DataMoveSuite) seedCampaign(customerID, state string) string {
	campaign := &Campaign{CustomerID: customerID, Name: "Renewals", ConfigID: "config-1", FromNumber: "+15550100"}
	s.Require().NoError(s.source.CreateCampaign(s.ctx, campaign))
	if state != Draft {
		s.Require().NoError(s.source.SetCampaignState(s.ctx, customerID, campaign.ID, state))
	}
	return campaign.ID
}

func (s *DataMoveSuite) TestARunningCampaignArrivesPausedAndTheCutOverNamesIt() {
	running := s.seedCampaign("acme", Running)
	paused := s.seedCampaign("acme", Paused)
	draft := s.seedCampaign("acme", Draft)
	s.move("acme", "moved")

	_, campaigns := s.resume("moved")

	s.Equal([]string{running}, campaigns)
	for id, want := range map[string]string{running: Paused, paused: Paused, draft: Draft} {
		campaign, err := s.destination.Campaign(s.ctx, "moved", id)
		s.Require().NoError(err)
		s.Equal(want, campaign.State, "only starting it here rings anybody")
	}
	source, err := s.source.Campaign(s.ctx, "acme", running)
	s.Require().NoError(err)
	s.Equal(Running, source.State, "the deployment it left goes on ringing")
}

func (s *DataMoveSuite) TestACampaignStartedHereIsNoLongerTheMovesToName() {
	running := s.seedCampaign("acme", Running)
	s.move("acme", "moved")

	s.Require().NoError(s.destination.SetCampaignState(s.ctx, "moved", running, Running))
	s.Require().NoError(s.destination.SetCampaignState(s.ctx, "moved", running, Paused))

	_, campaigns := s.resume("moved")
	s.Empty(campaigns, "the customer paused it here")
}

// ringOnSource rings a campaign's next contact on the source, as its runner does, and returns
// it in a call.
func (s *DataMoveSuite) ringOnSource(campaignID string) Contact {
	contact, found, err := s.source.ClaimContact(s.ctx, campaignID)
	s.Require().NoError(err)
	s.Require().True(found)
	return contact
}

// hangUpOnSource ends a call the source rang.
func (s *DataMoveSuite) hangUpOnSource(contact Contact) {
	contact.State = Done
	s.Require().NoError(s.source.FinishContact(s.ctx, contact))
}

// startHere starts a moved campaign on the destination as campaign.Runner.Start does, and
// returns the contact it rings first.
func (s *DataMoveSuite) startHere(customerID, campaignID string) Contact {
	s.Require().NoError(s.destination.ReleaseContacts(s.ctx, campaignID))
	s.Require().NoError(s.destination.SetCampaignState(s.ctx, customerID, campaignID, Running))
	contact, found, err := s.destination.ClaimContact(s.ctx, campaignID)
	s.Require().NoError(err)
	s.Require().True(found)
	return contact
}

// The cut-over the docs give: pause the campaign at the source, let its calls end, then run the
// last --once. The pause clears the move's mark, so --resume leaves the campaign alone, and
// starting it here rings only those the source never rang.
func (s *DataMoveSuite) TestACampaignPausedAtTheSourceBeforeTheLastCatchUpRingsNobodyTwiceHere() {
	s.capture("acme")
	running := s.seedCampaign("acme", Running)
	s.Require().NoError(s.source.AddContacts(s.ctx, []Contact{
		{CampaignID: running, ToNumber: "+15550101"}, {CampaignID: running, ToNumber: "+15550102"}}))
	inCall := s.ringOnSource(running)
	cursor := s.move("acme", "moved")

	s.Require().NoError(s.source.SetCampaignState(s.ctx, "acme", running, Paused))
	s.hangUpOnSource(inCall)
	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "moved", changes))

	_, campaigns := s.resume("moved")
	s.Empty(campaigns, "the user paused it at the source")
	campaign, err := s.destination.Campaign(s.ctx, "moved", running)
	s.Require().NoError(err)
	s.Equal(Paused, campaign.State, "it rings here only once someone starts it")

	next := s.startHere("moved", running)
	s.NotEqual(inCall.ID, next.ID, "the source rang them")
	s.Equal("+15550102", next.ToNumber)
	s.Equal(1, next.Attempts)
}

// A campaign running at the source when the last --once read it stays paused here, and the
// cut-over names it. The contacts the source rang up to then are done here.
func (s *DataMoveSuite) TestACampaignTheCopyPausedIsNamedAndStartsWhereTheCopyLeftOff() {
	running := s.seedCampaign("acme", Running)
	s.Require().NoError(s.source.AddContacts(s.ctx, []Contact{
		{CampaignID: running, ToNumber: "+15550101"}, {CampaignID: running, ToNumber: "+15550102"}}))
	s.hangUpOnSource(s.ringOnSource(running))
	s.move("acme", "moved")

	_, campaigns := s.resume("moved")
	s.Equal([]string{running}, campaigns)
	campaign, err := s.destination.Campaign(s.ctx, "moved", running)
	s.Require().NoError(err)
	s.Equal(Paused, campaign.State, "the cut-over starts no campaign")

	next := s.startHere("moved", running)
	s.Equal("+15550102", next.ToNumber)
	s.Equal(1, next.Attempts)
}

// A campaign the user paused at the source without waiting for its calls to end arrives paused
// and unmarked with a contact still calling. The cut-over names it with how many, since a start
// here rings them again.
func (s *DataMoveSuite) TestACampaignPausedAtTheSourceWithContactsCallingIsNamedWithTheirCount() {
	s.capture("acme")
	inCalls := s.seedCampaign("acme", Running)
	quiet := s.seedCampaign("acme", Running)
	s.Require().NoError(s.source.AddContacts(s.ctx, []Contact{
		{CampaignID: inCalls, ToNumber: "+15550101"}, {CampaignID: inCalls, ToNumber: "+15550102"},
		{CampaignID: inCalls, ToNumber: "+15550103"}, {CampaignID: quiet, ToNumber: "+15550104"}}))
	s.ringOnSource(inCalls)
	s.ringOnSource(inCalls)
	s.hangUpOnSource(s.ringOnSource(quiet))
	cursor := s.move("acme", "moved")
	s.Require().NoError(s.source.SetCampaignState(s.ctx, "acme", inCalls, Paused))
	s.Require().NoError(s.source.SetCampaignState(s.ctx, "acme", quiet, Paused))
	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "moved", changes))

	_, campaigns := s.resume("moved")
	s.Empty(campaigns, "the user paused both at the source")
	calling, err := s.destination.PausedCampaignsCalling(s.ctx, "moved")
	s.Require().NoError(err)
	s.Equal([]CampaignCalls{{ID: inCalls, Calls: 2}}, calling, "only the one with contacts in a call, counted")

	first := s.startHere("moved", inCalls)
	s.Equal(2, first.Attempts, "a start here rings the one the source was calling again")
}

func (s *DataMoveSuite) TestACampaignNotPausedHereIsNotNamedForItsCalls() {
	running := s.seedCampaign("acme", Running)
	s.Require().NoError(s.source.AddContacts(s.ctx, []Contact{{CampaignID: running, ToNumber: "+15550101"}}))
	s.ringOnSource(running)
	marked := s.seedCampaign("acme", Running)
	s.Require().NoError(s.source.AddContacts(s.ctx, []Contact{{CampaignID: marked, ToNumber: "+15550103"}}))
	s.ringOnSource(marked)
	other := s.seedCampaign("other", Running)
	s.Require().NoError(s.source.AddContacts(s.ctx, []Contact{{CampaignID: other, ToNumber: "+15550102"}}))
	s.ringOnSource(other)
	s.Require().NoError(s.source.SetCampaignState(s.ctx, "other", other, Paused))
	s.move("acme", "moved")
	s.move("other", "moved-other")
	s.startHere("moved", running)

	calling, err := s.destination.PausedCampaignsCalling(s.ctx, "moved")
	s.Require().NoError(err)
	s.Empty(calling, "one started here rings its own calls, the move's mark names its own, and another customer's is not its")
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

func (s *DataMoveSuite) TestAMovedNumberArrivesNotAttached() {
	s.Require().NoError(s.source.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000113", Vendor: "telnyx", Country: "US", CustomerID: "acme",
	}))
	s.Require().NoError(s.source.AttachNumber(s.ctx, "acme", "+15550000113", NumberAttachment{
		TrunkID: "trunk-1", RouteID: "route-1", CallType: "default",
	}))

	s.move("acme", "acme")

	moved, err := s.destination.Number(s.ctx, "acme", "+15550000113")
	s.Require().NoError(err)
	s.Nil(moved.AttachedAt)
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

func (s *DataMoveSuite) TestCopyingAgainKeepsANumberAttachedOnTheDestination() {
	attached := s.numberAttachedHere("+15550000130")
	s.retag(s.source, "+15550000130")

	s.move("acme", "acme")

	s.assertLineKept(attached)
}

func (s *DataMoveSuite) TestAChangeAppliedLaterKeepsANumberAttachedOnTheDestination() {
	s.capture("acme")
	attached := s.numberAttachedHere("+15550000131")
	// The source attaches it too, to another agent, and its row says so in the change.
	s.Require().NoError(s.source.AttachNumber(s.ctx, "acme", "+15550000131", NumberAttachment{
		TrunkID: "trunk-there", RouteID: "route-there", CallType: "default", CallID: "call-there",
		ConfigID: s.seedAgentConfig(s.source, "acme", "there"), BridgeURI: "sip:there.example.com",
	}))
	s.retag(s.source, "+15550000131")

	changes, _, err := s.source.Changes(s.ctx, "acme", 0, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))

	s.assertLineKept(attached)
}

func (s *DataMoveSuite) TestCopyingAgainKeepsANumbersLineOnTheDestination() {
	s.Require().NoError(s.source.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000132", Vendor: "telnyx", Country: "US", CustomerID: "acme",
	}))
	s.move("acme", "acme")
	// On a trunk to place calls, not attached.
	s.Require().NoError(s.destination.InTrunkGroup(s.ctx, TrunkGroup{CustomerID: "acme", StreamAppPK: 42, Vendor: "telnyx"},
		func(_ context.Context, tx *TrunkGroupTx) error {
			return tx.SetLine("+15550000132", "trunk-here", "route-here", NumberBridge{
				URI: "sip:here.example.com", Username: "agent-here", PasswordSealed: []byte("sealed here"), KEKVersion: 1,
			})
		}))
	lined, err := s.destination.Number(s.ctx, "acme", "+15550000132")
	s.Require().NoError(err)
	s.retag(s.source, "+15550000132")

	s.move("acme", "acme")

	s.assertLineKept(lined)
}

func (s *DataMoveSuite) TestCopyingAgainStillUpdatesANumberNotOnATrunkOnTheDestination() {
	s.Require().NoError(s.source.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000133", Vendor: "telnyx", Country: "US", CustomerID: "acme",
	}))
	s.move("acme", "acme")
	there := s.seedAgentConfig(s.source, "acme", "there")
	s.Require().NoError(s.source.AttachNumber(s.ctx, "acme", "+15550000133", NumberAttachment{
		TrunkID: "trunk-there", RouteID: "route-there", StreamAppPK: 42, CallType: "default", CallID: "call-there",
		ConfigID: there, BridgeURI: "sip:there.example.com", BridgeUsername: "agent-there",
	}))
	s.retag(s.source, "+15550000133")

	s.move("acme", "acme")

	// What the source's row says, as before a number on a trunk here kept its line.
	moved, err := s.destination.Number(s.ctx, "acme", "+15550000133")
	s.Require().NoError(err)
	s.Equal(map[string]string{"team": "sales"}, moved.Tags)
	s.Equal(there, moved.ConfigID)
	s.Equal(ForeignStreamApp, moved.StreamAppPK)
	s.Nil(moved.AttachedAt)
	s.Empty(moved.StreamTrunkID)
	s.Empty(moved.StreamCallID)
	s.Empty(moved.BridgeURI)
}

func (s *DataMoveSuite) TestCopyingAgainBringsTheSourcesAgentToANumberOnlyOnALineOnTheDestination() {
	first := s.seedAgentConfig(s.source, "acme", "first")
	s.Require().NoError(s.source.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15550000134", Vendor: "telnyx", Country: "US", CustomerID: "acme",
	}))
	s.Require().NoError(s.source.AttachNumber(s.ctx, "acme", "+15550000134", NumberAttachment{
		TrunkID: "trunk-there", RouteID: "route-there", CallType: "default", CallID: "call-there", ConfigID: first,
	}))
	s.move("acme", "acme")
	// On a line to place calls, which never wrote an agent.
	s.Require().NoError(s.destination.InTrunkGroup(s.ctx, TrunkGroup{CustomerID: "acme", StreamAppPK: 42, Vendor: "telnyx"},
		func(_ context.Context, tx *TrunkGroupTx) error {
			return tx.SetLine("+15550000134", "trunk-here", "route-here", NumberBridge{})
		}))
	second := s.seedAgentConfig(s.source, "acme", "second")
	s.Require().NoError(s.source.SetNumberAgent(s.ctx, "acme", "+15550000134", second))

	s.move("acme", "acme")

	moved, err := s.destination.Number(s.ctx, "acme", "+15550000134")
	s.Require().NoError(err)
	s.Equal(second, moved.ConfigID, "the source's agent arrives")
	s.Equal("trunk-here", moved.StreamTrunkID, "the line stays")
}

func (s *DataMoveSuite) TestCopyingAgainDoesNotKeepTheLineOfANumberTheSourceGaveBack() {
	attached := s.numberAttachedHere("+15550000135")
	s.Require().NoError(s.source.ReleaseNumber(s.ctx, "acme", "+15550000135", time.Now()))

	s.move("acme", "acme")

	numbers, err := s.destination.CustomerNumbers(s.ctx, "acme", true)
	s.Require().NoError(err)
	var moved *PhoneNumber
	for i := range numbers {
		if numbers[i].E164 == attached.E164 {
			moved = &numbers[i]
		}
	}
	s.Require().NotNil(moved)
	s.NotNil(moved.ReleasedAt)
	s.Nil(moved.AttachedAt)
	s.Empty(moved.StreamTrunkID)
	s.Empty(moved.StreamRouteID)
	s.Empty(moved.ConfigID)
}

// numberAttachedHere is a number the source holds, copied here and attached here, as it reads
// here once attached.
func (s *DataMoveSuite) numberAttachedHere(e164 string) PhoneNumber {
	s.Require().NoError(s.source.RecordNumber(s.ctx, &PhoneNumber{
		E164: e164, Vendor: "telnyx", Country: "US", CustomerID: "acme",
	}))
	s.move("acme", "acme")
	s.Require().NoError(s.destination.AttachNumber(s.ctx, "acme", e164, NumberAttachment{
		TrunkID: "trunk-here", RouteID: "route-here", StreamAppPK: 42, CallType: "default", CallID: "call-here",
		ConfigID: s.seedAgentConfig(s.destination, "acme", "here"), BridgeURI: "sip:here.example.com",
		BridgeUsername: "agent-here", BridgePasswordSealed: []byte("sealed here"), BridgePasswordKEKVersion: 1,
	}))
	attached, err := s.destination.Number(s.ctx, "acme", e164)
	s.Require().NoError(err)
	return attached
}

// retag edits a number on the source the way its owner would, in a column of its own.
func (s *DataMoveSuite) retag(store *Store, e164 string) {
	_, err := store.DB().ExecContext(s.ctx, `UPDATE phone_numbers SET tags = '{"team": "sales"}' WHERE e164 = ?`, e164)
	s.Require().NoError(err)
}

// assertLineKept checks the number reads here as it did before the source's row came again,
// except for the source's edit, which arrives.
func (s *DataMoveSuite) assertLineKept(before PhoneNumber) {
	after, err := s.destination.Number(s.ctx, "acme", before.E164)
	s.Require().NoError(err)
	s.Equal(map[string]string{"team": "sales"}, after.Tags, "the source's edit arrives")
	before.Tags = after.Tags
	s.Equal(before, after)
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

// movedRequest imports a request recorded in the source, as an export from before usage ids
// carries it when old is set, and returns the destination's row.
func (s *DataMoveSuite) movedRequest(old bool) map[string]any {
	exported := s.export("acme").of("requests")
	s.Require().Len(exported, 1)
	if old {
		exported[0]["usage_id"] = nil
	}
	encoded, err := json.Marshal(exported[0])
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ImportRow(s.ctx, "acme", "requests", encoded))
	moved := s.exportFrom(s.destination, "acme").of("requests")
	s.Require().Len(moved, 1)
	return moved[0]
}

func (s *DataMoveSuite) TestARequestFromBeforeUsageIDsIsImportedWithTheSameIDEveryTime() {
	s.Require().NoError(s.source.RecordRequest(s.ctx, &Request{Modality: "stt", CustomerID: "acme",
		Provider: "deepgram", Model: "flux", StartedAt: time.Now().UTC(), Success: true}))

	first := s.movedRequest(true)
	again := s.movedRequest(true)

	s.NotEmpty(first["usage_id"], "the table refuses a new row without a usage id")
	s.Equal(first["usage_id"], again["usage_id"], "importing it again names the same unit of work")
}

func (s *DataMoveSuite) TestAnImportedRequestKeepsItsUsageIDAndIsNeverShipped() {
	usageID := uuid.Must(uuid.NewV7())
	shippedAt := time.Now().UTC().Add(-time.Minute)
	s.Require().NoError(s.source.RecordRequest(s.ctx, &Request{UsageID: usageID, Modality: "stt", CustomerID: "acme",
		Provider: "deepgram", Model: "flux", StartedAt: time.Now().UTC(), Success: true, ToShip: true,
		ShippedAt: &shippedAt, ShipError: "missing_api_key"}))

	moved := s.movedRequest(false)

	s.Equal(usageID.String(), moved["usage_id"])
	s.Equal(false, moved["to_ship"], "the deployment that recorded it bills it")
	s.Nil(moved["shipped_at"], "what the other deployment's usage service answered stays there")
	s.Equal("", moved["ship_error"])
}

// takenRequest writes a request of acme's with the id given, as two deployments' own
// sequences can each hand out.
func (s *DataMoveSuite) takenRequest(store *Store, id int64) *Request {
	request := &Request{ID: id, UsageID: uuid.Must(uuid.NewV7()), Modality: "stt", CustomerID: "acme",
		Provider: "deepgram", Model: "flux", StartedAt: time.Now().UTC(), Success: true, ToShip: true}
	_, err := store.ShardDB().NewInsert().Model(request).Exec(s.ctx)
	s.Require().NoError(err)
	return request
}

// requestsByUsage is the destination's requests of acme by usage id.
func (s *DataMoveSuite) requestsByUsage() map[string]map[string]any {
	byUsage := map[string]map[string]any{}
	for _, row := range s.exportFrom(s.destination, "acme").of("requests") {
		byUsage[row["usage_id"].(string)] = row
	}
	return byUsage
}

func (s *DataMoveSuite) TestAMovedRequestNeverOverwritesTheDestinationsRowWithItsID() {
	id := time.Now().UnixNano() % 1_000_000_000_000
	own := s.takenRequest(s.destination, id)
	moved := s.takenRequest(s.source, id)

	s.move("acme", "acme")
	s.move("acme", "acme")

	rows := s.requestsByUsage()
	s.Require().Len(rows, 2, "the moved row beside the destination's own, once however often it is imported")
	s.EqualValues(id, rows[own.UsageID.String()]["id"])
	s.Equal(true, rows[own.UsageID.String()]["to_ship"], "the destination's own row is left as it was")
	s.NotEqualValues(id, rows[moved.UsageID.String()]["id"], "the moved row is given an id of the destination's")
	s.Equal(false, rows[moved.UsageID.String()]["to_ship"])
}

func (s *DataMoveSuite) TestADeletedRequestIsReplayedByItsUsageID() {
	s.capture("acme")
	id := time.Now().UnixNano() % 1_000_000_000_000
	doomed := s.takenRequest(s.source, id)
	own := s.takenRequest(s.destination, id)
	cursor := s.move("acme", "acme")
	s.Require().Len(s.requestsByUsage(), 2)

	_, err := s.source.ShardDB().ExecContext(s.ctx, "DELETE FROM requests WHERE id = ?", id)
	s.Require().NoError(err)
	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().Len(changes, 1)
	s.Require().Equal(ChangeDelete, changes[0].Op)
	s.JSONEq(fmt.Sprintf(`{"usage_id": %q, "id": %d}`, doomed.UsageID, id), string(changes[0].Key))
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))

	rows := s.requestsByUsage()
	s.NotContains(rows, doomed.UsageID.String())
	s.Contains(rows, own.UsageID.String(), "the destination's row with the deleted row's id is not the row deleted")
}

func (s *DataMoveSuite) TestADeletedRequestFromBeforeUsageIDsIsReplayedToo() {
	s.Require().NoError(s.source.RecordRequest(s.ctx, &Request{Modality: "stt", CustomerID: "acme",
		Provider: "deepgram", Model: "flux", StartedAt: time.Now().UTC(), Success: true}))
	moved := s.movedRequest(true)
	sourceID := int64(s.export("acme").of("requests")[0]["id"].(float64))
	kept := s.takenRequest(s.destination, sourceID+1_000_000_000)

	// What the source records deleting a row the migration left without a usage id.
	key := fmt.Sprintf(`{"usage_id": null, "id": %d}`, sourceID)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", []DataChange{
		{Table: "requests", Op: ChangeDelete, Key: json.RawMessage(key)},
	}))

	rows := s.requestsByUsage()
	s.NotContains(rows, moved["usage_id"], "found by the usage id its import made from its source id")
	s.Contains(rows, kept.UsageID.String())
}

// drewNothing restarts the sequence behind a column, so the store has drawn nothing from it yet.
// A truncate leaves sequences where they were, and the databases outlive a run.
func (s *DataMoveSuite) drewNothing(store *Store, table, column string) {
	_, err := store.DB().ExecContext(s.ctx, "SELECT setval(pg_get_serial_sequence(?, ?), 1, false)", table, column)
	s.Require().NoError(err)
}

func (s *DataMoveSuite) TestAContactAddedToAMovedCampaignIsRungAfterTheMovedOnes() {
	s.drewNothing(s.destination, "campaign_contacts", "seq")
	campaign := &Campaign{CustomerID: "acme", Name: "renewals"}
	s.Require().NoError(s.source.CreateCampaign(s.ctx, campaign))
	// The source has handed out more places than the destination: its contacts' places are
	// numbers the destination's sequence has not reached.
	_, err := s.source.DB().ExecContext(s.ctx, "SELECT setval(pg_get_serial_sequence('campaign_contacts', 'seq'), 1000)")
	s.Require().NoError(err)
	s.Require().NoError(s.source.AddContacts(s.ctx, []Contact{{CampaignID: campaign.ID, ToNumber: "+15550000001"}}))

	s.move("acme", "acme")
	s.Require().NoError(s.destination.AddContacts(s.ctx, []Contact{{CampaignID: campaign.ID, ToNumber: "+15550000002"}}))

	contacts, err := s.destination.CampaignContacts(s.ctx, campaign.ID)
	s.Require().NoError(err)
	s.Require().Len(contacts, 2)
	s.Equal("+15550000001", contacts[0].ToNumber, "a contact added here is rung after the ones that moved")
	s.Equal("+15550000002", contacts[1].ToNumber)
}

func (s *DataMoveSuite) TestAMovedRequestLeavesTheSequenceOfItsIDAlone() {
	s.drewNothing(s.destination, "requests", "id")
	s.takenRequest(s.source, 1_000_000_000)

	s.move("acme", "acme")
	usageID := uuid.Must(uuid.NewV7())
	s.Require().NoError(s.destination.RecordRequest(s.ctx, &Request{UsageID: usageID, Modality: "stt", CustomerID: "acme",
		Provider: "deepgram", Model: "flux", StartedAt: time.Now().UTC(), Success: true}))

	own := s.requestsByUsage()[usageID.String()]
	s.Require().NotNil(own)
	s.Less(own["id"].(float64), float64(1_000_000_000), "a moved request is given an id of the destination's, so the source's is reserved nowhere")
}

// seedLog writes a log of the customer's on the source, the way the router does, and answers its id.
func (s *DataMoveSuite) seedLog(customerID, message string) int64 {
	entry := &AgentLog{CustomerID: customerID, Source: "agent", Severity: "info", EventType: "turn", Message: message}
	s.Require().NoError(s.source.RecordAgentLog(s.ctx, entry))
	return entry.ID
}

// placeHere writes an agent log onto the destination as given, id included: the way an import
// from before AI-2445 wrote a copied one, or a log of the destination's own.
func (s *DataMoveSuite) placeHere(row map[string]any) {
	encoded, err := json.Marshal(row)
	s.Require().NoError(err)
	_, err = s.destination.DB().ExecContext(s.ctx,
		"INSERT INTO agent_logs SELECT * FROM jsonb_populate_record(NULL::agent_logs, ?::jsonb)", string(encoded))
	s.Require().NoError(err)
	// An explicit id leaves the sequence where it was, and SetupTest truncates without
	// restarting it, so both sides draw in step: the destination's next id would be this one.
	// A deployment that wrote the log itself has drawn past it.
	_, err = s.destination.DB().ExecContext(s.ctx, "SELECT setval('agent_logs_id_seq', GREATEST("+
		"(SELECT max(id) FROM agent_logs), (SELECT last_value FROM agent_logs_id_seq)))")
	s.Require().NoError(err)
}

// exportedLog is the one agent log of acme's the source exports, as a row the test may edit.
func (s *DataMoveSuite) exportedLog() map[string]any {
	logs := s.export("acme").of("agent_logs")
	s.Require().Len(logs, 1)
	return logs[0]
}

// logsHere are a customer's agent logs on the destination: their message by id.
func (s *DataMoveSuite) logsHere(customerID string) map[int64]string {
	rows, err := s.destination.DB().QueryContext(s.ctx, "SELECT id, message FROM agent_logs WHERE customer_id = ?", customerID)
	s.Require().NoError(err)
	defer rows.Close()
	logs := map[int64]string{}
	for rows.Next() {
		var id int64
		var message string
		s.Require().NoError(rows.Scan(&id, &message))
		logs[id] = message
	}
	s.Require().NoError(rows.Err())
	return logs
}

func (s *DataMoveSuite) TestACopiedLogWhoseIDAnotherAppHoldsHereLandsUnderANewID() {
	id := s.seedLog("acme", "copied")
	// The same words, which makes it no copy of acme's: it is another app's.
	theirs := s.exportedLog()
	theirs["customer_id"] = "resident"
	s.placeHere(theirs)

	s.move("acme", "acme")

	s.Equal(map[int64]string{id: "copied"}, s.logsHere("resident"), "the other app's log is left as it was")
	copied := s.logsHere("acme")
	s.Require().Len(copied, 1, "the log is copied, not skipped")
	for copiedID, message := range copied {
		s.NotEqual(id, copiedID)
		s.Equal("copied", message)
	}
}

func (s *DataMoveSuite) TestACopiedLogWhoseIDTheAppHoldsHereForAnotherLogLeavesThatLog() {
	id := s.seedLog("acme", "copied")
	own := s.exportedLog()
	own["message"] = "written here"
	s.placeHere(own)

	s.move("acme", "acme")

	logs := s.logsHere("acme")
	s.Len(logs, 2, "both the log written here and the copied one are kept")
	s.Equal("written here", logs[id], "the log written here is not overwritten")
	s.Contains(slices.Collect(maps.Values(logs)), "copied")
}

func (s *DataMoveSuite) TestALogHereThatDiffersOnlyInWhenItWasIngestedIsAnotherLog() {
	id := s.seedLog("acme", "copied")
	own := s.exportedLog()
	own["ingested_at"] = "2026-10-01T00:00:00Z"
	s.placeHere(own)

	s.move("acme", "acme")

	s.Len(s.logsHere("acme"), 2, "two logs written at different times are two logs")
	var ingested time.Time
	s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx, "SELECT ingested_at FROM agent_logs WHERE id = ?", id).Scan(&ingested))
	s.True(ingested.Equal(time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)), "the log written here is not overwritten")
}

func (s *DataMoveSuite) TestCopyingLogsTwiceCopiesThemOnce() {
	s.seedLog("acme", "first")
	s.seedLog("acme", "second")

	s.move("acme", "acme")
	s.move("acme", "acme")

	s.ElementsMatch([]string{"first", "second"}, slices.Collect(maps.Values(s.logsHere("acme"))))
}

func (s *DataMoveSuite) TestALogCopiedUnderItsSourceIDBeforeIsNotCopiedAgain() {
	id := s.seedLog("acme", "copied")
	s.placeHere(s.exportedLog())

	s.move("acme", "acme")

	s.Equal(map[int64]string{id: "copied"}, s.logsHere("acme"))
}

func (s *DataMoveSuite) TestALogCopiedFromAnotherSourceLogIsNotTakenForThisOne() {
	id := s.seedLog("acme", "copied")
	// The same words under this id, copied here from another log of the source's: the source
	// wrote the same log twice.
	twin := s.exportedLog()
	twin["source_id"] = id + 1
	s.placeHere(twin)

	s.move("acme", "acme")

	s.Len(s.logsHere("acme"), 2)
}

func (s *DataMoveSuite) TestAReplayedLogIsWrittenOnce() {
	s.capture("acme")
	cursor := s.move("acme", "acme")
	s.seedLog("acme", "after the copy")
	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().Len(changes, 1)

	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))
	s.move("acme", "acme")

	s.ElementsMatch([]string{"after the copy"}, slices.Collect(maps.Values(s.logsHere("acme"))))
}

func (s *DataMoveSuite) TestAReplayedDeleteOfALogRemovesItsCopyAndNotTheLogHoldingItsID() {
	s.capture("acme")
	id := s.seedLog("acme", "copied")
	own := s.exportedLog()
	own["message"] = "written here"
	s.placeHere(own)
	cursor := s.move("acme", "acme")

	_, err := s.source.DB().ExecContext(s.ctx, "DELETE FROM agent_logs WHERE id = ?", id)
	s.Require().NoError(err)
	changes, _, err := s.source.Changes(s.ctx, "acme", cursor, 100)
	s.Require().NoError(err)
	s.Require().NoError(s.destination.ApplyChanges(s.ctx, "acme", changes))

	s.Equal(map[int64]string{id: "written here"}, s.logsHere("acme"))
}

func (s *DataMoveSuite) TestACopiedLogIsNumberedUnderTheLockItsAppsLogsAreNumberedUnder() {
	s.seedLog("acme", "copied")
	encoded, err := json.Marshal(s.exportedLog())
	s.Require().NoError(err)
	// A log of acme's being written here, holding the lock RecordAgentLog takes.
	writing, err := s.destination.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = writing.Rollback() }()
	_, err = writing.ExecContext(s.ctx, "SELECT pg_advisory_xact_lock(hashtextextended('acme', 731))")
	s.Require().NoError(err)

	imported := make(chan error, 1)
	go func() { imported <- s.destination.ImportRow(s.ctx, "acme", "agent_logs", encoded) }()

	s.Eventually(func() bool {
		var waiting bool
		s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx, "SELECT EXISTS (SELECT 1 FROM pg_locks"+
			" WHERE locktype = 'advisory' AND NOT granted AND database = (SELECT oid FROM pg_database WHERE datname = current_database()))").Scan(&waiting))
		return waiting
	}, 10*time.Second, 10*time.Millisecond, "the copy waits for the log being written")
	s.Empty(s.logsHere("acme"))
	s.Require().NoError(writing.Commit())
	select {
	case err := <-imported:
		s.Require().NoError(err)
	case <-time.After(10 * time.Second):
		s.FailNow("the copy did not land once the log being written was")
	}
	s.Len(s.logsHere("acme"), 1)
}

func (s *DataMoveSuite) TestTheSameLogUnderAnotherIDHereIsNotTakenForACopy() {
	id := s.seedLog("acme", "copied")
	// Written here the same, but never copied from this log: a copy kept the source's id.
	same := s.exportedLog()
	same["id"] = id + 1000
	s.placeHere(same)

	s.move("acme", "acme")

	s.Len(s.logsHere("acme"), 2)
}

func (s *DataMoveSuite) TestMigrateBuildsTheIndexCopiedLogsAreKnownByAgain() {
	// As somebody dropping it by hand, or a CONCURRENTLY build that failed, leaves it.
	_, err := s.destination.DB().ExecContext(s.ctx, "DROP INDEX CONCURRENTLY agent_logs_source_idx")
	s.Require().NoError(err)
	s.seedLog("acme", "copied")

	s.Require().NoError(s.destination.Migrate(s.ctx))
	s.move("acme", "acme")
	s.move("acme", "acme")

	s.Len(s.logsHere("acme"), 1)
}

func (s *DataMoveSuite) TestACopiedRequestOfAnAgentIsLoggedHereOnlyByItsCopiedLog() {
	// With one database, the trigger on requests logs a request of an agent where it is written.
	s.Require().NoError(s.source.RecordRequest(s.ctx, &Request{Modality: "stt", CustomerID: "acme", AgentID: "agent",
		Provider: "deepgram", Model: "flux", StartedAt: time.Now().UTC(), Success: true}))
	logged := s.export("acme").of("agent_logs")
	s.Require().Len(logged, 1, "the source logged the request")

	s.move("acme", "acme")

	s.Equal([]string{logged[0]["message"].(string)}, slices.Collect(maps.Values(s.logsHere("acme"))),
		"the request is logged once here: by the log copied with it")
}

func (s *DataMoveSuite) TestALogHereThatDiffersInAnyColumnIsAnotherLog() {
	// Every column a copy writes but the source's id, each set apart from the copied log's.
	differing := map[string]any{
		"customer_id": "resident",
		"config_id":   "another-config",
		"agent_id":    "another-agent",
		"session_id":  "another-session",
		"user_id":     "another-user",
		"source":      "system",
		"severity":    "error",
		"event_type":  "provider_request",
		"message":     "written here",
		"occurred_at": "2026-10-01T00:00:00Z",
		"ingested_at": "2026-10-01T00:00:00Z",
		"details":     map[string]any{"written": "here"},
	}
	for column, value := range differing {
		s.Run(column, func() {
			s.SetupTest()
			s.seedLog("acme", "copied")
			here := s.exportedLog()
			s.Require().NotEqual(value, here[column], "the log here differs from the copied one")
			here[column] = value
			s.placeHere(here)

			s.move("acme", "acme")

			var logs int
			s.Require().NoError(s.destination.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM agent_logs").Scan(&logs))
			s.Equal(2, logs, "a log here differing in %s is not the copied one", column)
		})
	}
}

func (s *DataMoveSuite) TestALogCopiedBackWhereItCameFromIsNotCopiedAgain() {
	// Each deployment draws its own ids: the copy holds another id than the log it came from.
	_, err := s.destination.DB().ExecContext(s.ctx, "SELECT setval('agent_logs_id_seq', (SELECT last_value FROM agent_logs_id_seq) + 1000)")
	s.Require().NoError(err)
	id := s.seedLog("acme", "original")
	s.move("acme", "acme")

	for _, row := range s.exportFrom(s.destination, "acme").of("agent_logs") {
		s.Require().NotEqual(float64(id), row["id"], "the copy was given an id of the destination's")
		encoded, err := json.Marshal(row)
		s.Require().NoError(err)
		s.Require().NoError(s.source.ImportRow(s.ctx, "acme", "agent_logs", encoded))
	}

	var logs []int64
	s.Require().NoError(s.source.DB().NewSelect().Table("agent_logs").Column("id").Where("customer_id = 'acme'").Scan(s.ctx, &logs))
	s.Equal([]int64{id}, logs, "the log that went away and came back is the one log it was")
}
