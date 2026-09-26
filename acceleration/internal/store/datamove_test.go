//go:build integration

package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun/driver/pgdriver"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
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
	s.source = s.open(dsn)
	s.destination = s.open(elsewhere(dsn))
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

// elsewhere is the other deployment's database, which is this one's name with a word in
// front of the suffix that keeps it a test database.
func elsewhere(dsn string) string {
	return strings.Replace(dsn, "_test", "_moved_test", 1)
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
