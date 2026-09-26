package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/uptrace/bun"
)

// ErrChangesExpired says the cursor a caller is resuming from is older than anything
// still recorded, so there is no way to tell what happened in between. The answer is to
// export again rather than to carry on from a number nothing can honour.
var ErrChangesExpired = errors.New("store: the changes since that cursor are no longer kept")

// dataTable is one table a customer's rows live in.
//
// A table either holds the customer id itself or belongs to one through a parent, and
// the two need different queries in every direction: what to select, what to force on the
// way in, and how the trigger works out whose row it is. The order matters, because it is
// the order rows are exported and inserted in, and a child cannot be written before the
// row it points at.
type dataTable struct {
	name string
	// customer is the column holding the customer id, empty when parent owns the row.
	customer string
	// parent and parentColumn name the row this one belongs to.
	parent       string
	parentColumn string
	// filter is an extra predicate on an export, for a table that holds more than one
	// kind of row.
	filter string
}

// dataTables is everything that belongs to a customer, parents first.
//
// Deliberately absent: organizations, apps and api_keys, which are the credentials rather
// than the data, and which a deployment mints for itself; and goose_db_version, which is
// this schema's own bookkeeping. Organization-scope policies are absent too, since one
// organization's decisions cover apps the caller may not have.
var dataTables = []dataTable{
	{name: "agent_configs", customer: "customer_id"},
	{name: "skills", customer: "customer_id"},
	{name: "agent_plugin_connections", customer: "customer_id"},
	{name: "router_configs", customer: "customer_id"},
	{name: "voices", customer: "customer_id"},
	{name: "voice_samples", parent: "voices", parentColumn: "voice_id"},
	{name: "voice_bindings", parent: "voices", parentColumn: "voice_id"},
	{name: "phone_numbers", customer: "customer_id"},
	{name: "knowledge_urls", customer: "customer_id"},
	{name: "knowledge_documents", customer: "customer_id"},
	{name: "campaigns", customer: "customer_id"},
	{name: "campaign_contacts", parent: "campaigns", parentColumn: "campaign_id"},
	{name: "simulations", customer: "customer_id"},
	{name: "simulation_runs", customer: "customer_id"},
	{name: "simulation_cases", parent: "simulation_runs", parentColumn: "run_id"},
	{name: "guest_users", customer: "customer_id"},
	{name: "calls", customer: "customer_id"},
	{name: "call_events", customer: "customer_id"},
	{name: "call_bridges", customer: "customer_id"},
	{name: "recordings", customer: "customer_id"},
	{name: "agent_sessions", customer: "customer_id"},
	{name: "agent_responses", customer: "customer_id"},
	{name: "agent_response_items", parent: "agent_responses", parentColumn: "response_id"},
	{name: "agent_logs", customer: "customer_id"},
	{name: "requests", customer: "customer_id"},
	{name: "turns", customer: "customer_id"},
	{name: "stats_hourly", customer: "customer_id"},
	{name: "stats_daily", customer: "customer_id"},
	{name: "stats_tags_hourly", customer: "customer_id"},
	{name: "stats_tags_daily", customer: "customer_id"},
	{name: "turn_stats_hourly", customer: "customer_id"},
	{name: "turn_stats_daily", customer: "customer_id"},
	// Only what this app decided. The organization's policy covers apps whose rows are
	// not this caller's to read.
	{name: "policies", customer: "scope_id", filter: "scope = 'app'"},
}

// secretColumns never leave the deployment that holds them.
//
// A sealed key secret is worth nothing anywhere else, since the key that unseals it is
// this deployment's. An OAuth token is worse than worthless: it belongs to whoever
// granted it, to this deployment, and handing it out because somebody asked for a copy of
// their data would be a way to walk off with a customer's users' accounts. Both are left
// out on the way out and left alone on the way in, so a connection arrives needing to be
// authorized again rather than arriving broken.
var secretColumns = []string{"secret_sealed", "access_token", "refresh_token", "oauth_state", "code_verifier"}

// DataChange is one thing that happened to one row.
type DataChange struct {
	Seq   int64           `json:"seq"`
	Table string          `json:"table"`
	Op    string          `json:"op"`
	Key   json.RawMessage `json:"key"`
	// Payload is the row as it now reads, and is absent for a delete.
	Payload json.RawMessage `json:"payload,omitempty"`
	At      time.Time       `json:"at"`
}

// Operations a change records.
const (
	ChangeInsert = "insert"
	ChangeUpdate = "update"
	ChangeDelete = "delete"
)

// shape is what a table's columns are, read from the catalog rather than kept in a list
// here that a migration could leave behind.
type shape struct {
	columns []string
	keys    []string
}

// tableShapes caches those, since they only change when a migration does.
type tableShapes struct {
	mu     sync.Mutex
	byName map[string]shape
}

// DataTables names the tables an export carries, parents first. It is public so a test
// and the docs have one answer rather than two.
func DataTables() []string {
	names := make([]string, 0, len(dataTables))
	for _, table := range dataTables {
		names = append(names, table.name)
	}
	return names
}

// StartDataCapture records that a customer is moving, so their changes are written down
// for the deployment they are moving to.
//
// Nothing is recorded for a customer without one of these rows: on a busy deployment
// every model request is a row, and recording all of them for everybody in case somebody
// one day leaves is a cost nobody agreed to. It expires so that a move abandoned halfway
// stops recording on its own, and reading changes pushes it out again.
func (s *Store) StartDataCapture(ctx context.Context, customerID string, retention time.Duration) error {
	if customerID == "" {
		return errors.New("store: a customer id is required")
	}
	_, err := s.db.ExecContext(ctx, `
INSERT INTO data_change_capture (customer_id, expires_at)
VALUES (?, now() + make_interval(secs => ?))
ON CONFLICT (customer_id) DO UPDATE SET expires_at = EXCLUDED.expires_at`,
		customerID, retention.Seconds())
	if err != nil {
		return fmt.Errorf("store: start data capture: %w", err)
	}
	return nil
}

// ExportCustomer writes every row belonging to a customer and returns the cursor their
// changes carry on from.
//
// The rows are read in one repeatable-read transaction, so what comes out is the customer
// as they were at one moment rather than a stitch of several. The cursor is taken in the
// same transaction and is deliberately conservative: a change at or below it is already
// in these rows, and one above it may or may not be, which is safe because applying a
// change twice lands in the same place as applying it once.
//
// What is not here: the audio behind a voice sample and a call recording, which live in
// an object bucket rather than in Postgres. The rows naming them come across, and copying
// the bucket is a copy between two object stores that neither deployment should be in the
// middle of.
func (s *Store) ExportCustomer(ctx context.Context, customerID string, write func(table string, row json.RawMessage) error) (int64, error) {
	if customerID == "" {
		return 0, errors.New("store: a customer id is required")
	}

	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return 0, fmt.Errorf("store: export: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only, and the error on the way out says nothing

	var cursor int64
	if err := tx.QueryRowContext(ctx, changeWatermark).Scan(&cursor); err != nil {
		return 0, fmt.Errorf("store: export cursor: %w", err)
	}

	for _, table := range dataTables {
		if err := exportTable(ctx, tx, table, customerID, write); err != nil {
			return 0, err
		}
	}
	return cursor, nil
}

// changeWatermark is the highest sequence number every change at or below has committed.
//
// Taking the highest number that can be seen would be wrong: the numbers are handed out
// when a row is inserted and become visible when its transaction commits, so a reader
// could step over a lower number belonging to a transaction still in flight and never
// come back for it. This stops below the oldest transaction still running.
const changeWatermark = `
SELECT COALESCE(
    (SELECT MIN(seq) - 1 FROM data_changes WHERE tx >= pg_snapshot_xmin(pg_current_snapshot())),
    (SELECT COALESCE(MAX(seq), 0) FROM data_changes))`

func exportTable(ctx context.Context, tx bun.Tx, table dataTable, customerID string, write func(string, json.RawMessage) error) error {
	query, args := table.selectRows(customerID)
	rows, err := tx.QueryContext(ctx, query, args...)
	if err != nil {
		return fmt.Errorf("store: export %s: %w", table.name, err)
	}
	defer rows.Close()

	for rows.Next() {
		var row json.RawMessage
		if err := rows.Scan(&row); err != nil {
			return fmt.Errorf("store: export %s: %w", table.name, err)
		}
		if err := write(table.name, row); err != nil {
			return err
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("store: export %s: %w", table.name, err)
	}
	return nil
}

// selectRows reads one table's rows for one customer, without the columns that never
// leave.
func (t dataTable) selectRows(customerID string) (string, []any) {
	var subject strings.Builder
	subject.WriteString("to_jsonb(t)")
	for _, column := range secretColumns {
		subject.WriteString(" - '" + column + "'")
	}

	if t.parent != "" {
		return fmt.Sprintf(
			"SELECT %s FROM %s AS t JOIN %s AS p ON p.id = t.%s WHERE p.customer_id = ?",
			subject.String(), t.name, t.parent, t.parentColumn), []any{customerID}
	}
	where := t.customer + " = ?"
	if t.filter != "" {
		where += " AND " + t.filter
	}
	return fmt.Sprintf("SELECT %s FROM %s AS t WHERE %s", subject.String(), t.name, where), []any{customerID}
}

// ImportRow writes one exported row into this deployment, under the customer doing the
// importing.
//
// Whose row it is comes from the caller rather than from the row: an export names a
// customer inside every row it carries, and believing that would let anybody holding
// credentials for one app write into another. For a row that belongs to a customer
// through a parent there is no column to overwrite, so the parent is checked instead and
// a row whose parent is somebody else's is silently not written.
//
// It upserts, so importing the same export twice is the same as importing it once, which
// is what lets a move be resumed after it was interrupted.
func (s *Store) ImportRow(ctx context.Context, customerID, table string, row json.RawMessage) error {
	return s.importRow(ctx, s.db, customerID, table, row)
}

func (s *Store) importRow(ctx context.Context, db bun.IDB, customerID, table string, row json.RawMessage) error {
	spec, found := dataTableByName(table)
	if !found {
		return fmt.Errorf("store: %q is not a table an export carries", table)
	}
	shape, err := s.shapeOf(ctx, table)
	if err != nil {
		return err
	}

	columns := shape.insertable()
	quoted := strings.Join(columns, ", ")

	assignments := make([]string, 0, len(columns))
	for _, column := range columns {
		if !spec.isKey(shape, column) {
			assignments = append(assignments, column+" = EXCLUDED."+column)
		}
	}

	// The row first, then the customer, in the order the placeholders appear: the
	// customer is forced onto the row of a table that names one, and checked against the
	// parent of a table that does not.
	subject, where := "?::jsonb", ""
	if spec.customer != "" {
		subject = fmt.Sprintf("?::jsonb || %s", spec.identity())
	}
	if spec.parent != "" {
		where = " WHERE " + spec.owned("r")
	}

	// A row nobody may update is one whose primary key is all it has, and writing it
	// again would be the same row. Doing nothing is both the right answer and a legal
	// statement, where an empty SET list is neither.
	//
	// The update is guarded by who already holds the row rather than by who is writing
	// it. Primary keys are carried across as they are, so without this an import could
	// name a row it happened to know the id of and overwrite another customer's.
	conflict := "DO NOTHING"
	args := []any{string(row), customerID}
	if len(assignments) > 0 {
		conflict = "DO UPDATE SET " + strings.Join(assignments, ", ") + " WHERE " + spec.owned(spec.name)
		args = append(args, customerID)
	}

	query := fmt.Sprintf(
		"INSERT INTO %s (%s) SELECT %s FROM jsonb_populate_record(NULL::%s, %s) AS r%s ON CONFLICT (%s) %s",
		spec.name, quoted, quoted, spec.name, subject, where, strings.Join(shape.keys, ", "), conflict)

	if _, err := db.ExecContext(ctx, query, args...); err != nil {
		return fmt.Errorf("store: import %s: %w", table, err)
	}
	return nil
}

// identity is what the importing deployment overwrites on every row of this table, which
// is whose it is. A table holding more than one kind of row has the kind forced too, so
// an app-scope policy cannot be imported as its organization's.
func (t dataTable) identity() string {
	if t.name == "policies" {
		return "jsonb_build_object('scope_id', ?::text, 'scope', 'app')"
	}
	return fmt.Sprintf("jsonb_build_object('%s', ?::text)", t.customer)
}

// owned is the condition that a row of this table, as the given relation names it,
// belongs to the customer the query's next placeholder carries.
func (t dataTable) owned(relation string) string {
	if t.parent != "" {
		return fmt.Sprintf("EXISTS (SELECT 1 FROM %s AS p WHERE p.id = %s.%s AND p.customer_id = ?)",
			t.parent, relation, t.parentColumn)
	}
	return fmt.Sprintf("%s.%s = ?", relation, t.customer)
}

func (t dataTable) isKey(s shape, column string) bool {
	for _, key := range s.keys {
		if key == column {
			return true
		}
	}
	return false
}

// Changes returns what has happened to a customer's rows since a cursor, oldest first,
// and the cursor to ask from next time.
//
// A cursor older than what is still kept is ErrChangesExpired rather than a shorter
// answer: the caller would otherwise carry on from a gap it could not see, and a missing
// change is a row that is wrong on the other side forever.
func (s *Store) Changes(ctx context.Context, customerID string, after int64, limit int) ([]DataChange, int64, error) {
	if customerID == "" {
		return nil, 0, errors.New("store: a customer id is required")
	}
	if limit <= 0 {
		limit = 500
	}

	var earliest sql.NullInt64
	if err := s.db.QueryRowContext(ctx,
		"SELECT MIN(seq) FROM data_changes WHERE customer_id = ?", customerID).Scan(&earliest); err != nil {
		return nil, 0, fmt.Errorf("store: changes: %w", err)
	}
	if after > 0 && earliest.Valid && earliest.Int64 > after+1 {
		return nil, 0, ErrChangesExpired
	}

	rows, err := s.db.QueryContext(ctx, fmt.Sprintf(`
SELECT seq, table_name, op, key, payload, at
FROM data_changes
WHERE customer_id = ? AND seq > ? AND seq <= (%s)
ORDER BY seq
LIMIT ?`, changeWatermark), customerID, after, limit)
	if err != nil {
		return nil, 0, fmt.Errorf("store: changes: %w", err)
	}
	defer rows.Close()

	changes := []DataChange{}
	cursor := after
	for rows.Next() {
		var change DataChange
		var payload []byte
		if err := rows.Scan(&change.Seq, &change.Table, &change.Op, &change.Key, &payload, &change.At); err != nil {
			return nil, 0, fmt.Errorf("store: changes: %w", err)
		}
		change.Payload = payload
		changes = append(changes, change)
		cursor = change.Seq
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("store: changes: %w", err)
	}
	return changes, cursor, nil
}

// ApplyChanges replays changes onto this deployment, in one transaction so a batch either
// arrives or does not.
func (s *Store) ApplyChanges(ctx context.Context, customerID string, changes []DataChange) error {
	if customerID == "" {
		return errors.New("store: a customer id is required")
	}
	return s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		for _, change := range changes {
			switch change.Op {
			case ChangeDelete:
				if err := s.deleteRow(ctx, tx, customerID, change); err != nil {
					return err
				}
			default:
				if err := s.importRow(ctx, tx, customerID, change.Table, change.Payload); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

// deleteRow removes what a delete change names, scoped to the customer replaying it so a
// change carrying somebody else's key cannot reach their row.
func (s *Store) deleteRow(ctx context.Context, db bun.IDB, customerID string, change DataChange) error {
	spec, found := dataTableByName(change.Table)
	if !found {
		return fmt.Errorf("store: %q is not a table an export carries", change.Table)
	}
	shape, err := s.shapeOf(ctx, change.Table)
	if err != nil {
		return err
	}

	conditions := make([]string, 0, len(shape.keys)+1)
	args := make([]any, 0, len(shape.keys)+1)
	for _, column := range shape.keys {
		conditions = append(conditions, fmt.Sprintf("t.%s::text = (?::jsonb ->> '%s')", column, column))
		args = append(args, string(change.Key))
	}
	if spec.parent != "" {
		conditions = append(conditions, fmt.Sprintf(
			"EXISTS (SELECT 1 FROM %s AS p WHERE p.id = t.%s AND p.customer_id = ?)",
			spec.parent, spec.parentColumn))
	} else {
		conditions = append(conditions, fmt.Sprintf("t.%s = ?", spec.customer))
		if spec.filter != "" {
			conditions = append(conditions, spec.filter)
		}
	}
	args = append(args, customerID)

	query := fmt.Sprintf("DELETE FROM %s AS t WHERE %s", spec.name, strings.Join(conditions, " AND "))
	if _, err := db.ExecContext(ctx, query, args...); err != nil {
		return fmt.Errorf("store: delete %s: %w", change.Table, err)
	}
	return nil
}

// PruneDataChanges drops the changes and the capture rows nobody can still use. It
// returns how many changes it removed.
func (s *Store) PruneDataChanges(ctx context.Context, retention time.Duration) (int64, error) {
	if retention <= 0 {
		return 0, nil
	}
	result, err := s.db.ExecContext(ctx,
		"DELETE FROM data_changes WHERE at < now() - make_interval(secs => ?)", retention.Seconds())
	if err != nil {
		return 0, fmt.Errorf("store: prune data changes: %w", err)
	}
	if _, err := s.db.ExecContext(ctx, "DELETE FROM data_change_capture WHERE expires_at < now()"); err != nil {
		return 0, fmt.Errorf("store: prune data capture: %w", err)
	}
	return result.RowsAffected()
}

// WatchDataChanges attaches the recording trigger to every table an export carries.
//
// It runs on every start rather than once in a migration, so a table added later is
// covered by the deployment reaching it rather than by somebody remembering to write the
// trigger by hand. Replacing a trigger that is already right costs a catalogue write at
// startup and nothing afterwards.
func (s *Store) WatchDataChanges(ctx context.Context) error {
	for _, table := range dataTables {
		shape, err := s.shapeOf(ctx, table.name)
		if err != nil {
			return err
		}
		arguments := []string{"'" + table.customer + "'", "''", "''"}
		if table.parent != "" {
			arguments = []string{"'parent'", "'" + table.parentColumn + "'", "'" + table.parent + "'"}
		}
		for _, key := range shape.keys {
			arguments = append(arguments, "'"+key+"'")
		}

		query := fmt.Sprintf(`
CREATE OR REPLACE TRIGGER data_changes_recorded
AFTER INSERT OR UPDATE OR DELETE ON %s
FOR EACH ROW EXECUTE FUNCTION record_data_change(%s)`, table.name, strings.Join(arguments, ", "))
		if _, err := s.db.ExecContext(ctx, query); err != nil {
			return fmt.Errorf("store: watch %s: %w", table.name, err)
		}
	}
	return nil
}

// shapeOf reads a table's columns and primary key out of the catalogue, once.
func (s *Store) shapeOf(ctx context.Context, table string) (shape, error) {
	s.shapes.mu.Lock()
	defer s.shapes.mu.Unlock()

	if known, found := s.shapes.byName[table]; found {
		return known, nil
	}
	if _, known := dataTableByName(table); !known {
		return shape{}, fmt.Errorf("store: %q is not a table an export carries", table)
	}

	var read shape
	rows, err := s.db.QueryContext(ctx, `
SELECT a.attname,
       EXISTS (SELECT 1 FROM pg_index i
               WHERE i.indrelid = a.attrelid AND i.indisprimary AND a.attnum = ANY (i.indkey)) AS is_key
FROM pg_attribute a
WHERE a.attrelid = ?::regclass AND a.attnum > 0 AND NOT a.attisdropped AND a.attgenerated = ''
ORDER BY a.attnum`, table)
	if err != nil {
		return shape{}, fmt.Errorf("store: shape of %s: %w", table, err)
	}
	defer rows.Close()

	for rows.Next() {
		var column string
		var isKey bool
		if err := rows.Scan(&column, &isKey); err != nil {
			return shape{}, fmt.Errorf("store: shape of %s: %w", table, err)
		}
		read.columns = append(read.columns, column)
		if isKey {
			read.keys = append(read.keys, column)
		}
	}
	if err := rows.Err(); err != nil {
		return shape{}, fmt.Errorf("store: shape of %s: %w", table, err)
	}
	if len(read.keys) == 0 {
		return shape{}, fmt.Errorf("store: %s has no primary key to import against", table)
	}

	if s.shapes.byName == nil {
		s.shapes.byName = map[string]shape{}
	}
	s.shapes.byName[table] = read
	return read, nil
}

// insertable is every column an import writes: the whole row but the secrets, which are
// left to the local default so a connection arrives needing to be authorized again
// rather than arriving with somebody else's token in it.
func (s shape) insertable() []string {
	columns := make([]string, 0, len(s.columns))
	for _, column := range s.columns {
		secret := false
		for _, name := range secretColumns {
			if column == name {
				secret = true
				break
			}
		}
		if !secret {
			columns = append(columns, column)
		}
	}
	return columns
}

func dataTableByName(name string) (dataTable, bool) {
	for _, table := range dataTables {
		if table.name == name {
			return table, true
		}
	}
	return dataTable{}, false
}
