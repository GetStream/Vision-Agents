package store

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
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
	// shard is a table kept in the shards rather than the shared database.
	shard bool
	// across is the column a row is known by in every deployment, for a table whose primary
	// key each deployment draws from a sequence of its own, so the same id is two rows in
	// two deployments. Such a row is imported without its id, and is written once: importing
	// it again, or a row the destination already holds under it, changes nothing.
	across string
}

// rollup is a table of counts per key that every shard keeps its own rows of, so an export of
// several shards folds the rows of one key into one.
func (t dataTable) rollup() bool {
	return strings.HasPrefix(t.name, "stats_") || strings.HasPrefix(t.name, "turn_")
}

// dataTables is everything that belongs to a customer, parents first.
//
// Deliberately absent: organizations, apps and api_keys, which are the credentials rather
// than the data, and which a deployment mints for itself; and goose_db_version, which is
// this schema's own bookkeeping. Organization-scope policies are absent too, since one
// organization's decisions cover apps the caller may not have. So are
// connector_authorization_attempts: one lives minutes, is sealed under this deployment's
// key and finishes at this deployment's callback, so a copy could only expire. And
// connector_oauth_clients: a client's secret is sealed under this deployment's key, and a
// client without it cannot authenticate, so the app puts it again where it moved to. So is
// custom_models, whose endpoint keys are sealed the same way. And
// connector_config_tokens, sealed under this deployment's key for the provider app whose
// events URL is this deployment's. And connector_event_destinations, whose signing secrets are
// sealed under this deployment's key, with connector_event_deliveries, forwards in flight. And
// connection_event_subscriptions with connection_event_deliveries: their secrets are sealed
// under this deployment's key and their callbacks are this deployment's URL. And
// channel_threads, contact_map, episodes and episode_activity: the thread channels,
// omni-channels and cards they point at live in the Stream app, which a move does not carry
// either; and channel_thread_waiting, replies in flight for minutes. And
// connector_invocations and connector_audit, with connector_audit_credentials and
// connector_invocation_arguments: this deployment's record of what it did with credentials
// sealed under its own key. And connector_tool_pins: the deployment a connection moves to pins
// its tools again on first use. And connector_connection_clients: the next consent of the
// connection where it moved to records its client again. And connector_connection_validations:
// the next validate where it moved to records its result again.
var dataTables = []dataTable{
	{name: "agent_configs", customer: "customer_id"},
	{name: "skills", customer: "customer_id"},
	{name: "agent_plugin_connections", customer: "customer_id"},
	{name: "agent_plugin_clients", customer: "customer_id"},
	// Only the customer's own definitions: the built-ins are under no customer, and every
	// deployment seeds its own. They come before the connections that pin them.
	{name: "connector_definitions", customer: "customer_id"},
	{name: "connector_connections", customer: "customer_id"},
	{name: "router_configs", customer: "customer_id"},
	{name: "voices", customer: "customer_id"},
	{name: "voice_samples", parent: "voices", parentColumn: "voice_id"},
	{name: "voice_bindings", parent: "voices", parentColumn: "voice_id"},
	{name: "sip_trunks", customer: "customer_id"},
	{name: "phone_numbers", customer: "customer_id"},
	{name: "knowledge_urls", customer: "customer_id"},
	{name: "knowledge_documents", customer: "customer_id"},
	{name: "campaigns", customer: "customer_id"},
	{name: "campaign_contacts", parent: "campaigns", parentColumn: "campaign_id"},
	{name: "simulations", customer: "customer_id"},
	{name: "simulation_runs", customer: "customer_id"},
	{name: "simulation_cases", parent: "simulation_runs", parentColumn: "run_id"},
	{name: "users", customer: "customer_id"},
	{name: "scheduled_tasks", customer: "customer_id"},
	{name: "scheduled_task_runs", customer: "customer_id"},
	{name: "calls", customer: "customer_id", shard: true},
	{name: "call_bridges", customer: "customer_id", shard: true},
	{name: "recordings", customer: "customer_id", shard: true},
	{name: "agent_sessions", customer: "customer_id", shard: true},
	{name: "agent_session_tools", parent: "agent_sessions", parentColumn: "session_id", shard: true},
	{name: "agent_responses", customer: "customer_id", shard: true},
	{name: "agent_response_items", parent: "agent_responses", parentColumn: "response_id", shard: true},
	{name: "agent_logs", customer: "customer_id"},
	// No foreign key points at a request's id. An agent log row names it as text, and
	// Store.AgentLog only takes a request's error by that number from a row that carries none of
	// its own and whose time and agent are the request's, so a moved row cannot show the error
	// of whichever request holds its number here. Its usage id is the unit of work wherever it is.
	{name: "requests", customer: "customer_id", shard: true, across: "usage_id"},
	{name: "stats_hourly", customer: "customer_id", shard: true},
	{name: "stats_daily", customer: "customer_id", shard: true},
	{name: "stats_tags_hourly", customer: "customer_id", shard: true},
	{name: "stats_tags_daily", customer: "customer_id", shard: true},
	{name: "turn_stats_hourly", customer: "customer_id", shard: true},
	{name: "turn_latency_hourly", customer: "customer_id", shard: true},
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
var secretColumns = []string{"secret_sealed", "access_token", "refresh_token", "oauth_state", "code_verifier", "credentials_sealed", "password_sealed", "bridge_password_sealed"}

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

	if !s.Sharded() && len(s.peers) == 0 {
		for _, table := range dataTables {
			if err := s.exportTable(ctx, tx, table, customerID, write); err != nil {
				return 0, err
			}
		}
		return cursor, nil
	}

	// Every shard is read at one moment of its own. No table of the shared database points at
	// a shard's, so writing the shared tables first still writes every parent first.
	var shards []bun.Tx
	for _, db := range s.shards() {
		shardTx, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
		if err != nil {
			return 0, fmt.Errorf("store: export: shard: %w", err)
		}
		defer shardTx.Rollback() //nolint:errcheck // read-only, and the error on the way out says nothing
		shards = append(shards, shardTx)
	}
	for _, table := range dataTables {
		if !table.shard {
			if err := s.exportTable(ctx, tx, table, customerID, write); err != nil {
				return 0, err
			}
		}
	}
	for _, table := range dataTables {
		if !table.shard {
			continue
		}
		if table.rollup() {
			if err := s.exportRollup(ctx, shards, table, customerID, write); err != nil {
				return 0, err
			}
			continue
		}
		for _, shardTx := range shards {
			if err := s.exportTable(ctx, shardTx, table, customerID, write); err != nil {
				return 0, err
			}
		}
	}
	return cursor, nil
}

// exportRollup writes a rollup table's rows of every shard as one row per key: counts and
// totals summed, a percentile weighted by the requests or turns behind it, as the dashboard
// merges them, and uptime left for the importing side to work out again.
func (s *Store) exportRollup(ctx context.Context, shards []bun.Tx, table dataTable, customerID string, write func(string, json.RawMessage) error) error {
	shape, err := s.shapeOf(ctx, table.name)
	if err != nil {
		return err
	}
	// row is the first shard's row of a key, the others folded into it; weights are the
	// requests or turns behind each percentile so far.
	type keyed struct {
		row     map[string]any
		weights map[string]float64
	}
	var order []string
	byKey := map[string]*keyed{}
	for _, shardTx := range shards {
		err := s.exportTable(ctx, shardTx, table, customerID, func(_ string, raw json.RawMessage) error {
			decoder := json.NewDecoder(bytes.NewReader(raw))
			decoder.UseNumber()
			var row map[string]any
			if err := decoder.Decode(&row); err != nil {
				return fmt.Errorf("store: export %s: %w", table.name, err)
			}
			delete(row, "uptime")
			var key strings.Builder
			for _, column := range shape.keys {
				fmt.Fprintf(&key, "%v\x00", row[column])
			}
			weight := rollupWeight(row)
			into := byKey[key.String()]
			if into == nil {
				into = &keyed{row: row, weights: map[string]float64{}}
				for column, value := range row {
					if _, ok := value.(json.Number); ok && isPercentile(column) {
						into.weights[column] = weight
					}
				}
				byKey[key.String()] = into
				order = append(order, key.String())
				return nil
			}
			for column, value := range row {
				number, ok := value.(json.Number)
				if !ok || slices.Contains(shape.keys, column) {
					continue
				}
				if !isPercentile(column) {
					was, _ := into.row[column].(json.Number)
					into.row[column] = addNumbers(was, number)
					continue
				}
				before := into.weights[column]
				if before+weight > 0 {
					was, _ := into.row[column].(json.Number)
					x, _ := was.Float64()
					y, _ := number.Float64()
					into.row[column] = json.Number(strconv.FormatFloat((x*before+y*weight)/(before+weight), 'f', -1, 64))
				}
				into.weights[column] = before + weight
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	for _, key := range order {
		encoded, err := json.Marshal(byKey[key].row)
		if err != nil {
			return fmt.Errorf("store: export %s: %w", table.name, err)
		}
		if err := write(table.name, encoded); err != nil {
			return err
		}
	}
	return nil
}

// rollupWeight is how many requests or turns a rollup row counts.
func rollupWeight(row map[string]any) float64 {
	for _, column := range []string{"request_count", "turn_count", "count"} {
		if number, ok := row[column].(json.Number); ok {
			weight, _ := number.Float64()
			return weight
		}
	}
	return 0
}

func isPercentile(column string) bool {
	return strings.Contains(column, "_p50_") || strings.Contains(column, "_p95_") || strings.Contains(column, "_p99_")
}

// addNumbers sums two numbers, as integers while both are.
func addNumbers(a, b json.Number) json.Number {
	if a == "" {
		return b
	}
	x, errX := a.Int64()
	y, errY := b.Int64()
	if errX == nil && errY == nil {
		return json.Number(strconv.FormatInt(x+y, 10))
	}
	fx, _ := a.Float64()
	fy, _ := b.Float64()
	return json.Number(strconv.FormatFloat(fx+fy, 'f', -1, 64))
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

func (s *Store) exportTable(ctx context.Context, tx bun.Tx, table dataTable, customerID string, write func(string, json.RawMessage) error) error {
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
		if row, err = s.exportPin(table.name, row); err != nil {
			return err
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
	spec, _ := dataTableByName(table)
	return s.importRow(ctx, s.sideOf(spec), customerID, table, row)
}

// sideOf is the database a table's rows are written to here: this shard's or the shared one.
func (s *Store) sideOf(table dataTable) *bun.DB {
	if table.shard {
		return s.shard
	}
	return s.db
}

func (s *Store) importRow(ctx context.Context, db bun.IDB, customerID, table string, row json.RawMessage) error {
	spec, found := dataTableByName(table)
	if !found {
		return fmt.Errorf("store: %q is not a table an export carries", table)
	}
	row, err := s.importPin(ctx, customerID, table, row)
	if err != nil {
		return err
	}
	shape, err := s.shapeOf(ctx, table)
	if err != nil {
		return err
	}

	columns := shape.insertable()
	if spec.across != "" {
		// The id is the destination's to draw: the source's may be one of its own rows'.
		columns = slices.DeleteFunc(columns, func(column string) bool { return spec.isKey(shape, column) })
	}
	quoted := strings.Join(columns, ", ")
	selected := quoted
	if i := slices.Index(columns, "usage_id"); spec.name == "requests" && i >= 0 {
		// A row from before usage ids has none, and the table refuses a new row without one.
		// Its usage id is made from the source's id for it, so importing it again names the
		// same unit of work.
		picked := slices.Clone(columns)
		picked[i] = "COALESCE(usage_id, " + importedUsageID("customer_id", "id") + ")"
		selected = strings.Join(picked, ", ")
	}

	assignments := make([]string, 0, len(columns))
	for _, column := range columns {
		if !spec.isKey(shape, column) {
			assignments = append(assignments, column+" = EXCLUDED."+column)
		}
	}
	if spec.across != "" {
		assignments = nil
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
		spec.name, quoted, selected, spec.name, subject, where, strings.Join(spec.keys(shape), ", "), conflict)

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
	// A connector connection arrives without its credentials, so it arrives saying it needs
	// one, rather than connected with nothing to connect with. Key version 0 and no
	// expiry are what a connection without credentials holds (20261002193000_connector_connections.sql),
	// so the source credentials' key version and expiry do not come along either, on an import or
	// on every change applied after it. Nor does its provider unit: a row in an import is
	// whatever the importer wrote, and a unit stored from it would route another customer's
	// events to this one. The reconnect it needs proves the unit again
	// (SetConnectorConnectionProviderUnit).
	if t.name == "connector_connections" {
		return "jsonb_build_object('customer_id', ?::text, 'status', '" + ConnectionNeedsReauthorization + "'," +
			" 'credentials_kek_version', 0, 'expires_at', NULL, 'provider_unit_id', NULL)"
	}
	// A trunk arrives without its password, since the key that sealed it is the other
	// deployment's, so it arrives at key version 0, which is what a trunk without one
	// holds (20261007050000_sip_trunks.sql).
	if t.name == "sip_trunks" {
		return "jsonb_build_object('customer_id', ?::text, 'password_kek_version', 0)"
	}
	// An attached number arrives without its bridge password for the same reason, so its key
	// version stays behind too rather than describing a password that is not there.
	if t.name == "phone_numbers" {
		return "jsonb_build_object('customer_id', ?::text, 'bridge_password_kek_version', NULL)"
	}
	// A request was billed by the deployment that recorded it, so this one never ships it.
	if t.name == "requests" {
		return "jsonb_build_object('customer_id', ?::text, 'to_ship', false)"
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

// keys are the columns a row of this table is known by across deployments: what an import
// resolves a conflict on.
func (t dataTable) keys(s shape) []string {
	if t.across != "" {
		return []string{t.across}
	}
	return s.keys
}

// captured are the columns a change records of a row. A row known by another column than its
// primary key records both: the primary key is what a row without the other is found by.
func (t dataTable) captured(s shape) []string {
	if t.across != "" {
		return append([]string{t.across}, s.keys...)
	}
	return s.keys
}

// importedUsageID is the usage id a request from before usage ids is imported under, as SQL,
// made from the customer importing it and its id in the source, so importing it again and
// replaying its delete name the same row.
func importedUsageID(customer, sourceID string) string {
	return fmt.Sprintf("md5('import:' || %s || ':' || %s)::uuid", customer, sourceID)
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
		if change.Op != ChangeDelete && len(payload) > 0 {
			if change.Payload, err = s.exportPin(change.Table, payload); err != nil {
				return nil, 0, err
			}
		}
		changes = append(changes, change)
		cursor = change.Seq
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("store: changes: %w", err)
	}
	return changes, cursor, nil
}

// ApplyChanges replays changes onto this deployment, in one transaction so a batch either
// arrives or does not. With shards there is one for the shared tables and one for this
// shard's, each replayed in order, as no row of one points at a row of the other.
func (s *Store) ApplyChanges(ctx context.Context, customerID string, changes []DataChange) error {
	if customerID == "" {
		return errors.New("store: a customer id is required")
	}
	if !s.Sharded() {
		return s.applyChanges(ctx, s.db, customerID, changes)
	}
	var shared, sharded []DataChange
	for _, change := range changes {
		if spec, _ := dataTableByName(change.Table); spec.shard {
			sharded = append(sharded, change)
		} else {
			shared = append(shared, change)
		}
	}
	if err := s.applyChanges(ctx, s.db, customerID, shared); err != nil {
		return err
	}
	return s.applyChanges(ctx, s.shard, customerID, sharded)
}

func (s *Store) applyChanges(ctx context.Context, db *bun.DB, customerID string, changes []DataChange) error {
	if len(changes) == 0 {
		return nil
	}
	return db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
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

	var conditions []string
	var args []any
	if spec.across != "" {
		// Never by the id, which names another row here. A row from before usage ids has none,
		// and was imported under one made from its source id, which the change carries too.
		conditions = append(conditions, fmt.Sprintf("t.%s = COALESCE((?::jsonb ->> '%s')::uuid, %s)",
			spec.across, spec.across, importedUsageID("?::text", "(?::jsonb ->> 'id')")))
		args = append(args, string(change.Key), customerID, string(change.Key))
	} else {
		for _, column := range shape.keys {
			conditions = append(conditions, fmt.Sprintf("t.%s::text = (?::jsonb ->> '%s')", column, column))
			args = append(args, string(change.Key))
		}
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

// watchDataChanges attaches the recording trigger to every table an export carries, in one
// database.
//
// Changes are only recorded for a customer StartDataCapture marked, which it does in the shared
// database, so a shard's own tables record none: a deployment of several shards replicates the
// shared tables, and a move off it carries the shards' rows as the export read them.
//
// It runs on every start rather than once in a migration, so a table added later is
// covered by the deployment reaching it rather than by somebody remembering to write the
// trigger by hand. Replacing a trigger that is already right costs a catalogue write at
// startup and nothing afterwards.
func (s *Store) watchDataChanges(ctx context.Context, db *bun.DB) error {
	for _, table := range dataTables {
		shape, err := s.shapeOf(ctx, table.name)
		if err != nil {
			return err
		}
		arguments := []string{"'" + table.customer + "'", "''", "''"}
		if table.parent != "" {
			arguments = []string{"'parent'", "'" + table.parentColumn + "'", "'" + table.parent + "'"}
		}
		for _, key := range table.captured(shape) {
			arguments = append(arguments, "'"+key+"'")
		}

		query := fmt.Sprintf(`
CREATE OR REPLACE TRIGGER data_changes_recorded
AFTER INSERT OR UPDATE OR DELETE ON %s
FOR EACH ROW EXECUTE FUNCTION record_data_change(%s)`, table.name, strings.Join(arguments, ", "))
		if _, err := db.ExecContext(ctx, query); err != nil {
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
