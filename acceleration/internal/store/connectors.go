package store

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"slices"
	"strings"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// BuiltinCustomer is the customer the built-in connector definitions are stored under. It is
// empty because no caller can be it: every auth mode refuses an empty app id
// (internal/auth/auth.go) and api.CustomerFrom reports no customer for one, so no tenant can
// write a built-in or pass for its owner.
const BuiltinCustomer = ""

// CustomPrefix starts the id of every custom definition and of no built-in, so a customer's
// definition can never shadow a built-in and a lookup by id is never ambiguous. The prefix is
// the one the architecture's plan reserves for custom definitions
// (acceleration/docs/connectors/subtasks.md, T6 acceptance, on connectors/planning), which the
// core's custom_crm and custom_mtls_service fixtures already use. The migration checks it too.
const CustomPrefix = "custom_"

// definitionLockSeed is the second argument to hashtextextended for the lock a revision is
// written under. Any value other than the 731 RecordAgentLog uses would do, so a customer's
// log writes never wait on its definitions; 833 is this table's issue, AI-833.
const definitionLockSeed = 833

// ErrNoConnectorDefinition says there is no such definition for this customer, built-in or
// its own. A sentinel, because asking for an id nobody defined is an ordinary 404, and a
// caller has to tell it apart from the database being down.
var ErrNoConnectorDefinition = errors.New("store: no such connector definition")

// ConnectorDefinition is one revision of a connector's manifest. Built-ins are stored under
// BuiltinCustomer; a custom definition under the customer that made it.
type ConnectorDefinition struct {
	bun.BaseModel `bun:"table:connector_definitions,alias:cd"`

	CustomerID  string        `bun:"customer_id,pk"`
	ID          string        `bun:"id,pk"`
	Revision    int           `bun:"revision,pk"`
	Name        string        `bun:"name,notnull"`
	Category    string        `bun:"category,notnull"`
	Description string        `bun:"description,notnull"`
	Manifest    core.Manifest `bun:"manifest,type:jsonb,notnull"`
	CreatedAt   time.Time     `bun:"created_at,notnull"`
}

// SeedConnectorDefinitions stores every built-in manifest in fsys, one <id>.yaml per
// connector, as a built-in definition at the revision the file names. The file's author
// numbers built-in revisions, the way a migration is numbered, because two router builds can
// share a database at once: old and new pods in a rolling deploy, a rollback, or a branch
// build beside an accelerate build on staging. Each start of either build then finds its own
// revision already stored and changes nothing, rather than storing its manifest as the next
// revision and flipping latest back and forth. Every file is parsed before any is written, so
// one invalid manifest stores none of them.
func (s *Store) SeedConnectorDefinitions(ctx context.Context, fsys fs.FS) error {
	manifests, err := builtinManifests(fsys)
	if err != nil {
		return err
	}
	for _, manifest := range manifests {
		if err := s.seedBuiltin(ctx, manifest); err != nil {
			return err
		}
	}
	return nil
}

// CreateConnectorDefinition stores a customer's own definition as the next revision of its
// id, or returns the latest revision when that already says the same. The manifest's own
// revision is replaced: the store numbers revisions.
func (s *Store) CreateConnectorDefinition(ctx context.Context, customerID string, manifest core.Manifest) (ConnectorDefinition, error) {
	if customerID == BuiltinCustomer {
		return ConnectorDefinition{}, stack.Wrap(errors.New("store: customer id is required"))
	}
	if !strings.HasPrefix(manifest.ID, CustomPrefix) {
		return ConnectorDefinition{}, stack.Wrap(fmt.Errorf("store: a custom connector id starts with %s, so it cannot shadow a built-in: %q", CustomPrefix, manifest.ID))
	}
	return s.saveRevision(ctx, customerID, manifest)
}

// ConnectorDefinition returns one revision of a definition the customer can see: a built-in
// or its own.
func (s *Store) ConnectorDefinition(ctx context.Context, customerID, id string, revision int) (ConnectorDefinition, error) {
	if customerID == "" || id == "" {
		return ConnectorDefinition{}, stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}

	return connectorDefinition(ctx, s.db, customerID, id, revision)
}

// connectorDefinition is one revision of a definition the customer can see, read on db.
func connectorDefinition(ctx context.Context, db bun.IDB, customerID, id string, revision int) (ConnectorDefinition, error) {
	var definition ConnectorDefinition
	err := db.NewSelect().Model(&definition).
		Where("customer_id IN (?, ?)", BuiltinCustomer, customerID).
		Where("id = ?", id).
		Where("revision = ?", revision).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorDefinition{}, stack.Wrap(fmt.Errorf("%w: %s revision %d", ErrNoConnectorDefinition, id, revision))
	}
	if err != nil {
		return ConnectorDefinition{}, stack.Wrap(fmt.Errorf("store: connector definition: %w", err))
	}
	return definition, nil
}

// LatestConnectorDefinition returns the newest revision of a definition the customer can see.
func (s *Store) LatestConnectorDefinition(ctx context.Context, customerID, id string) (ConnectorDefinition, error) {
	if customerID == "" || id == "" {
		return ConnectorDefinition{}, stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	definition, err := latestDefinition(ctx, s.db, []string{BuiltinCustomer, customerID}, id)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorDefinition{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorDefinition, id))
	}
	if err != nil {
		return ConnectorDefinition{}, stack.Wrap(fmt.Errorf("store: latest connector definition: %w", err))
	}
	return definition, nil
}

// LatestBuiltinConnectorDefinition returns the newest revision of a built-in definition, for
// a caller that is no customer: a provider delivering an event to a connector's route.
func (s *Store) LatestBuiltinConnectorDefinition(ctx context.Context, id string) (ConnectorDefinition, error) {
	if id == "" {
		return ConnectorDefinition{}, stack.Wrap(errors.New("store: a connector id is required"))
	}
	definition, err := latestDefinition(ctx, s.db, []string{BuiltinCustomer}, id)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorDefinition{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorDefinition, id))
	}
	if err != nil {
		return ConnectorDefinition{}, stack.Wrap(fmt.Errorf("store: latest built-in connector definition: %w", err))
	}
	return definition, nil
}

// How many connector definitions are handed back at once: the session list's numbers
// (sessions.go), since a catalog is read a page at a time in a picker the same way a
// sidebar is. Not measured for a catalog.
const (
	defaultConnectorDefinitionLimit = 25
	maxConnectorDefinitionLimit     = 200
)

// ConnectorDefinitionLimit is the page size a definition list uses for the limit asked for.
// ListConnectorDefinitions returns one row more than this.
func ConnectorDefinitionLimit(asked int) int {
	return clampLimit(asked, defaultConnectorDefinitionLimit, maxConnectorDefinitionLimit)
}

// ConnectorDefinitionFilter narrows a definition list and says where its page starts.
type ConnectorDefinitionFilter struct {
	// Text keeps the definitions whose id, name, category or description holds it, ignoring
	// case, as the prototype's catalog search did (internal/api/connectors.go:72 on
	// codex/connector-support at cf62af0d). It is matched on the newest revision only.
	Text   string
	Cursor *ConnectorDefinitionPosition
	Limit  int
}

// ConnectorDefinitionPosition is the last definition of a page. Custom stands in for the
// customer id, so a cursor holds nothing of whose it was and the order is the same for
// every customer: built-ins, then the customer's own.
type ConnectorDefinitionPosition struct {
	Custom bool   `json:"c"`
	ID     string `json:"id"`
}

// ListConnectorDefinitions returns the newest revision of every definition the customer can
// see that matches the filter: the built-ins first, then its own, each by id. It returns up
// to one more than ConnectorDefinitionLimit, so a caller can tell the page is not the last.
func (s *Store) ListConnectorDefinitions(ctx context.Context, customerID string, filter ConnectorDefinitionFilter) ([]ConnectorDefinition, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	// The newest revision first, then the text matched on it, so an older revision that
	// matched cannot stand in for a newer one that does not.
	latest := s.db.NewSelect().Model((*ConnectorDefinition)(nil)).
		DistinctOn("customer_id, id").
		Where("customer_id IN (?, ?)", BuiltinCustomer, customerID).
		Order("customer_id", "id", "revision DESC")
	if after := filter.Cursor; after != nil {
		// customer_id <> '' is false for a built-in, so it sorts first, as customer_id does.
		latest = latest.Where("(customer_id <> ?, id) > (?, ?)", BuiltinCustomer, after.Custom, after.ID)
	}
	var definitions []ConnectorDefinition
	query := s.db.NewSelect().Model(&definitions).ModelTableExpr("(?) AS cd", latest)
	if filter.Text != "" {
		// Backslash is ILIKE's default escape, so the caller's % and _ match themselves.
		escaped := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`).Replace(filter.Text)
		// Each field on its own, so a query cannot match across the boundary of two.
		query = query.Where("(cd.id ILIKE ?0 OR cd.name ILIKE ?0 OR cd.category ILIKE ?0 OR cd.description ILIKE ?0)", "%"+escaped+"%")
	}
	err := query.
		OrderExpr("cd.customer_id, cd.id").
		Limit(ConnectorDefinitionLimit(filter.Limit) + 1).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list connector definitions: %w", err))
	}
	return definitions, nil
}

// saveRevision stores a custom manifest as the next revision of its id under the customer,
// unless the latest revision already says the same, in which case that one is returned. A
// custom definition has one writer, the customer through the API, so the store numbers it.
//
// It runs under a transaction-scoped advisory lock on the customer and id, so two routers
// starting at once, or two requests creating the same custom id, take turns: the second
// finds the first's revision and either matches it or follows it. The primary key would
// refuse a duplicate revision anyway; the lock is what turns that into a wait, not an error.
func (s *Store) saveRevision(ctx context.Context, customerID string, manifest core.Manifest) (ConnectorDefinition, error) {
	var saved ConnectorDefinition
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, ?))",
			customerID+"/"+manifest.ID, definitionLockSeed); err != nil {
			return err
		}

		latest, err := latestDefinition(ctx, tx, []string{customerID}, manifest.ID)
		switch {
		case errors.Is(err, sql.ErrNoRows):
			manifest.Revision = 1
		case err != nil:
			return err
		default:
			same, err := sameManifest(latest.Manifest, manifest)
			if err != nil {
				return err
			}
			if same {
				saved = latest
				return nil
			}
			manifest.Revision = latest.Revision + 1
		}

		// Validated as it is stored, with the revision it is stored under.
		if err := manifest.Validate(); err != nil {
			return err
		}
		saved = ConnectorDefinition{
			CustomerID:  customerID,
			ID:          manifest.ID,
			Revision:    manifest.Revision,
			Name:        manifest.Name,
			Category:    manifest.Category,
			Description: manifest.Description,
			Manifest:    manifest,
			CreatedAt:   time.Now().UTC(),
		}
		// Read back what Postgres stored, which keeps microseconds, so the answer to a create
		// is the row a later read returns.
		_, err = tx.NewInsert().Model(&saved).Returning("created_at").Exec(ctx)
		return err
	})
	if err != nil {
		return ConnectorDefinition{}, stack.Wrap(fmt.Errorf("store: save connector definition %s: %w", manifest.ID, err))
	}
	return saved, nil
}

// seedBuiltin stores a built-in manifest at the revision its file names, under the same lock
// saveRevision takes, so routers starting at once take turns:
//   - that revision is stored and says the same: nothing to do, the usual restart;
//   - that revision is stored and says something else: the file was edited without a new
//     revision, which is refused so a connection pinned to it never reads two manifests;
//   - it is not stored: it is stored now.
//
// The latest is the highest revision, so an older build starting last, even on a database
// that has never seen its revision, cannot make its manifest latest again. Reverting a
// manifest is a new revision whose content is the old one.
func (s *Store) seedBuiltin(ctx context.Context, manifest core.Manifest) error {
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, ?))",
			BuiltinCustomer+"/"+manifest.ID, definitionLockSeed); err != nil {
			return err
		}

		var stored ConnectorDefinition
		err := tx.NewSelect().Model(&stored).
			Where("customer_id = ?", BuiltinCustomer).
			Where("id = ?", manifest.ID).
			Where("revision = ?", manifest.Revision).
			Scan(ctx)
		switch {
		case err == nil:
			same, err := sameManifest(stored.Manifest, manifest)
			if err != nil {
				return err
			}
			if !same {
				return fmt.Errorf("%s.yaml says revision %d, which is already stored with other content: give the change a new revision", manifest.ID, manifest.Revision)
			}
			return markBroken(ctx, tx, manifest)
		case !errors.Is(err, sql.ErrNoRows):
			return err
		}

		_, err = tx.NewInsert().Model(&ConnectorDefinition{
			CustomerID:  BuiltinCustomer,
			ID:          manifest.ID,
			Revision:    manifest.Revision,
			Name:        manifest.Name,
			Category:    manifest.Category,
			Description: manifest.Description,
			Manifest:    manifest,
			CreatedAt:   time.Now().UTC(),
		}).Exec(ctx)
		if err != nil {
			return err
		}
		return markBroken(ctx, tx, manifest)
	})
	if err != nil {
		return fmt.Errorf("store: seed connector definition %s: %w", manifest.ID, err)
	}
	return nil
}

// ConnectorBrokenRevision marks one revision of a built-in connector as not working, as a later
// revision's manifest declared it (core.Manifest.BrokenRevisions). Table
// connector_broken_revisions; the seeder is its one writer.
type ConnectorBrokenRevision struct {
	bun.BaseModel `bun:"table:connector_broken_revisions,alias:cbr"`

	ConnectorID string `bun:"connector_id,pk"`
	Revision    int    `bun:"revision,pk"`
	Reason      string `bun:"reason,notnull"`
	// MarkedBy is the revision whose manifest declared it.
	MarkedBy  int       `bun:"marked_by,notnull"`
	CreatedAt time.Time `bun:"created_at,notnull"`
}

// markBroken stores the marks manifest declares, in the seeder's transaction. A mark already
// stored is kept as it is, so marks only accumulate: a later revision that lists one no longer,
// or gives another reason, changes nothing.
func markBroken(ctx context.Context, tx bun.Tx, manifest core.Manifest) error {
	broken := manifest.Broken()
	if len(broken) == 0 {
		return nil
	}
	marks := make([]ConnectorBrokenRevision, 0, len(broken))
	for revision, reason := range broken {
		marks = append(marks, ConnectorBrokenRevision{ConnectorID: manifest.ID, Revision: revision, Reason: reason,
			MarkedBy: manifest.Revision, CreatedAt: time.Now().UTC()})
	}
	_, err := tx.NewInsert().Model(&marks).On("CONFLICT (connector_id, revision) DO NOTHING").Exec(ctx)
	return err
}

// BrokenConnectorRevision is why revision of connector is marked broken, and whether it is.
func (s *Store) BrokenConnectorRevision(ctx context.Context, connectorID string, revision int) (string, bool, error) {
	var mark ConnectorBrokenRevision
	err := s.db.NewSelect().Model(&mark).
		Where("connector_id = ?", connectorID).
		Where("revision = ?", revision).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, stack.Wrap(fmt.Errorf("store: broken connector revision: %w", err))
	}
	return mark.Reason, true, nil
}

// How a connection's pinned revision compares with its connector's (DefinitionStatus.Status).
const (
	// DefinitionCurrent: the connection reads its connector's latest revision.
	DefinitionCurrent = "current"
	// DefinitionOutdated: a later revision exists, and the pinned one is not marked broken.
	DefinitionOutdated = "outdated"
	// DefinitionBroken: a later revision marked the pinned one broken.
	DefinitionBroken = "broken"
)

// DefinitionStatus is how one connection's pinned revision compares with its connector's.
type DefinitionStatus struct {
	Status string
	// Reason is the mark's, when Status is DefinitionBroken.
	Reason string
}

// ConnectorDefinitionStatuses is the DefinitionStatus of each of the customer's connections, by
// connection id, in two reads whatever their number: the latest revision of each connector,
// and the marks on them.
func (s *Store) ConnectorDefinitionStatuses(ctx context.Context, customerID string, connections []ConnectorConnection) (map[string]DefinitionStatus, error) {
	statuses := make(map[string]DefinitionStatus, len(connections))
	if len(connections) == 0 {
		return statuses, nil
	}
	ids := make([]string, 0, len(connections))
	for _, connection := range connections {
		if !slices.Contains(ids, connection.ConnectorID) {
			ids = append(ids, connection.ConnectorID)
		}
	}
	var latest []struct {
		ID       string `bun:"id"`
		Revision int    `bun:"revision"`
	}
	err := s.db.NewSelect().Model((*ConnectorDefinition)(nil)).
		Column("id").ColumnExpr("max(revision) AS revision").
		Where("customer_id IN (?, ?)", BuiltinCustomer, customerID).
		Where("id IN (?)", bun.In(ids)).
		Group("id").
		Scan(ctx, &latest)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: latest connector revisions: %w", err))
	}
	var marks []ConnectorBrokenRevision
	err = s.db.NewSelect().Model(&marks).Where("connector_id IN (?)", bun.In(ids)).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: broken connector revisions: %w", err))
	}
	newest := make(map[string]int, len(latest))
	for _, row := range latest {
		newest[row.ID] = row.Revision
	}
	for _, connection := range connections {
		status := DefinitionStatus{Status: DefinitionCurrent}
		if connection.DefinitionRevision < newest[connection.ConnectorID] {
			status.Status = DefinitionOutdated
		}
		for _, mark := range marks {
			if mark.ConnectorID == connection.ConnectorID && mark.Revision == connection.DefinitionRevision {
				status = DefinitionStatus{Status: DefinitionBroken, Reason: mark.Reason}
			}
		}
		statuses[connection.ID] = status
	}
	return statuses, nil
}

// ErrConnectorDefinitionInUse says an unforced delete found a live connection made from the
// custom definition, or a live agent config binding it. DeletedConnector.Uses names them.
var ErrConnectorDefinitionInUse = errors.New("store: the connector is in use")

// ConnectorUses are what names one custom definition: what deleting it would break.
type ConnectorUses struct {
	// Connections are the ids of its live connections, app-owned and users', oldest first.
	Connections []string
	// Bindings are the bindings of the customer's live agent configs that name it, fixed or
	// chosen per session, by config name and alias.
	Bindings []ConnectorBindingUse
}

// ConnectorBindingUse is one binding of a live agent config that names a connector.
type ConnectorBindingUse struct {
	ConfigID   string `bun:"config_id"`
	ConfigName string `bun:"config_name"`
	// Binding is the alias the config binds it under.
	Binding string `bun:"binding"`
}

// DeletedConnector is what DeleteConnectorDefinition found and removed.
type DeletedConnector struct {
	// Uses is what named the definition when the delete looked, set on ErrConnectorDefinitionInUse too.
	Uses ConnectorUses
	// Connections are the live connections a forced delete removed.
	Connections []DeletedConnection
}

// DeleteConnectorDefinition removes every revision of one of the customer's own definitions.
// Unforced, a live connection made from it or a live agent config binding it refuses the
// delete with ErrConnectorDefinitionInUse, and nothing changes. Forced, its live connections
// are deleted as a forced connection delete deletes one (softDeleteConnection: credentials and
// tool pins dropped), and the bindings are left in place, naming a connector that no longer
// exists, as a forced connection delete leaves its bindings. Either way the rows that hold the
// customer's sealed secrets for the connector go in the same transaction: its OAuth client and
// provider app (connector_oauth_clients), its configuration token and its event destinations
// with their pending deliveries, so no secret outlives the connector it was for. Only the
// customer's own rows are read, so a built-in, stored under BuiltinCustomer, is never deleted:
// its id is ErrNoConnectorDefinition, as one nobody defined is.
//
// Every revision is locked FOR UPDATE first, under the lock saveRevision takes. A writer of a
// row that names the definition, a connection, an OAuth client or a config's binding, locks
// it FOR KEY SHARE before its write (lockCustomDefinitions), so of a delete and that write one
// always waits for the other: the write that came first is seen by the check that follows the
// lock, and one that came second finds no definition. The connections are locked FOR NO KEY
// UPDATE, which a config save's FOR KEY SHARE on the connections it binds does not wait for
// (https://www.postgresql.org/docs/current/explicit-locking.html#LOCKING-ROWS, table 13.3),
// so a save holding a connection while it waits for the definition cannot deadlock with this.
func (s *Store) DeleteConnectorDefinition(ctx context.Context, customerID, id string, force bool) (DeletedConnector, error) {
	if customerID == BuiltinCustomer || id == "" {
		return DeletedConnector{}, stack.Wrap(errors.New("store: a customer and a connector id are required"))
	}
	var deleted DeletedConnector
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, ?))",
			customerID+"/"+id, definitionLockSeed); err != nil {
			return err
		}
		var revisions []int
		err := tx.NewSelect().Model((*ConnectorDefinition)(nil)).Column("revision").
			Where("customer_id = ?", customerID).
			Where("id = ?", id).
			For("UPDATE").
			Scan(ctx, &revisions)
		if err != nil {
			return fmt.Errorf("store: lock connector definition: %w", err)
		}
		if len(revisions) == 0 {
			return fmt.Errorf("%w: %s", ErrNoConnectorDefinition, id)
		}
		var connections []struct {
			ID        string `bun:"id"`
			OwnerType string `bun:"owner_type"`
			HadGrant  bool   `bun:"had_grant"`
		}
		err = tx.NewSelect().Model((*ConnectorConnection)(nil)).
			Column("cc.id", "cc.owner_type").
			ColumnExpr("cc.credentials_sealed <> ''::bytea AS had_grant").
			Where("cc.customer_id = ?", customerID).
			Where("cc.connector_id = ?", id).
			Where("cc.deleted_at IS NULL").
			Order("cc.created_at", "cc.id").
			For("NO KEY UPDATE").
			Scan(ctx, &connections)
		if err != nil {
			return fmt.Errorf("store: lock the connector's connections: %w", err)
		}
		for _, connection := range connections {
			deleted.Uses.Connections = append(deleted.Uses.Connections, connection.ID)
		}
		// A row whose connectors is not an array binds nothing, as in ConnectorConnectionUses.
		err = tx.NewSelect().
			TableExpr("agent_configs AS ac").
			Join("CROSS JOIN LATERAL jsonb_array_elements(CASE WHEN jsonb_typeof(ac.connectors) = 'array' THEN ac.connectors ELSE '[]'::jsonb END) AS binding").
			ColumnExpr("ac.id AS config_id, ac.name AS config_name").
			ColumnExpr("binding ->> 'name' AS binding").
			Where("ac.customer_id = ?", customerID).
			Where("ac.deleted_at IS NULL").
			Where("binding ->> 'connector_id' = ?", id).
			OrderExpr("ac.name, ac.id, binding ->> 'name'").
			Scan(ctx, &deleted.Uses.Bindings)
		if err != nil {
			return fmt.Errorf("store: connector binding uses: %w", err)
		}
		if !force && (len(deleted.Uses.Connections) > 0 || len(deleted.Uses.Bindings) > 0) {
			return fmt.Errorf("%w: %s", ErrConnectorDefinitionInUse, id)
		}
		for _, connection := range connections {
			if _, err := softDeleteConnection(ctx, tx, customerID, connection.ID, false); err != nil {
				return err
			}
			deleted.Connections = append(deleted.Connections, DeletedConnection{
				ID: connection.ID, ConnectorID: id, OwnerType: connection.OwnerType, HadGrant: connection.HadGrant,
			})
		}
		// The deliveries go with their destination (ON DELETE CASCADE,
		// 20261007040000_connector_event_destinations.sql).
		for _, table := range []string{"connector_oauth_clients", "connector_config_tokens", "connector_event_destinations"} {
			if _, err := tx.ExecContext(ctx, "DELETE FROM "+table+" WHERE customer_id = ? AND connector_id = ?", customerID, id); err != nil {
				return fmt.Errorf("store: delete the connector's %s: %w", table, err)
			}
		}
		_, err = tx.NewDelete().Model((*ConnectorDefinition)(nil)).
			Where("customer_id = ?", customerID).
			Where("id = ?", id).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: delete connector definition: %w", err)
		}
		return nil
	})
	if err != nil {
		return deleted, stack.Wrap(err)
	}
	return deleted, nil
}

// lockCustomDefinitions locks every revision of the customer's own definitions among ids FOR
// KEY SHARE until tx ends, and returns the custom ids among them that have none. A writer of
// a row naming a custom definition calls it before its write, so DeleteConnectorDefinition
// and the write take turns. A built-in is never deleted, so it is neither locked nor reported.
// FOR KEY SHARE, not FOR SHARE, as in lockBoundConnections: definitions are never updated,
// and nothing but the delete needs to wait.
func lockCustomDefinitions(ctx context.Context, tx bun.Tx, customerID string, ids []string) ([]string, error) {
	var custom []string
	for _, id := range ids {
		if strings.HasPrefix(id, CustomPrefix) && !slices.Contains(custom, id) {
			custom = append(custom, id)
		}
	}
	if len(custom) == 0 {
		return nil, nil
	}
	var live []string
	err := tx.NewSelect().Model((*ConnectorDefinition)(nil)).Column("id").
		Where("customer_id = ?", customerID).
		Where("id IN (?)", bun.In(custom)).
		For("KEY SHARE").
		Scan(ctx, &live)
	if err != nil {
		return nil, fmt.Errorf("store: lock connector definitions: %w", err)
	}
	return slices.DeleteFunc(custom, func(id string) bool { return slices.Contains(live, id) }), nil
}

// latestDefinition is the newest revision of id under any of customers.
func latestDefinition(ctx context.Context, db bun.IDB, customers []string, id string) (ConnectorDefinition, error) {
	var definition ConnectorDefinition
	err := db.NewSelect().Model(&definition).
		Where("customer_id IN (?)", bun.In(customers)).
		Where("id = ?", id).
		Order("revision DESC").
		Limit(1).
		Scan(ctx)
	return definition, stack.Wrap(err)
}

// builtinManifests parses every *.yaml in fsys as a built-in manifest. Each must be valid,
// be named for its id, and not take the custom prefix; an error names the file, and the
// manifest's own error names the field.
func builtinManifests(fsys fs.FS) ([]core.Manifest, error) {
	paths, err := fs.Glob(fsys, "*.yaml")
	if err != nil {
		return nil, fmt.Errorf("store: built-in connectors: %w", err)
	}
	manifests := make([]core.Manifest, 0, len(paths))
	for _, path := range paths {
		raw, err := fs.ReadFile(fsys, path)
		if err != nil {
			return nil, fmt.Errorf("store: built-in connector %s: %w", path, err)
		}
		manifest, err := core.ParseManifest(raw)
		if err != nil {
			return nil, fmt.Errorf("store: built-in connector %s: %w", path, err)
		}
		// The file name is the id, so the file to edit for a connector is never in doubt.
		if name := strings.TrimSuffix(path, ".yaml"); manifest.ID != name {
			return nil, fmt.Errorf("store: built-in connector %s: id %q is not the file name %q", path, manifest.ID, name)
		}
		if strings.HasPrefix(manifest.ID, CustomPrefix) {
			return nil, fmt.Errorf("store: built-in connector %s: id %q takes the %s prefix kept for custom definitions", path, manifest.ID, CustomPrefix)
		}
		manifests = append(manifests, manifest)
	}
	return manifests, nil
}

// sameManifest is whether two manifests say the same thing, which is what decides whether a
// seeded manifest is a new revision. Both are compared as the JSON core.Manifest marshals to,
// with the revision left out: the YAML's comments, spacing, key order and quoting are gone by
// then, and Go writes struct fields in declaration order and map keys sorted, so equal
// manifests marshal to equal bytes. The stored side is decoded into the current
// core.Manifest first, so a key that model no longer has does not count as a change.
func sameManifest(stored, seeded core.Manifest) (bool, error) {
	stored.Revision, seeded.Revision = 0, 0
	a, err := json.Marshal(stored)
	if err != nil {
		return false, err
	}
	b, err := json.Marshal(seeded)
	if err != nil {
		return false, err
	}
	return bytes.Equal(a, b), nil
}

// ConnectorConnectionClient is the OAuth client a connection's grant was issued to (AI-990
// F16), kept in connector_connection_clients
// (20261011210000_connector_clients_and_argument_shapes.sql) so it is read without unsealing the
// credentials. A client_id is not a secret (RFC 6749 section 2.2).
type ConnectorConnectionClient struct {
	bun.BaseModel `bun:"table:connector_connection_clients,alias:ccc"`

	ConnectionID string                        `bun:"connection_id,pk"`
	Registration core.ClientRegistrationMethod `bun:"registration,notnull"`
	ClientID     string                        `bun:"client_id,notnull"`
	UpdatedAt    time.Time                     `bun:"updated_at,notnull"`
}

// PutConnectorConnectionClient records the client a consent of the connection used, replacing
// the one an earlier consent recorded.
func (s *Store) PutConnectorConnectionClient(ctx context.Context, client *ConnectorConnectionClient) error {
	if client.ConnectionID == "" || client.ClientID == "" {
		return stack.Wrap(errors.New("store: a connection client needs a connection and a client id"))
	}
	client.UpdatedAt = time.Now().UTC().Truncate(time.Microsecond)
	_, err := s.db.NewInsert().Model(client).
		On("CONFLICT (connection_id) DO UPDATE").
		Set("registration = EXCLUDED.registration, client_id = EXCLUDED.client_id, updated_at = EXCLUDED.updated_at").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: put connection client: %w", err))
	}
	return nil
}

// ConnectorConnectionClients is the recorded client of each of connectionIDs that has one, by
// connection id. The caller has already scoped the ids to its customer.
func (s *Store) ConnectorConnectionClients(ctx context.Context, connectionIDs []string) (map[string]ConnectorConnectionClient, error) {
	byConnection := map[string]ConnectorConnectionClient{}
	if len(connectionIDs) == 0 {
		return byConnection, nil
	}
	clients := []ConnectorConnectionClient{}
	if err := s.db.NewSelect().Model(&clients).Where("connection_id IN (?)", bun.In(connectionIDs)).Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list connection clients: %w", err))
	}
	for _, client := range clients {
		byConnection[client.ConnectionID] = client
	}
	return byConnection, nil
}

// ConnectorConnectionValidation is the last validate of a connection (AI-1052), kept in
// connector_connection_validations (20261016120000_connector_connection_validations.sql): its
// status, its code (a stable reason, or the provider's HTTP status), its error and when it ran,
// and the revision of the credentials it checked. The error is the validate's with the
// credential's values cut out, and capped (api.storedError).
type ConnectorConnectionValidation struct {
	bun.BaseModel `bun:"table:connector_connection_validations,alias:ccv"`

	ConnectionID string `bun:"connection_id,pk"`
	// Revision is the connection's revision (ConnectorConnection.Revision) whose credentials
	// the validate checked. A connection read at a later revision has newer credentials than
	// the validate saw.
	Revision  int       `bun:"revision,notnull"`
	Status    string    `bun:"status,notnull"`
	Code      string    `bun:"code,notnull"`
	Error     string    `bun:"error,notnull"`
	CheckedAt time.Time `bun:"checked_at,notnull"`
}

// PutConnectorConnectionValidation records a validate of the connection, replacing the one an
// earlier validate recorded. Of two validates at once, the one of the later revision stays,
// and of two of the same revision the one that ran later, whichever writes last: a validate
// of credentials replaced while it ran never covers one of the new credentials.
func (s *Store) PutConnectorConnectionValidation(ctx context.Context, validation *ConnectorConnectionValidation) error {
	if validation.ConnectionID == "" || validation.Status == "" || validation.CheckedAt.IsZero() {
		return stack.Wrap(errors.New("store: a connection validation needs a connection, a status and a time"))
	}
	_, err := s.db.NewInsert().Model(validation).
		On("CONFLICT (connection_id) DO UPDATE").
		Set("revision = EXCLUDED.revision, status = EXCLUDED.status, code = EXCLUDED.code, error = EXCLUDED.error, checked_at = EXCLUDED.checked_at").
		Where("(ccv.revision, ccv.checked_at) <= (EXCLUDED.revision, EXCLUDED.checked_at)").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: put connection validation: %w", err))
	}
	return nil
}

// ConnectorConnectionValidations is the last validate of each of connectionIDs that was
// validated, by connection id. The caller has already scoped the ids to its customer.
func (s *Store) ConnectorConnectionValidations(ctx context.Context, connectionIDs []string) (map[string]ConnectorConnectionValidation, error) {
	byConnection := map[string]ConnectorConnectionValidation{}
	if len(connectionIDs) == 0 {
		return byConnection, nil
	}
	validations := []ConnectorConnectionValidation{}
	if err := s.db.NewSelect().Model(&validations).Where("connection_id IN (?)", bun.In(connectionIDs)).Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list connection validations: %w", err))
	}
	for _, validation := range validations {
		byConnection[validation.ConnectionID] = validation
	}
	return byConnection, nil
}
