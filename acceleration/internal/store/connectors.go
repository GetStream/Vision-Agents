package store

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"strings"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
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
		return ConnectorDefinition{}, errors.New("store: customer id is required")
	}
	if !strings.HasPrefix(manifest.ID, CustomPrefix) {
		return ConnectorDefinition{}, fmt.Errorf("store: a custom connector id starts with %s, so it cannot shadow a built-in: %q", CustomPrefix, manifest.ID)
	}
	return s.saveRevision(ctx, customerID, manifest)
}

// ConnectorDefinition returns one revision of a definition the customer can see: a built-in
// or its own.
func (s *Store) ConnectorDefinition(ctx context.Context, customerID, id string, revision int) (ConnectorDefinition, error) {
	if customerID == "" || id == "" {
		return ConnectorDefinition{}, errors.New("store: a customer and a connector id are required")
	}

	var definition ConnectorDefinition
	err := s.db.NewSelect().Model(&definition).
		Where("customer_id IN (?, ?)", BuiltinCustomer, customerID).
		Where("id = ?", id).
		Where("revision = ?", revision).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorDefinition{}, fmt.Errorf("%w: %s revision %d", ErrNoConnectorDefinition, id, revision)
	}
	if err != nil {
		return ConnectorDefinition{}, fmt.Errorf("store: connector definition: %w", err)
	}
	return definition, nil
}

// LatestConnectorDefinition returns the newest revision of a definition the customer can see.
func (s *Store) LatestConnectorDefinition(ctx context.Context, customerID, id string) (ConnectorDefinition, error) {
	if customerID == "" || id == "" {
		return ConnectorDefinition{}, errors.New("store: a customer and a connector id are required")
	}
	definition, err := latestDefinition(ctx, s.db, []string{BuiltinCustomer, customerID}, id)
	if errors.Is(err, sql.ErrNoRows) {
		return ConnectorDefinition{}, fmt.Errorf("%w: %s", ErrNoConnectorDefinition, id)
	}
	if err != nil {
		return ConnectorDefinition{}, fmt.Errorf("store: latest connector definition: %w", err)
	}
	return definition, nil
}

// ListConnectorDefinitions returns the newest revision of every definition the customer can
// see: the built-ins first, then its own, each by id.
func (s *Store) ListConnectorDefinitions(ctx context.Context, customerID string) ([]ConnectorDefinition, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	var definitions []ConnectorDefinition
	err := s.db.NewSelect().Model(&definitions).
		DistinctOn("customer_id, id").
		Where("customer_id IN (?, ?)", BuiltinCustomer, customerID).
		Order("customer_id", "id", "revision DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: list connector definitions: %w", err)
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
		_, err = tx.NewInsert().Model(&saved).Exec(ctx)
		return err
	})
	if err != nil {
		return ConnectorDefinition{}, fmt.Errorf("store: save connector definition %s: %w", manifest.ID, err)
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
			return nil
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
		return err
	})
	if err != nil {
		return fmt.Errorf("store: seed connector definition %s: %w", manifest.ID, err)
	}
	return nil
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
	return definition, err
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
