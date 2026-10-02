// Package pgsealed is the core.Backend that keeps a grant in Postgres: the connection row in
// connector_connections, its Material sealed with auth.Sealer, under a session-level
// advisory lock held across replicas (store.WithLockedConnectorConnection).
package pgsealed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// unreadableError is what LastError says when the stored material does not open: the key
// that sealed it is gone, or the blob is not this row's at this revision. Either way only a
// reconnect gives the connection a grant again (core.Grant.LastError says what to do).
const unreadableError = "The stored credential could not be opened; reconnect the account"

// errRevisionIsTheBackends says fn moved Grant.Revision. The backend advances it, by one,
// whenever Material changes, so the revision a blob is sealed for is never chosen by fn.
var errRevisionIsTheBackends = errors.New("pgsealed: the backend advances the revision; fn must not change it")

// Backend implements core.Backend over a store and a sealer.
type Backend struct {
	store  *store.Store
	sealer *auth.Sealer
}

var _ core.Backend = (*Backend)(nil)

// New is a backend over db, sealing with sealer's current key version.
func New(db *store.Store, sealer *auth.Sealer) (*Backend, error) {
	if db == nil || sealer == nil {
		return nil, errors.New("pgsealed: a store and a sealer are required")
	}
	return &Backend{store: db, sealer: sealer}, nil
}

// WithLocked loads the grant under the connection's lock, opens its Material and runs fn.
//
// The backend owns Grant.Revision: a commit whose Material differs from the stored one seals
// it for the next revision; one that keeps it keeps the revision. Material under an older key
// is sealed again under the current one by the first use that returns no error, even one
// that changed nothing. Material that does not open is handed to fn empty, with the grant at
// needs_reauthorization, committed before fn runs: no use can mint from it, and a reconnect
// can still replace it. Its blob stays as it is until then.
func (b *Backend) WithLocked(ctx context.Context, ref core.ConnectionRef,
	fn func(g *core.Grant, checkpoint func() error) (changed bool, err error)) error {
	return b.store.WithLockedConnectorConnection(ctx, ref.CustomerID, ref.ConnectionID,
		func(connection *store.ConnectorConnection, save func() error) (bool, error) {
			stored := &locked{connection: connection}
			stored.material, stored.opened = b.open(ref, connection)
			grant := stored.grant()
			if !stored.opened && connection.Status != store.ConnectionNeedsReauthorization {
				grant.Status = store.ConnectionNeedsReauthorization
				grant.LastError = unreadableError
				if err := b.commit(ref, stored, &grant, save); err != nil {
					return false, err
				}
			}
			changed, err := fn(&grant, func() error { return b.commit(ref, stored, &grant, save) })
			if err != nil {
				return false, err
			}
			if changed || b.stale(stored) {
				return false, b.commit(ref, stored, &grant, save)
			}
			return false, nil
		})
}

// locked is the connection as last committed, and the Material it holds when it opened.
type locked struct {
	connection *store.ConnectorConnection
	material   core.Material
	opened     bool
}

// grant is what fn sees of the connection.
func (l *locked) grant() core.Grant {
	grant := core.Grant{
		Revision:  l.connection.Revision,
		Status:    l.connection.Status,
		Material:  cloneMaterial(l.material),
		LastError: l.connection.LastError,
	}
	if l.connection.ExpiresAt != nil {
		grant.ExpiresAt = *l.connection.ExpiresAt
	}
	return grant
}

// stale reports material that opened under a key version other than the current one.
func (b *Backend) stale(stored *locked) bool {
	return stored.opened && len(stored.connection.MaterialSealed) > 0 &&
		stored.connection.MaterialKEKVersion != b.sealer.CurrentVersion()
}

// commit writes grant onto the connection and saves it at the revision last committed. New
// Material is sealed for the next revision; stale material is sealed again under the current
// key at the same revision, since the AAD binds the revision, not the key. A save that fails
// leaves the connection as it was.
func (b *Backend) commit(ref core.ConnectionRef, stored *locked, grant *core.Grant, save func() error) error {
	connection := stored.connection
	if grant.Revision != connection.Revision {
		return errRevisionIsTheBackends
	}
	previous := *connection
	revision := connection.Revision
	if !sameMaterial(grant.Material, stored.material) {
		revision++
	}
	if revision != connection.Revision || b.stale(stored) {
		sealed, version, err := b.seal(ref, revision, grant.Material)
		if err != nil {
			return err
		}
		connection.Revision, connection.MaterialSealed, connection.MaterialKEKVersion = revision, sealed, version
	}
	connection.Status = grant.Status
	connection.LastError = grant.LastError
	connection.ExpiresAt = nil
	if !grant.ExpiresAt.IsZero() {
		expires := grant.ExpiresAt.UTC().Truncate(time.Microsecond)
		connection.ExpiresAt = &expires
	}
	if err := save(); err != nil {
		*connection = previous
		return err
	}
	if revision != previous.Revision {
		stored.material, stored.opened = cloneMaterial(grant.Material), true
	}
	grant.Revision = connection.Revision
	return nil
}

// seal is Material as stored: empty, at key version 0 (the migration's "no material"), when
// there is none, else its JSON sealed under the current key for this row and revision.
func (b *Backend) seal(ref core.ConnectionRef, revision int, m core.Material) ([]byte, int, error) {
	if isEmpty(m) {
		return []byte{}, 0, nil
	}
	raw, err := json.Marshal(m)
	if err != nil {
		return nil, 0, fmt.Errorf("pgsealed: encode material: %w", err)
	}
	sealed, err := b.sealer.SealWithAAD(string(raw), materialAAD(ref, revision))
	if err != nil {
		return nil, 0, fmt.Errorf("pgsealed: seal material: %w", err)
	}
	return sealed, b.sealer.CurrentVersion(), nil
}

// open is the connection's Material, and false when there is a blob that does not open for
// this row at this revision under the key version it names.
func (b *Backend) open(ref core.ConnectionRef, connection *store.ConnectorConnection) (core.Material, bool) {
	if len(connection.MaterialSealed) == 0 {
		return core.Material{}, true
	}
	plain, err := b.sealer.OpenWithAADVersion(connection.MaterialSealed,
		materialAAD(ref, connection.Revision), connection.MaterialKEKVersion)
	if err != nil {
		return core.Material{}, false
	}
	var material core.Material
	if err := json.Unmarshal([]byte(plain), &material); err != nil {
		return core.Material{}, false
	}
	return material, true
}

// materialAAD binds a sealed Material to its tenant, connection and revision, so a blob
// copied to another row, or written back at a later revision, does not open. Each string is
// length-prefixed, so no customer and id split the same bytes two ways. The label is not the
// prototype's ("accelerate:connector-credentials:v1", connectors/secrets.go:113 at cf62af0d):
// what is sealed is a core.Material, not its Credentials, so neither opens as the other. v1
// changes with the layout.
func materialAAD(ref core.ConnectionRef, revision int) []byte {
	return fmt.Appendf(nil, "accelerate:connector-material:v1:%d:%s:%d:%s:%d",
		len(ref.CustomerID), ref.CustomerID, len(ref.ConnectionID), ref.ConnectionID, revision)
}

func isEmpty(m core.Material) bool {
	return m.Scheme == "" && m.Version == 0 && len(m.Payload) == 0
}

func sameMaterial(a, b core.Material) bool {
	return a.Scheme == b.Scheme && a.Version == b.Version && bytes.Equal(a.Payload, b.Payload)
}

// cloneMaterial copies Payload, so fn changing it in place is a change commit sees.
func cloneMaterial(m core.Material) core.Material {
	m.Payload = bytes.Clone(m.Payload)
	return m
}
