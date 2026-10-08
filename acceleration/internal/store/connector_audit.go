package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// What an audit row records (ConnectorAuditEvent.Action). The three grant actions are T47's
// (subtasks.md on connectors/planning: «grant created, refreshed or revoked»); the proxy call
// and the token export are T44's and T45's, allowed now so neither needs a migration.
const (
	AuditGrantCreated   = "grant_created"
	AuditGrantRefreshed = "grant_refreshed"
	AuditGrantRevoked   = "grant_revoked"
	AuditProxyCall      = "proxy_call"
	AuditTokenExport    = "token_export"
)

// Why a grant was created or revoked (ConnectorAuditEvent.Reason). A revocation the resolver
// writes gives the provider's outcome or signal kind instead (core.OutcomeKind,
// core.SignalKind), so the reason is what the provider said, in its own word.
const (
	// AuditReasonConsent: an OAuth consent finished at the callback.
	AuditReasonConsent = "consent"
	// AuditReasonCredentials: the app's backend stored a credential it holds itself.
	AuditReasonCredentials = "credentials"
	// AuditReasonDeleted: the connection was deleted.
	AuditReasonDeleted = "deleted"
	// AuditReasonUserDeleted: every connection of its user was deleted, for offboarding.
	AuditReasonUserDeleted = "user_deleted"
)

// How many audit rows are handed back at once: the connection list's sizes. Neither is
// measured.
const (
	defaultAuditLimit = 25
	maxAuditLimit     = 200
)

// ConnectorAuditEvent is one grant a connection got, renewed or lost, with the ids that tie it
// to what caused it. It names no owner id and no account id, so it outlives a user delete
// (20261007170000_connector_invocations_and_audit.sql).
type ConnectorAuditEvent struct {
	bun.BaseModel `bun:"table:connector_audit,alias:ca"`

	ID           string `bun:"id,pk"`
	CustomerID   string `bun:"customer_id,notnull"`
	ConnectionID string `bun:"connection_id,notnull"`
	ConnectorID  string `bun:"connector_id,notnull"`
	OwnerType    string `bun:"owner_type,notnull"`
	Action       string `bun:"action,notnull"`
	Reason       string `bun:"reason,notnull"`
	// Revision is the connection's credential revision once the change committed, 0 when it
	// names none.
	Revision int `bun:"revision,notnull"`
	// RequestID, SessionID and AttemptID are the correlation ids, each empty when nothing of
	// that kind caused it.
	RequestID string `bun:"request_id,notnull"`
	SessionID string `bun:"session_id,notnull"`
	AttemptID string `bun:"attempt_id,notnull"`
	// StatusCode, LatencyMs and Target are a proxy call's (T44), nil and empty for a grant.
	StatusCode *int      `bun:"status_code"`
	LatencyMs  *int64    `bun:"latency_ms"`
	Target     string    `bun:"target,notnull"`
	CreatedAt  time.Time `bun:"created_at,notnull"`
}

// AuditFilter picks a customer's audit rows, of one connection when ConnectionID is set, a
// page at a time.
type AuditFilter struct {
	ConnectionID string
	Limit        int
	// After is the last row of the previous page.
	After *AuditPosition
}

// AuditPosition is where a page of audit rows ended, newest first.
type AuditPosition struct {
	CreatedAt time.Time `json:"c"`
	ID        string    `json:"id"`
}

// AuditLimit is the page size an audit list uses for the limit asked for.
// ConnectorAuditEvents returns one row more than this.
func AuditLimit(asked int) int {
	return clampLimit(asked, defaultAuditLimit, maxAuditLimit)
}

// RecordConnectorAudit stores one audit row, at now.
func (s *Store) RecordConnectorAudit(ctx context.Context, event *ConnectorAuditEvent) error {
	if event.CustomerID == "" || event.ConnectionID == "" {
		return stack.Wrap(errors.New("store: an audit row needs a customer and a connection"))
	}
	switch event.Action {
	case AuditGrantCreated, AuditGrantRefreshed, AuditGrantRevoked, AuditProxyCall, AuditTokenExport:
	default:
		return stack.Wrap(fmt.Errorf("store: %q is not an audit action", event.Action))
	}
	event.ID = newID()
	event.CreatedAt = time.Now().UTC().Truncate(time.Microsecond)
	if _, err := s.db.NewInsert().Model(event).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: record connector audit: %w", err))
	}
	return nil
}

// ConnectorAuditEvents lists the customer's audit rows, newest first, one more than
// AuditLimit(filter.Limit). A deleted connection's rows are listed too.
func (s *Store) ConnectorAuditEvents(ctx context.Context, customerID string, filter AuditFilter) ([]ConnectorAuditEvent, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: a customer id is required"))
	}
	events := []ConnectorAuditEvent{}
	query := s.db.NewSelect().Model(&events).Where("customer_id = ?", customerID)
	if filter.ConnectionID != "" {
		query = query.Where("connection_id = ?", filter.ConnectionID)
	}
	if after := filter.After; after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
	}
	err := query.
		Order("created_at DESC", "id DESC").
		Limit(AuditLimit(filter.Limit) + 1).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list connector audit: %w", err))
	}
	return events, nil
}
