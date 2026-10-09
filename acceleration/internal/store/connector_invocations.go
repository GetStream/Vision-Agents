package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// How a connector tool call failed (ConnectorInvocation.ErrorType). The set is the subtask's
// (subtasks.md T29 on connectors/planning); the first four are the architecture doc's
// («Add» item 8), after ElevenLabs' tool executions (competitor-analysis.md, «Call history»).
const (
	// InvocationCustomerAuth: the provider refused the connection's credential, or the
	// connection had none to give. Whoever owns it has to reconnect.
	InvocationCustomerAuth = "customer_auth"
	// InvocationExternalServer: the provider answered with a failure, or could not be reached.
	InvocationExternalServer = "external_server"
	// InvocationClientTimeout: the connection's own client gave up on the request before the
	// provider answered, and before the binding's deadline.
	InvocationClientTimeout = "client_timeout"
	// InvocationOutcomeUnknown: the call was sent and cut off before it answered, by the
	// binding's deadline or an interrupted turn, so the provider may have done it.
	InvocationOutcomeUnknown = "outcome_unknown"
	// InvocationDenied: the router refused the call before anything was sent.
	InvocationDenied = "denied"
)

// How many invocations are handed back at once: the connection list's sizes, since both are
// lists a dashboard shows a page of. Neither is measured.
const (
	defaultInvocationLimit = 25
	maxInvocationLimit     = 200
)

// ConnectorInvocation is one connector tool call a session's dispatcher ran. It holds no
// argument and no result (20261007170000_connector_invocations_and_audit.sql).
type ConnectorInvocation struct {
	bun.BaseModel `bun:"table:connector_invocations,alias:ci"`

	ID           string `bun:"id,pk"`
	CustomerID   string `bun:"customer_id,notnull"`
	ConnectionID string `bun:"connection_id,notnull"`
	ConnectorID  string `bun:"connector_id,notnull"`
	ConfigID     string `bun:"config_id,notnull"`
	// Binding is the alias the config binds the connector under.
	Binding string `bun:"binding,notnull"`
	// Tool is the tool's name at the provider, without the alias.
	Tool string `bun:"tool,notnull"`
	// SessionID is empty for an incognito session.
	SessionID string    `bun:"session_id,notnull"`
	StartedAt time.Time `bun:"started_at,notnull"`
	LatencyMs int64     `bun:"latency_ms,notnull"`
	// ErrorType is one of the Invocation* values, empty for a call that answered.
	ErrorType string `bun:"error_type,notnull"`
	// Arguments is the shape of what the call was asked, never a value (AI-990 F40), nil when
	// its arguments were not a JSON object. Kept in connector_invocation_arguments
	// (20261011210000_connector_clients_and_argument_shapes.sql).
	Arguments []ArgumentShape `bun:"-"`
}

// ArgumentShape is one top-level argument of a tool call without its value: its name, its JSON
// type (RFC 8259 section 3: object, array, string, number, boolean or null) and, for a string
// or an array, its length, so an empty one shows (thread_ts: string of length 0).
type ArgumentShape struct {
	Name string `json:"name"`
	Type string `json:"type"`
	// Length is a string's characters (Unicode code points) or an array's elements, nil for
	// any other type.
	Length *int `json:"length,omitempty"`
}

// connectorInvocationArguments is one invocation's ArgumentShapes as connector_invocation_arguments
// keeps them.
type connectorInvocationArguments struct {
	bun.BaseModel `bun:"table:connector_invocation_arguments,alias:cia"`

	InvocationID string          `bun:"invocation_id,pk"`
	Shape        []ArgumentShape `bun:"shape,type:jsonb,notnull"`
}

// InvocationPosition is where a page of invocations ended, newest first.
type InvocationPosition struct {
	StartedAt time.Time `json:"s"`
	ID        string    `json:"id"`
}

// InvocationLimit is the page size an invocation list uses for the limit asked for.
// ConnectorInvocations returns one row more than this, so a caller can tell the page is not
// the last without counting.
func InvocationLimit(asked int) int {
	return clampLimit(asked, defaultInvocationLimit, maxInvocationLimit)
}

// RecordConnectorInvocation stores one call.
func (s *Store) RecordConnectorInvocation(ctx context.Context, invocation *ConnectorInvocation) error {
	if invocation.CustomerID == "" || invocation.ConnectionID == "" || invocation.Tool == "" {
		return stack.Wrap(errors.New("store: an invocation needs a customer, a connection and a tool"))
	}
	switch invocation.ErrorType {
	case "", InvocationCustomerAuth, InvocationExternalServer, InvocationClientTimeout, InvocationOutcomeUnknown, InvocationDenied:
	default:
		return stack.Wrap(fmt.Errorf("store: %q is not an invocation error type", invocation.ErrorType))
	}
	invocation.ID = newID()
	// Truncated to what Postgres keeps, so the row handed back is the row a read returns.
	invocation.StartedAt = invocation.StartedAt.UTC().Truncate(time.Microsecond)
	var err error
	if invocation.Arguments == nil {
		_, err = s.db.NewInsert().Model(invocation).Exec(ctx)
	} else {
		err = s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
			if _, err := tx.NewInsert().Model(invocation).Exec(ctx); err != nil {
				return err
			}
			_, err := tx.NewInsert().Model(&connectorInvocationArguments{InvocationID: invocation.ID, Shape: invocation.Arguments}).Exec(ctx)
			return err
		})
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: record connector invocation: %w", err))
	}
	return nil
}

// ConnectorInvocations lists one connection's calls, newest first, one more than
// InvocationLimit(limit).
func (s *Store) ConnectorInvocations(ctx context.Context, customerID, connectionID string, limit int, after *InvocationPosition) ([]ConnectorInvocation, error) {
	if customerID == "" || connectionID == "" {
		return nil, stack.Wrap(errors.New("store: a customer and a connection id are required"))
	}
	invocations := []ConnectorInvocation{}
	query := s.db.NewSelect().Model(&invocations).
		Where("customer_id = ?", customerID).
		Where("connection_id = ?", connectionID)
	if after != nil {
		query = query.Where("(started_at, id) < (?, ?)", after.StartedAt, after.ID)
	}
	err := query.
		Order("started_at DESC", "id DESC").
		Limit(InvocationLimit(limit) + 1).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list connector invocations: %w", err))
	}
	if len(invocations) == 0 {
		return invocations, nil
	}
	ids := make([]string, len(invocations))
	for i, invocation := range invocations {
		ids[i] = invocation.ID
	}
	shapes := []connectorInvocationArguments{}
	if err := s.db.NewSelect().Model(&shapes).Where("invocation_id IN (?)", bun.In(ids)).Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list connector invocation arguments: %w", err))
	}
	byInvocation := make(map[string][]ArgumentShape, len(shapes))
	for _, shape := range shapes {
		byInvocation[shape.InvocationID] = shape.Shape
	}
	for i := range invocations {
		invocations[i].Arguments = byInvocation[invocations[i].ID]
	}
	return invocations, nil
}
