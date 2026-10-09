package store

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ConnectorToolPin is the schema digest one tool of one connection had when a session first
// offered it, for a tool a session binding grants by name alone
// (20261011150000_connector_tool_pins.sql). It holds for the grant the connection had then,
// its ConnectedAt: a reconnect is a new trust event, and the next session pins again.
type ConnectorToolPin struct {
	bun.BaseModel `bun:"table:connector_tool_pins,alias:ctp"`

	ConnectionID string `bun:"connection_id,pk"`
	ToolName     string `bun:"tool_name,pk"`
	SchemaDigest string `bun:"schema_digest,notnull"`
	// ConnectedAt is the connection's ConnectedAt when the pin was taken; nil for a connection
	// connected before connected_at was kept.
	ConnectedAt *time.Time `bun:"connected_at"`
	PinnedAt    time.Time  `bun:"pinned_at,notnull"`
}

// ConnectorToolPins are the digests a connection's tools are pinned at, by tool name, for the
// grant that began at connectedAt: the connection's ConnectedAt as the caller read it. A pin
// taken under another grant is not one.
func (s *Store) ConnectorToolPins(ctx context.Context, connectionID string, connectedAt *time.Time) (map[string]string, error) {
	if connectionID == "" {
		return nil, stack.Wrap(errors.New("store: a connection id is required"))
	}
	var pins []ConnectorToolPin
	err := s.db.NewSelect().Model(&pins).
		Where("ctp.connection_id = ?", connectionID).
		Where("ctp.connected_at IS NOT DISTINCT FROM ?", connectedAt).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: connector tool pins: %w", err))
	}
	digests := make(map[string]string, len(pins))
	for _, pin := range pins {
		digests[pin.ToolName] = pin.SchemaDigest
	}
	return digests, nil
}

// PinConnectorTools pins each tool of digests, by name, at its digest for the grant that
// began at connectedAt, and returns the pins that hold for that grant afterwards, as
// ConnectorToolPins does.
//
// A pin is written once per grant: a tool already pinned for this grant keeps its digest,
// so of two sessions that pin at once the first one's stands and both read it back. A pin of
// an older grant is replaced; one of a newer grant, written by a router that read the
// connection after a reconnect this caller has not seen, is kept.
func (s *Store) PinConnectorTools(ctx context.Context, connectionID string, connectedAt *time.Time, digests map[string]string) (map[string]string, error) {
	if connectionID == "" {
		return nil, stack.Wrap(errors.New("store: a connection id is required"))
	}
	if len(digests) > 0 {
		now := time.Now().UTC()
		pins := make([]ConnectorToolPin, 0, len(digests))
		// In one order, so two sessions pinning the same tools at once take the rows' locks in
		// the same order and neither waits on the other in a cycle.
		for _, name := range slices.Sorted(maps.Keys(digests)) {
			pins = append(pins, ConnectorToolPin{ConnectionID: connectionID, ToolName: name,
				SchemaDigest: digests[name], ConnectedAt: connectedAt, PinnedAt: now})
		}
		_, err := s.db.NewInsert().Model(&pins).
			On("CONFLICT (connection_id, tool_name) DO UPDATE").
			Set("schema_digest = EXCLUDED.schema_digest").
			Set("connected_at = EXCLUDED.connected_at").
			Set("pinned_at = EXCLUDED.pinned_at").
			Where("EXCLUDED.connected_at > ctp.connected_at OR (ctp.connected_at IS NULL AND EXCLUDED.connected_at IS NOT NULL)").
			Exec(ctx)
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("store: pin connector tools: %w", err))
		}
	}
	return s.ConnectorToolPins(ctx, connectionID, connectedAt)
}
