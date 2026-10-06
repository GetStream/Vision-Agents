package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ConnectionsByIdentity is every live connection of a connector, of any customer, whose
// inputs hold every pair of inputs and whose metadata hold every pair of metadata: the
// connections of the account a provider event names (core.Signal.Identity), split by whether
// each identity part is an input or a captured value. An event names fewer parts for every
// account that has them, such as a workspace uninstall that names the team and no user.
//
// It reads across customers because an event on a connector's own route is signed with the
// operator's secret, which every customer's connections to that connector share; a custom
// connector's id is its customer's alone (CustomPrefix), so no other customer's connection
// can match. The parts are compared one by one as stored (JSONB containment, @>), never by
// splitting the joined account id, whose parts may themselves hold the ":" it is joined with.
func (s *Store) ConnectionsByIdentity(ctx context.Context, connectorID string, inputs, metadata map[string]string) ([]core.ConnectionRef, error) {
	if connectorID == "" || len(inputs)+len(metadata) == 0 {
		return nil, stack.Wrap(errors.New("store: a connector id and at least one identity part are required"))
	}
	inputsJSON, err := json.Marshal(nonNil(inputs))
	if err != nil {
		return nil, stack.Wrap(err)
	}
	metadataJSON, err := json.Marshal(nonNil(metadata))
	if err != nil {
		return nil, stack.Wrap(err)
	}
	var connections []ConnectorConnection
	err = s.db.NewSelect().Model(&connections).
		Column("customer_id", "id").
		Where("connector_id = ?", connectorID).
		Where("inputs @> ?::jsonb", string(inputsJSON)).
		Where("metadata @> ?::jsonb", string(metadataJSON)).
		Where("deleted_at IS NULL").
		Order("customer_id", "id").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: connections by identity: %w", err))
	}
	refs := make([]core.ConnectionRef, 0, len(connections))
	for _, connection := range connections {
		refs = append(refs, core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID})
	}
	return refs, nil
}

// nonNil is pairs, or no pairs when it is nil, so it marshals as an object.
func nonNil(pairs map[string]string) map[string]string {
	if pairs == nil {
		return map[string]string{}
	}
	return pairs
}
