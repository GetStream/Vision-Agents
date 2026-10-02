package store

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
)

// ForeignStreamApp is the pin an imported row is given when the app it names is not one
// the importing customer acts in here. No Stream app has it, so work pinned to it is
// parked rather than finished anywhere.
const ForeignStreamApp int64 = -1

// pinColumn is the column a Stream app pin is kept in.
const pinColumn = "stream_app_pk"

// pinnedTables are the tables an export carries whose rows are pinned to a Stream app.
var pinnedTables = map[string]bool{"agent_sessions": true, "calls": true, "phone_numbers": true}

// StreamPins is how a pin crosses from one deployment to another.
//
// A pin names the same Stream app wherever it is read, except NULL, which means the app
// of whichever deployment wrote it. So an export says what NULL meant when it knows, and
// an import never takes a pin on trust: a row may name only the app its customer acts in
// on the importing side.
type StreamPins struct {
	// Deployment is this deployment's own Stream app id, zero while it is not known.
	Deployment func() int64
	// For is the pin new work for a customer is given here: the customer's app, or zero
	// for the deployment's own.
	For func(ctx context.Context, customerID string) (int64, error)
}

// SetStreamPins says how this deployment's pins cross. Without it NULL stays NULL on the
// way out and every other pin is foreign on the way in, which is right for a deployment
// that has never pinned anything.
func (s *Store) SetStreamPins(pins StreamPins) { s.pins.Store(&pins) }

func (s *Store) streamPins() StreamPins {
	if pins := s.pins.Load(); pins != nil {
		return *pins
	}
	return StreamPins{}
}

func (p StreamPins) deployment() int64 {
	if p.Deployment == nil {
		return 0
	}
	return p.Deployment()
}

// exportPin writes the app a NULL pin meant onto an exported row, when it is known.
func (s *Store) exportPin(table string, row json.RawMessage) (json.RawMessage, error) {
	if !pinnedTables[table] {
		return row, nil
	}
	deployment := s.streamPins().deployment()
	if deployment == 0 {
		return row, nil
	}
	fields, pin, err := readPin(row)
	if err != nil || pin != nil {
		return row, err
	}
	return writePin(fields, &deployment)
}

// importPin decides what an imported row is pinned to. It keeps the pin only when it names
// the app the customer acts in here, or for a customer in the deployment's own app, that
// app. A row from a deployment that never pinned anything arrives unpinned on a deployment
// that does not pin either, which is how every move worked before pins existed.
func (s *Store) importPin(ctx context.Context, customerID, table string, row json.RawMessage) (json.RawMessage, error) {
	if !pinnedTables[table] {
		return row, nil
	}
	fields, pin, err := readPin(row)
	if err != nil {
		return nil, err
	}
	pins := s.streamPins()
	var here int64
	if pins.For != nil {
		if here, err = pins.For(ctx, customerID); err != nil {
			return nil, fmt.Errorf("store: which Stream app %s acts in: %w", customerID, err)
		}
	}
	deployment := pins.deployment()

	var kept *int64
	switch {
	case pin == nil && here == 0:
		kept = nil
	case pin != nil && *pin != 0 && *pin == here:
		kept = pin
	case pin != nil && here == 0 && deployment != 0 && *pin == deployment:
		kept = pin
	default:
		foreign := ForeignStreamApp
		kept = &foreign
	}
	return writePin(fields, kept)
}

func readPin(row json.RawMessage) (map[string]json.RawMessage, *int64, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(row, &fields); err != nil {
		return nil, nil, fmt.Errorf("store: read a row's pin: %w", err)
	}
	raw, ok := fields[pinColumn]
	if !ok || string(raw) == "null" {
		return fields, nil, nil
	}
	var pin int64
	if err := json.Unmarshal(raw, &pin); err != nil {
		return nil, nil, fmt.Errorf("store: read a row's pin: %w", err)
	}
	return fields, &pin, nil
}

func writePin(fields map[string]json.RawMessage, pin *int64) (json.RawMessage, error) {
	encoded, err := json.Marshal(pin)
	if err != nil {
		return nil, err
	}
	fields[pinColumn] = encoded
	return json.Marshal(fields)
}

// pins is held atomically because the router sets it once at startup while suites share a
// store across goroutines.
type pinsHolder = atomic.Pointer[StreamPins]
