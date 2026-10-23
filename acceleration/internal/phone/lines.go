package phone

import (
	"cmp"
	"context"
	"fmt"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// lineCallTemplate names the call a leg arriving on a number's rule lands in: the one the
// vendor named in the call id header, or, for a caller with none, a fresh call of their own.
// Stream refuses the leg when the template is written as {{#if}}…{{else}}{{uuid}}{{/if}}.
const lineCallTemplate = "{{x_stream_call_id}}{{#unless x_stream_call_id}}{{uuid}}{{/unless}}"

// trunkCapacity is how many numbers Stream lets one trunk have.
const trunkCapacity = 50

// TrunkLine is the trunk and rule a number's calls reach Stream through.
type TrunkLine struct {
	TrunkID string
	RouteID string
	// Bridge is the trunk's address for the number, with its password opened.
	Bridge Bridge
}

// line puts a held number on a trunk it shares with the customer's other numbers of the
// vendor in the app, with a rule of its own, and answers it. A number already on one is
// answered as it is. A failure leaves Stream as it found it and stores nothing.
func (s *Service) line(ctx context.Context, stream *Stream, pin int64, held store.PhoneNumber) (TrunkLine, error) {
	if s.sealer == nil {
		return TrunkLine{}, stack.Wrap(ErrNoSealer)
	}
	group := store.TrunkGroup{CustomerID: held.CustomerID, StreamAppPK: pin, Vendor: held.Vendor}

	var line TrunkLine
	var joined *joining
	err := s.store.InTrunkGroup(ctx, group, func(ctx context.Context, tx *store.TrunkGroupTx) error {
		trunks, err := tx.Trunks()
		if err != nil {
			return err
		}
		if held.StreamTrunkID != "" && held.StreamAppPK == pin {
			if i := slices.IndexFunc(trunks, func(t store.StreamTrunk) bool { return t.ID == held.StreamTrunkID }); i >= 0 {
				line, err = s.lineOn(trunks[i], held.E164, held.StreamRouteID)
				return err
			}
			line, err = s.adopt(ctx, stream, tx, held)
			return err
		}
		joined = &joining{}
		line, err = s.join(ctx, stream, tx, trunks, held, joined)
		return err
	})
	if err != nil {
		if joined != nil {
			s.leave(ctx, stream, group, joined)
		}
		return TrunkLine{}, err
	}
	return line, nil
}

// joining is what join changed at Stream, for leave to undo.
type joining struct {
	trunkID string
	routeID string
}

// join puts a number on the group's emptiest trunk with room, or a new one, and makes it a
// rule of its own.
func (s *Service) join(ctx context.Context, stream *Stream, tx *store.TrunkGroupTx, trunks []store.StreamTrunk, held store.PhoneNumber, joined *joining) (TrunkLine, error) {
	allowedIPs, err := s.allowlistFor(held.Vendor)
	if err != nil {
		return TrunkLine{}, err
	}
	name := "phone-" + held.Vendor + "-" + held.CustomerID

	var trunk store.StreamTrunk
	var password string
	roomiest := -1
	for i, candidate := range trunks {
		if len(candidate.Numbers) < trunkCapacity && (roomiest < 0 || len(candidate.Numbers) < len(trunks[roomiest].Numbers)) {
			roomiest = i
		}
	}
	if roomiest >= 0 {
		trunk = trunks[roomiest]
		if password, err = s.trunkPassword(trunk); err != nil {
			return TrunkLine{}, err
		}
		trunk.Numbers = append(slices.Clone(trunk.Numbers), held.E164)
		if err := stream.UpdateTrunk(ctx, trunk.ID, Trunk{Name: name, Numbers: trunk.Numbers, AllowedIPs: allowedIPs}); err != nil {
			return TrunkLine{}, err
		}
		joined.trunkID = trunk.ID
	} else {
		id, bridge, err := stream.CreateTrunk(ctx, Trunk{Name: name, Numbers: []string{held.E164}, AllowedIPs: allowedIPs})
		if err != nil {
			return TrunkLine{}, err
		}
		joined.trunkID = id
		sealed, err := s.sealer.SealWithAAD(bridge.Password, trunkAAD(held.CustomerID, id))
		if err != nil {
			return TrunkLine{}, stack.Wrap(err)
		}
		password = bridge.Password
		trunk = store.StreamTrunk{
			ID: id, Numbers: []string{held.E164}, BridgeURI: bridge.URI, BridgeUsername: bridge.Username,
			BridgePasswordSealed: sealed, BridgePasswordKEKVersion: s.sealer.CurrentVersion(),
		}
	}

	routeID, err := stream.CreateRoute(ctx, Route{
		Name:          "phone-" + held.E164,
		TrunkIDs:      []string{trunk.ID},
		CalledNumbers: []string{held.E164},
		CallID:        lineCallTemplate,
		CallType:      defaultCallType,
		Custom:        map[string]string{NumberKey: held.E164},
	})
	if err != nil {
		return TrunkLine{}, err
	}
	joined.routeID = routeID

	bridge, err := Bridge{URI: trunk.BridgeURI, Username: trunk.BridgeUsername, Password: password}.WithNumber(held.E164)
	if err != nil {
		return TrunkLine{}, err
	}
	// The number keeps a copy sealed to itself, which is what InboundBridge opens.
	sealed, err := s.sealer.SealWithAAD(password, bridgeAAD(held.CustomerID, held.E164))
	if err != nil {
		return TrunkLine{}, stack.Wrap(err)
	}
	if err := tx.SaveTrunk(trunk); err != nil {
		return TrunkLine{}, err
	}
	err = tx.SetLine(held.E164, trunk.ID, routeID, store.NumberBridge{
		URI: bridge.URI, Username: bridge.Username, PasswordSealed: sealed, KEKVersion: s.sealer.CurrentVersion(),
	})
	if err != nil {
		return TrunkLine{}, err
	}
	return TrunkLine{TrunkID: trunk.ID, RouteID: routeID, Bridge: bridge}, nil
}

// leave undoes a join that failed. Its transaction rolled back, so the stored trunk, read
// again under the group's lock, is what Stream should have: without the number, or not
// there at all for a trunk the join made. Best-effort: a failure is logged, never returned,
// so it cannot mask the join's own.
func (s *Service) leave(ctx context.Context, stream *Stream, group store.TrunkGroup, joined *joining) {
	if joined.routeID != "" {
		if err := stream.DeleteRoute(ctx, joined.routeID); err != nil {
			s.logger.Error("could not delete a routing rule", "why", "rolling back a failed line",
				"route", joined.routeID, "error", err)
		}
	}
	if joined.trunkID == "" {
		return
	}
	err := s.store.InTrunkGroup(ctx, group, func(ctx context.Context, tx *store.TrunkGroupTx) error {
		trunks, err := tx.Trunks()
		if err != nil {
			return err
		}
		i := slices.IndexFunc(trunks, func(t store.StreamTrunk) bool { return t.ID == joined.trunkID })
		if i < 0 {
			return stream.DeleteTrunk(ctx, joined.trunkID)
		}
		allowedIPs, err := s.allowlistFor(group.Vendor)
		if err != nil {
			return err
		}
		return stream.UpdateTrunk(ctx, joined.trunkID, Trunk{
			Name: "phone-" + group.Vendor + "-" + group.CustomerID, Numbers: trunks[i].Numbers, AllowedIPs: allowedIPs,
		})
	})
	if err != nil {
		s.logger.Error("could not take a number back off a trunk", "why", "rolling back a failed line",
			"trunk", joined.trunkID, "error", err)
	}
}

// adopt takes a number attached on a trunk of its own, before trunks were shared, into the
// group as it is: its trunk becomes one the customer's next numbers can join, and its rule
// reads the call id header.
func (s *Service) adopt(ctx context.Context, stream *Stream, tx *store.TrunkGroupTx, held store.PhoneNumber) (TrunkLine, error) {
	if len(held.BridgePasswordSealed) == 0 {
		return TrunkLine{}, stack.Wrap(fmt.Errorf("phone: adopt the trunk of %s: %w", held.E164, store.ErrNoStoredBridge))
	}
	password, err := s.sealer.OpenWithAADVersion(held.BridgePasswordSealed, bridgeAAD(held.CustomerID, held.E164), held.BridgePasswordKEKVersion)
	if err != nil {
		return TrunkLine{}, stack.Wrap(fmt.Errorf("phone: open the trunk password of %s: %w", held.E164, err))
	}
	sealed, err := s.sealer.SealWithAAD(password, trunkAAD(held.CustomerID, held.StreamTrunkID))
	if err != nil {
		return TrunkLine{}, stack.Wrap(err)
	}

	// An update resends every field, so the rule is given again all it was made with.
	err = stream.UpdateRoute(ctx, held.StreamRouteID, Route{
		Name:          "phone-" + held.E164,
		TrunkIDs:      []string{held.StreamTrunkID},
		CalledNumbers: []string{held.E164},
		CallID:        lineCallTemplate,
		CallType:      cmp.Or(held.StreamCallType, defaultCallType),
		Custom:        map[string]string{NumberKey: held.E164},
	})
	if err != nil {
		return TrunkLine{}, err
	}
	err = tx.SaveTrunk(store.StreamTrunk{
		ID: held.StreamTrunkID, Numbers: []string{held.E164}, BridgeURI: held.BridgeURI, BridgeUsername: held.BridgeUsername,
		BridgePasswordSealed: sealed, BridgePasswordKEKVersion: s.sealer.CurrentVersion(),
	})
	if err != nil {
		return TrunkLine{}, err
	}
	bridge, err := Bridge{URI: held.BridgeURI, Username: held.BridgeUsername, Password: password}.WithNumber(held.E164)
	if err != nil {
		return TrunkLine{}, err
	}
	return TrunkLine{TrunkID: held.StreamTrunkID, RouteID: held.StreamRouteID, Bridge: bridge}, nil
}

// lineOn is the line of a number already on one of the group's trunks.
func (s *Service) lineOn(trunk store.StreamTrunk, e164, routeID string) (TrunkLine, error) {
	password, err := s.trunkPassword(trunk)
	if err != nil {
		return TrunkLine{}, err
	}
	bridge, err := Bridge{URI: trunk.BridgeURI, Username: trunk.BridgeUsername, Password: password}.WithNumber(e164)
	if err != nil {
		return TrunkLine{}, err
	}
	return TrunkLine{TrunkID: trunk.ID, RouteID: routeID, Bridge: bridge}, nil
}

// trunkPassword opens a shared trunk's password.
func (s *Service) trunkPassword(trunk store.StreamTrunk) (string, error) {
	password, err := s.sealer.OpenWithAADVersion(trunk.BridgePasswordSealed, trunkAAD(trunk.CustomerID, trunk.ID), trunk.BridgePasswordKEKVersion)
	if err != nil {
		return "", stack.Wrap(fmt.Errorf("phone: open the password of trunk %s: %w", trunk.ID, err))
	}
	return password, nil
}

// trunkAAD binds a shared trunk's sealed password to the customer and the trunk, so the
// ciphertext does not open on another row.
func trunkAAD(customerID, trunkID string) []byte {
	return []byte("stream_trunks/" + customerID + "/" + trunkID)
}
