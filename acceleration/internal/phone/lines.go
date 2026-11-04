package phone

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// lineCallTemplate names the call a leg arriving on a number's rule lands in: the one the
// vendor named in the call id header, or, for a caller with none, a fresh call of their own.
// Stream refuses the leg when the template is written as {{#if}}…{{else}}{{uuid}}{{/if}}.
const lineCallTemplate = "{{x_stream_call_id}}{{#unless x_stream_call_id}}{{uuid}}{{/unless}}"

// trunkCapacity is how many numbers Stream lets one trunk have.
const trunkCapacity = 50

// errLinedElsewhere is line's error for a number on a trunk in another app.
var errLinedElsewhere = errors.New("lined in another app")

// errRoutedElsewhere is line's error for a number whose stored rule or trunk Stream no longer
// has while another rule or trunk there still routes it, made by hand or by another router
// on the app. A second one of ours would take its calls from whoever made that one.
var errRoutedElsewhere = errors.New("its rule or trunk is gone at Stream, and Stream routes it through another one")

// errTrunkLost is adopt's error for a number whose own trunk Stream no longer has.
var errTrunkLost = errors.New("its trunk is gone at Stream")

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
	err := s.inTrunkGroup(ctx, group, func(ctx context.Context, tx *store.TrunkGroupTx) error {
		// Read again under the lock: a concurrent line() for the same number may have just
		// put it on a trunk, and joining again would list it twice.
		held, err := tx.Number(held.E164)
		if err != nil {
			return err
		}
		// Lined in another app since the caller read it: joining here would point its row
		// away from the trunk the vendor still sends its callers to.
		if held.StreamTrunkID != "" && held.StreamAppPK != pin {
			return stack.Wrap(fmt.Errorf("phone: line %s: %w", held.E164, errLinedElsewhere))
		}
		trunks, err := tx.Trunks()
		if err != nil {
			return err
		}
		if held.StreamTrunkID != "" {
			if i := slices.IndexFunc(trunks, func(t store.StreamTrunk) bool { return t.ID == held.StreamTrunkID }); i >= 0 {
				line, err = s.lineOn(trunks[i], held.E164, held.StreamRouteID)
				return err
			}
		}
		joined = &joining{}
		if held.StreamTrunkID != "" {
			line, err = s.adopt(ctx, stream, tx, held, joined)
			if !errors.Is(err, errTrunkLost) {
				return err
			}
			// Nothing at Stream answers the number any more, so it joins as a new one would.
		}
		line, err = s.join(ctx, stream, tx, trunks, held, joined)
		return err
	})
	if err != nil {
		if joined != nil {
			// The join may have failed because ctx was cancelled; the undo still has to run.
			s.leave(context.WithoutCancel(ctx), stream, group, joined)
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
	name := sharedTrunkName(held.Vendor, held.CustomerID)

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
		err := stream.UpdateTrunk(ctx, trunk.ID, Trunk{Name: name, Numbers: trunk.Numbers, AllowedIPs: allowedIPs})
		switch {
		case notFound(err):
			if err := s.forgetLostTrunk(ctx, stream, tx, held, trunk.ID); err != nil {
				return TrunkLine{}, err
			}
			roomiest = -1
		case err != nil:
			return TrunkLine{}, err
		default:
			joined.trunkID = trunk.ID
		}
	}
	if roomiest < 0 {
		id, bridge, err := stream.CreateTrunk(ctx, Trunk{Name: name, Numbers: []string{held.E164}, AllowedIPs: allowedIPs})
		if err != nil {
			return TrunkLine{}, err
		}
		joined.trunkID = id
		sealed, err := s.sealer.SealWithAAD(bridge.Password, trunkAAD(held.CustomerID, id))
		if err != nil {
			return TrunkLine{}, stack.Wrap(fmt.Errorf("phone: seal the password of trunk %s: %w", id, err))
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
		return TrunkLine{}, stack.Wrap(fmt.Errorf("phone: seal the trunk password of %s: %w", held.E164, err))
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

// forgetLostTrunk drops the row of a shared trunk Stream no longer has, so the number
// joining gets a new trunk and each number left on the row, no longer finding it, lines again
// on its next call. The number must not be routed at Stream some other way.
func (s *Service) forgetLostTrunk(ctx context.Context, stream *Stream, tx *store.TrunkGroupTx, held store.PhoneNumber, trunkID string) error {
	rules, trunks, err := stream.answering(ctx, held.E164)
	if err != nil {
		return err
	}
	if len(rules) > 0 || len(trunks) > 0 {
		return s.routedElsewhere(held, rules, trunks)
	}
	return tx.DeleteTrunk(trunkID)
}

// routedElsewhere is errRoutedElsewhere for the number, with what Stream routes it through
// logged: the answer to the client does not name another's rule or trunk.
func (s *Service) routedElsewhere(held store.PhoneNumber, rules, trunks []string) error {
	s.logger.Warn("a number's line is gone at Stream and another routes it", "number", held.E164,
		"stored_trunk", held.StreamTrunkID, "stored_route", held.StreamRouteID, "rules", rules, "trunks", trunks)
	return stack.Wrap(fmt.Errorf("phone: line %s: %w", held.E164, errRoutedElsewhere))
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
	err := s.inTrunkGroup(ctx, group, func(ctx context.Context, tx *store.TrunkGroupTx) error {
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
			Name: sharedTrunkName(group.Vendor, group.CustomerID), Numbers: trunks[i].Numbers, AllowedIPs: allowedIPs,
		})
	})
	if err != nil {
		s.logger.Error("could not take a number back off a trunk", "why", "rolling back a failed line",
			"trunk", joined.trunkID, "error", err)
	}
}

// adopt takes a number attached on a trunk of its own, before trunks were shared, into the
// group as it is: its trunk becomes one the customer's next numbers can join, and its rule
// reads the call id header. A rule Stream no longer has is made again; a trunk it no longer
// has is errTrunkLost.
func (s *Service) adopt(ctx context.Context, stream *Stream, tx *store.TrunkGroupTx, held store.PhoneNumber, joined *joining) (TrunkLine, error) {
	if len(held.BridgePasswordSealed) == 0 {
		return TrunkLine{}, stack.Wrap(fmt.Errorf("phone: adopt the trunk of %s: %w", held.E164, store.ErrNoStoredBridge))
	}
	password, err := s.sealer.OpenWithAADVersion(held.BridgePasswordSealed, bridgeAAD(held.CustomerID, held.E164), held.BridgePasswordKEKVersion)
	if err != nil {
		return TrunkLine{}, stack.Wrap(fmt.Errorf("phone: open the trunk password of %s: %w", held.E164, err))
	}
	sealed, err := s.sealer.SealWithAAD(password, trunkAAD(held.CustomerID, held.StreamTrunkID))
	if err != nil {
		return TrunkLine{}, stack.Wrap(fmt.Errorf("phone: seal the password of trunk %s: %w", held.StreamTrunkID, err))
	}

	// An update resends every field, so the rule is given again all it was made with.
	route := Route{
		Name:          "phone-" + held.E164,
		TrunkIDs:      []string{held.StreamTrunkID},
		CalledNumbers: []string{held.E164},
		CallID:        lineCallTemplate,
		CallType:      cmp.Or(held.StreamCallType, defaultCallType),
		Custom:        map[string]string{NumberKey: held.E164},
	}
	routeID := held.StreamRouteID
	err = stream.UpdateRoute(ctx, routeID, route)
	if notFound(err) {
		routeID, err = s.routeAgain(ctx, stream, tx, held, route, joined)
	}
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
	return TrunkLine{TrunkID: held.StreamTrunkID, RouteID: routeID, Bridge: bridge}, nil
}

// routeAgain makes the rule of a number on a trunk of its own again, Stream having lost the
// stored one, and records it in place of that one.
func (s *Service) routeAgain(ctx context.Context, stream *Stream, tx *store.TrunkGroupTx, held store.PhoneNumber, route Route, joined *joining) (string, error) {
	rules, trunks, err := stream.answering(ctx, held.E164)
	if err != nil {
		return "", err
	}
	own := slices.Contains(trunks, held.StreamTrunkID)
	others := slices.DeleteFunc(trunks, func(id string) bool { return id == held.StreamTrunkID })
	if len(rules) > 0 || len(others) > 0 {
		return "", s.routedElsewhere(held, rules, others)
	}
	if !own {
		return "", stack.Wrap(fmt.Errorf("phone: line %s: %w", held.E164, errTrunkLost))
	}
	routeID, err := stream.CreateRoute(ctx, route)
	if err != nil {
		return "", err
	}
	joined.routeID = routeID
	err = tx.SetLine(held.E164, held.StreamTrunkID, routeID, store.NumberBridge{
		URI: held.BridgeURI, Username: held.BridgeUsername,
		PasswordSealed: held.BridgePasswordSealed, KEKVersion: held.BridgePasswordKEKVersion,
	})
	if err != nil {
		return "", err
	}
	return routeID, nil
}

// inTrunkGroup is store.InTrunkGroup behind a lock of this process's own. A caller waiting
// on the database's lock holds a pooled connection, so a burst of calls for one group, a
// campaign starting, would otherwise take every connection and leave none for the rest of
// the router.
func (s *Service) inTrunkGroup(ctx context.Context, group store.TrunkGroup, fn func(ctx context.Context, tx *store.TrunkGroupTx) error) error {
	unlock, err := s.groups.lock(ctx, group)
	if err != nil {
		return err
	}
	defer unlock()
	return s.store.InTrunkGroup(ctx, group, fn)
}

// groupLocks are the trunk groups this process is working on, one waiter at a time each.
type groupLocks struct {
	mu   sync.Mutex
	held map[store.TrunkGroup]*groupLock
}

type groupLock struct {
	// turn holds a value while someone has the group.
	turn  chan struct{}
	users int
}

// lock waits for the group, or for ctx to end.
func (g *groupLocks) lock(ctx context.Context, group store.TrunkGroup) (func(), error) {
	g.mu.Lock()
	if g.held == nil {
		g.held = map[store.TrunkGroup]*groupLock{}
	}
	l, ok := g.held[group]
	if !ok {
		l = &groupLock{turn: make(chan struct{}, 1)}
		g.held[group] = l
	}
	l.users++
	g.mu.Unlock()

	done := func() {
		g.mu.Lock()
		if l.users--; l.users == 0 {
			delete(g.held, group)
		}
		g.mu.Unlock()
	}
	select {
	case l.turn <- struct{}{}:
		return func() { <-l.turn; done() }, nil
	case <-ctx.Done():
		done()
		return nil, stack.Wrap(fmt.Errorf("phone: wait for trunk group: %w", ctx.Err()))
	}
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

// sharedTrunkName is the name at Stream of a customer's trunk for a vendor's numbers.
func sharedTrunkName(vendor, customerID string) string {
	return "phone-" + vendor + "-" + customerID
}

// trunkAAD binds a shared trunk's sealed password to the customer and the trunk, so the
// ciphertext does not open on another row.
func trunkAAD(customerID, trunkID string) []byte {
	return []byte("stream_trunks/" + customerID + "/" + trunkID)
}
