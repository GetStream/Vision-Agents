package outcome

import (
	"context"
	"errors"
	"log/slog"
	"slices"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ErrUnknownCall is a call no row was kept for, or whose token was swept.
var ErrUnknownCall = errors.New("outcome: no such call")

// Keeper keeps placed calls in the store and runs every event about one through decide. It
// counts each call once, inbound ones too, in router_phone_call_outcomes_total.
type Keeper struct {
	store *store.Store
	now   func() time.Time
	// vendors are the vendor names a call is counted under; any other is counted as other, so
	// the metric's labels stay a fixed set.
	vendors []string
	logger  *slog.Logger
}

// NewKeeper keeps calls in st, times what it is told without a time by now, and counts calls by
// the vendors named.
func NewKeeper(st *store.Store, now func() time.Time, vendors []string, logger *slog.Logger) *Keeper {
	return &Keeper{store: st, now: now, vendors: vendors, logger: logger}
}

// Placed keeps a call about to be dialled and returns its token, for the vendor's status URL.
func (k *Keeper) Placed(ctx context.Context, call Call) (string, error) {
	row := store.PhoneCall{
		Token:        uuid.NewString(),
		CustomerID:   call.CustomerID,
		StreamAppPK:  call.StreamAppPK,
		CallType:     call.CallType,
		CallID:       call.CallID,
		Vendor:       call.Vendor,
		LocalNumber:  call.Local,
		RemoteNumber: call.Remote,
		Reports:      call.Reports,
		PlacedAt:     k.now(),
	}
	if err := k.store.InsertPhoneCall(ctx, &row); err != nil {
		return "", err
	}
	return row.Token, nil
}

// Dialed keeps the vendor's own id for the call once it took the dial.
func (k *Keeper) Dialed(ctx context.Context, token, vendorCallID string) error {
	_, err := k.store.ChangePhoneCall(ctx, store.PhoneCallWhere{Token: token}, func(row *store.PhoneCall) bool {
		row.VendorCallID = vendorCallID
		return true
	})
	return asUnknown(err)
}

// Failed forgets a call that was never dialled, or that the vendor refused to dial: it never
// rang, so it has no outcome.
func (k *Keeper) Failed(ctx context.Context, token string) error {
	return k.store.DeletePhoneCall(ctx, token)
}

// Refused forgets a call the vendor was asked to dial and refused, as Failed does, and counts
// it as refused. It has no row, so nothing else counts it.
func (k *Keeper) Refused(ctx context.Context, token, vendor string) error {
	callOutcomes.WithLabelValues(k.vendorLabel(vendor), outbound, refused).Inc()
	return k.Failed(ctx, token)
}

// Reported takes a vendor's report and counts what it says of the carrier.
func (k *Keeper) Reported(ctx context.Context, token string, report Report) (Call, error) {
	e := event{kind: reportedEvent, reported: report.Outcome, word: report.Word, at: k.now()}
	before, call, err := k.decide(ctx, store.PhoneCallWhere{Token: token}, e)
	if err != nil {
		return Call{}, asUnknown(err)
	}
	k.countReport(before, report, e.at)
	return call, nil
}

// Canceled is the router hanging up the call before anyone answered. It is decided as a
// vendor's final report is, so whichever of the two came first is kept.
func (k *Keeper) Canceled(ctx context.Context, key Key) (Call, error) {
	_, call, err := k.decide(ctx, where(key), event{kind: reportedEvent, reported: Canceled, at: k.now()})
	if err != nil {
		return Call{}, asUnknown(err)
	}
	return call, nil
}

// Answered is the call's phone joining the Stream call at at. A leg with no row, a transfer's
// or an inbound call's, is nothing to keep.
func (k *Keeper) Answered(ctx context.Context, key Key, at time.Time) error {
	_, _, err := k.decide(ctx, where(key), event{kind: answeredEvent, at: at})
	return skipRowless(err)
}

// Ended is the call's leg ending at at. A leg with no row is nothing to keep, as in Answered.
// A placed call is counted the first time its leg ends, and an inbound one each time, since the
// store ends a leg once: failed when the router refused it, else completed or dropped. A
// transfer's leg is not counted.
func (k *Keeper) Ended(ctx context.Context, key Key, at time.Time, leg Leg) error {
	before, call, err := k.decide(ctx, where(key), event{kind: endedEvent, at: at})
	switch {
	case err != nil && !errors.Is(err, store.ErrNoPhoneCall):
		return err
	case leg.Inbound && leg.Refused:
		k.count(ctx, key, leg.Vendor, inbound, Failed, nil, at)
	case leg.Inbound:
		k.count(ctx, key, leg.Vendor, inbound, Completed, leg.AnsweredAt, at)
	case err == nil && before.EndedAt == nil:
		k.count(ctx, key, call.Vendor, outbound, call.Outcome, call.AnsweredAt, at)
	}
	return nil
}

// ByToken reads the call a token was made for.
func (k *Keeper) ByToken(ctx context.Context, token string) (Call, error) {
	row, err := k.store.PhoneCallByToken(ctx, token)
	if err != nil {
		return Call{}, asUnknown(err)
	}
	return callOf(row), nil
}

// Read reads the newest call placed into the customer's Stream call callID.
func (k *Keeper) Read(ctx context.Context, customerID, callID string) (Call, error) {
	calls, err := k.ReadMany(ctx, customerID, []string{callID})
	if err != nil {
		return Call{}, err
	}
	call, ok := calls[callID]
	if !ok {
		return Call{}, ErrUnknownCall
	}
	return call, nil
}

// ReadMany reads the newest call for each of callIDs, by call id. One with no call is left out.
func (k *Keeper) ReadMany(ctx context.Context, customerID string, callIDs []string) (map[string]Call, error) {
	rows, err := k.store.PhoneCalls(ctx, customerID, callIDs)
	if err != nil {
		return nil, err
	}
	calls := make(map[string]Call, len(rows))
	for _, row := range rows {
		calls[row.CallID] = callOf(row)
	}
	return calls, nil
}

// Sweep clears the tokens of calls placed before placedBefore, so a vendor's late post for one
// is refused, and keeps their outcomes.
func (k *Keeper) Sweep(ctx context.Context, placedBefore time.Time) (int64, error) {
	return k.store.ClearPhoneCallTokens(ctx, placedBefore)
}

// decide runs e on the call where finds, under the row's lock, so events arriving at once each
// see what the others decided. before is the call as e found it: e is what first ended the
// call when before has no end, which one event of a call is, whatever order they come in.
func (k *Keeper) decide(ctx context.Context, where store.PhoneCallWhere, e event) (before, call Call, err error) {
	row, err := k.store.ChangePhoneCall(ctx, where, func(row *store.PhoneCall) bool {
		before = callOf(*row)
		after := decide(before, e)
		if after == before {
			return false
		}
		row.AnsweredAt, row.EndedAt, row.DecidedAt = after.AnsweredAt, after.EndedAt, after.DecidedAt
		row.Outcome, row.VendorStatus = string(after.Outcome), after.VendorStatus
		return true
	})
	if err != nil {
		return Call{}, Call{}, err
	}
	return before, callOf(row), nil
}

// count counts a call that ended at ended under outcome, or as dropped when it was answered
// (at answered, nil if never) and its agent left it on the line, and as short when it was
// answered for less than shortCall. The router's session of a call has the call's id. A
// session that cannot be read leaves the call counted under outcome, and is logged.
func (k *Keeper) count(ctx context.Context, key Key, vendor, direction string, outcome Outcome, answered *time.Time,
	ended time.Time) {
	label := string(outcome)
	vendor = k.vendorLabel(vendor)
	if answered != nil {
		session, err := k.store.StoredSession(ctx, key.CustomerID, key.CallID)
		switch {
		case err == nil && leftOnTheLine(session, ended):
			label = dropped
		case err != nil && !errors.Is(err, store.ErrNoSession):
			k.logger.Warn("could not tell whether a phone call was dropped; it is counted as it is stored",
				"customer", key.CustomerID, "call", key.CallType+":"+key.CallID, "outcome", label, "error", err)
		}
	}
	callOutcomes.WithLabelValues(vendor, direction, label).Inc()
	if answered != nil && isShort(ended.Sub(*answered)) {
		shortCalls.WithLabelValues(vendor, direction).Inc()
	}
}

// vendorLabel is the vendor a call is counted under: one the keeper names, else other.
func (k *Keeper) vendorLabel(vendor string) string {
	if !slices.Contains(k.vendors, vendor) {
		return otherVendor
	}
	return vendor
}

func where(key Key) store.PhoneCallWhere {
	return store.PhoneCallWhere{
		CustomerID:   key.CustomerID,
		StreamAppPK:  key.StreamAppPK,
		CallType:     key.CallType,
		CallID:       key.CallID,
		LocalNumber:  key.Local,
		RemoteNumber: key.Remote,
	}
}

func callOf(row store.PhoneCall) Call {
	return Call{
		Key: Key{
			CustomerID:  row.CustomerID,
			StreamAppPK: row.StreamAppPK,
			CallType:    row.CallType,
			CallID:      row.CallID,
			Local:       row.LocalNumber,
			Remote:      row.RemoteNumber,
		},
		Vendor:       row.Vendor,
		Reports:      row.Reports,
		VendorCallID: row.VendorCallID,
		PlacedAt:     row.PlacedAt,
		AnsweredAt:   row.AnsweredAt,
		EndedAt:      row.EndedAt,
		Outcome:      Outcome(row.Outcome),
		VendorStatus: row.VendorStatus,
		DecidedAt:    row.DecidedAt,
	}
}

// asUnknown is err, with a call the store holds no row for as ErrUnknownCall.
func asUnknown(err error) error {
	if errors.Is(err, store.ErrNoPhoneCall) {
		return ErrUnknownCall
	}
	return err
}

// skipRowless is err, with a call the store holds no row for as no error.
func skipRowless(err error) error {
	if errors.Is(err, store.ErrNoPhoneCall) {
		return nil
	}
	return err
}
