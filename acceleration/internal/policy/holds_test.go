package policy

import (
	"log/slog"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

const heldOrg = "org-1"

// holding is an enforcer with no loops and an organization billed by a usage service, whose
// holds a test moves by hand.
func holding(t *testing.T) (*Enforcer, *account) {
	t.Helper()
	e := &Enforcer{logger: slog.New(slog.DiscardHandler)}
	m := &e.money
	m.enforcer, m.settings = e, Settings{Usage: &usage.Client{}, PendingTTL: time.Hour, Now: time.Now}
	m.accounts, m.holdOf = map[string]*account{}, map[uuid.UUID]string{}
	acct := &account{holds: map[uuid.UUID]*hold{}}
	m.accounts[heldOrg] = acct
	t.Cleanup(m.forgetFolded)
	return e, acct
}

func spend(e *Enforcer, micros int64) uuid.UUID {
	id := uuid.Must(uuid.NewV7())
	spendAs(e, id, micros)
	return id
}

func spendAs(e *Enforcer, id uuid.UUID, micros int64) {
	e.Spent(routing.Owner{CustomerID: "app-1", Attribution: routing.Attribution{OrganizationID: heldOrg}}, id, micros)
}

func ack(e *Enforcer, ids ...uuid.UUID) {
	acked := make([]routing.Acked, len(ids))
	for i, id := range ids {
		acked[i] = routing.Acked{UsageID: id, OrganizationID: heldOrg}
	}
	e.Acked(acked)
}

func TestARowAckedIntoAnAckedFoldWaitsForAReadAfterItsOwnAck(t *testing.T) {
	e, acct := holding(t)
	first, second := spend(e, 1), spend(e, 1)
	ack(e, first)
	require.True(t, e.money.fold(acct, first))
	require.True(t, e.money.fold(acct, second))
	time.Sleep(time.Millisecond)
	// A standing asked for after the first ack, and before the second.
	acct.standingFrom = time.Now()
	time.Sleep(time.Millisecond)

	ack(e, second)
	e.money.release(heldOrg, acct, time.Now())

	require.EqualValues(t, 2, acct.held, "1 micro is due, but not 2: the standing was asked for before the second ack")
	acct.standingFrom = time.Now()
	e.money.release(heldOrg, acct, time.Now())
	require.Zero(t, acct.held, "both go with a standing asked for after both acks")
}

// limitsHeldOf is what the account holds against the customer's limits.
func limitsHeldOf(acct *account) int64 {
	var sum int64
	for _, held := range acct.holds {
		if held.limits {
			sum += held.micros
		}
	}
	return sum
}

func TestALimitsHoldOutlastsAUsageServiceThatAcksNothing(t *testing.T) {
	e, acct := holding(t)
	e.money.settings.PendingTTL, e.money.settings.Fresh = 20*time.Millisecond, time.Millisecond
	acct.limited = true
	spend(e, 5)
	time.Sleep(40 * time.Millisecond)

	// Reads of the counters land, and none can have the row, which nothing acked.
	acct.spentFrom = time.Now()
	e.money.release(heldOrg, acct, time.Now())
	require.EqualValues(t, 5, limitsHeldOf(acct), "kept, however long, while no row of the account is acked")
	time.Sleep(40 * time.Millisecond)
	acct.spentFrom = time.Now()
	e.money.release(heldOrg, acct, time.Now())
	require.EqualValues(t, 5, limitsHeldOf(acct))

	ack(e, spend(e, 1))
	acked := time.Now()
	time.Sleep(time.Millisecond)
	// A read of the service itself, after the ack, counts the acked row.
	acct.spentFrom, acct.throughFrom = time.Now(), time.Now()
	e.money.release(heldOrg, acct, time.Now())
	require.EqualValues(t, 5, limitsHeldOf(acct), "the acks are back, and its timer starts again from then")
	for limitsHeldOf(acct) > 0 && time.Since(acked) < time.Second {
		time.Sleep(2 * time.Millisecond)
		acct.spentFrom = time.Now()
		e.money.release(heldOrg, acct, time.Now())
	}
	require.Zero(t, limitsHeldOf(acct), "a row the service never acked, among ones it does, expires")
	require.GreaterOrEqual(t, time.Since(acked), 15*time.Millisecond, "about PendingTTL after the acks came back")
}

func TestAnAcklessHoldIsKeptAtMostSoLong(t *testing.T) {
	e, acct := holding(t)
	e.money.settings.PendingTTL, e.money.settings.Fresh = 10*time.Millisecond, time.Millisecond
	acct.limited = true
	spend(e, 5)

	time.Sleep(30 * time.Millisecond)
	acct.spentFrom = time.Now()
	e.money.release(heldOrg, acct, time.Now())
	require.EqualValues(t, 5, limitsHeldOf(acct), "kept past PendingTTL while nothing is acked")
	time.Sleep(acklessAges * 10 * time.Millisecond)
	acct.spentFrom = time.Now()
	e.money.release(heldOrg, acct, time.Now())
	require.Zero(t, limitsHeldOf(acct), "let go past acklessAges PendingTTLs, acks or not")
}

func TestAnIdleAccountGivesItsFoldedIDsBackToTheProcess(t *testing.T) {
	e, acct := holding(t)
	e.money.settings.Idle = time.Millisecond
	before := foldedInProcess.Load()
	first, second := spend(e, 1), spend(e, 1)
	require.True(t, e.money.fold(acct, first))
	require.True(t, e.money.fold(acct, second))
	require.Equal(t, before+2, foldedInProcess.Load())
	// The folds go while their ids are still remembered, as a fold emptied by an ack does.
	for id, held := range acct.holds {
		acct.count(held, -1)
		delete(acct.holds, id)
	}
	acct.used = time.Now().Add(-time.Hour)

	e.money.inUse(time.Now())

	require.NotContains(t, e.money.accounts, heldOrg, "evicted, holding nothing")
	require.Equal(t, before, foldedInProcess.Load(), "its folded ids no longer count against the process")
	require.NotContains(t, e.money.holdOf, first)
	require.NotContains(t, e.money.holdOf, second)
}

func TestPastWhatTheProcessRemembersAFoldedIDIsHeldUntilItsFoldsTimer(t *testing.T) {
	e, acct := holding(t)
	before := foldedInProcess.Load()
	foldedInProcess.Store(maxFoldedInProcess)
	t.Cleanup(func() { foldedInProcess.Add(before - maxFoldedInProcess) })
	forgotten := testutil.ToFloat64(forgottenFolded)
	id := spend(e, 5)

	require.False(t, e.money.fold(acct, id))

	require.NotContains(t, acct.folded, id, "the process remembers no more")
	require.Equal(t, 1.0, testutil.ToFloat64(forgottenFolded)-forgotten, "counted")
	ack(e, id)
	acct.standingFrom = time.Now().Add(time.Second)
	e.money.release(heldOrg, acct, time.Now())
	require.EqualValues(t, 5, acct.held, "an ack it cannot find lets nothing go")
	spendAs(e, id, 5)
	require.EqualValues(t, 10, acct.held, "recorded again it is held twice: more refused, never less")
}
