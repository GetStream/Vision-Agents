package policy

import (
	"cmp"
	"slices"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// grantState is a live grant and what the request rows naming it had paid for at the last read.
type grantState struct {
	grant store.LimitGrant
	used  int64
}

// remaining is what the grant has left: its amount, less what it paid for, less what this
// process holds against it.
func (g grantState) remaining(held map[int64]int64) int64 {
	return g.grant.AmountMicros - g.used - held[g.grant.ID]
}

// grantFor is the grant a subject's work is admitted on once one of its limits has no room: of
// its grants live at now with something left, the one that expires first, a grant with no
// expiry last, then the oldest. False when none has.
func grantFor(grants []grantState, held map[int64]int64, ref usage.SubjectRef, now time.Time) (grantState, bool) {
	var found []grantState
	for _, grant := range grants {
		if grant.grant.Ref() == ref && grant.grant.LiveAt(now) && grant.remaining(held) > 0 {
			found = append(found, grant)
		}
	}
	if len(found) == 0 {
		return grantState{}, false
	}
	slices.SortFunc(found, func(a, b grantState) int {
		switch {
		case a.grant.ExpiresAt == nil && b.grant.ExpiresAt != nil:
			return 1
		case a.grant.ExpiresAt != nil && b.grant.ExpiresAt == nil:
			return -1
		case a.grant.ExpiresAt != nil && !a.grant.ExpiresAt.Equal(*b.grant.ExpiresAt):
			return a.grant.ExpiresAt.Compare(*b.grant.ExpiresAt)
		}
		return cmp.Compare(a.grant.ID, b.grant.ID)
	})
	return found[0], true
}

// keepGrants replaces an account's grants with what was read, under m.mu.
func (a *account) keepGrants(grants []store.LimitGrant, from time.Time) {
	a.grants = make([]grantState, len(grants))
	for i, grant := range grants {
		a.grants[i] = grantState{grant: grant, used: grant.UsedMicros}
	}
	a.grantsFrom = from
}

// countGrants adds sign times the hold to what it holds against each of its grants.
func (a *account) countGrants(h *hold, sign int64) {
	if len(h.grants) == 0 {
		return
	}
	if a.grantHeld == nil {
		a.grantHeld = map[int64]int64{}
	}
	for _, id := range h.grants {
		if a.grantHeld[id] += sign * h.micros; a.grantHeld[id] == 0 {
			delete(a.grantHeld, id)
		}
	}
}

func (a *account) letGoGrants(h *hold) {
	a.countGrants(h, -1)
	h.granted = false
	a.reopen(h)
}
