package policy

import (
	"cmp"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// inForce is the settings in force for an organization and the tier they are of, as its
// standing last said, for checks made on the work's own path rather than here, such as a
// channel's (dlc.Gate). Nil with no organization, an app's own account, money not enforced,
// or no standing read yet: nothing is read for it, so a text or a call costs no read.
func (m *money) inForce(organization string) (*usage.Limits, string) {
	if organization == "" || !billed(organization) || m.off() {
		return nil, ""
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	acct := m.accounts[organization]
	if acct == nil || acct.read.IsZero() {
		return nil, ""
	}
	// A pass replaces the standing whole, never its maps in place, so this is safe to share.
	limits := acct.standing.Limits
	return &limits, cmp.Or(acct.standing.OrganizationTier, acct.standing.Tier.Name)
}
