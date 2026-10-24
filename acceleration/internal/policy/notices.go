package policy

import (
	"context"
	"encoding/json"
	"strconv"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// noticeTimeout bounds writing one organization's warning and queueing its deliveries.
const noticeTimeout = 5 * time.Second

// The event types a warning is sent as.
const (
	EventBalanceLow       = "account.balance_low"
	EventBalanceExhausted = "account.balance_exhausted"
)

// notice is a warning a pass found due for one organization, taken under m.mu and written
// after it.
type notice struct {
	organization string
	tier         string
	kind         string
	granted      int64
	balance      int64
	threshold    *int64
}

// noticeDue is the warning the account's balance calls for now, under m.mu: balance_low once
// what it has to spend is at or below its threshold, balance_exhausted at or below nothing, as
// the answers' X-Stream-Balance-Warning says. A warning of a kind already written for what the
// organization has been granted is not due again: the next one waits for a top-up or a grant.
// Only an organization the usage service bills, with a standing read, is warned, and only one
// that was ever granted credit: one that never had any has not run out of it.
func (m *money) noticeDue(key string, acct *account) (notice, bool) {
	if !billed(key) || m.settings.Usage == nil || acct.read.IsZero() || acct.standing.GrantedMicros <= 0 {
		return notice{}, false
	}
	threshold := acct.standing.Limits.LowBalanceMicros
	if acct.warning != nil {
		threshold = acct.warning
	}
	available := acct.available(key)
	kind := ""
	switch stateOf(available, threshold) {
	case BalanceLow:
		kind = store.NoticeBalanceLow
	case BalanceExhausted:
		kind = store.NoticeBalanceExhausted
	default:
		return notice{}, false
	}
	granted := acct.standing.GrantedMicros
	if written, ok := acct.noticed[kind]; ok && written == granted {
		return notice{}, false
	}
	return notice{
		organization: key, tier: acct.standing.OrganizationTier, kind: kind,
		granted: granted, balance: available, threshold: threshold,
	}, true
}

// notify writes each warning due and queues it for the organization's stream destinations,
// off m.mu. The router that writes a warning first is the one that sends it; the others find
// it written and send nothing. One that could not be written is due again at the next pass.
func (m *money) notify(ctx context.Context, due []notice) {
	db := m.enforcer.config.DB()
	if len(due) == 0 || db == nil {
		return
	}
	for _, warning := range due {
		writing, cancel := context.WithTimeout(ctx, noticeTimeout)
		sent, queued, err := db.NoteBalance(writing, warning.row(), warning.delivery(), warning.body)
		cancel()
		if err != nil {
			m.failedRead("could not write that an organization's credit is running out", err,
				"organization", warning.organization, "kind", warning.kind)
			continue
		}
		m.mu.Lock()
		if acct := m.accounts[warning.organization]; acct != nil {
			if acct.noticed == nil {
				acct.noticed = map[string]int64{}
			}
			acct.noticed[warning.kind] = warning.granted
		}
		m.mu.Unlock()
		if !sent {
			continue
		}
		m.enforcer.logger.Info("warned an organization that its credit is running out",
			"organization", warning.organization, "kind", warning.kind, "balance_micros", warning.balance,
			"granted_micros", warning.granted, "destinations", queued)
		if queued > 0 && m.settings.Notified != nil {
			m.settings.Notified()
		}
	}
}

func (n notice) row() store.BalanceNotice {
	threshold := int64(0)
	if n.threshold != nil {
		threshold = *n.threshold
	}
	return store.BalanceNotice{
		OrganizationID: n.organization, Kind: n.kind, GrantedMicros: n.granted,
		BalanceMicros: n.balance, ThresholdMicros: threshold,
	}
}

// delivery is what each destination is sent beside the body: the webhook-id names the
// organization, the kind and what it had been granted, so a receiver that gets it twice can
// tell.
func (n notice) delivery() store.EventDelivery {
	return store.EventDelivery{
		ID:            "balance:" + n.organization + ":" + n.kind + ":" + strconv.FormatInt(n.granted, 10),
		Headers:       map[string]string{"Content-Type": "application/json"},
		NextAttemptAt: time.Now(),
	}
}

// balanceEvent is the body a warning is sent with, in the Standard Webhooks shape.
type balanceEvent struct {
	Type      string           `json:"type"`
	Timestamp time.Time        `json:"timestamp"`
	Data      balanceEventData `json:"data"`
}

type balanceEventData struct {
	OrganizationID  string `json:"organization_id"`
	AppID           string `json:"app_id"`
	BalanceMicros   int64  `json:"balance_micros"`
	ThresholdMicros *int64 `json:"threshold_micros,omitempty"`
	Tier            string `json:"tier,omitempty"`
}

// body is the event as the app's destinations are sent it.
func (n notice) body(appID string) ([]byte, error) {
	kind := EventBalanceLow
	if n.kind == store.NoticeBalanceExhausted {
		kind = EventBalanceExhausted
	}
	return json.Marshal(balanceEvent{Type: kind, Timestamp: time.Now().UTC().Truncate(time.Second), Data: balanceEventData{
		OrganizationID: n.organization, AppID: appID, BalanceMicros: n.balance, ThresholdMicros: n.threshold, Tier: n.tier,
	}})
}
