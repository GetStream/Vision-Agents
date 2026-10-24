package policy

// BalanceState is where an organization's available balance stands against the threshold it
// is warned at.
type BalanceState string

const (
	BalanceOK        BalanceState = "ok"
	BalanceLow       BalanceState = "low"
	BalanceExhausted BalanceState = "exhausted"
)

// Balance is what an organization has to spend now, as admission counts it, and what it is
// warned at.
type Balance struct {
	// AvailableMicros is the balance the usage service last answered less every hold, the other
	// routers' included: the figure admission refuses billable work at.
	AvailableMicros int64
	// ThresholdMicros is the balance at or below which it is low: its policy's
	// balance_warning_micros, else its tier's low_balance_micros. Nil with neither.
	ThresholdMicros *int64
	State           BalanceState
}

// stateOf is where available stands: exhausted at or below 0, low at or below the threshold.
func stateOf(available int64, threshold *int64) BalanceState {
	switch {
	case available <= 0:
		return BalanceExhausted
	case threshold != nil && available <= *threshold:
		return BalanceLow
	}
	return BalanceOK
}

// Below reports whether a is further down than b: low is below ok, exhausted below both.
func (a BalanceState) Below(b BalanceState) bool { return a.rank() > b.rank() }

func (a BalanceState) rank() int {
	switch a {
	case BalanceLow:
		return 1
	case BalanceExhausted:
		return 2
	}
	return 0
}

// Balance is the app's organization's balance, from memory: organizationID, or the one the
// app's decision names when it is empty. False until the organization's account has been read,
// for an app in no organization, with money off, and with no usage service, which bills nobody.
func (e *Enforcer) Balance(appID, organizationID string) (Balance, bool) {
	if e == nil || appID == "" || e.money.off() || e.settings.Usage == nil {
		return Balance{}, false
	}
	decided, _ := e.decided(appID)
	if organizationID == "" {
		organizationID = decided.organization
	}
	if organizationID == "" {
		return Balance{}, false
	}
	m := &e.money
	m.mu.Lock()
	acct := m.accounts[organizationID]
	if acct == nil || acct.read.IsZero() {
		m.mu.Unlock()
		return Balance{}, false
	}
	available := acct.available(organizationID)
	threshold := acct.standing.Limits.LowBalanceMicros
	m.mu.Unlock()
	if decided.organization == organizationID && decided.balanceWarning != nil {
		threshold = decided.balanceWarning
	}
	return Balance{AvailableMicros: available, ThresholdMicros: threshold, State: stateOf(available, threshold)}, true
}

// available is the balance less every hold, under m.mu.
func (a *account) available(key string) int64 {
	return a.standing.BalanceMicros - a.held - a.remoteOrganization(key)
}
