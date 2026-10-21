package twilio

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

// The state below is shared: Wait runs beside the run's main flow, and Hangup runs at the end.

// api builds the client on first use, so the credentials are only needed once a call is made.
func (p *Person) api() (api, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.client != nil {
		return *p.client, nil
	}
	a, err := newAPI()
	if err != nil {
		return api{}, err
	}
	p.client = &a
	return a, nil
}

// remember keeps the call so Hangup can look for its live calls without being given it.
func (p *Person) remember(call person.Call) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.call, p.called = call, true
}

func (p *Person) track(sid, status string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.statuses[sid] = status
}

// placedSID is the call Dial placed. A Person makes one call at a time.
func (p *Person) placedSID() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	for sid := range p.statuses {
		return sid
	}
	return ""
}

// liveCalls is every tracked call not known to be over, plus those listed live to the
// person's number since the run began.
func (p *Person) liveCalls(ctx context.Context, a api) ([]string, error) {
	p.mu.Lock()
	call, called := p.call, p.called
	live := map[string]bool{}
	for sid, status := range p.statuses {
		if !isFinal(status) {
			live[sid] = true
		}
	}
	p.mu.Unlock()

	var err error
	if called {
		var recs []callRecord
		recs, err = a.callsTo(ctx, call.PersonNumber, call.Started)
		for _, rec := range recs {
			if !isFinal(rec.Status) {
				live[rec.SID] = true
			}
		}
	}
	sids := make([]string, 0, len(live))
	for sid := range live {
		sids = append(sids, sid)
	}
	return sids, err
}
