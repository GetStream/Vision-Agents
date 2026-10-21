// Package twilio is the person side of a call on Twilio: it places, watches and ends calls,
// and never changes a number's settings.
package twilio

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

var (
	pollEvery = 2 * time.Second
	// waitMargin is how long past the talk time a call may take to appear or to end, so Wait
	// never blocks forever.
	waitMargin = 60 * time.Second
)

var finalStatuses = []string{"completed", "busy", "failed", "no-answer", "canceled"}

func isFinal(status string) bool { return slices.Contains(finalStatuses, status) }

// Person follows one call at a time. Wait runs beside the run's main flow and Hangup runs at
// the end, so the call is kept under a lock.
type Person struct {
	mu     sync.Mutex
	sid    string // the call Dial placed or Wait found; empty when none
	status string // the last status seen for it
}

func New() person.Person { return &Person{} }

func (p *Person) Name() string { return "twilio" }

func (p *Person) Flags(*flag.FlagSet) {}

func (p *Person) Ready(person.Direction) error {
	_, err := newAPI()
	return err
}

// Dial calls the agent number from the person number, which says the script and hangs up.
func (p *Person) Dial(ctx context.Context, call person.Call) error {
	a, err := newAPI()
	if err != nil {
		return err
	}
	sid, err := a.placeCall(ctx, call.PersonNumber, call.AgentNumber, dialScript(call.Talk))
	if err != nil {
		return err
	}
	p.setCall(sid, "")
	return nil
}

// Wait follows the person's call to a final status. Inbound it is the call Dial placed;
// outbound it is the newest call to the person's number since the run began.
func (p *Person) Wait(ctx context.Context, call person.Call) error {
	a, err := newAPI()
	if err != nil {
		return err
	}
	if call.Direction == person.Inbound {
		sid, _ := p.current()
		if sid == "" {
			return errors.New("no call was placed, so there is none to follow")
		}
		return p.follow(ctx, a, call, sid)
	}
	sid, found, err := p.findOutbound(ctx, a, call)
	if err != nil {
		return err
	}
	if !found {
		call.Logger.Warn("person call not observed")
		return nil
	}
	return p.follow(ctx, a, call, sid)
}

// Hangup ends the call it placed or found, if it is still live. An outbound call it never
// found, as when the run was stopped while looking, is looked for once more.
func (p *Person) Hangup(ctx context.Context, call person.Call) error {
	defer p.setCall("", "")
	sid, status := p.current()
	if sid != "" && isFinal(status) {
		return nil
	}
	if sid == "" && call.Direction == person.Inbound {
		return nil
	}
	a, err := newAPI()
	if err != nil {
		return err
	}
	if sid != "" {
		return a.complete(ctx, sid)
	}
	return endLiveCallsTo(ctx, a, call)
}

func (p *Person) current() (sid, status string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.sid, p.status
}

func (p *Person) setCall(sid, status string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.sid, p.status = sid, status
}

// findOutbound waits for the router's call to the person's number to show up. Not found
// with no error means the search ran out of time.
func (p *Person) findOutbound(ctx context.Context, a api, call person.Call) (sid string, found bool, err error) {
	searchCtx, cancel := context.WithTimeout(ctx, call.Talk+waitMargin)
	defer cancel()
	for {
		recs, listErr := a.callsTo(searchCtx, call.PersonNumber, call.Started)
		if listErr != nil {
			if searchCtx.Err() != nil {
				return "", false, ctx.Err() // nil when only the search's own time ran out
			}
			return "", false, listErr
		}
		if newest, ok := newestOf(recs); ok {
			p.setCall(newest.SID, newest.Status)
			return newest.SID, true, nil
		}
		if !sleep(searchCtx, pollEvery) {
			return "", false, ctx.Err()
		}
	}
}

func (p *Person) follow(ctx context.Context, a api, call person.Call, sid string) error {
	followCtx, cancel := context.WithTimeout(ctx, call.Talk+waitMargin)
	defer cancel()
	last := ""
	for {
		rec, err := a.call(followCtx, sid)
		if err != nil {
			if followCtx.Err() != nil {
				return didNotEnd(ctx)
			}
			return err
		}
		p.setCall(sid, rec.Status)
		if rec.Status != last {
			logStatus(call, rec)
			last = rec.Status
		}
		if isFinal(rec.Status) {
			return nil
		}
		if !sleep(followCtx, pollEvery) {
			return didNotEnd(ctx)
		}
	}
}

// didNotEnd is the run's own error when it was stopped, else says the call outlived its time.
func didNotEnd(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fmt.Errorf("the person's call did not end within %s of the talk time", waitMargin)
}

// endLiveCallsTo ends every call to the person's number since the run began that is not over.
func endLiveCallsTo(ctx context.Context, a api, call person.Call) error {
	recs, err := a.callsTo(ctx, call.PersonNumber, call.Started)
	errs := []error{err}
	for _, rec := range recs {
		if !isFinal(rec.Status) {
			errs = append(errs, a.complete(ctx, rec.SID))
		}
	}
	return errors.Join(errs...)
}

func logStatus(call person.Call, rec callRecord) {
	warnUnread(call, "start", rec.StartTime)
	warnUnread(call, "end", rec.EndTime)
	call.Logger.Info("person call", "status", rec.Status, "start", formatTime(rec.StartTime.Time),
		"end", formatTime(rec.EndTime.Time), "region_label", call.RegionLabel)
}

func warnUnread(call person.Call, field string, t twilioTime) {
	if t.Unread != "" {
		call.Logger.Warn("could not read a time of the person's call", "field", field, "value", t.Unread)
	}
}

func formatTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339)
}

func newestOf(recs []callRecord) (callRecord, bool) {
	if len(recs) == 0 {
		return callRecord{}, false
	}
	return slices.MaxFunc(recs, func(a, b callRecord) int { return a.StartTime.Compare(b.StartTime.Time) }), true
}

func sleep(ctx context.Context, d time.Duration) bool {
	select {
	case <-time.After(d):
		return true
	case <-ctx.Done():
		return false
	}
}
