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

type Person struct {
	mu       sync.Mutex
	client   *api
	call     person.Call
	called   bool
	statuses map[string]string // call sid -> last status seen; "" until seen
}

func New() person.Person { return &Person{statuses: map[string]string{}} }

func (p *Person) Name() string { return "twilio" }

func (p *Person) Flags(*flag.FlagSet) {}

// Dial calls the agent number from the person number, which says the script and hangs up.
func (p *Person) Dial(ctx context.Context, call person.Call) error {
	a, err := p.api()
	if err != nil {
		return err
	}
	p.remember(call)
	sid, err := a.placeCall(ctx, call.PersonNumber, call.AgentNumber, dialScript(call.Talk))
	if err != nil {
		return err
	}
	p.track(sid, "")
	return nil
}

// Wait follows the person's call to a final status. Inbound it is the call Dial placed;
// outbound it is the newest call to the person's number since the run began.
func (p *Person) Wait(ctx context.Context, call person.Call) error {
	a, err := p.api()
	if err != nil {
		return err
	}
	p.remember(call)
	sid, err := p.callToFollow(ctx, a, call)
	if err != nil || sid == "" {
		return err
	}
	return p.follow(ctx, a, call, sid)
}

// Hangup ends every call it placed or found that is not over.
func (p *Person) Hangup(ctx context.Context) error {
	a, err := p.api()
	if err != nil {
		return err
	}
	live, err := p.liveCalls(ctx, a)
	var errs []error
	if err != nil {
		errs = append(errs, err)
	}
	for _, sid := range live {
		if err := a.complete(ctx, sid); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (p *Person) callToFollow(ctx context.Context, a api, call person.Call) (string, error) {
	if call.Direction == person.Inbound {
		return p.placedSID(), nil
	}
	sid, err := p.findOutbound(ctx, a, call)
	if sid == "" && err == nil {
		call.Logger.Warn("person call not observed")
	}
	return sid, err
}

// findOutbound waits for the router's call to the person's number to show up.
func (p *Person) findOutbound(ctx context.Context, a api, call person.Call) (string, error) {
	searchCtx, cancel := context.WithTimeout(ctx, call.Talk+waitMargin)
	defer cancel()
	for {
		recs, err := a.callsTo(searchCtx, call.PersonNumber, call.Started)
		if err != nil && searchCtx.Err() == nil {
			return "", err
		}
		if newest, ok := newestOf(recs); ok {
			p.track(newest.SID, newest.Status)
			return newest.SID, nil
		}
		if !sleep(searchCtx, pollEvery) {
			return "", ctx.Err()
		}
	}
}

func (p *Person) follow(ctx context.Context, a api, call person.Call, sid string) error {
	followCtx, cancel := context.WithTimeout(ctx, call.Talk+waitMargin)
	defer cancel()
	last := ""
	for {
		rec, err := a.call(followCtx, sid)
		if err != nil && followCtx.Err() == nil {
			return err
		}
		if err == nil {
			p.track(sid, rec.Status)
			if rec.Status != last {
				logStatus(call, rec)
				last = rec.Status
			}
			if isFinal(rec.Status) {
				return nil
			}
		}
		if !sleep(followCtx, pollEvery) {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return fmt.Errorf("the person's call did not end within %s of the talk time", waitMargin)
		}
	}
}

func logStatus(call person.Call, rec callRecord) {
	call.Logger.Info("person call", "status", rec.Status, "start", formatTime(rec.StartTime.Time),
		"end", formatTime(rec.EndTime.Time), "region_label", call.RegionLabel)
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
