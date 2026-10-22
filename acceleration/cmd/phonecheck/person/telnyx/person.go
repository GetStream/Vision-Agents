// Package telnyx is the person side of a call on Telnyx: it places, watches and ends calls,
// and never changes a number, connection or app setting.
package telnyx

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
	// waitMargin is how long past the talk time a call may take to end, so Wait never
	// blocks forever.
	waitMargin = 60 * time.Second
	// observeMargin is how long past the talk time an unobserved call is given to end.
	observeMargin = 15 * time.Second
)

var finalStatuses = []string{"completed", "busy", "failed", "no-answer", "canceled"}

func isFinal(status string) bool { return slices.Contains(finalStatuses, status) }

// Person places one call at a time. Wait runs beside the run's main flow and Hangup runs at
// the end, so the call it placed is kept under a lock.
type Person struct {
	mu     sync.Mutex
	sid    string // the call Dial placed; empty when none
	status string // the last status seen for it
}

func New() person.Person { return &Person{} }

func (p *Person) Name() string { return "telnyx" }

func (p *Person) Flags(*flag.FlagSet) {}

// Ready checks the credentials only for inbound: outbound, the adapter never calls Telnyx.
func (p *Person) Ready(dir person.Direction) error {
	if dir == person.Outbound {
		return nil
	}
	a, err := newAPI()
	if err != nil {
		return err
	}
	if a.appID == "" {
		return errors.New("TELNYX_PERSON_CONNECTION_ID is not set")
	}
	return nil
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

// Wait follows the call Dial placed to a final status. Telnyx gives no call id for a call
// to the person's number, so outbound it only waits out the call.
func (p *Person) Wait(ctx context.Context, call person.Call) (time.Duration, error) {
	if call.Direction == person.Outbound {
		return 0, waitUnobserved(ctx, call)
	}
	sid, _ := p.placed()
	if sid == "" {
		return 0, errors.New("no call was placed, so there is none to follow")
	}
	a, err := newAPI()
	if err != nil {
		return 0, err
	}
	return p.follow(ctx, a, call, sid)
}

// Hangup ends the call it placed, if it is still live. The call is not needed: Telnyx only
// knows of calls the adapter placed.
func (p *Person) Hangup(ctx context.Context, _ person.Call) error {
	defer p.setCall("", "")
	sid, status := p.placed()
	if sid == "" || isFinal(status) {
		return nil
	}
	a, err := newAPI()
	if err != nil {
		return err
	}
	return a.hangup(ctx, sid)
}

func (p *Person) placed() (sid, status string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.sid, p.status
}

func (p *Person) setCall(sid, status string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.sid, p.status = sid, status
}

// waitUnobserved lets the hand-set script's <Hangup/> end the call.
func waitUnobserved(ctx context.Context, call person.Call) error {
	call.Logger.Warn("person call not observed")
	if !sleep(ctx, call.Talk+observeMargin) {
		return ctx.Err()
	}
	return nil
}

func (p *Person) follow(ctx context.Context, a api, call person.Call, sid string) (time.Duration, error) {
	followCtx, cancel := context.WithTimeout(ctx, call.Talk+waitMargin)
	defer cancel()
	last, seen := "", false
	for {
		rec, err := a.call(followCtx, sid)
		switch {
		case err != nil && followCtx.Err() != nil:
			return 0, didNotEnd(ctx)
		case errors.Is(err, errNotFound) && !seen:
			// Telnyx reads are eventually consistent: a call just placed may 404 for a moment.
		case err != nil:
			return 0, err
		default:
			seen = true
			p.setCall(sid, rec.Status)
			if rec.Status != last {
				logStatus(call, rec)
				last = rec.Status
			}
			if isFinal(rec.Status) {
				return lengthOf(rec), nil
			}
		}
		if !sleep(followCtx, pollEvery) {
			return 0, didNotEnd(ctx)
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

func logStatus(call person.Call, rec callRecord) {
	warnUnread(call, "start", rec.StartTime)
	warnUnread(call, "end", rec.EndTime)
	logged := []any{"status", rec.Status, "start", formatTime(rec.StartTime.Time),
		"end", formatTime(rec.EndTime.Time), "region_label", call.RegionLabel}
	if length := lengthOf(rec); length > 0 {
		logged = append(logged, "person_call_s", int(length.Seconds()))
	}
	call.Logger.Info("person call", logged...)
}

// lengthOf is how long the person's call lasted by its record, zero until it has both times.
func lengthOf(rec callRecord) time.Duration {
	return person.LengthOf(rec.StartTime.Time, rec.EndTime.Time)
}

func warnUnread(call person.Call, field string, t telnyxTime) {
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

func sleep(ctx context.Context, d time.Duration) bool {
	select {
	case <-time.After(d):
		return true
	case <-ctx.Done():
		return false
	}
}
