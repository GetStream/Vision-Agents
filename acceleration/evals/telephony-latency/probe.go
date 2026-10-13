package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net/netip"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// Result is one call between two of the bench's numbers.
type Result struct {
	// Region is where the bench ran, e.g. "us-east1".
	Region string    `json:"region"`
	Run    int       `json:"run"`
	At     time.Time `json:"at"`
	Caller Line      `json:"caller"`
	Callee Line      `json:"callee"`

	// RingMs is from asking the caller's vendor to dial to the callee's leg arriving.
	RingMs float64 `json:"ring_ms,omitempty"`
	// BridgeMs is from answering the callee's leg to the caller's vendor bridging its own.
	BridgeMs float64 `json:"bridge_ms,omitempty"`

	// ForwardMs is each stamp's trip from the caller's side to the callee's.
	ForwardMs []float64 `json:"forward_ms,omitempty"`
	// BackwardMs is each stamp's trip from the callee's side back to the caller's.
	BackwardMs []float64 `json:"backward_ms,omitempty"`
	// Sent is how many stamps went each way.
	Sent int `json:"sent"`
	// Garbled is stamps that arrived but did not decode, both ways together.
	Garbled int `json:"garbled,omitempty"`

	// CallerSignal and CallerMedia are the caller's vendor's addresses, as seen here.
	CallerSignal string `json:"caller_signal,omitempty"`
	CallerMedia  string `json:"caller_media,omitempty"`
	CalleeSignal string `json:"callee_signal,omitempty"`
	CalleeMedia  string `json:"callee_media,omitempty"`

	// Error is why the call did not complete, empty when it did.
	Error string `json:"error,omitempty"`
}

// Prober places measured calls.
type Prober struct {
	Endpoint  *Endpoint
	Providers map[string]phone.Provider
	// Host is the address carriers send calls to.
	Host netip.Addr
	// Port is the SIP port carriers send calls to.
	Port int
	// Stamps is how many stamps go each way on a call.
	Stamps int
	// Gap is the time between one stamp and the next on the same call.
	Gap time.Duration
	// Patience bounds how long a leg may take to arrive and a stamp to be heard.
	Patience time.Duration
}

// Probe calls callee from caller, sends stamps both ways and hangs up. A call that fails
// still returns a result, with the error in it, so a failing cell is reported rather than
// missing.
func (p *Prober) Probe(ctx context.Context, caller, callee Line) Result {
	result := Result{At: time.Now().UTC(), Caller: caller, Callee: callee}
	if err := p.probe(ctx, &result); err != nil {
		result.Error = err.Error()
	}
	return result
}

func (p *Prober) probe(ctx context.Context, result *Result) error {
	provider, ok := p.Providers[result.Caller.Vendor]
	if !ok {
		return fmt.Errorf("no provider for %s", result.Caller.Vendor)
	}

	id, err := probeID()
	if err != nil {
		return err
	}
	calleeArrived := p.Endpoint.Expect(result.Callee.E164)
	defer p.Endpoint.Forget(result.Callee.E164)
	callerArrived := p.Endpoint.Expect(id)
	defer p.Endpoint.Forget(id)

	outbound := phone.Outbound{
		From:   result.Caller.E164,
		To:     result.Callee.E164,
		Bridge: phone.Bridge{URI: p.uri(id)},
	}
	if provider.Dials(phone.FeatureRingTimeout) {
		outbound.RingTimeout = p.Patience
	}
	dialed := time.Now()
	if _, err := provider.Dial(ctx, outbound); err != nil {
		return fmt.Errorf("dial: %w", err)
	}

	callee, err := wait(ctx, calleeArrived, p.Patience, "the callee's leg")
	if err != nil {
		return err
	}
	defer p.hangup(callee)
	result.RingMs = millis(callee.Answered.Sub(dialed))
	result.CalleeSignal = callee.Signal.String()

	caller, err := wait(ctx, callerArrived, p.Patience, "the caller's leg")
	if err != nil {
		return err
	}
	defer p.hangup(caller)
	result.BridgeMs = millis(caller.Answered.Sub(callee.Answered))
	result.CallerSignal = caller.Signal.String()

	// Media starts flowing a moment after the answer; a stamp sent into that moment is
	// lost to the call setting up rather than to the network.
	if err := sleep(ctx, time.Second); err != nil {
		return err
	}

	// The two directions take turns, half a gap apart, so a stamp is never sent while one
	// from the other side is arriving.
	for range p.Stamps {
		caller.Media.Say()
		if err := sleep(ctx, p.Gap/2); err != nil {
			return err
		}
		callee.Media.Say()
		if err := sleep(ctx, p.Gap/2); err != nil {
			return err
		}
	}
	result.Sent = p.Stamps

	forward := collect(ctx, callee.Media.Heard(), p.Stamps, p.Patience)
	backward := collect(ctx, caller.Media.Heard(), p.Stamps, p.Patience)
	for _, heard := range forward {
		result.ForwardMs = append(result.ForwardMs, millis(heard.Latency()))
	}
	for _, heard := range backward {
		result.BackwardMs = append(result.BackwardMs, millis(heard.Latency()))
	}
	result.Garbled = caller.Media.Garbled() + callee.Media.Garbled()
	result.CallerMedia = caller.Media.Remote().String()
	result.CalleeMedia = callee.Media.Remote().String()

	if len(forward) == 0 && len(backward) == 0 {
		return errors.New("no stamp was heard either way")
	}
	return nil
}

// uri is the bench's address for a leg, the one a vendor dials to bridge its call here.
func (p *Prober) uri(user string) string {
	return fmt.Sprintf("sip:%s@%s:%d", user, p.Host, p.Port)
}

func (p *Prober) hangup(leg *Leg) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = leg.Hangup(ctx)
}

// collect waits for stamps until all of them have arrived or none has for a while.
func collect(ctx context.Context, heard <-chan Heard, want int, patience time.Duration) []Heard {
	var got []Heard
	for len(got) < want {
		select {
		case h, open := <-heard:
			if !open {
				return got
			}
			got = append(got, h)
		case <-time.After(patience):
			return got
		case <-ctx.Done():
			return got
		}
	}
	return got
}

func wait(ctx context.Context, arrived <-chan *Leg, patience time.Duration, what string) (*Leg, error) {
	select {
	case leg := <-arrived:
		return leg, nil
	case <-time.After(patience):
		return nil, fmt.Errorf("%s did not arrive within %s", what, patience)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func sleep(ctx context.Context, d time.Duration) error {
	select {
	case <-time.After(d):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func probeID() (string, error) {
	raw := make([]byte, 8)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("probe id: %w", err)
	}
	return "probe-" + hex.EncodeToString(raw), nil
}

func millis(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
