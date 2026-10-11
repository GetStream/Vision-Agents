package agents

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// NumberSearch narrows what the vendor is asked to offer.
type NumberSearch = stream.NumberSearch

// Phone reaches the telephony paths on the router this agent talks to.
func (a *Agent) Phone() (*stream.Phone, error) {
	client, err := a.Client()
	if err != nil {
		return nil, err
	}
	return stream.NewPhone(client), nil
}

// PurchaseAnyNumber buys the first number a vendor offers that matches the search.
//
// It starts a monthly charge, so it is not something to call on every run. An agent that
// answers the same number every day should buy it once and pass it to WaitForCall.
func (a *Agent) PurchaseAnyNumber(ctx context.Context, search NumberSearch) (string, error) {
	telephony, err := a.Phone()
	if err != nil {
		return "", err
	}
	if len(search.Tags) == 0 && len(a.options.CostTracking) > 0 {
		search.Tags = a.options.CostTracking
	}

	number, err := telephony.PurchaseAnyNumber(ctx, search)
	if err != nil {
		return "", err
	}
	a.logger.Info("bought a number", "number", number.E164, "vendor", number.Vendor)
	return number.E164, nil
}

// WaitForCall answers the next call to a number.
//
// The number is attached, so every caller lands in a call of their own, and this waits for
// the router to hand the next one over, then blocks until the caller says something. The
// session it returns is the conversation with whoever that was, from their second sentence:
// the one that unblocked this is read here and does not arrive again on Events.
func (a *Agent) WaitForCall(ctx context.Context, number string) (*Session, error) {
	if number == "" {
		return nil, errors.New("agents: there is no number to answer on")
	}

	telephony, err := a.Phone()
	if err != nil {
		return nil, err
	}
	if _, err := telephony.Attach(ctx, number); err != nil {
		return nil, err
	}
	backend, err := a.options.LLM.Backend()
	if err != nil {
		return nil, err
	}
	dispatch, err := stream.NewDispatch(stream.DispatchOptions{Backend: backend, Capacity: 1, Logger: a.logger})
	if err != nil {
		return nil, err
	}

	arrived := make(chan InboundCall, 1)
	dispatch.OnCall(func(_ context.Context, call InboundCall) error {
		if call.CalledNumber != number {
			return fmt.Errorf("agents: this agent is waiting on %s, not %s", number, call.CalledNumber)
		}
		select {
		case arrived <- call:
			return nil
		default:
			return errors.New("agents: this agent is already answering a call")
		}
	})
	waiting, stop := context.WithCancel(ctx)
	defer stop()
	failed := make(chan error, 1)
	go func() { failed <- dispatch.Run(waiting) }()

	a.logger.Info("waiting for a call", "number", number)
	var call InboundCall
	select {
	case call = <-arrived:
	case err := <-failed:
		if err == nil {
			err = errors.New("agents: stopped waiting before anybody rang")
		}
		return nil, err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	stop()

	session, err := a.Answer(ctx, call)
	if err != nil {
		return nil, err
	}
	if err := session.waitForCaller(ctx); err != nil {
		_ = session.Close(context.WithoutCancel(ctx))
		return nil, err
	}
	return session, nil
}

// Answer holds the conversation with somebody who rang one of the customer's numbers, on
// the call the router routed them into, which is named for the session opened here.
func (a *Agent) Answer(ctx context.Context, call InboundCall) (*Session, error) {
	if call.SessionID == "" {
		return nil, errors.New("agents: the call names no session; attach its number again")
	}
	return a.join(ctx, stream.Call{
		SessionID: call.SessionID,
		Voice:     true,
		Phone:     &acceleration.SessionPhone{Number: call.CalledNumber},
	})
}

// StartCall rings somebody and holds the conversation when they answer.
//
// The agent placed this call, so it is told it is navigating: recordings are let finish and
// menus are answered rather than talked over.
func (a *Agent) StartCall(ctx context.Context, from, to string) (*Session, error) {
	if from == "" || to == "" {
		return nil, errors.New("agents: a call needs a number to ring from and one to ring")
	}

	telephony, err := a.Phone()
	if err != nil {
		return nil, err
	}
	// Placing the call makes its own trunk and routing rule, pinned to the call of the
	// session it names, so the answered leg arrives in the call this agent is about to join.
	// Attaching the number first would be a second rule for the same number.
	placed, err := telephony.Place(ctx, stream.OutboundCall{
		From: from,
		To:   to,
		Tags: a.options.CostTracking,
	})
	if err != nil {
		return nil, err
	}
	if placed.SessionId == nil {
		return nil, errors.New("agents: the router placed the call for no session")
	}

	a.logger.Info("ringing", "to", to, "from", from, "session", *placed.SessionId)
	return a.join(ctx, stream.Call{
		SessionID: *placed.SessionId,
		Voice:     true,
		Phone: &acceleration.SessionPhone{
			Number:       from,
			VendorCallId: &placed.VendorCallId,
		},
		Navigating: true,
	})
}

// waitForCaller blocks until somebody says something on the call.
//
// A phone call the agent is holding for is not a conversation until it is answered, and
// there is nothing to say to an empty room.
func (s *Session) waitForCaller(ctx context.Context) error {
	events := s.Events()
	for {
		select {
		case event, open := <-events:
			if !open {
				return fmt.Errorf("agents: the call ended before anybody rang")
			}
			if event.Kind == "heard" {
				return nil
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}
