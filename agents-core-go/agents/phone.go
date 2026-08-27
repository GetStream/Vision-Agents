package agents

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/agents-core-go/acceleration"
	"github.com/GetStream/Vision-Agents/agents-core-go/edge"
	"github.com/GetStream/Vision-Agents/agents-core-go/stream"
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
// The number is pointed at a fresh Stream call, the agent joins it, and this blocks until
// somebody rings and says something. The session it returns is the conversation with
// whoever that was, from their second sentence: the one that unblocked this is read here
// and does not arrive again on Events.
func (a *Agent) WaitForCall(ctx context.Context, number string) (*Session, error) {
	if number == "" {
		return nil, errors.New("agents: there is no number to answer on")
	}

	telephony, err := a.Phone()
	if err != nil {
		return nil, err
	}
	transport, err := a.edge()
	if err != nil {
		return nil, err
	}

	call, err := transport.CreateCall(ctx, edge.Call{}, edge.User{ID: a.options.UserID, Name: a.options.Name})
	if err != nil {
		return nil, err
	}
	if _, err := telephony.Attach(ctx, number, call.ID, call.Type); err != nil {
		return nil, err
	}

	session, err := a.join(ctx, call, &acceleration.SessionPhone{Number: number}, false)
	if err != nil {
		return nil, err
	}

	a.logger.Info("waiting for a call", "number", number, "call", call.ID)
	if err := session.waitForCaller(ctx); err != nil {
		_ = session.Close(context.WithoutCancel(ctx))
		return nil, err
	}
	return session, nil
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
	transport, err := a.edge()
	if err != nil {
		return nil, err
	}

	call, err := transport.CreateCall(ctx, edge.Call{}, edge.User{ID: a.options.UserID, Name: a.options.Name})
	if err != nil {
		return nil, err
	}
	// Placing the call makes its own trunk and routing rule, pinned to the call named
	// here, so the answered leg arrives in the call this agent is about to join. Attaching
	// the number first would be a second rule for the same number.
	placed, err := telephony.Place(ctx, stream.OutboundCall{
		From:     from,
		To:       to,
		CallID:   call.ID,
		CallType: call.Type,
		Tags:     a.options.CostTracking,
	})
	if err != nil {
		return nil, err
	}

	a.logger.Info("ringing", "to", to, "from", from, "call", call.ID)
	return a.join(ctx, call, &acceleration.SessionPhone{
		Number:       from,
		VendorCallId: &placed.VendorCallId,
	}, true)
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
