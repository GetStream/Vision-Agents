package phone

import (
	"context"
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// LineOptions describes the call an agent is on.
type LineOptions struct {
	Owner routing.Owner
	// From is the customer's number this call is on. A transfer is dialled from it, so
	// the human sees the same number the caller rang.
	From string
	// CallID is the Stream call the agent is in, and the one a transferred human joins.
	CallID string
	// CallType is the Stream call type. Empty means "default".
	CallType string
	// Vendor is who is carrying the leg digits would be pressed on.
	Vendor string
	// VendorCallID is that leg. It is set for a call the agent placed and empty for one
	// that came to it, which is why an agent that was rung cannot press anything.
	VendorCallID string
}

// Line is one call as the agent sees it: somewhere to bring a human, and a leg to press
// digits at.
//
// It exists so the agent can act on a call without knowing what a trunk is. The agent holds
// this as an interface of two methods; everything about numbers, vendors and SIP stays on
// this side of it.
type Line struct {
	service *Service
	options LineOptions
}

// Line returns the call operations for one conversation.
func (s *Service) Line(options LineOptions) *Line {
	return &Line{service: s, options: options}
}

// Transfer brings a human onto the call.
func (l *Line) Transfer(ctx context.Context, to string) error {
	_, err := l.service.Transfer(ctx, TransferRequest{
		Owner:    l.options.Owner,
		From:     l.options.From,
		To:       to,
		CallID:   l.options.CallID,
		CallType: l.options.CallType,
	})
	return err
}

// SendDigits presses digits on the leg this call was placed over.
func (l *Line) SendDigits(ctx context.Context, digits string) error {
	if l.options.VendorCallID == "" {
		return errors.New("phone: this call was not placed from here, so there is no " +
			"leg to press digits on")
	}
	return l.service.SendDigits(ctx, l.options.Vendor, l.options.VendorCallID, digits)
}
