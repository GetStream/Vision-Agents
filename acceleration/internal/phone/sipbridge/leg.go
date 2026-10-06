package sipbridge

import "context"

// leg is one side of the call. sipLeg is the real one; tests use fakeLeg.
type leg interface {
	// Invite sends the initial INVITE (body nil for a late offer) and waits for a 2xx,
	// answering digest challenges. Cancelling ctx before the answer sends CANCEL, and a
	// 2xx that crosses it is ACKed and hung up. It returns the 2xx body and does not ACK.
	Invite(ctx context.Context, body []byte) ([]byte, error)
	// Ack confirms the initial 2xx. body carries our answer when the 2xx held the offer.
	Ack(ctx context.Context, body []byte) error
	// Reinvite sends a new SDP offer in the dialog, ACKs the 2xx and returns its body.
	Reinvite(ctx context.Context, body []byte) ([]byte, error)
	Info(ctx context.Context, contentType string, body []byte) error
	// Bye ends the dialog. On a dialog that already ended it does nothing.
	Bye(ctx context.Context) error
}

type side int

const (
	customerSide side = iota
	streamSide
)

func (s side) String() string {
	if s == customerSide {
		return "customer"
	}
	return "stream"
}
