package conversation

import (
	"context"
	"errors"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// Chats says which Stream app a conversation is kept in, and reaches it there.
//
// A conversation is written in the app its session was pinned to, and finished there:
// delivered, read back and recovered after a restart in that app, whatever app its customer
// acts in by then.
type Chats interface {
	// For is the client and pin a new conversation for a customer is kept with.
	For(ctx context.Context, customer string) (*getstream.Stream, int64, error)
	// ForApp is the client a conversation already pinned to an app is reached with.
	ForApp(ctx context.Context, customer string, app int64) (*getstream.Stream, error)
	// DeploymentApp is the deployment's own app id, zero while it is not known.
	DeploymentApp() int64
}

// Pins finds which app a conversation with no record here was written in, from wherever
// the deployment remembers its sessions. Found is false for a conversation it has no word
// of.
type Pins func(ctx context.Context, customer, cid string) (app int64, found bool, err error)

// oneChat is a single client answering for every customer and pin, which is what a caller
// that already holds one client means.
type oneChat struct {
	client *getstream.Stream
}

func (o oneChat) For(context.Context, string) (*getstream.Stream, int64, error) {
	return o.client, 0, nil
}

func (o oneChat) ForApp(context.Context, string, int64) (*getstream.Stream, error) {
	return o.client, nil
}

func (o oneChat) DeploymentApp() int64 { return 0 }

// StreamApps is Chats over the router's own resolution of customers to Stream apps.
func StreamApps(clients *streamapp.Clients) Chats { return streamApps{clients: clients} }

type streamApps struct {
	clients *streamapp.Clients
}

func (a streamApps) For(ctx context.Context, customer string) (*getstream.Stream, int64, error) {
	bound, err := a.clients.For(ctx, customer)
	if err != nil {
		return nil, 0, err
	}
	return bound.Client, bound.Identity.StreamApp, nil
}

func (a streamApps) ForApp(ctx context.Context, customer string, app int64) (*getstream.Stream, error) {
	bound, err := a.clients.ForApp(ctx, customer, app)
	if err != nil {
		return nil, err
	}
	return bound.Client, nil
}

// ForAppReading is the client a conversation is read back with, which reaches one kept in
// the deployment's app even once it can no longer be written there.
func (a streamApps) ForAppReading(ctx context.Context, customer string, app int64) (*getstream.Stream, error) {
	bound, err := a.clients.ForAppReading(ctx, customer, app)
	if err != nil {
		return nil, err
	}
	return bound.Client, nil
}

func (a streamApps) DeploymentApp() int64 { return a.clients.DeploymentApp() }

// readingChats is Chats that can read back a conversation it would no longer write to.
type readingChats interface {
	ForAppReading(ctx context.Context, customer string, app int64) (*getstream.Stream, error)
}

// parked reports whether an error means the conversation's app cannot be reached from here
// at all, or can only be read. Its writes are held and tried again much later, rather than
// delivered anywhere else.
func parked(err error) bool {
	return errors.Is(err, streamapp.ErrStreamAppMoved) ||
		errors.Is(err, streamapp.ErrStreamAppDisconnected) ||
		errors.Is(err, streamapp.ErrReadOnly)
}

// parkedRetry is how long a conversation whose app is out of reach waits to try again.
const parkedRetry = 5 * time.Minute

// SetPins says where to find which app a conversation with no record here was written in.
func (s *Service) SetPins(pins Pins) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pins = pins
}

// pinOf is the app a conversation is kept in, from its record or the deployment's memory of
// its sessions. Found is false for a conversation nobody has word of.
func (s *Service) pinOf(ctx context.Context, customer, cid string) (int64, bool, error) {
	s.mu.Lock()
	open, pins := s.all[known{customer, cid}], s.pins
	s.mu.Unlock()
	if open != nil {
		open.mu.Lock()
		owner, app := open.data.Customer, open.data.StreamApp
		open.mu.Unlock()
		if owner == customer {
			return app, true, nil
		}
	}
	return lookupPin(ctx, pins, customer, cid)
}

// lookupPin asks the deployment's memory of its sessions which app a conversation is in.
func lookupPin(ctx context.Context, pins Pins, customer, cid string) (int64, bool, error) {
	if pins == nil {
		return 0, false, nil
	}
	return pins(ctx, customer, cid)
}

// clientFor is the client a conversation for a customer is written with: in the app it is
// kept in, or a new one's in the customer's own.
func (s *Service) clientFor(ctx context.Context, customer, cid string) (*getstream.Stream, int64, error) {
	app, err := s.appOf(ctx, customer, cid, false)
	if err != nil {
		return nil, 0, err
	}
	client, err := s.chats.ForApp(ctx, customer, app)
	return client, app, err
}

// readerFor is the client a conversation is read back with, which is clientFor's except
// for a conversation kept where it can no longer be written, which can still be read.
func (s *Service) readerFor(ctx context.Context, customer, cid string) (*getstream.Stream, int64, error) {
	reading, ok := s.chats.(readingChats)
	if !ok {
		return s.clientFor(ctx, customer, cid)
	}
	app, err := s.appOf(ctx, customer, cid, true)
	if err != nil {
		return nil, 0, err
	}
	client, err := reading.ForAppReading(ctx, customer, app)
	return client, app, err
}

// appOf is the app a conversation is kept in, or for one nobody has word of, the app the
// customer acts in. A customer with no app at all reads as the deployment's. So does one
// whose app is disconnected, for reading only: what it wrote is still its to read.
func (s *Service) appOf(ctx context.Context, customer, cid string, reading bool) (int64, error) {
	// Where the conversation was written settles it; which app the customer acts in now is
	// asked only for one nothing has a record of.
	if app, found, err := s.pinOf(ctx, customer, cid); err != nil || found {
		return app, err
	}
	_, own, err := s.chats.For(ctx, customer)
	tolerated := errors.Is(err, streamapp.ErrNoIdentity) || (reading && errors.Is(err, streamapp.ErrStreamAppDisconnected))
	if err != nil && !tolerated {
		return 0, err
	}
	return own, nil
}

// AppOf is the app a conversation is kept in, for a caller deciding whether a session in
// another app may be bound to it.
func (s *Service) AppOf(ctx context.Context, customer, cid string) (int64, error) {
	return s.appOf(ctx, customer, cid, true)
}
