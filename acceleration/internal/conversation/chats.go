package conversation

import (
	"context"
	"encoding/hex"
	"errors"
	"path/filepath"
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

func (a streamApps) DeploymentApp() int64 { return a.clients.DeploymentApp() }

// parked reports whether an error means the conversation's app cannot be reached from here
// at all. Its writes stay on disk and are tried again much later, rather than delivered
// anywhere else.
func parked(err error) bool {
	return errors.Is(err, streamapp.ErrStreamAppMoved) ||
		errors.Is(err, streamapp.ErrStreamAppDisconnected) ||
		errors.Is(err, streamapp.ErrReadOnly)
}

// parkedRetry is how long a conversation whose app is out of reach waits to try again.
const parkedRetry = 5 * time.Minute

// appsDir holds the records of conversations kept in apps other than the deployment's,
// one directory per customer.
const appsDir = "apps"

// legacy reports whether a pin is the deployment's own app, whose records keep the layout
// every record had before apps had identities, so an older binary still finds them.
func (s *Service) legacy(app int64) bool {
	return app == 0 || app == s.chats.DeploymentApp()
}

// recordDir is where one conversation's record lives. A conversation kept in a customer's
// own app is filed under that customer, hex-encoded so no customer id can name a path, and
// out of sight of an older binary that would deliver it into the deployment's app.
func (s *Service) recordDir(customer string, app int64, id string) string {
	if s.legacy(app) {
		return filepath.Join(s.root, id)
	}
	return filepath.Join(s.root, appsDir, hex.EncodeToString([]byte(customer)), id)
}

// SetPins says where to find which app a conversation with no record here was written in.
func (s *Service) SetPins(pins Pins) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pins = pins
}

// pinOf is the app a conversation is kept in: its record's, the deployment's memory of its
// sessions, or for a conversation nobody has word of, the app given.
func (s *Service) pinOf(ctx context.Context, customer, cid string, otherwise int64) (int64, error) {
	s.mu.Lock()
	open, pins := s.all[cid], s.pins
	s.mu.Unlock()
	if open != nil {
		open.mu.Lock()
		owner, app := open.data.Customer, open.data.StreamApp
		open.mu.Unlock()
		if owner == customer {
			return app, nil
		}
	}
	return s.lookupPin(ctx, pins, customer, cid, otherwise)
}

func (s *Service) lookupPin(ctx context.Context, pins Pins, customer, cid string, otherwise int64) (int64, error) {
	if pins != nil {
		app, found, err := pins(ctx, customer, cid)
		if err != nil {
			return 0, err
		}
		if found {
			return app, nil
		}
	}
	return otherwise, nil
}

// clientFor is the client a conversation for a customer is reached with: in the app it is
// kept in, or a new one's in the customer's own.
func (s *Service) clientFor(ctx context.Context, customer, cid string) (*getstream.Stream, int64, error) {
	_, own, err := s.chats.For(ctx, customer)
	if err != nil && !errors.Is(err, streamapp.ErrNoIdentity) {
		return nil, 0, err
	}
	app, err := s.pinOf(ctx, customer, cid, own)
	if err != nil {
		return nil, 0, err
	}
	client, err := s.chats.ForApp(ctx, customer, app)
	return client, app, err
}
