package conversation

import (
	"context"
	"encoding/hex"
	"errors"
	"os"
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
// at all, or can only be read. Its writes stay on disk and are tried again much later,
// rather than delivered anywhere else.
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
	_, own, err := s.chats.For(ctx, customer)
	tolerated := errors.Is(err, streamapp.ErrNoIdentity) || (reading && errors.Is(err, streamapp.ErrStreamAppDisconnected))
	if err != nil && !tolerated {
		return 0, err
	}
	return s.pinOf(ctx, customer, cid, own)
}

// AppOf is the app a conversation is kept in, for a caller deciding whether a session in
// another app may be bound to it.
func (s *Service) AppOf(ctx context.Context, customer, cid string) (int64, error) {
	return s.appOf(ctx, customer, cid, true)
}

// LegacyRecords counts, by customer, the conversations recorded in the outbox at root in
// the layout every record had before apps had identities: those kept in the deployment's own
// app. A record that cannot be read is counted under the empty customer.
func LegacyRecords(root string) (map[string]int, error) {
	entries, err := os.ReadDir(root)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]int{}, nil
	}
	if err != nil {
		return nil, err
	}
	counted := map[string]int{}
	for _, entry := range entries {
		if !entry.IsDir() || !validID.MatchString(entry.Name()) {
			continue
		}
		record, err := loadDisk(filepath.Join(root, entry.Name()))
		if err != nil {
			counted[""]++
			continue
		}
		counted[record.Customer]++
	}
	return counted, nil
}
