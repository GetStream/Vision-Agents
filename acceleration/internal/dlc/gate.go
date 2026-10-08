package dlc

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// retention outlives a day by enough that a counter lasts the whole of its day, as in quota.
const retention = 48 * time.Hour

// writeTimeout bounds a count taken after the text or call already happened.
const writeTimeout = 2 * time.Second

const (
	messagesField = "messages"
	secondsField  = "audio_seconds"
)

// Sandbox is what an app may do before any use case of its is approved.
type Sandbox struct {
	// Enabled turns the sandbox on. Off, which a self-hosted router is, only opt-outs are
	// enforced: whether to register is then the operator's own business.
	Enabled bool
	// Recipients is how many numbers an app may text and call.
	Recipients int
	// MessagesPerDay and AudioMinutesPerDay are what it may send and talk in a UTC day.
	MessagesPerDay     int64
	AudioMinutesPerDay int64
}

// Gate decides whether a text or call may go out. A nil Gate lets everything through.
//
// The opt-out check fails closed, because reaching somebody who said stop is the one thing
// a carrier will not forgive. The daily counters fail open, as quota's do: an unreachable
// Redis should not stop an app from testing.
type Gate struct {
	store   *store.Store
	redis   rueidis.Client
	sandbox Sandbox
	logger  *slog.Logger
}

// NewGate returns a Gate. redis may be nil, which leaves the daily limits uncounted.
func NewGate(store *store.Store, redis rueidis.Client, sandbox Sandbox, logger *slog.Logger) *Gate {
	if logger == nil {
		logger = slog.Default()
	}
	return &Gate{store: store, redis: redis, sandbox: sandbox, logger: logger}
}

// Usage is where a sandboxed app stands today.
type Usage struct {
	Sandboxed    bool
	Recipients   []string
	Messages     int64
	AudioSeconds int64
	Limits       Sandbox
}

// Allow reports whether an app may text or call to on channel, as an error wrapping
// ErrRefused when it may not.
func (g *Gate) Allow(ctx context.Context, customerID, channel, to string) error {
	if g == nil || g.store == nil {
		return nil
	}
	optedOut, err := g.store.OptedOut(ctx, customerID, to, channel)
	if err != nil {
		return err
	}
	if optedOut {
		return stack.Wrap(fmt.Errorf("%w: %s opted out of %s", ErrRefused, to, channel))
	}

	sandboxed, err := g.sandboxed(ctx, customerID)
	if err != nil || !sandboxed {
		return err
	}
	recipients, err := g.store.SandboxRecipients(ctx, customerID)
	if err != nil {
		return err
	}
	if !slices.Contains(recipients, to) {
		return stack.Wrap(fmt.Errorf("%w: until a use case is approved, only sandbox recipients can be reached, and %s is not one", ErrRefused, to))
	}
	spent := g.spent(ctx, customerID)
	if channel == Voice {
		if limit := g.sandbox.AudioMinutesPerDay; limit > 0 && spent.AudioSeconds >= limit*60 {
			return stack.Wrap(fmt.Errorf("%w: the sandbox allows %d minutes of calls a day", ErrRefused, limit))
		}
		return nil
	}
	if limit := g.sandbox.MessagesPerDay; limit > 0 && spent.Messages >= limit {
		return stack.Wrap(fmt.Errorf("%w: the sandbox allows %d messages a day", ErrRefused, limit))
	}
	return nil
}

// Sent counts one message a sandboxed app sent.
func (g *Gate) Sent(ctx context.Context, customerID string) {
	g.count(ctx, customerID, messagesField, 1)
}

// Talked counts how long a sandboxed app's call lasted.
func (g *Gate) Talked(ctx context.Context, customerID string, lasted time.Duration) {
	g.count(ctx, customerID, secondsField, int64(lasted.Seconds()))
}

// Usage reports where an app stands against the sandbox today.
func (g *Gate) Usage(ctx context.Context, customerID string) (Usage, error) {
	if g == nil || g.store == nil {
		return Usage{Recipients: []string{}}, nil
	}
	sandboxed, err := g.sandboxed(ctx, customerID)
	if err != nil {
		return Usage{}, err
	}
	recipients, err := g.store.SandboxRecipients(ctx, customerID)
	if err != nil {
		return Usage{}, err
	}
	usage := g.spent(ctx, customerID)
	usage.Sandboxed, usage.Recipients, usage.Limits = sandboxed, recipients, g.sandbox
	return usage, nil
}

// SetRecipients replaces the numbers an app may reach while sandboxed.
func (g *Gate) SetRecipients(ctx context.Context, customerID string, recipients []string) error {
	if g.sandbox.Recipients > 0 && len(recipients) > g.sandbox.Recipients {
		return stack.Wrap(fmt.Errorf("%w: the sandbox allows %d recipients", ErrInvalid, g.sandbox.Recipients))
	}
	return g.store.SetSandboxRecipients(ctx, customerID, recipients)
}

// sandboxed reports whether an app is held to the sandbox: it is on, and none of the
// app's use cases is approved.
func (g *Gate) sandboxed(ctx context.Context, customerID string) (bool, error) {
	if !g.sandbox.Enabled {
		return false, nil
	}
	approved, err := g.store.HasUseCaseIn(ctx, customerID, Approved)
	return !approved, err
}

func (g *Gate) count(ctx context.Context, customerID, field string, by int64) {
	if g == nil || g.redis == nil || g.store == nil || by <= 0 {
		return
	}
	if sandboxed, err := g.sandboxed(ctx, customerID); err != nil || !sandboxed {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, writeTimeout)
	defer cancel()
	key := g.key(customerID)
	for _, response := range g.redis.DoMulti(ctx,
		g.redis.B().Hincrby().Key(key).Field(field).Increment(by).Build(),
		g.redis.B().Expire().Key(key).Seconds(int64(retention.Seconds())).Build(),
	) {
		if err := response.Error(); err != nil {
			g.logger.Error("could not count what a sandboxed app sent", "error", err)
			return
		}
	}
}

// spent reads today's counters. Any trouble reads as nothing spent.
func (g *Gate) spent(ctx context.Context, customerID string) Usage {
	if g.redis == nil {
		return Usage{}
	}
	entries, err := g.redis.Do(ctx, g.redis.B().Hgetall().Key(g.key(customerID)).Build()).AsStrMap()
	if err != nil {
		g.logger.Error("could not read the sandbox counters, allowing", "error", err)
		return Usage{}
	}
	return Usage{Messages: parseInt(entries[messagesField]), AudioSeconds: parseInt(entries[secondsField])}
}

// key is the app's counter for today, a new key each UTC day so nothing is ever reset.
func (g *Gate) key(customerID string) string {
	return "dlc:" + time.Now().UTC().Format("20060102") + ":" + customerID
}

func parseInt(value string) int64 {
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0
	}
	return parsed
}
