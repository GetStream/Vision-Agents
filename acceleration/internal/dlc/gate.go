package dlc

import (
	"cmp"
	"context"
	"fmt"
	"log/slog"
	"math"
	"slices"
	"strconv"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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
// Every check fails closed: the opt-out, because reaching somebody who said stop is the one
// thing a carrier will not forgive, and the sandbox and the channel limits, because a text or
// a minute let through unchecked is reach and money nobody agreed to. A gate with no Redis
// keeps no daily count, and so holds nobody to one.
type Gate struct {
	store   *store.Store
	redis   rueidis.Client
	sandbox Sandbox
	money   Money
	// limited has every organization kept to its tier's channel limits, as mode says.
	limited bool
	mode    config.LimitsMode
	logs    *throttle
	logger  *slog.Logger
}

// Money is what says whether a customer's organization may spend money now: the admission
// every routed request asks (policy.Enforcer). Its error refuses everything, as a suspension
// does; its Spend refuses work that costs money, which every text, call and number does. Its
// Limits are the channel limits the organization's tier sets. Attribute names the organization
// a text or a minute already sent is counted against.
type Money interface {
	Admit(ctx context.Context, owner routing.Owner) (routing.Admission, error)
	Attribute(ctx context.Context, customerID string) (routing.Admission, error)
}

// Charging has the gate refuse what money refuses too. It is set once, before the gate is used.
func (g *Gate) Charging(money Money) *Gate {
	if g != nil {
		g.money = money
	}
	return g
}

// Limiting has the gate keep every organization to the channel limits its tier and overrides
// set (usage.ChannelLimits), read from the admission Charging set: in enforce they refuse, in
// shadow they log "would refuse" and allow, and off they are neither checked nor counted. Opt-
// outs and the sandbox are kept in every mode. It is set once, before the gate is used.
func (g *Gate) Limiting(mode config.LimitsMode) *Gate {
	if g != nil {
		g.limited, g.mode = true, cmp.Or(mode, config.LimitsEnforce)
	}
	return g
}

// Admit refuses a customer whose organization may not pay for telephony: suspended, out of
// credit or past a spend limit. With no Money set it refuses nobody. Whatever a vendor charges
// for is billed work, so a refusal of money refuses it, unlike a model that costs nothing.
func (g *Gate) Admit(ctx context.Context, customerID string) error {
	_, err := g.admit(ctx, customerID)
	return err
}

// admit is Admit, with the admission it was answered from.
func (g *Gate) admit(ctx context.Context, customerID string) (routing.Admission, error) {
	if g == nil || g.money == nil {
		return routing.Admission{}, nil
	}
	// The key, user and categories are the context's, when the work carries them.
	admission, err := g.money.Admit(ctx, routing.Owner{CustomerID: customerID, Attribution: routing.AttributionFrom(ctx)})
	if err == nil {
		err = admission.Spend
	}
	if err == nil {
		return admission, nil
	}
	return admission, stack.Wrap(fmt.Errorf("%w: %w", ErrUnpaid, err))
}

// NewGate returns a Gate. redis may be nil, which leaves the daily limits uncounted.
func NewGate(store *store.Store, redis rueidis.Client, sandbox Sandbox, logger *slog.Logger) *Gate {
	if logger == nil {
		logger = slog.Default()
	}
	return &Gate{store: store, redis: redis, sandbox: sandbox, logs: &throttle{}, logger: logger}
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
// ErrRefused when it may not, or the refusal of money Admit answers. The order is money, the
// opt-out, the sandbox, then the organization's channel limits (Limiting).
func (g *Gate) Allow(ctx context.Context, customerID, channel, to string) error {
	return g.allow(ctx, customerID, channel, to, Call{})
}

// Call is a Stream call, by its type and id.
type Call struct {
	Type, ID string
}

// AllowTransfer is Allow for a transfer of the call into to: the leg being transferred is
// already up and is not counted against the calls allowed at once, so the human's new leg is.
func (g *Gate) AllowTransfer(ctx context.Context, customerID, to string, into Call) error {
	return g.allow(ctx, customerID, Voice, to, into)
}

func (g *Gate) allow(ctx context.Context, customerID, channel, to string, into Call) error {
	admission, err := g.admit(ctx, customerID)
	if err != nil {
		return err
	}
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
	if err := g.inSandbox(ctx, customerID, channel, to); err != nil {
		return err
	}
	return g.withinTier(ctx, admission, customerID, channel, to, into)
}

// inSandbox refuses what the sandbox does not allow an app none of whose use cases is approved.
func (g *Gate) inSandbox(ctx context.Context, customerID, channel, to string) error {
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
	spent, err := g.spent(ctx, customerID)
	if err != nil {
		return err
	}
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

// Sent counts one message an app sent to on channel: against the sandbox while it holds the
// app, and against its organization's channel limits whatever its use cases.
func (g *Gate) Sent(ctx context.Context, customerID, channel, to string) {
	g.count(ctx, customerID, messagesField, 1)
	g.countChannel(ctx, customerID, "", channel, to, 1)
}

// Dialed counts the person a call or a transfer of an app's in organizationID rings as reached
// today, against the organization's voice limit on recipients. What the call lasts is Talked's.
func (g *Gate) Dialed(ctx context.Context, customerID, organizationID, to string) {
	g.countChannel(ctx, customerID, organizationID, Voice, to, 0)
}

// Talked counts how long a leg of an app's call in organizationID lasted: its seconds against
// the sandbox while it holds the app, and for an outbound leg its minutes, rounded up, against
// the organization's voice limit a day. Calls people make to the app's numbers are not what
// that limit holds back, so they never use it up.
func (g *Gate) Talked(ctx context.Context, customerID, organizationID string, outbound bool, lasted time.Duration) {
	g.count(ctx, customerID, secondsField, int64(lasted.Seconds()))
	if minutes := int64(math.Ceil(lasted.Minutes())); outbound && minutes > 0 {
		g.countChannel(ctx, customerID, organizationID, Voice, "", minutes)
	}
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
	usage, err := g.spent(ctx, customerID)
	if err != nil {
		return Usage{}, err
	}
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

// Sandboxed reports whether the sandbox holds an app now, which is what gives its texts and
// minutes away. A nil Gate holds nobody, and neither does one that could not tell.
func (g *Gate) Sandboxed(ctx context.Context, customerID string) (bool, error) {
	if g == nil || g.store == nil {
		return false, nil
	}
	return g.sandboxed(ctx, customerID)
}

// sandboxed reports whether an app is held to the sandbox: it is on, and none of the
// app's use cases is approved.
func (g *Gate) sandboxed(ctx context.Context, customerID string) (bool, error) {
	if !g.sandbox.Enabled {
		return false, nil
	}
	approved, err := g.store.HasUseCaseIn(ctx, customerID, Approved)
	if err != nil {
		return false, err
	}
	return !approved, nil
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

// spent reads today's counters, none without Redis.
func (g *Gate) spent(ctx context.Context, customerID string) (Usage, error) {
	if g.redis == nil {
		return Usage{}, nil
	}
	entries, err := g.redis.Do(ctx, g.redis.B().Hgetall().Key(g.key(customerID)).Build()).AsStrMap()
	if err != nil {
		return Usage{}, stack.Wrap(fmt.Errorf("dlc: read the sandbox counters: %w", err))
	}
	return Usage{Messages: parseInt(entries[messagesField]), AudioSeconds: parseInt(entries[secondsField])}, nil
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
