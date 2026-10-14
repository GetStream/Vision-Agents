// Package quota caps what one end user may spend in a day.
//
// It exists because a token is spent by whoever asks for it, and a customer's own backend
// is trusted with that while a browser holding a token that backend minted is not. Two
// buckets are counted rather than one: a user id, which is the honest answer to "who is
// this", and an address, which is the answer that still holds when a backend mints a fresh
// user id per request.
//
// A day is a UTC calendar day, so a new day is a new key and nothing has to be reset.
package quota

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
)

var tracer = tracing.Tracer("quota")

// retention outlives a day by enough that a counter is still there for the whole of the day
// it belongs to, whatever the clock does at the boundary, and is gone long before it could
// be confused with the same date a year later.
const retention = 48 * time.Hour

// writeTimeout bounds a debit, because it is taken on the goroutine draining a response and
// a slow Redis should cost that goroutine a moment rather than the rest of the answer.
const writeTimeout = 2 * time.Second

const (
	messagesField = "messages"
	tokensField   = "tokens"
)

// ErrExhausted is what a caller with nothing left to spend is refused with. It is one error
// for both buckets and both limits, with the limit that was reached named in the message.
var ErrExhausted = errors.New("quota: the daily limit is spent")

// Limits is what one caller may spend in a day. A limit of zero is not enforced, which is
// how a deployment turns one half off without turning the other off with it.
type Limits struct {
	// MessagesPerDay is how many responses a caller may ask for.
	MessagesPerDay int64
	// TokensPerDay is the backstop under the message count, for the caller who asks for
	// few responses and makes each one enormous.
	TokensPerDay int64
}

// Enforced reports whether these limits constrain anything.
func (l Limits) Enforced() bool { return l.MessagesPerDay > 0 || l.TokensPerDay > 0 }

// Source answers which limits one customer's users are held to. It is given the
// deployment's own, which are what a customer that decided nothing is held to.
type Source interface {
	Limits(ctx context.Context, customerID string, defaults Limits) Limits
}

// Status is what a caller has left of the limit nearest to refusing them. A zero Status is
// a caller nothing was counted for.
type Status struct {
	// Limit is the day's allowance, in messages or in tokens.
	Limit int64
	// Remaining is what is left of it, never below zero.
	Remaining int64
	// Reset is when the day ends and the count starts again: the next UTC midnight.
	Reset time.Time
}

// Limiter counts what a caller has spent today and refuses them once it is gone.
//
// Every path through it fails open: an unreachable Redis, an unreadable counter or a nil
// Limiter all allow the request. Redis is optional in this deployment, and a limiter that
// failed closed would turn a cache blip into an outage of the thing it is protecting.
type Limiter struct {
	redis    rueidis.Client
	defaults Limits
	source   Source
	logger   *slog.Logger
}

// New returns a Limiter counting against the limits source gives each customer, or against
// defaults for every customer when source is nil. Defaults of zero are allowed, since a
// customer may still set limits of its own; a customer held to none is not counted.
func New(client rueidis.Client, defaults Limits, source Source, logger *slog.Logger) (*Limiter, error) {
	if client == nil {
		return nil, errors.New("quota: a redis client is required")
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Limiter{redis: client, defaults: defaults, source: source, logger: logger}, nil
}

// Allow reports whether the caller may ask for another response, and what they have left
// of the limit nearest to refusing them. The Status is returned with a refusal too.
//
// A caller with nothing to count against is allowed: that is a customer's own backend
// working for itself, which is trusted with its own tokens.
func (l *Limiter) Allow(ctx context.Context, customerID string, caller routing.Caller) (Status, error) {
	if l == nil {
		return Status{}, nil
	}
	// One clock reading names the day counted and the midnight it resets at.
	now := time.Now()
	keys := l.keys(customerID, caller, now)
	if len(keys) == 0 {
		return Status{}, nil
	}

	ctx, span := tracer.Start(ctx, "quota.allow")
	defer span.End()

	limits := l.limits(ctx, customerID)
	if !limits.Enforced() {
		return Status{}, nil
	}
	reset := nextMidnight(now)
	var nearest Status
	for _, key := range keys {
		spent, err := l.spent(ctx, key)
		if err != nil {
			l.logger.Error("could not read a quota, allowing the request", "error", err)
			return Status{}, nil
		}
		nearest = closer(nearest, limits.MessagesPerDay, spent.messages, reset)
		nearest = closer(nearest, limits.TokensPerDay, spent.tokens, reset)
		if limits.MessagesPerDay > 0 && spent.messages >= limits.MessagesPerDay {
			return nearest, stack.Wrap(fmt.Errorf("%w: %d messages a day", ErrExhausted, limits.MessagesPerDay))
		}
		if limits.TokensPerDay > 0 && spent.tokens >= limits.TokensPerDay {
			return nearest, stack.Wrap(fmt.Errorf("%w: %d tokens a day", ErrExhausted, limits.TokensPerDay))
		}
	}
	return nearest, nil
}

// Debit records one response and what it cost against every bucket the caller falls in.
//
// It reports nothing, because it is called once a response has already been generated and
// there is no longer a decision left for the answer to inform. A debit that does not land
// is logged and the caller keeps the tokens.
func (l *Limiter) Debit(ctx context.Context, customerID string, caller routing.Caller, tokens int64) {
	if l == nil {
		return
	}
	keys := l.keys(customerID, caller, time.Now())
	if len(keys) == 0 {
		return
	}
	if !l.limits(ctx, customerID).Enforced() {
		return
	}

	ctx, cancel := context.WithTimeout(ctx, writeTimeout)
	defer cancel()

	seconds := int64(retention.Seconds())
	commands := make([]rueidis.Completed, 0, len(keys)*3)
	for _, key := range keys {
		commands = append(commands,
			l.redis.B().Hincrby().Key(key).Field(messagesField).Increment(1).Build(),
			l.redis.B().Hincrby().Key(key).Field(tokensField).Increment(tokens).Build(),
			l.redis.B().Expire().Key(key).Seconds(seconds).Build(),
		)
	}

	for _, response := range l.redis.DoMulti(ctx, commands...) {
		if err := response.Error(); err != nil {
			l.logger.Error("could not record what a caller spent", "error", err)
			return
		}
	}
}

// spend is what one bucket has been debited today.
type spend struct {
	messages int64
	tokens   int64
}

// spent reads one bucket. A bucket nobody has spent from reads as zero rather than as an
// error, since the first request of a day is the common case.
func (l *Limiter) spent(ctx context.Context, key string) (spend, error) {
	entries, err := l.redis.Do(ctx, l.redis.B().Hgetall().Key(key).Build()).AsStrMap()
	if err != nil {
		return spend{}, stack.Wrap(fmt.Errorf("quota: read %s: %w", key, err))
	}
	return spend{
		messages: parseInt(entries[messagesField]),
		tokens:   parseInt(entries[tokensField]),
	}, nil
}

// limits are what the customer's users are held to: what the source says, or the
// deployment's own when there is no source.
func (l *Limiter) limits(ctx context.Context, customerID string) Limits {
	if l.source == nil {
		return l.defaults
	}
	return l.source.Limits(ctx, customerID, l.defaults)
}

// keys are the buckets a caller falls in, which is none for one there is nothing to count
// against and none for a customer's own process, which is trusted with its spend even
// when it names a user. Both buckets are keyed by the customer as well: a user id is the
// customer's own name for somebody, and each customer sets its own limit for an address.
func (l *Limiter) keys(customerID string, caller routing.Caller, now time.Time) []string {
	if caller.ServerSide {
		return nil
	}
	day := now.UTC().Format("20060102")

	var keys []string
	if customerID != "" && caller.UserID != "" {
		keys = append(keys, "quota:"+day+":user:"+customerID+":"+caller.UserID)
	}
	if caller.IP != "" {
		keys = append(keys, "quota:"+day+":ip:"+customerID+":"+caller.IP)
	}
	return keys
}

// closer returns whichever of nearest and one limit's standing has the smaller share left,
// so a caller is told about the limit that will refuse them first. A limit of zero is not
// enforced and is never the nearest.
func closer(nearest Status, limit, spent int64, reset time.Time) Status {
	if limit <= 0 {
		return nearest
	}
	candidate := Status{Limit: limit, Remaining: max(limit-spent, 0), Reset: reset}
	if nearest.Limit == 0 || share(candidate) < share(nearest) {
		return candidate
	}
	return nearest
}

// share is the part of a limit that is left, from 0 to 1.
func share(status Status) float64 { return float64(status.Remaining) / float64(status.Limit) }

// nextMidnight is when the UTC day containing now ends.
func nextMidnight(now time.Time) time.Time {
	return now.UTC().Truncate(24 * time.Hour).Add(24 * time.Hour)
}

func parseInt(value string) int64 {
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0
	}
	return parsed
}
