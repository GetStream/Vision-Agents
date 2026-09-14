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
)

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

// Limiter counts what a caller has spent today and refuses them once it is gone.
//
// Every path through it fails open: an unreachable Redis, an unreadable counter or a nil
// Limiter all allow the request. Redis is optional in this deployment, and a limiter that
// failed closed would turn a cache blip into an outage of the thing it is protecting.
type Limiter struct {
	redis  rueidis.Client
	limits Limits
	logger *slog.Logger
}

// New returns a Limiter counting against the given limits.
func New(client rueidis.Client, limits Limits, logger *slog.Logger) (*Limiter, error) {
	if client == nil {
		return nil, errors.New("quota: a redis client is required")
	}
	if !limits.Enforced() {
		return nil, errors.New("quota: at least one of the daily limits must be set")
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &Limiter{redis: client, limits: limits, logger: logger}, nil
}

// Allow reports whether the caller may ask for another response.
//
// A caller with nothing to count against is allowed: that is a customer's own backend
// working for itself, which is trusted with its own tokens.
func (l *Limiter) Allow(ctx context.Context, customerID string, caller routing.Caller) error {
	if l == nil {
		return nil
	}

	for _, key := range l.keys(customerID, caller) {
		spent, err := l.spent(ctx, key)
		if err != nil {
			l.logger.Error("could not read a quota, allowing the request", "error", err)
			return nil
		}
		if l.limits.MessagesPerDay > 0 && spent.messages >= l.limits.MessagesPerDay {
			return fmt.Errorf("%w: %d messages a day", ErrExhausted, l.limits.MessagesPerDay)
		}
		if l.limits.TokensPerDay > 0 && spent.tokens >= l.limits.TokensPerDay {
			return fmt.Errorf("%w: %d tokens a day", ErrExhausted, l.limits.TokensPerDay)
		}
	}
	return nil
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
	keys := l.keys(customerID, caller)
	if len(keys) == 0 {
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
		return spend{}, fmt.Errorf("quota: read %s: %w", key, err)
	}
	return spend{
		messages: parseInt(entries[messagesField]),
		tokens:   parseInt(entries[tokensField]),
	}, nil
}

// keys are the buckets a caller falls in, which is none for one there is nothing to count
// against. The user bucket is keyed by the customer as well, because a user id is the
// customer's own name for somebody and two customers may use the same one.
func (l *Limiter) keys(customerID string, caller routing.Caller) []string {
	day := time.Now().UTC().Format("20060102")

	var keys []string
	if customerID != "" && caller.UserID != "" {
		keys = append(keys, "quota:"+day+":user:"+customerID+":"+caller.UserID)
	}
	if caller.IP != "" {
		keys = append(keys, "quota:"+day+":ip:"+caller.IP)
	}
	return keys
}

func parseInt(value string) int64 {
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0
	}
	return parsed
}
