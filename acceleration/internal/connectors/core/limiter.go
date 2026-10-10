package core

import (
	"context"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/redis/rueidis"
)

// limiterPrefix starts every key a Limiter writes, so its keys stay apart from the other
// users of the same Redis (quota, live health, the config cache).
const limiterPrefix = "connectors:rate_limit:"

// callsPrefix starts every key Take counts a customer's direct calls under, apart from the
// blocks under limiterPrefix.
const callsPrefix = "connectors:proxy_calls:"

// MaxBlock is the longest a Limiter holds a key, whatever the Retry-After asked for. No
// provider doc sets it: it bounds what one bad Retry-After can do (a proxy's 429, an epoch
// timestamp sent as seconds), which would otherwise refuse a connector's calls for years. A
// provider still limiting after it answers 429 again and the key is held again, so the router
// holds too little, never too much. The value is unverified: chosen for the wave, Kanat may
// change it.
const MaxBlock = time.Hour

// blockScript sets KEYS[1] to ARGV[1], the end of a block in Unix milliseconds, for ARGV[2]
// milliseconds, unless it already holds a later end. Two routers that see 429s at once keep
// the longer wait, whichever writes last. One script, so the read and the write are atomic.
var blockScript = rueidis.NewLuaScript(`
local held = tonumber(redis.call('GET', KEYS[1]))
if held == nil or held < tonumber(ARGV[1]) then
  redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
end
return 0
`)

// countScript adds one to KEYS[1], the count of one window, and returns the count. The call
// that starts the count sets it to expire after ARGV[1] milliseconds. One script, so no count
// is left without an expiry.
var countScript = rueidis.NewLuaScript(`
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
return count
`)

// Limiter keeps the providers' own rate limits. After a provider answers a call with 429 and
// Retry-After, Block holds the call's key (ResolvedManifest.RateLimitKey) until then, and Wait
// says how long a call on that key must still wait, so it is refused without being sent. The
// block lives in Redis, so every router sharing that Redis refuses the same calls. A Limiter
// never retries a call; it only says when one may go.
//
// Every path fails open, as the daily quota does (internal/quota): a nil Limiter, an empty
// key, an unreachable Redis or an unreadable block refuses nothing. The provider is still the
// one that limits, and answers 429 to whatever gets through.
type Limiter struct {
	redis rueidis.Client
	now   func() time.Time
}

// NewLimiter is a Limiter over client. now is the clock a block's end is measured with; nil
// is time.Now.
func NewLimiter(client rueidis.Client, now func() time.Time) *Limiter {
	if now == nil {
		now = time.Now
	}
	return &Limiter{redis: client, now: now}
}

// Block holds key for wait from now, at most MaxBlock, and reports the wait it held: zero
// when the block was not written. A block already held past then is kept.
func (l *Limiter) Block(ctx context.Context, key string, wait time.Duration) time.Duration {
	if l == nil || key == "" || wait <= 0 {
		return 0
	}
	wait = min(wait, MaxBlock)
	until := l.now().Add(wait).UnixMilli()
	err := blockScript.Exec(ctx, l.redis, []string{key},
		[]string{strconv.FormatInt(until, 10), strconv.FormatInt(wait.Milliseconds(), 10)}).Error()
	if err != nil {
		return 0
	}
	return wait
}

// Wait is how long calls on key must still wait, zero when they may go.
func (l *Limiter) Wait(ctx context.Context, key string) time.Duration {
	if l == nil || key == "" {
		return 0
	}
	until, err := l.redis.Do(ctx, l.redis.B().Get().Key(key).Build()).AsInt64()
	if err != nil {
		return 0
	}
	return max(time.UnixMilli(until).Sub(l.now()), 0)
}

// Take counts one call on key in the window it falls in, and says how long until that window
// ends when the call is over limit, zero when it may go. Windows are fixed: they start at
// multiples of window, so every router sharing the Redis counts the same one. A refused call
// is counted as well, which changes nothing: the window it is in is already over limit.
//
// Like Wait, it fails open: a nil Limiter, an empty key, a limit or window of zero, or an
// unreachable Redis refuses nothing.
//
// Example: limit 60, window a minute. The 61st call of 12:00 gets 25s at 12:00:35, and the
// first call of 12:01 goes.
func (l *Limiter) Take(ctx context.Context, key string, limit int64, window time.Duration) time.Duration {
	if l == nil || key == "" || limit <= 0 || window <= 0 {
		return 0
	}
	now := l.now()
	start := now.Truncate(window)
	// Two windows: the count outlives the window it belongs to whatever the call that started
	// it saw of the clock, and leaves Redis soon after.
	expiry := strconv.FormatInt(2*window.Milliseconds(), 10)
	count, err := countScript.Exec(ctx, l.redis, []string{key + ":" + strconv.FormatInt(start.UnixMilli(), 10)},
		[]string{expiry}).AsInt64()
	if err != nil || count <= limit {
		return 0
	}
	return start.Add(window).Sub(now)
}

// CallsKey is the key Take counts the customer's direct calls to a connector under: one count
// per customer and connector, whichever connection of theirs the call goes through.
func CallsKey(customerID, connectorID string) string {
	// Escaped, so a ":" in an id cannot make two keys one.
	return callsPrefix + url.QueryEscape(customerID) + ":" + url.QueryEscape(connectorID)
}

// RateLimitKey is the key calls on connection c of the customer are limited under, as the
// manifest's rate_limit.per says, or "" when it names no scope: such a connector is never
// limited by the router. Every key holds the customer, so one customer's 429 never refuses
// another's calls.
//
//	app     the customer and the connector. An app the customer registered (customer or
//	        managed) is the customer's own; the operator's app serves only Stream's own agents
//	        (registrationOrder in schemes/oauth2code/client.go), so the customer stands for it.
//	tenant  the value of the manifest's first identity part, from the connection's inputs or
//	        captured metadata: a Slack team_id, a Microsoft tid. Without one, the account.
//	user    the connection's account id, or the connection itself when the provider gives
//	        none.
//
// Example: Slack, per tenant, identity [team_id, user_id]: two people of workspace T1 share
// one key, and a person of T2 has another.
func (m ResolvedManifest) RateLimitKey(customerID string, c Connection) string {
	parts := []string{customerID, m.ConnectorID, string(m.RateLimit.Per)}
	switch m.RateLimit.Per {
	case RateLimitPerApp:
	case RateLimitPerTenant:
		parts = append(parts, m.tenantOf(c)...)
	case RateLimitPerUser:
		parts = append(parts, accountOf(c)...)
	default:
		return ""
	}
	// Escaped, so a ":" in an account id cannot make two keys one.
	for i, part := range parts {
		parts[i] = url.QueryEscape(part)
	}
	return limiterPrefix + strings.Join(parts, ":")
}

// tenantOf is the provider-side tenant c is to: the manifest's first identity part, as the
// connection keeps it (an input or a captured value), or else the account.
func (m ResolvedManifest) tenantOf(c Connection) []string {
	if len(m.Identity) > 0 {
		name := m.Identity[0]
		if value := c.Inputs[name]; value != "" {
			return []string{"tenant", value}
		}
		if value := c.Metadata[name]; value != "" {
			return []string{"tenant", value}
		}
	}
	return accountOf(c)
}

// accountOf is the account c is to, or c itself when the provider gives no account id.
func accountOf(c Connection) []string {
	if c.AccountID != "" {
		return []string{"account", c.AccountID}
	}
	return []string{"connection", c.ID}
}
