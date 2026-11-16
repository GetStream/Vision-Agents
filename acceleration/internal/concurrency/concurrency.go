// Package concurrency caps how many of one thing a customer may have running at once.
//
// The count lives in Redis, so every router sharing it holds the same count. Each thing
// running holds a lease that ends on its own, so a router that dies mid-run frees its slots
// when their leases run out rather than holding them forever.
package concurrency

import (
	"context"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
)

// releaseTimeout bounds a release, which runs after the request that held the slot may
// already have given up.
const releaseTimeout = 2 * time.Second

// takeScript holds a slot of KEYS[1], a sorted set of leases scored by when each ends in
// Unix milliseconds. It drops the leases that ended before ARGV[1], and adds ARGV[3], ending
// at ARGV[2], unless ARGV[4] are still held. The set lives at least ARGV[5] milliseconds, as
// long as the longest lease in it. One script, so two routers cannot both take the last slot.
var takeScript = rueidis.NewLuaScript(`
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[4]) then
  return 0
end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3])
if redis.call('PTTL', KEYS[1]) < tonumber(ARGV[5]) then
  redis.call('PEXPIRE', KEYS[1], ARGV[5])
end
return 1
`)

// Slots counts what is running under each key against a limit.
//
// Every path fails open, as the daily quota does (internal/quota): a nil Slots, a limit of
// zero or an unreachable Redis refuses nothing. The cap protects the deployment from one
// customer, and refusing everyone because Redis blinked would be the larger outage.
type Slots struct {
	redis rueidis.Client
	now   func() time.Time
}

// New is Slots counted in client. now is the clock leases are measured with; nil is time.Now.
func New(client rueidis.Client, now func() time.Time) *Slots {
	if now == nil {
		now = time.Now
	}
	return &Slots{redis: client, now: now}
}

// Take holds one of limit slots under key for at most lease, and reports whether one was
// free. release gives it back, and is safe to call whatever Take answered.
func (s *Slots) Take(ctx context.Context, key string, limit int64, lease time.Duration) (release func(), taken bool) {
	if s == nil || limit <= 0 {
		return func() {}, true
	}
	now := s.now()
	member := uuid.NewString()
	held, err := takeScript.Exec(ctx, s.redis, []string{key}, []string{
		strconv.FormatInt(now.UnixMilli(), 10),
		strconv.FormatInt(now.Add(lease).UnixMilli(), 10),
		member,
		strconv.FormatInt(limit, 10),
		strconv.FormatInt(lease.Milliseconds(), 10),
	}).AsInt64()
	if err != nil {
		return func() {}, true
	}
	if held == 0 {
		return func() {}, false
	}
	return func() {
		ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), releaseTimeout)
		defer cancel()
		s.redis.Do(ctx, s.redis.B().Zrem().Key(key).Member(member).Build())
	}, true
}
