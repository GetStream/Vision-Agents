// Package concurrency caps how many of one thing a customer may have running at once.
//
// The count lives in Redis, so every router sharing it holds the same count. Each thing
// running holds a lease that ends on its own, so a router that dies mid-run frees its slots
// when their leases run out rather than holding them forever.
package concurrency

import (
	"context"
	"strconv"
	"sync"
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

// renewScript moves the end of ARGV[2]'s lease in KEYS[1] to ARGV[1], unless it ended and
// was dropped already, and keeps the set alive at least ARGV[3] milliseconds.
var renewScript = rueidis.NewLuaScript(`
if not redis.call('ZSCORE', KEYS[1], ARGV[2]) then
  return 0
end
redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2])
if redis.call('PTTL', KEYS[1]) < tonumber(ARGV[3]) then
  redis.call('PEXPIRE', KEYS[1], ARGV[3])
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
	member, taken := s.take(ctx, key, limit, lease)
	if member == "" {
		return func() {}, taken
	}
	return func() { s.release(ctx, key, member) }, true
}

// Held is a slot Hold took, kept until Release. A nil Held holds nothing.
type Held struct {
	slots       *Slots
	key, member string
	lease       time.Duration
	stop, done  chan struct{}
	once        sync.Once
}

// Hold takes one of limit slots under key as Take does, for work with no end it can name: the
// lease is renewed every third of it until Release, so the slot is held as long as the work
// runs, and a router that dies frees it once lease has passed. It answers a nil Held, which
// holds nothing, where Take holds nothing.
func (s *Slots) Hold(ctx context.Context, key string, limit int64, lease time.Duration) (*Held, bool) {
	member, taken := s.take(ctx, key, limit, lease)
	if member == "" {
		return nil, taken
	}
	held := &Held{slots: s, key: key, member: member, lease: lease, stop: make(chan struct{}), done: make(chan struct{})}
	go held.keep(context.WithoutCancel(ctx))
	return held, true
}

func (h *Held) keep(ctx context.Context) {
	defer close(h.done)
	ticker := time.NewTicker(h.lease / 3)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			h.slots.renew(ctx, h.key, h.member, h.lease)
		case <-h.stop:
			return
		}
	}
}

// Release stops renewing the slot and gives it back. Calling it twice gives it back once.
func (h *Held) Release() {
	if h == nil {
		return
	}
	h.once.Do(func() {
		close(h.stop)
		<-h.done
		h.slots.release(context.Background(), h.key, h.member)
	})
}

// take holds one of limit slots under key for lease, answering the lease's member. Empty is no
// slot held: none was free (taken false), or nothing is counted (taken true).
func (s *Slots) take(ctx context.Context, key string, limit int64, lease time.Duration) (member string, taken bool) {
	if s == nil || limit <= 0 {
		return "", true
	}
	now := s.now()
	member = uuid.NewString()
	held, err := takeScript.Exec(ctx, s.redis, []string{key}, []string{
		strconv.FormatInt(now.UnixMilli(), 10),
		strconv.FormatInt(now.Add(lease).UnixMilli(), 10),
		member,
		strconv.FormatInt(limit, 10),
		strconv.FormatInt(lease.Milliseconds(), 10),
	}).AsInt64()
	if err != nil {
		return "", true
	}
	if held == 0 {
		return "", false
	}
	return member, true
}

// renew moves the end of member's lease to lease from now. One that fails lets the lease run
// out, which frees the slot, as an unreachable Redis does for Take.
func (s *Slots) renew(ctx context.Context, key, member string, lease time.Duration) {
	ctx, cancel := context.WithTimeout(ctx, releaseTimeout)
	defer cancel()
	renewScript.Exec(ctx, s.redis, []string{key}, []string{
		strconv.FormatInt(s.now().Add(lease).UnixMilli(), 10),
		member,
		strconv.FormatInt(lease.Milliseconds(), 10),
	})
}

func (s *Slots) release(ctx context.Context, key, member string) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), releaseTimeout)
	defer cancel()
	s.redis.Do(ctx, s.redis.B().Zrem().Key(key).Member(member).Build())
}
