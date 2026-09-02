package main

import (
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// limiter throttles each app separately.
//
// Per app rather than per organization on purpose: an account's production
// traffic should not slow down because a script somebody wrote is running hot in
// the same organization.
type limiter struct {
	rate  rate.Limit
	burst int

	mu    sync.Mutex
	apps  map[string]*entry
	sweep time.Time
}

type entry struct {
	limiter *rate.Limiter
	seen    time.Time
}

// idleFor is how long an app's bucket is kept after its last request. Without
// eviction the map grows once per app that ever called and never shrinks.
const idleFor = time.Hour

func newLimiter(perSecond float64, burst int) *limiter {
	return &limiter{
		rate:  rate.Limit(perSecond),
		burst: burst,
		apps:  make(map[string]*entry),
		sweep: time.Now(),
	}
}

// allow reports whether this app may make one more request now.
func (l *limiter) allow(appID string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := time.Now()
	if now.Sub(l.sweep) > idleFor {
		for id, held := range l.apps {
			if now.Sub(held.seen) > idleFor {
				delete(l.apps, id)
			}
		}
		l.sweep = now
	}

	held, ok := l.apps[appID]
	if !ok {
		held = &entry{limiter: rate.NewLimiter(l.rate, l.burst)}
		l.apps[appID] = held
	}
	held.seen = now

	return held.limiter.Allow()
}
