package routing

import (
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// How a stream's lease is kept: LeaseTTL long, renewed every LeaseRenew, so a process that
// dies holds its streams' money at most LeaseTTL longer.
const (
	LeaseTTL   = 90 * time.Second
	LeaseRenew = 30 * time.Second
)

// Lease sets aside what a minute of a stream may cost while it runs, so the other routers
// count a stream in progress before its units are recorded: one hold under the stream's own
// usage id, renewed every LeaseRenew and let go when the stream ends. Each unit the stream
// records settles a hold of its own. A renewal that is refused keeps the stream going, and the
// last renewal's hold expires on its own; Err says so. A nil Lease holds nothing.
type Lease struct {
	recorder *Recorder
	config   ProviderConfig
	owner    Owner
	id       uuid.UUID
	minute   pricing.Quantities
	every    time.Duration
	ttl      time.Duration

	mu      sync.Mutex
	refused error

	stop     chan struct{}
	done     chan struct{}
	stopOnce sync.Once
}

// StartLease reserves a minute of the stream named kind and id, as minute says it uses, on the
// provider for its owner, and keeps renewing it until Stop. A recorder that holds nothing, or a
// provider that bills nothing, leases nothing.
func StartLease(recorder *Recorder, config ProviderConfig, owner Owner, kind, id string, minute pricing.Quantities) *Lease {
	return StartLeaseEvery(recorder, config, owner, kind, id, minute, LeaseRenew, LeaseTTL)
}

// StartLeaseEvery is StartLease renewing every every, for ttl each time.
func StartLeaseEvery(recorder *Recorder, config ProviderConfig, owner Owner, kind, id string, minute pricing.Quantities, every, ttl time.Duration) *Lease {
	if recorder == nil || !recorder.writer.records() || !recorder.Billable(config) {
		return nil
	}
	if _, holds := recorder.moneyGate().(Reserver); !holds {
		return nil
	}
	l := &Lease{
		recorder: recorder, config: config, owner: owner, id: UsageID("lease", kind, id), minute: minute,
		every: every, ttl: ttl, stop: make(chan struct{}), done: make(chan struct{}),
	}
	l.renew()
	go l.keep()
	return l
}

func (l *Lease) keep() {
	defer close(l.done)
	ticker := time.NewTicker(l.every)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			l.renew()
		case <-l.stop:
			return
		}
	}
}

func (l *Lease) renew() {
	err := l.recorder.Reserve(l.config, l.owner, l.id, l.minute, l.ttl)
	l.mu.Lock()
	l.refused = err
	l.mu.Unlock()
}

// ID is the usage id the lease is held under.
func (l *Lease) ID() uuid.UUID {
	if l == nil {
		return uuid.Nil
	}
	return l.id
}

// Err is why the last renewal was refused, nil while the lease is held.
func (l *Lease) Err() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.refused
}

// Stop ends the lease and lets go of its hold. Calling it twice lets go once.
func (l *Lease) Stop() {
	if l == nil {
		return
	}
	l.stopOnce.Do(func() {
		close(l.stop)
		<-l.done
		l.recorder.Release(l.id)
	})
}
