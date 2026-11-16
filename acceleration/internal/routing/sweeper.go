package routing

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

const (
	defaultSweepEvery = 30 * time.Second
	// defaultSettle leaves the rows the writer's shipper is about to send alone. It is counted
	// from when a row's work started, which is all a row says: a row for long work is swept as
	// soon as it is written, beside the shipper, and the service records it once.
	defaultSettle = 30 * time.Second
	// defaultSweepTimeout bounds one sweep, so a holder whose database stopped answering lets
	// go of the lock rather than keep it while nobody sweeps. A backlog bigger than one sweep
	// sends is carried on by the next.
	defaultSweepTimeout = 2 * time.Minute
	// lockTimeout bounds taking or checking the lock, and countTimeout counting what waits.
	lockTimeout  = 10 * time.Second
	countTimeout = 10 * time.Second
	// sweepPage is how many rows one page of a sweep reads and ships.
	sweepPage = usage.MaxEvents
	// unownedLogEvery spaces the line naming an app whose rows wait for an organization.
	unownedLogEvery = time.Hour
)

// Sweeper ships the rows of this router's shard the writer's shipper did not: the usage
// service was down, the shipper had no room, the row had no organization yet, or the process
// stopped first. Every router with a usage service runs one, and the one holding the shard's
// ship lock does the work; the others try the lock again at the next tick. Every one counts
// what waits, so a holder that is stuck does not hide that nothing ships.
type Sweeper struct {
	sending *shipping
	store   *store.Store
	lock    *store.ShipLock
	logger  *slog.Logger
	every   time.Duration
	settle  time.Duration
	timeout time.Duration

	// unshipped and oldest are what the last count found waiting in the shard, in rows and
	// nanoseconds since the oldest started.
	unshipped atomic.Int64
	oldest    atomic.Int64

	// errorLogged is when a failed sweep was last logged, in Unix nanoseconds, and
	// unownedLogged when each app whose rows have no organization was. Only the sweeping
	// goroutine reads unownedLogged.
	errorLogged   atomic.Int64
	unownedLogged map[string]time.Time
}

// SweeperOptions configure a Sweeper. Store and Usage are required.
type SweeperOptions struct {
	Store *store.Store
	Usage *usage.Client
	// Source names this process in the events it ships (see Source).
	Source string
	// Shipped is told which rows the usage service has, a row possibly more than once (see
	// Acked). Optional.
	Shipped func(acked []Acked)
	Logger  *slog.Logger
	// Every is how often the lock is tried and the shard swept; defaults to 30 s.
	Every time.Duration
	// Settle is how long ago a row must have started for the sweeper to send it, leaving the
	// newer ones to the writer's shipper; defaults to 30 s, and a negative one sends every row
	// however new.
	Settle time.Duration
	// Timeout bounds one sweep; defaults to 2 minutes.
	Timeout time.Duration
}

// NewSweeper returns a sweeper of the store's shard; Run starts it.
func NewSweeper(options SweeperOptions) *Sweeper {
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	every := options.Every
	if every <= 0 {
		every = defaultSweepEvery
	}
	settle := options.Settle
	if settle < 0 {
		settle = 0
	} else if settle == 0 {
		settle = defaultSettle
	}
	timeout := options.Timeout
	if timeout <= 0 {
		timeout = defaultSweepTimeout
	}
	return &Sweeper{
		sending: &shipping{
			store: options.Store, usage: options.Usage, source: options.Source, shipped: options.Shipped, logger: logger,
		},
		store:         options.Store,
		lock:          options.Store.ShipLock(),
		logger:        logger,
		every:         every,
		settle:        settle,
		timeout:       timeout,
		unownedLogged: map[string]time.Time{},
	}
}

// Run sweeps at once and then every interval until ctx is done, and then gives the lock back.
func (s *Sweeper) Run(ctx context.Context) {
	sweepers.Store(s, struct{}{})
	defer sweepers.Delete(s)
	defer s.lock.Release()
	ticker := time.NewTicker(s.every)
	defer ticker.Stop()
	for {
		if _, err := s.sweep(ctx); err != nil && ctx.Err() == nil {
			s.logError(err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// sweep counts what waits in the shard, and ships it when this sweeper holds the shard's lock,
// reporting whether it does. A holder whose sweep fails lets the lock go, so that a router that
// can sweep takes it at its next tick. One that ran out of time after marking rows shipped is
// draining a backlog, and keeps the lock to carry on.
func (s *Sweeper) sweep(ctx context.Context) (bool, error) {
	counted := s.count(ctx)
	locking, cancel := context.WithTimeout(ctx, lockTimeout)
	held, err := s.lock.Hold(locking)
	cancel()
	if err != nil || !held {
		return false, errors.Join(err, counted)
	}
	sweeping, cancel := context.WithTimeout(ctx, s.timeout)
	marked, err := s.ship(sweeping)
	outOfTime := ctx.Err() == nil && errors.Is(sweeping.Err(), context.DeadlineExceeded)
	cancel()
	if err != nil && outOfTime && marked > 0 {
		s.logger.Info("a usage sweep ran out of time with rows still waiting, the next carries on", "rows_marked", marked)
		err = nil
	}
	if err != nil {
		s.lock.Release()
		return true, err
	}
	return true, s.count(ctx)
}

// count reads how many rows wait in the shard and how long ago the oldest started, for the
// gauges. A count that fails leaves them as they were.
func (s *Sweeper) count(ctx context.Context) error {
	counting, cancel := context.WithTimeout(ctx, countTimeout)
	defer cancel()
	rows, oldest, err := s.store.Unshipped(counting)
	if err != nil {
		return err
	}
	s.unshipped.Store(rows)
	s.oldest.Store(int64(oldest))
	return nil
}

// ship sends the shard's waiting rows a page at a time, oldest first, until a page comes back
// short. A page is read after the last row of the one before, so the rows still waiting for an
// organization are passed over rather than read again. It fails only when the database does or
// ctx ends: a shipment that fails is counted and logged by send, and sent again next sweep. It
// returns how many rows it marked with the service's answer.
func (s *Sweeper) ship(ctx context.Context) (int, error) {
	organizations := map[string]string{}
	var after *store.Request
	marked := 0
	for {
		rows, err := s.store.RequestsToShip(ctx, s.settle, after, sweepPage)
		if err != nil {
			return marked, err
		}
		if len(rows) == 0 {
			return marked, nil
		}
		last := rows[len(rows)-1]
		after = &last
		owned, err := s.owned(ctx, rows, organizations)
		if err != nil {
			return marked, err
		}
		if err := s.sending.send(ctx, owned); err != nil {
			return marked, ctx.Err()
		}
		marked += len(owned)
		if len(rows) < sweepPage {
			return marked, nil
		}
	}
}

// owned is the rows that name their organization, looking up and writing the organization of
// those written without one. A row whose app has joined no organization yet is left waiting:
// it normally does within seconds of its first request, and the next sweep finds it.
func (s *Sweeper) owned(ctx context.Context, rows []store.Request, organizations map[string]string) ([]store.Request, error) {
	owned := make([]store.Request, 0, len(rows))
	found := map[string][]uuid.UUID{}
	for _, row := range rows {
		if row.OrganizationID == "" {
			organization, known := organizations[row.CustomerID]
			if !known {
				var err error
				if organization, err = s.store.OrganizationOf(ctx, row.CustomerID); err != nil {
					return nil, err
				}
				organizations[row.CustomerID] = organization
			}
			if organization == "" {
				usageUnowned.Inc()
				s.logUnowned(row.CustomerID)
				continue
			}
			row.OrganizationID = organization
			found[organization] = append(found[organization], row.UsageID)
		}
		owned = append(owned, row)
	}
	for organization, ids := range found {
		if err := s.store.SetRequestsOrganization(ctx, ids, organization); err != nil {
			return nil, err
		}
	}
	return owned, nil
}

// logUnowned logs an app whose rows wait for an organization, once an hour per app.
func (s *Sweeper) logUnowned(app string) {
	now := time.Now()
	if last, ok := s.unownedLogged[app]; ok && now.Sub(last) < unownedLogEvery {
		return
	}
	for logged, at := range s.unownedLogged {
		if now.Sub(at) >= unownedLogEvery {
			delete(s.unownedLogged, logged)
		}
	}
	s.unownedLogged[app] = now
	s.logger.Error("usage rows wait to ship: their app has joined no organization", "app", app)
}

// logError logs a sweep that failed, at most every errorLogEvery.
func (s *Sweeper) logError(err error) {
	now := time.Now().UnixNano()
	last := s.errorLogged.Load()
	if last != 0 && now-last < errorLogEvery.Nanoseconds() {
		return
	}
	if s.errorLogged.CompareAndSwap(last, now) {
		s.logger.Error("could not sweep unshipped usage rows, trying again", "in", s.every, "error", err)
	}
}
