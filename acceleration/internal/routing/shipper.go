package routing

import (
	"cmp"
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

const (
	// shipQueue is how many written batches wait for the shipper before the next is left to
	// the sweeper.
	shipQueue = 100
	// shipTimeout bounds one call to the usage service, and marking in the shard what it said.
	shipTimeout = 5 * time.Second
	// rejectionsLogged is how many of a batch's rejections its log line names.
	rejectionsLogged = 5
)

// Unsharded names the shard of a router in a deployment of one, in its events' source and its
// alerts.
const Unsharded = "unsharded"

// Source names a router process in the events it ships, router/<shard>/<host>: where an event
// came from, and what a shard's rows are counted by in the usage service.
func Source(shard, host string) string {
	return "router/" + cmp.Or(shard, Unsharded) + "/" + cmp.Or(host, "unknown")
}

// Acked is a row the usage service has, accepted now or before. A row may be acked more than
// once: the shipper and the sweeper can both send it, and a row whose shipment was not marked
// in the shard is sent again.
type Acked struct {
	UsageID        uuid.UUID
	OrganizationID string
	CostMicros     int64
	// Mirrored is the ack's: whether Redis took the window totals of every event of the
	// shipment the row was in. The service answers it per batch, not per row.
	Mirrored bool
}

// shipping sends rows to the usage service and marks in the shard what it answered. The
// writer's shipper and the sweeper each have one.
type shipping struct {
	store   *store.Store
	usage   *usage.Client
	source  string
	shipped func([]Acked)
	logger  *slog.Logger
	// errorLogged is when a failed shipment was last logged, in Unix nanoseconds.
	errorLogged atomic.Int64
}

// send ships rows that each name their organization, and marks in the shard which the service
// has and which it refused. On an error the rows it has not marked stay to be sent again by
// the sweeper; the service records a usage id once, so sending one twice charges it once.
func (s *shipping) send(ctx context.Context, rows []store.Request) error {
	if len(rows) == 0 {
		return nil
	}
	events := make([]usage.Event, len(rows))
	for i := range rows {
		events[i] = eventOf(rows[i], s.source)
	}
	call, cancel := context.WithTimeout(ctx, shipTimeout)
	ack, err := s.usage.Ship(call, events)
	cancel()
	if err != nil {
		s.failed(ctx, "could not ship usage rows, the sweeper sends them again", len(rows), err)
		return err
	}

	refused, err := refusedRows(rows, ack)
	if err != nil {
		s.failed(ctx, "the usage service's answer does not match the usage rows sent, the sweeper sends them again", len(rows), err)
		return err
	}
	var named []string
	for _, rejection := range ack.Rejected[:min(len(ack.Rejected), rejectionsLogged)] {
		named = append(named, fmt.Sprintf("%s %s: %s", rows[rejection.Index].UsageID, rejection.Reason, rejection.Message))
	}
	shipped := make([]uuid.UUID, 0, len(rows)-len(refused))
	acked := make([]Acked, 0, len(rows)-len(refused))
	for _, row := range rows {
		if _, no := refused[row.UsageID]; no {
			continue
		}
		shipped = append(shipped, row.UsageID)
		acked = append(acked, Acked{UsageID: row.UsageID, OrganizationID: row.OrganizationID, CostMicros: row.CostMicros, Mirrored: ack.Mirrored})
	}
	usageShipped.Add(float64(len(shipped)))
	if len(refused) > 0 {
		usageRejected.Add(float64(len(refused)))
		s.logger.Error("the usage service refused usage rows, kept with their ship_error until reshipped",
			"rows", len(refused), "rejections", strings.Join(named, "; "))
	}
	if s.shipped != nil && len(acked) > 0 {
		s.shipped(acked)
	}

	marking, cancel := context.WithTimeout(ctx, shipTimeout)
	defer cancel()
	if err := s.store.RecordShipment(marking, shipped, refused); err != nil {
		s.failed(ctx, "could not mark usage rows shipped, the sweeper sends them again", len(rows), err)
		return err
	}
	return nil
}

// refusedRows is why the service refused each row it refused, by usage id. An answer that does
// not account for every row sent once, accepted, a duplicate or refused, or that names a row
// not sent, is an error: nothing of it is marked, so no refused row is taken for shipped, and
// the sweeper sends the rows again. A service of another version could answer so.
func refusedRows(rows []store.Request, ack usage.Ack) (map[uuid.UUID]string, error) {
	if answered := ack.Accepted + ack.Duplicates + len(ack.Rejected); answered != len(rows) {
		return nil, fmt.Errorf("routing: the usage service answered for %d events of %d", answered, len(rows))
	}
	refused := make(map[uuid.UUID]string, len(ack.Rejected))
	for _, rejection := range ack.Rejected {
		if rejection.Index < 0 || rejection.Index >= len(rows) {
			return nil, fmt.Errorf("routing: the usage service refused event %d of %d", rejection.Index, len(rows))
		}
		id := rows[rejection.Index].UsageID
		// A rejection names its event by index when it cannot by usage id.
		if rejection.UsageID != uuid.Nil && rejection.UsageID != id {
			return nil, fmt.Errorf("routing: the usage service refused %s as event %d, which is %s", rejection.UsageID, rejection.Index, id)
		}
		if _, twice := refused[id]; twice {
			return nil, fmt.Errorf("routing: the usage service refused event %d twice", rejection.Index)
		}
		refused[id] = rejection.Reason
	}
	return refused, nil
}

// failed counts a shipment that failed, and logs it at most every errorLogEvery. One cut short
// by the process stopping is neither: its rows wait in the shard for the next sweep.
func (s *shipping) failed(ctx context.Context, message string, rows int, err error) {
	if ctx.Err() != nil {
		return
	}
	usageShipErrors.Inc()
	now := time.Now().UnixNano()
	last := s.errorLogged.Load()
	if last != 0 && now-last < errorLogEvery.Nanoseconds() {
		return
	}
	if s.errorLogged.CompareAndSwap(last, now) {
		s.logger.Error(message, "rows", rows, "error", err)
	}
}

// eventOf is the event a row ships as. A row written before the writer defaulted its key and
// tier, and since queued again, takes the same defaults, so the service accepts it.
func eventOf(request store.Request, source string) usage.Event {
	quantities := make(map[string]int64)
	for meter, quantity := range request.Quantities() {
		quantities[string(meter)] = quantity
	}
	return usage.Event{
		UsageID: request.UsageID,
		// bun writes a time truncated to microseconds, so a row sent from memory and one read
		// back say the same, and fall in the same day.
		OccurredAt:       request.StartedAt.UTC().Truncate(time.Microsecond),
		OrganizationID:   request.OrganizationID,
		OrganizationTier: cmp.Or(request.OrganizationTier, usage.DefaultTier),
		AppID:            request.CustomerID,
		APIKey:           cmp.Or(request.APIKey, Unattributed),
		UserID:           request.UserID,
		UsageCategories:  request.UsageCategories,
		AgentID:          request.AgentID,
		CallID:           request.CallID,
		SessionID:        request.SessionID,
		Meter:            request.Modality,
		Provider:         request.Provider,
		Model:            request.Model,
		Purpose:          request.Purpose,
		Quantities:       quantities,
		Attributes:       request.Attributes,
		CostMicros:       request.CostMicros,
		BaseCostMicros:   request.BaseCostMicros,
		PriceVersion:     request.PriceVersion,
		SKU:              request.SKU,
		Billable:         request.Billable,
		Success:          request.Success,
		ErrorCode:        request.ErrorCode,
		Tags:             request.Tags,
		Source:           source,
	}
}

// shipper ships the batches the writer has written, off the writer's goroutine: the writer
// never waits on the usage service, and a batch it has no room for waits in the shard for the
// sweeper. So does one that waited for the shipper longer than settle, which the sweeper may
// be sending already.
type shipper struct {
	*shipping
	settle  time.Duration
	batches chan shipment
	ctx     context.Context
	cancel  context.CancelFunc
	done    chan struct{}
}

// shipment is a written batch and when it was queued to ship.
type shipment struct {
	rows []store.Request
	at   time.Time
}

func newShipper(sending *shipping, settle time.Duration) *shipper {
	ctx, cancel := context.WithCancel(context.Background())
	s := &shipper{
		shipping: sending, settle: settle, batches: make(chan shipment, shipQueue), ctx: ctx, cancel: cancel, done: make(chan struct{}),
	}
	go s.run()
	return s
}

// offer queues written rows to ship without waiting. A row with no organization is left to
// the sweeper, which looks it up.
func (s *shipper) offer(rows []store.Request) {
	owned := make([]store.Request, 0, len(rows))
	for _, row := range rows {
		if row.OrganizationID != "" {
			owned = append(owned, row)
		}
	}
	if len(owned) == 0 {
		return
	}
	select {
	case s.batches <- shipment{rows: owned, at: time.Now()}:
	default:
		usageShipDeferred.Add(float64(len(owned)))
	}
}

func (s *shipper) run() {
	defer close(s.done)
	for batch := range s.batches {
		if s.ctx.Err() != nil {
			continue
		}
		if time.Since(batch.at) > s.settle {
			usageShipDeferred.Add(float64(len(batch.rows)))
			continue
		}
		_ = s.send(s.ctx, batch.rows)
	}
}

// close ships what is queued until ctx is done. What is left waits in the shard for a sweep.
func (s *shipper) close(ctx context.Context) {
	close(s.batches)
	select {
	case <-s.done:
	case <-ctx.Done():
		s.cancel()
		<-s.done
	}
	s.cancel()
}
