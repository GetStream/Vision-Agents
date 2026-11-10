package limitnotice

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// anomalyEvery is how often the check runs. AWS Cost Anomaly Detection, the reference for
	// this alert, runs about three times a day
	// (https://docs.aws.amazon.com/cost-management/latest/userguide/manage-ad.html); once an hour
	// tells the operator of a runaway agent loop sooner, and costs one read of the day's
	// requests on each shard.
	anomalyEvery = time.Hour
	// anomalyFactor is how many times its median day an organization must spend today to be
	// noticed. Steady growth stays under it: 20% a day for the four days between the median day
	// and today is about 2x. A bug, abuse or a loop that keeps calling a model is many times more.
	anomalyFactor = 5
	// anomalyDays is how many days before today make an organization's baseline: one of each
	// weekday, so a quiet weekend does not lower it. Days without spend count as nothing, so an
	// organization that spent on fewer than half of them has a median of nothing and is never
	// noticed: a week of history is the least the check judges.
	anomalyDays = 7
	// anomalyMinSpendMicros is the least an organization must spend today, in millionths of a
	// dollar, to be noticed, so a tiny organization going from cents to a few times that stays
	// quiet: a jump that small is somebody trying the API, and nothing for an operator to do.
	anomalyMinSpendMicros = 20_000_000
	// anomalyTimeout bounds one check, so a holder whose database stopped answering lets go of
	// the lock rather than keep it while nobody checks.
	anomalyTimeout = 2 * time.Minute
)

// AnomalyOptions configure Anomalies. Store and Poster are required.
type AnomalyOptions struct {
	Store  *store.Store
	Poster Poster
	// Environment is named in every message, as Notifier's are (EnvironmentTag).
	Environment string
	// Links are the pages a message links to; only Staff applies, an anomaly having no app.
	Links  Links
	Logger *slog.Logger
	// Every is how often the check runs; defaults to an hour.
	Every time.Duration
	// Now is the time the check runs at; defaults to time.Now.
	Now func() time.Time
}

// Anomalies tells the operator, in the money limit notices' channel, of each organization whose
// spend today is anomalyFactor times its median day of the anomalyDays before, once per
// organization and day. Every router runs one, and the one holding the deployment's cost anomaly
// lock does the check; the others try the lock again at the next tick.
type Anomalies struct {
	store       *store.Store
	lock        *store.ShipLock
	poster      Poster
	environment string
	links       Links
	logger      *slog.Logger
	every       time.Duration
	now         func() time.Time

	// baselines is each organization's median day before baselineDay, read once per day. Only
	// the checking goroutine uses them.
	baselines   map[string]int64
	baselineDay time.Time

	stop context.CancelFunc
	done chan struct{}
}

// NewAnomalies returns a check of every organization's spend; Start starts it.
func NewAnomalies(options AnomalyOptions) *Anomalies {
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	every := options.Every
	if every <= 0 {
		every = anomalyEvery
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Anomalies{
		store: options.Store, lock: options.Store.CostAnomalyLock(), poster: options.Poster,
		environment: options.Environment, links: options.Links, logger: logger, every: every, now: now,
	}
}

// Start checks at once and then every interval until Close.
func (a *Anomalies) Start() {
	ctx, stop := context.WithCancel(context.Background())
	a.stop, a.done = stop, make(chan struct{})
	go a.run(ctx)
}

// Close stops the check, waits for one under way to end, and gives the lock back. What it had
// noted and not posted is posted by the next check, on whichever router holds the lock.
func (a *Anomalies) Close() {
	if a.stop == nil {
		return
	}
	a.stop()
	<-a.done
}

func (a *Anomalies) run(ctx context.Context) {
	defer close(a.done)
	defer a.lock.Release()
	ticker := time.NewTicker(a.every)
	defer ticker.Stop()
	for {
		a.check(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// check notes and posts today's anomalies when this router holds the lock, then posts those
// whose post failed before. A holder whose check fails lets the lock go, so that a router that
// can check takes it at its next tick.
func (a *Anomalies) check(ctx context.Context) {
	now := a.now().UTC()
	ctx, cancel := context.WithTimeout(ctx, anomalyTimeout)
	defer cancel()
	held, err := a.lock.Hold(ctx)
	if err != nil {
		a.logger.Error("could not take the cost anomaly lock, trying again", "in", a.every, "error", err)
		return
	}
	if !held {
		return
	}
	if err := a.note(ctx, now); err != nil {
		a.lock.Release()
		a.logger.Error("could not check the organizations' spend for anomalies, trying again", "in", a.every, "error", err)
		return
	}
	anomalies, err := a.store.ClaimUnpostedCostAnomalies(ctx, stale, sweepBatch)
	if err != nil {
		a.logger.Error("could not read the cost anomalies never posted", "error", err)
		return
	}
	for _, anomaly := range anomalies {
		a.post(ctx, anomaly)
	}
}

// note writes the anomaly of each organization that has one on now's UTC day, and posts those
// it wrote.
func (a *Anomalies) note(ctx context.Context, now time.Time) error {
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
	if !a.baselineDay.Equal(today) {
		baselines, err := a.baselinesBefore(ctx, today)
		if err != nil {
			return err
		}
		a.baselines, a.baselineDay = baselines, today
	}
	spent, err := a.store.DailySpend(ctx, today, today.AddDate(0, 0, 1))
	if err != nil {
		return err
	}
	for _, spend := range spent {
		baseline := a.baselines[spend.OrganizationID]
		if spend.CostMicros < anomalyMinSpendMicros || baseline <= 0 || spend.CostMicros < anomalyFactor*baseline {
			continue
		}
		anomaly := store.CostAnomaly{
			OrganizationID: spend.OrganizationID, Day: today, SpentMicros: spend.CostMicros, BaselineMicros: baseline,
		}
		written, err := a.store.NoteCostAnomaly(ctx, anomaly)
		if err != nil {
			return err
		}
		if written {
			a.post(ctx, anomaly)
		}
	}
	return nil
}

// baselinesBefore is each organization's median daily spend over the anomalyDays before day,
// for the organizations that spent on any of them.
func (a *Anomalies) baselinesBefore(ctx context.Context, day time.Time) (map[string]int64, error) {
	spent, err := a.store.DailySpend(ctx, day.AddDate(0, 0, -anomalyDays), day)
	if err != nil {
		return nil, err
	}
	days := map[string][]int64{}
	for _, spend := range spent {
		days[spend.OrganizationID] = append(days[spend.OrganizationID], spend.CostMicros)
	}
	baselines := make(map[string]int64, len(days))
	for organization, costs := range days {
		baselines[organization] = median(costs, anomalyDays)
	}
	return baselines, nil
}

// median is the middle of costs, of days days, the days missing from costs having cost nothing.
// days is odd, so the middle is one of them.
func median(costs []int64, days int) int64 {
	all := make([]int64, days)
	copy(all, costs)
	slices.Sort(all)
	return all[days/2]
}

func (a *Anomalies) post(ctx context.Context, anomaly store.CostAnomaly) {
	if err := a.poster.Post(ctx, AnomalyMessage(a.environment, a.links, anomaly)); err != nil {
		a.logger.Error("could not post that an organization's spend jumped; it is retried",
			"organization", anomaly.OrganizationID, "error", err)
		return
	}
	if err := a.store.MarkCostAnomalyPosted(ctx, anomaly.OrganizationID, anomaly.Day); err != nil {
		a.logger.Error("posted that an organization's spend jumped, and could not record it; it may be posted again",
			"organization", anomaly.OrganizationID, "error", err)
		return
	}
	a.logger.Info("posted that an organization's spend jumped", "organization", anomaly.OrganizationID)
}

// AnomalyMessage is the anomaly as it is posted, in Slack's mrkdwn, tagged with its environment
// as Message is.
func AnomalyMessage(environment string, links Links, anomaly store.CostAnomaly) string {
	var b strings.Builder
	b.WriteString(EnvironmentTag(environment))
	fmt.Fprintf(&b, ":chart_with_upwards_trend: Organization `%s` spent %s on %s (UTC), %.1fx its median day of the %d before (%s).",
		anomaly.OrganizationID, dollars(anomaly.SpentMicros), anomaly.Day.UTC().Format(time.DateOnly),
		float64(anomaly.SpentMicros)/float64(anomaly.BaselineMicros), anomalyDays, dollars(anomaly.BaselineMicros))
	if staff := link(links.Staff, store.LimitNotice{OrganizationID: anomaly.OrganizationID}); staff != "" {
		b.WriteString("\n<" + staff + "|Staff page>")
	}
	return b.String()
}

// dollars is an amount in millionths of a dollar, to the cent.
func dollars(micros int64) string {
	return fmt.Sprintf("$%.2f", float64(micros)/1e6)
}
