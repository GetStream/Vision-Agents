package main

import (
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"slices"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// sandboxMismatch is the alert a reconciliation sends, once, naming the day, when what was
// billed for a sandbox and what Daytona's records say it used are too far apart, or Daytona's
// records could not be read.
const sandboxMismatch = "sandbox_mismatch"

// sandboxOrigin is the key a sandbox's time recorded from Daytona's own figures is owned under:
// the router's own reconciliation job.
var sandboxOrigin = routing.JobOrigin("sandbox_reconcile")

const (
	// sandboxRecordsLag is how late Daytona's records may be: a day is reconciled once it ended
	// that long ago.
	sandboxRecordsLag = 48 * time.Hour
	// orphanAge is how old a sandbox must be before the sweep judges it: a session's own row
	// may not be written yet in its first moments.
	orphanAge = 10 * time.Minute
	// runtimeSilence is how long a session with no row may go without recording runtime and
	// still be taken to be running: runtime is recorded a minute at a time.
	runtimeSilence = 10 * time.Minute
	// sandboxApart and sandboxApartFloor are how far what was billed and Daytona's records may
	// be apart, on any of the three meters, before it is a mismatch: 5% of Daytona's figure, or
	// 60 units (vCPU or GiB seconds), whichever is more.
	sandboxApart      = 0.05
	sandboxApartFloor = 60
)

var (
	sandboxUsageMismatch = promauto.NewCounter(prometheus.CounterOpts{
		Name: "router_sandbox_usage_mismatch_total",
		Help: "Sandboxes whose billed vCPU, memory or disk seconds were too far from Daytona's records for a day.",
	})
	sandboxOrphansDeleted = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "router_sandbox_orphans_deleted_total",
		Help: "Sandboxes the sweep deleted after their session had closed, by why they were left: a router's delete " +
			"failed, or no router was metering them.",
	}, []string{"reason"})
)

// sandboxEnvironment names the deployment on every sandbox it makes: the environment Sentry
// reports it as, which the chart sets per release.
func sandboxEnvironment() string {
	return cmp.Or(os.Getenv("SENTRY_ENVIRONMENT"), "development")
}

// sandboxRouter names the shard on every sandbox it makes.
func sandboxRouter(settings config.Config) string {
	return cmp.Or(settings.Shard, routing.Unsharded)
}

// newSandboxes makes the sessions' sandboxes, labelled with the deployment and shard and
// billed through the process's writer.
func newSandboxes(settings config.Config, writer *routing.Writer, prices pricing.Pricer, logger *slog.Logger) *metered.Sandboxes {
	return metered.NewSandboxes(metered.SandboxesOptions{
		Recorder:    routing.NewRecorderOn(routing.Sandbox, writer, prices, logger),
		Environment: sandboxEnvironment(),
		Router:      sandboxRouter(settings),
		Lifecycle:   sandbox.Lifecycle{IdleStop: settings.CodeSandbox.IdleStop, StoppedDelete: settings.CodeSandbox.StoppedDelete},
		Logger:      logger,
	})
}

// runUsageSandboxes deletes the sandboxes of this shard no session needs any more and, asked
// to, reconciles a day of what sandboxes were billed with Daytona's records.
func runUsageSandboxes(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("usage sandboxes", flag.ContinueOnError)
	reconcile := flags.Bool("reconcile", false, "also reconcile a day of sandbox usage with Daytona's records")
	dayText := flags.String("day", "", "the day to reconcile, YYYY-MM-DD; empty is the day before yesterday")
	if err := flags.Parse(args); err != nil {
		return err
	}
	now := time.Now()
	day, err := sandboxDayOf(*dayText, now)
	if err != nil {
		return err
	}
	logger := slog.Default()
	fleet, err := daytona.NewFleet(daytona.FleetOptions{})
	if err != nil {
		return fmt.Errorf("usage sandboxes: %w", err)
	}
	defer fleet.Close(ctx) //nolint:errcheck // nothing is left to send
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	recording, stop, err := jobRecorder(ctx, settings, pgStore, routing.Sandbox, logger)
	if err != nil {
		return err
	}
	defer stop()

	job := &sandboxJob{
		fleet: fleet, store: pgStore, sandboxes: metered.NewSandboxes(metered.SandboxesOptions{Recorder: recording, Logger: logger}),
		environment: sandboxEnvironment(), router: sandboxRouter(settings), logger: logger, out: out, now: now,
	}
	var failed []error
	if *reconcile {
		failed = append(failed, job.reconcile(ctx, day))
	}
	failed = append(failed, job.sweep(ctx))
	return errors.Join(failed...)
}

// sandboxDayOf is the day a reconciliation covers: the one --day names, or the day before
// yesterday, the last Daytona's records are complete for.
func sandboxDayOf(text string, now time.Time) (time.Time, error) {
	if text == "" {
		return now.UTC().Add(-sandboxRecordsLag).Truncate(24 * time.Hour), nil
	}
	day, err := time.Parse(time.DateOnly, text)
	if err != nil {
		return time.Time{}, fmt.Errorf("usage sandboxes: --day %q is not a YYYY-MM-DD day", text)
	}
	return day, nil
}

// sandboxJob looks after this shard's sandboxes in the deployment's Daytona organization.
type sandboxJob struct {
	fleet     *daytona.Fleet
	store     *store.Store
	sandboxes *metered.Sandboxes
	// environment and router are the labels this shard's sandboxes carry.
	environment, router string
	logger              *slog.Logger
	out                 io.Writer
	now                 time.Time
}

// ours reports whether labels are this shard's.
func (j *sandboxJob) ours(labels map[string]string) bool {
	return labels[metered.LabelEnv] == j.environment && labels[metered.LabelRouter] == j.router
}

// sweep deletes each of this shard's sandboxes whose session has ended, or has no session,
// once it is old enough to judge. Its time until now was billed to the organization and app on
// its labels: by the router metering it, whose delete failed, or by the reconciliation from
// Daytona's records when no router was. A sandbox that could not be deleted fails the run.
func (j *sandboxJob) sweep(ctx context.Context) error {
	listed, err := j.fleet.Labelled(ctx, map[string]string{metered.LabelEnv: j.environment, metered.LabelRouter: j.router})
	if err != nil {
		j.logger.Error("could not list this shard's sandboxes, so none is swept", "error", err)
		return fmt.Errorf("usage sandboxes: %w", err)
	}
	var sessions, ids []string
	for _, box := range listed {
		ids = append(ids, box.ID)
		if session := box.Labels[metered.LabelSession]; session != "" {
			sessions = append(sessions, session)
		}
	}
	ends, err := j.store.SessionEnds(ctx, sessions, j.now.Add(-runtimeSilence))
	if err != nil {
		return fmt.Errorf("usage sandboxes: %w", err)
	}
	recorded, err := j.store.SandboxUsage(ctx, ids, j.now.Add(-sandboxRecordsLag), j.now.Add(time.Minute))
	if err != nil {
		return fmt.Errorf("usage sandboxes: %w", err)
	}

	var deleted, kept, young int
	var failed []string
	for _, box := range listed {
		age := j.now.Sub(box.CreatedAt)
		session := box.Labels[metered.LabelSession]
		switch {
		case age < orphanAge:
			young++
			continue
		case session != "" && !ends[session].Ended():
			kept++
			continue
		}
		reason := "unmetered"
		if recorded[box.ID].Rows > 0 {
			reason = "delete_failed"
		}
		if err := j.fleet.Delete(ctx, box.ID); err != nil {
			j.logger.Error("could not delete a sandbox its session no longer needs", "sandbox", box.ID, "session", session,
				"error", err)
			failed = append(failed, box.ID)
			continue
		}
		deleted++
		sandboxOrphansDeleted.WithLabelValues(reason).Inc()
		j.logger.Warn("deleted a sandbox its session no longer needs", "sandbox", box.ID, "session", session,
			"age", age.Round(time.Second), "reason", reason, "state", box.State)
	}
	if _, err := fmt.Fprintf(j.out, "sandboxes: %d orphans deleted, %d kept for open sessions, %d too new to judge, %d could not be deleted\n",
		deleted, kept, young, len(failed)); err != nil {
		return err
	}
	if len(failed) > 0 {
		return fmt.Errorf("usage sandboxes: %d orphaned sandboxes could not be deleted: %v", len(failed), failed)
	}
	return nil
}

// tally is what one sandbox used over the day, in vCPU, memory GiB and disk GiB seconds.
type tally struct{ cpu, memory, disk int64 }

func (t tally) zero() bool { return t == tally{} }

// apart reports whether recorded is too far from billed, on any meter.
func apart(billed, recorded tally) bool {
	for _, pair := range [][2]int64{{billed.cpu, recorded.cpu}, {billed.memory, recorded.memory}, {billed.disk, recorded.disk}} {
		allowed := max(sandboxApart*float64(pair[0]), sandboxApartFloor)
		if math.Abs(float64(pair[0]-pair[1])) > allowed {
			return true
		}
	}
	return false
}

// reconcile checks a day of each sandbox of this shard's that Daytona's records say used
// something against what was billed for it, and records what was billed short from Daytona's
// figures: a whole sandbox no router metered a period at a time, a sandbox billed short by
// the day's difference. Owned by the sandbox's labels' organization and app, under the
// reconciliation's origin, and priced as the rest of the sandbox's life. Too far apart is a
// mismatch, logged at ERROR with both figures, counted, and alerted on once for the day as
// sandbox_mismatch. A sandbox with no deployment label was made by hand and is left out; one
// Daytona no longer knows, which this shard recorded nothing of, cannot be told apart from
// another deployment's. Records that cannot be read fail the run, so a rerun fills the day in;
// a rerun records nothing twice.
func (j *sandboxJob) reconcile(ctx context.Context, day time.Time) error {
	from, to := day, day.Add(24*time.Hour)
	dayText := day.Format(time.DateOnly)
	billed, err := j.fleet.Usage(ctx, from, to)
	if err != nil {
		j.logger.Error("could not read Daytona's records, so no sandbox is reconciled for the day", "day", dayText, "error", err)
		alerts.SendItem(sandboxMismatch, dayText, alerts.Warn, map[string]any{"day": dayText, "reading": err.Error(),
			"rerun": "router usage sandboxes --reconcile --day " + dayText})
		return fmt.Errorf("usage sandboxes: read Daytona's records for %s: %w", dayText, err)
	}
	ids := make([]string, 0, len(billed))
	for _, used := range billed {
		ids = append(ids, used.SandboxID)
	}
	recorded, err := j.store.SandboxUsage(ctx, ids, from, to)
	if err != nil {
		return fmt.Errorf("usage sandboxes: %w", err)
	}

	var matched, filled, handMade, notOurs, unknown int
	var mismatched, unread []string
	for _, used := range billed {
		daytonas := tally{cpu: rounded(used.CPUSeconds), memory: rounded(used.RAMGBSeconds), disk: rounded(used.DiskGBSeconds)}
		if daytonas.zero() {
			continue
		}
		ours := recorded[used.SandboxID]
		owner, known, err := j.owner(ctx, used.SandboxID, ours)
		switch {
		case err != nil:
			j.logger.Error("could not read a sandbox Daytona billed, so it is not reconciled", "sandbox", used.SandboxID, "error", err)
			unread = append(unread, used.SandboxID)
			continue
		case known == ownerHandMade:
			j.logger.Info("a sandbox Daytona billed carries no deployment label: made by hand, so not reconciled", "sandbox", used.SandboxID)
			handMade++
			continue
		case known == ownerTheirs:
			notOurs++
			continue
		case known == ownerUnknown:
			j.logger.Warn("Daytona billed a sandbox it no longer knows and this shard recorded none of: whose it was cannot "+
				"be told, so it is not reconciled", "sandbox", used.SandboxID, "day", dayText, "vcpu_seconds", daytonas.cpu)
			unknown++
			continue
		}
		have := tally{cpu: ours.VCPUSeconds, memory: ours.MemoryGBSeconds, disk: ours.DiskGBSeconds}
		if apart(daytonas, have) {
			mismatched = append(mismatched, used.SandboxID)
			sandboxUsageMismatch.Inc()
			j.logger.Error("a sandbox's billed time is too far from Daytona's records", "sandbox", used.SandboxID, "day", dayText,
				"vcpu_seconds", have.cpu, "daytona_vcpu_seconds", daytonas.cpu,
				"memory_gb_seconds", have.memory, "daytona_memory_gb_seconds", daytonas.memory,
				"disk_gb_seconds", have.disk, "daytona_disk_gb_seconds", daytonas.disk)
		} else {
			matched++
		}
		gaps, err := j.gaps(ctx, used.SandboxID, owner, daytonas, have, ours.Rows == 0, from, to)
		if err != nil {
			j.logger.Error("could not record what a sandbox was billed short", "sandbox", used.SandboxID, "day", dayText, "error", err)
			unread = append(unread, used.SandboxID)
			continue
		}
		filled += gaps
	}

	if _, err := fmt.Fprintf(j.out, "%s: %d sandboxes matched Daytona's records, %d did not, %d gaps recorded from them, "+
		"%d made by hand, %d another deployment's, %d no longer known, %d could not be read\n",
		dayText, matched, len(mismatched), filled, handMade, notOurs, unknown, len(unread)); err != nil {
		return err
	}
	if len(mismatched) > 0 {
		alerts.SendItem(sandboxMismatch, dayText, alerts.Warn, map[string]any{
			"day": dayText, "mismatched": len(mismatched), "sandboxes": mismatched[:min(len(mismatched), 20)],
		})
	}
	if len(unread) > 0 {
		return fmt.Errorf("usage sandboxes: %d sandboxes could not be reconciled for %s; run router usage sandboxes --reconcile --day %s "+
			"once they can", len(unread), dayText, dayText)
	}
	return nil
}

// What a reconciliation knows of a sandbox's owner.
const (
	ownerOurs = iota
	ownerTheirs
	ownerHandMade
	ownerUnknown
)

// owner is whose a sandbox Daytona billed was: the organization, app and session on its rows,
// when this shard recorded some, or on its labels.
func (j *sandboxJob) owner(ctx context.Context, id string, recorded store.SandboxUsed) (routing.Owner, int, error) {
	owned := func(app, organization, session string) routing.Owner {
		return routing.Owner{CustomerID: app, Attribution: routing.Attribution{
			OrganizationID: organization, APIKey: sandboxOrigin, SessionID: session,
		}}
	}
	if recorded.Rows > 0 {
		return owned(recorded.CustomerID, recorded.OrganizationID, recorded.SessionID), ownerOurs, nil
	}
	box, found, err := j.fleet.Sandbox(ctx, id)
	switch {
	case err != nil:
		return routing.Owner{}, ownerUnknown, err
	case !found:
		return routing.Owner{}, ownerUnknown, nil
	case box.Labels[metered.LabelEnv] == "":
		return routing.Owner{}, ownerHandMade, nil
	case !j.ours(box.Labels) || box.Labels[metered.LabelApp] == "":
		return routing.Owner{}, ownerTheirs, nil
	}
	labels := box.Labels
	return owned(labels[metered.LabelApp], labels[metered.LabelOrganization], labels[metered.LabelSession]), ownerOurs, nil
}

// gaps records what a sandbox was billed short of Daytona's records over the day, and says how
// many rows that took: nothing billed at all is recorded a Daytona period at a time; billed
// short, the day's difference on each meter in one row.
func (j *sandboxJob) gaps(ctx context.Context, id string, owner routing.Owner, daytonas, have tally, unmetered bool,
	from, to time.Time) (int, error) {
	closed := j.closedAt(ctx, owner.SessionID)
	if !unmetered {
		short := tally{cpu: max(daytonas.cpu-have.cpu, 0), memory: max(daytonas.memory-have.memory, 0), disk: max(daytonas.disk-have.disk, 0)}
		if short.zero() {
			return 0, nil
		}
		return 1, j.sandboxes.Reconcile(metered.SandboxGap{Owner: owner, SandboxID: id, Key: from.Format(time.DateOnly), At: from,
			Model: model(short), Used: short.used(), Orphan: closed != nil && closed.Before(from)})
	}
	periods, err := j.fleet.Periods(ctx, id, from, to)
	if err != nil {
		return 0, err
	}
	recorded := 0
	for _, period := range periods {
		start, startErr := time.Parse(time.RFC3339Nano, period.StartAt)
		end, endErr := time.Parse(time.RFC3339Nano, period.EndAt)
		if startErr != nil || endErr != nil {
			return recorded, fmt.Errorf("a period of sandbox %s is dated %q to %q", id, period.StartAt, period.EndAt)
		}
		start, end = maxTime(start, from), minTime(end, to)
		seconds := end.Sub(start).Seconds()
		if seconds <= 0 {
			continue
		}
		used := tally{cpu: rounded(period.CPU * seconds), memory: rounded(period.RAMGB * seconds), disk: rounded(period.DiskGB * seconds)}
		if used.zero() {
			continue
		}
		err := j.sandboxes.Reconcile(metered.SandboxGap{Owner: owner, SandboxID: id, Key: period.StartAt, At: start,
			Model: model(used), Used: used.used(), Orphan: closed != nil && !start.Before(*closed)})
		if err != nil {
			return recorded, err
		}
		recorded++
	}
	return recorded, nil
}

// closedAt is when a session closed, nil while it is open or when this shard cannot say.
func (j *sandboxJob) closedAt(ctx context.Context, session string) *time.Time {
	if session == "" {
		return nil
	}
	ends, err := j.store.SessionEnds(ctx, []string{session}, j.now)
	if err != nil {
		j.logger.Warn("could not read whether a sandbox's session closed, so none of its time is marked an orphan's",
			"session", session, "error", err)
		return nil
	}
	return ends[session].ClosedAt
}

// model is what time that used is billed as: running when it held vCPUs or memory, stopped
// when it held disk alone.
func model(used tally) string {
	if used.cpu > 0 || used.memory > 0 {
		return metered.SandboxRunning
	}
	return metered.SandboxStopped
}

func (t tally) used() pricing.Quantities {
	used := pricing.Quantities{}
	for meter, quantity := range map[pricing.Meter]int64{pricing.VCPUSeconds: t.cpu, pricing.MemoryGBSeconds: t.memory,
		pricing.DiskGBSeconds: t.disk} {
		if quantity > 0 {
			used[meter] = quantity
		}
	}
	return used
}

func rounded(value float64) int64 { return int64(math.Round(value)) }

func maxTime(a, b time.Time) time.Time { return slices.MaxFunc([]time.Time{a, b}, time.Time.Compare) }

func minTime(a, b time.Time) time.Time { return slices.MinFunc([]time.Time{a, b}, time.Time.Compare) }
