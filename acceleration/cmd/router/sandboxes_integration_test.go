//go:build integration

package main

import (
	"bytes"
	"cmp"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// sandboxDay is the day the reconciliation tests check.
var sandboxDay = time.Date(2026, 10, 21, 0, 0, 0, 0, time.UTC)

// ourDeployment is the deployment the job runs for, and theirDeployment another sharing its
// Daytona organization, its environment and its shard's name.
const (
	ourDeployment   = "deploy-a"
	theirDeployment = "deploy-b"
)

// ourLabels are the labels of a sandbox of this deployment's shard, made for app's session.
func ourLabels(app, session string) map[string]string {
	return labelsOf(ourDeployment, app, session)
}

// labelsOf are the labels of a sandbox of deployment's unsharded shard, made for app's session.
// An empty deployment is a router without code_sandbox.deployment, which labels none.
func labelsOf(deployment, app, session string) map[string]string {
	labels := map[string]string{metered.LabelEnv: "test", metered.LabelRouter: routing.Unsharded,
		metered.LabelOrganization: "org-" + app, metered.LabelApp: app, metered.LabelSession: session}
	if deployment != "" {
		labels[metered.LabelDeployment] = deployment
	}
	return labels
}

// sandboxJob is the job of ourDeployment against server, writing through a writer of its own as
// the CronJob's run does; stop drains it.
func (s *UsageCommandsSuite) sandboxJob(server *daytonatest.Server, logger *slog.Logger, now time.Time) (*sandboxJob, *bytes.Buffer, func()) {
	return s.sandboxJobOf(ourDeployment, server, logger, now)
}

// sandboxJobOf is sandboxJob for deployment.
func (s *UsageCommandsSuite) sandboxJobOf(deployment string, server *daytonatest.Server, logger *slog.Logger,
	now time.Time) (*sandboxJob, *bytes.Buffer, func()) {
	fleet, err := daytona.NewFleet(daytona.FleetOptions{APIKey: "key-1", APIURL: server.APIURL()})
	s.Require().NoError(err)
	recorder, stop, err := jobRecorder(s.ctx, s.settings, s.store, routing.Sandbox, logger)
	s.Require().NoError(err)
	var out bytes.Buffer
	job := &sandboxJob{fleet: fleet, store: s.store,
		sandboxes:  metered.NewSandboxes(metered.SandboxesOptions{Recorder: recorder, Deployment: deployment, Logger: logger}),
		deployment: deployment, router: routing.Unsharded, logger: logger, out: &out, now: now}
	return job, &out, func() {
		stop()
		_ = fleet.Close(s.ctx)
	}
}

// metered writes a slice of a sandbox's time as a router of ourDeployment metering it does.
func (s *UsageCommandsSuite) metered(id, app, session string, at time.Time, used pricing.Quantities) {
	s.meteredBy(ourDeployment, id, app, session, at, used)
}

// meteredBy is metered by a router of deployment; empty is one without code_sandbox.deployment.
func (s *UsageCommandsSuite) meteredBy(deployment, id, app, session string, at time.Time, used pricing.Quantities) {
	recorder := routing.NewRecorder(routing.Sandbox, s.store, nil, nil, slog.New(slog.DiscardHandler))
	stat := routing.Stat{Owner: routing.Owner{CustomerID: app, Attribution: routing.Attribution{APIKey: "key-" + app,
		OrganizationID: "org-" + app, SessionID: session}}, UsageID: uuid.New(), StartedAt: at, OperationID: id,
		Success: true, Unmeasured: true}
	stat.Use(used)
	stat.MeasuredByRouter()
	stat.Attribute(metered.AttributeSandboxID, id)
	stat.Attribute(metered.AttributeSandboxClass, "container")
	if deployment != "" {
		stat.Attribute(metered.AttributeDeployment, deployment)
	}
	recorder.Record(routing.ProviderConfig{Provider: "daytona", Model: metered.SandboxRunning}, stat)
	recorder.Close()
}

// session keeps a session of app's, closed at closed unless that is zero.
func (s *UsageCommandsSuite) session(id, app string, closed time.Time) {
	s.Require().NoError(s.store.SaveSession(s.ctx, &store.AgentSession{ID: id, CustomerID: app}))
	if !closed.IsZero() {
		s.Require().NoError(s.store.CloseSession(s.ctx, id, closed))
	}
}

func (s *UsageCommandsSuite) testdata(name string) []byte {
	data, err := os.ReadFile(filepath.Join("testdata", "daytona", name))
	s.Require().NoError(err)
	return data
}

// reconciled is the rows the reconciliation recorded, by sandbox, oldest first.
func (s *UsageCommandsSuite) reconciled() map[string][]store.Request {
	byBox := map[string][]store.Request{}
	for _, row := range s.requests() {
		if row.Attributes[metered.AttributeSource] == "daytona" {
			byBox[row.OperationID] = append(byBox[row.OperationID], row)
		}
	}
	return byBox
}

func (s *UsageCommandsSuite) TestSandboxUsageIsReconciled() {
	server := daytonatest.Start(s.T())
	server.Bills(s.testdata("usage_sandbox.json"))
	server.Periods("sandbox-b", s.testdata("periods_sandbox-b.json"))
	server.Holds(daytonatest.Box{ID: "sandbox-b", State: daytonatest.Destroyed, Labels: ourLabels("app-b", "session-b")})
	server.Holds(daytonatest.Box{ID: "sandbox-c", Labels: map[string]string{metered.LabelEnv: "elsewhere", metered.LabelApp: "app-c"}})
	server.Holds(daytonatest.Box{ID: "sandbox-d"})
	s.session("session-b", "app-b", time.Date(2026, 10, 21, 10, 20, 0, 0, time.UTC))
	morning := sandboxDay.Add(9 * time.Hour)
	s.metered("sandbox-a", "app-a", "session-a", morning, pricing.Quantities{pricing.VCPUSeconds: 3000,
		pricing.MemoryGBSeconds: 3000, pricing.DiskGBSeconds: 9000})
	s.metered("sandbox-f", "app-f", "session-f", morning, pricing.Quantities{pricing.VCPUSeconds: 590,
		pricing.MemoryGBSeconds: 590, pricing.DiskGBSeconds: 1770})
	s.metered("sandbox-g", "app-g", "session-g", morning, pricing.Quantities{pricing.VCPUSeconds: 610,
		pricing.MemoryGBSeconds: 610, pricing.DiskGBSeconds: 1830})
	s.metered("sandbox-a", "app-a", "session-a", sandboxDay.Add(-time.Minute), pricing.Quantities{pricing.VCPUSeconds: 9999})
	var logs bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logs, nil))
	defaultLogger := slog.Default()
	slog.SetDefault(logger)
	defer slog.SetDefault(defaultLogger)

	job, out, stop := s.sandboxJob(server, logger, time.Now())
	s.Require().NoError(job.reconcile(s.ctx, sandboxDay))
	stop()

	s.Equal("2026-10-21: 2 sandboxes matched Daytona's records, 2 did not, 4 gaps recorded from them, 1 made by hand, "+
		"1 another deployment's, 1 no longer known, 0 could not be read\n", out.String())
	s.Contains(server.Asked(), "/analytics/organization/org-test/usage/sandbox?from=2026-10-21T00%3A00%3A00Z&to=2026-10-22T00%3A00%3A00Z",
		"the organization read off a sandbox the key lists, and the whole day")
	s.Contains(logs.String(), `level=ERROR msg="a sandbox's billed time is too far from Daytona's records" sandbox=sandbox-a day=2026-10-21 `+
		`vcpu_seconds=3000 daytona_vcpu_seconds=3600 memory_gb_seconds=3000 daytona_memory_gb_seconds=3600 disk_gb_seconds=9000 daytona_disk_gb_seconds=10800`)
	s.Contains(logs.String(), `msg="usage alert" alert=sandbox_mismatch level=warn`)
	s.Contains(logs.String(), "item=2026-10-21", "the alert names the day")
	s.Contains(logs.String(), `msg="a sandbox Daytona billed carries no deployment label: made by hand, so not reconciled" sandbox=sandbox-d`)

	gaps := s.reconciled()
	s.Require().Len(gaps["sandbox-a"], 1, "billed short by a router that died: the day's difference")
	short := gaps["sandbox-a"][0]
	s.Equal(pricing.Quantities{pricing.VCPUSeconds: 600, pricing.MemoryGBSeconds: 600, pricing.DiskGBSeconds: 1800}, short.Meters)
	s.Equal(metered.SandboxRunning, short.Model)
	s.Equal(routing.UsageID(metered.SandboxReconciledKind, "sandbox-a", "2026-10-21"), short.UsageID)
	s.Equal("app-a", short.CustomerID, "owned as its rows were")
	s.Equal("session-a", short.SessionID)
	s.Equal(sandboxOrigin, short.APIKey, "under the reconciliation's own origin")
	s.EqualValues(8_400+2_700+54, short.BaseCostMicros, "costed at Daytona's rates, as the rest of its life")
	s.EqualValues(13_385, short.CostMicros, "and charged at the price the seed wrote: 11,154 × 1.2, rounded once")
	s.True(short.Billable)

	s.Require().Len(gaps["sandbox-f"], 1, "a sandbox within its tolerance is billed what Daytona billed all the same")
	s.Equal(pricing.Quantities{pricing.VCPUSeconds: 10, pricing.MemoryGBSeconds: 10, pricing.DiskGBSeconds: 30}, gaps["sandbox-f"][0].Meters)
	s.Empty(gaps["sandbox-g"], "billed more than Daytona's records: nothing to make up")
	for _, theirs := range []string{"sandbox-c", "sandbox-d", "sandbox-e", "sandbox-h"} {
		s.Empty(gaps[theirs], theirs)
	}

	rerun, again, stopAgain := s.sandboxJob(server, slog.New(slog.DiscardHandler), time.Now())
	s.Require().NoError(rerun.reconcile(s.ctx, sandboxDay))
	stopAgain()
	s.Equal("2026-10-21: 4 sandboxes matched Daytona's records, 0 did not, 0 gaps recorded from them, 1 made by hand, "+
		"1 another deployment's, 1 no longer known, 0 could not be read\n", again.String(), "a rerun records nothing twice")
	s.Equal(gaps, s.reconciled())
}

func (s *UsageCommandsSuite) TestAnOrphansTimeIsBilledToItsLabels() {
	server := daytonatest.Start(s.T())
	server.Bills(s.testdata("usage_sandbox.json"))
	server.Periods("sandbox-b", s.testdata("periods_sandbox-b.json"))
	server.Holds(daytonatest.Box{ID: "sandbox-b", State: daytonatest.Destroyed, Labels: ourLabels("app-b", "session-b")})
	// A sandbox the key can list is what says which organization the key is in.
	server.Holds(daytonatest.Box{ID: "sandbox-live", Labels: ourLabels("app-b", "session-later")})
	s.session("session-b", "app-b", time.Date(2026, 10, 21, 10, 20, 0, 0, time.UTC))

	job, _, stop := s.sandboxJob(server, slog.New(slog.DiscardHandler), time.Now())
	s.Require().NoError(job.reconcile(s.ctx, sandboxDay))
	stop()

	rows := s.reconciled()["sandbox-b"]
	s.Require().Len(rows, 2, "a sandbox no router metered, recorded a Daytona period at a time")
	running, stopped := rows[0], rows[1]
	s.Equal(metered.SandboxRunning, running.Model)
	s.Equal(pricing.Quantities{pricing.VCPUSeconds: 1800, pricing.MemoryGBSeconds: 1800, pricing.DiskGBSeconds: 5400}, running.Meters)
	s.Equal(time.Date(2026, 10, 21, 10, 0, 0, 0, time.UTC), running.StartedAt.UTC(), "dated to its period")
	s.Empty(running.Attributes[metered.AttributeOrphan], "the session was open when the period began")
	s.Equal(metered.SandboxStopped, stopped.Model)
	s.Equal(pricing.Quantities{pricing.DiskGBSeconds: 10_800}, stopped.Meters)
	s.Equal("true", stopped.Attributes[metered.AttributeOrphan], "time after its session closed, so a customer can be shown it")
	s.EqualValues(324, stopped.BaseCostMicros, "an hour of a 3 GiB disk at Daytona's rate")
	for _, row := range rows {
		s.Equal("app-b", row.CustomerID, "the app on its labels")
		s.Equal("session-b", row.SessionID, "the session on its labels")
		s.Equal(sandboxOrigin, row.APIKey)
		s.Equal(routing.UsageID(metered.SandboxReconciledKind, "sandbox-b", row.StartedAt.UTC().Format(time.RFC3339)), row.UsageID)
		s.True(row.Billable, "an orphan's time is billed like the rest of its life")
	}
}

func (s *UsageCommandsSuite) TestDaytonasRecordsThatCannotBeReadFailTheRun() {
	server := daytonatest.Start(s.T())
	server.Bills([]byte(`{"not": "a list"}`))
	server.Holds(daytonatest.Box{ID: "sandbox-x", Labels: ourLabels("app-x", "session-x")})

	job, _, stop := s.sandboxJob(server, slog.New(slog.DiscardHandler), time.Now())
	err := job.reconcile(s.ctx, sandboxDay)
	stop()

	s.ErrorContains(err, "read Daytona's records for 2026-10-21", "the CronJob's retry, or a run with --day, fills the day in")
}

func (s *UsageCommandsSuite) TestAnOrphanIsDeleted() {
	server := daytonatest.Start(s.T())
	now := time.Now().UTC()
	old := now.Add(-20 * time.Minute)
	s.session("session-closed", "app-1", now.Add(-15*time.Minute))
	s.session("session-open", "app-1", time.Time{})
	s.session("session-failed", "app-1", now.Add(-15*time.Minute))
	runtime := routing.NewRecorder(routing.Runtime, s.store, nil, nil, slog.New(slog.DiscardHandler))
	runtime.Record(routing.ProviderConfig{Provider: "stream", Model: "text"}, routing.Stat{
		Owner:   routing.Owner{CustomerID: "app-1", Attribution: routing.Attribution{SessionID: "session-incognito"}},
		UsageID: uuid.New(), StartedAt: now.Add(-time.Minute), Success: true, Unmeasured: true})
	runtime.Close()
	s.metered("box-failed", "app-1", "session-failed", now.Add(-18*time.Minute), pricing.Quantities{pricing.VCPUSeconds: 60})
	for id, session := range map[string]string{"box-closed": "session-closed", "box-open": "session-open",
		"box-incognito": "session-incognito", "box-silent": "session-silent", "box-failed": "session-failed"} {
		server.Holds(daytonatest.Box{ID: id, Labels: ourLabels("app-1", session), CreatedAt: old})
	}
	server.Holds(daytonatest.Box{ID: "box-young", Labels: ourLabels("app-1", "session-closed"), CreatedAt: now.Add(-time.Minute)})
	server.Holds(daytonatest.Box{ID: "box-stopped", State: daytonatest.Stopped, Labels: ourLabels("app-1", "session-closed"),
		CreatedAt: old})
	theirs := ourLabels("app-1", "session-closed")
	theirs[metered.LabelRouter] = "shard-b"
	server.Holds(daytonatest.Box{ID: "box-other-shard", Labels: theirs, CreatedAt: old})
	server.Holds(daytonatest.Box{ID: "box-by-hand", CreatedAt: old})
	var logs bytes.Buffer

	job, out, stop := s.sandboxJob(server, slog.New(slog.NewTextHandler(&logs, nil)), now)
	s.Require().NoError(job.sweep(s.ctx))
	stop()

	s.Equal([]string{"box-closed", "box-failed", "box-silent", "box-stopped"}, server.Deleted(),
		"each whose session closed, or that has no session that is still running")
	s.Equal("sandboxes: 4 orphans deleted, 2 kept for open sessions, 1 too new to judge, 0 could not be deleted\n", out.String())
	s.Contains(logs.String(), `level=WARN msg="deleted a sandbox its session no longer needs" sandbox=box-failed session=session-failed age=20m0s reason=delete_failed`)
	s.Contains(logs.String(), `sandbox=box-silent session=session-silent age=20m0s reason=unmetered`)
}

func (s *UsageCommandsSuite) TestTwoDeploymentsLeaveEachOthersSandboxesAlone() {
	server := daytonatest.Start(s.T())
	now := time.Now().UTC()
	old := now.Add(-20 * time.Minute)
	for _, deployment := range []string{ourDeployment, theirDeployment, ""} {
		id, session := "box-"+cmp.Or(deployment, "unlabelled"), "session-"+cmp.Or(deployment, "unlabelled")
		s.session(session, "app-1", now.Add(-15*time.Minute))
		server.Holds(daytonatest.Box{ID: id, Labels: labelsOf(deployment, "app-1", session), CreatedAt: old})
		s.meteredBy(deployment, id, "app-1", session, sandboxDay.Add(9*time.Hour),
			pricing.Quantities{pricing.VCPUSeconds: 590, pricing.MemoryGBSeconds: 590, pricing.DiskGBSeconds: 1770})
	}
	server.Bills([]byte(`[
  {"sandboxId": "box-deploy-a", "totalCPUSeconds": 600, "totalRAMGBSeconds": 600, "totalDiskGBSeconds": 1800},
  {"sandboxId": "box-deploy-b", "totalCPUSeconds": 600, "totalRAMGBSeconds": 600, "totalDiskGBSeconds": 1800},
  {"sandboxId": "box-unlabelled", "totalCPUSeconds": 600, "totalRAMGBSeconds": 600, "totalDiskGBSeconds": 1800}
]`))

	ours, out, stop := s.sandboxJob(server, slog.New(slog.DiscardHandler), now)
	s.Require().NoError(ours.reconcile(s.ctx, sandboxDay))
	s.Require().NoError(ours.sweep(s.ctx))
	stop()

	s.Equal([]string{"box-deploy-a"}, server.Deleted(), "the other deployment's and the unlabelled router's are left alone")
	s.Equal("2026-10-21: 1 sandboxes matched Daytona's records, 0 did not, 1 gaps recorded from them, 0 made by hand, "+
		"2 another deployment's, 0 no longer known, 0 could not be read\n"+
		"sandboxes: 1 orphans deleted, 0 kept for open sessions, 0 too new to judge, 0 could not be deleted\n", out.String(),
		"another deployment's rows are not read as this one's, though they share a database")
	gaps := s.reconciled()
	s.Require().Len(gaps["box-deploy-a"], 1)
	s.Equal(ourDeployment, gaps["box-deploy-a"][0].Attributes[metered.AttributeDeployment])
	s.Empty(gaps["box-deploy-b"])
	s.Empty(gaps["box-unlabelled"])

	theirs, _, stopTheirs := s.sandboxJobOf(theirDeployment, server, slog.New(slog.DiscardHandler), now)
	s.Require().NoError(theirs.reconcile(s.ctx, sandboxDay))
	s.Require().NoError(theirs.sweep(s.ctx))
	stopTheirs()

	s.Equal([]string{"box-deploy-a", "box-deploy-b"}, server.Deleted(), "each deployment's job deletes its own")
	gaps = s.reconciled()
	s.Len(gaps["box-deploy-a"], 1, "nothing of ours billed again by theirs")
	s.Require().Len(gaps["box-deploy-b"], 1)
	s.Equal(theirDeployment, gaps["box-deploy-b"][0].Attributes[metered.AttributeDeployment])
	s.Empty(gaps["box-unlabelled"], "a router without the setting is never touched")
	_, held := server.Sandbox("box-unlabelled")
	s.True(held)
}

func (s *UsageCommandsSuite) TestTheSandboxJobRefusesToRunWithoutADeployment() {
	s.settings.CodeSandbox.Deployment = ""
	for _, args := range [][]string{nil, {"--reconcile"}} {
		var out bytes.Buffer
		err := runUsageSandboxes(s.ctx, args, s.settings, &out)
		s.ErrorContains(err, "set code_sandbox.deployment (ROUTER_CODE_SANDBOX_DEPLOYMENT)", args)
		s.Empty(out.String())
	}
}

func (s *UsageCommandsSuite) TestAPeriodAcrossMidnightIsBilledOnEachDay() {
	server := daytonatest.Start(s.T())
	server.Holds(daytonatest.Box{ID: "sandbox-m", State: daytonatest.Destroyed, Labels: ourLabels("app-m", "session-m")})
	server.Holds(daytonatest.Box{ID: "sandbox-live", Labels: ourLabels("app-m", "session-later")})
	server.Bills([]byte(`[{"sandboxId": "sandbox-m", "totalCPUSeconds": 1800, "totalRAMGBSeconds": 1800, "totalDiskGBSeconds": 5400}]`))
	server.Periods("sandbox-m", []byte(`[{"startAt": "2026-10-20T23:30:00Z", "endAt": "2026-10-21T00:30:00Z", "cpu": 1, "ramGB": 1, "diskGB": 3}]`))

	for _, day := range []time.Time{sandboxDay.Add(-24 * time.Hour), sandboxDay, sandboxDay} {
		job, _, stop := s.sandboxJob(server, slog.New(slog.DiscardHandler), time.Now())
		s.Require().NoError(job.reconcile(s.ctx, day))
		stop()
	}

	rows := s.reconciled()["sandbox-m"]
	s.Require().Len(rows, 2, "half an hour on each day, and a rerun records neither again")
	midnight := sandboxDay
	s.Equal([]time.Time{midnight.Add(-30 * time.Minute), midnight}, []time.Time{rows[0].StartedAt.UTC(), rows[1].StartedAt.UTC()})
	for _, row := range rows {
		s.Equal(pricing.Quantities{pricing.VCPUSeconds: 1800, pricing.MemoryGBSeconds: 1800, pricing.DiskGBSeconds: 5400}, row.Meters)
		s.Equal(routing.UsageID(metered.SandboxReconciledKind, "sandbox-m", row.StartedAt.UTC().Format(time.RFC3339)), row.UsageID)
		s.Positive(row.CostMicros, "priced as the container it was")
	}
}
