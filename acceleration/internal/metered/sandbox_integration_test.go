//go:build integration

package metered

import (
	"log/slog"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// sandboxes makes the test's sandboxes on a Daytona in process, on lifecycle, each slice an
// hour long so only a switch or the end writes one.
func (s *MeteredSuite) sandboxes(lifecycle sandbox.Lifecycle) (*Sandboxes, *daytonatest.Server) {
	server := daytonatest.Start(s.T())
	return NewSandboxes(SandboxesOptions{Recorder: s.recorder, Deployment: "deploy-a", Environment: "test", Router: "shard-a",
		Lifecycle: lifecycle, APIKey: "key-1", APIURL: server.APIURL(), Every: time.Hour, Logger: slog.New(slog.DiscardHandler)}), server
}

// sandboxOwner is the test's owner in a session of its own.
func (s *MeteredSuite) sandboxOwner() routing.Owner {
	owner := s.owner
	owner.OrganizationID = "org-" + owner.CustomerID
	owner.SessionID = "session-" + uuid.NewString()
	return owner
}

func models(rows []store.Request) []string {
	var billed []string
	for _, row := range rows {
		billed = append(billed, row.Model)
	}
	return billed
}

func (s *MeteredSuite) TestASandboxIsMeteredFromCreateToDelete() {
	sandboxes, server := s.sandboxes(sandbox.Lifecycle{IdleStop: time.Minute})
	owner := s.sandboxOwner()
	box, err := sandboxes.Open(owner, SandboxSpec{Config: sandbox.Config{CPU: 2, MemoryGB: 4, DiskGB: 10}})
	s.Require().NoError(err)
	opened := time.Now()

	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	time.Sleep(1200 * time.Millisecond)
	s.Require().NoError(box.Close())
	life := time.Since(opened)

	rows := s.rows()
	s.Require().Len(rows, 1, "one slice: the sandbox lived less than one")
	row := rows[0]
	seconds := row.Meters[pricing.DiskGBSeconds] / 10
	s.InDelta(life.Seconds(), float64(seconds), 1, "the sandbox's life, from Daytona's create to its delete")
	s.Equal(pricing.Quantities{pricing.VCPUSeconds: 2 * seconds, pricing.MemoryGBSeconds: 4 * seconds, pricing.DiskGBSeconds: 10 * seconds},
		row.Meters, "the size Daytona made, by the second")
	s.Equal(string(routing.Sandbox), row.Modality)
	s.Equal(daytonaProvider, row.Provider)
	s.Equal(SandboxRunning, row.Model)
	s.Equal("sandbox-1", row.OperationID)
	s.Equal(owner.SessionID, row.SessionID, "the session the sandbox ran in")
	s.Equal(owner.OrganizationID, row.OrganizationID)
	s.Equal(map[string]string{AttributeSandboxID: "sandbox-1", AttributeSandboxClass: "container", AttributeSandboxCPU: "2",
		AttributeSandboxMemoryGB: "4", AttributeSandboxDiskGB: "10", routing.AttributeMeasuredBy: routing.MeasuredByRouter,
		routing.AttributeSlice: "0", AttributeDeployment: "deploy-a"}, row.Attributes)
	perSecond := (2*0.0504 + 4*0.0162 + 10*0.000108) / 3600 * 1e6
	s.InDelta(perSecond*float64(seconds), float64(row.CostMicros), 1, "at Daytona's rates")
	s.True(row.Billable)
	s.Equal([]string{"sandbox-1"}, server.Deleted())
	held, _ := server.Sandbox("sandbox-1")
	s.Equal(map[string]string{LabelDeployment: "deploy-a", LabelEnv: "test", LabelRouter: "shard-a", LabelOrganization: owner.OrganizationID,
		LabelApp: owner.CustomerID, LabelSession: owner.SessionID, "code-toolbox-language": "python"}, held.Labels)
}

func (s *MeteredSuite) TestAnIdleSandboxIsMeteredAsStopped() {
	sandboxes, _ := s.sandboxes(sandbox.Lifecycle{IdleStop: time.Second})
	box, err := sandboxes.Open(s.sandboxOwner(), SandboxSpec{})
	s.Require().NoError(err)

	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	time.Sleep(2300 * time.Millisecond)
	s.Require().NoError(box.Close())

	rows := s.rows()
	s.Require().Equal([]string{SandboxRunning, SandboxStopped}, models(rows), "running until it idled, then stopped")
	s.Equal(pricing.Quantities{pricing.VCPUSeconds: 1, pricing.MemoryGBSeconds: 1, pricing.DiskGBSeconds: 3}, rows[0].Meters,
		"a second running at Daytona's default size")
	s.Equal(pricing.Quantities{pricing.DiskGBSeconds: 6}, rows[1].Meters, "then its disk alone")
	s.Zero(rows[1].CostMicros, "6 GiB-seconds of disk is 0.18 micro-dollars, rounded half up to none")
	s.True(rows[1].Billable)
}

func (s *MeteredSuite) TestAStoppedSandboxIsMeteredAsRunningAgain() {
	sandboxes, server := s.sandboxes(sandbox.Lifecycle{IdleStop: time.Second})
	box, err := sandboxes.Open(s.sandboxOwner(), SandboxSpec{})
	s.Require().NoError(err)
	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	time.Sleep(1200 * time.Millisecond)
	server.Stop("sandbox-1")

	_, err = box.Run(s.ctx, "print(2)", nil)
	s.Require().NoError(err)
	s.Require().NoError(box.Close())

	s.Equal([]string{SandboxRunning, SandboxStopped, SandboxRunning}, models(s.rows()))
}

func (s *MeteredSuite) TestAFailedDeleteKeepsTheMeterRunning() {
	sandboxes, server := s.sandboxes(sandbox.Lifecycle{IdleStop: 2 * time.Second, StoppedDelete: time.Second})
	box, err := sandboxes.Open(s.sandboxOwner(), SandboxSpec{})
	s.Require().NoError(err)
	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	server.RefuseDeletes(true)

	s.Require().Error(box.Close(), "Daytona would not delete it, as with a revoked key")
	time.Sleep(3600 * time.Millisecond)

	rows := s.rows()
	s.Require().Equal([]string{SandboxRunning, SandboxRunning, SandboxStopped}, models(rows),
		"billed on: running until it idled, then stopped until Daytona's timer deletes it")
	s.Empty(rows[0].Attributes[AttributeOrphan], "the session's own time")
	s.Equal("true", rows[1].Attributes[AttributeOrphan], "time after the session closed")
	s.Equal("true", rows[2].Attributes[AttributeOrphan])
	var seconds int64
	for _, row := range rows {
		seconds += row.Meters[pricing.DiskGBSeconds] / 3
	}
	s.InDelta(3, seconds, 1, "until Daytona deleted it: two idle seconds and one stopped")
	_, held := server.Sandbox("sandbox-1")
	s.True(held, "the router never managed to delete it; Daytona's own timer does")
}

func (s *MeteredSuite) TestAGapIsRecordedFromDaytonasFiguresOnce() {
	sandboxes, _ := s.sandboxes(sandbox.Lifecycle{})
	owner := s.owner
	owner.APIKey = routing.JobOrigin("sandbox_reconcile")
	id := "sandbox-" + uuid.NewString()
	at := time.Date(2026, 10, 21, 10, 0, 0, 0, time.UTC)
	running := SandboxGap{Owner: owner, SandboxID: id, Key: "2026-10-21T10:00:00Z", At: at, Model: SandboxRunning,
		Used:  pricing.Quantities{pricing.VCPUSeconds: 3600, pricing.MemoryGBSeconds: 3600, pricing.DiskGBSeconds: 3 * 3600},
		Class: "container"}
	stopped := SandboxGap{Owner: owner, SandboxID: id, Key: "2026-10-21T11:00:00Z", At: at.Add(time.Hour),
		Model: SandboxStopped, Used: pricing.Quantities{pricing.DiskGBSeconds: 3 * 3600}, Class: "container", Orphan: true}

	s.Require().NoError(sandboxes.Reconcile(running))
	s.Require().NoError(sandboxes.Reconcile(stopped))
	s.Require().NoError(sandboxes.Reconcile(running), "recorded again")

	rows := s.rows()
	s.Require().Len(rows, 2, "a gap recorded twice is billed once")
	s.Equal(routing.UsageID(SandboxReconciledKind, id, "2026-10-21T10:00:00Z"), rows[0].UsageID)
	s.Equal(at, rows[0].StartedAt.UTC(), "dated to the period it bills")
	s.EqualValues(66_924, rows[0].CostMicros, "an hour of Daytona's default sandbox running: 1 vCPU, 1 GiB and 3 GiB")
	s.EqualValues(324, rows[1].CostMicros, "an hour of its 3 GiB disk")
	s.Equal(map[string]string{AttributeSandboxID: id, AttributeSource: daytonaProvider, AttributeSandboxClass: "container",
		AttributeDeployment: "deploy-a"}, rows[0].Attributes, "the deployment's, so only its own job reads it")
	s.Equal("true", rows[1].Attributes[AttributeOrphan])
	s.Equal(routing.JobOrigin("sandbox_reconcile"), rows[0].APIKey)
	s.True(rows[0].Billable)
}

// daytonaProvider is the provider sandbox rows are billed under.
const daytonaProvider = "daytona"
