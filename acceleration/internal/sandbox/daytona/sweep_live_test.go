//go:build daytona_live

package daytona

import (
	"context"
	"fmt"
	"log/slog"
	"runtime"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
)

// SweepSuite runs the live suite's sweep against a Daytona in process, with no key: what it
// deletes and what it reports are checked without a call to the real one.
type SweepSuite struct {
	suite.Suite
	fake *daytonatest.Server
}

func TestSweepSuite(t *testing.T) {
	suite.Run(t, new(SweepSuite))
}

// reports is a testing.T that keeps what the sweep reports.
type reports struct{ errors []string }

func (r *reports) Errorf(format string, args ...any) {
	r.errors = append(r.errors, fmt.Sprintf(format, args...))
}
func (r *reports) FailNow() { runtime.Goexit() }

func (s *SweepSuite) SetupTest() {
	s.fake = daytonatest.Start(s.T())
	s.T().Setenv(apiKeyEnvVar, "not-a-token")
	s.T().Setenv("DAYTONA_API_URL", s.fake.APIURL())
}

// made makes a sandbox with labels on the fake.
func (s *SweepSuite) made(labels map[string]string) string {
	box, err := New(Options{APIURL: s.fake.APIURL(), Logger: slog.New(slog.DiscardHandler), Labels: labels})
	s.Require().NoError(err)
	_, err = box.Ready(context.Background())
	s.Require().NoError(err)
	return box.box.ID
}

func (s *SweepSuite) TestItDeletesTheBoxesOfItsRunOnly() {
	ours := map[string]string{"stream.env": "live-test", runLabel: "run-a"}
	first, second := s.made(ours), s.made(ours)
	s.made(map[string]string{"stream.env": "live-test", runLabel: "run-b"})
	var seen reports

	sweepRun(&seen, "run-a")

	s.Empty(seen.errors)
	s.Equal([]string{first, second}, s.fake.Deleted())
}

func (s *SweepSuite) TestABoxWithOtherLabelsIsLeftAloneAndReported() {
	s.fake.LooseLabels(true)
	ours := s.made(map[string]string{"stream.env": "live-test", runLabel: "run-a"})
	otherRun := s.made(map[string]string{"stream.env": "live-test", runLabel: "run-b"})
	otherEnv := s.made(map[string]string{"stream.env": "staging", runLabel: "run-a"})
	var seen reports

	sweepRun(&seen, "run-a")

	s.Equal([]string{ours}, s.fake.Deleted())
	s.Require().Len(seen.errors, 4, "the two boxes left alone, then them in the final list")
	s.Contains(seen.errors[0], otherRun)
	s.Contains(seen.errors[1], otherEnv)
}

func (s *SweepSuite) TestABoxInTheErrorStateIsDeletedWithoutWaiting() {
	id := s.made(map[string]string{"stream.env": "live-test", runLabel: "run-a"})
	s.fake.SetState(id, "error")
	var seen reports

	sweepRun(&seen, "run-a")

	s.Empty(seen.errors)
	s.Equal([]string{id}, s.fake.Deleted())
}

func (s *SweepSuite) TestABoxLeftInTheListIsReported() {
	id := s.made(map[string]string{"stream.env": "live-test", runLabel: "run-a"})
	s.fake.RefuseDeletes(true)
	var seen reports

	sweepRun(&seen, "run-a")

	s.Require().Len(seen.errors, 2)
	s.Contains(seen.errors[1], "is left")
	s.Contains(seen.errors[1], id)
}
