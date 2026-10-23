//go:build daytona_live

package daytona

import (
	"context"
	"os"
	"testing"
	"time"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// LiveLifecycleSuite runs a sandbox through Daytona's own states, against Daytona with
// DAYTONA_API_KEY: go test -tags daytona_live -run TestLiveLifecycleSuite
// ./internal/sandbox/daytona. Each sandbox is labelled stream.env=live-test and deleted when
// the test ends.
type LiveLifecycleSuite struct {
	suite.Suite
	ctx    context.Context
	client *sdk.Client
	meter  *told
}

func TestLiveLifecycleSuite(t *testing.T) {
	if os.Getenv(apiKeyEnvVar) == "" {
		t.Skip("DAYTONA_API_KEY is not set")
	}
	suite.Run(t, new(LiveLifecycleSuite))
}

func (s *LiveLifecycleSuite) SetupTest() {
	s.ctx = context.Background()
	client, err := sdk.NewClient()
	s.Require().NoError(err)
	s.client = client
	s.meter = &told{}
}

func (s *LiveLifecycleSuite) box() *Sandbox {
	box, err := New(Options{Labels: map[string]string{"stream.env": "live-test", "stream.session": "live-" + random()},
		Lifecycle: sandbox.Lifecycle{IdleStop: 15 * time.Minute, StoppedDelete: time.Hour}, Meter: s.meter,
		Timeout: 2 * time.Minute})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

func (s *LiveLifecycleSuite) TestASandboxIsCreatedWithLabelsAndALifecycle() {
	box := s.box()

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	held, err := s.client.Get(s.ctx, box.box.ID)
	s.Require().NoError(err)
	s.Equal("live-test", held.Labels["stream.env"])
	s.Equal(15, held.AutoStopInterval)
	s.Equal(60, held.AutoDeleteInterval)
	s.T().Logf("Daytona made %s: %g vCPU, %g GiB, %g GiB, %v", held.ID, held.Cpu, held.Memory, held.Disk, held.SandboxClass)
}

// TestAStoppedSandboxIsStartedAgain stops the sandbox as Daytona's timer would, and runs code
// in it again. Before T4 E the second run did not start it: it waited for a start that never
// came, or reached a toolbox that refused it.
func (s *LiveLifecycleSuite) TestAStoppedSandboxIsStartedAgain() {
	box := s.box()
	_, err := box.Run(s.ctx, "open('/tmp/kept', 'w').write('here')", nil)
	s.Require().NoError(err)
	held, err := s.client.Get(s.ctx, box.box.ID)
	s.Require().NoError(err)
	s.Require().NoError(held.Stop(s.ctx))
	box.used = time.Now().Add(-time.Hour)

	result, err := box.Run(s.ctx, "print(open('/tmp/kept').read())", nil)

	s.Require().NoError(err)
	s.Equal("here\n", result.Output, "the same sandbox, its files where they were")
	generation, err := box.Ready(s.ctx)
	s.Require().NoError(err)
	s.Equal(1, generation)
}

func (s *LiveLifecycleSuite) TestADeletedSandboxIsMadeAgain() {
	box := s.box()
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	held, err := s.client.Get(s.ctx, box.box.ID)
	s.Require().NoError(err)
	s.Require().NoError(held.DeleteAndWait(s.ctx, 2*time.Minute))
	box.used = time.Now().Add(-time.Hour)

	_, err = box.Run(s.ctx, "print(2)", nil)

	s.Require().NoError(err)
	generation, err := box.Ready(s.ctx)
	s.Require().NoError(err)
	s.Equal(2, generation, "a new sandbox")
	s.Len(s.meter.all(), 3, "created, gone, created: %v", s.meter.all())
}
