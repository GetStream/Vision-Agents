//go:build integration

package session

import (
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// daytona is a Daytona in process the sessions' sandboxes are made on.
func (s *RuntimeSuite) daytona() *daytonatest.Server {
	server := daytonatest.Start(s.T())
	s.T().Setenv("DAYTONA_API_KEY", "key-1")
	s.T().Setenv("DAYTONA_API_URL", server.APIURL())
	return server
}

// sandboxRows shuts the manager down, which writes what its sandboxes' time is still queued
// as, and returns the session's sandbox rows, oldest first.
func (s *RuntimeSuite) sandboxRows(sessionID string) []store.Request {
	s.Require().NoError(s.manager.Shutdown())
	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).
		Where("customer_id = ?", s.customer).
		Where("session_id = ?", sessionID).
		Where("modality = ?", string(routing.Sandbox)).
		Order("started_at").Scan(s.ctx))
	return rows
}

func (s *RuntimeSuite) TestASessionThatNeverDelegatesRecordsNoSandbox() {
	server := s.daytona()
	s.manages()
	created, err := s.opens(Spec{Text: true, Sandbox: daytonaProvider})
	s.Require().NoError(err)

	s.Require().NoError(created.Close())

	s.Empty(s.sandboxRows(created.ID()), "a session that never runs code never pays for a sandbox")
	s.Empty(server.Created(), "and none is made")
}

func (s *RuntimeSuite) TestASessionsSandboxIsLabelledAndBilledToTheSession() {
	server := s.daytona()
	s.manages()
	id := "runtime-" + uuid.NewString()
	spec := Spec{ID: id, CustomerID: s.customer, Sandbox: daytonaProvider, AgentID: "agent-1",
		Attribution: routing.Attribution{APIKey: "key-runtime", OrganizationID: "org-1", SessionID: id}}
	box, err := s.manager.box(s.ctx, spec, nil)
	s.Require().NoError(err)

	_, err = box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	time.Sleep(1100 * time.Millisecond)
	s.Require().NoError(box.Close())

	held, ok := server.Sandbox("sandbox-1")
	s.Require().True(ok)
	s.Equal(s.customer, held.Labels[metered.LabelApp])
	s.Equal("org-1", held.Labels[metered.LabelOrganization])
	s.Equal(id, held.Labels[metered.LabelSession])
	rows := s.sandboxRows(id)
	s.Require().NotEmpty(rows, "the sandbox's time is the session's")
	var seconds int64
	for _, row := range rows {
		s.Equal("daytona", row.Provider)
		s.Equal(metered.SandboxRunning, row.Model)
		s.Equal("sandbox-1", row.OperationID)
		s.Equal("key-runtime", row.APIKey, "billed to whoever opened the session")
		s.Equal("agent-1", row.AgentID)
		s.Equal("sandbox-1", row.Attributes[metered.AttributeSandboxID])
		seconds += row.Meters[pricing.VCPUSeconds]
	}
	s.InDelta(2, seconds, 1, "a vCPU for the second or two it lived")
}
