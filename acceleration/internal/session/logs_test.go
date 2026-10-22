package session

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func (s *SessionSuite) joinedRowFor(sfu agent.SFU) store.AgentLog {
	queue := make(chan store.AgentLog, 1)
	sess := &Session{
		logs: &logRecorder{queue: queue, store: &store.Store{}},
		spec: Spec{CustomerID: "customer-1"},
	}
	sess.recordLog(agent.Joined{At: time.Now(), SFU: sfu})
	return <-queue
}

func (s *SessionSuite) TestTheJoinedRowNamesTheAgentsSFU() {
	row := s.joinedRowFor(agent.SFU{Edge: "sfu-a1", Host: "sfu-a1.example.com"})

	s.Equal("joined", row.EventType)
	s.Equal(map[string]any{"sfu_edge": "sfu-a1", "sfu_host": "sfu-a1.example.com"}, row.Details)
}

func (s *SessionSuite) TestAJoinedRowWithoutAnSFUHasNoSFUKeys() {
	row := s.joinedRowFor(agent.SFU{})

	s.Equal("joined", row.EventType)
	s.Empty(row.Details)
}
