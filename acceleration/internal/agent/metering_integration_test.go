//go:build integration

package agent

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// AgentMeteringSuite runs agents whose knowledge and memory record into Postgres, priced by
// the vendor list, and reads back the rows a conversation left.
type AgentMeteringSuite struct {
	AgentSuite
	pgStore *store.Store
	list    *pricing.List
	// since is when the test started: the suite's database is its own and its tests run one
	// at a time, so a row since then is the test's.
	since time.Time
}

func TestAgentMeteringSuite(t *testing.T) {
	suite.Run(t, new(AgentMeteringSuite))
}

func (s *AgentMeteringSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s must be set", dsnEnvVar)
	}
	pgStore, err := store.Open(testenv.Database(dsn, "agentmetering"))
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(context.Background()))
	s.pgStore = pgStore
	s.T().Cleanup(func() { s.Require().NoError(pgStore.Close()) })
	s.list, err = pricing.NewList("vendor", prices.Vendor().Entries, 0, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
}

func (s *AgentMeteringSuite) SetupTest() {
	s.AgentSuite.SetupTest()
	s.since = time.Now().UTC()
	s.metering = routing.NewRecorder(routing.Knowledge, s.pgStore, nil, s.list, slog.New(slog.DiscardHandler))
}

func (s *AgentMeteringSuite) TearDownTest() {
	s.metering.Close()
}

// rows drains what the agent's knowledge and memory recorded and returns the test's rows of
// the modality.
func (s *AgentMeteringSuite) rows(modality routing.Modality) []store.Request {
	s.metering.Close()
	var rows []store.Request
	s.Require().NoError(s.pgStore.DB().NewSelect().Model(&rows).
		Where("started_at >= ? AND modality = ?", s.since, string(modality)).Scan(context.Background()))
	return rows
}

func (s *AgentMeteringSuite) TestALookupRecordsItsBytes() {
	s.reads(knowledge.Document{Text: "Delivery is free over $50.", Source: "shipping.md"})
	s.join(false)
	s.model.reply = []string{"Let me check that."}
	s.asksFor("lookup", `{"query":"what does delivery cost"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "how much is delivery")
	s.Require().NoError(s.awaitToolRan().Err)

	rows := s.rows(routing.Knowledge)
	s.Require().Len(rows, 1, "one lookup, one row")
	row := rows[0]
	s.Equal("acme", row.CustomerID, "the agent's customer pays for what it looked up")
	s.Equal("stub", row.Provider)
	s.Equal(metered.KnowledgeLookup, row.Model)
	s.True(row.Success)
	s.Equal(pricing.Quantities{
		pricing.Requests: 1, pricing.BytesQueried: stubQueried, pricing.Results: 1,
		pricing.BytesReturned: int64(len("Delivery is free over $50.") + len("shipping.md")),
	}, row.Quantities(), "what the store billed, not a row at $0 with nothing on it")
}

func (s *AgentMeteringSuite) TestAMemorySearchIsOneRequest() {
	s.remembers = &stubMemory{knows: []memory.Memory{{Text: "Prefers to be called Al"}}}
	s.join(false)
	s.eventually(func() bool { return s.remembers.scopedTo().UserID != "" }, "the agent never recalled anything")

	rows := s.rows(routing.Memory)
	s.Require().Len(rows, 1)
	s.Equal("acme", rows[0].CustomerID)
	s.Equal(metered.MemorySearch, rows[0].Model, "a recall is a search, priced apart from an add")
	s.Equal(pricing.Quantities{pricing.Requests: 1}, rows[0].Quantities())
	s.True(rows[0].Success)
}
