//go:build integration

package metered

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer/turbopuffertest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// listCost is turbopuffer's list price (turbopuffer.com/pricing, read 2026-10-23), the
// router's default knowledge.stream_cost.
var listCost = turbopuffer.Cost{
	pricing.BytesQueried:  {USD: 1, Per: 1_000_000_000_000_000},
	pricing.BytesReturned: {USD: 0.05, Per: 1_000_000_000},
	pricing.BytesWritten:  {USD: 2, Per: 1_000_000_000, Minimum: 10_000},
	pricing.ByteHours:     {USD: 0.33, Per: 730_000_000_000},
}

// MeteredSuite calls knowledge and memory through this package into Postgres, priced by the
// vendor list, and reads back the rows each call left.
type MeteredSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	prices *pricing.List
	// recorder records the test's rows; owner is a customer of the test's own.
	recorder *routing.Recorder
	owner    routing.Owner
}

func TestMeteredSuite(t *testing.T) {
	suite.Run(t, new(MeteredSuite))
}

func (s *MeteredSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "metered"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	s.prices, err = pricing.NewList("vendor", pricing.AllAtCost(prices.Vendor().Entries), slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
}

func (s *MeteredSuite) SetupTest() {
	s.recorder = routing.NewRecorder(routing.LLM, s.store, nil, s.prices, slog.New(slog.DiscardHandler))
	s.owner = routing.Owner{CustomerID: uuid.NewString(), Attribution: routing.Attribution{APIKey: "key-a"}}
}

// rows drains the recorder and returns the test customer's rows, oldest first.
func (s *MeteredSuite) rows() []store.Request {
	s.recorder.Close()
	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", s.owner.CustomerID).
		Order("started_at", "id").Scan(s.ctx))
	return rows
}

// fixture is a turbopuffer serving its documented responses, metered for the test.
func (s *MeteredSuite) fixture() (*Knowledge, *turbopuffertest.Server) {
	server := turbopuffertest.Start(s.T())
	backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: server.URL, Cost: listCost})
	s.Require().NoError(err)
	return NewKnowledge(backend, s.recorder, nil), server
}

func (s *MeteredSuite) TestEachKnowledgeCallWritesOneRowOfWhatItWasBilled() {
	base, _ := s.fixture()
	owned := base.For(s.owner)
	namespace := "docs"

	s.Require().NoError(owned.Upsert(s.ctx, namespace, []knowledge.Document{
		{ID: "pricing.md#0", Text: "A call costs a penny.", Source: "pricing.md"},
	}))
	found, err := owned.Search(s.ctx, knowledge.Query{Namespace: namespace, Text: "penny"})
	s.Require().NoError(err)
	s.Require().Len(found, 1)
	fetched, err := owned.Fetch(s.ctx, namespace, []string{"pricing.md#0"})
	s.Require().NoError(err)
	s.Require().Len(fetched, 1)
	s.Require().NoError(owned.Delete(s.ctx, namespace, []string{"pricing.md#0"}))
	s.Require().NoError(owned.Delete(s.ctx, namespace, nil), "deleting nothing")

	rows := s.rows()
	s.Require().Len(rows, 4, "one row a call, and none for a call that asked for nothing")
	passage := int64(len("pricing.md#0") + len("A call costs a penny.") + len("pricing.md"))
	for i, want := range []struct {
		model string
		used  pricing.Quantities
		cost  int64
	}{
		{KnowledgeWrite, pricing.Quantities{pricing.Requests: 1, pricing.BytesWritten: passage}, 20},
		{KnowledgeLookup, pricing.Quantities{pricing.Requests: 1, pricing.BytesQueried: turbopuffertest.MinimumQueried,
			pricing.BytesReturned: passage, pricing.Results: 1}, 1},
		{KnowledgeLookup, pricing.Quantities{pricing.Requests: 1, pricing.BytesQueried: turbopuffertest.MinimumQueried,
			pricing.BytesReturned: passage, pricing.Results: 1}, 1},
		{KnowledgeWrite, pricing.Quantities{pricing.Requests: 1, pricing.BytesWritten: int64(len("pricing.md#0"))}, 20},
	} {
		row := rows[i]
		s.Equal("knowledge", row.Modality, "row %d", i)
		s.Equal("turbopuffer", row.Provider)
		s.Equal(want.model, row.Model, "row %d", i)
		s.Equal("key-a", row.APIKey)
		s.True(row.Success)
		s.Equal(want.used, row.Quantities(), "row %d", i)
		s.True(row.Billable, "row %d is priced", i)
		s.EqualValues(want.cost, row.CostMicros, "row %d at the vendor list's rate", i)
		s.NotEmpty(row.Attributes[routing.AttributeProviderCostMicros], "row %d keeps what it cost under turbopuffer's formula", i)
	}
	s.Equal("1.282", rows[1].Attributes[routing.AttributeProviderCostMicros],
		"1.28 GB queried at $1 a PB, and 43 bytes returned at $0.05 a GB")
	s.Equal("20.000", rows[0].Attributes[routing.AttributeProviderCostMicros], "a small write is billed its 10 KB minimum")
}

func (s *MeteredSuite) TestALookupOnALargeNamespaceSellsAtTheMinimumAndKeepsWhatItCost() {
	base, server := s.fixture()
	owned := base.For(s.owner)
	s.Require().NoError(owned.Upsert(s.ctx, "docs", []knowledge.Document{
		{ID: "pricing.md#0", Text: "A call costs a penny.", Source: "pricing.md"},
	}))
	server.Holds(knowledge.Scoped(s.owner.CustomerID, "docs"), 10_000_000_000)

	_, err := owned.Search(s.ctx, knowledge.Query{Namespace: "docs", Text: "penny"})
	s.Require().NoError(err)

	rows := s.rows()
	s.Require().Len(rows, 2, "the write and the lookup")
	lookup := rows[1]
	s.Equal(KnowledgeLookup, lookup.Model)
	s.EqualValues(10_000_000_000, lookup.Quantities()[pricing.BytesQueried], "the bytes turbopuffer billed are on the row")
	s.EqualValues(1, lookup.CostMicros, "priced per lookup, at turbopuffer's minimum, until a large-namespace rate exists")
	s.Equal("10.002", lookup.Attributes[routing.AttributeProviderCostMicros],
		"10 GB queried at $1 a PB, and 43 bytes returned: what shows a namespace costing more than it pays")
}

func (s *MeteredSuite) TestTwoCustomersNamingTheSameBaseReadOnlyTheirOwn() {
	base, _ := s.fixture()
	other := routing.Owner{CustomerID: uuid.NewString()}
	s.Require().NoError(base.For(s.owner).Upsert(s.ctx, "docs", []knowledge.Document{
		{ID: "pricing.md#0", Text: "A call costs a penny.", Source: "pricing.md"},
	}))
	s.Require().NoError(base.For(other).Upsert(s.ctx, "docs", []knowledge.Document{
		{ID: "pricing.md#0", Text: "A call costs a pound.", Source: "pricing.md"},
	}))

	found, err := base.For(s.owner).Search(s.ctx, knowledge.Query{Namespace: "docs", Text: "penny pound"})
	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("A call costs a penny.", found[0].Text)
	fetched, err := base.For(other).Fetch(s.ctx, "docs", []string{"pricing.md#0"})
	s.Require().NoError(err)
	s.Require().Len(fetched, 1)
	s.Equal("A call costs a pound.", fetched[0].Text)
}

func (s *MeteredSuite) TestAnEmptyNamespaceIsRefused() {
	base, _ := s.fixture()

	err := base.For(s.owner).Upsert(s.ctx, "", []knowledge.Document{
		{ID: "pricing.md#0", Text: "A call costs a penny.", Source: "pricing.md"},
	})

	s.Require().Error(err)
}

func (s *MeteredSuite) TestAFailedLookupIsRecordedAndNotBilled() {
	base, _ := s.fixture()

	_, err := base.For(s.owner).Search(s.ctx, knowledge.Query{Text: "no namespace"})
	s.Require().Error(err)

	row := s.rows()[0]
	s.False(row.Success)
	s.Equal(knowledgeFailed, row.ErrorCode)
	s.Zero(row.CostMicros)
}

func (s *MeteredSuite) TestAnHourOfStorageIsBilledOnceHoweverOftenItIsSampled() {
	base, server := s.fixture()
	namespace := knowledge.Scoped(s.owner.CustomerID, "docs")
	server.Holds(namespace, 2_000_000_000)
	hour := time.Date(2026, 10, 23, 14, 0, 0, 0, time.UTC)

	for _, at := range []time.Time{hour.Add(17 * time.Minute), hour.Add(48 * time.Minute), hour.Add(time.Hour)} {
		stored, err := base.Storage(s.ctx, s.owner, namespace, at)
		s.Require().NoError(err)
		s.EqualValues(2_000_000_000, stored.ByteHours)
	}

	rows := s.rows()
	s.Require().Len(rows, 2, "two samples of one hour are one row, and the next hour is another")
	s.Equal(routing.UsageID(KnowledgeStorageKind, namespace, "2026-10-23T14:00:00Z"), rows[0].UsageID)
	s.Equal(hour, rows[0].StartedAt.UTC(), "dated to the hour it bills")
	s.Equal(hour.Add(time.Hour), rows[1].StartedAt.UTC())
	s.Equal(KnowledgeStorage, rows[0].Model)
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.ByteHours: 2_000_000_000}, rows[0].Quantities())
	s.EqualValues(904, rows[0].CostMicros, "2 GB for an hour at $0.33 a GB-month of 730 hours")
	s.Equal(routing.MeasuredByRouter, rows[0].Attributes[routing.AttributeMeasuredBy])
}

func (s *MeteredSuite) TestAnEmptyKnowledgeBaseWritesNoStorageRow() {
	base, server := s.fixture()
	namespace := knowledge.Scoped(s.owner.CustomerID, "docs")
	server.Holds(namespace, 0)

	stored, err := base.Storage(s.ctx, s.owner, namespace, time.Now())

	s.Require().NoError(err)
	s.Zero(stored.ByteHours)
	s.Empty(s.rows(), "an hour of nothing is not a row")
}

func (s *MeteredSuite) TestAWriteThatFailsPartwayBillsTheRequestsThatSucceeded() {
	base, server := s.fixture()
	passages := make([]knowledge.Document, knowledge.WriteBatch+1)
	for i := range passages {
		passages[i] = knowledge.Document{ID: fmt.Sprintf("handbook.md#%d", i), Text: "A passage.", Source: "handbook.md"}
	}
	server.RefuseWritesAfter(1)

	err := base.For(s.owner).Upsert(s.ctx, "docs", passages)

	s.Require().Error(err, "the caller hears the write failed")
	rows := s.rows()
	s.Require().Len(rows, 2, "a row per request")
	s.True(rows[0].Success)
	var written int64
	for _, passage := range passages[:knowledge.WriteBatch] {
		written += int64(len(passage.ID) + len(passage.Text) + len(passage.Source))
	}
	s.Equal(pricing.Quantities{pricing.Requests: 1, pricing.BytesWritten: written}, rows[0].Quantities(),
		"the first request, which turbopuffer billed")
	s.True(rows[0].Billable)
	s.Positive(rows[0].CostMicros)
	s.False(rows[1].Success, "the request that failed")
	s.Zero(rows[1].CostMicros)
}

// mem0Platform answers mem0's endpoints with the bodies its API reference documents, and
// counts what it was asked.
func (s *MeteredSuite) mem0Platform() *mem0.Store {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method + " " + r.URL.Path {
		case "POST /v3/memories/search/":
			_ = json.NewEncoder(w).Encode(map[string]any{"results": []map[string]any{{"id": "m1", "memory": "Lives in Austin", "score": 0.9}}})
		case "POST /v3/memories/add/":
			_ = json.NewEncoder(w).Encode(map[string]string{"event_id": "e1", "status": "PENDING"})
		case "POST /v3/memories/":
			_ = json.NewEncoder(w).Encode(map[string]any{"results": []any{}})
		default:
			http.NotFound(w, r)
		}
	}))
	s.T().Cleanup(server.Close)
	backend, err := mem0.New(mem0.Options{APIKey: "test-key", BaseURL: server.URL})
	s.Require().NoError(err)
	return backend
}

func (s *MeteredSuite) TestEachMemoryCallIsARowOfItsRequests() {
	owned := NewMemory(s.mem0Platform(), s.recorder).For(s.owner)
	scope := memory.Scope{AppID: s.owner.CustomerID, UserID: "ada", AgentID: "agent-1", RunID: "session-1"}

	recalled, err := owned.Recall(s.ctx, memory.Query{Scope: scope, Text: "where do they live"})
	s.Require().NoError(err)
	s.Require().Len(recalled, 1)
	s.Require().NoError(owned.Remember(s.ctx, scope, []llm.Message{{Role: llm.User, Content: "I moved to Austin"}}))
	s.Require().NoError(owned.Remember(s.ctx, scope, []llm.Message{{Role: llm.User}}), "an exchange with nothing said")
	s.Require().NoError(owned.ForgetRun(s.ctx, s.owner.CustomerID, "session-1"))

	rows := s.rows()
	s.Require().Len(rows, 3)
	for i, want := range []struct {
		model string
		cost  int64
	}{{MemorySearch, 2_490}, {MemoryAdd, 249}, {MemoryForget, 0}} {
		s.Equal("memory", rows[i].Modality)
		s.Equal("mem0", rows[i].Provider)
		s.Equal(want.model, rows[i].Model, "row %d", i)
		s.Equal(pricing.Quantities{pricing.Requests: 1}, rows[i].Quantities(), "row %d is one request", i)
		s.EqualValues(want.cost, rows[i].CostMicros, "row %d", i)
	}
	s.True(rows[0].Billable, "$2.49 a thousand searches")
	s.True(rows[1].Billable, "$0.249 a thousand adds")
	s.False(rows[2].Billable, "mem0 bills no delete")
}

// TestEveryKnowledgeCallIsRecorded makes each call against turbopuffer itself, in a scratch
// namespace emptied after. Its customer is a random id no deployment knows, so no storage
// run bills it.
func (s *MeteredSuite) TestEveryKnowledgeCallIsRecorded() {
	if os.Getenv("TURBOPUFFER_API_KEY") == "" {
		s.T().Skip("TURBOPUFFER_API_KEY not set")
	}
	backend, err := turbopuffer.New(turbopuffer.Options{Cost: listCost, Timeout: 30 * time.Second})
	s.Require().NoError(err)
	owned := NewKnowledge(backend, s.recorder, nil).For(s.owner)
	namespace := fmt.Sprintf("acceleration-test-metered-%d", time.Now().UnixNano())
	passages := []knowledge.Document{{ID: "pricing.md#0", Text: "A call costs a penny.", Source: "pricing.md"}}

	s.Require().NoError(owned.Upsert(s.ctx, namespace, passages))
	_, err = owned.Search(s.ctx, knowledge.Query{Namespace: namespace, Text: "penny"})
	s.Require().NoError(err)
	_, err = owned.Fetch(s.ctx, namespace, []string{"pricing.md#0"})
	s.Require().NoError(err)
	s.Require().NoError(owned.Delete(s.ctx, namespace, []string{"pricing.md#0"}))

	rows := s.rows()
	s.Require().Len(rows, 4)
	for _, row := range rows {
		s.True(row.Success)
		s.Positive(row.Quantities()[pricing.Requests])
		s.NotEmpty(row.Attributes[routing.AttributeProviderCostMicros])
	}
	s.Positive(rows[0].Quantities()[pricing.BytesWritten])
	s.GreaterOrEqual(rows[1].Quantities()[pricing.BytesQueried], int64(turbopuffertest.MinimumQueried))
}
