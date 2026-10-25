package turbopuffer

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// listCost is what turbopuffer's pricing page charges (turbopuffer.com/pricing, read
// 2026-10-23): $1 per PB queried, $0.05 per GB returned, $2 per GB written with at least 10 KB a
// request, $0.33 per GB-month stored.
var listCost = Cost{
	pricing.BytesQueried:  {USD: 1, Per: 1_000_000_000_000_000},
	pricing.BytesReturned: {USD: 0.05, Per: 1_000_000_000},
	pricing.BytesWritten:  {USD: 2, Per: 1_000_000_000, Minimum: 10_000},
	pricing.ByteHours:     {USD: 0.33, Per: 730_000_000_000},
}

// BillingSuite reads turbopuffer's documented response bodies (testdata/README.md) as they
// come off the wire.
type BillingSuite struct {
	suite.Suite
	ctx   context.Context
	store *Store
	// answers is the testdata file each path is answered with.
	answers map[string]string
}

func TestBillingSuite(t *testing.T) {
	suite.Run(t, new(BillingSuite))
}

func (s *BillingSuite) SetupTest() {
	s.ctx = context.Background()
	s.answers = map[string]string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.Method + " " + r.URL.Path
		if cursor := r.URL.Query().Get("cursor"); cursor != "" {
			key += "?cursor=" + cursor
		}
		name, ok := s.answers[key]
		if !ok {
			http.NotFound(w, r)
			return
		}
		body, err := os.ReadFile(filepath.Join("testdata", name))
		s.Require().NoError(err)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	s.T().Cleanup(server.Close)

	store, err := New(Options{APIKey: "test-key", BaseURL: server.URL, Cost: listCost, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.store = store
}

func (s *BillingSuite) TestTheBillingObjectIsRead() {
	s.answers["POST /v2/namespaces/app-1__docs/query"] = "query.json"
	s.answers["POST /v2/namespaces/app-1__docs"] = "upsert.json"

	found, err := s.store.Search(s.ctx, knowledge.Query{Namespace: "app-1__docs", Text: "delivery"})
	s.Require().NoError(err)
	s.Len(found.Documents, 2)
	s.Equal(knowledge.Billed{Requests: 1, BytesQueried: 1_280_000_000, BytesReturned: 112, Results: 2, CostNanos: 1_286},
		found.Billed, "1.28 GB at $1 a PB is 1,280 billionths of a dollar, and 112 bytes returned 6 more")

	fetched, err := s.store.Fetch(s.ctx, "app-1__docs", []string{"shipping.md#0", "shipping.md#1"})
	s.Require().NoError(err)
	s.Len(fetched.Documents, 2)
	s.Equal(found.Billed, fetched.Billed, "a fetch is a query, billed the same way")

	written, err := s.store.Upsert(s.ctx, "app-1__docs", []knowledge.Document{
		{ID: "shipping.md#0", Text: "Delivery is free over $50.", Source: "shipping.md"},
		{ID: "shipping.md#1", Text: "Next-day costs $9.", Source: "shipping.md"},
	})
	s.Require().NoError(err)
	s.Equal(knowledge.Billed{Requests: 1, BytesWritten: 1_084, CostNanos: 20_000}, written,
		"turbopuffer's bytes as it reported them, and the cost at its 10 KB minimum")

	s.answers["POST /v2/namespaces/app-1__docs"] = "delete.json"
	deleted, err := s.store.Delete(s.ctx, "app-1__docs", []string{"shipping.md#0", "shipping.md#1"})
	s.Require().NoError(err)
	s.Equal(knowledge.Billed{Requests: 1, BytesWritten: 26, CostNanos: 20_000}, deleted, "a delete is a write")
}

func (s *BillingSuite) TestNamespacesAreListedPageByPage() {
	s.answers["GET /v1/namespaces"] = "namespaces_1.json"
	s.answers["GET /v1/namespaces?cursor=app-1__handbook"] = "namespaces_2.json"

	names, err := s.store.Namespaces(s.ctx, "")

	s.Require().NoError(err)
	s.Equal([]string{"app-1__docs", "app-1__handbook", "app-2__docs"}, names)
}

func (s *BillingSuite) TestStorageIsReadFromTheNamespacesMetadata() {
	s.answers["GET /v1/namespaces/app-1__docs/metadata"] = "metadata.json"

	stored, err := s.store.Stored(s.ctx, "app-1__docs")

	s.Require().NoError(err)
	s.Equal(knowledge.Billed{Requests: 1, ByteHours: 52_428_800, CostNanos: 23_701}, stored,
		"50 MiB for an hour at $0.33 a GB-month of 730 hours")
}

func (s *BillingSuite) TestANamespaceThatIsGoneHasNoStorage() {
	_, err := s.store.Stored(s.ctx, "app-1__deleted")

	s.ErrorIs(err, knowledge.ErrNoNamespace)
}

func TestProviderCostFollowsTurbopuffersFormula(t *testing.T) {
	cases := []struct {
		name  string
		nanos int64
		want  int64
	}{
		// turbopuffer.com/pricing, "How are writes billed?": every write is billed at least
		// 10 KB, and its batch discount, min(0.5, max(0, 0.2 × (log10(KB) − 1))), reaches 50%
		// at about 3.2 MB.
		{"a 1 KB write costs as 10 KB", listCost.write(1_000), listCost.write(10_000)},
		{"10 KB is $0.00002", listCost.write(10_000), 20_000},
		{"a 6 MB batch gets 50% off", listCost.write(6_000_000), 6_000_000},
		{"a 128 KB batch gets about 22% off", listCost.write(128_000), 199_310},
		{"a lookup on a small namespace costs 1.28 GB queried",
			listCost.query(queryBilling{BytesQueried: 1_280_000_000}), 1_280},
		{"a lookup on a 10 GB namespace costs its 10 GB queried",
			listCost.query(queryBilling{BytesQueried: 10_000_000_000}), 10_000},
		{"a GB stored for an hour", listCost.stored(1_000_000_000), 452_055},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if c.nanos != c.want {
				t.Fatalf("got %d billionths of a dollar, want %d", c.nanos, c.want)
			}
		})
	}
	if (Cost{}).write(1_000) != 0 {
		t.Fatal("a store with no cost configured works out none")
	}
}
