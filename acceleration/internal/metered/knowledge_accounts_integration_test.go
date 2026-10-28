//go:build integration

package metered

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer/turbopuffertest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing/pricingtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// customerTurbopufferKey is the key the test's customer keeps for their own turbopuffer account.
const customerTurbopufferKey = "tpuf_customer_0123456789abcdef"

// keptKeys are the keys customers keep, by customer, provider and region. A test puts and
// deletes one by writing the map.
type keptKeys map[string]string

func (k keptKeys) CustomerKey(_ context.Context, customerID, provider, region string) (string, bool, error) {
	key, ok := k[customerID+"/"+provider+"/"+region]
	return key, ok, nil
}

// account is a customer's own turbopuffer account, which remembers the credential each request
// to it carried.
type account struct {
	*turbopuffertest.Server
	front *httptest.Server
	mu    sync.Mutex
	seen  []string
}

func (a *account) credentials() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.seen
}

// accounts is a knowledge base in the us whose deployment's account is ours, and whose
// customers' own accounts are all theirs, priced by the fixtures' price file; keys are the keys
// customers keep.
func (s *MeteredSuite) accounts(keys keptKeys) (*Knowledge, *turbopuffertest.Server, *account, *pricing.List) {
	list := pricingtest.List(s.T(), "fixtures")
	s.recorder = routing.NewRecorder(routing.Knowledge, s.store, nil, list, slog.New(slog.DiscardHandler))
	ours := turbopuffertest.Start(s.T())
	backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: ours.URL, Cost: listCost})
	s.Require().NoError(err)
	theirs := &account{Server: turbopuffertest.Start(s.T())}
	theirs.front = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		theirs.mu.Lock()
		theirs.seen = append(theirs.seen, r.Header.Get("Authorization"))
		theirs.mu.Unlock()
		theirs.Config.Handler.ServeHTTP(w, r)
	}))
	s.T().Cleanup(theirs.front.Close)
	base := NewRegionalKnowledge(map[string]knowledge.Backend{pricing.RegionUS: backend}, s.recorder, nil, s.store)
	base.WithCustomerAccounts(keys, func(region, key string) (knowledge.Backend, error) {
		s.Equal(pricing.RegionUS, region)
		return turbopuffer.New(turbopuffer.Options{APIKey: key, BaseURL: theirs.front.URL})
	})
	return base, ours, theirs, list
}

// keyOf is where keptKeys keeps the test customer's turbopuffer key for the us.
func (s *MeteredSuite) keyOf() string { return s.owner.CustomerID + "/turbopuffer/us" }

// holding is what a turbopuffer lists of the test customer's.
func (s *MeteredSuite) holding(server *turbopuffertest.Server) []string {
	backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: server.URL})
	s.Require().NoError(err)
	names, err := backend.Namespaces(s.ctx, s.owner.CustomerID)
	s.Require().NoError(err)
	return names
}

func (s *MeteredSuite) TestAnOwnerWhoKeepsATurbopufferKeyIsKeptInTheirOwnAccountAndBilledAtTheBYOKFee() {
	base, ours, theirs, list := s.accounts(keptKeys{s.keyOf(): customerTurbopufferKey})
	owned := base.For(s.owner)

	s.Require().NoError(owned.Upsert(s.ctx, "docs", passage))
	found, err := owned.Search(s.ctx, knowledge.Query{Namespace: "docs", Text: "penny"})

	s.Require().NoError(err)
	s.Len(found, 1)
	s.Equal([]string{knowledge.Scoped(s.owner.CustomerID, "docs")}, s.holding(theirs.Server))
	s.Empty(s.holding(ours), "nothing of theirs is written to the deployment's account")
	for _, credential := range theirs.credentials() {
		s.Equal("Bearer "+customerTurbopufferKey, credential)
	}
	kept, recorded, err := s.store.KnowledgePlace(s.ctx, s.owner.CustomerID, "docs")
	s.Require().NoError(err)
	s.True(recorded)
	s.Equal(store.KnowledgePlace{Region: pricing.RegionUS, Account: store.KnowledgeAccountCustomer}, kept)
	rows := s.rows()
	s.Require().Len(rows, 2)
	s.Equal("knowledge-turbopuffer-write-us-byok", rows[0].SKU)
	s.Zero(rows[0].BaseCostMicros, "the vendor bills the customer")
	s.Equal(pricingtest.Micros(s.T(), list, pricingtest.Knowledge(pricing.RegionUS, true), rows[0].Quantities()), rows[0].CostMicros)
	s.True(strings.HasSuffix(rows[1].SKU, "-us-byok"), "the lookup is priced at its BYOK entry too: %s", rows[1].SKU)
	s.Zero(rows[1].BaseCostMicros)
}

func (s *MeteredSuite) TestABaseKeptInTheDeploymentsAccountStaysThereWhenItsOwnerPutsAKey() {
	keys := keptKeys{}
	base, ours, theirs, _ := s.accounts(keys)
	owned := base.For(s.owner)
	s.Require().NoError(owned.Upsert(s.ctx, "docs", passage))

	keys[s.keyOf()] = customerTurbopufferKey
	s.Require().NoError(owned.Upsert(s.ctx, "docs", passage))

	s.Len(s.holding(ours), 1)
	s.Empty(s.holding(theirs.Server))
	rows := s.rows()
	s.Require().Len(rows, 2)
	s.Equal("knowledge-turbopuffer-write-us", rows[1].SKU)
}

func (s *MeteredSuite) TestABaseInTheCustomersOwnAccountIsRefusedOnceItsKeyIsDeletedAndNeverReadFromTheDeployments() {
	keys := keptKeys{s.keyOf(): customerTurbopufferKey}
	base, ours, _, _ := s.accounts(keys)
	owned := base.For(s.owner)
	s.Require().NoError(owned.Upsert(s.ctx, "docs", passage))

	delete(keys, s.keyOf())
	_, searched := owned.Search(s.ctx, knowledge.Query{Namespace: "docs", Text: "penny"})
	wrote := owned.Upsert(s.ctx, "docs", passage)
	deleted := owned.Delete(s.ctx, "docs", []string{"pricing.md#0"})

	for _, err := range []error{searched, wrote, deleted} {
		s.Require().ErrorIs(err, ErrNoAccountKey)
		s.Require().ErrorIs(err, routing.ErrRequest)
		s.ErrorContains(err, `"docs" is in your turbopuffer account in us; put a turbopuffer key for us back to reach it`)
	}
	s.Empty(s.holding(ours))
	s.Len(s.rows(), 1, "only the first write was done")
}

func (s *MeteredSuite) TestAnHourOfStorageInTheCustomersOwnAccountIsReadOnTheirKeyAndBilledAtTheBYOKFee() {
	base, _, theirs, _ := s.accounts(keptKeys{s.keyOf(): customerTurbopufferKey})
	namespace := knowledge.Scoped(s.owner.CustomerID, "docs")
	theirs.Holds(namespace, 1_000_000)
	hour := time.Date(2026, 10, 23, 14, 0, 0, 0, time.UTC)

	_, err := base.StorageAt(s.ctx, s.owner, store.KnowledgePlace{Region: pricing.RegionUS, Account: store.KnowledgeAccountCustomer},
		namespace, hour)

	s.Require().NoError(err)
	rows := s.rows()
	s.Require().Len(rows, 1)
	s.Equal(KnowledgeStorage, rows[0].Model)
	s.True(strings.HasSuffix(rows[0].SKU, "-us-byok"), rows[0].SKU)
	s.Equal([]string{"Bearer " + customerTurbopufferKey}, theirs.credentials())
}
