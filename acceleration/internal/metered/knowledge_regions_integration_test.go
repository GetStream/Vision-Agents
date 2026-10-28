//go:build integration

package metered

import (
	"context"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer/turbopuffertest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing/pricingtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// keptIn is a customer's policies keeping their work in a region, admitting everything. A test
// moves them by setting region.
type keptIn struct{ region string }

func (k *keptIn) Admit(context.Context, routing.Owner) (routing.Admission, error) {
	return routing.Admission{}, nil
}

func (k *keptIn) Region(context.Context, string) (string, error) { return k.region, nil }

// regional is a knowledge base with a turbopuffer in the us and one in the eu, each region
// recorded in the suite's store, its rows priced by the fixtures' price file, which prices a
// write in the eu.
func (s *MeteredSuite) regional(policies *keptIn) (*Knowledge, *pricing.List) {
	list := pricingtest.List(s.T(), "fixtures")
	s.recorder = routing.NewRecorder(routing.Knowledge, s.store, nil, list, slog.New(slog.DiscardHandler))
	backends := map[string]knowledge.Backend{}
	for _, region := range []string{pricing.RegionUS, pricing.RegionEU} {
		server := turbopuffertest.Start(s.T())
		backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: server.URL, Cost: listCost})
		s.Require().NoError(err)
		backends[region] = backend
	}
	return NewRegionalKnowledge(backends, s.recorder, policies, s.store), list
}

// listed is what the region's backend holds of the test customer's.
func (s *MeteredSuite) listed(base *Knowledge, region string) []string {
	names, err := base.Namespaces(s.ctx, region, s.owner.CustomerID)
	s.Require().NoError(err)
	return names
}

var passage = []knowledge.Document{{ID: "pricing.md#0", Text: "A call costs a penny.", Source: "pricing.md"}}

func (s *MeteredSuite) TestAnOwnerKeptInTheEUWritesThereAndIsPricedAtTheEUEntry() {
	base, list := s.regional(&keptIn{region: pricing.RegionEU})

	s.Require().NoError(base.For(s.owner).Upsert(s.ctx, "docs", passage))

	s.Equal([]string{knowledge.Scoped(s.owner.CustomerID, "docs")}, s.listed(base, pricing.RegionEU))
	s.Empty(s.listed(base, pricing.RegionUS), "nothing of an eu customer's is written in the us")
	kept, recorded, err := s.store.KnowledgePlace(s.ctx, s.owner.CustomerID, "docs")
	s.Require().NoError(err)
	s.True(recorded)
	s.Equal(store.KnowledgePlace{Region: pricing.RegionEU, Account: store.KnowledgeAccountStream}, kept)
	rows := s.rows()
	s.Require().Len(rows, 1)
	s.Equal("knowledge-turbopuffer-write-eu", rows[0].SKU)
	s.Equal(pricingtest.Micros(s.T(), list, pricingtest.Knowledge(pricing.RegionEU, false), rows[0].Quantities()), rows[0].CostMicros)
}

func (s *MeteredSuite) TestAnOwnerWhosePoliciesNameNoRegionIsKeptInTheUS() {
	base, _ := s.regional(&keptIn{region: pricing.DefaultRegion})

	s.Require().NoError(base.For(s.owner).Upsert(s.ctx, "docs", passage))

	s.Len(s.listed(base, pricing.RegionUS), 1)
	s.Equal("knowledge-turbopuffer-write-us", s.rows()[0].SKU)
}

func (s *MeteredSuite) TestAKnowledgeBaseIsNeverReadOrWrittenFromAnotherRegion() {
	policies := &keptIn{region: pricing.RegionUS}
	base, _ := s.regional(policies)
	owned := base.For(s.owner)
	s.Require().NoError(owned.Upsert(s.ctx, "docs", passage))

	policies.region = pricing.RegionEU
	_, searched := owned.Search(s.ctx, knowledge.Query{Namespace: "docs", Text: "penny"})
	wrote := owned.Upsert(s.ctx, "docs", passage)

	for _, err := range []error{searched, wrote} {
		s.Require().ErrorIs(err, routing.ErrRequest)
		s.ErrorContains(err, `"docs" is kept in us and your policies keep your work in eu`)
	}
	s.Empty(s.listed(base, pricing.RegionEU), "nothing is moved")
}

func (s *MeteredSuite) TestAKnowledgeBaseIsDeletedWhereItIsKeptWhateverThePoliciesSayNow() {
	policies := &keptIn{region: pricing.RegionUS}
	base, _ := s.regional(policies)
	owned := base.For(s.owner)
	s.Require().NoError(owned.Upsert(s.ctx, "docs", passage))

	policies.region = pricing.RegionEU
	s.Require().NoError(owned.Delete(s.ctx, "docs", []string{"pricing.md#0"}))

	rows := s.rows()
	s.Require().Len(rows, 2)
	s.Equal("knowledge-turbopuffer-write-us", rows[1].SKU, "the delete was done, and billed, in the us")
}

func (s *MeteredSuite) TestARegionWithNoKnowledgeBackendIsRefusedNamingIt() {
	server := turbopuffertest.Start(s.T())
	backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: server.URL, Cost: listCost})
	s.Require().NoError(err)
	owned := NewRegionalKnowledge(map[string]knowledge.Backend{pricing.RegionUS: backend}, s.recorder,
		&keptIn{region: pricing.RegionEU}, s.store).For(s.owner)

	err = owned.Upsert(s.ctx, "docs", passage)

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.ErrorContains(err, "keeps no knowledge in region eu")
	_, recorded, err := s.store.KnowledgePlace(s.ctx, s.owner.CustomerID, "docs")
	s.Require().NoError(err)
	s.False(recorded, "a base nothing was written to is kept nowhere")
}

func (s *MeteredSuite) TestTwoFirstWritesInTwoRegionsKeepTheBaseWhereTheFirstLanded() {
	eu := store.KnowledgePlace{Region: pricing.RegionEU, Account: store.KnowledgeAccountCustomer}
	kept, err := s.store.ClaimKnowledgePlace(s.ctx, s.owner.CustomerID, "docs", eu)
	s.Require().NoError(err)
	s.Equal(eu, kept)

	kept, err = s.store.ClaimKnowledgePlace(s.ctx, s.owner.CustomerID, "docs",
		store.KnowledgePlace{Region: pricing.RegionUS, Account: store.KnowledgeAccountStream})

	s.Require().NoError(err)
	s.Equal(eu, kept, "the region and the account of the first")
}

func (s *MeteredSuite) TestAnHourOfStorageOutsideTheDefaultRegionIsAUsageOfItsOwn() {
	base, _ := s.regional(&keptIn{region: pricing.RegionEU})
	namespace := knowledge.Scoped(s.owner.CustomerID, "docs")
	s.Require().NoError(base.For(s.owner).Upsert(s.ctx, "docs", passage))
	hour := time.Date(2026, 10, 23, 14, 0, 0, 0, time.UTC)

	_, err := base.Storage(s.ctx, s.owner, pricing.RegionEU, namespace, hour)

	s.Require().NoError(err)
	rows := s.rows()
	s.Require().Len(rows, 2)
	s.Equal(KnowledgeStorage, rows[0].Model, "dated to the hour it bills, before the write")
	s.Equal(routing.UsageID(KnowledgeStorageKind, namespace, "2026-10-23T14:00:00Z", pricing.RegionEU), rows[0].UsageID)
}
