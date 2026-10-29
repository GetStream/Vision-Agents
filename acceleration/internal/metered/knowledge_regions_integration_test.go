//go:build integration

package metered

import (
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

// regional is a knowledge base with a turbopuffer in the us and one in the eu, each region
// recorded in the suite's store, its rows priced by the fixtures' price file, which prices a
// write in the eu.
func (s *MeteredSuite) regional() (*Knowledge, *pricing.List) {
	list := pricingtest.List(s.T(), "fixtures")
	s.recorder = routing.NewRecorder(routing.Knowledge, s.store, nil, list, slog.New(slog.DiscardHandler))
	backends := map[string]knowledge.Backend{}
	for _, region := range []string{pricing.RegionUS, pricing.RegionEU} {
		server := turbopuffertest.Start(s.T())
		backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: server.URL, Cost: listCost})
		s.Require().NoError(err)
		backends[region] = backend
	}
	return NewRegionalKnowledge(backends, s.recorder, nil, s.store), list
}

// in is the suite's owner doing work in region, as an agent kept there does.
func (s *MeteredSuite) in(region string) routing.Owner {
	owner := s.owner
	owner.Region = region
	return owner
}

// listed is what the region's backend holds of the test customer's.
func (s *MeteredSuite) listed(base *Knowledge, region string) []string {
	names, err := base.Namespaces(s.ctx, region, s.owner.CustomerID)
	s.Require().NoError(err)
	return names
}

var passage = []knowledge.Document{{ID: "pricing.md#0", Text: "A call costs a penny.", Source: "pricing.md"}}

func (s *MeteredSuite) TestWorkInTheEUWritesThereAndIsPricedAtTheEUEntry() {
	base, list := s.regional()

	s.Require().NoError(base.For(s.in(pricing.RegionEU)).Upsert(s.ctx, "docs", passage))

	s.Equal([]string{knowledge.Scoped(s.owner.CustomerID, "docs")}, s.listed(base, pricing.RegionEU))
	s.Empty(s.listed(base, pricing.RegionUS), "nothing of eu work is written in the us")
	kept, recorded, err := s.store.KnowledgePlace(s.ctx, s.owner.CustomerID, "docs")
	s.Require().NoError(err)
	s.True(recorded)
	s.Equal(store.KnowledgePlace{Region: pricing.RegionEU, Account: store.KnowledgeAccountStream}, kept)
	rows := s.rows()
	s.Require().Len(rows, 1)
	s.Equal("knowledge-turbopuffer-write-eu", rows[0].SKU)
	s.Equal(pricingtest.Micros(s.T(), list, pricingtest.Knowledge(pricing.RegionEU, false), rows[0].Quantities()), rows[0].CostMicros)
}

func (s *MeteredSuite) TestABaseMadeByWorkNamingNoRegionIsKeptInTheUS() {
	base, _ := s.regional()

	s.Require().NoError(base.For(s.owner).Upsert(s.ctx, "docs", passage))

	s.Len(s.listed(base, pricing.RegionUS), 1)
	s.Equal("knowledge-turbopuffer-write-us", s.rows()[0].SKU)
}

func (s *MeteredSuite) TestAKnowledgeBaseIsNeverReadOrWrittenByWorkInAnotherRegion() {
	base, _ := s.regional()
	s.Require().NoError(base.For(s.in(pricing.RegionUS)).Upsert(s.ctx, "docs", passage))

	elsewhere := base.For(s.in(pricing.RegionEU))
	_, searched := elsewhere.Search(s.ctx, knowledge.Query{Namespace: "docs", Text: "penny"})
	wrote := elsewhere.Upsert(s.ctx, "docs", passage)

	for _, err := range []error{searched, wrote} {
		s.Require().ErrorIs(err, routing.ErrRequest)
		s.ErrorContains(err, `"docs" is kept in us and this work is done in eu`)
	}
	s.Empty(s.listed(base, pricing.RegionEU), "nothing is moved")
}

func (s *MeteredSuite) TestWorkNamingNoRegionGoesWhereTheBaseIsKept() {
	base, list := s.regional()
	s.Require().NoError(base.For(s.in(pricing.RegionEU)).Upsert(s.ctx, "docs", passage))
	unnamed := base.For(s.owner)

	found, err := unnamed.Search(s.ctx, knowledge.Query{Namespace: "docs", Text: "penny"})
	s.Require().NoError(err)
	s.Require().NoError(unnamed.Upsert(s.ctx, "docs", passage))

	s.Len(found, 1, "read from the eu")
	s.Empty(s.listed(base, pricing.RegionUS), "and written there, never in the us")
	rows := s.rows()
	s.Require().Len(rows, 3)
	s.Equal(KnowledgeWrite, rows[2].Model)
	s.Equal("knowledge-turbopuffer-write-eu", rows[2].SKU)
	s.Equal(pricingtest.Micros(s.T(), list, pricingtest.Knowledge(pricing.RegionEU, false), rows[2].Quantities()), rows[2].CostMicros)
}

func (s *MeteredSuite) TestKeepKeepsABaseNothingWasWrittenToInTheWorksRegion() {
	base, _ := s.regional()

	s.Require().NoError(base.For(s.in(pricing.RegionEU)).Keep(s.ctx, "docs"))
	s.Require().NoError(base.For(s.owner).Upsert(s.ctx, "docs", passage))

	kept, recorded, err := base.Place(s.ctx, s.owner.CustomerID, "docs")
	s.Require().NoError(err)
	s.True(recorded)
	s.Equal(pricing.RegionEU, kept.Region)
	s.Len(s.listed(base, pricing.RegionEU), 1, "a write naming no region went where the base was kept")
	s.Empty(s.listed(base, pricing.RegionUS))
	s.ErrorContains(base.For(s.in(pricing.RegionUS)).Keep(s.ctx, "docs"), "is kept in eu and this work is done in us")
}

func (s *MeteredSuite) TestAKnowledgeBaseIsDeletedWhereItIsKeptWhateverRegionTheWorkNames() {
	base, _ := s.regional()
	s.Require().NoError(base.For(s.in(pricing.RegionUS)).Upsert(s.ctx, "docs", passage))

	s.Require().NoError(base.For(s.in(pricing.RegionEU)).Delete(s.ctx, "docs", []string{"pricing.md#0"}))

	rows := s.rows()
	s.Require().Len(rows, 2)
	s.Equal("knowledge-turbopuffer-write-us", rows[1].SKU, "the delete was done, and billed, in the us")
}

func (s *MeteredSuite) TestARegionWithNoKnowledgeBackendIsRefusedNamingIt() {
	server := turbopuffertest.Start(s.T())
	backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: server.URL, Cost: listCost})
	s.Require().NoError(err)
	owned := NewRegionalKnowledge(map[string]knowledge.Backend{pricing.RegionUS: backend}, s.recorder, nil, s.store).
		For(s.in(pricing.RegionEU))

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
	base, _ := s.regional()
	namespace := knowledge.Scoped(s.owner.CustomerID, "docs")
	s.Require().NoError(base.For(s.in(pricing.RegionEU)).Upsert(s.ctx, "docs", passage))
	hour := time.Date(2026, 10, 23, 14, 0, 0, 0, time.UTC)

	_, err := base.Storage(s.ctx, s.owner, pricing.RegionEU, namespace, hour)

	s.Require().NoError(err)
	rows := s.rows()
	s.Require().Len(rows, 2)
	s.Equal(KnowledgeStorage, rows[0].Model, "dated to the hour it bills, before the write")
	s.Equal(routing.UsageID(KnowledgeStorageKind, namespace, "2026-10-23T14:00:00Z", pricing.RegionEU), rows[0].UsageID)
}
