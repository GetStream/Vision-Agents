//go:build integration

package imagerouter

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ImageRouterIntegrationSuite checks what a generation writes down, against a real
// Postgres. The providers are stubs: what is under test is the row, not the picture.
type ImageRouterIntegrationSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	customerID string
}

func TestImageRouterIntegrationSuite(t *testing.T) {
	suite.Run(t, new(ImageRouterIntegrationSuite))
}

func (s *ImageRouterIntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore
}

func (s *ImageRouterIntegrationSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *ImageRouterIntegrationSuite) SetupTest() {
	s.customerID = "customer-" + time.Now().Format("150405.000000000")
}

func (s *ImageRouterIntegrationSuite) TestAGenerationIsOneRowCountingItsPicturesAndAFailureCostsNothing() {
	drawing := &stubImages{name: "quick", drawn: pictures(2, 1024, 1024)}
	refusing := &stubImages{name: "strict", err: imagegen.Fail(imagegen.ContentFiltered, true, errors.New("flagged"))}
	registry := NewRegistry()
	registry.Register("quick", func(routing.Spec) (imagegen.Provider, error) { return drawing, nil })
	registry.Register("strict", func(routing.Spec) (imagegen.Provider, error) { return refusing, nil })
	router, err := New(Options{
		Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{
			{Provider: "quick", Model: "fast", Languages: []string{"en"}, Price: routing.Price{PerImage: 0.04}},
			{Provider: "strict", Model: "fast", Languages: []string{"en"}, Price: routing.Price{PerImage: 0.04}},
		}},
		Registry: registry,
		Store:    s.store,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)

	from := time.Now().UTC().Truncate(time.Hour)
	tags := routing.Tags{"employee": "e1"}
	_, err = router.Generate(s.ctx, Request{
		CustomerID: s.customerID, Tags: tags, Target: "quick/fast", Image: imagegen.Request{Prompt: "a field", N: 2},
	})
	s.Require().NoError(err)
	_, err = router.Generate(s.ctx, Request{
		CustomerID: s.customerID, Tags: tags, Target: "strict/fast", Image: imagegen.Request{Prompt: "a field"},
	})
	s.Require().Error(err)
	// Closing drains the writer, so both rows are in by the time it returns.
	router.Close()
	to := from.Add(2 * time.Hour)

	buckets, err := s.store.CustomerStats(s.ctx, string(routing.Image), s.customerID, store.Hourly, from, to, nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 2)
	drawn, refused := buckets[0], buckets[1]
	s.Equal("quick", drawn.Provider)
	s.EqualValues(2, drawn.ImagesTotal)
	s.EqualValues(80_000, drawn.CostMicrosTotal, "two pictures at four cents")
	s.EqualValues(1, drawn.RequestCount, "one request is one row, however many pictures it drew")
	s.Zero(drawn.ErrorCount)
	s.Equal("strict", refused.Provider)
	s.Zero(refused.ImagesTotal)
	s.Zero(refused.CostMicrosTotal, "a picture that was refused is not billed")
	s.EqualValues(1, refused.ErrorCount)

	_, err = s.store.Rollup(s.ctx, store.Hourly, from, to)
	s.Require().NoError(err)
	labelled, err := s.store.CustomerTagStats(s.ctx, string(routing.Image), s.customerID, "employee", store.Hourly, from, to)
	s.Require().NoError(err)
	s.Require().Len(labelled, 1)
	s.Equal("e1", labelled[0].TagValue)
	s.EqualValues(2, labelled[0].ImagesTotal)
	s.EqualValues(80_000, labelled[0].CostMicrosTotal)
	s.EqualValues(2, labelled[0].RequestCount)
}

func (s *ImageRouterIntegrationSuite) TestAnImageRowSaysTheSizeItWasDrawnAtWhichCanChooseThePrice() {
	drawing := &stubImages{name: "quick", drawn: pictures(1, 1024, 1792)}
	registry := NewRegistry()
	registry.Register("quick", func(routing.Spec) (imagegen.Provider, error) { return drawing, nil })
	perImage := func(usd float64) map[pricing.Meter]pricing.Rate {
		return map[pricing.Meter]pricing.Rate{pricing.Images: {USD: usd, Per: 1}}
	}
	prices, err := pricing.NewList("sizes", pricing.AllAtCost([]pricing.Entry{
		{Modality: string(routing.Image), Provider: "quick", Model: "fast", Region: pricing.RegionUS, Rates: perImage(0.04)},
		{Modality: string(routing.Image), Provider: "quick", Model: "fast", Region: pricing.RegionUS, Rates: perImage(0.08),
			When: map[string]string{imagegen.AttributeWidth: "1024", imagegen.AttributeHeight: "1792"}},
	}), slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	router, err := New(Options{
		Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{
			{Provider: "quick", Model: "fast", Languages: []string{"en"}},
		}},
		Registry: registry,
		Store:    s.store,
		Pricer:   prices,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)

	generation, err := router.Generate(s.ctx, Request{
		CustomerID: s.customerID, Target: "quick/fast", Image: imagegen.Request{Prompt: "a tall field", Width: 1024, Height: 1792},
	})
	s.Require().NoError(err)
	router.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID).Scan(s.ctx))
	s.Require().Len(rows, 1)
	s.Equal("1024", rows[0].Attributes[imagegen.AttributeWidth])
	s.Equal("1792", rows[0].Attributes[imagegen.AttributeHeight])
	s.EqualValues(80_000, rows[0].CostMicros, "priced at the rate for its size")
	s.Equal(rows[0].CostMicros, generation.CostMicros, "the caller is told what the row costs")
}
