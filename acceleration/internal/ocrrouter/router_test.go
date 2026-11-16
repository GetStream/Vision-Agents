package ocrrouter

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stubReader stands in for a real provider so routing can be driven without credentials.
// One that hangs answers nothing until its context ends, as a stuck vendor does.
type stubReader struct {
	name  string
	read  ocr.Result
	err   error
	hangs bool

	mu    sync.Mutex
	asked int
}

func (s *stubReader) Read(ctx context.Context, _ ocr.Request) (ocr.Result, error) {
	s.mu.Lock()
	s.asked++
	s.mu.Unlock()
	if s.hangs {
		<-ctx.Done()
		return ocr.Result{}, ocr.Fail(ocr.CodeOf(ctx.Err()), ctx.Err())
	}
	if s.err != nil {
		return ocr.Result{}, s.err
	}
	return s.read, nil
}

func (s *stubReader) times() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.asked
}

func (s *stubReader) Start(context.Context) error { return nil }
func (s *stubReader) Close() error                { return nil }
func (s *stubReader) Provider() string            { return s.name }
func (s *stubReader) Model() string               { return "stub" }

type OCRRouterSuite struct {
	suite.Suite
	ctx context.Context
	// Provider names are unique to each test so health from one does not rank the next.
	pages  *stubReader
	vision *stubReader
	blind  *stubReader
}

func TestOCRRouterSuite(t *testing.T) {
	suite.Run(t, new(OCRRouterSuite))
}

func (s *OCRRouterSuite) SetupTest() {
	s.ctx = context.Background()
	stamp := time.Now().Format("150405.000000000")
	s.pages = &stubReader{name: "pages-" + stamp, read: ocr.Result{Markdown: "# Fryer cleaning", Pages: 1}}
	s.vision = &stubReader{name: "vision-" + stamp, read: ocr.Result{Markdown: "# Fryer cleaning", InputTokens: 1500, OutputTokens: 500}}
	s.blind = &stubReader{name: "blind-" + stamp, read: ocr.Result{Markdown: "I cannot see images."}}
}

func (s *OCRRouterSuite) router() *Router {
	registry := NewRegistry()
	for _, stub := range []*stubReader{s.pages, s.vision, s.blind} {
		registry.Register(stub.name, func(routing.Spec) (ocr.Provider, error) { return stub, nil })
	}
	router, err := New(Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{
				{Provider: s.pages.name, Model: "stub", Languages: []string{"en"}, Tier: routing.HighQuality,
					Price: routing.Price{Rates: map[pricing.Meter]pricing.Rate{pricing.Pages: {USD: 4, Per: 1000}}}},
				{Provider: s.vision.name, Model: "stub", Languages: []string{"en"}, Tier: routing.LowLatency,
					InputModalities: []string{"image"},
					Price:           routing.Price{PerMillionInputTokens: 0.1, PerMillionOutputTokens: 0.5}},
				{Provider: s.blind.name, Model: "stub", Languages: []string{"en"}, Tier: routing.LowLatency},
			},
			Aliases: map[string]routing.Alias{
				DefaultTarget: {Only: []string{s.pages.name + "/stub", s.vision.name + "/stub"}, Prefer: s.pages.name + "/stub"},
			},
		},
		Registry:       registry,
		AttemptTimeout: 50 * time.Millisecond,
		Logger:         slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	return router
}

func (s *OCRRouterSuite) TestThePreferredModelReadsThePage() {
	reading, err := s.router().Read(s.ctx, Request{CustomerID: "c1"})

	s.Require().NoError(err)
	s.Equal(s.pages.name, reading.Provider)
	s.Equal("# Fryer cleaning", reading.Markdown)
	s.Zero(s.vision.times())
}

func (s *OCRRouterSuite) TestAModelThatTimesOutHandsThePageToTheNext() {
	s.pages.hangs = true

	started := time.Now()
	reading, err := s.router().Read(s.ctx, Request{CustomerID: "c1"})

	s.Require().NoError(err)
	s.Equal(s.vision.name, reading.Provider)
	s.Equal(1, s.pages.times())
	s.Less(time.Since(started), 5*time.Second, "the first model is given its own time, not the caller's")
}

func (s *OCRRouterSuite) TestAPageIsNeverFailedOverToAModelOutsideTheShortcut() {
	s.pages.err = ocr.Fail(ocr.ProviderFailed, errors.New("down"))
	s.vision.err = ocr.Fail(ocr.ProviderFailed, errors.New("down"))

	_, err := s.router().Read(s.ctx, Request{CustomerID: "c1"})

	s.Equal(ocr.ProviderFailed, ocr.CodeOf(err))
	s.Zero(s.blind.times(), "a model that cannot read a page is no fallback for one that can")
}

func (s *OCRRouterSuite) TestNamingAModelAsksThatModel() {
	reading, err := s.router().Read(s.ctx, Request{CustomerID: "c1", Target: s.vision.name + "/stub"})

	s.Require().NoError(err)
	s.Equal(s.vision.name, reading.Provider)
	s.Zero(s.pages.times())
}

func (s *OCRRouterSuite) TestACallerThatStopsWaitingIsNotFailedOver() {
	s.pages.hangs = true
	ctx, cancel := context.WithCancel(s.ctx)
	time.AfterFunc(10*time.Millisecond, cancel)

	_, err := s.router().Read(ctx, Request{CustomerID: "c1"})

	s.Equal(ocr.Cancelled, ocr.CodeOf(err))
	s.Zero(s.vision.times())
}

func (s *OCRRouterSuite) TestAPageIsPricedByThePageOrByTheTokensItsModelBills() {
	byPage, err := s.router().Read(s.ctx, Request{CustomerID: "c1"})
	s.Require().NoError(err)
	byToken, err := s.router().Read(s.ctx, Request{CustomerID: "c1", Target: s.vision.name + "/stub"})
	s.Require().NoError(err)

	s.Equal(int64(4000), byPage.CostMicros, "$4 per 1,000 pages")
	s.Equal(int64(400), byToken.CostMicros, "1,500 input tokens at $0.10 and 500 output at $0.50 a million")
}

func (s *OCRRouterSuite) TestAScanReadWithItsFiguresGoesToAModelThatSeesFirst() {
	reading, err := s.router().Read(s.ctx, Request{CustomerID: "c1", Page: ocr.Request{Mode: ocr.TextAndFigures}})

	s.Require().NoError(err)
	s.Equal(s.vision.name, reading.Provider, "the shortcut prefers the OCR model, which cannot summarise a chart")
	s.Zero(s.pages.times())
}

func (s *OCRRouterSuite) TestAScanReadWithItsFiguresIsTranscribedWhenNoModelThatSeesCan() {
	s.vision.err = ocr.Fail(ocr.ProviderFailed, errors.New("down"))

	reading, err := s.router().Read(s.ctx, Request{CustomerID: "c1", Page: ocr.Request{Mode: ocr.TextAndFigures}})

	s.Require().NoError(err)
	s.Equal(s.pages.name, reading.Provider, "its text is more than nothing")
}

func (s *OCRRouterSuite) TestFiguresAloneAreNeverAskedOfAModelThatOnlyTranscribes() {
	s.vision.err = ocr.Fail(ocr.ProviderFailed, errors.New("down"))

	_, err := s.router().Read(s.ctx, Request{CustomerID: "c1", Page: ocr.Request{Mode: ocr.Figures}})

	s.Equal(ocr.ProviderFailed, ocr.CodeOf(err))
	s.Zero(s.pages.times(), "it would answer with the page's text, which the passages hold already")
}

// failing stands in for every model in the built-in config, writing down who was asked and
// failing, so each is asked in turn.
type failing struct {
	provider string
	model    string
	asked    *[]string
}

func (f *failing) Read(context.Context, ocr.Request) (ocr.Result, error) {
	*f.asked = append(*f.asked, f.provider+"/"+f.model)
	return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, errors.New("down"))
}

func (f *failing) Start(context.Context) error { return nil }
func (f *failing) Close() error                { return nil }
func (f *failing) Provider() string            { return f.provider }
func (f *failing) Model() string               { return f.model }

// askedThroughFigures is the order ocr-figures, in the built-in config, asks its models to
// read a page in mode.
func (s *OCRRouterSuite) askedThroughFigures(mode ocr.Mode) []string {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	var asked []string
	registry := NewRegistry()
	for _, provider := range config[routing.OCR].Providers {
		registry.Register(provider.Provider, func(spec routing.Spec) (ocr.Provider, error) {
			return &failing{provider: provider.Provider, model: spec.Model, asked: &asked}, nil
		})
	}
	router, err := New(Options{Config: config[routing.OCR], Registry: registry, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)

	_, err = router.Read(s.ctx, Request{CustomerID: "c1", Target: "ocr-figures", Page: ocr.Request{Mode: mode}})

	s.Require().Error(err)
	return asked
}

func (s *OCRRouterSuite) TestFiguresAreSummarisedByTheStrongestModelForTheMoneyFirst() {
	s.Equal([]string{"openai/gpt-6-luna", "anthropic/claude-haiku-5-5", "gemini/gemini-3.8-flash", "mistral/mistral-ocr-4-1"},
		s.askedThroughFigures(ocr.TextAndFigures))
	s.Equal([]string{"openai/gpt-6-luna", "anthropic/claude-haiku-5-5", "gemini/gemini-3.8-flash"},
		s.askedThroughFigures(ocr.Figures))
}
