//go:build integration

package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocrrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// OCRSuite covers reading a page: what comes back, what it cost and is recorded as, and
// what a request nothing can serve is answered with.
type OCRSuite struct {
	RouterSuite
}

func TestOCRSuite(t *testing.T) {
	runSuite(t, new(OCRSuite))
}

func (s *OCRSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *OCRSuite) TestAPageComesBackAsMarkdownAndWhatItCost() {
	read := s.read(OcrRequest{Image: drawn.Images[0].Data, Tags: &map[string]string{"manual": "fryer"}})

	s.True(strings.HasPrefix(read.Id, "ocr_"))
	s.Equal(OcrStatusCompleted, read.Status)
	s.Equal("pages", *read.Provider)
	s.Equal("# Fryer cleaning\n\nDrain the oil.", read.Markdown)
	s.EqualValues(4000, read.CostMicros, "$4 per 1,000 pages")
}

func (s *OCRSuite) TestEveryPageIsAUsageRowWithItsModelAndCost() {
	s.useApp(s.data.createApp())
	since := time.Now().UTC()

	s.read(OcrRequest{Image: drawn.Images[0].Data})
	s.read(OcrRequest{Image: drawn.Images[0].Data, Options: &OcrOptions{Target: pointerTo("vision/stub")}})

	rows := s.meteredRows(routing.OCR, since, 2)
	s.Equal("pages", rows[0].Provider)
	s.Equal(pricing.Quantities{pricing.Pages: 1}, rows[0].Quantities())
	s.EqualValues(4000, rows[0].CostMicros)
	s.Equal("vision", rows[1].Provider)
	s.Equal(pricing.Quantities{pricing.Pages: 1, pricing.InputTokens: 1500, pricing.OutputTokens: 500}, rows[1].Quantities())
	s.EqualValues(400, rows[1].CostMicros, "a vision model is priced by the tokens it took")
}

func (s *OCRSuite) TestAReadingEveryModelFailedIsAnAnswerSayingWhy() {
	read := s.read(OcrRequest{Image: drawn.Images[0].Data, Options: &OcrOptions{Target: pointerTo("broken/stub")}})

	s.Equal(OcrStatusFailed, read.Status)
	s.Equal(OcrErrorCode(ocr.ProviderFailed), *read.ErrorCode)
	s.Contains(*read.Error, "vendor down")
	s.Empty(read.Markdown)
	s.Zero(read.CostMicros)
}

func (s *OCRSuite) TestSomethingThatIsNotAPictureIsRefused() {
	status, _ := s.serverClient.call(http.MethodPost, "/v1/ocr", OcrRequest{Image: []byte("%PDF-1.7")})

	s.Equal(http.StatusBadRequest, status)
}

func (s *OCRSuite) TestAModelNamedOutsideTheOptionsIsRefused() {
	for _, model := range []any{"nobody/nothing", "", 5, map[string]any{}} {
		s.assertTopLevelModelRefused("/v1/ocr", map[string]any{"image": drawn.Images[0].Data, "model": model})
	}
}

func (s *OCRSuite) TestTheModelsThatReadPagesAreListed() {
	var providers []Provider
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/ocr/providers", nil, &providers))

	var listed []string
	for _, provider := range providers {
		listed = append(listed, provider.Provider+"/"+provider.Model)
	}
	s.ElementsMatch([]string{"pages/stub", "vision/stub", "broken/stub"}, listed)
}

func (s *OCRSuite) TestOnlyTheCustomersOwnBackendMayReadPages() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/ocr", OcrRequest{Image: drawn.Images[0].Data})
		return status
	})
}

// read sends a page the endpoint must answer for, whether or not it was read.
func (s *OCRSuite) read(request OcrRequest) OcrResult {
	var read OcrResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/ocr", request, &read))
	return read
}

// reader stands in for an OCR provider: it reads the same text every time, or fails.
type reader struct {
	name string
	read ocr.Result
	err  error
}

func (r *reader) Read(context.Context, ocr.Request) (ocr.Result, error) { return r.read, r.err }
func (r *reader) Start(context.Context) error                           { return nil }
func (r *reader) Close() error                                          { return nil }
func (r *reader) Provider() string                                      { return r.name }
func (r *reader) Model() string                                         { return "stub" }

var readers = []*reader{
	{name: "pages", read: ocr.Result{Markdown: "# Fryer cleaning\n\nDrain the oil.", Pages: 1}},
	{name: "vision", read: ocr.Result{Markdown: "# Fryer cleaning\n\nDrain the oil.", InputTokens: 1500, OutputTokens: 500}},
	{name: "broken", err: ocr.Fail(ocr.ProviderFailed, errors.New("vendor down"))},
}

func ocrConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{Provider: "pages", Model: "stub", Languages: []string{"en"}, Tier: routing.HighQuality,
				Price: routing.Price{Rates: map[pricing.Meter]pricing.Rate{pricing.Pages: {USD: 4, Per: 1000}}}},
			{Provider: "vision", Model: "stub", Languages: []string{"en"}, Tier: routing.LowLatency, InputModalities: []string{"image"},
				Price: routing.Price{PerMillionInputTokens: 0.1, PerMillionOutputTokens: 0.5}},
			{Provider: "broken", Model: "stub", Languages: []string{"en"}, Tier: routing.LowLatency,
				Price: routing.Price{Rates: map[pricing.Meter]pricing.Rate{pricing.Pages: {USD: 4, Per: 1000}}}},
		},
		Aliases: map[string]routing.Alias{
			ocrrouter.DefaultTarget: {Title: "OCR", Only: []string{"pages/stub", "vision/stub"}, Prefer: "pages/stub"},
		},
	}
}

func ocrRegistry() *ocrrouter.Registry {
	registry := ocrrouter.NewRegistry()
	for _, stub := range readers {
		registry.Register(stub.name, func(routing.Spec) (ocr.Provider, error) { return stub, nil })
	}
	return registry
}
