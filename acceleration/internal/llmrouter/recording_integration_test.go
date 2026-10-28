//go:build integration

package llmrouter

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing/pricingtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// replayLLM answers every request with a response that has already happened.
type replayLLM struct {
	*stubLLM
	response llm.Response
}

func (r *replayLLM) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	response := r.response
	response.ID = params.ID
	return llm.Replay(llm.ResponseCompleted{Response: response}), nil
}

// RecordingSuite records responses through a session over the built-in price list into
// Postgres, and reads back the row each one stored.
type RecordingSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	config routing.Config
	prices *pricing.List
	// logs is what the recorder of the test logged.
	logs *bytes.Buffer
}

func (s *RecordingSuite) SetupTest() {
	s.logs = &bytes.Buffer{}
}

func TestRecordingSuite(t *testing.T) {
	suite.Run(t, new(RecordingSuite))
}

func (s *RecordingSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })

	s.config, err = routing.DefaultConfig()
	s.Require().NoError(err)
	listed, err := routing.Prices(s.config, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
	// At the vendors' rates, so each cost below is what a vendor's published rate makes of the
	// tokens it billed.
	s.prices, err = pricing.NewList("vendor", pricing.AllAtCost(listed.Entries()), slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

// builtIn is the built-in config's LLM provider of that name.
func (s *RecordingSuite) builtIn(name string) routing.ProviderConfig {
	for _, provider := range s.config[routing.LLM].Providers {
		if provider.Name() == name {
			return provider
		}
	}
	s.FailNow("the built-in config has no " + name)
	return routing.ProviderConfig{}
}

// recorded asks the provider for one response through a session of the built-in config's
// model, and returns the row it stored and the cost the caller was told.
func (s *RecordingSuite) recorded(name string, response llm.Response) (store.Request, int64) {
	return s.recordedFor(name, llm.ResponseParams{Input: prompt()}, response)
}

// recordedFor is recorded, asking with params. A response that has already happened never
// fails its stream, however it ended.
func (s *RecordingSuite) recordedFor(name string, params llm.ResponseParams, response llm.Response) (store.Request, int64) {
	row, told, err := s.recordedVia(s.ctx, name, &replayLLM{stubLLM: newStubLLM(), response: response}, params)
	s.Require().NoError(err)
	return row, told
}

// recordedVia asks provider for one response with ctx, through a session priced as the
// built-in config's model, and returns the row it stored, the cost the caller was told and
// the stream's failure.
func (s *RecordingSuite) recordedVia(ctx context.Context, name string, provider Provider, params llm.ResponseParams) (store.Request, int64, error) {
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.LLM, s.store, nil, s.prices, slog.New(slog.NewTextHandler(s.logs, nil)))
	// A key, as every owner site names one: a row without one is logged as an error.
	owner := routing.Owner{CustomerID: customer, Attribution: routing.Attribution{APIKey: "key-a"}}
	session := newSession(provider, s.builtIn(name), owner, recorder, nil)

	var told int64
	params.ID = "r1"
	params.OnTiming = func(timing llm.CallTiming) { told = timing.CostMicros }
	stream, err := session.Create(ctx, params)
	s.Require().NoError(err)
	drain(stream)
	recorder.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Scan(s.ctx))
	s.Require().Len(rows, 1)
	return rows[0], told, stream.Err()
}

// stalling is a chat completions provider whose server streams the opening of an answer and
// then nothing, until the request is given up on, giving up itself after limit.
func (s *RecordingSuite) stalling(limit time.Duration) Provider {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{"content":"Paris is"}}]}`+"\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	s.T().Cleanup(server.Close)
	provider, err := Started(openaicompat.New(openaicompat.Options{
		Provider: "stalling", Model: "m", APIKey: "k", BaseURL: server.URL, Timeout: limit,
	}))
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = provider.Close() })
	return provider
}

// requireBilledOnTheEstimate checks that a response a deadline cut short after "Paris is" was
// billed as a cut is, counted as a timeout and alarming nobody.
func (s *RecordingSuite) requireBilledOnTheEstimate(row store.Request, told int64, err error, before float64) {
	s.ErrorIs(err, context.DeadlineExceeded, "whoever waited sees the deadline")
	s.False(row.Success)
	s.Equal(routing.ErrorTimeout, row.ErrorCode)
	s.EqualValues(101, row.InputTokens, "the vendor read the whole prompt before the deadline")
	s.EqualValues(2, row.OutputTokens, "and wrote what was forwarded")
	s.Equal(map[string]string{routing.AttributeUsageEstimated: "true"}, row.Attributes)
	s.True(row.Billable)
	s.Positive(row.CostMicros)
	s.Equal(row.CostMicros, told)
	s.Equal(before+1, s.counted("router_usage_estimated_total", item("openai/gpt-6-luna", "reason", "timeout")))
	s.NotContains(s.logs.String(), "level=ERROR")
}

func (s *RecordingSuite) TestAResponseACallersDeadlineCutShortIsBilledOnTheEstimate() {
	before := s.counted("router_usage_estimated_total", item("openai/gpt-6-luna", "reason", "timeout"))
	ctx, cancel := context.WithTimeout(s.ctx, 300*time.Millisecond)
	defer cancel()

	row, told, err := s.recordedVia(ctx, "openai/gpt-6-luna", s.stalling(0), briefly())

	s.requireBilledOnTheEstimate(row, told, err, before)
}

func (s *RecordingSuite) TestAResponseTheProvidersOwnLimitCutShortIsBilledOnTheEstimate() {
	before := s.counted("router_usage_estimated_total", item("openai/gpt-6-luna", "reason", "timeout"))

	row, told, err := s.recordedVia(s.ctx, "openai/gpt-6-luna", s.stalling(300*time.Millisecond), briefly())

	s.requireBilledOnTheEstimate(row, told, err, before)
}

func (s *RecordingSuite) TestAResponseTheVendorFailedIsNotBilled() {
	row, _ := s.recordedFor("openai/gpt-6-luna", briefly(), llm.Response{
		Status:     llm.StatusFailed,
		OutputText: "Paris is",
	})

	s.Equal("provider_error", row.ErrorCode)
	s.Zero(row.InputTokens)
	s.Zero(row.CostMicros)
	s.Empty(row.Attributes, "a vendor is not taken to bill its own failure")
}

// TestCacheWritesAndServiceTierReachTheRow uses the first turn of a cached conversation as
// gpt-5.6-terra answered it, served at the flex tier.
func (s *RecordingSuite) TestCacheWritesAndServiceTierReachTheRow() {
	row, told := s.recorded("openai/gpt-5.6-terra", llm.Response{
		Status:      llm.StatusCompleted,
		ServiceTier: "flex",
		Usage: llm.Usage{
			InputTokens:         2_389,
			InputTokensDetails:  llm.InputTokensDetails{CacheWriteTokens: 2_386},
			OutputTokens:        22,
			OutputTokensDetails: llm.OutputTokensDetails{ReasoningTokens: 7},
		},
	})

	s.Equal(pricing.Quantities{pricing.CacheWriteInputTokens: 2_386}, row.Meters)
	s.Equal(map[string]string{"service_tier": "flex", "reasoning_tokens": "7"}, row.Attributes)
	s.EqualValues(2_389, row.InputTokens, "the input stays the whole prompt")
	// 3 fresh tokens, 2,386 written and 22 generated, at the flex tier's rates.
	flex := routing.ItemOf(routing.LLM, s.builtIn("openai/gpt-5.6-terra"), map[string]string{"service_tier": "flex"})
	s.Equal(pricingtest.Micros(s.T(), s.prices, flex, pricing.Quantities{pricing.InputTokens: 3,
		pricing.CacheWriteInputTokens: 2_386, pricing.OutputTokens: 22}), row.CostMicros)
	s.Equal(row.CostMicros, told, "the caller is told what the row costs")
	repriced := s.prices.Quote(routing.ItemOf(routing.LLM, s.builtIn("openai/gpt-5.6-terra"), row.Attributes), row.Quantities())
	s.Equal(row.CostMicros, repriced.CostMicros, "the row alone prices to what it was charged")
}

func (s *RecordingSuite) TestAResponseWithNothingBeyondItsTokensStoresNoMeters() {
	row, _ := s.recorded("openai/gpt-6-luna", llm.Response{
		Status: llm.StatusCompleted,
		Usage:  llm.Usage{InputTokens: 9, OutputTokens: 5},
	})

	s.Empty(row.Meters)
	s.Empty(row.Attributes)
	s.Equal(pricingtest.Micros(s.T(), s.prices, routing.ItemOf(routing.LLM, s.builtIn("openai/gpt-6-luna"), nil),
		pricing.Quantities{pricing.InputTokens: 9, pricing.OutputTokens: 5}), row.CostMicros)
}

// counted is what the router's counter of that name has counted for the series with those
// labels, read as /metrics serves it.
func (s *RecordingSuite) counted(name string, labels map[string]string) float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	var total float64
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			pairs := map[string]string{}
			for _, pair := range metric.GetLabel() {
				pairs[pair.GetName()] = pair.GetValue()
			}
			if maps.Equal(pairs, labels) {
				total += metric.GetCounter().GetValue()
			}
		}
	}
	return total
}

// item is the labels of the built-in model's series, with more labels added.
func item(name string, more ...string) map[string]string {
	provider, model, _ := strings.Cut(name, "/")
	labels := map[string]string{"modality": "llm", "provider": provider, "model": model}
	for i := 0; i+1 < len(more); i += 2 {
		labels[more[i]] = more[i+1]
	}
	return labels
}

// briefly is a request whose prompt is 101 tokens by the router's own count.
func briefly() llm.ResponseParams {
	return llm.ResponseParams{Instructions: strings.Repeat("Answer briefly. ", 25), Input: prompt()}
}

func (s *RecordingSuite) TestAnAnswerWithoutUsageIsEstimated() {
	missing := item("openai/gpt-6-luna", "reason", "missing")
	before := s.counted("router_usage_estimated_total", missing)

	row, told := s.recordedFor("openai/gpt-6-luna", briefly(), llm.Response{
		Status:     llm.StatusCompleted,
		OutputText: "Paris is the capital of France.",
	})

	s.EqualValues(101, row.InputTokens, "the prompt as the router counts it")
	s.EqualValues(8, row.OutputTokens, "and the answer")
	s.Equal(map[string]string{routing.AttributeUsageEstimated: "true"}, row.Attributes)
	s.True(row.Billable)
	s.Positive(row.CostMicros)
	s.Equal(row.CostMicros, told, "the caller is told what the estimate costs")
	s.Equal(before+1, s.counted("router_usage_estimated_total", missing))
	s.Contains(s.logs.String(), `msg="billable work recorded an estimate" modality=llm provider=openai model=gpt-6-luna`)
}

func (s *RecordingSuite) TestACutResponseIsEstimatedWithoutAnAlarm() {
	cancelled := item("openai/gpt-6-luna", "reason", "cancelled")
	missing := item("openai/gpt-6-luna", "reason", "missing")
	before, alarmed := s.counted("router_usage_estimated_total", cancelled), s.counted("router_usage_estimated_total", missing)

	row, _ := s.recordedFor("openai/gpt-6-luna", briefly(), llm.Response{
		Status:     llm.StatusCancelled,
		OutputText: "Paris is",
	})

	s.Equal(routing.ErrorCancelled, row.ErrorCode)
	s.EqualValues(101, row.InputTokens, "the vendor read the whole prompt before the cut")
	s.EqualValues(2, row.OutputTokens, "and wrote what was forwarded")
	s.Equal(map[string]string{routing.AttributeUsageEstimated: "true"}, row.Attributes)
	s.Positive(row.CostMicros)
	s.Equal(before+1, s.counted("router_usage_estimated_total", cancelled))
	s.Equal(alarmed, s.counted("router_usage_estimated_total", missing), "a cut is expected, not an adapter to fix")
	s.NotContains(s.logs.String(), "level=ERROR")
}

func (s *RecordingSuite) TestAPartialUsageFrameIsKeptOnACut() {
	row, _ := s.recordedFor("openai/gpt-6-luna", briefly(), llm.Response{
		Status:        llm.StatusCancelled,
		OutputText:    "Paris is",
		Usage:         llm.Usage{InputTokens: 120, OutputTokens: 3, TotalTokens: 123},
		UsageReported: true,
	})

	s.EqualValues(120, row.InputTokens, "the vendor's own count up to the cut")
	s.EqualValues(3, row.OutputTokens)
	s.Empty(row.Attributes, "and it is not marked as an estimate")
}

func (s *RecordingSuite) TestAVendorTotalThatDoesNotAddUpIsCounted() {
	luna := item("openai/gpt-6-luna")
	before := s.counted("router_usage_mismatch_total", luna)

	s.recorded("openai/gpt-6-luna", llm.Response{
		Status:        llm.StatusCompleted,
		OutputText:    "4",
		Usage:         llm.Usage{InputTokens: 10, OutputTokens: 4, TotalTokens: 20},
		UsageReported: true,
	})

	s.Equal(before+1, s.counted("router_usage_mismatch_total", luna))
	s.Contains(s.logs.String(), "vendor_total_tokens=20 input_tokens=10 output_tokens=4")
}

// TestReasoningTokensAreAnAttribute is xAI's "think" frame as openaicompat reports it: the
// thinking inside the output, its count shown beside it, and xAI's own cost kept to check the
// price by.
func (s *RecordingSuite) TestReasoningTokensAreAnAttribute() {
	grok := item("xai/grok-4.7")
	before := s.counted("router_usage_mismatch_total", grok)

	row, _ := s.recorded("xai/grok-4.7", llm.Response{
		Status:     llm.StatusCompleted,
		OutputText: "4",
		Usage: llm.Usage{
			InputTokens:         1_301,
			InputTokensDetails:  llm.InputTokensDetails{CachedTokens: 1_152},
			OutputTokens:        1_446,
			OutputTokensDetails: llm.OutputTokensDetails{ReasoningTokens: 1_425},
			TotalTokens:         2_747,
			VendorCostMicros:    9_550,
		},
		UsageReported: true,
	})

	s.Equal(map[string]string{"reasoning_tokens": "1425", store.VendorCostAttribute: "9550"}, row.Attributes)
	s.Empty(row.Meters, "reasoning is priced inside the output tokens, never on its own")
	s.EqualValues(1_446, row.OutputTokens)
	s.EqualValues(9_550, row.BaseCostMicros, "at the vendor's rates, what xAI said it charged")
	s.Equal(before, s.counted("router_usage_mismatch_total", grok), "the tokens add up to xAI's total")
}
