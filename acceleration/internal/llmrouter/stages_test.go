package llmrouter

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

type StagesSuite struct {
	suite.Suite
}

func TestStages(t *testing.T) { suite.Run(t, new(StagesSuite)) }

// answer is one chunk of text, as an OpenAI-compatible provider streams it.
const answer = "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n"

// done ends an OpenAI-compatible stream.
const done = "data: [DONE]\n\n"

// provider is a TLS provider that waits beforeHeaders before its headers, beforeToken before it
// streams body and afterToken before it ends the stream, each cut short when the request is
// cancelled.
func (s *StagesSuite) provider(beforeHeaders, beforeToken, afterToken time.Duration, body string) *httptest.Server {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		if !pause(r.Context(), beforeHeaders) {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		if !pause(r.Context(), beforeToken) {
			return
		}
		fmt.Fprint(w, body)
		w.(http.Flusher).Flush()
		if !pause(r.Context(), afterToken) {
			return
		}
		fmt.Fprint(w, done)
	}))
	s.T().Cleanup(server.Close)
	return server
}

func pause(ctx context.Context, d time.Duration) bool {
	select {
	case <-time.After(d):
		return true
	case <-ctx.Done():
		return false
	}
}

// config is a provider and model no other test observes stages under.
func (s *StagesSuite) config() routing.ProviderConfig {
	return routing.ProviderConfig{Provider: "stages-" + rand.Text(), Model: "model-" + rand.Text(), Languages: []string{"en"}}
}

// session asks server under config, with nothing in front of it but the router's own steps.
func (s *StagesSuite) session(server *httptest.Server, config routing.ProviderConfig) *Session {
	provider, err := Started(openaicompat.New(openaicompat.Options{
		Provider: config.Provider, Model: config.Model, APIKey: "test",
		BaseURL: server.URL + "/v1", HTTPClient: server.Client(),
	}))
	s.Require().NoError(err)
	session := newSession(provider, config, routing.Owner{CustomerID: "acme"}, nil, nil)
	session.recorder = &statSink{}
	s.T().Cleanup(func() { _ = session.Close() })
	return session
}

func (s *StagesSuite) ask(ctx context.Context, session *Session) llm.Response {
	stream, err := session.Create(ctx, llm.ResponseParams{Input: prompt()})
	s.Require().NoError(err)
	response, err := llm.Collect(stream)
	s.Require().NoError(err)
	return response
}

// series is the metric of that name with those labels, nil when nothing was observed under them.
func (s *StagesSuite) series(name string, labels map[string]string) *dto.Metric {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			matched := 0
			for _, pair := range metric.GetLabel() {
				if labels[pair.GetName()] == pair.GetValue() {
					matched++
				}
			}
			if matched == len(labels) {
				return metric
			}
		}
	}
	return nil
}

// stage is how many requests router_llm_stage_seconds observed a stage of, and their total.
func (s *StagesSuite) stage(config routing.ProviderConfig, model, stage string) (uint64, time.Duration) {
	histogram := s.series("router_llm_stage_seconds",
		map[string]string{"provider": config.Provider, "model": model, "stage": stage}).GetHistogram()
	return histogram.GetSampleCount(), time.Duration(histogram.GetSampleSum() * float64(time.Second))
}

// connections is router_llm_connections_total for a provider.
func (s *StagesSuite) connections(provider string, reused bool) float64 {
	return s.series("router_llm_connections_total",
		map[string]string{"provider": provider, "reused": fmt.Sprint(reused)}).GetCounter().GetValue()
}

func (s *StagesSuite) TestEachStageIsTheTimeSpentInItAlone() {
	// Each stage takes a different multiple of unit, and a stage that took in any other's time
	// is one unit or more off. The client stamps a byte when its goroutine reads it, so a busy
	// machine moves the line between two stages by a little. The router's stage is admission and
	// the screen's start; the stream goes on after its first token.
	const unit, slack = 200 * time.Millisecond, 50 * time.Millisecond
	const admitting, screening, dialling, beforeHeaders, beforeToken, afterToken = unit, 2 * unit, 2 * unit, 4 * unit, 8 * unit, unit
	config := s.config()
	server := s.provider(beforeHeaders, beforeToken, afterToken, answer)
	client := server.Client()
	transport := client.Transport.(*http.Transport)
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		time.Sleep(dialling)
		return (&net.Dialer{}).DialContext(ctx, network, address)
	}
	session := s.session(server, config)
	session.admit = func(context.Context, routing.Owner) (routing.Admission, error) {
		time.Sleep(admitting)
		return routing.Admission{}, nil
	}
	session.screen = func(context.Context, routing.Owner, []llm.Message) <-chan error {
		time.Sleep(screening)
		passed := make(chan error)
		close(passed)
		return passed
	}

	s.Equal("hi", s.ask(s.T().Context(), session).OutputText)

	for stage, least := range map[string]time.Duration{
		routing.StageRouter:  admitting + screening,
		routing.StageConnect: dialling,
		routing.StageWait:    beforeHeaders,
		routing.StageStream:  beforeToken,
	} {
		count, took := s.stage(config, config.Model, stage)
		s.EqualValues(1, count, stage)
		s.GreaterOrEqual(took, least-slack, stage)
		s.Less(took, least+unit-slack, stage)
	}
	s.Equal(1.0, s.connections(config.Provider, false))
	s.Zero(s.connections(config.Provider, true))
}

func (s *StagesSuite) TestAReusedConnectionIsCountedButHasNoConnectStage() {
	config := s.config()
	session := s.session(s.provider(0, 0, 0, answer), config)

	s.ask(s.T().Context(), session)
	s.ask(s.T().Context(), session)

	connects, _ := s.stage(config, config.Model, routing.StageConnect)
	waits, _ := s.stage(config, config.Model, routing.StageWait)
	s.EqualValues(1, connects, "only the first request dialled")
	s.EqualValues(2, waits)
	s.Equal(1.0, s.connections(config.Provider, false))
	s.Equal(1.0, s.connections(config.Provider, true))
}

func (s *StagesSuite) TestACustomModelIsObservedUnderCustom() {
	config := s.config()
	config.Provider = routing.CustomProvider
	before, _ := s.stage(config, routing.CustomProvider, routing.StageWait)
	session := s.session(s.provider(0, 0, 0, answer), config)

	s.ask(s.T().Context(), session)

	after, _ := s.stage(config, routing.CustomProvider, routing.StageWait)
	s.EqualValues(1, after-before)
	named, _ := s.stage(config, config.Model, routing.StageWait)
	s.Zero(named, "a customer's own model name is no label")
}

func (s *StagesSuite) TestARequestCancelledBeforeItsHeadersHasNoWaitOrStream() {
	config := s.config()
	session := s.session(s.provider(time.Minute, 0, 0, answer), config)
	ctx, cancel := context.WithTimeout(s.T().Context(), 200*time.Millisecond)
	defer cancel()

	_, err := session.Create(ctx, llm.ResponseParams{Input: prompt()})
	s.Require().Error(err)

	for stage, want := range map[string]uint64{
		routing.StageRouter: 1, routing.StageConnect: 1, routing.StageWait: 0, routing.StageStream: 0,
	} {
		count, _ := s.stage(config, config.Model, stage)
		s.Equal(want, count, stage)
	}
}

func (s *StagesSuite) TestARequestRefusedAfterItsHeadersHasAWaitButNoStream() {
	config := s.config()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"no","type":"invalid_request_error"}}`)
	}))
	s.T().Cleanup(server.Close)
	session := s.session(server, config)

	_, err := session.Create(s.T().Context(), llm.ResponseParams{Input: prompt()})
	s.Require().Error(err)

	waits, _ := s.stage(config, config.Model, routing.StageWait)
	streams, _ := s.stage(config, config.Model, routing.StageStream)
	s.EqualValues(1, waits, "the provider took that long to answer, if only with a refusal")
	s.Zero(streams)
}

func (s *StagesSuite) TestARetryThatGotNoHeadersHasNoWaitFromTheTryBefore() {
	config := s.config()
	var tries atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		if tries.Add(1) == 1 {
			// Asked again at once: the client honours how soon the provider says to retry.
			w.Header().Set("Retry-After-Ms", "1")
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		<-r.Context().Done()
	}))
	s.T().Cleanup(server.Close)
	session := s.session(server, config)
	ctx, cancel := context.WithCancel(s.T().Context())
	go func() {
		for tries.Load() < 2 {
			time.Sleep(time.Millisecond)
		}
		cancel()
	}()

	_, err := session.Create(ctx, llm.ResponseParams{Input: prompt()})
	s.Require().Error(err)

	s.EqualValues(2, tries.Load())
	waits, _ := s.stage(config, config.Model, routing.StageWait)
	s.Zero(waits, "the request that counts never heard back")
}

func (s *StagesSuite) TestARequestAnsweredBeforeItWasWrittenHasNoWait() {
	// A provider may answer, a 413 say, while the request is still being written; the write
	// then fails, or ends after the answer was read. The hooks run as the transport calls them.
	for name, answered := range map[string]func(*httptrace.ClientTrace){
		"still writing": func(trace *httptrace.ClientTrace) { trace.GotFirstResponseByte() },
		"write failed": func(trace *httptrace.ClientTrace) {
			trace.WroteRequest(httptrace.WroteRequestInfo{Err: errors.New("broken pipe")})
			trace.GotFirstResponseByte()
		},
		"written after the answer": func(trace *httptrace.ClientTrace) {
			trace.GotFirstResponseByte()
			time.Sleep(time.Millisecond)
			trace.WroteRequest(httptrace.WroteRequestInfo{})
		},
	} {
		config := s.config()
		ctx, stages := traceStages(s.T().Context())
		trace := httptrace.ContextClientTrace(ctx)
		trace.GetConn("provider:443")
		trace.GotConn(httptrace.GotConnInfo{})
		answered(trace)

		stages.observe(config)

		waits, _ := s.stage(config, config.Model, routing.StageWait)
		s.Zero(waits, name)
	}
}

func (s *StagesSuite) TestAnAnswerReadAfterTheAttemptWasCancelledHasNoWait() {
	// Cancelling closes the connection, and the transport can still read the provider's answer
	// in that window. The hooks run as the transport calls them.
	config := s.config()
	ctx, cancel := context.WithCancel(s.T().Context())
	ctx, stages := traceStages(ctx)
	trace := httptrace.ContextClientTrace(ctx)
	trace.GetConn("provider:443")
	trace.GotConn(httptrace.GotConnInfo{})
	trace.WroteRequest(httptrace.WroteRequestInfo{})
	cancel()
	trace.GotFirstResponseByte()

	stages.observe(config)

	waits, _ := s.stage(config, config.Model, routing.StageWait)
	s.Zero(waits)
}

func (s *StagesSuite) TestAnAnswerThatNeverSaidAnythingHasNoStream() {
	config := s.config()
	session := s.session(s.provider(0, 0, 0, ""), config)

	s.ask(s.T().Context(), session)

	waits, _ := s.stage(config, config.Model, routing.StageWait)
	streams, _ := s.stage(config, config.Model, routing.StageStream)
	s.EqualValues(1, waits)
	s.Zero(streams)
}

func (s *StagesSuite) TestAProviderThatMakesNoHTTPRequestHasOnlyTheRoutersStage() {
	config := s.config()
	provider := newStubLLM()
	session := newSession(provider, config, routing.Owner{CustomerID: "acme"}, nil, nil)
	session.recorder = &statSink{}
	s.T().Cleanup(func() { _ = session.Close() })

	stream, err := session.Create(s.T().Context(), llm.ResponseParams{Input: prompt()})
	s.Require().NoError(err)
	provider.script(0).OutputText("hi")
	provider.script(0).Done()
	s.Equal("hi", drain(stream).OutputText)

	for stage, want := range map[string]uint64{
		routing.StageRouter: 1, routing.StageConnect: 0, routing.StageWait: 0, routing.StageStream: 0,
	} {
		count, _ := s.stage(config, config.Model, stage)
		s.Equal(want, count, stage)
	}
	s.Zero(s.connections(config.Provider, false))
	s.Zero(s.connections(config.Provider, true))
}

func (s *StagesSuite) TestATraceTheCallerSetStillRuns() {
	config := s.config()
	session := s.session(s.provider(0, 0, 0, answer), config)
	var heard atomic.Bool
	ctx := httptrace.WithClientTrace(s.T().Context(), &httptrace.ClientTrace{
		GotFirstResponseByte: func() { heard.Store(true) },
	})

	s.ask(ctx, session)

	s.True(heard.Load())
	waits, _ := s.stage(config, config.Model, routing.StageWait)
	s.EqualValues(1, waits)
}

func (s *StagesSuite) TestEachHedgedAttemptRecordsItsOwnStages() {
	stalled := s.provider(time.Minute, 0, 0, answer)
	quick := s.provider(0, 0, 0, answer)
	original, hedge := s.config(), s.config()
	registry := NewRegistry()
	for _, candidate := range []struct {
		config routing.ProviderConfig
		server *httptest.Server
	}{{original, stalled}, {hedge, quick}} {
		registry.Register(candidate.config.Provider, func(spec routing.Spec) (Provider, error) {
			return Started(openaicompat.New(openaicompat.Options{
				Provider: candidate.config.Provider, Model: spec.Model, APIKey: "test",
				BaseURL: candidate.server.URL + "/v1", HTTPClient: candidate.server.Client(),
			}))
		})
	}
	members := []string{original.Provider + "/" + original.Model, hedge.Provider + "/" + hedge.Model}
	router, err := New(Options{
		Registry:   registry,
		ReplyHedge: 100 * time.Millisecond,
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{original, hedge},
			Aliases:   map[string]routing.Alias{"voice": {Prefer: members[0], Only: members}},
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	session, err := router.Start(s.T().Context(), Request{CustomerID: "acme", Target: "voice"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	stream, err := session.Create(s.T().Context(), llm.ResponseParams{Purpose: replyPurpose, Input: prompt()})
	s.Require().NoError(err)
	_, err = llm.Collect(stream)
	s.Require().NoError(err)

	// The original is cancelled when it loses, and records what it passed once its Create returns.
	s.Eventually(func() bool {
		connects, _ := s.stage(original, original.Model, routing.StageConnect)
		return connects == 1
	}, 5*time.Second, 10*time.Millisecond)
	for _, want := range []struct {
		config routing.ProviderConfig
		stage  string
		count  uint64
	}{
		{original, routing.StageRouter, 1},
		{original, routing.StageConnect, 1},
		{original, routing.StageWait, 0},
		{hedge, routing.StageRouter, 0},
		{hedge, routing.StageConnect, 1},
		{hedge, routing.StageWait, 1},
		{hedge, routing.StageStream, 1},
	} {
		count, _ := s.stage(want.config, want.config.Model, want.stage)
		s.Equal(want.count, count, "%s %s", want.config.Provider, want.stage)
	}
}

func (s *StagesSuite) TestConcurrentRequestsEachRecordTheirOwnStages() {
	const requests = 8
	config := s.config()
	session := s.session(s.provider(10*time.Millisecond, 10*time.Millisecond, 0, answer), config)

	var wg sync.WaitGroup
	for range requests {
		wg.Go(func() {
			stream, err := session.Create(s.T().Context(), llm.ResponseParams{Input: prompt()})
			if s.NoError(err) {
				_, err = llm.Collect(stream)
				s.NoError(err)
			}
		})
	}
	wg.Wait()

	waits, _ := s.stage(config, config.Model, routing.StageWait)
	streams, _ := s.stage(config, config.Model, routing.StageStream)
	s.EqualValues(requests, waits)
	s.EqualValues(requests, streams)
	s.Equal(float64(requests), s.connections(config.Provider, false)+s.connections(config.Provider, true))
}
