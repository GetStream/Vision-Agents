package llmrouter

import (
	"context"
	"crypto/rand"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/suite"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// recordingSpans is the provider the package's tracers record into, installed once per process:
// a tracer taken at package initialisation is handed to the first provider installed and to no
// later one (delegateTraceOnce in otel's internal/global/state.go).
var recordingSpans = sync.OnceValue(func() *sdktrace.TracerProvider {
	provider := sdktrace.NewTracerProvider()
	otel.SetTracerProvider(provider)
	return provider
})

type GenAISuite struct {
	suite.Suite
	recorded *tracetest.SpanRecorder
}

func TestGenAI(t *testing.T) { suite.Run(t, new(GenAISuite)) }

func (s *GenAISuite) SetupTest() {
	s.recorded = tracetest.NewSpanRecorder()
	recordingSpans().RegisterSpanProcessor(s.recorded)
	recorded := s.recorded
	s.T().Cleanup(func() { recordingSpans().UnregisterSpanProcessor(recorded) })
}

// session is a session over a stub provider routed as provider, under a model no other test
// traces, which is what the test finds its spans by.
func (s *GenAISuite) session(provider string) (*Session, *stubLLM, string) {
	stub := newStubLLM()
	model := "model-" + rand.Text()
	session := newSession(stub, routing.ProviderConfig{Provider: provider, Model: model},
		routing.Owner{CustomerID: "acme"}, nil, nil)
	session.recorder = &statSink{}
	s.T().Cleanup(func() { _ = session.Close() })
	return session, stub, model
}

// ended is the ended span of that name, nil when none has ended.
func (s *GenAISuite) ended(name string) sdktrace.ReadOnlySpan {
	var found sdktrace.ReadOnlySpan
	for _, span := range s.recorded.Ended() {
		if span.Name() == name {
			s.Require().Nil(found, "one model call is one span")
			found = span
		}
	}
	return found
}

// inference is the model call's span, which the test needs to have ended.
func (s *GenAISuite) inference(model string) sdktrace.ReadOnlySpan {
	span := s.ended("chat " + model)
	s.Require().NotNil(span, "the model call has a span named for its operation and model")
	return span
}

func attributes(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	values := map[attribute.Key]attribute.Value{}
	for _, kv := range span.Attributes() {
		values[kv.Key] = kv.Value
	}
	return values
}

func (s *GenAISuite) TestAModelCallIsTracedUnderTheGenAINamesWithWhatItsProviderReported() {
	session, stub, model := s.session("stub-" + rand.Text())
	provider := session.Provider()

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	s.Require().NoError(err)
	stub.script(0).OutputText("hello")
	stub.script(0).Usage(llm.Usage{
		InputTokens:         120,
		InputTokensDetails:  llm.InputTokensDetails{CachedTokens: 50, CacheWriteTokens: 25},
		OutputTokens:        30,
		OutputTokensDetails: llm.OutputTokensDetails{ReasoningTokens: 10},
	})
	s.Require().True(stream.Next())
	s.Nil(s.ended("chat "+model), "the span covers the whole response, not only the request")
	stub.script(0).Done()
	drain(stream)

	span := s.inference(model)
	s.Equal(trace.SpanKindClient, span.SpanKind(), "the vendor is called over the network")
	s.Equal(map[attribute.Key]attribute.Value{
		"gen_ai.operation.name":                 attribute.StringValue("chat"),
		"gen_ai.provider.name":                  attribute.StringValue(provider),
		"gen_ai.request.model":                  attribute.StringValue(model),
		"gen_ai.usage.input_tokens":             attribute.Int64Value(120),
		"gen_ai.usage.output_tokens":            attribute.Int64Value(30),
		"gen_ai.usage.cache_read.input_tokens":  attribute.Int64Value(50),
		"gen_ai.usage.cache_write.input_tokens": attribute.Int64Value(25),
		"gen_ai.usage.reasoning.output_tokens":  attribute.Int64Value(10),
	}, attributes(span))
	s.Equal(codes.Unset, span.Status().Code)
}

func (s *GenAISuite) TestGeminiAndXAIGoByTheConventionsWellKnownProviderNames() {
	for provider, named := range map[string]string{"gemini": "gcp.gemini", "xai": "x_ai"} {
		session, stub, model := s.session(provider)
		stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
		s.Require().NoError(err)
		stub.script(0).Done()
		drain(stream)

		s.Equal(attribute.StringValue(named), attributes(s.inference(model))["gen_ai.provider.name"], provider)
	}
}

func (s *GenAISuite) TestTokenKindsTheProviderReportedNoneOfAreLeftOff() {
	session, stub, model := s.session("stub-" + rand.Text())

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	s.Require().NoError(err)
	stub.script(0).OutputText("hello")
	stub.script(0).Usage(llm.Usage{InputTokens: 12, OutputTokens: 3})
	stub.script(0).Done()
	drain(stream)

	traced := attributes(s.inference(model))
	s.Equal(attribute.Int64Value(12), traced["gen_ai.usage.input_tokens"])
	s.Equal(attribute.Int64Value(3), traced["gen_ai.usage.output_tokens"])
	for _, key := range []attribute.Key{
		"gen_ai.usage.cache_read.input_tokens",
		"gen_ai.usage.cache_write.input_tokens",
		"gen_ai.usage.reasoning.output_tokens",
	} {
		s.NotContains(traced, key, "recommended only when it is non-zero")
	}
}

func (s *GenAISuite) TestAResponseWhoseProviderReportedNoUsageCarriesNoTokenCounts() {
	session, stub, model := s.session("stub-" + rand.Text())

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	s.Require().NoError(err)
	stub.script(0).OutputText("an answer the router bills on its own count")
	stub.script(0).Done()
	drain(stream)

	traced := attributes(s.inference(model))
	for key := range traced {
		s.NotContains(string(key), "gen_ai.usage.", "an estimate is not what the provider reported")
	}
	s.Contains(traced, attribute.Key("gen_ai.request.model"))
}

func (s *GenAISuite) TestAModelCallThatCouldNotBeMadeEndsAsAnErrorOfItsCode() {
	session, stub, model := s.session("stub-" + rand.Text())
	stub.createErr = errors.New("stub: vendor down")

	_, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	s.Require().Error(err)

	span := s.inference(model)
	s.Equal(attribute.StringValue("create_failed"), attributes(span)["error.type"])
	s.Equal(codes.Error, span.Status().Code)
}

func (s *GenAISuite) TestAResponseThatFailedPartWayCarriesItsErrorAndWhatItUsed() {
	session, stub, model := s.session("stub-" + rand.Text())

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	s.Require().NoError(err)
	stub.script(0).OutputText("half an ans")
	stub.script(0).Usage(llm.Usage{InputTokens: 9, OutputTokens: 4})
	stub.script(0).Fail(errors.New("stub: connection reset"), "read")
	drain(stream)

	span := s.inference(model)
	traced := attributes(span)
	s.Equal(attribute.StringValue("provider_error"), traced["error.type"])
	s.Equal(attribute.Int64Value(4), traced["gen_ai.usage.output_tokens"])
	s.Equal(codes.Error, span.Status().Code)
}

// Catches an original that lost its hedge being traced without the timeout its row carries.
func (s *GenAISuite) TestAStalledAttemptThatLostItsHedgeIsATimeoutOnItsSpan() {
	synctest.Test(s.T(), func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha", "beta")
		asked := r.ask(context.Background(), r.params())
		pass(lateAfter)
		pass(300 * time.Millisecond)
		r.script("beta").Usage(llm.Usage{InputTokens: 7, OutputTokens: 2})
		r.say("beta", "Hello")
		s.Require().True(asked.answered())
		r.finish("beta", asked)
		s.Require().Equal(map[string]string{"alpha": routing.ErrorTimeout, "beta": "served"}, r.called())
	})

	stalled, served := s.inference("quick"), s.inference("steady")
	s.Equal(attribute.StringValue(routing.ErrorTimeout), attributes(stalled)["error.type"])
	s.Equal(codes.Error, stalled.Status().Code)
	s.NotContains(attributes(stalled), attribute.Key("gen_ai.usage.input_tokens"), "the winner's tokens are not the loser's")
	s.NotContains(attributes(served), attribute.Key("error.type"))
	s.Equal(attribute.Int64Value(7), attributes(served)["gen_ai.usage.input_tokens"])
}

// Catches a refused customer key being traced as create_failed, not as the key_refused its row carries.
func (s *GenAISuite) TestAKeyTheProviderRefusedIsKeyRefusedOnItsSpan() {
	refusing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("invalid api key"))
	}))
	s.T().Cleanup(refusing.Close)
	model := "model-" + rand.Text()
	provider, err := Started(openaicompat.New(openaicompat.Options{
		Provider: "gemma", Model: model, APIKey: "k", BaseURL: refusing.URL,
	}))
	s.Require().NoError(err)
	session := newSession(provider, routing.ProviderConfig{Provider: "gemma", Model: model, BYOK: true},
		routing.Owner{CustomerID: "acme"}, nil, nil)
	session.recorder = &statSink{}
	s.T().Cleanup(func() { _ = session.Close() })

	_, err = session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	s.Require().Error(err)

	span := s.inference(model)
	s.Equal(attribute.StringValue(routing.ErrorKeyRefused), attributes(span)["error.type"])
	s.Equal(codes.Error, span.Status().Code)
}

func (s *GenAISuite) TestAResponseItsCallerClosedIsNoError() {
	session, stub, model := s.session("stub-" + rand.Text())

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	s.Require().NoError(err)
	stub.script(0).OutputText("half a sen")
	s.Require().True(stream.Next())
	s.Require().NoError(stream.Close())
	s.Require().Equal(llm.StatusCancelled, drain(stream).Status)

	span := s.inference(model)
	s.NotContains(attributes(span), attribute.Key("error.type"), "barge-in is the caller's choice")
	s.Equal(codes.Unset, span.Status().Code)
}

func (s *GenAISuite) TestTheModelCallSpanLeavesTheCallsOtherSpansWhereTheyWere() {
	session, stub, model := s.session("stub-" + rand.Text())
	ctx, root := recordingSpans().Tracer("genai-test").Start(context.Background(), "caller")

	stream, err := session.Create(ctx, llm.ResponseParams{ID: "r1", Input: prompt()})
	s.Require().NoError(err)
	stub.script(0).Done()
	drain(stream)
	root.End()

	var create, providerCreate sdktrace.ReadOnlySpan
	for _, span := range s.recorded.Ended() {
		if span.SpanContext().TraceID() != root.SpanContext().TraceID() {
			continue
		}
		switch span.Name() {
		case "llm.create":
			create = span
		case "llm.provider.create":
			providerCreate = span
		}
	}
	s.Require().NotNil(create)
	s.Require().NotNil(providerCreate)
	s.Equal(create.SpanContext().SpanID(), providerCreate.Parent().SpanID(),
		"llm.provider.create stays under llm.create")
	s.Equal(create.SpanContext().SpanID(), s.inference(model).Parent().SpanID(),
		"the model call's span sits beside it, in the same trace")
}
