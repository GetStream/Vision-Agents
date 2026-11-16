package llmrouter

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/xai"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// The names of the OpenTelemetry GenAI inference span, as
// github.com/open-telemetry/semantic-conventions-genai defines them at commit e6658f14c9d9
// (2026-11-06), docs/gen-ai/client-inference.md. The conventions are still in development and
// that repository has no release yet, so they are pinned here: the last Go semconv package
// with GenAI names, v1.41.0, still says cache_creation where the repository now says
// cache_write.
const (
	genAIOperationName    = attribute.Key("gen_ai.operation.name")
	genAIProviderName     = attribute.Key("gen_ai.provider.name")
	genAIRequestModel     = attribute.Key("gen_ai.request.model")
	genAIInputTokens      = attribute.Key("gen_ai.usage.input_tokens")
	genAIOutputTokens     = attribute.Key("gen_ai.usage.output_tokens")
	genAICacheReadTokens  = attribute.Key("gen_ai.usage.cache_read.input_tokens")
	genAICacheWriteTokens = attribute.Key("gen_ai.usage.cache_write.input_tokens")
	genAIReasoningTokens  = attribute.Key("gen_ai.usage.reasoning.output_tokens")
	genAIErrorType        = attribute.Key("error.type")
)

// genAIChat is the operation every model call is: one response to a conversation.
const genAIChat = "chat"

// genAIProviders are the providers the conventions' list of well-known values names other than
// the router does, which then MUST be used: Gemini as gcp.gemini for its generativelanguage
// endpoint, xAI as x_ai. Any other provider goes by its own name, as the list allows.
var genAIProviders = map[string]string{
	gemini.ProviderName: "gcp.gemini",
	xai.ProviderName:    "x_ai",
}

// startInference opens the span one model call is traced under, from the request to the end of
// its response, named as the conventions ask: the operation, then the model. Every provider
// config names a model (ModalityConfig.validate). The span is a sibling of the call's other
// spans rather than their parent, so the traces they were already in keep their shape.
func startInference(ctx context.Context, config routing.ProviderConfig) trace.Span {
	provider := config.Provider
	if named, ok := genAIProviders[provider]; ok {
		provider = named
	}
	_, span := tracer.Start(ctx, genAIChat+" "+config.Model, trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			genAIOperationName.String(genAIChat),
			genAIProviderName.String(provider),
			genAIRequestModel.String(config.Model)))
	return span
}

// endInference ends a model call's span with what its response used and, when it did not run
// to its end, the router's error code for why: a bounded set, as error.type asks. A response
// its caller closed is no failure, as tracing.Fail has it. Usage is set only as the provider
// reported it, which is what the conventions count: the router's own estimate, which a row is
// billed on when the provider said nothing, is counted only in router_tokens_total.
func endInference(span trace.Span, response *llm.Response, code string) {
	defer span.End()
	if response != nil && response.UsageReported {
		usage := response.Usage
		span.SetAttributes(genAIInputTokens.Int64(usage.InputTokens), genAIOutputTokens.Int64(usage.OutputTokens))
		// Each of these is recommended only when it is non-zero.
		for key, count := range map[attribute.Key]int64{
			genAICacheReadTokens:  usage.InputTokensDetails.CachedTokens,
			genAICacheWriteTokens: usage.InputTokensDetails.CacheWriteTokens,
			genAIReasoningTokens:  usage.OutputTokensDetails.ReasoningTokens,
		} {
			if count != 0 {
				span.SetAttributes(key.Int64(count))
			}
		}
	}
	if code == "" || code == routing.ErrorCancelled {
		return
	}
	span.SetAttributes(genAIErrorType.String(code))
	span.SetStatus(codes.Error, code)
}
