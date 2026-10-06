// Package tracing records what a request spent its time on and, where a deployment names
// a collector, sends it there over OTLP.
//
// Nothing is exported unless OTEL_EXPORTER_OTLP_ENDPOINT or its traces-only counterpart is
// set. Until one is, the global provider stays the no-op OpenTelemetry installs by default
// and a span costs an interface call, which is what lets the instrumentation sit on the
// hot path unconditionally.
package tracing

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

// scope prefixes every tracer name, so a span says which package opened it.
const scope = "github.com/GetStream/Vision-Agents/acceleration/"

// Tracer returns the tracer a package records its spans with. It is safe to call at
// package initialisation: the global provider delegates to whatever Setup installs later.
func Tracer(name string) trace.Tracer { return otel.Tracer(scope + name) }

// Setup installs the global tracer provider and returns the function that flushes it.
//
// A deployment that names no collector gets a shutdown that does nothing and keeps the
// no-op provider, rather than a provider buffering spans nobody will ever read.
func Setup(ctx context.Context, service, version string) (func(context.Context) error, error) {
	// The same two variables the OTLP exporter reads for itself. They are checked here as
	// well because the exporter defaults to localhost rather than refusing, and a router
	// retrying against a collector nobody deployed is worse than no traces.
	if endpoint("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == "" && endpoint("OTEL_EXPORTER_OTLP_ENDPOINT") == "" {
		return func(context.Context) error { return nil }, nil
	}

	exporter, err := otlptracehttp.New(ctx)
	if err != nil {
		return nil, fmt.Errorf("tracing: otlp exporter: %w", err)
	}

	attributes := []attribute.KeyValue{semconv.ServiceName(service)}
	if version != "" {
		attributes = append(attributes, semconv.ServiceVersion(version))
	}
	// Merge against the default resource rather than replacing it, so the host and the
	// process the collector groups by are still there.
	described, err := resource.Merge(resource.Default(), resource.NewWithAttributes(
		semconv.SchemaURL, attributes...))
	if err != nil {
		return nil, fmt.Errorf("tracing: describe this service: %w", err)
	}

	// Sampling is left to OTEL_TRACES_SAMPLER and its argument, which is what an operator
	// turning the volume down at a busy deployment reaches for.
	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(described),
	)
	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{}, propagation.Baggage{}))

	return provider.Shutdown, nil
}

// Fail marks a span as the failure it ended in. A context cancelled on the way out is not
// one: it is the caller hanging up, and a trace full of red for it hides the real errors.
func Fail(span trace.Span, err error) {
	if err == nil || errors.Is(err, context.Canceled) {
		return
	}
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
}

func endpoint(name string) string { return strings.TrimSpace(os.Getenv(name)) }
