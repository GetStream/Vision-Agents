package agent

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
)

// tracer records what a turn spent its wait on.
var tracer = tracing.Tracer("internal/agent")

// traceTurn records a finished turn as a span and its stages as children.
//
// The spans are written after the fact, from the moments the tracker already holds,
// rather than opened where each stage starts: a turn is assembled by four goroutines and
// a span handed between them would be a lifetime to get wrong for something nobody reads
// until the turn is over. Giving the spans their real timestamps makes the two the same
// picture.
//
// A stage that never happened is left out rather than recorded as instant, the way the
// rows leave it null: a turn interrupted before the model answered has no reply to show.
func traceTurn(owner routing.Owner, turnID string, current *openTurn) {
	ended := lastMoment(current)
	if current.transcriptAt.IsZero() || ended.IsZero() {
		return
	}

	ctx, turn := tracer.Start(context.Background(), "agent.turn",
		trace.WithTimestamp(current.transcriptAt),
		trace.WithAttributes(
			attribute.String("turn.id", turnID),
			attribute.String("agent.id", owner.AgentID),
			attribute.String("call.id", owner.CallID),
			attribute.String("participant.id", current.participant.ID),
			attribute.Bool("turn.interrupted", current.interrupted),
		))
	defer turn.End(trace.WithTimestamp(ended))

	// Speech-to-text is the one stage with no pair of moments: what the transcriber spent
	// settling the turn is reported as a duration, and it ran up to the transcript.
	if current.sttLatencyMs > 0 {
		stage(ctx, "turn.stt",
			current.transcriptAt.Add(-time.Duration(current.sttLatencyMs)*time.Millisecond),
			current.transcriptAt)
	}
	stage(ctx, "turn.cadence", current.transcriptAt, current.readyAt)
	decisionAt := current.decisionAt
	if decisionAt.IsZero() {
		decisionAt = current.modelAt
	}
	stage(ctx, "turn.decision", current.readyAt, decisionAt)
	stage(ctx, "turn.llm", current.modelAt, current.firstTextAt)
	stage(ctx, "turn.tts", current.firstTextAt, current.firstAudioAt)
	stage(ctx, "turn.playout", current.queuedAt, current.pulledAt)
}

// stage records one leg of a turn, and nothing for a leg that never ran.
func stage(ctx context.Context, name string, from, to time.Time) {
	if from.IsZero() || to.IsZero() || to.Before(from) {
		return
	}
	_, span := tracer.Start(ctx, name, trace.WithTimestamp(from))
	span.End(trace.WithTimestamp(to))
}

// lastMoment is the latest thing known to have happened in a turn, which is where its
// span ends. A turn cut short ends at whatever it got to.
func lastMoment(current *openTurn) time.Time {
	latest := time.Time{}
	for _, at := range []time.Time{
		current.readyAt, current.decisionAt, current.modelAt, current.firstTextAt, current.ttsAt,
		current.firstAudioAt, current.queuedAt, current.pulledAt,
	} {
		if at.After(latest) {
			latest = at
		}
	}
	return latest
}
