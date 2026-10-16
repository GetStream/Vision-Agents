package agent

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func TestTurnTracingSuite(t *testing.T) { suite.Run(t, new(TurnTracingSuite)) }

type TurnTracingSuite struct {
	suite.Suite

	recorded *tracetest.SpanRecorder
	owner    routing.Owner
}

// SetupSuite installs the recorder once. The global provider takes a delegate exactly
// once per process, so a provider per test would leave every test after the first
// recording into the first one's recorder.
func (s *TurnTracingSuite) SetupSuite() {
	s.recorded = tracetest.NewSpanRecorder()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(s.recorded)))
	s.owner = routing.Owner{AgentID: "agent-1", CallID: "call-1"}
}

func (s *TurnTracingSuite) SetupTest() { s.recorded.Reset() }

// turn is one exchange whose stages each took 100ms, starting from the transcript.
func (s *TurnTracingSuite) turn() *openTurn {
	at := time.Now().Add(-time.Second)
	step := func(n int) time.Time { return at.Add(time.Duration(n) * 100 * time.Millisecond) }
	return &openTurn{
		participant:  stt.Participant{ID: "caller"},
		transcriptAt: at,
		readyAt:      step(1),
		modelAt:      step(2),
		firstTextAt:  step(3),
		ttsAt:        step(4),
		firstAudioAt: step(5),
		queuedAt:     step(5),
		pulledAt:     step(6),
		sttLatencyMs: 50,
	}
}

// spans returns what was recorded, by name.
func (s *TurnTracingSuite) spans() map[string]sdktrace.ReadOnlySpan {
	byName := map[string]sdktrace.ReadOnlySpan{}
	for _, span := range s.recorded.Ended() {
		byName[span.Name()] = span
	}
	return byName
}

func (s *TurnTracingSuite) TestEveryStageOfATurnIsASpanOfItsOwnLength() {
	current := s.turn()

	traceTurn(s.owner, "turn-1", current)

	spans := s.spans()
	s.Require().Contains(spans, "agent.turn")
	for _, name := range []string{"turn.stt", "turn.cadence", "turn.decision", "turn.llm", "turn.tts", "turn.playout"} {
		s.Require().Containsf(spans, name, "%s should be a stage of the turn", name)
		s.Equal(spans["agent.turn"].SpanContext().SpanID(), spans[name].Parent().SpanID(),
			"%s belongs to the turn it was part of", name)
	}

	s.Equal(50*time.Millisecond, spans["turn.stt"].EndTime().Sub(spans["turn.stt"].StartTime()))
	s.Equal(100*time.Millisecond, spans["turn.decision"].EndTime().Sub(spans["turn.decision"].StartTime()))
	s.Equal(200*time.Millisecond, spans["turn.tts"].EndTime().Sub(spans["turn.tts"].StartTime()),
		"the voice stage runs from the first words to the first audio")
	s.Equal(600*time.Millisecond, spans["agent.turn"].EndTime().Sub(spans["agent.turn"].StartTime()))
}

func (s *TurnTracingSuite) TestAnEarlyPreviewKeepsTheDecisionAndOverlappingModelStages() {
	current := s.turn()
	current.modelAt = current.transcriptAt.Add(50 * time.Millisecond)
	current.decisionAt = current.transcriptAt.Add(200 * time.Millisecond)
	current.firstTextAt = current.transcriptAt.Add(300 * time.Millisecond)

	traceTurn(s.owner, "turn-1", current)

	spans := s.spans()
	s.Require().Contains(spans, "turn.decision")
	s.Require().Contains(spans, "turn.llm")
	s.Equal(100*time.Millisecond, spans["turn.decision"].EndTime().Sub(spans["turn.decision"].StartTime()))
	s.Equal(250*time.Millisecond, spans["turn.llm"].EndTime().Sub(spans["turn.llm"].StartTime()))
	s.True(spans["turn.llm"].StartTime().Before(spans["turn.decision"].EndTime()),
		"the preview's model interval starts before the decision completes")
}

func (s *TurnTracingSuite) TestTheTurnSaysWhoItWasFor() {
	traceTurn(s.owner, "turn-1", s.turn())

	var attributes []string
	for _, attribute := range s.spans()["agent.turn"].Attributes() {
		attributes = append(attributes, string(attribute.Key)+"="+attribute.Value.Emit())
	}
	s.Subset(attributes, []string{"turn.id=turn-1", "agent.id=agent-1", "call.id=call-1", "participant.id=caller"})
}

func (s *TurnTracingSuite) TestAStageThatNeverRanIsNotRecorded() {
	cut := s.turn()
	cut.modelAt, cut.firstTextAt, cut.ttsAt = time.Time{}, time.Time{}, time.Time{}
	cut.firstAudioAt, cut.queuedAt, cut.pulledAt = time.Time{}, time.Time{}, time.Time{}
	cut.interrupted = true

	traceTurn(s.owner, "turn-1", cut)

	spans := s.spans()
	s.Contains(spans, "turn.cadence")
	s.NotContains(spans, "turn.llm", "a turn cut off before the model answered has no reply to show")
	s.NotContains(spans, "turn.playout")
}

func (s *TurnTracingSuite) TestATurnThatNeverStartedIsNotRecorded() {
	traceTurn(s.owner, "turn-1", &openTurn{})

	s.Empty(s.recorded.Ended())
}
