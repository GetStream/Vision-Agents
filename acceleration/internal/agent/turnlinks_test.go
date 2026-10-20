package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Messages of the lines that tie a call's turns, tools and utterances together.
const (
	turnTiming  = "voice turn timing"
	toolBack    = "a tool call came back"
	tookSpeech  = "the voice took an utterance"
	spokeSpeech = "finished speaking"
	choseLine   = "saying a line the agent chose"
	chased      = "the reply owed a call and made none, asking for it"
)

// waitsOnATool has the caller ask for an order, the model reach for a tool without a word, a
// hold line said while the tool runs, and the answer once it is released.
func (s *AgentSuite) waitsOnATool() {
	s.ownsTools("")
	s.join(false)
	runner := &holdingTool{began: make(chan struct{}), release: make(chan struct{})}
	release := sync.OnceFunc(func() { close(runner.release) })
	defer release()
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{}
	s.model.then = []string{"It ships tomorrow."}
	s.model.composes = []string{"Pulling up order twelve now."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return s.spokenText("Pulling up order twelve now.") }, "the hold line was never said")
	release()
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 2 }, "the answer never followed the hold line")
}

// voiceRequests is what the agent sent the voice, each as the kind of turn its id names,
// its text and whether it closed the utterance.
func (s *AgentSuite) voiceRequests() []string {
	var sent []string
	for _, request := range s.voice.spoken() {
		kind := strings.TrimRight(request.ID[:strings.IndexAny(request.ID, "0123456789")], "-")
		sent = append(sent, fmt.Sprintf("%s|%s|%t", kind, request.Text, request.Final))
	}
	return sent
}

// callerTurn is the turn id the agent answered the caller's words with.
func (s *AgentSuite) callerTurn() string {
	responding, ok := firstOf[Responding](s.reported())
	s.Require().True(ok, "the caller was never answered")
	return responding.TurnID
}

// loggedOnce waits for one record with message and returns it.
func (s *AgentSuite) loggedOnce(message string) map[string]any {
	s.eventually(func() bool { return len(s.logs.records(message)) == 1 }, "never logged: "+message)
	return s.logs.records(message)[0]
}

// byKind is the records with message for the turns of kind.
func byKind(records []map[string]any, kind string) []map[string]any {
	var found []map[string]any
	for _, record := range records {
		if record["kind"] == kind {
			found = append(found, record)
		}
	}
	return found
}

// histogramOf reads one series of a histogram from the default registry: how many samples
// it has, and how many of them were above one.
func histogramOf(name, label, value string) (samples, aboveOne uint64) {
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		panic(err)
	}
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, pair := range metric.GetLabel() {
				if pair.GetName() != label || pair.GetValue() != value {
					continue
				}
				histogram := metric.GetHistogram()
				for _, bucket := range histogram.GetBucket() {
					if bucket.GetUpperBound() == 1 {
						return histogram.GetSampleCount(), histogram.GetSampleCount() - bucket.GetCumulativeCount()
					}
				}
				return histogram.GetSampleCount(), 0
			}
		}
	}
	return 0, 0
}

func (s *AgentSuite) TestWhatIsSaidAroundAToolIsSaidAsBefore() {
	s.waitsOnATool()

	// Probed on 7bf225e2b, before turns were linked and tool calls logged.
	s.Equal([]string{"back|Pulling up order twelve now.|true", "tool|It ships tomorrow.|true"}, s.voiceRequests())
}

func (s *AgentSuite) TestATurnTheAgentStartedNamesTheCallerTurnItFollows() {
	s.logs = &logLines{}
	s.waitsOnATool()

	s.eventually(func() bool { return len(s.logs.records(turnTiming)) == 2 }, "both turns were never measured")
	timings := s.logs.records(turnTiming)
	callers, tools := byKind(timings, kindCaller), byKind(timings, kindTool)
	s.Require().Len(callers, 1)
	s.Require().Len(tools, 1)
	s.Equal(s.callerTurn(), callers[0]["turn"])
	s.Empty(callers[0]["parent"], "a caller turn follows nothing")
	s.Equal(s.callerTurn(), tools[0]["parent"])
}

func (s *AgentSuite) TestAToolCallIsLoggedWhenItComesBack() {
	s.logs = &logLines{}
	before, _ := histogramOf("router_agent_tool_call_seconds", "outcome", "ok")
	s.waitsOnATool()

	back := s.loggedOnce(toolBack)
	s.Equal("INFO", back["level"])
	s.Equal("lookup_order", back["tool"])
	s.Equal("call-1", back["call"])
	s.Equal(s.callerTurn(), back["turn"])
	s.Equal("ok", back["outcome"])
	s.Contains(back, "duration_ms")
	s.NotContains(back, "arguments", "what a caller's tool is given is theirs")
	s.NotContains(back, "result", "what a caller's tool returns is theirs")
	after, _ := histogramOf("router_agent_tool_call_seconds", "outcome", "ok")
	s.Equal(before+1, after)
}

func (s *AgentSuite) TestAToolThatFailedIsLoggedAsAnError() {
	s.logs = &logLines{}
	s.ownsTools("")
	s.runner.err = errors.New("stub: the order service is down")
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.Equal("error", s.loggedOnce(toolBack)["outcome"])
}

// interruptsATool has the caller interrupt while a tool runs, the tool asking policy of the
// agent, then lets the tool return.
func (s *AgentSuite) interruptsATool(policy ToolPolicy, runner ToolRunner, began <-chan struct{}) {
	s.ownsTools("")
	s.toolPolicy = func(string) ToolPolicy { return policy }
	s.join(true)
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "where is my order")
	select {
	case <-began:
	case <-time.After(3 * time.Second):
		s.FailNow("the tool did not start")
	}
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.mutters(participant, "stop cancel that lookup")
}

func (s *AgentSuite) TestAToolAnsweredAsStillRunningIsLoggedSoWhenItComesBack() {
	s.logs = &logLines{}
	runner := &holdingTool{began: make(chan struct{}), release: make(chan struct{})}
	release := sync.OnceFunc(func() { close(runner.release) })
	defer release()
	s.interruptsATool(ToolPolicy{Waits: true}, runner, runner.began)
	s.eventually(func() bool {
		return strings.Contains(fmt.Sprint(s.history()), stillRunning)
	}, "the interrupted call was not answered")

	release()

	s.Equal("still-running", s.loggedOnce(toolBack)["outcome"])
}

// cancelledTool runs until its call is cancelled.
type cancelledTool struct{ began chan struct{} }

func (r *cancelledTool) Run(ctx context.Context, _ llm.ToolCall) ([]llm.ContentPart, error) {
	close(r.began)
	<-ctx.Done()
	return nil, ctx.Err()
}

func (s *AgentSuite) TestAToolCancelledByTheCallerIsLoggedAsCancelled() {
	s.logs = &logLines{}
	runner := &cancelledTool{began: make(chan struct{})}
	s.interruptsATool(ToolPolicy{Cancels: true}, runner, runner.began)

	s.Equal("cancelled", s.loggedOnce(toolBack)["outcome"])
}

func (s *AgentSuite) TestAnUtteranceIsLoggedWithWhatStartedItsTurn() {
	s.logs = &logLines{}
	s.waitsOnATool()

	s.eventually(func() bool { return len(s.logs.records(spokeSpeech)) == 2 }, "the utterances never finished")
	for _, kind := range []string{kindHold, kindTool} {
		spoke := byKind(s.logs.records(spokeSpeech), kind)
		s.Require().Len(spoke, 1, kind)
		s.Equal("INFO", spoke[0]["level"])
		s.Contains(spoke[0], "audio_ms")
		s.Equal(false, spoke[0]["interrupted"])
	}
}

func (s *AgentSuite) TestAnUtteranceTakenIsLoggedWithWhatStartedItsTurn() {
	s.logs = &logLines{}
	s.join(true)
	s.agent.turns.started("turn-a", kindCaller)
	s.agent.turns.started("tool-b", kindTool)

	s.voiceSends(2, interleaved()...)

	took := s.logs.records(tookSpeech)
	s.Require().Len(took, 2)
	s.Equal("INFO", took[0]["level"])
	s.Equal([]any{"turn-a", kindCaller}, []any{took[0]["synthesis"], took[0]["kind"]})
	s.Equal([]any{"tool-b", kindTool}, []any{took[1]["synthesis"], took[1]["kind"]})
}

func (s *AgentSuite) TestAHoldLineIsLoggedWithoutItsWords() {
	s.logs = &logLines{}
	s.waitsOnATool()

	chose := s.loggedOnce(choseLine)
	s.Equal("INFO", chose["level"])
	s.Equal(kindHold, chose["kind"])
	s.NotContains(chose, "line", "what the agent says is the caller's conversation")
}

func (s *AgentSuite) TestAChaseIsLoggedWithTheTurnItChases() {
	s.logs = &logLines{}
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Order twelve, one moment, checking..."}
	s.model.then = []string{}
	s.model.callsOn = 2
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is order twelve")

	chase := s.loggedOnce(chased)
	s.Equal("INFO", chase["level"])
	s.Equal(s.callerTurn(), chase["chases"])
	s.NotContains(chase, "note")
	s.eventually(func() bool { return len(byKind(s.logs.records(turnTiming), kindChase)) == 1 }, "the chase was never measured")
	timing := byKind(s.logs.records(turnTiming), kindChase)[0]
	s.Equal(chase["turn"], timing["turn"])
	s.Equal(s.callerTurn(), timing["parent"])
}

func (s *AgentSuite) TestTwoUtterancesPublishedAtOnceAreCountedOpenTogether() {
	s.join(true)
	samples, aboveOne := histogramOf("router_agent_open_utterances", "provider", "stub")

	s.voiceSends(2, interleaved()...)

	afterSamples, afterAboveOne := histogramOf("router_agent_open_utterances", "provider", "stub")
	s.Equal(uint64(4), afterSamples-samples, "one sample a published chunk")
	s.Equal(uint64(3), afterAboveOne-aboveOne, "every chunk after the second utterance began went out with two open")
}
