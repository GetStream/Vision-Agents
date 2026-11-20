package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// requesterTool records who each call it runs was made for.
type requesterTool struct{ seen chan Requester }

func (r *requesterTool) Run(ctx context.Context, _ llm.ToolCall) ([]llm.ContentPart, error) {
	requester, _ := RequesterFrom(ctx)
	r.seen <- requester
	return llm.TextParts("order 12 ships tomorrow"), nil
}

func (s *AgentSuite) TestAToolAskedForInASpokenTurnIsForWhoeverSpokeIt() {
	s.ownsTools("")
	s.join(true)
	runner := &requesterTool{seen: make(chan Requester, 1)}
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	jean := stt.Participant{ID: "jean-phone", UserID: "jean", Name: "Jean"}

	s.speak(jean)
	s.says(jean, "where is my order")

	select {
	case requester := <-runner.seen:
		s.Equal(Requester{UserID: "jean", Name: "Jean"}, requester)
	case <-time.After(settleFor):
		s.Fail("the tool never ran")
	}
}

// AI-3464: in a channel of a type an agent config answers in, a tool call is for the person who
// wrote the message its turn answers (WithWriter), turn by turn: Anya's turn is hers, Ben's next
// one his, a turn nobody wrote there is nobody's, and a writer goes before the caller who relayed
// their message.
func (s *AgentSuite) TestAToolAskedForInAWrittenTurnIsForWhoeverWroteIt() {
	s.ownsTools("")
	s.joinText()
	runner := &requesterTool{seen: make(chan Requester, 1)}
	s.agent.options.ToolRunner = runner
	s.asksFor("lookup_order", `{"order":"12"}`)
	relayed := routing.WithCaller(s.ctx, routing.Caller{UserID: "backend-sam"})

	s.Equal(Requester{UserID: "anya-1", Name: "Anya"}, s.toolAskedIn(WithWriter(s.ctx, "anya-1", "Anya"), runner))
	s.Equal(Requester{UserID: "ben-2", Name: "Ben"}, s.toolAskedIn(WithWriter(s.ctx, "ben-2", "Ben"), runner), "Anya's turn named Ben's call")
	s.Equal(Requester{}, s.toolAskedIn(s.ctx, runner), "a turn nobody wrote named the last writer")
	s.Equal(Requester{UserID: "backend-sam"}, s.toolAskedIn(relayed, runner), "a caller's turn")
	s.Equal(Requester{UserID: "anya-1", Name: "Anya"}, s.toolAskedIn(WithWriter(relayed, "anya-1", "Anya"), runner),
		"a writer's message relayed by a caller")
}

// toolAskedIn has the agent answer a turn on ctx whose first reply calls the tool asksFor named,
// and is who that call was for, once the turn is answered. Each turn's call has an id of its own.
func (s *AgentSuite) toolAskedIn(ctx context.Context, runner *requesterTool) Requester {
	answered := countOf[Responded](s.reported())
	s.model.mu.Lock()
	s.model.callsOn = len(s.model.asked) + 1
	s.model.calls[0].ID = fmt.Sprintf("call-%d", s.model.callsOn)
	s.model.mu.Unlock()
	_, err := s.agent.RespondTo(ctx, "where is my order", nil, options.LLM{})
	s.Require().NoError(err)
	var requester Requester
	select {
	case requester = <-runner.seen:
	case <-time.After(settleFor):
		s.Fail("the tool never ran")
	}
	// One reply asks for the tool and the next one answers with its result.
	s.eventually(func() bool { return countOf[Responded](s.reported()) == answered+2 }, "the turn was never answered")
	return requester
}
