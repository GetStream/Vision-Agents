package agent

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
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
