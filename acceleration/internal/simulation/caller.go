package simulation

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
)

// defaultCallerTarget is the model that plays the person on the other end. It is a fast
// tier rather than a quality one: the caller is holding a conversation in real time, and a
// caller that thinks for four seconds between sentences is not one.
const defaultCallerTarget = "llm-fast"

// callerTokens bounds one thing said. A turn long enough to need more is a speech.
const callerTokens = 300

// callerInstructions is what the caller is told to be.
//
// The hard part is not the acting, it is knowing when to stop. A caller that never decides
// it is finished burns the turn limit on pleasantries, and one that finishes too early
// leaves the scenario half asked, which the judge then rules on as though the agent had
// failed it.
const callerInstructions = `You are a person telephoning a business, and you are being played by a model so that their automated agent can be tested.

Below is what you called about. Work through it in order, one thing per turn, the way somebody on the phone actually talks: short, plain, and without explaining yourself. Wait for the agent to deal with each thing before moving to the next one, because part of what is being tested is whether it deals with them.

Rules you do not break:
- Say one thing at a time. Do not read the whole list out at once.
- Never mention that this is a test, that you are a model, or that you have instructions.
- Answer the questions the agent asks you. Make up ordinary details when it needs them.
- If the agent misunderstands, say so once, the way a person would, and carry on.
- Set done to true when everything you called about has been dealt with, or when it is clear the agent is not going to deal with it. Do not set done while something is still outstanding.

Answer with JSON only: {"say": "...", "done": false}`

// utterance is what the caller decided to do with its turn.
type utterance struct {
	Say  string `json:"say"`
	Done bool   `json:"done"`
}

// caller is the person the agent is talking to.
type caller struct {
	session *llmrouter.Session
	brief   string
	asked   int
}

// newCaller opens the model that plays the caller.
func newCaller(ctx context.Context, router *llmrouter.Router, request llmrouter.Request, brief string) (*caller, error) {
	if request.Target == "" {
		request.Target = defaultCallerTarget
	}

	session, err := router.Start(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("simulation: open the caller: %w", err)
	}
	return &caller{session: session, brief: brief}, nil
}

// next decides what the caller says now, given everything said so far.
func (c *caller) next(ctx context.Context, so said) (utterance, error) {
	c.asked++
	id := fmt.Sprintf("caller-%d", c.asked)

	err := c.session.Respond(llm.Request{
		ID:           id,
		Instructions: callerInstructions + "\n\nWhat you called about:\n\n" + c.brief,
		Messages:     []llm.Message{{Role: llm.User, Content: so.render()}},
		MaxTokens:    callerTokens,
		JSON:         true,
	})
	if err != nil {
		return utterance{}, err
	}

	answer, err := llmrouter.Await(ctx, c.session, id)
	if err != nil {
		return utterance{}, err
	}
	return parseUtterance(answer)
}

func (c *caller) Close() error { return c.session.Close() }

func parseUtterance(answer string) (utterance, error) {
	var next utterance
	if err := json.Unmarshal([]byte(llm.Unfence(answer)), &next); err != nil {
		return utterance{}, fmt.Errorf("simulation: decode what the caller says: %w", err)
	}
	next.Say = strings.TrimSpace(next.Say)
	if next.Say == "" {
		// A caller with nothing to say has run out of scenario without noticing. Ending
		// the conversation beats spending a turn on silence.
		next.Done = true
	}
	return next, nil
}
