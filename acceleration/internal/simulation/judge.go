package simulation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
)

// defaultJudgeTarget is what rules on a conversation when the simulation did not name a
// model. It is a quality tier rather than a fast one, for the same reason a review is:
// nobody is waiting for it, and the point of it is the judgement.
const defaultJudgeTarget = "multilingual-high-accuracy"

// judgeTimeout bounds the ruling, and judgeTokens bounds what it may write.
const (
	judgeTimeout = 90 * time.Second
	judgeTokens  = 500
)

// judgeInstructions is what the judge is told to be.
//
// It answers the question it was given and nothing else. A judge that also has opinions
// about tone turns every simulation into a review, and the customer already wrote down what
// they wanted checked.
const judgeInstructions = `You are checking a recorded conversation between a caller and an automated agent against one question.

Answer only the question you are given. Do not rule on how polite the agent was, how long it took, or anything else you happened to notice: if the question is about whether an order was placed, an unhelpful agent that placed the order correctly still passes.

Judge what the agent actually did, not what it said it would do. An agent that promised to place an order and never confirmed one has not placed it.

Score your confidence in the ruling from 1 to 5, where 5 is the conversation plainly settles it and 1 is that you are guessing. Say in one or two sentences what in the conversation decided it, quoting the line that settled it where there is one.

Answer with JSON only: {"passed": true, "reason": "...", "score": 5}`

// verdict is what the judge answers with.
type verdict struct {
	Passed bool   `json:"passed"`
	Reason string `json:"reason"`
	Score  int    `json:"score"`
}

// rule asks the judge whether the assertion held.
func rule(
	ctx context.Context,
	router *llmrouter.Router,
	request llmrouter.Request,
	id, assertion string,
	so said,
) (verdict, error) {
	if request.Target == "" {
		request.Target = defaultJudgeTarget
	}

	ctx, cancel := context.WithTimeout(ctx, judgeTimeout)
	defer cancel()

	session, err := router.Start(ctx, request)
	if err != nil {
		return verdict{}, fmt.Errorf("simulation: open the judge: %w", err)
	}
	defer session.Close()

	asked := "The question:\n\n" + assertion + "\n\n" + heard(so)
	err = session.Respond(llm.Request{
		ID:           id,
		Instructions: judgeInstructions,
		Messages:     []llm.Message{{Role: llm.User, Content: asked}},
		MaxTokens:    judgeTokens,
		JSON:         true,
	})
	if err != nil {
		return verdict{}, err
	}

	answer, err := llmrouter.Await(ctx, session, id)
	if err != nil {
		return verdict{}, err
	}
	return parseVerdict(answer)
}

// heard is the conversation as the judge reads it, which names the two sides rather than
// addressing one of them the way the caller's copy does.
func heard(so said) string {
	var written strings.Builder
	written.WriteString("The conversation, in order:\n\n")
	for _, line := range so {
		speaker := "Agent"
		if line.Caller {
			speaker = "Caller"
		}
		written.WriteString(speaker)
		written.WriteString(": ")
		written.WriteString(line.Text)
		written.WriteString("\n")
	}
	return written.String()
}

// parseVerdict reads the ruling. A ruling that cannot be read is an error rather than a
// failure: a conversation nobody managed to judge has not been judged to have failed.
func parseVerdict(answer string) (verdict, error) {
	var ruled verdict
	if err := json.Unmarshal([]byte(llm.Unfence(answer)), &ruled); err != nil {
		return verdict{}, fmt.Errorf("simulation: decode the ruling: %w", err)
	}
	if strings.TrimSpace(ruled.Reason) == "" {
		return verdict{}, errors.New("simulation: the judge ruled without saying why")
	}
	return ruled, nil
}
