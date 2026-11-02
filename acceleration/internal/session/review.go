package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// defaultReviewTarget is what judges a call when the agent had no subagent of its
// own. It is a quality tier rather than a fast one: nobody is waiting for this, and the
// point of it is the judgement.
const defaultReviewTarget = "multilingual-high-accuracy"

// reviewLimit is how many lines of a call are read back to the reviewer. A call longer
// than this is summarised from its beginning, which is where what it was about is.
const reviewLimit = 400

// reviewTimeout bounds the whole pass, and reviewTokens bounds the answer it may write. A
// summary long enough to need more is not a summary. A model that thinks first is given
// llm.ThinkingRoom on top, since its thinking counts against the same cap.
const (
	reviewTimeout = 90 * time.Second
	reviewTokens  = 500
)

// reviewGuide is what the reviewer is told to judge a call by, unless the agent's config
// says otherwise.
//
// It judges how the agent handled the call rather than how the call went: a caller who
// could not be helped because the answer is no is not a badly handled call, and an agent
// that was warm about being useless is not a well handled one.
const reviewGuide = `You are reviewing a recorded call between an automated voice agent and a caller.

Summarise what the caller wanted and what happened, in two or three sentences, written for
someone who did not hear the call. Then score how well the agent handled it from 1 to 5:

5 - the caller got what they came for, quickly and without repeating themselves.
4 - the caller was helped, with some friction.
3 - the caller was neither helped nor harmed: passed on, or told to call back.
2 - the agent misunderstood, stalled, or made the caller work for it.
1 - the agent was wrong, unhelpful, or left the caller worse off.

Judge the handling, not the outcome. An answer the caller did not want, given clearly and
early, is a good call. Note anything worth acting on: something the agent did not know,
a question it kept getting wrong, a place it should have handed over sooner.`

// reviewOutputRule ends every review prompt, a config's own guide included, because
// parseJudgement reads nothing else.
const reviewOutputRule = `Answer with JSON only: {"summary": "...", "score": 3, "notes": "..."}`

// reviewPrompt is what the reviewer is told: the config's guide, or the default one when
// it gave none, then the output rule. A guide of only whitespace is none: it would leave
// the reviewer nothing to judge by.
func reviewPrompt(guide string) string {
	guide = strings.TrimSpace(guide)
	if guide == "" {
		guide = reviewGuide
	}
	return guide + "\n\n" + reviewOutputRule
}

// judgement is what the reviewer answers with.
type judgement struct {
	Summary string `json:"summary"`
	Score   int    `json:"score"`
	Notes   string `json:"notes"`
}

// spoken is one line of a call, kept in memory as it is said so that reviewing it needs
// nothing read back over the network.
type spoken struct {
	// agent is which side said it, which is all the reviewer needs to know about who.
	agent bool
	text  string
}

// reviewer looks back at a finished call and writes down what it made of it.
//
// The pass runs after the call ended and off everybody's path: nobody is waiting for it,
// a failure costs a summary rather than a conversation, and a router on its way down
// abandons the ones still running rather than holding the process open for them.
type reviewer struct {
	llm    *llmrouter.Router
	store  *store.Store
	logger *slog.Logger

	// ctx is cancelled by Close, which is what abandons the reviews still running.
	ctx    context.Context
	cancel context.CancelFunc

	running sync.WaitGroup
}

func newReviewer(router *llmrouter.Router, pgStore *store.Store, logger *slog.Logger) *reviewer {
	ctx, cancel := context.WithCancel(context.Background())
	return &reviewer{
		llm:    router,
		store:  pgStore,
		logger: logger,
		ctx:    ctx,
		cancel: cancel,
	}
}

// Review starts a pass over a finished call, judged by guide, or by the default guide when
// it is empty. A call nobody said anything on is not reviewed: there is nothing to
// summarise, and asking would cost the customer money to be told so. The review is owned by
// whoever the session was, and done in the session's region.
func (r *reviewer) Review(call store.Call, attribution routing.Attribution, region, target, guide string, said []spoken) {
	if len(said) == 0 {
		return
	}
	if target == "" {
		target = defaultReviewTarget
	}

	r.running.Add(1)
	go func() {
		defer r.running.Done()

		ctx, cancel := context.WithTimeout(routing.WithAttribution(r.ctx, attribution), reviewTimeout)
		defer cancel()

		err := r.review(ctx, call, region, target, guide, said)
		switch {
		case err == nil:
		case r.ctx.Err() != nil:
			// Close cut it short, which a shutdown does on purpose: whatever the review
			// failed with once that happened is the shutdown, not a fault.
			r.logger.Warn("a shutdown abandoned the call's review", "call", call.ID, "error", err)
		default:
			r.logger.Error("could not review the call", "call", call.ID, "error", err)
		}
	}()
}

// Close abandons the reviews still running and waits for them to stop.
func (r *reviewer) Close() {
	r.cancel()
	r.running.Wait()
}

func (r *reviewer) review(ctx context.Context, call store.Call, region, target, guide string, said []spoken) error {
	verdict, err := r.judge(ctx, call, region, target, guide, said)
	if err != nil {
		return err
	}

	var score *int
	if verdict.Score >= 1 && verdict.Score <= 5 {
		score = &verdict.Score
	}
	return r.store.ReviewCall(ctx, call.CustomerID, call.ID, verdict.Summary, score, verdict.Notes)
}

// judge asks the reviewer's model what it made of the call.
func (r *reviewer) judge(ctx context.Context, call store.Call, region, target, guide string, said []spoken) (judgement, error) {
	session, err := r.llm.Start(ctx, llmrouter.Request{
		CustomerID: call.CustomerID,
		AgentID:    call.AgentID,
		CallID:     call.CallID,
		Region:     region,
		Tags:       call.Tags,
		Target:     target,
	})
	if err != nil {
		return judgement{}, err
	}
	defer session.Close()

	stream, err := session.Create(ctx, llm.ResponseParams{
		ID:              call.ID,
		Instructions:    reviewPrompt(guide),
		Input:           []llm.Message{{Role: llm.User, Content: conversation(said)}},
		MaxOutputTokens: reviewTokens + llm.ThinkingRoom,
		Text:            llm.TextParams{Format: llm.FormatJSONObject},
	})
	if err != nil {
		return judgement{}, err
	}

	response, err := llm.Collect(stream)
	if err != nil {
		return judgement{}, err
	}
	read, err := parseJudgement(response.OutputText)
	// An answer the cap cut off reads as broken JSON, which hides what broke it.
	if err != nil && response.Status == llm.StatusIncomplete {
		return judgement{}, fmt.Errorf("session: the review stopped early (%s) after %d output tokens, %d of them thinking",
			response.IncompleteReason, response.Usage.OutputTokens, response.Usage.OutputTokensDetails.ReasoningTokens)
	}
	return read, err
}

// conversation is the call as the reviewer reads it. It is quoted rather than replayed as
// turns, because a model handed a conversation to continue continues it.
func conversation(said []spoken) string {
	if len(said) > reviewLimit {
		said = said[:reviewLimit]
	}

	var written strings.Builder
	written.WriteString("The call, in order:\n\n")
	for _, line := range said {
		speaker := "Caller"
		if line.agent {
			speaker = "Agent"
		}
		written.WriteString(speaker)
		written.WriteString(": ")
		written.WriteString(line.text)
		written.WriteString("\n")
	}
	return written.String()
}

func parseJudgement(answer string) (judgement, error) {
	var verdict judgement
	if err := json.Unmarshal([]byte(llm.Unfence(answer)), &verdict); err != nil {
		return judgement{}, fmt.Errorf("session: decode review: %w", err)
	}
	if strings.TrimSpace(verdict.Summary) == "" {
		return judgement{}, errors.New("session: the review said nothing about the call")
	}
	return verdict, nil
}
