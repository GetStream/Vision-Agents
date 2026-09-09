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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// defaultReviewTarget is what judges a call when the agent had no thinking model of its
// own. It is a quality tier rather than a fast one: nobody is waiting for this, and the
// point of it is the judgement.
const defaultReviewTarget = "multilingual-high-accuracy"

// reviewLimit is how many lines of a call are read back to the reviewer. A call longer
// than this is summarised from its beginning, which is where what it was about is.
const reviewLimit = 400

// reviewTimeout bounds the whole pass, and reviewTokens bounds what it may write. A
// summary long enough to need more is not a summary.
const (
	reviewTimeout = 90 * time.Second
	reviewTokens  = 500
)

// reviewInstructions is what the reviewer is told to be.
//
// It judges how the agent handled the call rather than how the call went: a caller who
// could not be helped because the answer is no is not a badly handled call, and an agent
// that was warm about being useless is not a well handled one.
const reviewInstructions = `You are reviewing a recorded call between an automated voice agent and a caller.

Summarise what the caller wanted and what happened, in two or three sentences, written for
someone who did not hear the call. Then score how well the agent handled it from 1 to 5:

5 - the caller got what they came for, quickly and without repeating themselves.
4 - the caller was helped, with some friction.
3 - the caller was neither helped nor harmed: passed on, or told to call back.
2 - the agent misunderstood, stalled, or made the caller work for it.
1 - the agent was wrong, unhelpful, or left the caller worse off.

Judge the handling, not the outcome. An answer the caller did not want, given clearly and
early, is a good call. Note anything worth acting on: something the agent did not know,
a question it kept getting wrong, a place it should have handed over sooner.

Answer with JSON only: {"summary": "...", "score": 3, "notes": "..."}`

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

// Review starts a pass over a finished call. A call nobody said anything on is not
// reviewed: there is nothing to summarise, and asking would cost the customer money to
// be told so.
func (r *reviewer) Review(call store.Call, target string, said []spoken) {
	if len(said) == 0 {
		return
	}
	if target == "" {
		target = defaultReviewTarget
	}

	r.running.Add(1)
	go func() {
		defer r.running.Done()

		ctx, cancel := context.WithTimeout(r.ctx, reviewTimeout)
		defer cancel()

		if err := r.review(ctx, call, target, said); err != nil {
			r.logger.Error("could not review the call", "call", call.ID, "error", err)
		}
	}()
}

// Close abandons the reviews still running and waits for them to stop.
func (r *reviewer) Close() {
	r.cancel()
	r.running.Wait()
}

func (r *reviewer) review(ctx context.Context, call store.Call, target string, said []spoken) error {
	session, err := r.llm.Start(ctx, llmrouter.Request{
		CustomerID: call.CustomerID,
		AgentID:    call.AgentID,
		CallID:     call.CallID,
		Tags:       call.Tags,
		Target:     target,
	})
	if err != nil {
		return err
	}
	defer session.Close()

	stream, err := session.Create(ctx, llm.ResponseParams{
		ID:              call.ID,
		Instructions:    reviewInstructions,
		Input:           []llm.Message{{Role: llm.User, Content: conversation(said)}},
		MaxOutputTokens: reviewTokens,
		Text:            llm.TextParams{Format: llm.FormatJSONObject},
	})
	if err != nil {
		return err
	}

	response, err := llm.Collect(stream)
	if err != nil {
		return err
	}

	verdict, err := parseJudgement(response.OutputText)
	if err != nil {
		return err
	}

	var score *int
	if verdict.Score >= 1 && verdict.Score <= 5 {
		score = &verdict.Score
	}
	return r.store.ReviewCall(ctx, call.CustomerID, call.ID, verdict.Summary, score, verdict.Notes)
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
