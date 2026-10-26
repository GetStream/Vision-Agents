package session

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/stretchr/testify/suite"
)

type ReviewSuite struct {
	suite.Suite
}

func TestReviewSuite(t *testing.T) {
	suite.Run(t, new(ReviewSuite))
}

// reviewerOver is a reviewer whose model answers only when the test says, logging as text
// into logs. The tests review on "en-low-latency", an alias stubConfig routes to the stub.
func (s *ReviewSuite) reviewerOver(logs *bytes.Buffer) (*reviewer, *gatedLLM) {
	model := newGatedLLM()
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return model, nil })
	router, err := llmrouter.New(llmrouter.Options{Config: stubConfig(), Registry: reasoning, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return newReviewer(router, nil, slog.New(slog.NewTextHandler(logs, nil))), model
}

// asked waits for the review to reach the model, which is where it is held.
func (s *ReviewSuite) asked(model *gatedLLM, logs *bytes.Buffer) {
	select {
	case <-model.entered:
	case <-time.After(settleFor):
		s.FailNow("the review never asked the model", logs.String())
	}
}

// promptedLLM keeps the instructions each review was given. It answers with prose, so the
// review stops at reading the answer and never reaches the store, which these tests lack.
type promptedLLM struct {
	mu   sync.Mutex
	told []string
}

func (p *promptedLLM) Start(context.Context) error { return nil }

func (p *promptedLLM) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	p.mu.Lock()
	p.told = append(p.told, params.Instructions)
	p.mu.Unlock()
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: p.Provider(), Model: p.Model()})
	script.OutputText("It went fine.")
	script.Done()
	return script.Stream(), nil
}

func (p *promptedLLM) Provider() string               { return "stub" }
func (p *promptedLLM) Model() string                  { return "stub-llm" }
func (p *promptedLLM) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (p *promptedLLM) Close() error                   { return nil }

// toldWith is the instructions the reviewer's model is given for a call whose agent's
// config holds guide.
func (s *ReviewSuite) toldWith(guide string) string {
	model := &promptedLLM{}
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return model, nil })
	router, err := llmrouter.New(llmrouter.Options{Config: stubConfig(), Registry: reasoning, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	reviews := newReviewer(router, nil, slog.New(slog.DiscardHandler))

	reviews.Review(store.Call{ID: "call-prompted", CustomerID: "customer-reviewed"}, routing.Attribution{}, "en-low-latency", guide, []spoken{{text: "hello?"}})
	reviews.running.Wait()
	reviews.Close()

	s.Require().Len(model.told, 1, "the review asked the model once")
	return model.told[0]
}

// basePrompt is reviewInstructions as it stood before a config could set the guide
// (internal/session/review.go at adc9d6015), copied rather than read from the code under
// test, so an agent with no guide of its own is reviewed exactly as before.
const basePrompt = `You are reviewing a recorded call between an automated voice agent and a caller.

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

func (s *ReviewSuite) TestACallWhoseAgentHasNoGuideIsReviewedAsBefore() {
	s.Equal(basePrompt, s.toldWith(""))
}

func (s *ReviewSuite) TestAGuideOfOnlyWhitespaceIsNoGuide() {
	s.Equal(basePrompt, s.toldWith(" \n\t "))
}

// The guide replaces the default one, and the reviewer is still told how to answer, since
// nothing else can be read back.
func (s *ReviewSuite) TestAnAgentsGuideReplacesTheDefaultAndKeepsTheAnswersFormat() {
	told := s.toldWith("Score how well the agent sold the upgrade.\n")

	s.Equal("Score how well the agent sold the upgrade.\n\n"+
		`Answer with JSON only: {"summary": "...", "score": 3, "notes": "..."}`, told)
}

func (s *ReviewSuite) TestAReviewAShutdownAbandonsIsAWarningNotAnError() {
	var logs bytes.Buffer
	reviews, model := s.reviewerOver(&logs)

	reviews.Review(store.Call{ID: "call-cut-short", CustomerID: "customer-reviewed"}, routing.Attribution{}, "en-low-latency", "", []spoken{{text: "hello?"}})
	s.asked(model, &logs)
	reviews.Close()

	s.Contains(logs.String(), "level=WARN")
	s.Contains(logs.String(), "call=call-cut-short")
	s.NotContains(logs.String(), "level=ERROR")
}

func (s *ReviewSuite) TestAReviewThatFailsWhileTheReviewerIsOpenIsAnError() {
	// The model answers with prose rather than the JSON the review asks for, so the review
	// fails of its own accord rather than because anything closed it.
	var logs bytes.Buffer
	reviews, model := s.reviewerOver(&logs)

	reviews.Review(store.Call{ID: "call-unreadable", CustomerID: "customer-reviewed"}, routing.Attribution{}, "en-low-latency", "", []spoken{{text: "hello?"}})
	s.asked(model, &logs)
	model.answers(1)
	reviews.running.Wait()
	reviews.Close()

	s.Contains(logs.String(), "level=ERROR")
	s.Contains(logs.String(), "call=call-unreadable")
	s.Contains(logs.String(), "decode review")
	s.NotContains(logs.String(), "level=WARN")
}

// A review that outlives reviewTimeout fails with the reviewer still open, so it stays an
// error: only a Close may turn a failure into a warning. The bubble's clock runs the 90 s
// out at once.
func (s *ReviewSuite) TestAReviewThatRunsOutOfTimeIsAnError() {
	synctest.Test(s.T(), func(t *testing.T) {
		var logs bytes.Buffer
		reviews, model := s.reviewerOver(&logs)

		reviews.Review(store.Call{ID: "call-too-slow", CustomerID: "customer-reviewed"}, routing.Attribution{}, "en-low-latency", "", []spoken{{text: "hello?"}})
		<-model.entered
		time.Sleep(reviewTimeout)
		reviews.running.Wait()
		reviews.Close()
		reviews.llm.Close()

		s.Contains(logs.String(), "level=ERROR")
		s.Contains(logs.String(), "call=call-too-slow")
		s.Contains(logs.String(), "deadline exceeded")
		s.NotContains(logs.String(), "level=WARN")
	})
}

func (s *ReviewSuite) TestTheCallIsQuotedWithBothSidesNamed() {
	written := conversation([]spoken{
		{text: "are you open on Sundays?"},
		{agent: true, text: "we are, from ten."},
	})

	s.Contains(written, "Caller: are you open on Sundays?")
	s.Contains(written, "Agent: we are, from ten.")
}

func (s *ReviewSuite) TestALongCallIsReviewedFromItsBeginning() {
	// What a call was about is said at the start of it, so the part that is dropped is
	// the end rather than the beginning.
	var said []spoken
	said = append(said, spoken{text: "I want to change my flight"})
	for range reviewLimit + 50 {
		said = append(said, spoken{agent: true, text: "one moment."})
	}

	written := conversation(said)

	s.Contains(written, "I want to change my flight")
	s.Equal(reviewLimit, strings.Count(written, "\n")-2, "more of the call was sent than fits")
}

func (s *ReviewSuite) TestAJudgementIsReadOutOfWhatTheModelWrote() {
	verdict, err := parseJudgement(
		"```json\n{\"summary\":\"they wanted a refund\",\"score\":4,\"notes\":\"knew the policy\"}\n```")

	s.Require().NoError(err)
	s.Equal("they wanted a refund", verdict.Summary)
	s.Equal(4, verdict.Score)
	s.Equal("knew the policy", verdict.Notes)
}

func (s *ReviewSuite) TestAReviewThatSaidNothingIsNotOne() {
	// An empty summary written onto the row would read as a call nobody could describe
	// rather than as a review that failed.
	_, err := parseJudgement(`{"summary":"","score":5}`)

	s.Require().Error(err)
}

func (s *ReviewSuite) TestAnAnswerThatIsNotJsonIsRefused() {
	_, err := parseJudgement("I think the call went quite well.")

	s.Require().Error(err)
}
