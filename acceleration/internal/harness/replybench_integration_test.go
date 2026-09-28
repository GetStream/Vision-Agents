//go:build integration

package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// replyBenchmarkEnvVar has to be set for the reply benchmark to run.
const replyBenchmarkEnvVar = "REPLY_BENCHMARK"

// replyModelsEnvVar names the reply models to compare, comma separated, as router targets, or
// all for every configured LLM.
const replyModelsEnvVar = "REPLY_BENCHMARK_MODELS"

// defaultReplyModels is the voice model as deployed.
const defaultReplyModels = "llm-fast"

// replyJudgeEnvVar names the model that grades the replies. It is the quality tier by
// default and should stay pinned across runs being compared, or the judge is what moved.
const replyJudgeEnvVar = "REPLY_BENCHMARK_JUDGE"

// replyJudgeInstructions is what the judge grades against. Each property is one a caller
// would notice on a call, and none of them is about style the contract did not ask for.
const replyJudgeInstructions = `You grade one reply a voice agent spoke on a live call.

You are given what the agent was told, the conversation so far, the words it is replying to, and its reply. Judge only these:

- relevant: the reply engages with the words it is replying to, in the context of the conversation.
- grounded: the reply claims nothing the agent could not know or has not done. Saying a booking is made, a record was checked or a fact is true, with nothing in the conversation to show it, is not grounded. Offering to do it, or asking for what is needed to do it, is.
- spoken: the reply works said out loud: no markdown, lists, links or emoji, and short enough that a caller would not lose the thread, which is rarely more than three sentences.

Then score the reply from 1 to 5 as a person on the call would: 5 is what a skilled human in the agent's role would have said, 1 is a reply that derails the call. Say in one sentence what decided the score.

Answer with JSON only: {"relevant": true, "grounded": true, "spoken": true, "score": 4, "reason": "..."}`

// replyDeadline bounds one reply, or one grading of it.
const replyDeadline = 30 * time.Second

// graded is one reply and what the judge made of it.
type graded struct {
	CaseID string `json:"case_id"`
	Heard  string `json:"heard"`
	Reply  string `json:"reply"`
	// FirstTextMs is what a caller waits for the first word to be written, which is what
	// the voice waits on; TookMs is the whole reply.
	FirstTextMs float64 `json:"first_text_ms"`
	TookMs      float64 `json:"took_ms"`
	Words       int     `json:"words"`

	Relevant bool   `json:"relevant"`
	Grounded bool   `json:"grounded"`
	Spoken   bool   `json:"spoken"`
	Score    int    `json:"score"`
	Reason   string `json:"reason"`
	// Failed is why the reply or its grading could not be had.
	Failed string `json:"failed,omitempty"`

	Usage routing.Usage `json:"usage"`
}

func (g graded) passed() bool { return g.Failed == "" && g.Relevant && g.Grounded && g.Spoken }

type ReplyBenchmarkSuite struct {
	suite.Suite
	ctx    context.Context
	config routing.Config
	router *llmrouter.Router
	judge  *llmrouter.Session
}

func TestReplyBenchmarkSuite(t *testing.T) {
	suite.Run(t, new(ReplyBenchmarkSuite))
}

func (s *ReplyBenchmarkSuite) SetupSuite() {
	if os.Getenv(replyBenchmarkEnvVar) == "" {
		s.T().Skip(replyBenchmarkEnvVar + " not set")
	}
	s.ctx = context.Background()

	var err error
	s.config, err = routing.DefaultConfig()
	s.Require().NoError(err)
	s.router, err = llmrouter.New(llmrouter.Options{
		Config:   s.config[routing.LLM],
		Registry: llmrouter.DefaultRegistry(),
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.router.Close)

	s.judge, err = s.router.Start(s.ctx, llmrouter.Request{
		CustomerID: "reply-benchmark", Target: envOr(replyJudgeEnvVar, "llm-judge"),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = s.judge.Close() })
}

// TestReplyBenchmark answers every case where the agent should speak next, with every model,
// and grades each reply.
func (s *ReplyBenchmarkSuite) TestReplyBenchmark() {
	var report strings.Builder
	results := map[string]map[string][]graded{}
	for _, name := range strings.Split(envOr(setsEnvVar, writtenSet+","+amiSet), ",") {
		set, err := loadFlowSet(name)
		s.Require().NoError(err)
		set = set.sample(sampleFraction())
		cases := replyCases(set)

		results[name] = map[string][]graded{}
		var models []string
		for _, target := range targets(s.config, envOr(replyModelsEnvVar, defaultReplyModels)) {
			session, err := s.router.Start(s.ctx, llmrouter.Request{
				CustomerID: "reply-benchmark", Target: target,
			})
			if err != nil {
				s.T().Logf("skipping %s, it is out of reach: %v", target, err)
				continue
			}
			s.T().Logf("%s answering %d cases of the %s set", target, len(cases), name)
			replies := make([]graded, 0, len(cases))
			for _, one := range cases {
				replies = append(replies, s.answer(session, set, one))
			}
			_ = session.Close()
			results[name][target] = replies
			models = append(models, target)
		}
		report.WriteString(replyReport(name, set, models, results[name]))
	}

	fmt.Print(report.String())
	out := filepath.Join("testdata", "replybench-out", time.Now().UTC().Format("20060102-150405"))
	s.Require().NoError(os.MkdirAll(out, 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(out, "report.md"), []byte(report.String()), 0o644))
	encoded, err := json.MarshalIndent(results, "", "  ")
	s.Require().NoError(err)
	s.Require().NoError(os.WriteFile(filepath.Join(out, "replies.json"), encoded, 0o644))
	s.T().Logf("wrote %s", out)
}

// replyCases are the cases the agent should answer next: a settled turn to reply to, or a
// settled interruption to reply to after being cut off.
func replyCases(set flowSet) []flowCase {
	var cases []flowCase
	for _, one := range set.Cases {
		if one.Expect == outcomeAnswer || (one.Expect == outcomeInterrupt && !one.Unfinished) {
			cases = append(cases, one)
		}
	}
	return cases
}

// answer has the model reply through the production harness, times it and grades it.
func (s *ReplyBenchmarkSuite) answer(session *llmrouter.Session, set flowSet, one flowCase) graded {
	result := graded{CaseID: one.ID, Heard: one.Heard}

	harness, err := New(Options{Model: session, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	history := one.turn(one.ID, set.Contracts).History
	if one.AgentSaid != "" {
		history = append(history, llm.Message{Role: llm.Assistant, Content: one.AgentSaid})
	}
	history = append(history, llm.Message{Role: llm.User, Content: one.Heard})

	ctx, cancel := context.WithTimeout(s.ctx, replyDeadline)
	defer cancel()
	askedAt := time.Now()
	stream, err := harness.Respond(ctx, Turn{
		ID: one.ID, Instructions: set.Contracts[one.Contract], History: history,
	})
	if err != nil {
		result.Failed = err.Error()
		return result
	}
	var reply strings.Builder
	for stream.Next() {
		delta, ok := stream.Current().(llm.OutputTextDelta)
		if !ok {
			continue
		}
		if spoken := harness.Filter(one.ID, delta.Delta); spoken != "" {
			if reply.Len() == 0 {
				result.FirstTextMs = float64(time.Since(askedAt).Microseconds()) / 1000
			}
			reply.WriteString(spoken)
		}
	}
	reply.WriteString(harness.Flush())
	result.TookMs = float64(time.Since(askedAt).Microseconds()) / 1000
	if err := stream.Err(); err != nil {
		result.Failed = err.Error()
		return result
	}
	usage := stream.Response().Usage
	result.Usage = routing.Usage{InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens}
	result.Reply = strings.TrimSpace(reply.String())
	result.Words = len(strings.Fields(result.Reply))

	if err := s.grade(set, one, &result); err != nil {
		result.Failed = "grading: " + err.Error()
	}
	return result
}

// grade asks the judge about one reply.
func (s *ReplyBenchmarkSuite) grade(set flowSet, one flowCase, result *graded) error {
	var asked strings.Builder
	fmt.Fprintf(&asked, "The agent was told:\n%s\n\nThe conversation so far:\n",
		set.Contracts[one.Contract])
	for _, said := range one.History {
		fmt.Fprintf(&asked, "%s: %s\n", said.Speaker, said.Text)
	}
	if one.AgentSaid != "" {
		fmt.Fprintf(&asked, "agent (cut off): %s\n", one.AgentSaid)
	}
	fmt.Fprintf(&asked, "\nThe words it is replying to, from %s:\n%s\n\nIts reply:\n%s\n",
		one.Participant, one.Heard, result.Reply)

	ctx, cancel := context.WithTimeout(s.ctx, replyDeadline)
	defer cancel()
	stream, err := s.judge.Create(ctx, llm.ResponseParams{
		ID:              "grade-" + one.ID,
		Instructions:    replyJudgeInstructions,
		Input:           []llm.Message{{Role: llm.User, Content: asked.String()}},
		MaxOutputTokens: 500,
		Text:            llm.TextParams{Format: llm.FormatJSONObject},
	})
	if err != nil {
		return err
	}
	response, err := llm.Collect(stream)
	if err != nil {
		return err
	}
	return json.Unmarshal([]byte(llm.Unfence(response.OutputText)), result)
}

// replyReport is the table a human reads for one set, and the replies that failed.
func replyReport(name string, set flowSet, models []string, results map[string][]graded) string {
	var out strings.Builder
	fmt.Fprintf(&out, "\n## Replies on the %s set\n\n", name)
	if set.Source != "" {
		fmt.Fprintf(&out, "Taken from the %s.\n\n", set.Source)
	}
	fmt.Fprintln(&out, "| Model | Passed | Relevant | Grounded | Spoken | Score | Words p50 |"+
		" First text p50 | p95 | Whole reply p50 | Failed |")
	fmt.Fprintln(&out, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
	for _, model := range models {
		replies := results[model]
		var passed, relevant, grounded, spoken, score, judged, failed int
		var words, first, took []float64
		for _, one := range replies {
			if one.Failed != "" {
				failed++
				continue
			}
			judged++
			if one.passed() {
				passed++
			}
			if one.Relevant {
				relevant++
			}
			if one.Grounded {
				grounded++
			}
			if one.Spoken {
				spoken++
			}
			score += one.Score
			words = append(words, float64(one.Words))
			first = append(first, one.FirstTextMs)
			took = append(took, one.TookMs)
		}
		mean := 0.0
		if judged > 0 {
			mean = float64(score) / float64(judged)
		}
		fmt.Fprintf(&out, "| `%s` | %s | %s | %s | %s | %.2f | %.0f | %.0fms | %.0fms | %.0fms | %d |\n",
			model, share(passed, judged), share(relevant, judged), share(grounded, judged),
			share(spoken, judged), mean, percentile(words, 0.5), percentile(first, 0.5),
			percentile(first, 0.95), percentile(took, 0.5), failed)
	}

	for _, model := range models {
		var worst []graded
		for _, one := range results[model] {
			if !one.passed() {
				worst = append(worst, one)
			}
		}
		if len(worst) == 0 {
			continue
		}
		fmt.Fprintf(&out, "\n### Where %s fell short\n\n", model)
		for _, one := range worst {
			why := one.Reason
			if one.Failed != "" {
				why = one.Failed
			}
			fmt.Fprintf(&out, "- `%s` heard %q, said %q: %s\n", one.CaseID, one.Heard, one.Reply, why)
		}
	}
	return out.String()
}
