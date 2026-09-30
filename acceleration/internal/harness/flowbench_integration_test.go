//go:build integration

package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm/typesafe"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// benchmarkEnvVar has to be set for the benchmark to run, because six hundred round trips to
// two vendors is not something an integration suite should do by walking past it.
const benchmarkEnvVar = "FLOW_BENCHMARK"

// modelsEnvVar names the router targets to compare, comma separated: an alias such as
// llm-flow, a provider and model such as cerebras/gemma-4-31b, all for every LLM the router is
// configured with, or none to run only the Jev arms. Each is asked the production prompt and read with the
// production parser.
const modelsEnvVar = "FLOW_BENCHMARK_MODELS"

// defaultModels are the controller as deployed and the Gemma 4 deployment of our own, behind
// GEMMA_BASE_URL. A target that cannot be reached is skipped rather than failing the run.
const defaultModels = "llm-flow,gemma/gemma-4-26B-A4B-it"

// localEnvVar is an OpenAI-compatible endpoint on this machine, up to and including /v1, such
// as gophonic-server serving Qwen3-8B. It is compared with the others as one more arm, free of
// charge and of the network.
const localEnvVar = "FLOW_BENCHMARK_LOCAL"

// localModelEnvVar is the model the local endpoint is asked for.
const localModelEnvVar = "FLOW_BENCHMARK_LOCAL_MODEL"

// effortEnvVar asks every model arm to reason this hard, such as max, with the output budget
// of teacherOutputTokens instead of the controller's. That measures a model as a labeller of
// training data, which may think as long as it likes, rather than as the controller.
const effortEnvVar = "FLOW_BENCHMARK_EFFORT"

// repeatsEnvVar overrides how often each case is put to a model arm, such as 1 when a judge
// checks training cases rather than a controller being measured.
const repeatsEnvVar = "FLOW_BENCHMARK_REPEATS"

// workersEnvVar puts that many cases to an arm at once. Latencies then include the contention,
// so it is for judging a large set, not for timing a controller.
const workersEnvVar = "FLOW_BENCHMARK_WORKERS"

// teacherOutputTokens is the budget a labeller answers within, room for long thinking.
const teacherOutputTokens = 32768

// sampleEnvVar runs a fraction of each set, such as 0.05, for a sweep across many models that
// would cost too much in full. Every model and every run is asked the same cases.
const sampleEnvVar = "BENCHMARK_SAMPLE"

// setsEnvVar picks the labelled sets, comma separated: written, ami, or the path of a set file
// in the same format, such as generated training cases to have a judge check.
const setsEnvVar = "FLOW_BENCHMARK_SETS"

// modelRepeats is how often each case is put to a model. It samples, so one answer measures a
// draw rather than the model, and a controller that flips between two answers for the same
// words is a controller that flips mid-call.
const modelRepeats = 3

// jevRepeats is one. Jev returns the distribution rather than a draw from it, so asking twice
// measures the network.
const jevRepeats = 1

// jevModel pins the version, because jev-latest moves when a release ships and a comparison
// across runs is only a comparison if the same model answered both.
const jevModel = "jev-1.13.0"

// jevRecentTurns is how much of the conversation the decomposed arm sends. TypeSafe's own
// guidance is that accuracy falls as the state fills with what the decision does not need, and
// who holds the floor now is decided by the last exchange rather than the call's history.
const jevRecentTurns = 2

// jevNeutral is the only threshold in the composed arm, and it is the point at which a
// probability stops leaning one way. Anything else would be a number fitted to this set.
const jevNeutral = 0.5

// jevConfident is where a choice's own confidence stops being worth acting on unasked. It is
// reported rather than applied, because what to do with an uncertain judgement is a decision
// about the call and not about the model.
const jevConfident = 0.6

// jevInputPerMillion is $42 per billion input tokens. Output tokens are not billed.
const jevInputPerMillion = 0.042

// benchDeadline bounds one judgement. It is longer than the production flowDeadline on purpose:
// a benchmark that scores a slow answer as a wrong one cannot tell the two apart.
const benchDeadline = 20 * time.Second

// retries is how often a rate-limited request is asked again before it is given up on.
const retries = 4

// judgement is one arm's answer to one case.
type judgement struct {
	CaseID  string      `json:"case_id"`
	State   flowState   `json:"state"`
	Attempt int         `json:"attempt"`
	Expect  flowOutcome `json:"expect"`
	Got     flowOutcome `json:"got"`
	TookMs  float64     `json:"took_ms"`
	// Unreadable says the model's answer could not be parsed, so what Got holds is the
	// fallback the controller takes rather than anything the model chose.
	Unreadable bool `json:"unreadable,omitempty"`
	// Raw is what the model wrote when it could not be read, so a failure can be looked at
	// rather than guessed at.
	Raw string `json:"raw,omitempty"`
	// Confidence is how peaked the distribution behind the answer was, and is zero for an
	// arm whose model does not report one.
	Confidence float64       `json:"confidence,omitempty"`
	Usage      routing.Usage `json:"usage"`
}

func (j judgement) right() bool { return j.Got == j.Expect }

// arm is one model, asked one way.
type arm struct {
	name    string
	model   string
	repeats int
	// price turns what a judgement consumed into millionths of a dollar.
	price func(routing.Usage) int64
	judge func(ctx context.Context, set flowSet, one flowCase, attempt int) (judgement, error)
}

type FlowBenchmarkSuite struct {
	suite.Suite
	ctx  context.Context
	sets []string
}

func TestFlowBenchmarkSuite(t *testing.T) {
	suite.Run(t, new(FlowBenchmarkSuite))
}

func (s *FlowBenchmarkSuite) SetupSuite() {
	if os.Getenv(benchmarkEnvVar) == "" {
		s.T().Skip(benchmarkEnvVar + " not set")
	}
	s.ctx = context.Background()
	s.sets = strings.Split(envOr(setsEnvVar, writtenSet+","+amiSet), ",")
}

// TestFlowControllerBenchmark puts every case of every set to every arm that can be reached and
// reports what each would have made the agent do.
func (s *FlowBenchmarkSuite) TestFlowControllerBenchmark() {
	arms := s.modelArms()
	if os.Getenv(localEnvVar) != "" {
		arms = append(arms, s.localChoiceArm())
	}
	arms = append(arms, s.jevArms()...)
	s.Require().NotEmpty(arms, "no arm can be reached: name a target in "+modelsEnvVar+
		" whose provider has a key, or set TYPESAFE_API_KEY")

	var report strings.Builder
	results := map[string][]tally{}
	for _, name := range s.sets {
		set, err := loadNamedSet(name)
		s.Require().NoError(err)
		set = set.sample(sampleFraction())
		tallies := make([]tally, 0, len(arms))
		for _, one := range arms {
			s.T().Logf("running %s over the %s set, %d cases, %d times each",
				one.name, name, len(set.Cases), one.repeats)
			tallies = append(tallies, s.run(one, set))
		}
		report.WriteString(s.report(name, set, tallies))
		results[name] = tallies
	}

	fmt.Print(report.String())
	s.write(report.String(), results)
}

// run puts every case to one arm, in order and one at a time, because a benchmark that reports
// latency cannot also be saturating the provider it is measuring.
func (s *FlowBenchmarkSuite) run(one arm, set flowSet) tally {
	counted := newTally(one)
	workers, _ := strconv.Atoi(envOr(workersEnvVar, "1"))
	var mu sync.Mutex
	var wg sync.WaitGroup
	queue := make(chan flowCase)
	for range max(workers, 1) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for subject := range queue {
				for attempt := range one.repeats {
					ctx, cancel := context.WithTimeout(s.ctx, benchDeadline)
					decided, err := s.ask(ctx, one, set, subject, attempt)
					cancel()
					mu.Lock()
					if err != nil {
						// A vendor that will not answer is worth reporting as its own failure
						// rather than as a wrong judgement, which is a claim about the model.
						s.T().Logf("%s could not judge %q: %v", one.name, subject.ID, err)
						counted.refused++
					} else {
						counted.add(decided)
					}
					mu.Unlock()
				}
			}
		}()
	}
	for _, subject := range set.Cases {
		queue <- subject
	}
	close(queue)
	wg.Wait()
	return counted
}

// ask asks once, and again after a wait when the vendor said it was too busy.
func (s *FlowBenchmarkSuite) ask(
	ctx context.Context, one arm, set flowSet, subject flowCase, attempt int,
) (judgement, error) {
	var last error
	for try := range retries {
		decided, err := one.judge(ctx, set, subject, attempt)
		if err == nil {
			return decided, nil
		}
		last = err
		if !busy(err) {
			return judgement{}, err
		}
		select {
		case <-ctx.Done():
			return judgement{}, ctx.Err()
		case <-time.After(time.Duration(1<<try) * time.Second):
		}
	}
	return judgement{}, last
}

// busy reports whether a failure is one that waiting fixes.
func busy(err error) bool {
	var refused *typesafe.StatusError
	if errors.As(err, &refused) {
		return refused.Retryable()
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "429") || strings.Contains(message, "rate limit") ||
		strings.Contains(message, "too many requests")
}

// modelArms are the router targets being compared, each asked exactly as the production
// controller asks: its prompt, its parser, and the fallback it takes when an answer will not
// parse.
//
// A target that cannot be reached is left out, which is how an undeployed Gemma or a missing
// key leaves the others to run rather than taking the whole benchmark with it.
func (s *FlowBenchmarkSuite) modelArms() []arm {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	router, err := llmrouter.New(llmrouter.Options{
		Config:   config[routing.LLM],
		Registry: llmrouter.DefaultRegistry(),
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	var arms []arm
	for _, target := range targets(config, envOr(modelsEnvVar, defaultModels)) {
		session, err := router.Start(s.ctx, llmrouter.Request{
			CustomerID: "flow-benchmark", Target: target,
		})
		if err != nil {
			s.T().Logf("skipping %s, it is out of reach: %v", target, err)
			continue
		}
		s.T().Cleanup(func() { _ = session.Close() })
		arms = append(arms, modelArm(target, os.Getenv(effortEnvVar), session, session.Price().CostMicros))
	}
	return arms
}

// localChoiceArm asks the local model the production policy as a multiple-choice question,
// through gophonic's /v1/classifications. Asked for JSON instead, Qwen3-8B wrote values the
// parser refuses or nothing at all. The model scores one answer letter rather than writing JSON,
// so it cannot answer with something the conversation cannot read, and the question is
// evaluated once and kept prepared: each case costs only its own words and one token.
func (s *FlowBenchmarkSuite) localChoiceArm() arm {
	endpoint := strings.TrimSuffix(os.Getenv(localEnvVar), "/") + "/classifications"
	model := envOr(localModelEnvVar, "Qwen3-8B")
	return arm{
		name:    "local-choice",
		model:   "local/" + model,
		repeats: 1,
		price:   func(routing.Usage) int64 { return 0 },
		judge: func(ctx context.Context, set flowSet, one flowCase, attempt int) (judgement, error) {
			asked := localChoices(one)
			input := localInput(one, set)
			labels := make([]string, len(asked))
			for i, option := range asked {
				labels[i] = option.label
			}
			body, err := json.Marshal(map[string]any{
				"model": model, "input": input,
				"question": localQuestion(one), "labels": labels,
			})
			if err != nil {
				return judgement{}, err
			}

			askedAt := time.Now()
			request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
			if err != nil {
				return judgement{}, err
			}
			request.Header.Set("Content-Type", "application/json")
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				return judgement{}, err
			}
			defer response.Body.Close()
			if response.StatusCode != http.StatusOK {
				raw, _ := io.ReadAll(response.Body)
				return judgement{}, fmt.Errorf("local classification: %s: %s", response.Status, raw)
			}
			var answered struct {
				Results []struct {
					Classes []struct {
						Label       string  `json:"label"`
						Probability float64 `json:"probability"`
					} `json:"classes"`
				} `json:"results"`
			}
			if err := json.NewDecoder(response.Body).Decode(&answered); err != nil {
				return judgement{}, err
			}
			if len(answered.Results) != 1 || len(answered.Results[0].Classes) != len(asked) {
				return judgement{}, fmt.Errorf("local classification: unexpected answer %+v", answered)
			}
			best := 0
			for i, class := range answered.Results[0].Classes {
				if class.Probability > answered.Results[0].Classes[best].Probability {
					best = i
				}
			}
			return judgement{
				CaseID:     one.ID,
				State:      one.State,
				Attempt:    attempt,
				Expect:     one.Expect,
				TookMs:     float64(time.Since(askedAt).Microseconds()) / 1000,
				Got:        asked[best].outcome,
				Confidence: answered.Results[0].Classes[best].Probability,
			}, nil
		},
	}
}

// localChoice is one option the local model may pick, and what the agent then does.
type localChoice struct {
	label   string
	outcome flowOutcome
}

// localChoices are the outcomes the conversation can reach from where the case stands, which
// is what keeps a small model from answering with one it cannot.
func localChoices(one flowCase) []localChoice {
	switch {
	case !one.AgentSpeaking:
		return []localChoice{
			{"respond: a complete thought addressed to the agent", outcomeAnswer},
			{"wait: probably unfinished, or a recorded menu still reading its options", outcomeWait},
			{"clarify: addressed to the agent, but what it wants is ambiguous", outcomeClarify},
			{"ignore: background speech, or addressed to somebody else", outcomeIgnore},
		}
	case one.Unfinished:
		return []localChoice{
			{"stop: a correction, a different request, a question, or a direct interruption", outcomeInterrupt},
			{"shorten: one more item of the same kind for the request being answered, neither correcting nor replacing it", outcomeShorten},
			{"continue: an acknowledgement, a noise, an echo of the agent, unrelated background speech, or too short to tell", outcomeContinue},
		}
	default:
		return []localChoice{
			{"stop: a correction, a different request, a question, or a direct interruption", outcomeInterrupt},
			{"shorten: one more item of the same kind for the request being answered, neither correcting nor replacing it", outcomeShorten},
			{"continue: a brief acknowledgement, a noise, or an echo of the agent's own words", outcomeContinue},
			{"ignore: background speech, or addressed to somebody else", outcomeIgnore},
		}
	}
}

// localInput is what the local model classifies: the agent's instructions and the case, in
// the production question's words, less its output format.
func localInput(one flowCase, set flowSet) string {
	turn := one.turn(one.ID, set.Contracts)
	return strings.TrimSpace("The agent has been told: " + turn.Instructions + "\n\n" +
		strings.TrimSuffix(strings.TrimSuffix(flowQuestion(turn), "Return the JSON object."),
			"Decide only the floor. "))
}

// trainExportEnvVar is a JSONL file to write every case of the sets in trainSetsEnvVar to,
// asked exactly as the local-choice arm asks: the question, its lettered options, the input,
// and which options are right. It is the training data for a local controller, and, for the
// benchmark's own sets, what that training is measured on.
const trainExportEnvVar = "FLOW_TRAIN_EXPORT"

// trainSetsEnvVar lists the sets to export, comma separated: written, ami, or the path of a
// set file in the same format, such as one testdata/ami/extract.go -all writes.
const trainSetsEnvVar = "FLOW_TRAIN_SETS"

func TestFlowTrainExport(t *testing.T) {
	out := os.Getenv(trainExportEnvVar)
	if out == "" {
		t.Skip(trainExportEnvVar + " not set")
	}
	file, err := os.Create(out)
	require.NoError(t, err)
	defer file.Close()
	encoder := json.NewEncoder(file)
	for _, name := range strings.Split(envOr(trainSetsEnvVar, "written,ami"), ",") {
		set, err := loadNamedSet(name)
		require.NoError(t, err, name)
		for _, one := range set.Cases {
			choices := localChoices(one)
			labels, correct := make([]string, len(choices)), []int{}
			for i, choice := range choices {
				labels[i] = choice.label
				if choice.outcome == one.Expect {
					correct = append(correct, i)
				}
			}
			if len(correct) == 0 {
				continue // no option reaches what the case expects
			}
			require.NoError(t, encoder.Encode(map[string]any{
				"id": one.ID, "set": name, "state": one.State, "question": localQuestion(one),
				"options": labels, "input": localInput(one, set), "correct": correct,
				// The words themselves, as they appear in input, so a trainer can vary how they
				// are written without that varying with the label.
				"heard": one.Heard,
			}))
		}
	}
}

// localQuestion is the production policy for the situation the case is in, in the production
// prompt's own words, less the output format the letters replace.
func localQuestion(one flowCase) string {
	const role = "You control the floor of a live voice conversation between an agent and a caller. " +
		"You never talk to the caller; another model answers them. "
	if !one.AgentSpeaking {
		return role + "The agent is not speaking and the words below have just been said. " +
			"Choose wait when the words are probably incomplete, especially when they end on a " +
			"PIN, member ID, phone number, or clock time that may still be growing. A recorded " +
			"menu reading out its options is one thought however long its pauses: wait until it " +
			"has asked for a choice. Words in a different voice usually come from somebody else " +
			"in the room, so lean towards ignore unless they plainly address the agent. What " +
			"should happen with the words?"
	}
	return role + "The agent is speaking when the words below arrive. Stop as soon as they are a " +
		"correction, a new request, a question, or a direct interruption such as \"wait\", " +
		"\"no\", or \"hang on\". Words that only repeat what the agent is saying are the " +
		"caller's line echoing it back. Never talk over a recorded menu. What should the agent do?"
}

// modelArm asks one model the production question.
func modelArm(target, effort string, session llm.LLM, price func(routing.Usage) int64) arm {
	name, budget := target, 512
	repeats, err := strconv.Atoi(envOr(repeatsEnvVar, strconv.Itoa(modelRepeats)))
	if err != nil || repeats < 1 {
		repeats = modelRepeats
	}
	if effort != "" {
		name, budget = target+"@"+effort, teacherOutputTokens
	}
	return arm{
		name:    name,
		model:   session.Provider() + "/" + session.Model(),
		repeats: repeats,
		price:   price,
		judge: func(ctx context.Context, set flowSet, one flowCase, attempt int) (judgement, error) {
			turn := one.turn(one.ID+"-"+strconv.Itoa(attempt), set.Contracts)
			askedAt := time.Now()
			stream, err := session.Create(ctx, llm.ResponseParams{
				ID:           turn.ID,
				Instructions: flowInstructions + "\n\nThe agent has been told:\n" + turn.Instructions,
				Input:        []llm.Message{{Role: llm.User, Content: flowQuestion(turn)}},
				// The same budget the controller runs with, unless it is being measured as a
				// labeller. A thinking model that spends it before the closing brace is a real
				// failure mode and is scored as one.
				MaxOutputTokens: budget,
				Reasoning:       llm.ReasoningParams{Effort: effort},
				Text:            llm.TextParams{Format: llm.FormatJSONObject},
			})
			if err != nil {
				return judgement{}, err
			}
			response, err := llm.Collect(stream)
			if err != nil {
				return judgement{}, err
			}

			decided := judgement{
				CaseID:  one.ID,
				State:   one.State,
				Attempt: attempt,
				Expect:  one.Expect,
				TookMs:  float64(time.Since(askedAt).Microseconds()) / 1000,
				Usage: routing.Usage{
					InputTokens:       response.Usage.InputTokens,
					CachedInputTokens: response.Usage.InputTokensDetails.CachedTokens,
					OutputTokens:      response.Usage.OutputTokens,
				},
			}
			answer, err := parseFlow(response.OutputText)
			if err != nil {
				// The fallbacks the controller itself takes, so the row says what a call
				// would have done rather than leaving a hole in the table.
				decided.Unreadable = true
				decided.Raw = response.OutputText
				answer = flowAnswer{Disposition: Respond, Floor: Continue}
				if turn.Unfinished {
					answer = flowAnswer{Disposition: Wait, Floor: Stop}
				}
			}
			decided.Got = one.outcome(answer.Disposition, answer.Floor)
			return decided, nil
		},
	}
}

// jevArms are the two ways of asking Jev the same thing: the two choices on their own, closest
// to what the model arms are asked, and the same two with the judgements the conversation already
// hard-codes asked for separately and applied in code. Empty when there is no key.
func (s *FlowBenchmarkSuite) jevArms() []arm {
	if os.Getenv("TYPESAFE_API_KEY") == "" {
		s.T().Log("TYPESAFE_API_KEY not set, skipping the Jev arms")
		return nil
	}

	client, err := typesafe.New(typesafe.Options{
		Model:   jevModel,
		Timeout: benchDeadline,
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)

	price := func(used routing.Usage) int64 {
		return routing.Price{PerMillionInputTokens: jevInputPerMillion}.CostMicros(used)
	}

	return []arm{
		{
			name:    "jev-direct",
			model:   client.Model(),
			repeats: jevRepeats,
			price:   price,
			judge: func(ctx context.Context, set flowSet, one flowCase, attempt int) (judgement, error) {
				return s.askJev(ctx, client, set, one, attempt, jevChoices(), false)
			},
		},
		{
			name:    "jev-composed",
			model:   client.Model(),
			repeats: jevRepeats,
			price:   price,
			judge: func(ctx context.Context, set flowSet, one flowCase, attempt int) (judgement, error) {
				return s.askJev(ctx, client, set, one, attempt, jevComposed(), true)
			},
		},
		{
			name:    "jev-decomposed",
			model:   client.Model(),
			repeats: jevRepeats,
			price:   price,
			judge: func(ctx context.Context, set flowSet, one flowCase, attempt int) (judgement, error) {
				return s.askJevDecomposed(ctx, client, set, one, attempt)
			},
		},
	}
}

// jevDecomposed is every fact the conversation's policy turns on, each asked as its own yes or
// no, which is how TypeSafe says its model is meant to be used: one well-scoped question at a
// time, answered literally, with the combining done in code. No question asks what to do.
func jevDecomposed() map[string]lcm.Question {
	return map[string]lcm.Question{
		"addressed_to_agent": lcm.Noul(
			"Were the words in `heard` meant for the agent described in `agent_was_told`?",
			"Spoken to the agent, whether or not they are finished.",
			"Spoken to somebody else in the room, to a pet or a child, read off a television, "+
				"or otherwise not meant for the agent."),
		"finished": lcm.Noul(
			"Is `heard` a complete thought that now waits for the agent to reply?",
			"The speaker has said what they meant to say and expects an answer.",
			"The words stop part way through a sentence, a list, a number or a name, so "+
				"more is coming."),
		"still_growing": lcm.Noul(
			"Does `heard` end part way through a number, an identifier or a time that the "+
				"speaker is still reading out?",
			"It ends mid-sequence, so more digits or words are still coming.",
			"Whatever number it contains is complete, or it contains none."),
		"recorded_menu": lcm.Noul(
			"Is `heard` a recording reading out its options rather than a person talking?",
			"An automated menu, hold message or greeting.",
			"A person speaking, however stilted."),
		"non_speech": lcm.Noul(
			"Is `heard` a noise rather than words: a cough, a sneeze, a laugh, a door, "+
				"static, or something else in the room?",
			"A noise, or a transcriber's description of one.",
			"Words the speaker meant to say, however short."),
		"ambiguous": lcm.Noul(
			"Is what `heard` asks the agent to do impossible to act on without asking which "+
				"thing the speaker means?",
			"It points at something `conversation` does not pin down, such as \"the usual\", "+
				"\"change it\" or \"put it back\" with nothing saying what it is.",
			"What is wanted is clear from `heard` and `conversation`, or it asks for nothing."),
		"echo": lcm.Noul(
			"Is `heard` the agent's own words from `agent_has_said` coming back, word for word "+
				"or nearly?",
			"The same words the agent just said, as a line echoing back.",
			"Words of the speaker's own."),
		"acknowledgement": lcm.Noul(
			"Is `heard` only a brief sign that the speaker is listening, such as \"yeah\", "+
				"\"okay\" or \"right\"?",
			"A listening noise that asks for nothing.",
			"It says or asks something."),
		"adds_to_request": lcm.Noul(
			"Does `heard` add to what the speaker asked for, without contradicting what the "+
				"agent is saying in `agent_has_said`?",
			"An addition such as \"and Saturday as well\" or \"and put us on the patio\".",
			"It corrects the agent, asks something unrelated, or adds nothing."),
		"objects": lcm.Noul(
			"Does `heard` correct the agent, tell it to stop or wait, or ask it something new, "+
				"while it is saying `agent_has_said`?",
			"A correction such as \"no, make it six\", \"that's the wrong date\", \"wait\", "+
				"or a new question.",
			"It agrees, acknowledges, repeats the agent, or is not for the agent at all."),
	}
}

// askJevDecomposed asks every fact at once over a trimmed state and decides in code.
func (s *FlowBenchmarkSuite) askJevDecomposed(
	ctx context.Context, client *typesafe.Client, set flowSet, one flowCase, attempt int,
) (judgement, error) {
	state := one.state(set.Contracts)
	if said := state["conversation"].([]map[string]string); len(said) > jevRecentTurns {
		state["conversation"] = said[len(said)-jevRecentTurns:]
	}

	askedAt := time.Now()
	answered, err := client.Classify(ctx, lcm.Request{State: state, Questions: jevDecomposed()})
	if err != nil {
		return judgement{}, err
	}
	return judgement{
		CaseID:  one.ID,
		State:   one.State,
		Attempt: attempt,
		Expect:  one.Expect,
		TookMs:  float64(time.Since(askedAt).Microseconds()) / 1000,
		Usage:   routing.Usage{InputTokens: answered.Usage.InputTokens},
		Got:     decomposedOutcome(one, answered.Answers),
	}, nil
}

// decomposedOutcome is the conversation's policy written over facts rather than over a model's
// choice of what to do. The order is the order of precedence: whether the words were speech,
// then whether they were for the agent, then what they ask of it.
func decomposedOutcome(one flowCase, answers map[string]lcm.Answer) flowOutcome {
	yes := func(fact string) bool { return answers[fact].Yes >= jevNeutral }

	if !one.AgentSpeaking {
		switch {
		case yes("non_speech"), !yes("addressed_to_agent"):
			return outcomeIgnore
		case yes("recorded_menu"), yes("still_growing"), !yes("finished"):
			return outcomeWait
		case yes("ambiguous"):
			return outcomeClarify
		default:
			return outcomeAnswer
		}
	}
	switch {
	case yes("non_speech"), yes("echo"), yes("acknowledgement"):
		return outcomeContinue
	case !yes("addressed_to_agent"):
		// Mid-utterance an ignore can only leave the agent talking, as in overlapRuled.
		if one.Unfinished {
			return outcomeContinue
		}
		return outcomeIgnore
	case yes("adds_to_request"):
		return outcomeShorten
	case yes("objects"):
		return outcomeInterrupt
	default:
		return outcomeContinue
	}
}

// The two choices, which are the two axes the conversation reads. Their options are described in
// the words the production prompt uses, so what differs between the arms is the model and the
// shape of the answer rather than the policy.
func jevChoices() map[string]lcm.Question {
	return map[string]lcm.Question{
		"disposition": lcm.Choice(
			"A voice agent is on a live call and has just heard `heard` from `speaker`. "+
				"What should it do with those words?",
			map[string]string{
				"respond": "A complete thought addressed to the agent, so it should answer.",
				"wait": "Probably unfinished, and especially so when it ends on a PIN, " +
					"a member ID, a phone number or a clock time that may still be growing, " +
					"or when it is a recorded menu that has not asked for a choice yet.",
				"clarify": "Addressed to the agent, but what it wants is ambiguous.",
				"ignore": "Background speech, or speech addressed to somebody other than " +
					"the agent.",
			}),
		// Asked whether or not the agent is speaking. When it is not, the answer decides
		// nothing and the code ignores it, which costs a few tokens and saves a round trip on
		// the cases where it does decide.
		"floor": lcm.Choice(
			"Assume the agent is in the middle of saying `agent_has_said` out loud when "+
				"`heard` arrives. Should it stop, cut its answer short, or carry on?",
			map[string]string{
				"stop": "`heard` is a correction, a new request, a question, or a direct " +
					"interruption such as \"wait\", \"no\" or \"hang on\".",
				"shorten": "`heard` adds to what was asked, which makes the answer being " +
					"spoken longer than it needs to be.",
				"continue": "`heard` is a brief acknowledgement, a cough or other " +
					"non-speech, the caller's line echoing the agent's own words back, or " +
					"speech that has nothing to do with the agent.",
			}),
	}
}

// jevComposed asks the two choices and, separately, the four judgements the conversation
// already makes in code rather than leaving to a model. Splitting them out is what lets the
// policy stay in Go: the model says what is true and the code says what to do about it.
func jevComposed() map[string]lcm.Question {
	questions := jevChoices()
	questions["addressed_to_agent"] = lcm.Noul(
		"Were the words in `heard` meant for the agent described in `agent_was_told`?",
		"Spoken to the agent, whether or not they are finished.",
		"Spoken to somebody else in the room, to a pet or a child, read off a television, "+
			"or otherwise not meant for the agent. `different_voice` being true is evidence "+
			"of this without settling it, because a second person may have leaned in to "+
			"answer for the caller.")
	questions["still_growing"] = lcm.Noul(
		"Does `heard` end part way through a number, an identifier or a time that the "+
			"speaker is still reading out?",
		"It ends mid-sequence, so more digits or words are still coming.",
		"Whatever number it contains is complete, or it contains none.")
	questions["recorded_menu"] = lcm.Noul(
		"Is `heard` a recording reading out its options rather than a person talking?",
		"An automated menu, hold message or greeting, which is one thought however long "+
			"the pauses between its parts.",
		"A person speaking, however stilted.")
	questions["non_speech"] = lcm.Noul(
		"Is `heard` a noise rather than words: a cough, a sneeze, a throat clear, a laugh, "+
			"a door, static, or something else in the room?",
		"A noise, or a transcriber's description of one.",
		"Words the speaker meant to say, however short.")
	return questions
}

// askJev puts one case to Jev and reads the answers the way the arm's own policy says to.
func (s *FlowBenchmarkSuite) askJev(
	ctx context.Context,
	client *typesafe.Client,
	set flowSet,
	one flowCase,
	attempt int,
	questions map[string]lcm.Question,
	composed bool,
) (judgement, error) {
	askedAt := time.Now()
	answered, err := client.Classify(ctx, lcm.Request{
		State:     one.state(set.Contracts),
		Questions: questions,
	})
	if err != nil {
		return judgement{}, err
	}

	disposition := Disposition(answered.Answers["disposition"].Chosen)
	floor := Floor(answered.Answers["floor"].Chosen)
	if !disposition.Valid() || !floor.Valid() {
		return judgement{}, fmt.Errorf(
			"harness: jev answered %q and %q, which are not a disposition and a floor",
			disposition, floor)
	}

	decided := judgement{
		CaseID:  one.ID,
		State:   one.State,
		Attempt: attempt,
		Expect:  one.Expect,
		TookMs:  float64(time.Since(askedAt).Microseconds()) / 1000,
		Usage:   routing.Usage{InputTokens: answered.Usage.InputTokens},
		Got:     one.outcome(disposition, floor),
	}
	// The axis that decides is the one whose confidence says whether to act unasked.
	decided.Confidence = answered.Answers["floor"].Confidence
	if !one.AgentSpeaking {
		decided.Confidence = answered.Answers["disposition"].Confidence
	}
	if composed {
		decided.Got = composedOutcome(one, answered.Answers, disposition, floor)
	}
	return decided, nil
}

// composedOutcome applies the three overrides the conversation already hard-codes, then reads
// the two choices as usual.
//
// They are the same three and no more: a noise does not take the floor (overlapNoise), a menu
// is never cut off, and words that were not for the agent are not answered. Every threshold is
// the neutral half, because a number chosen against this set would be a number this set could
// no longer measure.
func composedOutcome(
	one flowCase,
	answers map[string]lcm.Answer,
	disposition Disposition,
	floor Floor,
) flowOutcome {
	switch {
	case one.AgentSpeaking && answers["non_speech"].Yes >= jevNeutral:
		return outcomeContinue
	case !one.AgentSpeaking && answers["recorded_menu"].Yes >= jevNeutral:
		return outcomeWait
	case answers["addressed_to_agent"].Yes < jevNeutral:
		disposition = Ignore
	case !one.AgentSpeaking && answers["still_growing"].Yes >= jevNeutral:
		disposition = Wait
	}
	return one.outcome(disposition, floor)
}

// tally is what one arm scored.
type tally struct {
	Arm     string `json:"arm"`
	Model   string `json:"model"`
	Repeats int    `json:"repeats"`

	Judged  int `json:"judged"`
	Correct int `json:"correct"`
	// Refused counts the judgements a vendor would not make at all.
	Refused int `json:"refused"`
	// Unreadable counts the answers that had to fall back to what the controller does with
	// JSON it cannot parse.
	Unreadable int `json:"unreadable"`
	// MissedStop counts a caller who took the floor and did not get it, which is the agent
	// talking over a correction. FalseStop counts the agent giving the floor up to a cough.
	MissedStop int `json:"missed_stop"`
	FalseStop  int `json:"false_stop"`
	// StopWanted and StopNotWanted are what those two are out of. Without them the counts
	// cannot be compared between arms, because an arm asked three times per case has three
	// times as many chances to get one wrong.
	StopWanted    int `json:"stop_wanted"`
	StopNotWanted int `json:"stop_not_wanted"`
	// Unsure counts judgements whose own confidence was below jevConfident.
	Unsure int `json:"unsure"`
	// Flipped counts cases whose repeats did not agree with each other.
	Flipped int `json:"flipped"`

	QuietJudged     int `json:"quiet_judged"`
	QuietCorrect    int `json:"quiet_correct"`
	SpeakingJudged  int `json:"speaking_judged"`
	SpeakingCorrect int `json:"speaking_correct"`

	ByState   map[flowState]*stateTally           `json:"by_state"`
	Confusion map[flowOutcome]map[flowOutcome]int `json:"confusion"`

	P50Ms      float64       `json:"p50_ms"`
	P95Ms      float64       `json:"p95_ms"`
	CostMicros int64         `json:"cost_micros"`
	Usage      routing.Usage `json:"usage"`

	Judgements []judgement `json:"judgements"`

	refused   int
	latencies []float64
	price     func(routing.Usage) int64
	seen      map[string]map[flowOutcome]int
}

type stateTally struct {
	Judged  int `json:"judged"`
	Correct int `json:"correct"`
}

// quietFloor reports whether a state is one of the six the agent judges with nothing of its own
// being spoken, where the disposition is the axis that decides.
func quietFloor(state flowState) bool {
	switch state {
	case stateRespond, stateWait, stateWaitDigits, stateWaitMenu, stateClarify, stateIgnore:
		return true
	}
	return false
}

func newTally(one arm) tally {
	counted := tally{
		Arm:       one.name,
		Model:     one.model,
		Repeats:   one.repeats,
		ByState:   map[flowState]*stateTally{},
		Confusion: map[flowOutcome]map[flowOutcome]int{},
		price:     one.price,
		seen:      map[string]map[flowOutcome]int{},
	}
	for _, state := range flowStates {
		counted.ByState[state] = &stateTally{}
	}
	for _, expected := range flowOutcomes {
		counted.Confusion[expected] = map[flowOutcome]int{}
	}
	return counted
}

func (t *tally) add(decided judgement) {
	t.Judged++
	t.Judgements = append(t.Judgements, decided)
	t.latencies = append(t.latencies, decided.TookMs)
	t.Usage.InputTokens += decided.Usage.InputTokens
	t.Usage.CachedInputTokens += decided.Usage.CachedInputTokens
	t.Usage.OutputTokens += decided.Usage.OutputTokens
	t.CostMicros += t.price(decided.Usage)

	state := t.ByState[decided.State]
	state.Judged++
	t.Confusion[decided.Expect][decided.Got]++

	quiet := quietFloor(decided.State)
	if quiet {
		t.QuietJudged++
	} else {
		t.SpeakingJudged++
	}

	if decided.right() {
		t.Correct++
		state.Correct++
		if quiet {
			t.QuietCorrect++
		} else {
			t.SpeakingCorrect++
		}
	}
	if decided.Unreadable {
		t.Unreadable++
	}
	if decided.Expect == outcomeInterrupt {
		t.StopWanted++
		if decided.Got != outcomeInterrupt {
			t.MissedStop++
		}
	} else {
		t.StopNotWanted++
		if decided.Got == outcomeInterrupt {
			t.FalseStop++
		}
	}
	if decided.Confidence > 0 && decided.Confidence < jevConfident {
		t.Unsure++
	}

	if t.seen[decided.CaseID] == nil {
		t.seen[decided.CaseID] = map[flowOutcome]int{}
	}
	t.seen[decided.CaseID][decided.Got]++
}

// settle works out the numbers that need every judgement in before they mean anything.
func (t *tally) settle() {
	t.Refused = t.refused
	t.P50Ms = percentile(t.latencies, 0.5)
	t.P95Ms = percentile(t.latencies, 0.95)
	for _, answers := range t.seen {
		if len(answers) > 1 {
			t.Flipped++
		}
	}
}

func share(part, whole int) string {
	if whole == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f%%", 100*float64(part)/float64(whole))
}

// report writes the tables a human reads about one set.
func (s *FlowBenchmarkSuite) report(name string, set flowSet, tallies []tally) string {
	for i := range tallies {
		tallies[i].settle()
	}

	var out strings.Builder
	fmt.Fprintf(&out, "\n## The %s set\n\nRun %s over %d cases.\n\n",
		name, time.Now().UTC().Format(time.RFC3339), len(set.Cases))
	if set.Source != "" {
		fmt.Fprintf(&out, "Taken from the %s.\n\n", set.Source)
	}

	fmt.Fprintln(&out, "| Arm | Model | Correct | Floor free | Agent talking |"+
		" Missed stop | False stop | Unreadable | Flipped | p50 | p95 | $/1k |")
	fmt.Fprintln(&out, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"+
		" --- | --- | --- |")
	for _, counted := range tallies {
		perThousand := 0.0
		if counted.Judged > 0 {
			perThousand = float64(counted.CostMicros) / float64(counted.Judged) / 1000
		}
		// An arm asked once cannot disagree with itself, so it has no flip rate rather than
		// a flip rate of nothing.
		flipped := "n/a"
		if counted.Repeats > 1 {
			flipped = share(counted.Flipped, len(set.Cases))
		}
		fmt.Fprintf(&out,
			"| %s | `%s` | %s | %s | %s | %s | %s | %d | %s | %.0fms | %.0fms | $%.3f |\n",
			counted.Arm, counted.Model,
			share(counted.Correct, counted.Judged),
			share(counted.QuietCorrect, counted.QuietJudged),
			share(counted.SpeakingCorrect, counted.SpeakingJudged),
			share(counted.MissedStop, counted.StopWanted),
			share(counted.FalseStop, counted.StopNotWanted),
			counted.Unreadable,
			flipped, counted.P50Ms, counted.P95Ms, perThousand)
	}

	fmt.Fprintf(&out, "\n### By state\n\n")
	fmt.Fprint(&out, "| State | Wants |")
	for _, counted := range tallies {
		fmt.Fprintf(&out, " %s |", counted.Arm)
	}
	fmt.Fprint(&out, "\n| --- | --- |")
	for range tallies {
		fmt.Fprint(&out, " --- |")
	}
	fmt.Fprintln(&out)
	counts := set.counts()
	for _, state := range flowStates {
		if counts[state] == 0 {
			continue
		}
		fmt.Fprintf(&out, "| `%s` | `%s` |", state, set.expected(state))
		for _, counted := range tallies {
			scored := counted.ByState[state]
			fmt.Fprintf(&out, " %s |", share(scored.Correct, scored.Judged))
		}
		fmt.Fprintln(&out)
	}

	for _, counted := range tallies {
		fmt.Fprintf(&out, "\n### What %s did instead on the %s set\n\n", counted.Arm, name)
		fmt.Fprintf(&out,
			"Rows are what the case wanted, columns what the arm would have done.\n\n")
		fmt.Fprint(&out, "| wanted |")
		for _, got := range flowOutcomes {
			fmt.Fprintf(&out, " %s |", got)
		}
		fmt.Fprint(&out, "\n| --- |")
		for range flowOutcomes {
			fmt.Fprint(&out, " --- |")
		}
		fmt.Fprintln(&out)
		for _, wanted := range flowOutcomes {
			if len(counted.Confusion[wanted]) == 0 {
				continue
			}
			fmt.Fprintf(&out, "| `%s` |", wanted)
			for _, got := range flowOutcomes {
				count := counted.Confusion[wanted][got]
				if count == 0 {
					fmt.Fprint(&out, " |")
					continue
				}
				fmt.Fprintf(&out, " %d |", count)
			}
			fmt.Fprintln(&out)
		}
		if counted.Unsure > 0 {
			fmt.Fprintf(&out,
				"\n%s of judgements came back under %.2f confidence, which is what a "+
					"deployment would escalate rather than act on.\n",
				share(counted.Unsure, counted.Judged), jevConfident)
		}
		if counted.Refused > 0 {
			fmt.Fprintf(&out, "\n%d judgements were refused outright.\n", counted.Refused)
		}
	}
	return out.String()
}

// write keeps the run, because a table in a terminal is not a baseline.
func (s *FlowBenchmarkSuite) write(report string, results map[string][]tally) {
	out := filepath.Join("testdata", "flowbench-out",
		time.Now().UTC().Format("20060102-150405"))
	s.Require().NoError(os.MkdirAll(out, 0o755))

	s.Require().NoError(os.WriteFile(filepath.Join(out, "report.md"), []byte(report), 0o644))

	for _, tallies := range results {
		sort.Slice(tallies, func(i, j int) bool { return tallies[i].Arm < tallies[j].Arm })
	}
	encoded, err := json.MarshalIndent(results, "", "  ")
	s.Require().NoError(err)
	s.Require().NoError(os.WriteFile(filepath.Join(out, "summary.json"), encoded, 0o644))
	s.T().Logf("wrote %s", out)
}

// envOr is the variable's value, or fallback when it is unset.
func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

// targets reads a list of router targets, where all means every configured LLM and none
// means no model at all, which leaves the Jev arms to run on their own.
func targets(config routing.Config, listed string) []string {
	switch listed {
	case "none":
		return nil
	case "all":
	default:
		return strings.Split(listed, ",")
	}
	var every []string
	for _, one := range config[routing.LLM].Providers {
		every = append(every, one.Provider+"/"+one.Model)
	}
	return every
}

// sampleFraction is what sampleEnvVar asks for, or all of every set when it is unset.
func sampleFraction() float64 {
	fraction, err := strconv.ParseFloat(os.Getenv(sampleEnvVar), 64)
	if err != nil {
		return 1
	}
	return fraction
}
