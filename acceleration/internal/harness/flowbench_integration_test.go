//go:build integration

package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/typesafe"
)

// benchmarkEnvVar has to be set for the benchmark to run, because six hundred round trips to
// two vendors is not something an integration suite should do by walking past it.
const benchmarkEnvVar = "FLOW_BENCHMARK"

// defaultGemmaTarget is the Gemma 4 deployment of our own, behind GEMMA_BASE_URL.
const defaultGemmaTarget = "gemma/gemma-4-26B-A4B-it"

// gemmaTargetEnvVar names a different incumbent, which is how the same set is put to Gemma 4
// 31B on Cerebras' public inference API, or to whatever replaces it.
const gemmaTargetEnvVar = "FLOW_BENCHMARK_GEMMA"

// gemmaRepeats is how often each case is put to Gemma. It samples, so one answer measures a
// draw rather than the model, and a controller that flips between two answers for the same
// words is a controller that flips mid-call.
const gemmaRepeats = 3

// jevRepeats is one. Jev returns the distribution rather than a draw from it, so asking twice
// measures the network.
const jevRepeats = 1

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
	judge func(ctx context.Context, one flowCase, attempt int) (judgement, error)
}

type FlowBenchmarkSuite struct {
	suite.Suite
	ctx context.Context
	set flowSet
}

func TestFlowBenchmarkSuite(t *testing.T) {
	suite.Run(t, new(FlowBenchmarkSuite))
}

func (s *FlowBenchmarkSuite) SetupSuite() {
	if os.Getenv(benchmarkEnvVar) == "" {
		s.T().Skip(benchmarkEnvVar + " not set")
	}
	s.ctx = context.Background()

	set, err := loadFlowSet()
	s.Require().NoError(err)
	s.set = set
}

// TestFlowControllerBenchmark puts every case to every arm that has a key and reports what each
// would have made the agent do.
func (s *FlowBenchmarkSuite) TestFlowControllerBenchmark() {
	arms := []arm{}
	if gemma := s.gemmaArm(); gemma != nil {
		arms = append(arms, *gemma)
	}
	for _, jev := range s.jevArms() {
		arms = append(arms, jev)
	}
	s.Require().NotEmpty(arms, "no arm can be reached: set GEMMA_BASE_URL with "+
		"BASETEN_API_KEY, or TYPESAFE_API_KEY, or both")

	tallies := make([]tally, 0, len(arms))
	for _, one := range arms {
		s.T().Logf("running %s over %d cases, %d times each",
			one.name, len(s.set.Cases), one.repeats)
		tallies = append(tallies, s.run(one))
	}

	report := s.report(tallies)
	fmt.Print(report)
	s.write(report, tallies)
}

// run puts every case to one arm, in order and one at a time, because a benchmark that reports
// latency cannot also be saturating the provider it is measuring.
func (s *FlowBenchmarkSuite) run(one arm) tally {
	counted := newTally(one)
	for _, subject := range s.set.Cases {
		for attempt := range one.repeats {
			ctx, cancel := context.WithTimeout(s.ctx, benchDeadline)
			decided, err := s.ask(ctx, one, subject, attempt)
			cancel()
			if err != nil {
				// A vendor that will not answer is worth reporting as its own failure rather
				// than as a wrong judgement, which is a claim about the model.
				s.T().Logf("%s could not judge %q: %v", one.name, subject.ID, err)
				counted.refused++
				continue
			}
			counted.add(decided)
		}
	}
	return counted
}

// ask asks once, and again after a wait when the vendor said it was too busy.
func (s *FlowBenchmarkSuite) ask(
	ctx context.Context, one arm, subject flowCase, attempt int,
) (judgement, error) {
	var last error
	for try := range retries {
		decided, err := one.judge(ctx, subject, attempt)
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

// gemmaArm is the incumbent: the production prompt, the production parser, and the fallback the
// controller takes when the answer will not parse.
//
// Nil when the target cannot be reached, which is how an undeployed Gemma or a missing key
// leaves the Jev arms to run on their own rather than taking the whole benchmark with it.
func (s *FlowBenchmarkSuite) gemmaArm() *arm {
	target := os.Getenv(gemmaTargetEnvVar)
	if target == "" {
		target = defaultGemmaTarget
	}

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	router, err := llmrouter.New(llmrouter.Options{
		Config:   config[routing.LLM],
		Registry: llmrouter.DefaultRegistry(),
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	session, err := router.Start(s.ctx, llmrouter.Request{
		CustomerID: "flow-benchmark", Target: target,
	})
	if err != nil {
		s.T().Logf("skipping the Gemma arm, %s is out of reach: %v", target, err)
		return nil
	}
	s.T().Cleanup(func() { _ = session.Close() })

	price := session.Price()
	return &arm{
		name:    "gemma",
		model:   session.Provider() + "/" + session.Model(),
		repeats: gemmaRepeats,
		price:   price.CostMicros,
		judge: func(ctx context.Context, one flowCase, attempt int) (judgement, error) {
			turn := one.turn(one.ID+"-"+strconv.Itoa(attempt), s.set.Contracts)
			askedAt := time.Now()
			stream, err := session.Create(ctx, llm.ResponseParams{
				ID:           turn.ID,
				Instructions: flowInstructions + "\n\nThe agent has been told:\n" + turn.Instructions,
				Input:        []llm.Message{{Role: llm.User, Content: flowQuestion(turn)}},
				// The same budget the controller runs with. A thinking model that spends it
				// before the closing brace is a real failure mode and is scored as one.
				MaxOutputTokens: 512,
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
// to what Gemma is asked, and the same two with the judgements the conversation already
// hard-codes asked for separately and applied in code. Empty when there is no key.
func (s *FlowBenchmarkSuite) jevArms() []arm {
	if os.Getenv("TYPESAFE_API_KEY") == "" {
		s.T().Log("TYPESAFE_API_KEY not set, skipping both Jev arms")
		return nil
	}

	client, err := typesafe.New(typesafe.Options{
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
			judge: func(ctx context.Context, one flowCase, attempt int) (judgement, error) {
				return s.askJev(ctx, client, one, attempt, jevChoices(), false)
			},
		},
		{
			name:    "jev-composed",
			model:   client.Model(),
			repeats: jevRepeats,
			price:   price,
			judge: func(ctx context.Context, one flowCase, attempt int) (judgement, error) {
				return s.askJev(ctx, client, one, attempt, jevComposed(), true)
			},
		},
	}
}

// The two choices, which are the two axes the conversation reads. Their options are described in
// the words the production prompt uses, so what differs between the arms is the model and the
// shape of the answer rather than the policy.
func jevChoices() map[string]typesafe.Question {
	return map[string]typesafe.Question{
		"disposition": typesafe.Choice(
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
		"floor": typesafe.Choice(
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
func jevComposed() map[string]typesafe.Question {
	questions := jevChoices()
	questions["addressed_to_agent"] = typesafe.Noul(
		"Were the words in `heard` meant for the agent described in `agent_was_told`?",
		"Spoken to the agent, whether or not they are finished.",
		"Spoken to somebody else in the room, to a pet or a child, read off a television, "+
			"or otherwise not meant for the agent. `different_voice` being true is evidence "+
			"of this without settling it, because a second person may have leaned in to "+
			"answer for the caller.")
	questions["still_growing"] = typesafe.Noul(
		"Does `heard` end part way through a number, an identifier or a time that the "+
			"speaker is still reading out?",
		"It ends mid-sequence, so more digits or words are still coming.",
		"Whatever number it contains is complete, or it contains none.")
	questions["recorded_menu"] = typesafe.Noul(
		"Is `heard` a recording reading out its options rather than a person talking?",
		"An automated menu, hold message or greeting, which is one thought however long "+
			"the pauses between its parts.",
		"A person speaking, however stilted.")
	questions["non_speech"] = typesafe.Noul(
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
	one flowCase,
	attempt int,
	questions map[string]typesafe.Question,
	composed bool,
) (judgement, error) {
	askedAt := time.Now()
	answered, err := client.Ask(ctx, one.state(s.set.Contracts), questions)
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
	answers map[string]typesafe.Answer,
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

// report writes the tables a human reads.
func (s *FlowBenchmarkSuite) report(tallies []tally) string {
	for i := range tallies {
		tallies[i].settle()
	}

	var out strings.Builder
	fmt.Fprintf(&out, "\n## Results\n\nRun %s over %d cases.\n\n",
		time.Now().UTC().Format(time.RFC3339), len(s.set.Cases))

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
			flipped = share(counted.Flipped, len(s.set.Cases))
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
	for _, state := range flowStates {
		fmt.Fprintf(&out, "| `%s` | `%s` |", state, s.set.expected(state))
		for _, counted := range tallies {
			scored := counted.ByState[state]
			fmt.Fprintf(&out, " %s |", share(scored.Correct, scored.Judged))
		}
		fmt.Fprintln(&out)
	}

	for _, counted := range tallies {
		fmt.Fprintf(&out, "\n### What %s did instead\n\n", counted.Arm)
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
func (s *FlowBenchmarkSuite) write(report string, tallies []tally) {
	out := filepath.Join("testdata", "flowbench-out",
		time.Now().UTC().Format("20060102-150405"))
	s.Require().NoError(os.MkdirAll(out, 0o755))

	s.Require().NoError(os.WriteFile(filepath.Join(out, "report.md"), []byte(report), 0o644))

	sort.Slice(tallies, func(i, j int) bool { return tallies[i].Arm < tallies[j].Arm })
	encoded, err := json.MarshalIndent(tallies, "", "  ")
	s.Require().NoError(err)
	s.Require().NoError(os.WriteFile(filepath.Join(out, "summary.json"), encoded, 0o644))
	s.T().Logf("wrote %s", out)
}
