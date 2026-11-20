package agent

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// theRefusal is what the stub policy says instead of an answer.
const theRefusal = "I can only help with questions about Stream."

// stubGuardrail refuses whatever a test tells it to, so what the agent does with a verdict
// can be tested without a classifier having an opinion about pizza.
type stubGuardrail struct {
	policy guardrail.Policy
	// refuse names the turns to block. Anything else is allowed.
	refuse []string
	err    error
	// held keeps the check from answering until a test closes it, which is what lets a
	// reply be caught waiting on a verdict.
	held <-chan struct{}

	mu    sync.Mutex
	asked []string
	// before is the conversation each check was given, in the order they were asked.
	before []string
}

func (g *stubGuardrail) Policy() guardrail.Policy { return g.policy }

func (g *stubGuardrail) Close() error { return nil }

func (g *stubGuardrail) Check(ctx context.Context, _, text, conversation string) (guardrail.Verdict, error) {
	g.mu.Lock()
	g.asked = append(g.asked, text)
	g.before = append(g.before, conversation)
	held := g.held
	g.mu.Unlock()

	if held != nil {
		select {
		case <-held:
		case <-ctx.Done():
			return guardrail.Verdict{}, ctx.Err()
		}
	}
	if g.err != nil {
		return guardrail.Verdict{}, g.err
	}

	for _, blocked := range g.refuse {
		if strings.Contains(text, blocked) {
			return guardrail.Verdict{
				Reason:      "outside what the policy permits",
				Probability: 0.93,
				Model:       "stub/judge",
			}, nil
		}
	}
	return guardrail.Verdict{Allowed: true, Probability: 0.07, Model: "stub/judge"}, nil
}

// conversations is the conversation each check was given.
func (g *stubGuardrail) conversations() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.before...)
}

func (g *stubGuardrail) screened() []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]string(nil), g.asked...)
}

// screens gives the agent a policy that refuses the turns named, checked in the given
// mode.
func (s *AgentSuite) screens(mode guardrail.Mode, refuse ...string) {
	s.guards = &stubGuardrail{
		policy: guardrail.Policy{
			Kind:      guardrail.KindDecisionModel,
			Mode:      mode,
			Threshold: guardrail.DefaultThreshold,
			Refusal:   theRefusal,
			Text:      "Only answer questions about Stream's SDKs.",
		},
		refuse: refuse,
	}
}

// screening is the guardrail the agent joins with. A typed nil would make the interface
// non-nil, which is not the same as having no policy at all.
func (s *AgentSuite) screening() guardrail.Guardrail {
	if s.guards == nil {
		return nil
	}
	return s.guards
}

// answered is the text of every reply the agent finished, refusals included.
func (s *AgentSuite) answered() []string {
	var said []string
	for _, event := range s.reported() {
		if responded, ok := event.(Responded); ok {
			said = append(said, responded.Text)
		}
	}
	return said
}

// written is every word that left the agent in writing, whether as a streaming delta or
// as a finished reply. Nothing the model wrote may appear here for a refused turn.
func (s *AgentSuite) written() string {
	var out strings.Builder
	for _, event := range s.reported() {
		switch typed := event.(type) {
		case ResponseDelta:
			out.WriteString(typed.Text)
		case Responded:
			out.WriteString(typed.Text)
		}
	}
	return out.String()
}

func (s *AgentSuite) TestATurnThePolicyRefusesIsAnsweredWithTheRefusal() {
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()
	s.model.reply = []string{"First you make the dough, ", "then you add the sauce."}

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the turn was never answered")

	s.Equal([]string{theRefusal}, s.answered())
	s.NotContains(s.written(), "dough",
		"the model's reply reached the caller despite the refusal")
	s.NotContains(s.written(), "sauce")
}

func (s *AgentSuite) TestARefusedTurnIsReportedAsATurnAndAsARefusal() {
	// A client that has never heard of a guardrail should show a refused turn as an
	// ordinary exchange, and one that has should be able to say why. Both at once.
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Blocked](s.reported()) == 1 && countOf[Responded](s.reported()) == 1 },
		"nothing was reported as refused and answered")

	refused, _ := firstOf[Blocked](s.reported())
	s.Equal("outside what the policy permits", refused.Reason)
	s.InDelta(0.93, refused.Probability, 0.001)
	s.NotEmpty(refused.TurnID)

	responding, ok := firstOf[Responding](s.reported())
	s.Require().True(ok, "a refused turn never reported that the agent was answering")
	s.Equal(refused.TurnID, responding.TurnID)

	responded, ok := firstOf[Responded](s.reported())
	s.Require().True(ok, "a refused turn never reported an answer")
	s.Equal(refused.TurnID, responded.TurnID)
	s.Equal(theRefusal, responded.Text)
}

func (s *AgentSuite) TestATurnThePolicyPermitsIsAnsweredNormally() {
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()
	s.model.reply = []string{"Render it with the MessageList component."}

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I render a message list"))
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the turn was never answered")

	s.Equal([]string{"Render it with the MessageList component."}, s.answered())
	s.Zero(countOf[Blocked](s.reported()))
	s.Equal([]string{"how do I render a message list"}, s.guards.screened())
}

func (s *AgentSuite) TestScreeningInParallelSpendsTheTokensOfAReplyNobodyReads() {
	// This is the trade parallel mode makes, and it is worth pinning rather than
	// discovering on a bill: a refused turn pays for a reply that is thrown away, and in
	// exchange a permitted turn waits on nothing.
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the turn was never answered")

	s.Len(s.model.requests(), 1, "parallel mode did not ask the model")
	s.Equal([]string{theRefusal}, s.answered())
}

func (s *AgentSuite) TestScreeningFirstSpendsNothingOnATurnThatIsRefused() {
	// The other half of the trade: no tokens at all on a refused turn, and in exchange
	// every caller waits for the verdict before the model is even asked.
	s.screens(guardrail.ModeBlocking, "pizza")
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the turn was never answered")

	s.Empty(s.model.requests(), "blocking mode asked the model anyway")
	s.Equal([]string{theRefusal}, s.answered())
}

func (s *AgentSuite) TestAReplyHeldForAVerdictIsReleasedRatherThanDropped() {
	// Holding the reply is what makes the refusal safe. Forgetting to let it go again
	// would make every turn silent, which is the failure this guards against.
	release := make(chan struct{})
	s.screens(guardrail.ModeParallel, "pizza")
	s.guards.held = release
	s.joinText()
	s.model.reply = []string{"Use the MessageList component."}

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I render a message list"))
	s.eventually(func() bool { return len(s.guards.screened()) == 1 }, "the check never started")
	s.Zero(countOf[Responded](s.reported()), "the reply was delivered before the verdict was in")

	close(release)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the permitted reply was never released")
	s.Equal([]string{"Use the MessageList component."}, s.answered())
}

func (s *AgentSuite) TestARefusalCanBeAskedAbout() {
	// The refusal goes into the history like anything else the agent said, so "why not?"
	// has an antecedent and the transcript reads as what happened.
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the turn was never answered")

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "why not?"))
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the second turn never went out")

	var spoken []string
	for _, message := range harness.WithoutNotes(s.model.requests()[1].Input) {
		spoken = append(spoken, message.Content)
	}
	s.Contains(spoken, theRefusal)
	s.Contains(spoken, "how do I make a pizza")
}

func (s *AgentSuite) TestACheckThatCouldNotBeMadeLetsTheTurnThrough() {
	// Written down so it cannot be changed by accident: a classifier outage, or a
	// customer's own webhook being down, leaves the agent answering rather than mute.
	s.screens(guardrail.ModeParallel, "pizza")
	s.guards.err = errors.New("rate limited")
	s.joinText()
	s.model.reply = []string{"First you make the dough."}

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the turn was never answered")

	s.Equal([]string{"First you make the dough."}, s.answered())
	s.Zero(countOf[Blocked](s.reported()))
}

func (s *AgentSuite) TestOnlyWhatTheCallerSaidIsScreened() {
	// A turn carrying no new words from the caller has nothing to judge, and paying for a
	// judgement on words already judged is a bill with nothing behind it.
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()

	s.Require().NoError(s.agent.Say(s.ctx, "Hello, how can I help?"))
	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I render a message list"))
	s.eventually(func() bool { return len(s.model.requests()) == 1 },
		"the turn was never answered")
	// The check runs beside the model and the reply waits for its verdict, so the turn's
	// reply, not its request, is what says its words were screened.
	turn := s.model.requests()[0].ID
	s.eventually(func() bool {
		return slices.ContainsFunc(s.reported(), func(event Event) bool {
			responded, ok := event.(Responded)
			return ok && responded.TurnID == turn
		})
	}, "the turn's reply never came")

	s.Equal([]string{"how do I render a message list"}, s.guards.screened())
}

func (s *AgentSuite) TestAnAgentWithNoPolicyIsScreenedByNothing() {
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the turn was never answered")

	s.Equal([]string{"Hello there. How are you?"}, s.answered())
	s.Zero(countOf[Blocked](s.reported()))
}

func (s *AgentSuite) TestARefusalOutLoudIsSpokenInsteadOfTheReply() {
	// The voice path, where a leak would be audible: what reaches the synthesiser is the
	// refusal and only the refusal.
	s.screens(guardrail.ModeParallel, "pizza")
	s.join(true)
	s.model.reply = []string{"First you make the dough, ", "then you add the sauce."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "how do I make a pizza")
	s.eventually(func() bool { return countOf[Blocked](s.reported()) == 1 },
		"the turn was never refused")
	s.eventually(func() bool { return strings.Contains(said(s.voice.spoken()), theRefusal) },
		"the refusal was never spoken")

	s.NotContains(said(s.voice.spoken()), "dough",
		"the model's reply was spoken despite the refusal")
	s.NotContains(said(s.voice.spoken()), "sauce")
}

// A written aside is screened on what the model is asked, its words or imagesAlone for
// photos sent alone, not on the note history keeps of its photos.
func (s *AgentSuite) TestAWrittenAsideWithPhotosIsScreenedOnWhatIsAsked() {
	s.sees = true
	s.screens(guardrail.ModeBlocking)
	s.joinText()
	photo := llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8, 0xff}}

	_, err := s.agent.Ask(s.ctx, "is this fan broken?", photo)
	s.Require().NoError(err)
	_, err = s.agent.Ask(s.ctx, "", photo)
	s.Require().NoError(err)

	s.Equal([]string{"is this fan broken?", imagesAlone}, s.guards.screened())
}

func (s *AgentSuite) TestAWrittenAsideIsScreenedToo() {
	// Ask is the Stream chat path: another way in to the same agent, and a guardrail that
	// only covered the call would be a guardrail with a door beside it.
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()
	s.model.reply = []string{"First you make the dough."}

	answer, err := s.agent.Ask(s.ctx, "how do I make a pizza")
	s.Require().NoError(err)

	s.Equal(theRefusal, answer)
	s.eventually(func() bool { return countOf[Blocked](s.reported()) == 1 },
		"the aside was never reported as refused")
}

func (s *AgentSuite) TestATurnThePolicyPermitsReportsItsCheck() {
	// An allowed turn's check is what a threshold is tuned by: without it only refusals
	// say what the model made of a turn.
	release := make(chan struct{})
	s.screens(guardrail.ModeParallel, "pizza")
	s.guards.held = release
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I render a message list"))
	s.eventually(func() bool { return len(s.guards.screened()) == 1 }, "the check never started")
	close(release)
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")

	turn, _ := firstOf[Turn](s.reported())
	check := turn.Guardrail
	s.Require().NotNil(check, "the turn does not say it was screened")
	s.Equal("decision_model", check.Kind)
	s.Equal("parallel", check.Mode)
	s.Equal("stub/judge", check.Model)
	s.True(check.Allowed)
	s.False(check.Failed)
	s.Require().NotNil(check.Probability)
	s.InDelta(0.07, *check.Probability, 0.001)
	s.Require().NotNil(check.Threshold)
	s.InDelta(guardrail.DefaultThreshold, *check.Threshold, 0.001)
	s.Positive(check.LatencyMs)
	responded, _ := firstOf[Responded](s.reported())
	s.Equal(responded.TurnID, turn.TurnID)
}

func (s *AgentSuite) TestATurnThePolicyRefusesReportsItsCheck() {
	s.screens(guardrail.ModeBlocking, "pizza")
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")

	turn, _ := firstOf[Turn](s.reported())
	s.Require().NotNil(turn.Guardrail, "the turn does not say it was screened")
	s.False(turn.Guardrail.Allowed)
	s.Equal("blocking", turn.Guardrail.Mode)
	s.Require().NotNil(turn.Guardrail.Probability)
	s.InDelta(0.93, *turn.Guardrail.Probability, 0.001)
	s.Zero(turn.Guardrail.HeldMs, "no reply waits on a check that runs before the model is asked")
}

func (s *AgentSuite) TestACheckThatCouldNotBeMadeIsReportedAsFailedAndNotAsHarmless() {
	// A failed check lets the turn through. Reported as allowed at a probability of zero,
	// an outage would read as a model sure of every turn.
	s.screens(guardrail.ModeParallel, "pizza")
	s.guards.err = errors.New("rate limited")
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")

	turn, _ := firstOf[Turn](s.reported())
	s.Require().NotNil(turn.Guardrail, "the turn does not say its check failed")
	s.True(turn.Guardrail.Failed)
	s.True(turn.Guardrail.Allowed)
	s.Nil(turn.Guardrail.Probability)
	s.Empty(turn.Guardrail.Model)
	s.Require().NotNil(turn.Guardrail.Threshold)
}

func (s *AgentSuite) TestATurnTheModelFailedStillReportsItsCheck() {
	// The check answered beside a model that never did: the turn ends failed, and what the
	// guardrail made of it is still on it.
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()
	s.model.mu.Lock()
	s.model.refuses = errors.New("invalid api key")
	s.model.mu.Unlock()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I render a message list"))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the failed turn was never reported")

	turn, _ := firstOf[Turn](s.reported())
	s.Require().NotNil(turn.Guardrail, "the failed turn lost its check")
	s.True(turn.Guardrail.Allowed)
	s.Require().NotNil(turn.Guardrail.Probability)
	s.InDelta(0.07, *turn.Guardrail.Probability, 0.001)
	s.Zero(turn.Guardrail.HeldMs, "no reply was held: the model wrote none")
}

func (s *AgentSuite) TestATurnNothingScreenedReportsNoCheck() {
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I make a pizza"))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")

	turn, _ := firstOf[Turn](s.reported())
	s.Nil(turn.Guardrail)
}

func (s *AgentSuite) TestAFollowUpIsScreenedWithWhatCameBeforeIt() {
	// "and for my kid?" cannot be judged alone: the check is given the exchange before the
	// turn, and not the turn again as part of it.
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()
	s.model.reply = []string{"Use the MessageList component."}

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I render a message list"))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the first turn was never reported")
	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "and in React Native?"))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 2 }, "the follow-up was never reported")

	s.Equal([]string{
		"",
		"user: how do I render a message list\nassistant: Use the MessageList component.",
	}, s.guards.conversations())
}

func (s *AgentSuite) TestAWrittenAsideIsScreenedWithWhatCameBeforeIt() {
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()

	refusal, err := s.agent.Ask(s.ctx, "how do I make a pizza")
	s.Require().NoError(err)
	s.Equal(theRefusal, refusal)
	_, err = s.agent.Ask(s.ctx, "how do I render a message list")
	s.Require().NoError(err)

	s.Equal([]string{"", "user: how do I make a pizza\nassistant: " + theRefusal}, s.guards.conversations())
}

func (s *AgentSuite) TestALateToolResultIsNotPartOfTheConversationTheNextTurnIsScreenedWith() {
	// A call held over an interruption has its result added on the caller's side once it is
	// in. Nobody said it, so the next turn's check is not shown it as the caller's words.
	s.screens(guardrail.ModeParallel, "pizza")
	s.ownsTools("")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{Waits: true} }
	s.join(true)
	runner := &holdingTool{began: make(chan struct{}), release: make(chan struct{})}
	release := sync.OnceFunc(func() { close(runner.release) })
	defer release()
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.model.then = []string{"Fifteen."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "where is my order")
	select {
	case <-runner.began:
	case <-time.After(settleFor):
		s.FailNow("tool did not start")
	}
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.replyWentOut()
	s.mutters(participant, "stop cancel that lookup")
	s.eventually(func() bool {
		return slices.ContainsFunc(s.history(), func(m llm.Message) bool { return strings.HasPrefix(m.Content, stillRunning) })
	}, "the interrupted call was not answered")
	s.flow.then = nil
	s.says(participant, "what is seven plus eight")
	s.eventually(func() bool { return s.spokenText("Fifteen") }, "the next turn waited on the call")

	release()
	s.eventually(func() bool {
		return slices.ContainsFunc(s.history(), func(m llm.Message) bool { return strings.Contains(m.Content, "ships tomorrow") })
	}, "the late result never reached the conversation")
	s.eventually(func() bool { return countOf[Responded](s.reported()) >= 3 }, "the late result was never answered")
	s.says(participant, "can I change the address")
	s.eventually(func() bool { return slices.Contains(s.guards.screened(), "can I change the address") },
		"the turn after the late result was never screened")

	asked := s.guards.screened()
	before := s.guards.conversations()[slices.Index(asked, "can I change the address")]
	s.NotContains(before, "ships tomorrow", "the tool's output was screened as the caller's words")
	s.NotContains(before, "lookup_order")
	s.Contains(before, "user: what is seven plus eight", "what the caller said is still the conversation")
}

func (s *AgentSuite) TestTheNoteThatSilenceDeclinedACallIsNotConversation() {
	s.agentAsksAndWaits(100 * time.Millisecond)
	s.askFor()
	s.eventually(s.declinedBySilence, "silence never declined the call")

	history := append(s.history(), llm.Message{Role: llm.User, Content: "what now?"})
	before := conversationBefore(history)
	s.NotContains(before, "did not say yes")
	s.Contains(before, "user: book order 12")
}

func (s *AgentSuite) TestASummaryOfTheCallSoFarIsNotConversation() {
	// The summary is the router's words about the conversation, not the conversation, and it
	// is written from what tools returned too.
	prefix := []llm.Message{{Role: llm.User, Content: "first question"}, {Role: llm.Assistant, Content: "first answer"}}
	recent := []llm.Message{{Role: llm.User, Content: "new question"}, {Role: llm.Assistant, Content: "new answer"}}
	emitter := NewEmitter(eventBuffer)
	defer emitter.Close()
	agent := &Agent{emitter: emitter, history: append(append([]llm.Message(nil), prefix...), recent...)}
	agent.applyCompaction(harness.Compacted{Prefix: prefix, Summary: "The caller's card ends in 4242."})

	history := append(agent.History(), llm.Message{Role: llm.User, Content: "and then?"})
	s.Equal("user: new question\nassistant: new answer", conversationBefore(history))
}

func (s *AgentSuite) TestNotesAreNotConversation() {
	history := []llm.Message{
		{Role: llm.User, Content: "hi"},
		{Role: llm.Assistant, Content: "hello"},
		harness.DocumentMessage("menu.pdf", 1, "the whole menu"),
		{Role: llm.User, Content: "and now?"},
	}

	s.Equal("user: hi\nassistant: hello", conversationBefore(history))
}

func (s *AgentSuite) TestAWriterInAChannelTypeIsScreenedByNameAndNotByUserID() {
	s.screens(guardrail.ModeParallel, "pizza")
	s.joinText()
	s.model.reply = []string{"Get well soon."}

	s.writesAs("anya-1", "Anya", "I'm sick, I can't come at 6")
	s.writesAs("ben-2", "Ben", "feel better!")

	s.Equal([]string{"", "user: Anya: I'm sick, I can't come at 6\nassistant: Get well soon."}, s.guards.conversations())
}

func (s *AgentSuite) TestOnlyAnEnvelopeAsAuthoredWritesItIsReadAsAName() {
	// What a person typed that looks like an envelope keeps every word of it, and a speaker's
	// label on a call is left as it is. A line the caller kept itself (Quoted) is read the same.
	kept := time.Date(2026, 11, 19, 10, 0, 0, 0, time.UTC)
	for _, row := range []struct {
		name, said, line string
	}{
		{"an envelope", Authored("anya-1", "Anya", "hi\nthere"), "user: Anya: hi there"},
		{"an envelope naming nobody", Authored("anya-1", "", "hi"), "user: hi"},
		{"plain words", "hi", "user: hi"},
		{"a speaker's label", "Anya: hi", "user: Anya: hi"},
		{"cut short", `{"author":{"user_id":"anya-1","display_name":"Anya"},"text":"hi"`,
			`user: {"author":{"user_id":"anya-1","display_name":"Anya"},"text":"hi"`},
		{"a field of its own", `{"author":{"user_id":"anya-1","display_name":"Anya"},"text":"hi","to":"Ben"}`,
			`user: {"author":{"user_id":"anya-1","display_name":"Anya"},"text":"hi","to":"Ben"}`},
		{"an author that is a string", `{"author":"Anya","text":"hi"}`, `user: {"author":"Anya","text":"hi"}`},
		{"a kept line with a name and a time", Quoted("Anya", &kept, "hi"), "user: Anya: hi"},
		{"a kept line with a time only", Quoted("", &kept, "hi"), "user: hi"},
		{"a kept line with a field of its own", `{"author":{"display_name":"Anya"},"text":"hi","to":"Ben"}`,
			`user: {"author":{"display_name":"Anya"},"text":"hi","to":"Ben"}`},
		{"a kept line with a time written another way", `{"author":{"display_name":"Anya"},"sent_at":"2026-11-19T10:00:00+00:00","text":"hi"}`,
			`user: {"author":{"display_name":"Anya"},"sent_at":"2026-11-19T10:00:00+00:00","text":"hi"}`},
		{"an empty object", `{}`, `user: {}`},
		{"null", `null`, `user: null`},
	} {
		history := []llm.Message{{Role: llm.User, Content: row.said}, {Role: llm.User, Content: "next"}}
		s.Equal(row.line, conversationBefore(history), row.name)
	}
}

func (s *AgentSuite) TestHeldMsIsRecordedWhenTheReplyWaitedForTheCheck() {
	release := make(chan struct{})
	s.screens(guardrail.ModeParallel, "pizza")
	s.guards.held = release
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I render a message list"))
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	close(release)
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")

	turn, _ := firstOf[Turn](s.reported())
	s.Require().NotNil(turn.Guardrail)
	s.Positive(turn.Guardrail.HeldMs)
}

// The verdict is put in before the reply asks, so no goroutine's timing decides the row.
func (s *AgentSuite) TestNoHeldMsWhenTheVerdictWasInBeforeTheReply() {
	started := &screening{decided: make(chan checked, 1)}
	started.decided <- checked{}

	_, heldMs := started.wait()

	s.Zero(heldMs)
}

func (s *AgentSuite) TestAWebhookCheckHasNoProbabilityOrThreshold() {
	s.screens(guardrail.ModeParallel, "pizza")
	s.guards.policy.Kind = guardrail.KindWebhook
	s.joinText()

	s.Require().NoError(s.agent.SimpleResponse(s.ctx, "how do I render a message list"))
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")

	turn, _ := firstOf[Turn](s.reported())
	s.Require().NotNil(turn.Guardrail)
	s.Equal("webhook", turn.Guardrail.Kind)
	s.Nil(turn.Guardrail.Probability)
	s.Nil(turn.Guardrail.Threshold)
}
