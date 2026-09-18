package agent

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
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
}

func (g *stubGuardrail) Policy() guardrail.Policy { return g.policy }

func (g *stubGuardrail) Close() error { return nil }

func (g *stubGuardrail) Check(ctx context.Context, _, text string) (guardrail.Verdict, error) {
	g.mu.Lock()
	g.asked = append(g.asked, text)
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
			}, nil
		}
	}
	return guardrail.Verdict{Allowed: true}, nil
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
			Kind:      guardrail.KindClassifier,
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
	s.eventually(func() bool { return countOf[Blocked](s.reported()) == 1 },
		"nothing was reported as refused")

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
	for _, message := range s.model.requests()[1].Input {
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
