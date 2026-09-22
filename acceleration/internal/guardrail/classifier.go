package guardrail

import (
	"context"
	"errors"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcmrouter"
)

// violates is the id the one question is asked under.
const violates = "violates"

// classifier screens a turn by asking the lcm router for the probability that it violates
// the policy.
type classifier struct {
	policy  Policy
	session *lcmrouter.Session
	logger  *slog.Logger
}

// newClassifier opens the session the checks will run on.
//
// It is opened here rather than on the first turn so that a deployment with nothing to
// route the check to refuses the agent outright. A guardrail that starts and then cannot
// reach its classifier allows every turn, which is worse than the call not starting.
func newClassifier(ctx context.Context, policy Policy, deps Deps) (Guardrail, error) {
	if deps.Classifier == nil {
		return nil, errors.New(
			"guardrail: this deployment routes no lcm, so it cannot judge a turn against a policy")
	}

	session, err := deps.Classifier.Start(ctx, lcmrouter.Request{
		CustomerID: deps.Owner.CustomerID,
		AgentID:    deps.Owner.AgentID,
		CallID:     deps.Owner.CallID,
		Tags:       deps.Owner.Tags,
		Target:     policy.Target,
	})
	if err != nil {
		return nil, err
	}

	return &classifier{policy: policy, session: session, logger: deps.Logger}, nil
}

func (c *classifier) Policy() Policy { return c.policy }

func (c *classifier) Close() error { return c.session.Close() }

// Check asks whether the turn is something the policy permits.
//
// The policy travels as named state rather than pasted into the question, which is what
// lets the question point at `policy` and `message` as two things it is comparing. One noul
// rather than a battery of them: one policy file is one judgement, and what a battery would
// buy - named hazards scored separately - is a thing the file cannot yet ask for.
func (c *classifier) Check(ctx context.Context, turnID, text string) (Verdict, error) {
	ctx, cancel := context.WithTimeout(ctx, checkTimeout)
	defer cancel()

	answered, err := c.session.Classify(ctx, lcm.Request{
		State: map[string]string{"policy": c.policy.Text, "message": text},
		Questions: map[string]lcm.Question{
			violates: lcm.Noul(
				"Is `message` a request for something the assistant is not permitted to "+
					"answer under `policy`?",
				"It asks for something outside what the policy permits.",
				"It is a request the policy permits.",
			),
		},
	})
	if err != nil {
		return Verdict{}, err
	}

	probability := answered.Answers[violates].Yes
	if probability < c.policy.Threshold {
		return allowed(probability), nil
	}

	if c.logger.Enabled(ctx, slog.LevelDebug) {
		c.logger.Debug("a turn was refused by the policy",
			"turn", turnID, "probability", probability, "threshold", c.policy.Threshold,
			"model", answered.Model)
	}
	return Verdict{
		Reason:      "the classifier reads this as outside what the policy permits",
		Probability: probability,
	}, nil
}
