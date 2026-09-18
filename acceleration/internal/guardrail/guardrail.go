// Package guardrail screens what a caller asks before the agent answers it.
//
// An agent declares one in a guardrail.md beside its instructions: frontmatter says how to
// check and the body is the policy itself, in prose, so what the agent will and will not
// answer stays something a human reads and edits rather than a rule expressed in code.
//
// There are three ways to check, and they are different trades rather than three grades of
// the same thing. A classifier is a calibrated probability in a few hundred milliseconds. A
// webhook is the customer's own server deciding, which is the only option when the answer
// depends on who is asking. An LLM judge reads the policy as prose, which is the most
// forgiving of a policy written loosely and the slowest and dearest of the three.
//
// Screening lives here, in the backend, rather than in the agent process, because a
// guardrail an agent can skip is decoration.
package guardrail

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmclassifierrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// Kind is how a turn is checked.
type Kind string

const (
	// KindClassifier asks the llm_classifier router for the probability that a turn
	// violates the policy.
	KindClassifier Kind = "llm_classifier"
	// KindWebhook asks the customer's own server.
	KindWebhook Kind = "webhook"
	// KindLLM asks a language model to read the policy and judge the turn.
	KindLLM Kind = "llm"
)

// Mode is when the check runs relative to the model.
type Mode string

const (
	// ModeParallel runs the check beside the model and holds the reply until the verdict
	// arrives. It costs the tokens of a reply that may never be delivered, and it is the
	// default because it is the only one that does not add to what a caller waits.
	ModeParallel Mode = "parallel"
	// ModeBlocking runs the check first and asks the model only if it passes. It spends no
	// tokens on a turn that is refused, and every caller pays the check's latency for it.
	ModeBlocking Mode = "blocking"
)

// DefaultThreshold is where a classifier or a judge starts blocking. Slightly above the
// middle: a guardrail that fires on a coin flip refuses things it was never meant to.
const DefaultThreshold = 0.6

// DefaultRefusal is what an agent says when the policy does not say.
const DefaultRefusal = "I can't help with that."

// checkTimeout bounds one check. It is short on purpose: in parallel mode this is how long
// a finished reply may be held, and a guardrail that has not answered by now is one whose
// verdict has stopped being worth waiting for.
const checkTimeout = 2 * time.Second

// Policy is a parsed guardrail.md.
type Policy struct {
	Kind Kind
	Mode Mode
	// Threshold is the probability of a violation at or above which a turn is refused.
	// It means nothing to a webhook, which answers yes or no itself.
	Threshold float64
	// Target routes the check: a classifier or LLM "provider/model" or capability
	// shortcut. Empty takes the modality's default route.
	Target string
	// URL is the customer's endpoint, for a webhook.
	URL string
	// Refusal is what the agent says instead of the reply it did not give.
	Refusal string
	// Text is the policy itself: everything after the frontmatter.
	Text string
}

// Verdict is what a check decided.
type Verdict struct {
	Allowed bool
	// Reason is why, for the log and the event rather than for the caller. What the caller
	// hears is the policy's refusal, because a reason quoting the policy back is a map of
	// how to get around it.
	Reason string
	// Probability is what a classifier or judge put on the turn violating the policy. Zero
	// from a webhook, which does not deal in probabilities.
	Probability float64
	// Refusal replaces the policy's own line, for a check that has something more useful
	// to say than the policy could. Only a webhook fills it: a customer's server knows
	// things about who is asking that a file written in advance does not.
	Refusal string
}

// allowed is the verdict for a turn nothing objected to.
func allowed(probability float64) Verdict {
	return Verdict{Allowed: true, Probability: probability}
}

// Guardrail screens one agent's turns.
type Guardrail interface {
	// Check reports whether a turn may be answered. An error means the check could not be
	// made, which is not the same as a refusal and is not treated as one.
	Check(ctx context.Context, turnID, text string) (Verdict, error)
	// Policy is what this was built from, which is what tells the agent when to run the
	// check and what to say when it fails.
	Policy() Policy
	Close() error
}

// Deps is what the implementations need from the deployment.
type Deps struct {
	// Owner is who a routed check is billed to.
	Owner routing.Owner
	// Classifier routes a KindClassifier check.
	Classifier *llmclassifierrouter.Router
	// LLM routes a KindLLM check.
	LLM *llmrouter.Router
	// Secret signs a webhook, so the customer's server can tell our request from anyone
	// who found the URL.
	Secret string
	// HTTPClient calls a webhook. Nil builds one with the check timeout.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// New builds the guardrail a policy asks for.
//
// A policy whose backend this deployment does not route is refused here, as the session is
// created, rather than at the first turn. The alternative is an agent that starts, fails
// every check, and answers everything - which is an unguarded agent that looks guarded.
func New(ctx context.Context, policy Policy, deps Deps) (Guardrail, error) {
	if deps.Logger == nil {
		deps.Logger = slog.Default()
	}

	switch policy.Kind {
	case KindClassifier:
		return newClassifier(ctx, policy, deps)
	case KindWebhook:
		return newWebhook(policy, deps)
	case KindLLM:
		return newJudge(ctx, policy, deps)
	default:
		return nil, fmt.Errorf("guardrail: %q is not a way of checking a turn", policy.Kind)
	}
}

// Parse reads a guardrail.md: frontmatter saying how to check, then the policy itself.
//
// The recognised keys are type, mode, threshold, target, url and refusal. A file with no
// frontmatter at all is a classifier guardrail with default settings, since prose on its
// own is the shortest thing worth writing here.
func Parse(content string) (Policy, error) {
	policy := Policy{
		Kind:      KindClassifier,
		Mode:      ModeParallel,
		Threshold: DefaultThreshold,
		Refusal:   DefaultRefusal,
	}

	frontmatter, body, found := cutFrontmatter(content)
	policy.Text = strings.TrimSpace(body)

	if found {
		for line := range strings.SplitSeq(frontmatter, "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			key, value, ok := strings.Cut(line, ":")
			if !ok {
				return policy, fmt.Errorf("guardrail: %q is not a key and a value", line)
			}
			key = strings.TrimSpace(key)
			value = strings.Trim(strings.TrimSpace(value), `"'`)

			switch key {
			case "type":
				policy.Kind = Kind(value)
			case "mode":
				policy.Mode = Mode(value)
			case "threshold":
				threshold, err := strconv.ParseFloat(value, 64)
				if err != nil {
					return policy, fmt.Errorf("guardrail: %q is not a threshold", value)
				}
				policy.Threshold = threshold
			case "target":
				policy.Target = value
			case "url":
				policy.URL = value
			case "refusal":
				policy.Refusal = value
			default:
				// An unrecognised key is refused rather than ignored: a misspelt
				// threshold that is silently the default is a guardrail whose settings
				// are not what the file says they are.
				return policy, fmt.Errorf("guardrail: %q is not something a guardrail takes", key)
			}
		}
	}

	return policy, policy.Validate()
}

// Validate reports what would make a policy mean something other than what it says.
func (p Policy) Validate() error {
	switch p.Kind {
	case KindClassifier, KindLLM:
		if strings.TrimSpace(p.Text) == "" {
			return errors.New("guardrail: there is no policy to judge a turn against")
		}
		if p.URL != "" {
			return fmt.Errorf("guardrail: a %s guardrail calls nothing, so its url would be ignored", p.Kind)
		}
		if p.Threshold <= 0 || p.Threshold > 1 {
			return fmt.Errorf("guardrail: a threshold of %v is not a probability between 0 and 1", p.Threshold)
		}
	case KindWebhook:
		if p.URL == "" {
			return errors.New("guardrail: a webhook guardrail needs a url to ask")
		}
		if !strings.HasPrefix(p.URL, "https://") && !strings.HasPrefix(p.URL, "http://") {
			return fmt.Errorf("guardrail: %q is not a url a webhook can be posted to", p.URL)
		}
	default:
		return fmt.Errorf("guardrail: %q is not a way of checking a turn", p.Kind)
	}

	switch p.Mode {
	case ModeParallel, ModeBlocking:
	default:
		return fmt.Errorf("guardrail: %q is not when a check can run", p.Mode)
	}

	if strings.TrimSpace(p.Refusal) == "" {
		return errors.New("guardrail: there is nothing for the agent to say when a turn is refused")
	}
	return nil
}

// cutFrontmatter separates a leading --- block from the body, as an agent folder's skill
// files are read.
func cutFrontmatter(content string) (frontmatter, body string, found bool) {
	trimmed := strings.TrimLeft(content, "\ufeff \t\r\n")
	if !strings.HasPrefix(trimmed, "---") {
		return "", content, false
	}

	rest := strings.TrimPrefix(trimmed, "---")
	rest = strings.TrimLeft(rest, "\r\n")
	frontmatter, body, found = strings.Cut(rest, "\n---")
	if !found {
		return "", content, false
	}
	return frontmatter, strings.TrimLeft(body, "-\r\n"), true
}
