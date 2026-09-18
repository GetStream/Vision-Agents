package guardrail

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type PolicySuite struct {
	suite.Suite
}

func TestPolicySuite(t *testing.T) {
	suite.Run(t, new(PolicySuite))
}

func (s *PolicySuite) TestFrontmatterSaysHowToCheckAndTheBodyIsThePolicy() {
	policy, err := Parse(`---
type: llm_classifier
mode: blocking
threshold: 0.8
target: typesafe/jev-latest
refusal: I can only help with questions about Stream.
---
Only answer questions about Stream's SDKs.

Nothing else.
`)
	s.Require().NoError(err)

	s.Equal(KindClassifier, policy.Kind)
	s.Equal(ModeBlocking, policy.Mode)
	s.InDelta(0.8, policy.Threshold, 0.001)
	s.Equal("typesafe/jev-latest", policy.Target)
	s.Equal("I can only help with questions about Stream.", policy.Refusal)
	s.Equal("Only answer questions about Stream's SDKs.\n\nNothing else.", policy.Text)
}

func (s *PolicySuite) TestProseOnItsOwnIsAGuardrail() {
	// The shortest thing worth writing here is what the agent may be asked. Everything
	// else has a default, and the defaults are the ones that cost a caller the least.
	policy, err := Parse("Only answer questions about Stream.")
	s.Require().NoError(err)

	s.Equal(KindClassifier, policy.Kind)
	s.Equal(ModeParallel, policy.Mode,
		"a check that adds no latency is what a voice agent needs by default")
	s.InDelta(DefaultThreshold, policy.Threshold, 0.001)
	s.Equal(DefaultRefusal, policy.Refusal)
	s.Equal("Only answer questions about Stream.", policy.Text)
}

func (s *PolicySuite) TestAPolicyWithNothingToJudgeAgainstIsRefused() {
	_, err := Parse("---\ntype: llm_classifier\n---\n")

	s.ErrorContains(err, "no policy")
}

func (s *PolicySuite) TestAWayOfCheckingNobodyHasIsRefused() {
	// Refused rather than defaulted: an agent whose guardrail says "webhoook" and is
	// screened by a classifier is screened by something its own file does not name.
	_, err := Parse("---\ntype: vibes\n---\nOnly Stream questions.")

	s.ErrorContains(err, "vibes")
}

func (s *PolicySuite) TestAMisspeltSettingIsRefusedRatherThanIgnored() {
	// A threshold spelt "treshold" that is silently 0.6 is a guardrail whose settings are
	// not the ones the file says, and nothing about the running agent would show it.
	_, err := Parse("---\ntreshold: 0.9\n---\nOnly Stream questions.")

	s.ErrorContains(err, "treshold")
}

func (s *PolicySuite) TestAThresholdThatIsNotAProbabilityIsRefused() {
	_, err := Parse("---\nthreshold: 60\n---\nOnly Stream questions.")
	s.ErrorContains(err, "probability")

	_, err = Parse("---\nthreshold: high\n---\nOnly Stream questions.")
	s.ErrorContains(err, "not a threshold")
}

func (s *PolicySuite) TestAWebhookNeedsSomewhereToAsk() {
	_, err := Parse("---\ntype: webhook\n---\nAsk my server.")

	s.ErrorContains(err, "url")
}

func (s *PolicySuite) TestAUrlOnAPolicyThatCallsNothingIsRefused() {
	// A setting that is present and does nothing is worse than one that is absent: the
	// file says a server decides, and no server is ever asked.
	_, err := Parse("---\ntype: llm_classifier\nurl: https://example.test/check\n---\nOnly Stream questions.")

	s.ErrorContains(err, "ignored")
}

func (s *PolicySuite) TestAWebhookNeedsNoPolicyProseBecauseTheServerHoldsThePolicy() {
	policy, err := Parse("---\ntype: webhook\nurl: https://example.test/check\n---\n")
	s.Require().NoError(err)

	s.Equal(KindWebhook, policy.Kind)
	s.Empty(policy.Text)
}

func (s *PolicySuite) TestAModeNobodyHasIsRefused() {
	_, err := Parse("---\nmode: eventually\n---\nOnly Stream questions.")

	s.ErrorContains(err, "eventually")
}

func (s *PolicySuite) TestAPolicyWithNothingToSayOnARefusalIsRefused() {
	// The refusal is the only part of this a caller ever hears, so a blank one is an agent
	// that goes silent on the turns it was configured to decline.
	_, err := Parse("---\nrefusal: \"\"\n---\nOnly Stream questions.")

	s.ErrorContains(err, "nothing for the agent to say")
}
