//go:build integration

package session

import (
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
)

// limitedSandboxes has the manager ask a real enforcer, sharing slots through the suites' Redis,
// what sandbox a session may have, with the development tiers seeded, and returns an app of an
// organization on the default tier and that tier's sandbox limits.
func (s *RuntimeSuite) limitedSandboxes() (routing.Owner, usage.SandboxLimits) {
	_, sharing := policytest.SharedHolds(s.T())
	money := policytest.Start(s.T(), "session_sandbox", sharing)
	file, err := ledger.ReadTierFile("../../tiers/dev.json")
	s.Require().NoError(err)
	var limits *usage.SandboxLimits
	for _, tier := range file.Tiers {
		money.Usage.SetTier(s.T(), tier)
		if tier.Name == usage.DefaultTier {
			limits = tier.Limits.Sandbox
		}
	}
	s.Require().NotNil(limits, "the default tier limits sandboxes")
	s.manager.options.SandboxLimits = money.Enforcer
	return money.Owner(s.T(), 1_000_000), *limits
}

// sandboxSpec is a session of the owner's that runs code in a sandbox configured so.
func sandboxSpec(owner routing.Owner, config sandbox.Config) Spec {
	id := "runtime-" + uuid.NewString()
	attribution := owner.Attribution
	attribution.SessionID = id
	return Spec{ID: id, CustomerID: owner.CustomerID, Sandbox: daytonaProvider, SandboxOptions: config, Attribution: attribution}
}

func (s *RuntimeSuite) TestASessionsSandboxBeyondItsTierIsNotMade() {
	server := s.daytona()
	s.manages()
	owner, limits := s.limitedSandboxes()
	s.Require().NotNil(limits.MaxCPU)

	for name, config := range map[string]sandbox.Config{
		"more network": {Network: true},
		"bigger":       {CPU: *limits.MaxCPU + 1},
	} {
		box, err := s.manager.box(s.ctx, sandboxSpec(owner, config), nil)
		s.Require().NoError(err, "the session runs: %s", name)
		s.Nil(box, "without a sandbox, as one configured with none: %s", name)
	}
	s.Empty(server.Created())
}

func (s *RuntimeSuite) TestASessionsSandboxRunsOnlyInAFreeSlot() {
	s.daytona()
	s.manages()
	owner, limits := s.limitedSandboxes()
	s.Require().NotNil(limits.MaxConcurrent)
	s.Require().Equal(1, *limits.MaxConcurrent, "the default tier runs one sandbox at a time")
	first, err := s.manager.box(s.ctx, sandboxSpec(owner, sandbox.Config{}), nil)
	s.Require().NoError(err)
	second, err := s.manager.box(s.ctx, sandboxSpec(owner, sandbox.Config{}), nil)
	s.Require().NoError(err)
	_, err = first.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	_, err = second.Run(s.ctx, "print(1)", nil)
	s.Require().ErrorIs(err, policy.ErrSandboxLimit, "the organization's one sandbox is running")

	s.Require().NoError(first.Close())
	_, err = second.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err, "its slot was given back")
	s.Require().NoError(second.Close())
}
