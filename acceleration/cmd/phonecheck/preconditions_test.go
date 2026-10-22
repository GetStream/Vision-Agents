package main

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

func outbound() Config {
	return Config{Direction: person.Outbound, AgentNumber: agentNumber, PersonNumber: personNumber}
}

func inbound() Config {
	return Config{Direction: person.Inbound, AgentNumber: agentNumber, PersonNumber: personNumber}
}

func routerWith(numbers ...acceleration.PhoneNumber) *fakeRouter {
	return &fakeRouter{steps: &steps{}, numbers: numbers}
}

// refused asserts err is a refusal saying want, and that nothing but reads happened.
func refused(t *testing.T, r *fakeRouter, err error, want string) {
	t.Helper()
	require.ErrorContains(t, err, want)
	assert.False(t, r.steps.placedACall(), "steps: %v", r.steps.all())
}

func TestAnUnreachableLocalRouterSaysToStartItFirst(t *testing.T) {
	r := routerWith()
	r.healthErr = errors.New("connection refused")

	err := checkRouter(context.Background(), r, "http://localhost:8080")

	refused(t, r, err, "no router answers at http://localhost:8080; start it first (see README)")
}

func TestAnUnreachableRemoteRouterIsNotToldToStart(t *testing.T) {
	r := routerWith()
	r.healthErr = errors.New("connection refused")

	err := checkRouter(context.Background(), r, "https://router.example.com")

	refused(t, r, err, "no router answers at https://router.example.com")
	assert.NotContains(t, err.Error(), "start it first")
}

func TestAnAgentNumberNotRegisteredForTheCustomerIsRefused(t *testing.T) {
	r := routerWith(attached(otherNumber, "twilio"))

	err := checkNumbers(context.Background(), r, outbound())

	refused(t, r, err, fmt.Sprintf("the agent number %s is not registered", agentNumber))
}

func TestAPersonNumberTheRouterHasAttachedIsRefused(t *testing.T) {
	r := routerWith(attached(agentNumber, "twilio"), attached(personNumber, "twilio"))

	err := checkNumbers(context.Background(), r, outbound())

	refused(t, r, err, fmt.Sprintf("the person number %s is registered", personNumber))
}

func TestAPersonNumberRegisteredButNotAttachedIsRefusedToo(t *testing.T) {
	r := routerWith(attached(agentNumber, "twilio"), unattached(personNumber, "twilio"))

	err := checkNumbers(context.Background(), r, outbound())

	refused(t, r, err, fmt.Sprintf("the person number %s is registered", personNumber))
}

func TestARouterThatCannotStoreLogsIsRefused(t *testing.T) {
	r := routerWith()
	r.logsErr = fmt.Errorf("logs: %w", errNoLogStore)

	err := checkLogs(context.Background(), r)

	refused(t, r, err, "the router keeps no logs (it needs Postgres); the agent's SFU is read from them")
}

func TestInboundRefusesAnAgentNumberThatIsNotAttached(t *testing.T) {
	r := routerWith(unattached(agentNumber, "twilio"))

	err := checkNumbers(context.Background(), r, inbound())

	refused(t, r, err, fmt.Sprintf("the agent number %s is not attached", agentNumber))
}

func TestInboundRefusesASIPTrunkNumber(t *testing.T) {
	r := routerWith(attached(agentNumber, "sip_trunk"))

	err := checkNumbers(context.Background(), r, inbound())

	refused(t, r, err, fmt.Sprintf("the agent number %s is a sip_trunk number", agentNumber))
}

func TestInboundRefusesACustomerWithAnotherAttachedNumber(t *testing.T) {
	r := routerWith(attached(agentNumber, "twilio"), attached(otherNumber, "telnyx"))

	err := checkNumbers(context.Background(), r, inbound())

	refused(t, r, err, fmt.Sprintf("the customer has another attached number, %s", otherNumber))
}

func TestInboundAcceptsAnotherNumberThatIsNotAttached(t *testing.T) {
	r := routerWith(attached(agentNumber, "twilio"), unattached(otherNumber, "telnyx"))

	require.NoError(t, checkNumbers(context.Background(), r, inbound()))
}

func TestOutboundAcceptsASIPTrunkAgentNumber(t *testing.T) {
	r := routerWith(attached(agentNumber, "sip_trunk"), attached(otherNumber, "twilio"))

	require.NoError(t, checkNumbers(context.Background(), r, outbound()))
}
