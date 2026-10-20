//go:build integration

package session

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConcurrencySuite is the cap a manifest's rate_limit.max_concurrent_calls puts on a
// connection's connector calls in flight at once: a call over it waits for one to end, until
// its binding's timeout, and is then answered without being sent.
type ConcurrencySuite struct {
	connectorFixture
}

func TestConcurrencySuite(t *testing.T) {
	suite.Run(t, new(ConcurrencySuite))
}

// TestWithoutACapEveryCallIsInFlightAtOnce: the control. A manifest with no cap sends all five
// calls at once, as the base did (probe on 3f4cf612e: 5 in flight).
func (s *ConcurrencySuite) TestWithoutACapEveryCallIsInFlightAtOnce() {
	primary := s.connection("", "primary")
	d := s.attachAll(s.fixed("crm", primary, "whoami"))
	release := s.provider.hold()
	defer release()

	answers := s.callAtOnce(d, "crm__whoami", 5)
	s.Require().Eventually(func() bool { return s.provider.inFlight("primary") == 5 }, 5*time.Second, 10*time.Millisecond)
	release()

	s.Equal([]string{"primary", "primary", "primary", "primary", "primary"}, answers())
	s.Equal(5, s.provider.mostInFlight("primary"))
}

// TestACapOfTwoKeepsAtMostTwoCallsInFlight: the ticket's acceptance. Five calls at once reach
// the provider two at a time, and every one is answered.
func (s *ConcurrencySuite) TestACapOfTwoKeepsAtMostTwoCallsInFlight() {
	s.capAt(2)
	primary := s.connection("", "primary")
	d := s.attachAll(s.fixed("crm", primary, "whoami"))
	release := s.provider.hold()
	defer release()

	answers := s.callAtOnce(d, "crm__whoami", 5)
	s.Require().Eventually(func() bool { return s.provider.inFlight("primary") == 2 }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool { return s.provider.inFlight("primary") > 2 }, 300*time.Millisecond, 10*time.Millisecond)
	release()

	s.Equal([]string{"primary", "primary", "primary", "primary", "primary"}, answers())
	s.Equal(2, s.provider.mostInFlight("primary"))
	s.Equal(5, s.provider.calls("primary"))
}

// TestACallStillWaitingAtItsTimeoutIsNotSent: the model reads connector_busy, which is no
// error it would retry at once, and the row says the router held it.
func (s *ConcurrencySuite) TestACallStillWaitingAtItsTimeoutIsNotSent() {
	s.capAt(1)
	primary := s.connection("", "primary")
	quick := s.fixed("quick", primary, "whoami")
	quick.TimeoutMs = 200
	d := s.attachAll(s.fixed("crm", primary, "whoami"), quick)
	release := s.provider.hold()
	defer release()
	first := s.callAtOnce(d, "crm__whoami", 1)
	s.Require().Eventually(func() bool { return s.provider.inFlight("primary") == 1 }, 5*time.Second, 10*time.Millisecond)

	said, err := s.call(d, "quick__whoami", "{}")
	release()

	s.Require().NoError(err, "not an error the model would retry")
	s.Equal("connector_busy: quick__whoami was not run: the connector's limit of calls at once (1) was reached, "+
		"and none of them ended in time. Call it again once one has answered.", said)
	s.Equal([]string{"primary"}, first())
	s.Equal(1, s.provider.calls("primary"), "the held call was not sent")
	row := s.rowOf(primary, "quick")
	s.Equal(store.InvocationDenied, row.ErrorType)
	s.Equal("held by the router: in-flight limit 1 reached on the connection", row.ErrorMessage)
}

// TestACallInterruptedWhileWaitingIsNotSent: a turn that ends while the call waits ends the
// wait, and the call never reaches the provider.
func (s *ConcurrencySuite) TestACallInterruptedWhileWaitingIsNotSent() {
	s.capAt(1)
	primary := s.connection("", "primary")
	d := s.attachAll(s.fixed("crm", primary, "whoami"))
	release := s.provider.hold()
	defer release()
	first := s.callAtOnce(d, "crm__whoami", 1)
	s.Require().Eventually(func() bool { return s.provider.inFlight("primary") == 1 }, 5*time.Second, 10*time.Millisecond)

	turn, interrupt := context.WithTimeout(s.ctx, 100*time.Millisecond)
	defer interrupt()
	_, err := d.Run(turn, llm.ToolCall{ID: uuid.NewString(), Name: "crm__whoami", Arguments: "{}"})
	release()

	s.ErrorIs(err, context.DeadlineExceeded)
	s.Equal([]string{"primary"}, first())
	s.Equal(1, s.provider.calls("primary"))
}

// TestTheCapIsPerConnection: a cap of one on one connection holds nothing of another
// connection to the same connector.
func (s *ConcurrencySuite) TestTheCapIsPerConnection() {
	s.capAt(1)
	primary, secondary := s.connection("", "primary"), s.connection("", "secondary")
	d := s.attachAll(s.fixed("crm", primary, "whoami"), s.fixed("other", secondary, "whoami"))
	release := s.provider.hold()
	defer release()

	first := s.callAtOnce(d, "crm__whoami", 1)
	s.Require().Eventually(func() bool { return s.provider.inFlight("primary") == 1 }, 5*time.Second, 10*time.Millisecond)
	other := s.callAtOnce(d, "other__whoami", 1)
	s.Require().Eventually(func() bool { return s.provider.inFlight("secondary") == 1 }, 5*time.Second, 10*time.Millisecond)
	release()

	s.Equal([]string{"primary"}, first())
	s.Equal([]string{"secondary"}, other())
}

// capAt stores a revision of the test's connector capped at limit calls in flight, which the
// connections the test makes after it pin.
func (s *ConcurrencySuite) capAt(limit int) {
	manifest, err := core.ParseManifest([]byte(s.crmManifest(s.connectorID, 1) +
		fmt.Sprintf("rate_limit:\n  max_concurrent_calls: %d\n", limit)))
	s.Require().NoError(err)
	definition, err := s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
	s.Require().NoError(err)
	s.revision = definition.Revision
}

// attachAll opens an app's session of a config with bindings.
func (s *ConcurrencySuite) attachAll(bindings ...store.ConnectorBinding) *dispatcher {
	d, _, unavailable, err := s.attach(s.spec(s.config(bindings...), "", nil))
	s.Require().NoError(err)
	s.Require().Empty(unavailable)
	return d
}

// callAtOnce runs count calls of name through d at once, and returns what waits for them and
// gives each one's text, sorted.
func (s *ConcurrencySuite) callAtOnce(d *dispatcher, name string, count int) func() []string {
	var wg sync.WaitGroup
	answers := make([]string, count)
	failures := make([]error, count)
	for i := range count {
		wg.Go(func() { answers[i], failures[i] = s.call(d, name, "{}") })
	}
	return func() []string {
		wg.Wait()
		for _, err := range failures {
			s.Require().NoError(err)
		}
		return answers
	}
}

// rowOf is the one invocation row binding's calls on connection left, once written.
func (s *ConcurrencySuite) rowOf(connection, binding string) store.ConnectorInvocation {
	var found []store.ConnectorInvocation
	s.Require().Eventually(func() bool {
		rows, err := s.store.ConnectorInvocations(s.ctx, s.customerID, connection, 0, nil)
		s.Require().NoError(err)
		found = found[:0]
		for _, row := range rows {
			if row.Binding == binding {
				found = append(found, row)
			}
		}
		return len(found) == 1
	}, 5*time.Second, 20*time.Millisecond)
	return found[0]
}
