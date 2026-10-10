//go:build integration

package api

import (
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AI-1048: an event is subscribed to once per connection, by the oldest live config that
// declares it. A test copy subscribes to nothing.

// TestATestCopyOfAWatcherSubscribesToNothing: on base the copy got a subscription of its own at
// the next validate, and each event opened two conversations.
func (s *ConnectionEventsSuite) TestATestCopyOfAWatcherSubscribesToNothing() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	watcher := s.config(binding)
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "copy-" + s.utils.uuid(), "mode": "text", "llm": "noted/noted-model",
		"connectors": []map[string]any{binding}, "tags": map[string]string{store.DraftOfTag: watcher},
	}, nil))

	s.validate(connection)

	s.Require().Eventually(func() bool { return s.activeFor(connection, watcher) }, settleFor, 20*time.Millisecond)
	s.Never(func() bool { return len(s.held(connection)) != 1 }, time.Second, 50*time.Millisecond)
}

// TestATestCopyAloneSubscribesToNothing: a test copy never subscribes, even when no live config
// declares the event.
func (s *ConnectionEventsSuite) TestATestCopyAloneSubscribesToNothing() {
	connection := s.connection()
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "copy-" + s.utils.uuid(), "mode": "text", "llm": "noted/noted-model",
		"connectors": []map[string]any{s.binding(connection, issueCreated)}, "tags": map[string]string{store.DraftOfTag: s.utils.uuid()},
	}, nil))

	s.validate(connection)

	s.Never(func() bool { return len(s.held(connection)) > 0 || len(s.atTheFake(connection)) > 0 }, time.Second, 50*time.Millisecond)
}

// TestASecondConfigDeclaringTheSameEventIsSubscribedOnce: two live configs bind one connection
// (it is no channel, so both may) and declare one event. The oldest is subscribed, and an
// event opens one conversation.
func (s *ConnectionEventsSuite) TestASecondConfigDeclaringTheSameEventIsSubscribedOnce() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	first := s.config(binding)
	s.config(binding)

	s.validate(connection)

	s.Require().Eventually(func() bool { return s.activeFor(connection, first) }, settleFor, 20*time.Millisecond)
	s.Never(func() bool { return len(s.held(connection)) != 1 }, time.Second, 50*time.Millisecond)
	answered := s.provider.Emit(issueCreated, map[string]any{"title": s.utils.uuid()})
	s.Equal([]int{http.StatusAccepted}, s.statuses(answered, connection))
}

// TestAnOlderConfigThatDeclaresTheEventTakesItsSubscriptionOver: the newer config was
// subscribed while it alone declared the event; once the older declares it too, the next
// validate unsubscribes the newer one at the server and drops it, so the event still opens one
// conversation.
func (s *ConnectionEventsSuite) TestAnOlderConfigThatDeclaresTheEventTakesItsSubscriptionOver() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	quiet := map[string]any{}
	for key, value := range binding {
		quiet[key] = value
	}
	delete(quiet, "events")
	older := s.config(quiet)
	newer := s.config(binding)
	s.validate(connection)
	s.Require().Eventually(func() bool { return s.activeFor(connection, newer) }, settleFor, 20*time.Millisecond)
	replaced := s.atTheFake(connection)[0]
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+older,
		map[string]any{"connectors": []map[string]any{binding}}, nil))

	s.validate(connection)

	s.Eventually(func() bool { return s.activeFor(connection, older) && len(s.atTheFake(connection)) == 1 }, settleFor, 20*time.Millisecond)
	s.Equal(http.StatusGone, s.provider.DeliverEvent(replaced.URL, replaced.Secret, replaced.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{}))
}

// TestADeletedOwnerHandsItsEventsToTheNextConfig: the owner of the event is deleted, and no
// validate follows. The subscription the server delivers to carries the next config's events
// until that config's own is active, then goes; on base both configs were subscribed, and the
// second answered. At d17a7938 the delivery was a 410 and the event was lost (reviewer probe
// a: [410], then nothing held).
func (s *ConnectionEventsSuite) TestADeletedOwnerHandsItsEventsToTheNextConfig() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	first := s.config(binding)
	second := s.config(binding)
	s.validate(connection)
	s.Require().Eventually(func() bool { return s.activeFor(connection, first) }, settleFor, 20*time.Millisecond)
	carrier := s.atTheFake(connection)[0]
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+first, nil, nil))

	marker := s.utils.uuid()
	answered := s.provider.Emit(issueCreated, map[string]any{"title": marker})

	s.Equal(http.StatusAccepted, answered[carrier.ID], "the deleted owner's subscription carries the event")
	s.Eventually(func() bool { return s.modelWasAsked(marker) }, settleFor, 20*time.Millisecond)
	s.Require().Eventually(func() bool { return s.activeFor(connection, second) && len(s.atTheFake(connection)) == 1 },
		settleFor, 20*time.Millisecond, "the next config is subscribed, and the old subscription goes")
	again := s.provider.Emit(issueCreated, map[string]any{"title": s.utils.uuid()})
	s.Equal([]int{http.StatusAccepted}, s.statuses(again, connection))
}

// TestAnOlderConfigThatGainsTheEventIsAnsweredBeforeAValidate: the older config starts
// declaring the event the newer is subscribed to, and no validate follows. The newer's
// subscription carries the event as the older's until the older's is active, and an event it
// carried is not taken a second time through the older's (reviewer probe b: [410] at d17a7938).
func (s *ConnectionEventsSuite) TestAnOlderConfigThatGainsTheEventIsAnsweredBeforeAValidate() {
	connection := s.connection()
	binding := s.binding(connection, issueCreated)
	quiet := map[string]any{}
	for key, value := range binding {
		quiet[key] = value
	}
	delete(quiet, "events")
	older := s.config(quiet)
	newer := s.config(binding)
	s.validate(connection)
	s.Require().Eventually(func() bool { return s.activeFor(connection, newer) }, settleFor, 20*time.Millisecond)
	carrier := s.atTheFake(connection)[0]
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+older,
		map[string]any{"connectors": []map[string]any{binding}}, nil))
	event := "evt_" + s.utils.uuid()

	s.Equal(http.StatusAccepted, s.provider.DeliverEvent(carrier.URL, carrier.Secret, carrier.ID, event, issueCreated, map[string]any{}))

	s.Require().Eventually(func() bool { return s.activeFor(connection, older) && len(s.atTheFake(connection)) == 1 },
		settleFor, 20*time.Millisecond, "the older config is subscribed within a lease, and the newer's subscription goes")
	owner := s.atTheFake(connection)[0]
	s.Equal(http.StatusOK, s.provider.DeliverEvent(owner.URL, owner.Secret, owner.ID, event, issueCreated, map[string]any{}),
		"the event the newer's subscription carried opens no second conversation")
	s.Equal(http.StatusGone, s.provider.DeliverEvent(carrier.URL, carrier.Secret, carrier.ID, "evt_"+s.utils.uuid(), issueCreated, map[string]any{}))
}

// TestOneConfigWithTwoBindingsOfAnEventIsSubscribedOnce: one config binds the connection twice
// and both bindings declare the event. The first binding owns it, so the second's subscription,
// made while it alone declared the event, goes, and an event opens one conversation.
func (s *ConnectionEventsSuite) TestOneConfigWithTwoBindingsOfAnEventIsSubscribedOnce() {
	connection := s.connection()
	declaring := s.binding(connection, issueCreated)
	quiet := map[string]any{}
	for key, value := range declaring {
		quiet[key] = value
	}
	delete(quiet, "events")
	second := map[string]any{}
	for key, value := range declaring {
		second[key] = value
	}
	second["name"] = "crm_again"
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "watcher-" + s.utils.uuid(), "mode": "text", "llm": "noted/noted-model",
		"connectors": []map[string]any{quiet, second},
	}, &created))
	s.validate(connection)
	s.Require().Eventually(func() bool { return s.activeFor(connection, created.Id) }, settleFor, 20*time.Millisecond)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{declaring, second}}, nil))

	s.validate(connection)

	s.Require().Eventually(func() bool {
		held := s.held(connection)
		return len(held) == 1 && held[0].Binding == "crm" && held[0].Status == store.ConnectionEventActive && len(s.atTheFake(connection)) == 1
	}, settleFor, 20*time.Millisecond)
	answered := s.provider.Emit(issueCreated, map[string]any{"title": s.utils.uuid()})
	s.Equal([]int{http.StatusAccepted}, s.statuses(answered, connection))
}

// activeFor reports whether the connection's only subscription is configID's, and active.
func (s *ConnectionEventsSuite) activeFor(connection, configID string) bool {
	held := s.held(connection)
	return len(held) == 1 && held[0].ConfigID == configID && held[0].Status == store.ConnectionEventActive
}
