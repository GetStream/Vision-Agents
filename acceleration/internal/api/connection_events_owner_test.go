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

// activeFor reports whether the connection's only subscription is configID's, and active.
func (s *ConnectionEventsSuite) activeFor(connection, configID string) bool {
	held := s.held(connection)
	return len(held) == 1 && held[0].ConfigID == configID && held[0].Status == store.ConnectionEventActive
}
