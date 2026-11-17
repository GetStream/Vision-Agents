//go:build integration

package session

import (
	"log/slog"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// A call's episode is dated by Postgres's clock, as its number's link is, from when its session
// began: a router whose clock runs ahead of Postgres's, or a card written once a late caller
// joined, does not move the call past the link.

// cardOfCall writes the card of a call session to number as the session's start does
// (callCards.write), the session made when the router's own clock read created and began by the
// process's monotonic clock, and returns the session id.
func (s *EpisodeReadingSuite) cardOfCall(number string, created, began time.Time) string {
	cards := newCallCards(s.store, nil, s.apps, slog.New(slog.DiscardHandler))
	s.T().Cleanup(cards.Close)
	call := "call-" + uuid.NewString()
	created = created.UTC().Truncate(time.Microsecond)
	session := &Session{id: "session-" + uuid.NewString(), created: created, began: began, transcribedInto: "agent:" + call}
	spec := Spec{CustomerID: s.customerID, ConfigID: s.configID, AgentName: "Athena", CallID: call, StreamApp: suiteDeploymentApp}
	s.Require().NoError(cards.write(s.ctx, session, spec, number))
	return session.id
}

// callEpisode is the start of a call session's episode as Postgres holds it, and the
// omni-channel its card is in.
func (s *EpisodeReadingSuite) callEpisode(session string) (time.Time, string) {
	var started time.Time
	var omni string
	err := s.store.DB().QueryRowContext(s.ctx, `SELECT ep.started_at, cm.conversation_id FROM episodes AS ep
JOIN contact_map AS cm ON cm.id = ep.contact_id WHERE ep.customer_id = ? AND ep.session_id = ?`,
		s.customerID, session).Scan(&started, &omni)
	s.Require().NoError(err)
	return started, omni
}

// postgresNow is Postgres's clock.
func (s *EpisodeReadingSuite) postgresNow() time.Time {
	var now time.Time
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT statement_timestamp()").Scan(&now))
	return now
}

// Defect: a call episode dated by the router's clock (Session.created) while its number's link
// is dated by Postgres's: with the router an hour ahead, a call that began before the link, whose
// card is written after it, starts after the link's boundary, and every session reads what the
// number said before it was linked. The call that began after the link is read either way.
func (s *EpisodeReadingSuite) TestACallThatBeganBeforeItsNumbersLinkIsLeftOutWithTheRoutersClockAhead() {
	s.cardsConfig()
	number, user := "+15550100191", "anya-"+uuid.NewString()
	ahead := time.Now().Add(time.Hour)
	began := time.Now().Add(-time.Minute)
	s.Require().NoError(s.store.LinkUserPhone(s.ctx, &store.UserPhone{CustomerID: s.customerID,
		StreamAppPK: suiteDeploymentApp, UserID: user, Phone: number}))
	before := s.cardOfCall(number, ahead, began)
	after := s.cardOfCall(number, ahead, time.Now())

	_, omni := s.callEpisode(before)
	cards, err := s.store.EpisodeCards(s.ctx, store.CardsQuery{CustomerID: s.customerID, AgentConfigID: s.configID,
		ConversationID: omni, Limit: 10})
	s.Require().NoError(err)
	var read []string
	for _, card := range cards {
		read = append(read, card.SessionID)
	}
	s.Equal([]string{after}, read, "the call from before the link is left out, the one after it read")
}

// Defect: a call episode dated when its card is written, by Postgres's clock or the router's,
// starts when a late caller joined, not when the call began, and its card says so too.
func (s *EpisodeReadingSuite) TestACardWrittenTenMinutesIntoACallKeepsTheCallsStart() {
	s.cardsConfig()
	number := "+15550100192"
	postgres := s.postgresNow()
	began := time.Now().Add(-10*time.Minute - 500*time.Millisecond)

	session := s.cardOfCall(number, began, began)

	started, omni := s.callEpisode(session)
	s.WithinDuration(postgres.Add(-10*time.Minute-500*time.Millisecond), started, 200*time.Millisecond, "the call's start by Postgres's clock")
	var card []string
	for _, message := range s.chat.Stored(strings.TrimPrefix(omni, "agent:")) {
		custom, _ := message["custom"].(map[string]any)
		if custom["call_id"] != nil {
			card = append(card, custom["started_at"].(string))
		}
	}
	s.Equal([]string{started.UTC().Format(time.RFC3339)}, card, "the card gives the start Postgres holds")
}

// Defect: a session that keeps no start of its own by the monotonic clock dates its call's
// episode when the card is written, once its late caller joins, a second into the call here.
func (s *EpisodeReadingSuite) TestALateCallersCardStartsWhenTheSessionWasMade() {
	edge := s.attended()
	call := uuid.NewString()
	s.chat.PutCall("agent", call)
	made := s.postgresNow()
	created := s.opens(s.inboundSpec(call, true))
	opened := s.postgresNow()
	db := s.store
	s.Never(func() bool { return len(callEpisodesIn(db, created.ID())) > 0 }, time.Second, 20*time.Millisecond,
		"no card before anybody rings in")

	s.chat.PutCall("agent", call, "sip-"+lateCaller)
	edge.arrives(stt.Participant{ID: "caller-" + uuid.NewString(), UserID: "sip-" + lateCaller})

	s.Require().Eventually(func() bool { return len(callEpisodesIn(db, created.ID())) == 1 }, settleFor, 10*time.Millisecond)
	started, _ := s.callEpisode(created.ID())
	s.False(started.Before(made), "started %s, before the session was made at %s", started, made)
	s.False(started.After(opened.Add(500*time.Millisecond)), "started %s, after the session opened at %s", started, opened)
}
