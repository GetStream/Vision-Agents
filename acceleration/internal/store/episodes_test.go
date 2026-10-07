//go:build integration

package store

import (
	"fmt"
	"sync"
	"time"
)

// episodeOf is a thread episode of the acme-app customer for a contact, which a test changes
// one part of.
func (s *StoreSuite) episodeOf(contactID, threadChannel string) *Episode {
	return &Episode{
		CustomerID:    "acme-app",
		ContactID:     contactID,
		Source:        "sms",
		ThreadChannel: threadChannel,
		StartedAt:     s.base,
		StreamAppPK:   4242,
	}
}

// mapped is a contact map row of the acme-app customer, for an episode to name.
func (s *StoreSuite) mapped(address string) string {
	entry := contact(address, "agent:omni-"+address)
	_, err := s.store.MapContact(s.ctx, entry)
	s.Require().NoError(err)
	return entry.ID
}

func (s *StoreSuite) TestTheFirstMessageOfAThreadOpensAnEpisodeInProgress() {
	episode := s.episodeOf(s.mapped("+15550100"), "agent:thread-one")

	opened, err := s.store.OpenEpisode(s.ctx, episode)

	s.Require().NoError(err)
	s.True(opened)
	s.Equal(episodeInProgress, episode.Status)
	s.Equal("episode-"+episode.ID, episode.CardMessageID)
	s.Equal(s.base, episode.StartedAt)
}

// Three SMS in a row are one episode, so one card.
func (s *StoreSuite) TestTheNextMessagesOfAThreadFindItsOpenEpisode() {
	person := s.mapped("+15550100")
	first := s.episodeOf(person, "agent:thread-one")
	_, err := s.store.OpenEpisode(s.ctx, first)
	s.Require().NoError(err)

	for range 2 {
		next := s.episodeOf(person, "agent:thread-one")
		opened, err := s.store.OpenEpisode(s.ctx, next)
		s.Require().NoError(err)
		s.False(opened)
		s.Equal(first.ID, next.ID)
		s.Equal(first.CardMessageID, next.CardMessageID)
	}
	s.Equal(1, s.episodeRows())
}

// Two first messages of one thread taken at once by two routers open one episode, so the
// thread gets one card: the open-thread index decides which.
func (s *StoreSuite) TestTwoFirstMessagesOfOneThreadAtOnceOpenOneEpisode() {
	person := s.mapped("+15550100")
	pools := s.pools(8)
	var wg sync.WaitGroup
	ids := make([]string, len(pools))
	opened := make([]bool, len(pools))
	for i, pool := range pools {
		wg.Add(1)
		go func() {
			defer wg.Done()
			episode := s.episodeOf(person, "agent:thread-one")
			var err error
			opened[i], err = pool.OpenEpisode(s.ctx, episode)
			s.NoError(err)
			ids[i] = episode.ID
		}()
	}
	wg.Wait()

	count := 0
	for i, id := range ids {
		s.Equal(ids[0], id)
		if opened[i] {
			count++
		}
	}
	s.Equal(1, count, "one of them opened it, so one card is written")
	s.Equal(1, s.episodeRows())
}

func (s *StoreSuite) TestAnotherThreadIsAnotherEpisode() {
	person := s.mapped("+15550100")
	_, err := s.store.OpenEpisode(s.ctx, s.episodeOf(person, "agent:thread-one"))
	s.Require().NoError(err)

	opened, err := s.store.OpenEpisode(s.ctx, s.episodeOf(person, "agent:thread-two"))

	s.Require().NoError(err)
	s.True(opened)
}

// Two calls on one call channel are two episodes: the session keys a call, not the channel.
func (s *StoreSuite) TestEachCallSessionIsAnEpisodeOfItsOwnOnOneCallChannel() {
	person := s.mapped("+15550100")
	call := func(session string) *Episode {
		episode := s.episodeOf(person, "agent:phone-15550199")
		episode.Source, episode.CallID, episode.SessionID = EpisodeCall, "phone-15550199", session
		return episode
	}
	_, err := s.store.OpenEpisode(s.ctx, call("session-one"))
	s.Require().NoError(err)

	opened, err := s.store.OpenEpisode(s.ctx, call("session-two"))
	s.Require().NoError(err)
	s.True(opened)
	again, err := s.store.OpenEpisode(s.ctx, call("session-two"))
	s.Require().NoError(err)
	s.False(again, "a session's call is one episode")
	s.Equal(2, s.episodeRows())
}

// A thread episode closed by T55 leaves the thread free for the next one.
func (s *StoreSuite) TestAThreadWhoseEpisodeEndedOpensANewOne() {
	person := s.mapped("+15550100")
	first := s.episodeOf(person, "agent:thread-one")
	_, err := s.store.OpenEpisode(s.ctx, first)
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE episodes SET status = 'ended', ended_at = now() WHERE id = ?", first.ID)
	s.Require().NoError(err)

	next := s.episodeOf(person, "agent:thread-one")
	opened, err := s.store.OpenEpisode(s.ctx, next)

	s.Require().NoError(err)
	s.True(opened)
	s.NotEqual(first.ID, next.ID)
}

func (s *StoreSuite) TestOnlyACallEpisodeNamesASession() {
	person := s.mapped("+15550100")
	thread := s.episodeOf(person, "agent:thread-one")
	thread.SessionID = "session-one"
	_, err := s.store.OpenEpisode(s.ctx, thread)
	s.ErrorContains(err, "only a call episode, names its session")

	call := s.episodeOf(person, "agent:call-one")
	call.Source = EpisodeCall
	_, err = s.store.OpenEpisode(s.ctx, call)
	s.ErrorContains(err, "only a call episode, names its session")
}

// threadAt is a thread episode of person in thread, opened at at.
func (s *StoreSuite) threadAt(person, thread string, at time.Time) *Episode {
	episode := s.episodeOf(person, thread)
	episode.StartedAt = at
	_, err := s.store.OpenEpisode(s.ctx, episode)
	s.Require().NoError(err)
	return episode
}

// closeIdle closes the thread episodes with no message for an hour before now, leased for
// five minutes.
func (s *StoreSuite) closeIdle(store *Store, now time.Time) []ClosedEpisode {
	closed, err := store.CloseIdleEpisodes(s.ctx, now.Add(-time.Hour), now, 50, now.Add(5*time.Minute))
	s.Require().NoError(err)
	return closed
}

// episodeStatus is an episode's status as stored.
func (s *StoreSuite) episodeStatus(id string) string {
	var status string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT status FROM episodes WHERE id = ?", id).Scan(&status))
	return status
}

func (s *StoreSuite) TestAThreadQuietForTheIdlePeriodIsClosedAndLeasedToItsCloser() {
	person := s.mapped("+15550100")
	episode := s.threadAt(person, "agent:thread-one", s.base)
	now := s.base.Add(time.Hour)

	closed := s.closeIdle(s.store, now)

	s.Require().Len(closed, 1)
	s.Equal(episode.ID, closed[0].ID)
	s.Equal(EpisodeEnded, closed[0].Status)
	s.Require().NotNil(closed[0].EndedAt)
	s.True(now.Equal(*closed[0].EndedAt))
	s.Require().NotNil(closed[0].SummaryLeaseUntil)
	s.True(now.Add(5 * time.Minute).Equal(*closed[0].SummaryLeaseUntil))
	s.Equal("agent:omni-+15550100", closed[0].ConversationID, "the omni-channel the card is in")
	s.Equal("agent-one", closed[0].AgentConfigID, "the config whose LLM writes the summary")
	s.Equal(EpisodeEnded, s.episodeStatus(episode.ID))
}

// Each message of a thread is its last one, so a thread that is talking is not idle.
func (s *StoreSuite) TestAThreadWithAMessageInTheIdlePeriodStaysOpen() {
	person := s.mapped("+15550100")
	episode := s.threadAt(person, "agent:thread-one", s.base)
	s.threadAt(person, "agent:thread-one", s.base.Add(50*time.Minute))

	closed := s.closeIdle(s.store, s.base.Add(90*time.Minute))

	s.Empty(closed)
	s.Equal(episodeInProgress, s.episodeStatus(episode.ID))
}

// A call ends when its call does, never for being quiet.
func (s *StoreSuite) TestACallIsNeverClosedAsIdle() {
	call := s.episodeOf(s.mapped("+15550100"), "agent:call-one")
	call.Source, call.CallID, call.SessionID = EpisodeCall, "call-one", "session-one"
	_, err := s.store.OpenEpisode(s.ctx, call)
	s.Require().NoError(err)

	closed := s.closeIdle(s.store, s.base.Add(48*time.Hour))

	s.Empty(closed)
	s.Equal(episodeInProgress, s.episodeStatus(call.ID))
}

// Several routers sweep at once: each idle episode is closed by one of them, once, so one
// summary is written for it.
func (s *StoreSuite) TestTwoSweepersAtOnceCloseEachIdleEpisodeOnce() {
	person := s.mapped("+15550100")
	for i := range 20 {
		s.threadAt(person, fmt.Sprintf("agent:thread-%d", i), s.base)
	}
	pools := s.pools(8)
	now := s.base.Add(time.Hour)
	var wg sync.WaitGroup
	closed := make([][]ClosedEpisode, len(pools))
	for i, pool := range pools {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var err error
			closed[i], err = pool.CloseIdleEpisodes(s.ctx, now.Add(-time.Hour), now, 50, now.Add(5*time.Minute))
			s.NoError(err)
		}()
	}
	wg.Wait()

	seen := map[string]int{}
	for _, swept := range closed {
		for _, episode := range swept {
			seen[episode.ID]++
		}
	}
	s.Len(seen, 20)
	for id, times := range seen {
		s.Equal(1, times, "episode %s was closed by more than one sweeper", id)
	}
}

// A router that is closing an episode holds its row until it commits. Another sweep leaves
// that row to it, rather than waiting and closing it a second time.
func (s *StoreSuite) TestAnEpisodeAnotherRouterIsClosingIsLeftToIt() {
	episode := s.threadAt(s.mapped("+15550100"), "agent:thread-one", s.base)
	closing, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	_, err = closing.ExecContext(s.ctx, "UPDATE episodes SET status = 'ended', ended_at = ? WHERE id = ?", s.base.Add(time.Hour), episode.ID)
	s.Require().NoError(err)

	swept := make(chan []ClosedEpisode, 1)
	go func() {
		now := s.base.Add(time.Hour)
		closed, err := s.store.CloseIdleEpisodes(s.ctx, now.Add(-time.Hour), now, 50, now.Add(5*time.Minute))
		s.NoError(err)
		swept <- closed
	}()
	// A sweep that waited for the row would go on once the other router commits: give it
	// the time to reach the row before that.
	var closed []ClosedEpisode
	select {
	case closed = <-swept:
	case <-time.After(200 * time.Millisecond):
		s.Require().NoError(closing.Commit())
		closed = <-swept
	}
	_ = closing.Rollback()

	s.Empty(closed, "the episode was closed twice")
}

// A call id is unique only in its app: the event from one app ends that app's call only.
func (s *StoreSuite) TestACallsEpisodesEndWithItInItsOwnAppOnly() {
	person := s.mapped("+15550100")
	call := func(callID, session string) *Episode {
		episode := s.episodeOf(person, "agent:"+callID)
		episode.Source, episode.CallID, episode.SessionID = EpisodeCall, callID, session
		_, err := s.store.OpenEpisode(s.ctx, episode)
		s.Require().NoError(err)
		return episode
	}
	first, second, other := call("call-one", "session-one"), call("call-one", "session-two"), call("call-two", "session-three")
	now := s.base.Add(10 * time.Minute)

	elsewhere, err := s.store.EndCallEpisodes(s.ctx, AppScope{App: 7}, "call-one", now, now.Add(5*time.Minute))
	s.Require().NoError(err)
	s.Empty(elsewhere, "another app's call of the same id")
	ended, err := s.store.EndCallEpisodes(s.ctx, AppScope{App: 4242}, "call-one", now, now.Add(5*time.Minute))
	s.Require().NoError(err)

	s.ElementsMatch([]string{first.ID, second.ID}, []string{ended[0].ID, ended[1].ID})
	s.Equal(episodeInProgress, s.episodeStatus(other.ID))
	again, err := s.store.EndCallEpisodes(s.ctx, AppScope{App: 4242}, "call-one", now, now.Add(5*time.Minute))
	s.Require().NoError(err)
	s.Empty(again, "an event delivered twice ends each episode once")
}

// A router that closed an episode and stopped before its summary leaves it ended; once the
// lease runs out the next sweep takes it, once.
func (s *StoreSuite) TestAnEpisodeWhoseSummaryLeaseRanOutIsTakenAgainOnce() {
	episode := s.threadAt(s.mapped("+15550100"), "agent:thread-one", s.base)
	closedAt := s.base.Add(time.Hour)
	s.closeIdle(s.store, closedAt)

	early, err := s.store.ClaimEpisodeSummaries(s.ctx, closedAt.Add(4*time.Minute), 10, closedAt.Add(time.Hour))
	s.Require().NoError(err)
	s.Empty(early, "its closer still holds it")
	late := closedAt.Add(6 * time.Minute)
	taken, err := s.store.ClaimEpisodeSummaries(s.ctx, late, 10, late.Add(5*time.Minute))
	s.Require().NoError(err)
	s.Require().Len(taken, 1)
	s.Equal(episode.ID, taken[0].ID)
	s.Equal("agent:omni-+15550100", taken[0].ConversationID)
	again, err := s.store.ClaimEpisodeSummaries(s.ctx, late, 10, late.Add(5*time.Minute))
	s.Require().NoError(err)
	s.Empty(again, "the new lease holds it")
}

func (s *StoreSuite) TestAFinishedSummaryIsNeverTakenAgain() {
	episode := s.threadAt(s.mapped("+15550100"), "agent:thread-one", s.base)
	s.closeIdle(s.store, s.base.Add(time.Hour))

	s.Require().NoError(s.store.FinishEpisodeSummary(s.ctx, "acme-app", episode.ID, EpisodeSummarized))

	s.Equal(EpisodeSummarized, s.episodeStatus(episode.ID))
	taken, err := s.store.ClaimEpisodeSummaries(s.ctx, s.base.Add(48*time.Hour), 10, s.base.Add(49*time.Hour))
	s.Require().NoError(err)
	s.Empty(taken)
}

// Only an ended episode is finished: one still in progress has no summary to finish.
func (s *StoreSuite) TestASummaryFinishesOnlyAnEndedEpisode() {
	episode := s.threadAt(s.mapped("+15550100"), "agent:thread-one", s.base)

	s.Require().NoError(s.store.FinishEpisodeSummary(s.ctx, "acme-app", episode.ID, EpisodeSummaryFailed))

	s.Equal(episodeInProgress, s.episodeStatus(episode.ID))
}

func (s *StoreSuite) TestAnyEpisodeCardsIsWhetherAConfigTurnedThemOn() {
	off := &AgentConfig{CustomerID: "acme-app", Name: "plain"}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, off))
	any, err := s.store.AnyEpisodeCards(s.ctx)
	s.Require().NoError(err)
	s.False(any)

	on := &AgentConfig{CustomerID: "acme-app", Name: "carded", EpisodeCards: true}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, on))
	any, err = s.store.AnyEpisodeCards(s.ctx)
	s.Require().NoError(err)
	s.True(any)

	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, "acme-app", on.ID))
	any, err = s.store.AnyEpisodeCards(s.ctx)
	s.Require().NoError(err)
	s.False(any, "a deleted config writes no card")
}

// episodeRows is how many episodes there are.
func (s *StoreSuite) episodeRows() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM episodes").Scan(&count))
	return count
}
