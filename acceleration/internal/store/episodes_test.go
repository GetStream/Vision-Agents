//go:build integration

package store

import "sync"

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

// episodeRows is how many episodes there are.
func (s *StoreSuite) episodeRows() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM episodes").Scan(&count))
	return count
}
