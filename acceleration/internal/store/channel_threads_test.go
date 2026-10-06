//go:build integration

package store

import (
	"errors"
	"sync"
	"time"
)

// thread is a link for a Slack-shaped thread of the acme-app customer, which a test changes
// one part of.
func thread(channelID, threadKey string) *ChannelThread {
	return &ChannelThread{
		ChannelID:      channelID,
		CustomerID:     "acme-app",
		ConnectorID:    "acme",
		ProviderUnitID: "T0000TEAM",
		ThreadKey:      threadKey,
		ConnectionID:   "connection-one",
		ThreadParts:    map[string]string{"channel": "C0000CHAN", "thread_ts": "1759740000.000100"},
		StreamAppPK:    4242,
	}
}

// account gives a connection the account id a consent would.
func (s *StoreSuite) account(id, accountID string) {
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE connector_connections SET account_id = ? WHERE id = ?", accountID, id)
	s.Require().NoError(err)
}

func (s *StoreSuite) TestTheFirstMessageOfAThreadLinksItsChannel() {
	linked := thread("thread-one", "C0000CHAN:1759740000.000100")

	created, err := s.store.LinkChannelThread(s.ctx, linked)

	s.Require().NoError(err)
	s.True(created)
	found, err := s.store.ChannelThread(s.ctx, "thread-one")
	s.Require().NoError(err)
	s.Equal("C0000CHAN:1759740000.000100", found.ThreadKey)
	s.Equal(map[string]string{"channel": "C0000CHAN", "thread_ts": "1759740000.000100"}, found.ThreadParts)
	s.Equal(int64(4242), found.StreamAppPK)
	s.Equal("connection-one", found.ConnectionID)
}

func (s *StoreSuite) TestASecondMessageOfAThreadKeepsItsChannelAndTakesTheLatestConnection() {
	_, err := s.store.LinkChannelThread(s.ctx, thread("thread-one", "C0000CHAN:1"))
	s.Require().NoError(err)
	again := thread("thread-two", "C0000CHAN:1")
	again.ConnectionID = "connection-two"

	created, err := s.store.LinkChannelThread(s.ctx, again)

	s.Require().NoError(err)
	s.False(created)
	s.Equal("thread-one", again.ChannelID, "the thread keeps the channel it was first written to")
	found, err := s.store.ChannelThread(s.ctx, "thread-one")
	s.Require().NoError(err)
	s.Equal("connection-two", found.ConnectionID)
	_, err = s.store.ChannelThread(s.ctx, "thread-two")
	s.ErrorIs(err, ErrNoChannelThread)
}

func (s *StoreSuite) TestAnotherThreadGetsAnotherChannel() {
	_, err := s.store.LinkChannelThread(s.ctx, thread("thread-one", "C0000CHAN:1"))
	s.Require().NoError(err)

	created, err := s.store.LinkChannelThread(s.ctx, thread("thread-two", "C0000CHAN:2"))

	s.Require().NoError(err)
	s.True(created)
}

// One Slack workspace can install two customers' apps, so the same thread is two threads.
func (s *StoreSuite) TestTheSameThreadOfAnotherCustomerIsAThreadOfItsOwn() {
	_, err := s.store.LinkChannelThread(s.ctx, thread("thread-one", "C0000CHAN:1"))
	s.Require().NoError(err)
	other := thread("thread-two", "C0000CHAN:1")
	other.CustomerID = "other-app"

	created, err := s.store.LinkChannelThread(s.ctx, other)

	s.Require().NoError(err)
	s.True(created)
	s.Equal("thread-two", other.ChannelID)
}

func (s *StoreSuite) TestTwoFirstMessagesOfOneThreadAtOnceLinkOneChannel() {
	var wg sync.WaitGroup
	channels := make([]string, 8)
	for i := range channels {
		wg.Add(1)
		go func() {
			defer wg.Done()
			linked := thread("thread-"+newID(), "C0000CHAN:1")
			_, err := s.store.LinkChannelThread(s.ctx, linked)
			s.NoError(err)
			channels[i] = linked.ChannelID
		}()
	}
	wg.Wait()

	for _, channel := range channels {
		s.Equal(channels[0], channel)
	}
}

func (s *StoreSuite) TestAChannelNoThreadIsLinkedToIsNoThread() {
	_, err := s.store.ChannelThread(s.ctx, "plain-conversation")

	s.ErrorIs(err, ErrNoChannelThread)
}

func (s *StoreSuite) TestARetriedMessageIsClaimedOnce() {
	_, err := s.store.LinkChannelThread(s.ctx, thread("thread-one", "C0000CHAN:1"))
	s.Require().NoError(err)

	first, err := s.store.ClaimChannelThreadMessage(s.ctx, "thread-one", "1759740000.000200")
	s.Require().NoError(err)
	again, err := s.store.ClaimChannelThreadMessage(s.ctx, "thread-one", "1759740000.000200")
	s.Require().NoError(err)
	next, err := s.store.ClaimChannelThreadMessage(s.ctx, "thread-one", "1759740000.000300")
	s.Require().NoError(err)

	s.True(first)
	s.False(again)
	s.True(next)
}

func (s *StoreSuite) TestAMessageIdOlderThanTheKeepIsForgotten() {
	_, err := s.store.LinkChannelThread(s.ctx, thread("thread-one", "C0000CHAN:1"))
	s.Require().NoError(err)
	_, err = s.store.ClaimChannelThreadMessage(s.ctx, "thread-one", "old")
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE channel_thread_messages SET created_at = ?", time.Now().Add(-channelMessageKeep-time.Minute))
	s.Require().NoError(err)

	_, err = s.store.ClaimChannelThreadMessage(s.ctx, "thread-one", "new")
	s.Require().NoError(err)

	var kept int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM channel_thread_messages").Scan(&kept))
	s.Equal(1, kept)
}

func (s *StoreSuite) TestTheAppConnectionOfAnAccountIsFound() {
	bot := s.connection("acme-app", nil)
	s.account(bot.ID, "T0000TEAM")
	user := s.connection("acme-app", userOwned("alice"))
	s.account(user.ID, "T0000TEAM")

	found, err := s.store.AppConnectionByAccount(s.ctx, "acme-app", "acme", "T0000TEAM")

	s.Require().NoError(err)
	s.Equal(bot.ID, found.ID, "a user's connection is never the app's")
}

func (s *StoreSuite) TestAnotherCustomersAppConnectionOfTheSameAccountIsNotFound() {
	bot := s.connection("other-app", nil)
	s.account(bot.ID, "T0000TEAM")

	_, err := s.store.AppConnectionByAccount(s.ctx, "acme-app", "acme", "T0000TEAM")

	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestADeletedAppConnectionIsNotFound() {
	bot := s.connection("acme-app", nil)
	s.account(bot.ID, "T0000TEAM")
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", bot.ID))

	_, err := s.store.AppConnectionByAccount(s.ctx, "acme-app", "acme", "T0000TEAM")

	s.True(errors.Is(err, ErrNoConnectorConnection))
}

func (s *StoreSuite) TestTheConfigsThatBindAConnectionAreFound() {
	bot := s.connection("acme-app", nil)
	first := s.bind("acme-app", fixedBinding(bot.ID))
	second := s.bind("acme-app", fixedBinding(bot.ID))
	s.bind("acme-app", fixedBinding("another-connection"))
	s.bind("other-app", fixedBinding(bot.ID))

	configs, err := s.store.AgentConfigsBindingConnection(s.ctx, "acme-app", bot.ID)

	s.Require().NoError(err)
	s.Require().Len(configs, 2)
	s.Equal(first, configs[0].ID)
	s.Equal(second, configs[1].ID)
}
