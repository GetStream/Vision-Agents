//go:build integration

package store

import "sync"

// contact is a contact map row of the acme-app customer's agent for a phone number, which a
// test changes one part of.
func contact(address, conversationID string) *ContactMapEntry {
	return &ContactMapEntry{
		CustomerID:     "acme-app",
		AgentConfigID:  "agent-one",
		Kind:           ContactPhone,
		Address:        address,
		ConversationID: conversationID,
		StreamAppPK:    4242,
	}
}

func (s *StoreSuite) TestANewNumberMakesOneContactRow() {
	created, err := s.store.MapContact(s.ctx, contact("+15550100", "agent:omni-one"))

	s.Require().NoError(err)
	s.True(created)
	s.Equal(1, s.contactRows())
}

// An SMS and a call from one number are one person: the second sighting finds the first
// one's omni-channel and pin, whatever it proposed.
func (s *StoreSuite) TestTheSameNumberAgainKeepsItsOmniChannel() {
	_, err := s.store.MapContact(s.ctx, contact("+15550100", "agent:omni-one"))
	s.Require().NoError(err)
	again := contact("+15550100", "agent:omni-two")
	again.StreamAppPK = 7

	created, err := s.store.MapContact(s.ctx, again)

	s.Require().NoError(err)
	s.False(created)
	s.Equal("agent:omni-one", again.ConversationID)
	s.Equal(int64(4242), again.StreamAppPK)
	s.Equal(1, s.contactRows())
}

// The same person writing to two agents is two conversations, as channel_identities has it.
func (s *StoreSuite) TestTheSameNumberForAnotherAgentIsAnotherContact() {
	_, err := s.store.MapContact(s.ctx, contact("+15550100", "agent:omni-one"))
	s.Require().NoError(err)
	other := contact("+15550100", "agent:omni-two")
	other.AgentConfigID = "agent-two"

	created, err := s.store.MapContact(s.ctx, other)

	s.Require().NoError(err)
	s.True(created)
}

func (s *StoreSuite) TestTheSameNumberOfAnotherCustomerIsAnotherContact() {
	_, err := s.store.MapContact(s.ctx, contact("+15550100", "agent:omni-one"))
	s.Require().NoError(err)
	other := contact("+15550100", "agent:omni-two")
	other.CustomerID = "globex-app"

	created, err := s.store.MapContact(s.ctx, other)

	s.Require().NoError(err)
	s.True(created)
}

func (s *StoreSuite) TestAContactNeedsAnAddressAndAConversation() {
	_, err := s.store.MapContact(s.ctx, contact("", "agent:omni-one"))
	s.ErrorContains(err, "a kind, an address and a conversation are required")

	_, err = s.store.MapContact(s.ctx, contact("+15550100", ""))
	s.ErrorContains(err, "a kind, an address and a conversation are required")
}

// Two routers seeing a new number at once, an SMS and a call say, make one row and hand both
// the same omni-channel: the unique index on the address decides which proposal wins.
func (s *StoreSuite) TestTwoFirstSightingsOfOneAddressAtOnceAreOneRow() {
	pools := s.pools(8)
	var wg sync.WaitGroup
	omniChannels := make([]string, len(pools))
	for i, pool := range pools {
		wg.Add(1)
		go func() {
			defer wg.Done()
			entry := contact("+15550100", "agent:omni-"+newID())
			_, err := pool.MapContact(s.ctx, entry)
			s.NoError(err)
			omniChannels[i] = entry.ConversationID
		}()
	}
	wg.Wait()

	for _, omni := range omniChannels {
		s.Equal(omniChannels[0], omni)
	}
	s.Equal(1, s.contactRows())
}

// pools are count stores on the suite's database, each its own connection pool, as count
// routers would be. The suite's own store is the first.
func (s *StoreSuite) pools(count int) []*Store {
	pools := []*Store{s.store}
	for len(pools) < count {
		pool, err := Open(s.dsn)
		s.Require().NoError(err)
		s.T().Cleanup(func() { _ = pool.Close() })
		pools = append(pools, pool)
	}
	return pools
}

// contactRows is how many contact map rows there are.
func (s *StoreSuite) contactRows() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM contact_map").Scan(&count))
	return count
}
