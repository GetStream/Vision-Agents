//go:build integration

package store

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

// contactRows is how many contact map rows there are.
func (s *StoreSuite) contactRows() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM contact_map").Scan(&count))
	return count
}
