//go:build integration

package store

import "time"

// answering is a live config of the customer naming the channel types given, made at.
func (s *StoreSuite) answering(customerID string, at time.Time, types ...string) string {
	config := &AgentConfig{CustomerID: customerID, Name: "agent-" + newID()}
	for _, named := range types {
		config.ChatChannels = append(config.ChatChannels, ChatChannel{Type: named})
	}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE agent_configs SET created_at = ? WHERE id = ?", at, config.ID)
	s.Require().NoError(err)
	return config.ID
}

// The oldest of the customer's live configs naming the type answers in it: not a newer one,
// another type's, another customer's, a deleted one or a test copy.
func (s *StoreSuite) TestTheOldestConfigNamingAChannelTypeAnswersInIt() {
	customer, other := "acme-"+newID(), "other-"+newID()
	now := time.Now().UTC()
	deleted := s.answering(customer, now.Add(-4*time.Hour), "callouts")
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, customer, deleted))
	copied := s.answering(customer, now.Add(-3*time.Hour), "callouts")
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE agent_configs SET tags = ?::jsonb WHERE id = ?", `{"draft_of": "x"}`, copied)
	s.Require().NoError(err)
	s.answering(other, now.Add(-2*time.Hour), "callouts")
	s.answering(customer, now.Add(-90*time.Minute), "shifts")
	oldest := s.answering(customer, now.Add(-time.Hour), "shifts", "callouts")
	s.answering(customer, now, "callouts")

	config, found, err := s.store.AgentConfigAnsweringChatChannel(s.ctx, customer, "callouts")

	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal(oldest, config.ID)
	s.Equal([]ChatChannel{{Type: "shifts"}, {Type: "callouts"}}, config.ChatChannels)
}

// A type is matched whole: one no config names, or that only begins or ends like one, has none.
func (s *StoreSuite) TestAChannelTypeNoConfigNamesHasNone() {
	customer := "acme-" + newID()
	s.answering(customer, time.Now().UTC(), "callouts")
	s.answering(customer, time.Now().UTC())

	for _, asked := range []string{"call", "callouts2", "Callouts", "agent"} {
		_, found, err := s.store.AgentConfigAnsweringChatChannel(s.ctx, customer, asked)
		s.Require().NoError(err, asked)
		s.False(found, asked)
	}
}
