//go:build integration

package store

// A config keeps the end-of-turn choice it was given, and an update writes it whole.
func (s *StoreSuite) TestAConfigKeepsItsEndOfTurn() {
	config := &AgentConfig{CustomerID: "app", Name: "eager", Mode: AgentModeVoice, EndOfTurn: EndOfTurn{Preset: "fast"}}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))
	stored, err := s.store.AgentConfig(s.ctx, "app", config.ID)
	s.Require().NoError(err)
	s.Equal(EndOfTurn{Preset: "fast"}, stored.EndOfTurn)

	config.EndOfTurn = EndOfTurn{Bars: []EndOfTurnBar{{FromMs: 0, Bar: 0.9}, {FromMs: 800, Bar: 0.25}}}
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, config))
	stored, err = s.store.AgentConfig(s.ctx, "app", config.ID)
	s.Require().NoError(err)
	s.Equal(EndOfTurn{Bars: []EndOfTurnBar{{FromMs: 0, Bar: 0.9}, {FromMs: 800, Bar: 0.25}}}, stored.EndOfTurn)

	config.EndOfTurn = EndOfTurn{}
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, config))
	stored, err = s.store.AgentConfig(s.ctx, "app", config.ID)
	s.Require().NoError(err)
	s.True(stored.EndOfTurn.IsZero())
}

// A session keeps the choice it was opened with, and one opened with none reads back none.
func (s *StoreSuite) TestASessionKeepsTheEndOfTurnItWasOpenedWith() {
	s.opened("chosen", "app", s.base, func(session *AgentSession) { session.EndOfTurn = EndOfTurn{MaxWaitMs: 900} })
	s.opened("default", "app", s.base, nil)

	chosen, err := s.store.StoredSession(s.ctx, "app", "chosen")
	s.Require().NoError(err)
	s.Equal(EndOfTurn{MaxWaitMs: 900}, chosen.EndOfTurn)
	unchosen, err := s.store.StoredSession(s.ctx, "app", "default")
	s.Require().NoError(err)
	s.True(unchosen.EndOfTurn.IsZero())
}
