//go:build integration

package store

// A config keeps the floor control it was given, an update writes it, and one that chose none
// is NULL, the router's default, and reads back as none.
func (s *StoreSuite) TestAConfigKeepsItsFloorControl() {
	acting := FloorControl{Model: FloorModelChorusASR, Mode: FloorModeAct}
	config := &AgentConfig{CustomerID: "app", Name: "ruled", Mode: AgentModeVoice, FloorControl: acting}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))
	stored, err := s.store.AgentConfig(s.ctx, "app", config.ID)
	s.Require().NoError(err)
	s.Equal(acting, stored.FloorControl)

	config.FloorControl = FloorControl{Model: FloorModelChorusASR, Mode: FloorModeOff}
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, config))
	stored, err = s.store.AgentConfig(s.ctx, "app", config.ID)
	s.Require().NoError(err)
	s.Equal(FloorControl{Model: FloorModelChorusASR, Mode: FloorModeOff}, stored.FloorControl)

	config.FloorControl = FloorControl{}
	s.Require().NoError(s.store.UpdateAgentConfig(s.ctx, config))
	stored, err = s.store.AgentConfig(s.ctx, "app", config.ID)
	s.Require().NoError(err)
	s.True(stored.FloorControl.IsZero())
	var unset bool
	s.Require().NoError(s.store.DB().NewSelect().Table("agent_configs").ColumnExpr("floor_control IS NULL").
		Where("id = ?", config.ID).Scan(s.ctx, &unset))
	s.True(unset, "the router's default is stored as NULL")

	unchosen := &AgentConfig{CustomerID: "app", Name: "unruled", Mode: AgentModeVoice}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, unchosen))
	stored, err = s.store.AgentConfig(s.ctx, "app", unchosen.ID)
	s.Require().NoError(err)
	s.True(stored.FloorControl.IsZero())
}

// A session keeps the floor control it was opened with, and one opened with none reads back
// none.
func (s *StoreSuite) TestASessionKeepsTheFloorControlItWasOpenedWith() {
	shadow := FloorControl{Model: FloorModelChorusASR, Mode: FloorModeShadow}
	s.opened("chosen", "app", s.base, func(session *AgentSession) { session.FloorControl = shadow })
	s.opened("default", "app", s.base, nil)

	chosen, err := s.store.StoredSession(s.ctx, "app", "chosen")
	s.Require().NoError(err)
	s.Equal(shadow, chosen.FloorControl)
	unchosen, err := s.store.StoredSession(s.ctx, "app", "default")
	s.Require().NoError(err)
	s.True(unchosen.FloorControl.IsZero())
}
