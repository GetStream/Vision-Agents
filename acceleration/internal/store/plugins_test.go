//go:build integration

package store

// pluginClient sets an OAuth client on a config, with a sealed secret standing in for a
// real one.
func (s *StoreSuite) pluginClient(customerID, configID, pluginID, clientID string) PluginClient {
	client := PluginClient{
		CustomerID: customerID, ConfigID: configID, PluginID: pluginID, ClientID: clientID,
		SecretSealed: []byte("sealed " + clientID), SecretKEKVersion: 1,
	}
	s.Require().NoError(s.store.SavePluginClient(s.ctx, &client))
	return client
}

func (s *StoreSuite) TestSettingAPluginClientAgainReplacesItAndKeepsWhenItWasFirstSet() {
	customerID, configID := newID(), newID()
	first := s.pluginClient(customerID, configID, "google_calendar", "first")
	s.pluginClient(customerID, configID, "google_calendar", "second")

	held, err := s.store.PluginClient(s.ctx, customerID, configID, "google_calendar")
	s.Require().NoError(err)
	s.Equal("second", held.ClientID)
	s.Equal([]byte("sealed second"), held.SecretSealed)
	s.True(first.CreatedAt.Equal(held.CreatedAt))
}

func (s *StoreSuite) TestAPluginClientIsOnlyItsOwnConfigs() {
	customerID, configID := newID(), newID()
	s.pluginClient(customerID, configID, "google_calendar", "mine")

	_, otherConfig := s.store.PluginClient(s.ctx, customerID, newID(), "google_calendar")
	_, otherCustomer := s.store.PluginClient(s.ctx, newID(), configID, "google_calendar")
	listed, err := s.store.PluginClients(s.ctx, newID(), configID)
	s.Require().NoError(err)

	s.ErrorIs(otherConfig, ErrUnknownPluginClient)
	s.ErrorIs(otherCustomer, ErrUnknownPluginClient)
	s.Empty(listed)
}

func (s *StoreSuite) TestAConfigsPluginClientsAreListedByPlugin() {
	customerID, configID := newID(), newID()
	s.pluginClient(customerID, configID, "google_calendar", "calendar")
	s.pluginClient(customerID, configID, "google_drive", "drive")

	listed, err := s.store.PluginClients(s.ctx, customerID, configID)
	s.Require().NoError(err)

	s.Len(listed, 2)
	s.Equal("calendar", listed["google_calendar"].ClientID)
	s.Equal("drive", listed["google_drive"].ClientID)
}

func (s *StoreSuite) TestADeletedPluginClientIsGoneAndDeletingItAgainSaysSo() {
	customerID, configID := newID(), newID()
	s.pluginClient(customerID, configID, "google_calendar", "gone")

	s.Require().NoError(s.store.DeletePluginClient(s.ctx, customerID, configID, "google_calendar"))
	_, err := s.store.PluginClient(s.ctx, customerID, configID, "google_calendar")

	s.ErrorIs(err, ErrUnknownPluginClient)
	s.ErrorIs(s.store.DeletePluginClient(s.ctx, customerID, configID, "google_calendar"), ErrUnknownPluginClient)
}
