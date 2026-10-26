//go:build integration

package store

import (
	"context"
	"errors"
	"time"
)

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

// allowAll is an AddConfigPlugin check that allows every config.
func allowAll(context.Context, AgentConfig, Definitions) error { return nil }

// TestAnEditCommittedWhileAPluginIsNamedIsWhatItsCheckSees: another writer holds the config
// row with an uncommitted edit; AddConfigPlugin waits for its lock, and once the edit commits,
// the check is asked about the config with the edit, and both the edit and the plugin are kept.
func (s *StoreSuite) TestAnEditCommittedWhileAPluginIsNamedIsWhatItsCheckSees() {
	config := s.boundConfig("acme-app")
	tx, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	// A failed wait below still lets the naming finish; after the commit this does nothing.
	defer func() { _ = tx.Rollback() }()
	_, err = tx.ExecContext(s.ctx, "UPDATE agent_configs SET instructions = 'edited meanwhile' WHERE id = ?", config.ID)
	s.Require().NoError(err)

	checked := make(chan string, 1)
	done := make(chan error, 1)
	go func() {
		_, err := s.store.AddConfigPlugin(s.ctx, "acme-app", config.ID, "linear", func(_ context.Context, seen AgentConfig, _ Definitions) error {
			checked <- seen.Instructions
			return nil
		})
		done <- err
	}()
	s.Require().Eventually(func() bool {
		var waiting int
		s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
			"SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'").Scan(&waiting))
		return waiting > 0
	}, 5*time.Second, 10*time.Millisecond, "the naming waits on the edit's row lock")
	s.Require().NoError(tx.Commit())
	s.Require().NoError(<-done)

	s.Require().Len(checked, 1, "the naming asks its check")
	s.Equal("edited meanwhile", <-checked)
	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Equal("edited meanwhile", read.Instructions)
	s.Equal([]PluginEntry{{Name: "linear"}}, read.Plugins)
}

// A plugin the check refuses is not named, and the check's error is returned as it is.
func (s *StoreSuite) TestAPluginItsCheckRefusesIsNotNamed() {
	config := s.boundConfig("acme-app")
	refused := errors.New("not beside a sensitive connector")

	_, err := s.store.AddConfigPlugin(s.ctx, "acme-app", config.ID, "linear", func(context.Context, AgentConfig, Definitions) error { return refused })

	s.ErrorIs(err, refused)
	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Empty(read.Plugins)
}

// A plugin is named once, and one each end user connects becomes the app's, keeping its
// options; the check is asked about the config as it is written.
func (s *StoreSuite) TestAPluginIsNamedOnceAndAnEndUsersBecomesTheApps() {
	config := &AgentConfig{CustomerID: "acme-app", Name: "agent-" + newID(),
		Plugins: []PluginEntry{{Name: "google_calendar", User: true, Scopes: []string{"calendar.readonly"}}}}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))
	var seen []PluginEntry
	check := func(_ context.Context, config AgentConfig, _ Definitions) error {
		seen = config.Plugins
		return nil
	}

	_, err := s.store.AddConfigPlugin(s.ctx, "acme-app", config.ID, "google_calendar", check)
	s.Require().NoError(err)
	s.Equal([]PluginEntry{{Name: "google_calendar", Scopes: []string{"calendar.readonly"}}}, seen)
	_, err = s.store.AddConfigPlugin(s.ctx, "acme-app", config.ID, "linear", check)
	s.Require().NoError(err)
	_, err = s.store.AddConfigPlugin(s.ctx, "acme-app", config.ID, "linear", allowAll)
	s.Require().NoError(err)

	read, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	want := []PluginEntry{{Name: "google_calendar", Scopes: []string{"calendar.readonly"}}, {Name: "linear"}}
	s.Equal(want, seen)
	s.Equal(want, read.Plugins)
}

// A plugin is named only on a live config of the customer's.
func (s *StoreSuite) TestAPluginIsNamedOnlyOnALiveConfigOfItsCustomer() {
	deleted := s.boundConfig("acme-app")
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, "acme-app", deleted.ID))
	others := s.boundConfig("other-app")

	for _, id := range []string{deleted.ID, others.ID, newID()} {
		_, err := s.store.AddConfigPlugin(s.ctx, "acme-app", id, "linear", allowAll)
		s.ErrorIs(err, ErrNoAgentConfig, "deleted, another customer's, never made")
	}
	read, err := s.store.AgentConfig(s.ctx, "other-app", others.ID)
	s.Require().NoError(err)
	s.Empty(read.Plugins)
}
