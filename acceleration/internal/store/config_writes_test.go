//go:build integration

package store

import (
	"context"
	"errors"
	"time"
)

// onOneConnection runs write on a pool of one connection, which the write's transaction holds,
// so a check that reads anywhere but on that transaction waits until the deadline. Three
// seconds is the probe that found the wait (pool=1, deadline after 3.001s).
func (s *StoreSuite) onOneConnection(write func(ctx context.Context) error) {
	s.seed(acmeManifest)
	was := s.store.DB().Stats().MaxOpenConnections
	s.store.DB().SetMaxOpenConns(1)
	defer s.store.DB().SetMaxOpenConns(was)
	ctx, cancel := context.WithTimeout(s.ctx, 3*time.Second)
	defer cancel()

	s.Require().NoError(write(ctx))
}

// readsAcme is a check that reads acme's definition as SensitiveBindingsComplaint does.
func (s *StoreSuite) readsAcme(read *string) ConfigCheck {
	return func(ctx context.Context, _ AgentConfig, definitions Definitions) error {
		definition, err := definitions.LatestConnectorDefinition(ctx, "acme-app", "acme")
		*read = definition.Manifest.Name
		return err
	}
}

// waitForLockWaiters waits until n statements of this database wait on a lock.
func (s *StoreSuite) waitForLockWaiters(n int, what string) {
	s.Require().Eventually(func() bool {
		var waiting int
		s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
			"SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'").Scan(&waiting))
		return waiting >= n
	}, 5*time.Second, 10*time.Millisecond, what)
}

func (s *StoreSuite) TestAPluginsCheckReadsDefinitionsOnTheWritesConnection() {
	config := s.boundConfig("acme-app")
	var read string

	s.onOneConnection(func(ctx context.Context) error {
		_, err := s.store.AddConfigPlugin(ctx, "acme-app", config.ID, "linear", s.readsAcme(&read))
		return err
	})

	s.Equal("Acme", read)
}

func (s *StoreSuite) TestABindingsCheckReadsDefinitionsOnTheWritesConnection() {
	config := s.boundConfig("acme-app")
	binding := ConnectorBinding{Name: "inbox", ConnectorID: "acme", Connection: ConnectionBinding{Type: "session"}, Tools: []ToolGrant{}}
	var read string

	s.onOneConnection(func(ctx context.Context) error {
		_, _, err := s.store.AddConnectorBinding(ctx, "acme-app", config.ID, binding, s.readsAcme(&read))
		return err
	})

	s.Equal("Acme", read)
}

func (s *StoreSuite) TestAnUpdateButPluginsChecksOnTheWritesConnection() {
	config := s.boundConfig("acme-app")
	var read string

	s.onOneConnection(func(ctx context.Context) error {
		return s.store.UpdateAgentConfigButPlugins(ctx, &config, s.readsAcme(&read))
	})

	s.Equal("Acme", read)
}

// TestAnUpdateButPluginsKeepsPluginsNamedWhileItWaited: the config is read, then another
// writer names a plugin and holds the row until the update waits on it. The update writes
// its own instructions and the plugin, and its check is asked about the config with both.
func (s *StoreSuite) TestAnUpdateButPluginsKeepsPluginsNamedWhileItWaited() {
	created := s.boundConfig("acme-app")
	read, err := s.store.AgentConfig(s.ctx, "acme-app", created.ID)
	s.Require().NoError(err)
	read.Instructions = "be brief"
	tx, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	// A failed wait below still lets the update finish; after the commit this does nothing.
	defer func() { _ = tx.Rollback() }()
	_, err = tx.ExecContext(s.ctx, `UPDATE agent_configs SET plugins = '[{"name":"linear"}]' WHERE id = ?`, created.ID)
	s.Require().NoError(err)

	var checked []PluginEntry
	done := make(chan error, 1)
	go func() {
		done <- s.store.UpdateAgentConfigButPlugins(s.ctx, &read, func(_ context.Context, config AgentConfig, _ Definitions) error {
			checked = config.Plugins
			return nil
		})
	}()
	s.waitForLockWaiters(1, "the update waits on the naming's row lock")
	s.Require().NoError(tx.Commit())
	s.Require().NoError(<-done)

	want := []PluginEntry{{Name: "linear"}}
	s.Equal(want, checked)
	s.Equal(want, read.Plugins, "the config written is handed back")
	stored, err := s.store.AgentConfig(s.ctx, "acme-app", created.ID)
	s.Require().NoError(err)
	s.Equal("be brief", stored.Instructions)
	s.Equal(want, stored.Plugins)
}

// An update but plugins its check refuses writes nothing, and the check's error is returned
// as it is.
func (s *StoreSuite) TestAnUpdateButPluginsItsCheckRefusesWritesNothing() {
	config := s.boundConfig("acme-app")
	config.Instructions = "be brief"
	refused := errors.New("not beside a sensitive connector")

	err := s.store.UpdateAgentConfigButPlugins(s.ctx, &config, func(context.Context, AgentConfig, Definitions) error { return refused })

	s.ErrorIs(err, refused)
	stored, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Empty(stored.Instructions)
}

func (s *StoreSuite) TestAnUpdateButPluginsOfAConfigThatIsGoneIsUnknown() {
	config := s.boundConfig("acme-app")
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, "acme-app", config.ID))

	err := s.store.UpdateAgentConfigButPlugins(s.ctx, &config, allowAll)

	s.ErrorIs(err, ErrNoAgentConfig)
}

// An edit that changes no plugin writes nothing and asks no check; one with no check writes.
func (s *StoreSuite) TestAPluginEditThatChangesNothingWritesNothing() {
	config := s.boundConfig("acme-app")
	asked := false
	check := func(context.Context, AgentConfig, Definitions) error {
		asked = true
		return nil
	}
	keep := func(entries []PluginEntry) ([]PluginEntry, bool) { return entries, false }
	add := func(entries []PluginEntry) ([]PluginEntry, bool) {
		return append(entries, PluginEntry{Name: "linear", User: true}), true
	}

	_, err := s.store.EditConfigPlugins(s.ctx, "acme-app", config.ID, keep, check)
	s.Require().NoError(err)
	unchanged, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	_, err = s.store.EditConfigPlugins(s.ctx, "acme-app", config.ID, add, nil)
	s.Require().NoError(err)

	s.False(asked)
	s.Equal(config.UpdatedAt.Truncate(time.Microsecond), unchanged.UpdatedAt.Truncate(time.Microsecond))
	stored, err := s.store.AgentConfig(s.ctx, "acme-app", config.ID)
	s.Require().NoError(err)
	s.Equal([]PluginEntry{{Name: "linear", User: true}}, stored.Plugins)
}
