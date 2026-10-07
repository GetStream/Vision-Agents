package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginmigrate"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const pluginsUsage = `usage: router plugins migrate [--apply] [--customer id] [--include-rotating]

  migrate   move the plugin rows onto connectors: each app's plugin OAuth client
            (agent_plugin_clients) onto connector_oauth_clients, each plugin login
            (agent_plugin_connections) onto an oauth2_code connection with its tokens
            sealed, each agent_plugins entry onto a fixed binding and each user_plugins
            entry onto a session binding. The plugin tables are only read. Event
            subscriptions are not moved. Prints every row and what it becomes.
    --apply  write. Without it nothing is written and the plan is printed. Needs
             connectors.enabled: a binding wins over its plugin entry in a session, so
             a binding on a deployment with connectors off would leave the agent
             without the plugin's tools.
    --customer  move only this app's rows (its customer id). Empty moves every app's.
    --include-rotating  also move a grant whose connector rotates refresh tokens. The
             plugin row keeps its copy, and whichever side renews first retires the
             other's, so they are skipped unless asked for.

  It never migrates the database or seeds the built-in connectors: a database behind
  this binary is refused, and a connector not seeded yet is a skipped row.
`

// runPlugins looks after the plugin rows. Only a person runs it: nothing in serve calls it.
func runPlugins(args []string, settings config.Config, logger *slog.Logger) error {
	if len(args) == 0 {
		return errors.New(pluginsUsage)
	}
	if args[0] != "migrate" {
		return fmt.Errorf("unknown plugins command %q\n\n%s", args[0], pluginsUsage)
	}
	flags := flag.NewFlagSet("plugins migrate", flag.ContinueOnError)
	apply := flags.Bool("apply", false, "write; without it the plan is printed and nothing is written")
	customer := flags.String("customer", "", "move only this app's rows; empty moves every app's")
	includeRotating := flags.Bool("include-rotating", false, "also move grants of connectors that rotate refresh tokens")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	return migratePlugins(context.Background(), settings, logger, pluginmigrate.Options{Customer: *customer, IncludeRotating: *includeRotating}, *apply, os.Stdout)
}

// migratePlugins builds what the router builds for connectors, over settings' database and
// keyring, and runs the move once.
func migratePlugins(ctx context.Context, settings config.Config, logger *slog.Logger, scope pluginmigrate.Options, apply bool, out io.Writer) error {
	if apply && !settings.Connectors.Enabled {
		return errors.New("router plugins migrate --apply needs connectors.enabled: a binding wins over " +
			"its plugin entry, so with connectors off a session would have neither")
	}
	// Opened, not openStore: that migrates and seeds the built-ins, and this command writes
	// nothing a dry run does not show, so a binary newer than the database must not move its
	// schema. serve does that, when this build is deployed.
	if settings.Postgres.DSN == "" {
		return errors.New("this needs a database: set postgres.dsn")
	}
	pgStore, err := store.Open(settings.Postgres.DSN)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	pending, err := pgStore.PendingMigrations(ctx)
	if err != nil {
		return err
	}
	if len(pending) > 0 {
		return fmt.Errorf("the database is behind this binary: %d migrations not applied (the newest %d); "+
			"deploy this build first, which migrates it, then run this again", len(pending), pending[len(pending)-1])
	}
	secrets, err := loadKeyring(settings, "router plugins migrate needs")
	if err != nil {
		return err
	}
	configs, err := appconfig.New(appconfig.Options{Store: pgStore, Address: settings.Redis.Addr,
		Username: settings.Redis.Username, Password: settings.Redis.Password, Logger: logger})
	if err != nil {
		return err
	}
	defer configs.Close()

	clients := api.ConnectorClients(pgStore, secrets, os.Getenv)
	// The registry serve builds with connectors on. A dry run reads it on a deployment with
	// them off too: it writes nothing.
	enabled := settings
	enabled.Connectors.Enabled = true
	registry, err := newConnectorRegistry(enabled, clients)
	if err != nil {
		return err
	}
	resolver, err := newConnectorResolver(registry, pgStore, secrets)
	if err != nil {
		return err
	}
	transports, err := newConnectorTransports(resolver)
	if err != nil {
		return err
	}
	credentials, err := pgsealed.New(pgStore, secrets)
	if err != nil {
		return err
	}
	report, err := pluginmigrate.Run(ctx, pluginmigrate.Options{
		Customer:        scope.Customer,
		IncludeRotating: scope.IncludeRotating,
		Store:           pgStore,
		Configs:         configs,
		Registry:        registry,
		Credentials:     credentials,
		Transports:      transports,
		HTTP:            egress.NewClient(connectorHTTPTimeout, nil),
		PublicEndpoint:  egress.ValidatePublicHTTPSURL,
		Clients:         clients,
		PluginClients:   session.PluginClients(pgStore, secrets),
		Getenv:          os.Getenv,
		SealClientSecret: func(customerID, connectorID, secret string) ([]byte, int, error) {
			return api.SealConnectorOAuthClientSecret(secrets, customerID, connectorID, secret)
		},
	}, apply)
	if err != nil {
		return err
	}
	return report.Write(out)
}
