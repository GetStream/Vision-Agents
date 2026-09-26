package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const keysUsage = `usage: router keys create [flags]

  --app-id      the id to give the app, which is what every row's customer_id holds.
                Pass the id the deployment you are moving from used, so the rows you
                import still belong to something. Empty mints a new one
  --app-name    what to call the app, for a human reading the list
  --org         the organization to put a new app in
  --key-name    what to call the key, so an operator knows which to revoke
  --env         live or test
`

// runKeys mints the first credential a self-hosted router has.
//
// It exists because api_key mode has no way to bootstrap itself over HTTP: every endpoint
// wants a key, and there is nobody to issue the first one. A command run beside the
// database can, and it is the same command that moves an app id across from a deployment
// somebody is leaving.
func runKeys(args []string, settings config.Config, logger *slog.Logger) error {
	if len(args) == 0 {
		return errors.New(keysUsage)
	}
	if args[0] != "create" {
		return fmt.Errorf("unknown keys command %q\n\n%s", args[0], keysUsage)
	}

	flags := flag.NewFlagSet("keys create", flag.ContinueOnError)
	appID := flags.String("app-id", "", "the app id to use, or empty for a new one")
	appName := flags.String("app-name", "", "what to call the app")
	orgName := flags.String("org", "self-hosted", "the organization a new app belongs to")
	keyName := flags.String("key-name", "default", "what to call the key")
	environment := flags.String("env", string(auth.Live), "live or test")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}

	ctx := context.Background()
	pgStore, err := openStore(ctx, settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()

	sealer, err := auth.NewSealer(settings.Auth.KEK)
	if err != nil {
		return fmt.Errorf("minting a key needs auth.kek, which is what seals its secret: %w", err)
	}

	app, err := resolveApp(ctx, pgStore, *appID, *appName, *orgName)
	if err != nil {
		return err
	}

	key, secret, err := auth.NewCredential(auth.Environment(*environment))
	if err != nil {
		return err
	}
	sealed, err := sealer.Seal(secret)
	if err != nil {
		return err
	}
	if err := pgStore.CreateAPIKey(ctx, &store.APIKey{
		ID:         key,
		AppID:      app.ID,
		Name:       *keyName,
		Env:        *environment,
		Sealed:     sealed,
		KEKVersion: auth.KEKVersion,
		Last4:      auth.Last4(secret),
		CreatedBy:  "router keys create",
	}); err != nil {
		return err
	}
	logger.Info("created an api key", "app", app.ID, "key", key)

	// Printed rather than logged: the secret is never shown again, and a log is the one
	// place it must not be. Standard output is what the operator is looking at.
	fmt.Printf("organization  %s\napp           %s\nkey           %s\nsecret        %s\n\n"+
		"The secret is shown once. Set it where your SDK reads it and keep it out of the router's own config.\n",
		app.OrganizationID, app.ID, key, secret)
	return nil
}

// resolveApp finds the app the key is for, creating it and its organization when it is
// not there yet. An id that already exists is used as it is, so running this twice adds a
// second key rather than refusing.
func resolveApp(ctx context.Context, pgStore *store.Store, appID, appName, orgName string) (store.App, error) {
	if appID != "" {
		app, err := pgStore.AppByID(ctx, appID)
		if err == nil {
			return app, nil
		}
		if !errors.Is(err, store.ErrNoApp) {
			return store.App{}, err
		}
	}

	if orgName == "" {
		return store.App{}, errors.New("a new app needs an organization: pass --org")
	}
	org := store.Organization{Name: orgName}
	if err := pgStore.CreateOrganization(ctx, &org); err != nil {
		return store.App{}, err
	}

	name := appName
	if name == "" {
		name = orgName
	}
	app := store.App{ID: appID, OrganizationID: org.ID, Name: name}
	if err := pgStore.CreateApp(ctx, &app); err != nil {
		return store.App{}, err
	}
	return app, nil
}
