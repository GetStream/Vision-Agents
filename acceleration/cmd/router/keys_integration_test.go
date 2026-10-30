//go:build integration

package main

import (
	"context"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// KeysCLISuite mints keys with `router keys create` and authenticates with them as the
// router does, under whichever keyring the environment names at the time.
type KeysCLISuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	store    *store.Store
}

func TestKeysCLISuite(t *testing.T) {
	suite.Run(t, new(KeysCLISuite))
}

func (s *KeysCLISuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "router_keys")
	opened, err := openStore(s.ctx, s.settings)
	s.Require().NoError(err)
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *KeysCLISuite) SetupTest() {
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_V2", "second-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "2")
}

// created mints a key for the app with the command and returns its row.
func (s *KeysCLISuite) created(app string) store.APIKey {
	s.Require().NoError(runKeys([]string{"create", "--app-id", app, "--env", "test"}, s.settings, slog.New(slog.DiscardHandler)))
	var minted store.APIKey
	s.Require().NoError(s.store.DB().NewSelect().Model(&minted).
		Where("app_id = ?", app).Order("created_at DESC").Limit(1).Scan(s.ctx))
	return minted
}

// lookup authenticates the key under the keyring the environment names now.
func (s *KeysCLISuite) lookup(key string) (auth.App, error) {
	sealer, err := loadKeyring(s.ctx, s.settings, "the suite needs")
	s.Require().NoError(err)
	configs, err := appconfig.New(appconfig.Options{Store: s.store, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	return configs.Lookup(sealer)(s.ctx, key)
}

func (s *KeysCLISuite) TestAKeyStillOpensOnceTheKeyVersionIsRaised() {
	// Raising the version is how a key encryption key is rotated, keeping the old ones.
	minted := s.created("app-" + uuid.NewString())
	s.T().Setenv("ROUTER_AUTH_KEK_V3", "third-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "3")

	app, err := s.lookup(minted.ID)

	s.Require().NoError(err)
	s.Equal(minted.AppID, app.AppID)
}

func (s *KeysCLISuite) TestAKeysSealedSecretCopiedIntoAnotherKeysRowDoesNotOpen() {
	app := "app-" + uuid.NewString()
	first, second := s.created(app), s.created(app)
	_, err := s.store.DB().NewUpdate().Model((*store.APIKey)(nil)).
		Set("secret_sealed = ?", first.Sealed).Where("id = ?", second.ID).Exec(s.ctx)
	s.Require().NoError(err)

	_, err = s.lookup(second.ID)

	s.Error(err)
}
