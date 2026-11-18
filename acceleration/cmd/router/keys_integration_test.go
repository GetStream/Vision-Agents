//go:build integration

package main

import (
	"log/slog"
	"os"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// KeysCLISuite runs `router keys` against Postgres.
type KeysCLISuite struct {
	suite.Suite
	settings config.Config
}

func TestKeysCLISuite(t *testing.T) {
	suite.Run(t, new(KeysCLISuite))
}

func (s *KeysCLISuite) SetupTest() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	for _, name := range []string{devKEKEnvVar, "ROUTER_AUTH_KEK", "ROUTER_AUTH_KEK_V1"} {
		s.T().Setenv(name, "")
	}
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "router_cli")
}

func (s *KeysCLISuite) TestCreateRefusesTheKeySourcesTheRouterRefuses() {
	cases := []struct {
		name, kmsKey, want string
		env                map[string]string
	}{
		{"the dev key beside a KMS key", testKMSKey,
			"ROUTER_DEV_KEK is for a laptop or CI and is never set with auth.kms_key (ROUTER_AUTH_KMS_KEY): unset ROUTER_DEV_KEK",
			map[string]string{"ROUTER_DEV_KEK": "first-key"}},
		{"a retired plain key", "",
			"ROUTER_AUTH_KEK is no longer read: set ROUTER_DEV_KEK on a laptop or in CI, or " +
				"ROUTER_AUTH_KEK_V1_WRAPPED with auth.kms_key (ROUTER_AUTH_KMS_KEY) on a hosted router",
			map[string]string{"ROUTER_AUTH_KEK": "first-key"}},
	}
	for _, c := range cases {
		s.Run(c.name, func() {
			settings := s.settings
			settings.Auth.KMSKey = c.kmsKey
			for name, value := range c.env {
				s.T().Setenv(name, value)
			}

			err := runKeys([]string{"create", "--app-id", "refused-app"}, settings, slog.New(slog.DiscardHandler))

			s.EqualError(err, c.want)
		})
	}
}
