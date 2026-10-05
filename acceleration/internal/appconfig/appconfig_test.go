//go:build integration

package appconfig

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// settleFor is how long an invalidation has to reach the other node. Redis pushes it as
// soon as the key is deleted, so this is generous rather than expected.
const settleFor = 2 * time.Second

// AppConfigSuite runs two stores over one Postgres and one Redis, which is what two
// replicas of the router are. A change made through one has to be what the other reads,
// because the alternative is a revoked key that still works somewhere for an hour.
type AppConfigSuite struct {
	suite.Suite
	ctx   context.Context
	db    *store.Store
	first *Store
	other *Store
}

func TestAppConfigSuite(t *testing.T) {
	suite.Run(t, new(AppConfigSuite))
}

func (s *AppConfigSuite) SetupSuite() {
	dsn, redisAddr := os.Getenv("ROUTER_POSTGRES_DSN"), os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || redisAddr == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	s.ctx = context.Background()

	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.db = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })

	logger := slog.New(slog.DiscardHandler)
	s.first = s.node(redisAddr, logger)
	s.other = s.node(redisAddr, logger)
}

func (s *AppConfigSuite) node(redisAddr string, logger *slog.Logger) *Store {
	node, err := New(Options{Store: s.db, Address: redisAddr, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(node.Close)
	return node
}

// app writes an organization, an app and a live key, and returns the app and key ids.
func (s *AppConfigSuite) app(with ...func(*store.APIKey)) (string, string) {
	stamp := time.Now().UnixNano()
	organization := store.Organization{Name: fmt.Sprintf("org-%d", stamp)}
	s.Require().NoError(s.db.CreateOrganization(s.ctx, &organization))

	app := store.App{OrganizationID: organization.ID, Name: fmt.Sprintf("app-%d", stamp)}
	s.Require().NoError(s.db.CreateApp(s.ctx, &app))

	credential := store.APIKey{
		ID: fmt.Sprintf("key-%d", stamp), AppID: app.ID, Name: "test", Env: "test",
		Sealed: []byte("sealed"), KEKVersion: 1, Last4: "abcd", CreatedBy: "appconfig test",
	}
	for _, apply := range with {
		apply(&credential)
	}
	s.Require().NoError(s.db.CreateAPIKey(s.ctx, &credential))
	return app.ID, credential.ID
}

func (s *AppConfigSuite) TestAPIKey() {
	s.Run("a key is read back with the app behind it", func() {
		appID, keyID := s.app()

		owner, err := s.first.APIKey(s.ctx, keyID)
		s.Require().NoError(err)
		s.Equal(appID, owner.AppID)
		s.Equal([]byte("sealed"), owner.Sealed)
	})

	s.Run("an unknown key is unknown rather than empty", func() {
		_, err := s.first.APIKey(s.ctx, "key-nobody-minted")
		s.ErrorIs(err, store.ErrNoAPIKey)
	})

	s.Run("a key revoked on one node stops working on the other", func() {
		_, keyID := s.app()

		// Both nodes hold it locally before anything is revoked, which is the state a
		// revocation has to undo rather than one it gets to avoid.
		_, err := s.first.APIKey(s.ctx, keyID)
		s.Require().NoError(err)
		_, err = s.other.APIKey(s.ctx, keyID)
		s.Require().NoError(err)

		s.Require().NoError(s.first.RevokeAPIKey(s.ctx, keyID, "appconfig test"))

		s.Eventually(func() bool {
			_, err := s.other.APIKey(s.ctx, keyID)
			return err != nil
		}, settleFor, 10*time.Millisecond, "the other node still accepts a revoked key")
	})

	s.Run("a key cached while it was live stops working when it lapses", func() {
		_, keyID := s.app(func(key *store.APIKey) {
			lapses := time.Now().UTC().Add(300 * time.Millisecond)
			key.ExpiresAt = &lapses
		})

		_, err := s.first.APIKey(s.ctx, keyID)
		s.Require().NoError(err)

		time.Sleep(400 * time.Millisecond)

		_, err = s.first.APIKey(s.ctx, keyID)
		s.ErrorIs(err, store.ErrNoAPIKey)
	})
}

func (s *AppConfigSuite) TestPolicy() {
	s.Run("a policy saved on one node is read by the other", func() {
		appID, _ := s.app()

		was, err := s.other.Policy(s.ctx, store.ScopeApp, appID)
		s.Require().NoError(err)
		s.Nil(was.AllowedModels)

		allowed := []string{"openai/gpt-5"}
		s.Require().NoError(s.first.SavePolicy(s.ctx, store.ScopeApp, appID,
			store.PolicyDocument{AllowedModels: &allowed}))

		s.Eventually(func() bool {
			now, err := s.other.Policy(s.ctx, store.ScopeApp, appID)
			return err == nil && now.AllowedModels != nil && len(*now.AllowedModels) == 1
		}, settleFor, 10*time.Millisecond, "the other node is still routing on the old policy")
	})

	s.Run("an app moved to another organization is answered under the new one", func() {
		appID, _ := s.app()
		s.Require().NoError(s.first.JoinOrganization(s.ctx, appID, "org-first"))

		under, err := s.other.OrganizationOf(s.ctx, appID)
		s.Require().NoError(err)
		s.Equal("org-first", under)

		s.Require().NoError(s.first.JoinOrganization(s.ctx, appID, "org-second"))

		s.Eventually(func() bool {
			under, err := s.other.OrganizationOf(s.ctx, appID)
			return err == nil && under == "org-second"
		}, settleFor, 10*time.Millisecond, "the other node still has the old organization")
	})
}

func (s *AppConfigSuite) TestAgentConfig() {
	s.Run("a renamed config is no longer found under the name it had", func() {
		appID, _ := s.app()

		config := store.AgentConfig{CustomerID: appID, Name: "support", Mode: "text"}
		s.Require().NoError(s.first.CreateAgentConfig(s.ctx, &config))

		found, exists, err := s.other.AgentConfigByName(s.ctx, appID, "support")
		s.Require().NoError(err)
		s.Require().True(exists)
		s.Equal(config.ID, found.ID)

		config.Name = "sales"
		s.Require().NoError(s.first.UpdateAgentConfig(s.ctx, &config))

		s.Eventually(func() bool {
			_, exists, err := s.other.AgentConfigByName(s.ctx, appID, "support")
			return err == nil && !exists
		}, settleFor, 10*time.Millisecond, "the other node still answers to the old name")

		renamed, exists, err := s.other.AgentConfigByName(s.ctx, appID, "sales")
		s.Require().NoError(err)
		s.Require().True(exists)
		s.Equal(config.ID, renamed.ID)
	})

	s.Run("an edited skill is read with its new instructions", func() {
		appID, _ := s.app()

		config := store.AgentConfig{CustomerID: appID, Name: "refunds", Mode: "text"}
		s.Require().NoError(s.first.CreateAgentConfig(s.ctx, &config))

		skill := store.Skill{
			CustomerID: appID, ConfigID: config.ID,
			Name: "refund", Description: "issue a refund", Instructions: "ask first",
		}
		s.Require().NoError(s.first.CreateSkill(s.ctx, &skill))

		named, err := s.other.SkillsNamed(s.ctx, appID, config.ID, []string{"refund"})
		s.Require().NoError(err)
		s.Require().Len(named, 1)
		s.Equal("ask first", named[0].Instructions)

		skill.Instructions = "refund without asking"
		s.Require().NoError(s.first.UpdateSkill(s.ctx, &skill))

		s.Eventually(func() bool {
			named, err := s.other.SkillsNamed(s.ctx, appID, config.ID, []string{"refund"})
			return err == nil && len(named) == 1 && named[0].Instructions == "refund without asking"
		}, settleFor, 10*time.Millisecond, "the other node is still following the old instructions")
	})
}

func (s *AppConfigSuite) TestWithoutRedis() {
	s.Run("a deployment with no redis reads postgres and still sees its own writes", func() {
		plain, err := New(Options{Store: s.db, Logger: slog.New(slog.DiscardHandler)})
		s.Require().NoError(err)
		defer plain.Close()

		appID, keyID := s.app()
		owner, err := plain.APIKey(s.ctx, keyID)
		s.Require().NoError(err)
		s.Equal(appID, owner.AppID)

		s.Require().NoError(plain.RevokeAPIKey(s.ctx, keyID, "appconfig test"))
		_, err = plain.APIKey(s.ctx, keyID)
		s.ErrorIs(err, store.ErrNoAPIKey)
	})
}
