//go:build integration

package main

import (
	"context"
	"log/slog"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// EpisodeSweeperSuite is when the router starts the idle sweeper (D3, Kanat 2026-10-07): only
// with connectors on or an agent config with episode_cards on. Each test holds a text episode
// idle for longer than the closer's hour, on a database of the suite's own, and a closer that
// would sweep it every few milliseconds once started.
type EpisodeSweeperSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	store    *store.Store
	chat     *chattest.Server
	model    *countedModel
	closer   *omnichannel.Closer
	episode  string
}

func TestEpisodeSweeperSuite(t *testing.T) {
	suite.Run(t, new(EpisodeSweeperSuite))
}

func (s *EpisodeSweeperSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = testenv.Database(dsn, "episode_sweeper")
	opened, err := openStore(s.ctx, s.settings)
	s.Require().NoError(err)
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *EpisodeSweeperSuite) SetupTest() {
	// The gate reads every customer's configs, so no earlier test's may have the cards on,
	// and no earlier episode may be left to sweep. The database is the suite's own.
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = false")
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(s.ctx,
		"UPDATE episodes SET status = 'summary_failed', summary_lease_until = NULL WHERE status IN ('in_progress', 'ended')")
	s.Require().NoError(err)
	s.settings.Connectors.Enabled = false

	s.chat = chattest.NewServer(s.T())
	apps := streamapp.NewClients(streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: s.chat.URL,
	}), streamapp.ClientsOptions{})
	s.model = &countedModel{}
	registry := llmrouter.NewRegistry()
	registry.Register("counted", func(routing.Spec) (llmrouter.Provider, error) { return s.model, nil })
	router, err := llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "counted", Model: "counted-model", Languages: []string{"en"}}},
		Aliases:   map[string]routing.Alias{"llm-fast": {Languages: []string{"en"}}},
	}, Registry: registry, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	s.closer, err = omnichannel.NewCloser(omnichannel.CloserOptions{
		Store: s.store, Stream: apps, LLM: router, IdleAfter: time.Hour, Every: 10 * time.Millisecond,
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.closer.Close)

	customer := "sweep-" + uuid.NewString()
	contact := store.ContactMapEntry{
		CustomerID: customer, AgentConfigID: "agent-" + uuid.NewString(), Kind: store.ContactSlack,
		Address: "T1:U1", ConversationID: "agent:omni-" + uuid.NewString(),
	}
	_, err = s.store.MapContact(s.ctx, &contact)
	s.Require().NoError(err)
	episode := store.Episode{
		CustomerID: customer, ContactID: contact.ID, Source: store.EpisodeSlack,
		ThreadChannel: "agent:thread-" + uuid.NewString(), StartedAt: time.Now().Add(-2 * time.Hour),
	}
	_, err = s.store.OpenEpisode(s.ctx, &episode)
	s.Require().NoError(err)
	s.episode = episode.ID
}

// status is the test's episode's status as stored.
func (s *EpisodeSweeperSuite) status() string {
	var status string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT status FROM episodes WHERE id = ?", s.episode).Scan(&status))
	return status
}

// Staging runs with connectors off and no config has the cards on: the sweeper never starts,
// so an idle episode stays as it is, nothing is written and Stream and the LLM are asked
// nothing.
func (s *EpisodeSweeperSuite) TestWithNeitherConnectorsNorCardsTheSweeperNeverStarts() {
	started, err := startEpisodeSweeper(s.ctx, s.settings, s.store, s.closer)

	s.Require().NoError(err)
	s.False(started)
	s.Never(func() bool { return s.status() != "in_progress" }, 300*time.Millisecond, 10*time.Millisecond,
		"an idle episode was closed")
	s.Empty(s.chat.Requests("deploy-key"), "Stream was asked something")
	s.Zero(s.model.asked.Load(), "the LLM was asked something")
}

func (s *EpisodeSweeperSuite) TestWithConnectorsOnTheSweeperClosesAnIdleEpisode() {
	s.settings.Connectors.Enabled = true

	started, err := startEpisodeSweeper(s.ctx, s.settings, s.store, s.closer)

	s.Require().NoError(err)
	s.True(started)
	s.Eventually(func() bool { return s.status() != "in_progress" }, 5*time.Second, 10*time.Millisecond)
}

func (s *EpisodeSweeperSuite) TestWithAConfigThatTurnedCardsOnTheSweeperClosesAnIdleEpisode() {
	carded := &store.AgentConfig{CustomerID: "sweep-" + uuid.NewString(), Name: "carded", EpisodeCards: true}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, carded))

	started, err := startEpisodeSweeper(s.ctx, s.settings, s.store, s.closer)

	s.Require().NoError(err)
	s.True(started)
	s.Eventually(func() bool { return s.status() != "in_progress" }, 5*time.Second, 10*time.Millisecond)
}

// countedModel answers every request with a line and counts them.
type countedModel struct{ asked atomic.Int64 }

func (m *countedModel) Start(context.Context) error { return nil }

func (m *countedModel) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	m.asked.Add(1)
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: m.Provider(), Model: m.Model()})
	script.OutputText("A summary.")
	script.Done()
	return script.Stream(), nil
}

func (m *countedModel) Provider() string               { return "counted" }
func (m *countedModel) Model() string                  { return "counted-model" }
func (m *countedModel) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (m *countedModel) Close() error                   { return nil }
