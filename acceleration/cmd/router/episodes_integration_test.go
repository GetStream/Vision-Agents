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
// with connectors on, an agent config with episode_cards on, or an episode still to close or
// summarize (AI-2771). Each test runs on a database of the suite's own, with a closer that
// would sweep every few milliseconds once started; most hold a text episode idle for longer
// than the closer's hour.
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
		"UPDATE episodes SET status = 'summary_failed' WHERE status IN ('in_progress', 'ended')")
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE episode_activity SET summary_lease_until = NULL")
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
}

// idleEpisode opens the test's episode, a text episode idle for longer than the closer's
// hour, under the agent config configID of the customer customer.
func (s *EpisodeSweeperSuite) idleEpisode(customer, configID string) {
	contact := store.ContactMapEntry{
		CustomerID: customer, AgentConfigID: configID, Kind: store.ContactSlack,
		Address: "T1:U1", ConversationID: "agent:omni-" + uuid.NewString(),
	}
	_, err := s.store.MapContact(s.ctx, &contact)
	s.Require().NoError(err)
	episode := store.Episode{
		CustomerID: customer, ContactID: contact.ID, Source: store.EpisodeSlack,
		ThreadChannel: "agent:thread-" + uuid.NewString(), StartedAt: time.Now().Add(-2 * time.Hour),
	}
	_, err = s.store.OpenEpisode(s.ctx, &episode)
	s.Require().NoError(err)
	s.episode = episode.ID
}

// strayEpisode opens the test's episode under an agent config no row holds.
func (s *EpisodeSweeperSuite) strayEpisode() {
	s.idleEpisode("sweep-"+uuid.NewString(), "agent-"+uuid.NewString())
}

// status is the test's episode's status as stored.
func (s *EpisodeSweeperSuite) status() string {
	var status string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT status FROM episodes WHERE id = ?", s.episode).Scan(&status))
	return status
}

// A deployment where no config ever had the cards on, with connectors off, holds no episode:
// the sweeper never starts, so nothing is written and Stream and the LLM are asked nothing,
// as on base 668d15d5c.
func (s *EpisodeSweeperSuite) TestWithNeitherConnectorsNorCardsNorAnEpisodeTheSweeperNeverStarts() {
	plain := &store.AgentConfig{CustomerID: "sweep-" + uuid.NewString(), Name: "plain"}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, plain))

	started, err := startEpisodeSweeper(s.ctx, s.settings, s.store, s.closer)

	s.Require().NoError(err)
	s.False(started)
	s.Never(func() bool { return s.model.asked.Load() != 0 }, 300*time.Millisecond, 10*time.Millisecond,
		"the LLM was asked something")
	s.Empty(s.chat.Requests("deploy-key"), "Stream was asked something")
}

// Catches AI-2771: a config's chat episode was open when the config turned the cards off,
// and the router started again with no config's cards on and connectors off; the sweeper
// never started, so the episode stayed in progress and was never summarized.
func (s *EpisodeSweeperSuite) TestAnEpisodeLeftOpenAfterItsConfigTurnedTheCardsOffIsClosedAndItsSummaryFinished() {
	customer := "sweep-" + uuid.NewString()
	carded := &store.AgentConfig{CustomerID: customer, Name: "was carded", EpisodeCards: true}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, carded))
	s.idleEpisode(customer, carded.ID)
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE agent_configs SET episode_cards = false WHERE id = ?", carded.ID)
	s.Require().NoError(err)

	started, err := startEpisodeSweeper(s.ctx, s.settings, s.store, s.closer)

	s.Require().NoError(err)
	s.True(started)
	s.Eventually(func() bool { return s.status() != "in_progress" && s.status() != "ended" }, 5*time.Second, 10*time.Millisecond)
	s.NotEmpty(s.chat.Requests("deploy-key"), "the card and the thread channel were never reached")
}

// Catches an episode a router closed and stopped before summarizing, under a config that
// turned the cards off since: only a sweep takes its expired lease.
func (s *EpisodeSweeperSuite) TestAnEndedEpisodeWhoseSummaryLeaseRanOutHasItsSummaryFinished() {
	customer := "sweep-" + uuid.NewString()
	plain := &store.AgentConfig{CustomerID: customer, Name: "was carded"}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, plain))
	s.idleEpisode(customer, plain.ID)
	closedAt := time.Now()
	closed, err := s.store.CloseIdleEpisodes(s.ctx, closedAt.Add(-time.Hour), closedAt, 10, closedAt.Add(-time.Second))
	s.Require().NoError(err)
	s.Require().Len(closed, 1)

	started, err := startEpisodeSweeper(s.ctx, s.settings, s.store, s.closer)

	s.Require().NoError(err)
	s.True(started)
	s.Eventually(func() bool { return s.status() != "ended" }, 5*time.Second, 10*time.Millisecond)
	s.NotEmpty(s.chat.Requests("deploy-key"), "the card and the thread channel were never reached")
}

// No episode is in the database yet: the connectors alone start the sweeper, for the
// thread episodes their channel bridge opens.
func (s *EpisodeSweeperSuite) TestWithConnectorsOnTheSweeperStartsAndClosesAnIdleEpisode() {
	s.settings.Connectors.Enabled = true

	started, err := startEpisodeSweeper(s.ctx, s.settings, s.store, s.closer)

	s.Require().NoError(err)
	s.True(started)
	s.strayEpisode()
	s.Eventually(func() bool { return s.status() != "in_progress" }, 5*time.Second, 10*time.Millisecond)
}

// No episode is in the database yet: the config alone starts the sweeper, for the app chat
// episodes it will open.
func (s *EpisodeSweeperSuite) TestWithAConfigThatTurnedCardsOnTheSweeperStartsAndClosesAnIdleEpisode() {
	carded := &store.AgentConfig{CustomerID: "sweep-" + uuid.NewString(), Name: "carded", EpisodeCards: true}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, carded))

	started, err := startEpisodeSweeper(s.ctx, s.settings, s.store, s.closer)

	s.Require().NoError(err)
	s.True(started)
	s.strayEpisode()
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
