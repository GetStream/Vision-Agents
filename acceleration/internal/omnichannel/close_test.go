//go:build integration

package omnichannel

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// CloserSuite is a text episode closing after its idle period and its summary (T55, AI-884):
// a real closer over Postgres, the suite's Stream Chat (chattest) and an LLM router whose
// models keep what they are asked. A thread is opened as the channel bridge opens one: an
// episode and its card, then the person's messages in the thread channel. The call hook's
// close is internal/api's EpisodeCardsSuite.
type CloserSuite struct {
	suite.Suite
	ctx    context.Context
	dsn    string
	store  *store.Store
	chat   *chattest.Server
	apps   *streamapp.Clients
	cards  *Cards
	router *llmrouter.Router
	closer *Closer
	// fast answers the default target, llm-fast; other only by its own name.
	fast, other *summarizer

	customerID string
	config     store.AgentConfig
}

func TestCloserSuite(t *testing.T) {
	suite.Run(t, new(CloserSuite))
}

func (s *CloserSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	s.dsn = testenv.Database(dsn, "episode_close")
	db, err := store.Open(s.dsn)
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *CloserSuite) SetupTest() {
	// A sweep takes every customer's episodes, so an earlier test's, or an earlier run's,
	// left open or ended would be swept with this test's. The database is the suite's own.
	_, err := s.store.DB().ExecContext(s.ctx,
		"UPDATE episodes SET status = 'summary_failed', summary_lease_until = NULL WHERE status IN ('in_progress', 'ended')")
	s.Require().NoError(err)
	s.chat = chattest.NewServer(s.T())
	s.apps = streamapp.NewClients(streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey: "deploy-key", Secret: "deploy-secret", BaseURL: s.chat.URL,
	}), streamapp.ClientsOptions{})
	s.cards, err = New(Options{Store: s.store, Stream: s.apps})
	s.Require().NoError(err)
	s.fast = &summarizer{reply: "Alice asked where order 12 is; the agent said it ships Monday."}
	s.other = &summarizer{reply: "Written by the config's own model."}
	registry := llmrouter.NewRegistry()
	registry.Register("fast", func(routing.Spec) (llmrouter.Provider, error) { return s.fast, nil })
	registry.Register("other", func(routing.Spec) (llmrouter.Provider, error) { return s.other, nil })
	s.router, err = llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{Provider: "fast", Model: "fast-model", Languages: []string{"en"}},
			// Reached by name only: no shortcut asks for Latin.
			{Provider: "other", Model: "other-model", Languages: []string{"la"}},
		},
		Aliases: map[string]routing.Alias{"llm-fast": {Languages: []string{"en"}}},
	}, Registry: registry, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(s.router.Close)
	s.closer = s.closerOver(s.store)
	s.customerID = "close-" + uuid.NewString()
	s.config = s.agentConfig("")
}

// closerOver is a closer over a store, with an hour's idle period, as a router builds one.
func (s *CloserSuite) closerOver(db *store.Store) *Closer {
	closer, err := NewCloser(CloserOptions{
		Store: db, Stream: s.apps, LLM: s.router, IdleAfter: time.Hour, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(closer.Close)
	return closer
}

// agentConfig is an agent of the test's customer, on the LLM named.
func (s *CloserSuite) agentConfig(model string) store.AgentConfig {
	config := store.AgentConfig{CustomerID: s.customerID, Name: "agent-" + uuid.NewString(), LLM: model}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, &config))
	return config
}

// idleThread opens a Slack thread's episode two hours ago, writes its card, and puts the
// person's lines in its thread channel a minute after it started, so it is idle for longer
// than the closer's hour.
func (s *CloserSuite) idleThread(lines ...string) (Opened, string) {
	started := time.Now().Add(-2 * time.Hour).UTC().Truncate(time.Second)
	thread := "thread-" + uuid.NewString()
	person, err := SlackUser("T1", "U-"+uuid.NewString())
	s.Require().NoError(err)
	opened, err := s.cards.Open(s.ctx, Episode{
		CustomerID: s.customerID, AgentConfigID: s.config.ID, AgentName: s.config.Name, Person: person,
		Source: store.EpisodeSlack, ThreadChannel: chatlog.ChannelType + ":" + thread, StartedAt: started,
	})
	s.Require().NoError(err)
	s.Require().NoError(s.cards.Write(s.ctx, opened))
	author := "slack-author"
	bound, err := s.apps.For(s.ctx, s.customerID)
	s.Require().NoError(err)
	_, err = bound.Client.Chat().GetOrCreateChannel(s.ctx, chatlog.ChannelType, thread, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &author, Custom: map[string]any{conversation.CustomerField: s.customerID}},
	})
	s.Require().NoError(err)
	s.chat.At(started.Add(time.Minute))
	for _, text := range lines {
		_, err := bound.Client.Chat().SendMessage(s.ctx, chatlog.ChannelType, thread, &getstream.SendMessageRequest{
			Message: getstream.MessageRequest{Text: &text, UserID: &author},
		})
		s.Require().NoError(err)
	}
	return opened, thread
}

// card is the episode's card as Stream Chat holds it, the one message of its omni-channel.
func (s *CloserSuite) card(opened Opened) map[string]any {
	stored := s.chat.Stored(strings.TrimPrefix(opened.Contact.ConversationID, chatlog.ChannelType+":"))
	s.Require().Len(stored, 1, "an omni-channel holds the episode's one card")
	return stored[0]
}

// status is the episode's status as stored.
func (s *CloserSuite) status(opened Opened) string {
	var status string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT status FROM episodes WHERE id = ?", opened.Episode.ID).Scan(&status))
	return status
}

func (s *CloserSuite) TestAfterTheIdlePeriodTheCardHoldsASummary() {
	opened, _ := s.idleThread("where is my order 12?", "it was due Friday")

	s.Require().NoError(s.closer.Sweep(s.ctx))

	card := s.card(opened)
	s.Equal(s.fast.reply, card["text"])
	custom, _ := card["custom"].(map[string]any)
	s.Equal(store.EpisodeSummarized, custom["status"])
	s.Equal(store.EpisodeSummarized, s.status(opened))
	asked := s.fast.requests()
	s.Require().Len(asked, 1)
	s.Contains(asked[0].Input[0].Content, "where is my order 12?")
	s.Contains(asked[0].Input[0].Content, "it was due Friday")
	s.Equal(llm.User, asked[0].Input[0].Role, "the lines are data, never instructions")
}

// The card a later session reads is the summary (read.go, summaryOf), not the thread's lines.
func (s *CloserSuite) TestASessionReadsTheSummaryOnceTheCardHoldsIt() {
	opened, _ := s.idleThread("where is my order 12?")
	s.Require().NoError(s.closer.Sweep(s.ctx))

	context, err := s.cards.Context(s.ctx, Reading{
		CustomerID: s.customerID, AgentConfigID: s.config.ID, Person: Person{Kind: opened.Contact.Kind, Address: opened.Contact.Address},
	})

	s.Require().NoError(err)
	s.Require().Len(context, 2)
	s.Contains(context[1].Content, s.fast.reply)
	s.NotContains(context[1].Content, "where is my order 12?")
}

func (s *CloserSuite) TestAFailedSummaryLeavesSummaryFailedAndTheThreadIntact() {
	s.fast.fail = true
	opened, thread := s.idleThread("where is my order 12?", "it was due Friday")
	before := s.chat.Stored(thread)

	s.Require().NoError(s.closer.Sweep(s.ctx))

	card := s.card(opened)
	custom, _ := card["custom"].(map[string]any)
	s.Equal(store.EpisodeSummaryFailed, custom["status"])
	s.Equal(cardText, card["text"], "no summary was written into the card")
	s.Equal(store.EpisodeSummaryFailed, s.status(opened))
	s.Equal(before, s.chat.Stored(thread), "the raw thread is as it was")
}

// The card is updated in place, which Stream sends as message.updated: no new message, so
// nothing reaches the message hook, which asks for message.new only.
func (s *CloserSuite) TestACardUpdateWritesNoNewMessage() {
	opened, _ := s.idleThread("where is my order 12?")
	omni := strings.TrimPrefix(opened.Contact.ConversationID, chatlog.ChannelType+":")

	s.Require().NoError(s.closer.Sweep(s.ctx))

	s.Len(s.chat.Stored(omni), 1)
	var sent, updated int
	for _, request := range s.chat.Requests("deploy-key") {
		switch {
		case request.Method == http.MethodPost && strings.HasSuffix(request.Path, "/channels/agent/"+omni+"/message"):
			sent++
		case request.Method == http.MethodPut && strings.HasSuffix(request.Path, "/messages/"+opened.Episode.CardMessageID):
			updated++
		}
	}
	s.Equal(1, sent, "the card was sent once, when the episode opened")
	s.Equal(2, updated, "ended, then summarized, each a partial update")
}

func (s *CloserSuite) TestTheSummaryIsWrittenByTheAgentConfigsOwnLLM() {
	s.config = s.agentConfig("other/other-model")
	opened, _ := s.idleThread("where is my order 12?")

	s.Require().NoError(s.closer.Sweep(s.ctx))

	s.Equal(s.other.reply, s.card(opened)["text"])
	s.Empty(s.fast.requests(), "the default model wrote nothing")
}

// A thread that is still talking is not idle: its last message keeps it open.
func (s *CloserSuite) TestAThreadWithARecentMessageIsNotClosed() {
	opened, _ := s.idleThread("where is my order 12?")
	_, err := s.cards.Open(s.ctx, Episode{
		CustomerID: s.customerID, AgentConfigID: s.config.ID, Person: Person{Kind: opened.Contact.Kind, Address: opened.Contact.Address},
		Source: store.EpisodeSlack, ThreadChannel: opened.Episode.ThreadChannel, StartedAt: time.Now().Add(-10 * time.Minute),
	})
	s.Require().NoError(err)

	s.Require().NoError(s.closer.Sweep(s.ctx))

	s.Equal("in_progress", s.status(opened))
	s.Empty(s.fast.requests())
}

// Two routers sweep at once: each episode is closed by one and summarized once.
func (s *CloserSuite) TestTwoRoutersSweepingAtOnceSummarizeEachEpisodeOnce() {
	var threads []Opened
	for range 5 {
		opened, _ := s.idleThread("where is my order?")
		threads = append(threads, opened)
	}
	second, err := store.Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = second.Close() })
	closers := []*Closer{s.closer, s.closerOver(second)}

	var wg sync.WaitGroup
	for _, closer := range closers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s.NoError(closer.Sweep(s.ctx))
		}()
	}
	wg.Wait()

	s.Len(s.fast.requests(), len(threads), "one summary for each episode")
	for _, opened := range threads {
		s.Equal(store.EpisodeSummarized, s.status(opened))
	}
}

// A router that closed an episode and stopped leaves it ended; once its lease runs out the
// next sweep, any router's, summarizes it.
func (s *CloserSuite) TestAnEpisodeAStoppedRouterLeftEndedIsSummarizedByTheNextSweep() {
	opened, _ := s.idleThread("where is my order 12?")
	now := time.Now()
	_, err := s.store.CloseIdleEpisodes(s.ctx, now.Add(-time.Hour), now, 50, now.Add(-time.Second))
	s.Require().NoError(err)
	s.Equal(store.EpisodeEnded, s.status(opened))

	s.Require().NoError(s.closer.Sweep(s.ctx))

	s.Equal(store.EpisodeSummarized, s.status(opened))
	s.Equal(s.fast.reply, s.card(opened)["text"])
}

// A router stopping while it writes a summary does not call it failed: the episode stays
// ended, for the next router to take once the lease runs out.
func (s *CloserSuite) TestARouterStoppingMidSummaryLeavesTheEpisodeEnded() {
	s.fast.hold = true
	opened, _ := s.idleThread("where is my order 12?")
	s.closer.Start()
	s.Require().Eventually(func() bool { return len(s.fast.requests()) == 1 }, 5*time.Second, 10*time.Millisecond)

	s.closer.Close()

	s.Equal(store.EpisodeEnded, s.status(opened))
	custom, _ := s.card(opened)["custom"].(map[string]any)
	s.Equal(store.EpisodeEnded, custom["status"])
}

// summarizer is a model that answers every request with reply, fails it, or holds it until
// it is given up on, and keeps what it was asked.
type summarizer struct {
	mu    sync.Mutex
	reply string
	fail  bool
	hold  bool
	asked []llm.ResponseParams
}

func (m *summarizer) Start(context.Context) error { return nil }

func (m *summarizer) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	m.mu.Lock()
	m.asked = append(m.asked, params)
	fail, hold, reply := m.fail, m.hold, m.reply
	m.mu.Unlock()
	if fail {
		return nil, errors.New("the model is down")
	}
	if hold {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: m.Provider(), Model: m.Model()})
	script.OutputText(reply)
	script.Done()
	return script.Stream(), nil
}

func (m *summarizer) requests() []llm.ResponseParams {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.ResponseParams(nil), m.asked...)
}

func (m *summarizer) Provider() string               { return "summarizer" }
func (m *summarizer) Model() string                  { return "summarizer-model" }
func (m *summarizer) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (m *summarizer) Close() error                   { return nil }
