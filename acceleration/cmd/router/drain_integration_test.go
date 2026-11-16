//go:build integration

package main

import (
	"context"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	chathooks "github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatinbound"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// DrainSuite is what a router holds its shutdown drain for, on a database of
// the suite's own.
type DrainSuite struct {
	suite.Suite
	ctx context.Context
	db  *store.Store
	// dsn is the suite's database.
	dsn string
}

func TestDrainSuite(t *testing.T) {
	suite.Run(t, new(DrainSuite))
}

func (s *DrainSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	settings := config.Defaults()
	settings.Postgres.DSN = testenv.Database(dsn, "drain")
	s.dsn = settings.Postgres.DSN
	db, err := openStore(s.ctx, settings)
	s.Require().NoError(err)
	s.T().Cleanup(func() { db.Close() })
	s.db = db
}

func (s *DrainSuite) TestTheDrainWaitsForAScheduledDeliveryInFlight() {
	due := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	task := store.ScheduledTask{
		CustomerID: "customer-" + uuid.NewString(), ConfigID: "config-1", UserID: "user-1",
		CallerKind: store.UserKindAuthenticated, Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: due, Timezone: "UTC", NextRunAt: &due, State: store.ScheduledTaskActive, CreatedBy: "agent",
	}
	s.Require().NoError(s.db.CreateScheduledTask(s.ctx, &task, 25))
	s.T().Cleanup(func() {
		s.NoError(s.db.DeleteScheduledTask(context.Background(), task.CustomerID, task.ID))
	})
	release := make(chan struct{})
	scheduler := scheduling.NewScheduler(s.db, func(ctx context.Context, _ store.ClaimedScheduledTask) (string, string, error) {
		select {
		case <-release:
		case <-ctx.Done():
		}
		return "conversation", "", nil
	}, slog.New(slog.DiscardHandler))
	scheduler.Start(s.ctx)
	s.T().Cleanup(scheduler.Close)

	s.Require().Eventually(func() bool { return scheduler.Running() > 0 }, 10*time.Second, 10*time.Millisecond)
	s.Positive(running(nil, scheduler, nil, nil, nil, nil, nil), "the drain counts the delivery in flight")
	s.Equal(scheduler.Running(), running(nil, scheduler, nil, nil, nil, nil, nil))

	close(release)
	s.Require().Eventually(func() bool { return running(nil, scheduler, nil, nil, nil, nil, nil) == 0 }, 10*time.Second, 10*time.Millisecond)
}

func (s *DrainSuite) TestTheDrainWaitsForAChatAnswerInFlight() {
	id := "message-" + uuid.NewString()
	queued, err := s.db.QueueChatInbound(s.ctx, store.ChatInbound{
		StreamApp: 1, MessageID: id, Deployment: true, Body: []byte(`{}`), NextAttemptAt: time.Now(),
	})
	s.Require().NoError(err)
	s.Require().True(queued)
	s.T().Cleanup(func() { s.deleteChatInbound(id) })
	release := make(chan struct{})
	deliveries, err := chatinbound.NewWorker(chatinbound.Options{Store: s.db, Answer: func(ctx context.Context, _ store.ChatInbound) error {
		select {
		case <-release:
		case <-ctx.Done():
		}
		return nil
	}, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	deliveries.Start(s.ctx)
	s.T().Cleanup(deliveries.Close)

	s.Require().Eventually(func() bool { return deliveries.Running() > 0 }, 10*time.Second, 10*time.Millisecond)
	s.Positive(running(nil, nil, nil, deliveries, nil, nil, nil), "the drain counts the answer in flight")
	s.Equal(deliveries.Running(), running(nil, nil, nil, deliveries, nil, nil, nil))

	close(release)
	s.Require().Eventually(func() bool { return running(nil, nil, nil, deliveries, nil, nil, nil) == 0 }, 10*time.Second, 10*time.Millisecond)
}

// A router told to stop while it writes an inbound message takes no other message, of the
// bridge or of Stream Chat, that falls due during its drain, and its drain lasts until that
// write ends. Both workers claim until the signal (run's ctx), and the drain counts the
// bridge's write (running).
func (s *DrainSuite) TestARouterDrainingAnInboundWriteTakesNoMoreMessages() {
	chat := chattest.NewServer(s.T())
	public, _, err := ed25519.GenerateKey(nil)
	s.Require().NoError(err)
	settings := config.Defaults()
	settings.Postgres.DSN = s.dsn
	settings.Addr = "127.0.0.1:0"
	settings.Auth.Mode = string(auth.NoAuth)
	// Connectors, for the channel bridge, seal under a keyring.
	settings.Auth.KEK = "drain-key"
	settings.Connectors.Enabled = true
	settings.Stream.APIKey, settings.Stream.APISecret, settings.Stream.BaseURL = "deploy-key", "deploy-secret", chat.URL
	settings.Chat.InboundKeys = []string{base64.StdEncoding.EncodeToString(public)}
	settings.Shutdown.Drain = time.Minute
	// The peer listener closes when the drain ends, so a dial to it says whether the drain
	// still runs.
	free, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	settings.Node.Listen = free.Addr().String()
	s.Require().NoError(free.Close())

	customer := "customer-" + uuid.NewString()
	writing, later := s.thread(customer), s.thread(customer)
	waiting, release := chat.Hold("/" + writing.ChannelID + "/message")
	// Due after the router has started and been told to stop: the workers wait for it then.
	// Well under writeTimeout (30 s), so the write held is still in flight once it is due and
	// has the rest to finish in.
	due := time.Now().Add(12 * time.Second)
	s.queueInbound(writing, time.Now())
	s.queueInbound(later, due)
	message := "message-" + uuid.NewString()
	queued, err := s.db.QueueChatInbound(s.ctx, store.ChatInbound{
		StreamApp: 1, MessageID: message, Deployment: true, Body: []byte(`{}`), NextAttemptAt: due,
	})
	s.Require().NoError(err)
	s.Require().True(queued)
	s.T().Cleanup(func() { s.deleteChatInbound(message) })

	stopped := make(chan struct{})
	var runErr error
	go func() {
		runErr = run(settings, slog.New(slog.DiscardHandler))
		close(stopped)
	}()
	// A failed wait below ends the test with the router up or draining; run handles SIGTERM
	// only while it runs, so it is stopped here and not by the signal killing the binary.
	s.T().Cleanup(func() {
		release()
		if isClosed(stopped) {
			return
		}
		s.NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
		select {
		case <-stopped:
		case <-time.After(30 * time.Second):
			s.Fail("the router did not stop on SIGTERM")
		}
	})
	select {
	case <-waiting:
	case <-stopped:
		s.Require().Failf("the router stopped before it was told to", "error: %v", runErr)
	case <-time.After(15 * time.Second):
		s.Require().Fail("the router never wrote the first message")
	}
	// The worker starts before the listeners do; the signal is for a router that serves.
	s.Require().Eventually(func() bool { return serving(settings.Node.Listen) }, 15*time.Second, 10*time.Millisecond)
	s.Require().True(time.Now().Before(due.Add(-2*time.Second)), "the router took too long to start")

	s.Require().NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
	s.Never(func() bool {
		return isClosed(stopped) || !serving(settings.Node.Listen) || s.inbound(later).NextAttemptAt.After(due.Add(time.Second)) ||
			s.chatInbound(message).NextAttemptAt.After(due.Add(time.Second))
	}, time.Until(due.Add(2*time.Second)), 50*time.Millisecond,
		"the drain ended with the write in flight, or a draining router took a message")
	release()

	select {
	case <-stopped:
		s.Require().NoError(runErr)
	case <-time.After(30 * time.Second):
		s.Require().Fail("the drain did not end with the write")
	}
	s.Equal(store.ChannelInboundDone, s.inbound(writing).State)
	left := s.inbound(later)
	s.Equal([]any{store.ChannelInboundQueued, 0}, []any{left.State, left.Attempts})
	chatLeft := s.chatInbound(message)
	s.Equal([]any{store.ChatInboundQueued, 0}, []any{chatLeft.State, chatLeft.Attempts})
}

// A router told to stop while it writes the episode card of an app chat's message, which a
// message hook started off its request, drains until that card is written: a card lost then
// is lost for good, as the next message finds its episode open. Catches the server's chat
// cards left out of what the drain counts (running).
func (s *DrainSuite) TestARouterDrainsAChatCardInFlight() {
	chat := chattest.NewServer(s.T())
	settings := config.Defaults()
	settings.Postgres.DSN = s.dsn
	settings.Auth.Mode = string(auth.NoAuth)
	settings.Stream.APIKey, settings.Stream.APISecret, settings.Stream.BaseURL = "deploy-key", "deploy-secret", chat.URL
	settings.Stream.AppID = 1
	settings.Shutdown.Drain = time.Minute
	for _, address := range []*string{&settings.Addr, &settings.Node.Listen} {
		free, err := net.Listen("tcp", "127.0.0.1:0")
		s.Require().NoError(err)
		*address = free.Addr().String()
		s.Require().NoError(free.Close())
	}
	agent := store.AgentConfig{CustomerID: "customer-" + uuid.NewString(), Name: "chat-" + uuid.NewString(),
		Mode: store.AgentModeText, LLM: "llm-flow", EpisodeCards: true}
	s.Require().NoError(s.db.CreateAgentConfig(s.ctx, &agent))
	waiting, release := chat.Hold("/message")

	stopped := make(chan struct{})
	var runErr error
	go func() {
		runErr = run(settings, slog.New(slog.DiscardHandler))
		close(stopped)
	}()
	s.T().Cleanup(func() {
		release()
		if isClosed(stopped) {
			return
		}
		s.NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
		select {
		case <-stopped:
		case <-time.After(30 * time.Second):
			s.Fail("the router did not stop on SIGTERM")
		}
	})
	s.Require().Eventually(func() bool { return serving(settings.Addr) }, 15*time.Second, 10*time.Millisecond)
	body := fmt.Sprintf(`{"type":"message.new","app_id":1,"channel_id":%q,"channel_type":"agent","created_at":%q,
"channel_custom":{"agent_config_id":%q},"channel":{"members":[{"user_id":"anya_123"}]},"message":{"id":%q,"text":"where is my order?","user":{"id":"anya_123","name":"Anya"}}}`,
		"chat-"+uuid.NewString(), time.Now().UTC().Format(time.RFC3339Nano), agent.ID, "message-"+uuid.NewString())
	mac := hmac.New(sha256.New, []byte(settings.Stream.APISecret))
	mac.Write([]byte(body))
	request, err := http.NewRequest(http.MethodPost, "http://"+settings.Addr+chathooks.MessageHookPath, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Signature", hex.EncodeToString(mac.Sum(nil)))
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Require().Equal(http.StatusOK, response.StatusCode)
	select {
	case <-waiting:
	case <-time.After(15 * time.Second):
		s.Require().Fail("the router never wrote the card")
	}

	s.Require().NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
	s.Never(func() bool { return isClosed(stopped) || !serving(settings.Node.Listen) }, 3*time.Second, 50*time.Millisecond,
		"the drain ended with the card in flight")
	release()

	select {
	case <-stopped:
		s.Require().NoError(runErr)
	case <-time.After(30 * time.Second):
		s.Require().Fail("the drain did not end with the card")
	}
}

// Defect: run never hands stopServing the trunk call provider's StopTaking, so a draining
// router takes other nodes' trunk calls over only to hand them on again when it stops.
func (s *DrainSuite) TestADrainingRouterStopsSayingItTakesTrunkCallsOver() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	client, err := redisready.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	takers := func() []string {
		nodes, err := trunkCallTakers(s.ctx, client)
		s.Require().NoError(err)
		return nodes
	}
	chat := chattest.NewServer(s.T())
	settings := config.Defaults()
	settings.Postgres.DSN = s.dsn
	settings.Auth.Mode = string(auth.NoAuth)
	settings.Auth.KEK = "drain-key"
	settings.Redis.Addr = address
	settings.Stream.APIKey, settings.Stream.APISecret, settings.Stream.BaseURL = "deploy-key", "deploy-secret", chat.URL
	settings.Stream.AppID = 1
	settings.Shutdown.Drain = time.Minute
	for _, address := range []*string{&settings.Addr, &settings.Node.Listen} {
		free, err := net.Listen("tcp", "127.0.0.1:0")
		s.Require().NoError(err)
		*address = free.Addr().String()
		s.Require().NoError(free.Close())
	}
	// An episode card held in flight is what keeps the router draining.
	agent := store.AgentConfig{CustomerID: "customer-" + uuid.NewString(), Name: "chat-" + uuid.NewString(),
		Mode: store.AgentModeText, LLM: "llm-flow", EpisodeCards: true}
	s.Require().NoError(s.db.CreateAgentConfig(s.ctx, &agent))
	waiting, release := chat.Hold("/message")
	before := takers()

	stopped := make(chan struct{})
	var runErr error
	go func() {
		runErr = run(settings, slog.New(slog.DiscardHandler))
		close(stopped)
	}()
	s.T().Cleanup(func() {
		release()
		if isClosed(stopped) {
			return
		}
		s.NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
		select {
		case <-stopped:
		case <-time.After(30 * time.Second):
			s.Fail("the router did not stop on SIGTERM")
		}
	})
	s.Require().Eventually(func() bool { return serving(settings.Addr) }, 15*time.Second, 10*time.Millisecond)
	var node string
	s.Require().Eventually(func() bool {
		node = joined(before, takers())
		return node != ""
	}, 10*time.Second, 20*time.Millisecond, "the running router does not say it takes trunk calls over")
	body := fmt.Sprintf(`{"type":"message.new","app_id":1,"channel_id":%q,"channel_type":"agent","created_at":%q,
"channel_custom":{"agent_config_id":%q},"channel":{"members":[{"user_id":"anya_123"}]},"message":{"id":%q,"text":"where is my order?","user":{"id":"anya_123","name":"Anya"}}}`,
		"chat-"+uuid.NewString(), time.Now().UTC().Format(time.RFC3339Nano), agent.ID, "message-"+uuid.NewString())
	mac := hmac.New(sha256.New, []byte(settings.Stream.APISecret))
	mac.Write([]byte(body))
	request, err := http.NewRequest(http.MethodPost, "http://"+settings.Addr+chathooks.MessageHookPath, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Signature", hex.EncodeToString(mac.Sum(nil)))
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Require().Equal(http.StatusOK, response.StatusCode)
	select {
	case <-waiting:
	case <-time.After(15 * time.Second):
		s.Require().Fail("the router never wrote the card")
	}

	s.Require().NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
	s.Require().Eventually(func() bool { return !slices.Contains(takers(), node) }, 5*time.Second, 20*time.Millisecond,
		"a draining router still says it takes trunk calls over")
	s.Require().True(serving(settings.Node.Listen), "the drain ended before the check, so stopping alone withdrew the router")
	release()

	select {
	case <-stopped:
		s.Require().NoError(runErr)
	case <-time.After(30 * time.Second):
		s.Require().Fail("the drain did not end with the card")
	}
}

// thread links a thread channel of customer's, on a connector the router has no scheme for.
func (s *DrainSuite) thread(customer string) store.ChannelThread {
	thread := store.ChannelThread{
		ChannelID: conversation.ThreadChannelPrefix + uuid.NewString(), CustomerID: customer, ConnectorID: "fake_line",
		ProviderUnitID: "T0000TEAM", ThreadKey: uuid.NewString(), ConnectionID: "connection-1",
	}
	_, err := s.db.LinkChannelThread(s.ctx, &thread)
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		_, err := s.db.DB().NewDelete().Model((*store.ChannelInbound)(nil)).Where("channel_id = ?", thread.ChannelID).Exec(context.Background())
		s.NoError(err)
	})
	return thread
}

// queueInbound queues a message in thread, due at due.
func (s *DrainSuite) queueInbound(thread store.ChannelThread, due time.Time) {
	queued, err := s.db.AcceptChannelInbound(s.ctx, store.ChannelInbound{
		ChannelID: thread.ChannelID, ProviderMessageID: "m-1", CustomerID: thread.CustomerID,
		ConnectorID: thread.ConnectorID, ProviderUnitID: thread.ProviderUnitID, ThreadKey: thread.ThreadKey,
		ConfigID: "config-1", ConfigName: "Support", AuthorID: "U0000ALICE", Text: "hello",
		Raw: []byte(`{"text":"hello"}`), NextAttemptAt: due,
	}, nil)
	s.Require().NoError(err)
	s.Require().True(queued)
}

// inbound is thread's queued message as it is now.
func (s *DrainSuite) inbound(thread store.ChannelThread) store.ChannelInbound {
	var row store.ChannelInbound
	s.Require().NoError(s.db.DB().NewSelect().Model(&row).Where("channel_id = ?", thread.ChannelID).Scan(s.ctx))
	return row
}

// chatInbound is a queued Stream Chat message's row as it is now.
func (s *DrainSuite) chatInbound(messageID string) store.ChatInbound {
	var row store.ChatInbound
	s.Require().NoError(s.db.DB().NewSelect().Model(&row).Where("message_id = ?", messageID).Scan(s.ctx))
	return row
}

func (s *DrainSuite) deleteChatInbound(messageID string) {
	_, err := s.db.DB().NewDelete().Model((*store.ChatInbound)(nil)).Where("message_id = ?", messageID).Exec(context.Background())
	s.NoError(err)
}

// serving is whether a listener takes connections at address.
func serving(address string) bool {
	conn, err := net.DialTimeout("tcp", address, time.Second)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

// isClosed is whether done is closed, without waiting.
func isClosed(done <-chan struct{}) bool {
	select {
	case <-done:
		return true
	default:
		return false
	}
}

func (s *DrainSuite) TestTheDrainWaitsForACampaignCallInFlight() {
	customerID := "customer-" + uuid.NewString()
	sequence, err := json.Marshal(campaign.DefaultSequence("config-1"))
	s.Require().NoError(err)
	hours, err := json.Marshal(campaign.LegacyCallingHours())
	s.Require().NoError(err)
	created := store.Campaign{CustomerID: customerID, Name: "renewals", ConfigID: "config-1",
		FromNumber: "+15550100000", Concurrency: 1, Sequence: sequence, CallingHours: hours}
	s.Require().NoError(s.db.CreateCampaign(s.ctx, &created))
	_, _, err = s.db.AddContacts(s.ctx, created.ID, []store.Contact{{ToNumber: "+15550000001"}})
	s.Require().NoError(err)
	now := time.Now()
	_, err = s.db.LaunchCampaign(s.ctx, customerID, created.ID, store.Launch{StartAt: now, Now: now, APIKey: "key-a"})
	s.Require().NoError(err)
	// Stopped whatever happens, so no later test's dialer rings it.
	s.T().Cleanup(func() { s.NoError(s.db.StopCampaign(context.Background(), customerID, created.ID, time.Now())) })

	release := make(chan struct{})
	dialer, err := campaign.New(campaign.Options{
		Store: s.db, Dialer: heldDial(release), Transcripts: unreadTranscripts{}, Classifier: noOutcome{},
		Tick: 10 * time.Millisecond, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Require().NoError(dialer.Start(s.ctx))
	s.T().Cleanup(dialer.Close)

	s.Require().Eventually(func() bool { return dialer.Running() > 0 }, 10*time.Second, 10*time.Millisecond)
	s.Equal(1, running(nil, nil, nil, nil, nil, dialer, nil), "the drain counts the call in flight")

	close(release)
	s.Require().Eventually(func() bool { return running(nil, nil, nil, nil, nil, dialer, nil) == 0 }, 10*time.Second, 10*time.Millisecond)
}

// heldDial is a dial that lasts until release closes, as a campaign call lasts until its
// session ends, and that nobody answered.
type heldDial chan struct{}

func (h heldDial) Dial(ctx context.Context, _ campaign.Dial) (campaign.Dialed, error) {
	select {
	case <-h:
	case <-ctx.Done():
	}
	return campaign.Dialed{Outcome: outcome.NoAnswer}, nil
}

type unreadTranscripts struct{}

func (unreadTranscripts) CallTranscript(context.Context, string, string) ([]chatlog.Spoken, error) {
	return nil, nil
}

type noOutcome struct{}

func (noOutcome) Classify(context.Context, store.Campaign, string, campaign.Step, []chatlog.Spoken) (string, error) {
	return "", nil
}
