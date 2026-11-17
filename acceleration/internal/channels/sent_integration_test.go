//go:build integration

package channels

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// roundTrip is an http.RoundTripper that is a function.
type roundTrip func(*http.Request) (*http.Response, error)

func (f roundTrip) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

// SentSuite is whom the day's channel limit counts as reached when the service sends: the
// person written to, not the line or the customer. It runs a real gate over Redis and a real
// enforcer, with the provider stood in for by a local server.
type SentSuite struct {
	suite.Suite
	ctx     context.Context
	money   *policytest.Money
	redis   rueidis.Client
	service *Service
	owner   routing.Owner
	sealer  *auth.Sealer
}

func TestSentSuite(t *testing.T) { suite.Run(t, new(SentSuite)) }

func (s *SentSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	s.ctx = context.Background()
	s.money = policytest.Start(s.T(), "channels_sent")
	redis, err := redisready.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(redis.Close)
	s.redis = redis
	s.sealer, err = auth.NewSealer("sent-suite-kek")
	s.Require().NoError(err)
}

func (s *SentSuite) SetupTest() {
	s.owner = s.money.Owner(s.T(), 1_000_000_000)
	taken := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }))
	s.T().Cleanup(taken.Close)
	target, err := url.Parse(taken.URL)
	s.Require().NoError(err)
	provider := &http.Client{Transport: roundTrip(func(request *http.Request) (*http.Response, error) {
		request.URL.Scheme, request.URL.Host = target.Scheme, target.Host
		return http.DefaultTransport.RoundTrip(request)
	})}
	gate := dlc.NewGate(s.money.Store, s.redis, dlc.Sandbox{}, slog.New(slog.DiscardHandler)).
		Charging(s.money.Enforcer).Limiting(config.LimitsEnforce)
	s.service = &Service{store: s.money.Store, secrets: s.sealer, transport: provider, gate: gate,
		logger: slog.New(slog.DiscardHandler)}
}

// reached is who the owner's organization reached by text today, as the counters say.
func (s *SentSuite) reached() []string {
	key := "channel:" + time.Now().UTC().Format("20060102") + ":{" + s.owner.OrganizationID + "}:sms:to"
	members, err := s.redis.Do(s.ctx, s.redis.B().Smembers().Key(key).Build()).AsStrSlice()
	s.Require().NoError(err)
	slices.Sort(members)
	s.T().Cleanup(func() { s.Require().NoError(s.redis.Do(s.ctx, s.redis.B().Del().Key(key).Build()).Error()) })
	return members
}

// Catches Notify counting its recipient as the line or nobody: the second message a day to the
// same person must not take another of the organization's recipients.
func (s *SentSuite) TestANoticeCountsTheUserItReachedAsReached() {
	line, writer := "+15550100"+uuid.NewString()[:2], "+13475550101"
	sealed, err := s.sealer.SealWithAAD(`{"token":"telnyx-key"}`, []byte(s.owner.CustomerID))
	s.Require().NoError(err)
	s.Require().NoError(s.money.Store.SaveChannelAccount(s.ctx, &store.ChannelAccount{
		CustomerID: s.owner.CustomerID, Kind: string(SMS), E164: line, Token: "line-" + uuid.NewString(),
		SecretsSealed: sealed, SecretsKEKVersion: s.sealer.CurrentVersion(),
	}))
	agentConfig := store.AgentConfig{CustomerID: s.owner.CustomerID, Name: "sent-" + uuid.NewString(),
		Channels: store.AgentChannels{SMS: &store.ChannelLine{Number: line}}}
	s.Require().NoError(s.money.Store.CreateAgentConfig(s.ctx, &agentConfig))
	s.Require().NoError(s.money.Store.SaveChannelIdentity(s.ctx, &store.ChannelIdentity{
		CustomerID: s.owner.CustomerID, ConfigID: agentConfig.ID, Kind: string(SMS), Address: writer, UserID: "user-1",
	}))

	err = s.service.Notify(s.ctx, Notice{CustomerID: s.owner.CustomerID, ConfigID: agentConfig.ID, Kind: SMS,
		UserID: "user-1", Text: "Your table is ready."})

	s.Require().NoError(err)
	s.Equal([]string{writer}, s.reached(), "the person written to, not the line, the customer or nobody")
}

// Catches a reply counting nobody, or the line, as the person reached: the reached set is who
// the reply went to.
func (s *SentSuite) TestAReplyCountsTheSenderItAnsweredAsReached() {
	line, writer := Account{Kind: SMS, E164: "+15550100999", Token: "telnyx-key"}, "+13475550102"
	provider, ok := For(SMS, s.service.transport)
	s.Require().True(ok)
	events := make(chan session.Event, 1)
	events <- agent.Responded{Text: "On my way."}
	close(events)

	s.service.send(s.ctx, nil, s.owner, line, provider, Message{Kind: SMS, From: writer, To: line.E164, Thread: writer}, events)

	s.Equal([]string{writer}, s.reached(), "the sender answered, not the line, the customer or nobody")
}
