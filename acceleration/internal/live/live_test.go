//go:build integration

package live

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// AddressEnvVar is where the tests look for a Redis to run against.
const AddressEnvVar = "ROUTER_REDIS_ADDR"

type LiveSuite struct {
	suite.Suite
	client *Client
	ctx    context.Context
	// customer is unique per test so parallel runs cannot collide on keys.
	customer string
	model    string
}

func TestLiveSuite(t *testing.T) {
	suite.Run(t, new(LiveSuite))
}

func (s *LiveSuite) SetupSuite() {
	address := os.Getenv(AddressEnvVar)
	if address == "" {
		s.T().Skipf("%s not set", AddressEnvVar)
	}

	client, err := New(Options{Address: address})
	s.Require().NoError(err)
	s.client = client
	s.ctx = context.Background()
	s.Require().NoError(client.Ping(s.ctx))
}

func (s *LiveSuite) TearDownSuite() {
	if s.client != nil {
		s.client.Close()
	}
}

func (s *LiveSuite) SetupTest() {
	// Fresh keys per test, so no cleanup is needed and the TTL logic stays observable.
	unique := time.Now().UnixNano()
	s.customer = fmt.Sprintf("customer-%d", unique)
	s.model = fmt.Sprintf("model-%d", unique)
}

// usage is a successful speech-to-text request, which is what most of these tests record.
func (s *LiveSuite) usage(latencyMs float64, audioMs int64, success bool) Usage {
	return Usage{
		Modality:   "stt",
		CustomerID: s.customer,
		Provider:   "deepgram",
		Model:      s.model,
		LatencyMs:  latencyMs,
		AudioMs:    audioMs,
		Success:    success,
	}
}

func (s *LiveSuite) TestUnseenProviderIsAvailableWithNoHistory() {
	health, err := s.client.Health(s.ctx, "stt", "deepgram", s.model)
	s.Require().NoError(err)

	s.True(health.Available, "a provider with no history should not be penalised")
	s.Zero(health.Requests)
	s.Zero(health.Errors)
	s.InDelta(1.0, health.SuccessRate(), 0.001)
}

func (s *LiveSuite) TestSuccessfulRequestsBuildUpHealth() {
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1000, true)))
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(300, 2000, true)))

	health, err := s.client.Health(s.ctx, "stt", "deepgram", s.model)
	s.Require().NoError(err)

	s.EqualValues(2, health.Requests)
	s.Zero(health.Errors)
	s.InDelta(200.0, health.LatencyMsAvg, 0.001, "latency is the mean over the window")
	s.True(health.Available)
}

func (s *LiveSuite) TestErrorsLowerTheSuccessRate() {
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1000, true)))
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1000, true)))
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1000, false)))

	health, err := s.client.Health(s.ctx, "stt", "deepgram", s.model)
	s.Require().NoError(err)

	s.EqualValues(3, health.Requests)
	s.EqualValues(1, health.Errors)
	s.InDelta(2.0/3.0, health.SuccessRate(), 0.001)
	s.True(health.Available, "one failure in three is under the default threshold")
}

func (s *LiveSuite) TestProviderBecomesUnavailableWhenErrorsDominate() {
	for range 3 {
		s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1000, false)))
	}
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1000, true)))

	health, err := s.client.Health(s.ctx, "stt", "deepgram", s.model)
	s.Require().NoError(err)

	s.False(health.Available, "a 75% error rate is over the default 50% threshold")
}

func (s *LiveSuite) TestErrorRateThresholdIsConfigurable() {
	strict, err := New(Options{Address: os.Getenv(AddressEnvVar), MaxErrorRate: 0.1})
	s.Require().NoError(err)
	defer strict.Close()

	for range 9 {
		s.Require().NoError(strict.RecordRequest(s.ctx, s.usage(100, 1000, true)))
	}
	s.Require().NoError(strict.RecordRequest(s.ctx, s.usage(100, 1000, false)))

	health, err := strict.Health(s.ctx, "stt", "deepgram", s.model)
	s.Require().NoError(err)
	s.InDelta(0.1, health.ErrorRate(), 0.0001)
	s.False(health.Available, "one failure in ten meets a 10% threshold")
}

func (s *LiveSuite) TestHealthIsTrackedPerModel() {
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1000, false)))

	other, err := s.client.Health(s.ctx, "stt", "deepgram", s.model+"-other")
	s.Require().NoError(err)
	s.Zero(other.Requests, "a sibling model should not inherit failures")
}

func (s *LiveSuite) TestHealthIsTrackedPerModality() {
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1000, false)))

	spoken, err := s.client.Health(s.ctx, "tts", "deepgram", s.model)
	s.Require().NoError(err)
	s.Zero(spoken.Requests, "one modality's failures must not rank another's providers")
}

func (s *LiveSuite) TestUsageAccumulatesEveryBillableUnit() {
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1500, true)))

	synthesis := Usage{
		Modality: "stt", CustomerID: s.customer,
		Provider: "parakeet", Model: s.model,
		LatencyMs: 100, AudioMs: 2500, Characters: 42, CostMicros: 700,
	}
	s.Require().NoError(s.client.RecordRequest(s.ctx, synthesis))

	usage, err := s.client.Usage(s.ctx, "stt", s.customer)
	s.Require().NoError(err)

	s.EqualValues(2, usage.Requests, "usage spans every provider the customer used")
	s.EqualValues(1, usage.Errors)
	s.EqualValues(4000, usage.AudioMs)
	s.EqualValues(42, usage.Characters)
	s.EqualValues(700, usage.CostMicros)
}

func (s *LiveSuite) TestUsageIsScopedToOneModality() {
	s.Require().NoError(s.client.RecordRequest(s.ctx, s.usage(100, 1500, true)))

	usage, err := s.client.Usage(s.ctx, "tts", s.customer)
	s.Require().NoError(err)
	s.Zero(usage.Requests, "text-to-speech spend is reported on its own")
}

func (s *LiveSuite) TestUsageIsZeroForAnUnknownCustomer() {
	usage, err := s.client.Usage(s.ctx, "stt", "nobody-"+s.customer)
	s.Require().NoError(err)

	s.Zero(usage.Requests)
	s.Zero(usage.Errors)
	s.Zero(usage.AudioMs)
}

func (s *LiveSuite) TestCountersExpireSoHealthStaysRecent() {
	short, err := New(Options{Address: os.Getenv(AddressEnvVar), Window: time.Second})
	s.Require().NoError(err)
	defer short.Close()

	s.Require().NoError(short.RecordRequest(s.ctx, s.usage(100, 1000, false)))

	health, err := short.Health(s.ctx, "stt", "deepgram", s.model)
	s.Require().NoError(err)
	s.EqualValues(1, health.Requests)

	time.Sleep(1500 * time.Millisecond)

	health, err = short.Health(s.ctx, "stt", "deepgram", s.model)
	s.Require().NoError(err)
	s.Zero(health.Requests, "observations older than the window should age out")
	s.True(health.Available, "a provider with no recent history is available again")
}

func (s *LiveSuite) TestNewRequiresAnAddress() {
	_, err := New(Options{})
	s.ErrorContains(err, "redis address is required")
}
