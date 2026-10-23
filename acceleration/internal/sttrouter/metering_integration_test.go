//go:build integration

package sttrouter

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// MeteringSuite records turns through a session over the built-in price list into a
// database of its own, and reads back the rows they stored.
type MeteringSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	config routing.Config
	prices *pricing.List
	logs   *bytes.Buffer
}

func TestMeteringSuite(t *testing.T) {
	suite.Run(t, new(MeteringSuite))
}

func (s *MeteringSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "sttmeter"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })

	s.config, err = routing.DefaultConfig()
	s.Require().NoError(err)
	s.prices, err = routing.Prices(s.config, slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil)))
	s.Require().NoError(err)
}

func (s *MeteringSuite) SetupTest() { s.logs = &bytes.Buffer{} }

// flux is the built-in config's Flux, which is priced by the hour of audio.
func (s *MeteringSuite) flux() routing.ProviderConfig {
	for _, provider := range s.config[routing.STT].Providers {
		if provider.Name() == "deepgram/flux-general-en" {
			return provider
		}
	}
	s.FailNow("the built-in config has no deepgram/flux-general-en")
	return routing.ProviderConfig{}
}

// estimated is how many Flux turns were billed on the router's own count because Flux did not
// say how long they were.
func estimated() float64 {
	families, _ := prometheus.DefaultGatherer.Gather()
	for _, family := range families {
		if family.GetName() != "router_usage_estimated_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["modality"] == "stt" && labels["model"] == "flux-general-en" && labels["reason"] == "missing" {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
}

// speech is that much caller audio at the rate every transcriber is fed.
func speech(ms int) stt.PcmData {
	return stt.PcmData{Samples: make([]int16, stt.SampleRate*ms/1000), SampleRate: stt.SampleRate, Channels: 1}
}

// settled waits for the session to pass on the next settled turn.
func (s *MeteringSuite) settled(session *Session) {
	for event := range session.Events() {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			return
		}
	}
	s.FailNow("the session closed before the turn settled")
}

func (s *MeteringSuite) TestATurnTheVendorDidNotMeasureIsBilledForWhatWasStreamed() {
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.STT, s.store, nil, s.prices, slog.New(slog.NewTextHandler(s.logs, nil)))
	provider := &quietSTT{emitter: stt.NewEmitter(64)}
	session := newSession(provider, s.flux(), routing.Owner{CustomerID: customer}, recorder, nil, nil)
	before := estimated()

	s.Require().NoError(session.ProcessAudio(speech(1_000), stt.Participant{}))
	s.Require().NoError(session.ProcessAudio(speech(500), stt.Participant{}))
	provider.emitter.Send(stt.Transcript{Text: "book a table", Mode: stt.ModeFinal})
	s.settled(session)
	s.Require().NoError(session.ProcessAudio(speech(500), stt.Participant{}))
	provider.emitter.Send(stt.Transcript{Text: "for two", Mode: stt.ModeFinal, AudioDurationMs: 480})
	s.settled(session)
	s.Require().NoError(session.Close())
	recorder.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Order("id").Scan(s.ctx))
	s.Require().Len(rows, 2)
	s.EqualValues(1_500, rows[0].AudioMs, "the audio streamed since the session opened")
	s.Equal(map[string]string{routing.AttributeUsageEstimated: "true"}, rows[0].Attributes)
	s.True(rows[0].Billable)
	s.EqualValues(115, rows[0].CostMicros, "1.5 s at $0.276 an hour")
	s.EqualValues(480, rows[1].AudioMs, "a turn the vendor measured keeps its figure")
	s.Empty(rows[1].Attributes)
	s.Equal(before+1, estimated())
	s.Contains(s.logs.String(), "billable work recorded an estimate", "the adapter is told to get fixed")
}
