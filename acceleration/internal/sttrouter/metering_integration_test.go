//go:build integration

package sttrouter

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing/pricingtest"
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
	live   *live.Client
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
	listed, err := routing.Prices(s.config, slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil)))
	s.Require().NoError(err)
	// At the vendors' rates, so each cost below is what a vendor's published rate makes of the
	// audio it took.
	s.prices, err = pricing.NewList("vendor", pricing.AllAtCost(listed.Entries()), slog.New(slog.NewTextHandler(&bytes.Buffer{}, nil)))
	s.Require().NoError(err)

	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	s.live, err = live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.T().Cleanup(s.live.Close)
}

// health is how many requests the model's live health has counted.
func (s *MeteringSuite) health(provider, model string) int64 {
	health, err := s.live.Health(s.ctx, string(routing.STT), provider, model)
	s.Require().NoError(err)
	return health.Requests
}

func (s *MeteringSuite) SetupTest() { s.logs = &bytes.Buffer{} }

// flux is the built-in config's Flux, which is priced by the hour of audio.
func (s *MeteringSuite) flux() routing.ProviderConfig { return s.builtIn("deepgram/flux-general-en") }

// estimated is how many Flux turns were billed on the router's own count because Flux did not
// say how long they were.
func estimated() float64 {
	return counted("router_usage_estimated_total", "deepgram", "flux-general-en", map[string]string{"reason": "missing"})
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

// builtIn is the built-in config's transcriber of that name.
func (s *MeteringSuite) builtIn(name string) routing.ProviderConfig {
	for _, provider := range s.config[routing.STT].Providers {
		if provider.Name() == name {
			return provider
		}
	}
	s.FailNow("the built-in config has no " + name)
	return routing.ProviderConfig{}
}

// measuredSTT is a transcriber whose vendor bills the audio it is sent and reports none of it.
type measuredSTT struct{ quietSTT }

func (*measuredSTT) Metering() stt.Metering { return stt.MeteredByRouter }

// counted is the metering counter of that name for the item, with the labels given.
func counted(name, provider, model string, labels map[string]string) float64 {
	families, _ := prometheus.DefaultGatherer.Gather()
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
	metrics:
		for _, metric := range family.GetMetric() {
			found := map[string]string{}
			for _, label := range metric.GetLabel() {
				found[label.GetName()] = label.GetValue()
			}
			if found["modality"] != "stt" || found["provider"] != provider || found["model"] != model {
				continue
			}
			for key, value := range labels {
				if found[key] != value {
					continue metrics
				}
			}
			return metric.GetCounter().GetValue()
		}
	}
	return 0
}

// billed is the customer's successful rows, in the order they were written.
func (s *MeteringSuite) billed(customer string) []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ? AND success", customer).Order("id").Scan(s.ctx))
	return rows
}

// TestARouterMeasuredTranscriberIsBilledEveryMillisecondItWasSent: Ink 2 reports nothing of a
// turn's length, and Cartesia bills the audio it is sent, silence included. Two turns and the
// half second the caller spoke before hanging up are each billed what was streamed, and add
// up to it.
func (s *MeteringSuite) TestARouterMeasuredTranscriberIsBilledEveryMillisecondItWasSent() {
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.STT, s.store, s.live, s.prices, slog.New(slog.NewTextHandler(s.logs, nil)))
	health := s.health("cartesia", "ink-2")
	provider := &measuredSTT{quietSTT{emitter: stt.NewEmitter(64)}}
	owner := routing.Owner{CustomerID: customer, Attribution: routing.Attribution{APIKey: "key-1"}}
	session := newSession(provider, s.builtIn("cartesia/ink-2"), owner, recorder, nil, nil)
	estimatedBefore := counted("router_usage_estimated_total", "cartesia", "ink-2", nil)
	missingBefore := counted("router_usage_missing_total", "cartesia", "ink-2", nil)

	s.Require().NoError(session.ProcessAudio(speech(1_000), stt.Participant{}))
	provider.emitter.Send(stt.Transcript{Text: "book a table", Mode: stt.ModeFinal, AudioDurationMs: 640})
	s.settled(session)
	provider.emitter.Send(stt.Transcript{Text: "please", Mode: stt.ModeFinal})
	s.settled(session)
	s.Require().NoError(session.ProcessAudio(speech(1_500), stt.Participant{}))
	provider.emitter.Send(stt.Transcript{Text: "for two", Mode: stt.ModeFinal})
	s.settled(session)
	s.Require().NoError(session.ProcessAudio(speech(500), stt.Participant{}))
	s.Require().NoError(session.Close())
	for range session.Events() {
	}
	recorder.Close()

	rows := s.billed(customer)
	var audio, costs []int64
	for _, row := range rows {
		audio = append(audio, row.AudioMs)
		costs = append(costs, row.CostMicros)
		s.Equal(map[string]string{routing.AttributeMeasuredBy: routing.MeasuredByRouter}, row.Attributes)
	}
	s.Equal([]int64{1_000, 0, 1_500, 500}, audio,
		"what was streamed, not the 640 the vendor said; a turn that followed none is the zero it was; "+
			"what the caller said before hanging up is billed as the session ends")
	var priced []int64
	for _, ms := range audio {
		priced = append(priced, pricingtest.Micros(s.T(), s.prices, routing.ItemOf(routing.STT, s.builtIn("cartesia/ink-2"), nil),
			pricing.Quantities{pricing.AudioMs: ms}))
	}
	s.Equal(priced, costs, "what was streamed at Cartesia's rate")
	s.Equal(estimatedBefore, counted("router_usage_estimated_total", "cartesia", "ink-2", nil), "the router's count is the bill, not an estimate")
	s.Equal(missingBefore, counted("router_usage_missing_total", "cartesia", "ink-2", nil), "a zero the router counted is no missing usage")
	s.NotContains(s.logs.String(), "level=ERROR")
	s.Equal(health+3, s.health("cartesia", "ink-2"), "the three turns measure the vendor, and the audio after them does not")
}

// TestAVendorThatMeasuresItsTurnsIsBilledOnlyThem: a vendor that reports exactly what it
// bills on each turn is billed those turns, and the half second the caller spoke after the
// last one before hanging up is no row of its own.
func (s *MeteringSuite) TestAVendorThatMeasuresItsTurnsIsBilledOnlyThem() {
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.STT, s.store, s.live, s.prices, slog.New(slog.NewTextHandler(s.logs, nil)))
	provider := &quietSTT{emitter: stt.NewEmitter(64)}
	session := newSession(provider, s.flux(), routing.Owner{CustomerID: customer}, recorder, nil, nil)

	s.Require().NoError(session.ProcessAudio(speech(1_000), stt.Participant{}))
	provider.emitter.Send(stt.Transcript{Text: "book a table", Mode: stt.ModeFinal, AudioDurationMs: 960})
	s.settled(session)
	s.Require().NoError(session.ProcessAudio(speech(500), stt.Participant{}))
	s.Require().NoError(session.Close())
	for range session.Events() {
	}
	recorder.Close()

	rows := s.billed(customer)
	s.Require().Len(rows, 1, "no row of what was streamed after the last turn")
	s.EqualValues(960, rows[0].AudioMs, "the turn the vendor measured keeps its figure")
	s.Empty(rows[0].Attributes)
}

// TestAFailoverBillsEachVendorTheAudioItTook: Ink 2 takes 700 ms and refuses the session,
// and Gemini takes over for 300 ms more. Each is billed what it was sent.
func (s *MeteringSuite) TestAFailoverBillsEachVendorTheAudioItTook() {
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.STT, s.store, nil, s.prices, slog.New(slog.NewTextHandler(s.logs, nil)))
	first := &measuredSTT{quietSTT{emitter: stt.NewEmitter(64)}}
	next := &measuredSTT{quietSTT{emitter: stt.NewEmitter(64)}}
	gemini := s.builtIn("gemini/gemini-3.5-transcribe-live")
	reopen := func([]string) (stt.STT, routing.ProviderConfig, error) { return next, gemini, nil }
	session := newSession(first, s.builtIn("cartesia/ink-2"), routing.Owner{CustomerID: customer}, recorder, reopen, nil)

	s.Require().NoError(session.ProcessAudio(speech(700), stt.Participant{}))
	first.emitter.Send(stt.Error{Err: errors.New("account suspended"), Context: "session", Fatal: true})
	for event := range session.Events() {
		if _, ok := event.(stt.Failover); ok {
			break
		}
	}
	s.Require().NoError(session.ProcessAudio(speech(300), stt.Participant{}))
	next.emitter.Send(stt.Transcript{Text: "book a table", Mode: stt.ModeFinal})
	s.settled(session)
	s.Require().NoError(session.Close())
	for range session.Events() {
	}
	recorder.Close()

	rows := s.billed(customer)
	s.Require().Len(rows, 2)
	s.Equal("ink-2", rows[0].Model)
	s.EqualValues(700, rows[0].AudioMs, "the vendor that refused is billed what it took before it did")
	s.Equal("gemini-3.5-transcribe-live", rows[1].Model)
	s.EqualValues(300, rows[1].AudioMs, "the next vendor is billed only what it took")
	s.Equal(pricingtest.Micros(s.T(), s.prices, routing.ItemOf(routing.STT, gemini, nil), pricing.Quantities{pricing.AudioMs: 300}),
		rows[1].CostMicros, "300 ms at Gemini's rate")
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
	s.Equal(pricingtest.Micros(s.T(), s.prices, routing.ItemOf(routing.STT, s.flux(), nil), pricing.Quantities{pricing.AudioMs: 1_500}),
		rows[0].CostMicros, "1.5 s at Flux's rate")
	s.EqualValues(480, rows[1].AudioMs, "a turn the vendor measured keeps its figure")
	s.Empty(rows[1].Attributes)
	s.Equal(before+1, estimated())
	s.Contains(s.logs.String(), "billable work recorded an estimate", "the adapter is told to get fixed")
}
