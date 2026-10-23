//go:build integration

package llmrouter

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// recordedHedge is how long a reply may say nothing before these tests' requests are asked
// again: long enough that the original is surely asked first, on the wall clock.
const recordedHedge = 100 * time.Millisecond

// HedgeRecordingSuite races requests across two candidates in real time, recording into
// Postgres, and reads back the rows each attempt left.
type HedgeRecordingSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	customer string
}

func TestHedgeRecordingSuite(t *testing.T) {
	suite.Run(t, new(HedgeRecordingSuite))
}

func (s *HedgeRecordingSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "llmrouter_hedge"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

// rig is alpha/quick and beta/steady behind one target, priced, recording for a customer of
// the test's own, with replies hedged after recordedHedge.
func (s *HedgeRecordingSuite) rig() (*rig, *Router) {
	return s.rigOf("alpha", "beta")
}

// rigOf is rig with only the named candidates, in that order.
func (s *HedgeRecordingSuite) rigOf(candidates ...string) (*rig, *Router) {
	s.customer = "hedge-" + uuid.NewString()
	r := &rig{headers: map[string]time.Duration{}, failing: map[string]error{}, scripts: map[string][]*llmtest.Script{}}
	registry := NewRegistry()
	price := routing.Price{PerMillionInputTokens: 1, PerMillionOutputTokens: 2}
	models := map[string]string{"alpha": "quick", "beta": "steady"}
	var config routing.ModalityConfig
	var members []string
	for _, name := range candidates {
		config.Providers = append(config.Providers,
			routing.ProviderConfig{Provider: name, Model: models[name], Languages: []string{"en"}, Price: price})
		members = append(members, name+"/"+models[name])
		registry.Register(name, func(spec routing.Spec) (Provider, error) { return &pace{rig: r, name: name, model: spec.Model}, nil })
	}
	config.Aliases = map[string]routing.Alias{"voice": {Prefer: members[0], Only: members}}
	router, err := New(Options{Config: config, Registry: registry, Store: s.store, ReplyHedge: recordedHedge})
	s.Require().NoError(err)
	r.session, err = router.Start(routing.WithAttribution(s.ctx, routing.Attribution{APIKey: "key-hedge"}),
		Request{CustomerID: s.customer, Target: "voice"})
	s.Require().NoError(err)
	return r, router
}

// asked puts a reply to the rig, which returns once one of its attempts has said something.
func (s *HedgeRecordingSuite) asked(r *rig) <-chan *llm.Stream {
	return s.askedWith(r, r.params())
}

// askedWith is asked, with params.
func (s *HedgeRecordingSuite) askedWith(r *rig, params llm.ResponseParams) <-chan *llm.Stream {
	answered := make(chan *llm.Stream, 1)
	go func() {
		stream, err := r.session.Create(s.ctx, params)
		s.NoError(err)
		answered <- stream
	}()
	return answered
}

// started waits for a candidate to have been asked.
func (s *HedgeRecordingSuite) started(r *rig, name string) {
	s.Require().Eventually(func() bool { return r.requests(name) == 1 }, 3*time.Second, time.Millisecond, name+" was never asked")
}

// rows waits for the customer's rows, each attempt's by provider, and closes the router.
func (s *HedgeRecordingSuite) rows(router *Router, want int) map[string]store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", s.customer).Scan(s.ctx))
		return len(rows) == want
	}, 5*time.Second, 10*time.Millisecond, "every attempt is recorded")
	router.Close()
	byProvider := map[string]store.Request{}
	for _, row := range rows {
		byProvider[row.Provider] = row
	}
	return byProvider
}

func (s *HedgeRecordingSuite) TestEveryHedgeAttemptIsTagged() {
	r, router := s.rig()
	answered := s.asked(r)
	s.started(r, "alpha")
	s.started(r, "beta")

	r.script("beta").OutputText("Hello")
	stream := <-answered
	r.script("beta").Done()
	s.Equal("beta", drain(stream).Provider)

	rows := s.rows(router, 2)
	loser, winner := rows["alpha"], rows["beta"]
	s.Equal(map[string]string{routing.AttributeHedge: "turn-1", routing.AttributeKept: "false"},
		withoutEstimate(loser.Attributes), "the original was thrown away the moment the hedge spoke")
	s.Equal(map[string]string{routing.AttributeHedge: "turn-1", routing.AttributeKept: "true"},
		withoutEstimate(winner.Attributes))
	s.Equal(routing.ErrorCancelled, loser.ErrorCode)
	for _, row := range []store.Request{loser, winner} {
		s.Equal("reply", row.Purpose, "thrown away or not, it was a reply")
		s.True(row.Billable, "both are billed")
		s.Positive(row.CostMicros)
	}
}

func (s *HedgeRecordingSuite) TestAnOriginalThatStillWinsIsKept() {
	r, router := s.rig()
	answered := s.asked(r)
	s.started(r, "beta")

	r.script("alpha").OutputText("Hello")
	stream := <-answered
	r.script("alpha").Done()
	drain(stream)

	rows := s.rows(router, 2)
	s.Equal("true", rows["alpha"].Attributes[routing.AttributeKept])
	s.Equal("false", rows["beta"].Attributes[routing.AttributeKept])
	s.Equal("turn-1", rows["beta"].Attributes[routing.AttributeHedge])
}

func (s *HedgeRecordingSuite) TestARequestThatWasNeverHedgedIsNotTagged() {
	r, router := s.rig()
	answered := s.asked(r)
	s.started(r, "alpha")

	r.script("alpha").OutputText("Hello")
	stream := <-answered
	r.script("alpha").Done()
	drain(stream)

	row := s.rows(router, 1)["alpha"]
	s.NotContains(row.Attributes, routing.AttributeHedge, "it answered in time, so nothing was asked twice")
	s.NotContains(row.Attributes, routing.AttributeKept)
}

func (s *HedgeRecordingSuite) TestADroppedSpeculativeReplyIsNotKeptWhicheverAttemptWon() {
	r, router := s.rig()
	params := r.params()
	// A speculative reply its ruling dropped, as the agent labels it.
	params.Attributes = func() map[string]string {
		return map[string]string{routing.AttributeSpeculation: "spec-1", routing.AttributeKept: "false"}
	}
	answered := s.askedWith(r, params)
	s.started(r, "alpha")
	s.started(r, "beta")

	r.script("beta").OutputText("Hello")
	stream := <-answered
	r.script("beta").Done()
	drain(stream)

	rows := s.rows(router, 2)
	for provider, row := range rows {
		s.Equal("spec-1", row.Attributes[routing.AttributeSpeculation], provider)
		s.Equal("turn-1", row.Attributes[routing.AttributeHedge], provider)
		s.Equal("false", row.Attributes[routing.AttributeKept], "%s: nobody heard the reply, so neither attempt was kept", provider)
	}
}

func (s *HedgeRecordingSuite) TestALateRequestWithNoOtherCandidateIsNotTagged() {
	r, router := s.rigOf("alpha")
	answered := s.asked(r)
	s.started(r, "alpha")

	time.Sleep(3 * recordedHedge)
	r.script("alpha").OutputText("Late, but alone")
	stream := <-answered
	r.script("alpha").Done()
	drain(stream)

	row := s.rows(router, 1)["alpha"]
	s.NotContains(row.Attributes, routing.AttributeHedge, "nothing else could be asked, so it was asked once")
	s.NotContains(row.Attributes, routing.AttributeKept)
}

// withoutEstimate is a row's attributes less the mark of a count the router made itself,
// which a script that reports no usage leaves on every row.
func withoutEstimate(attributes map[string]string) map[string]string {
	kept := map[string]string{}
	for name, value := range attributes {
		if name != routing.AttributeUsageEstimated {
			kept[name] = value
		}
	}
	return kept
}
