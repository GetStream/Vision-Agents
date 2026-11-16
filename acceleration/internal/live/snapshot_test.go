//go:build integration

package live

import (
	"fmt"
	"os"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

// snapshotClient is a client whose snapshot refreshes only when a test says so.
func (s *LiveSuite) snapshotClient() *Client {
	client, err := New(Options{Address: os.Getenv(AddressEnvVar), Username: os.Getenv("ROUTER_REDIS_USERNAME"),
		Password: os.Getenv("ROUTER_REDIS_PASSWORD"), Refresh: -1})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	return client
}

// models are n speech-to-text models of the test's own.
func (s *LiveSuite) models(n int) []Model {
	models := make([]Model, n)
	for i := range models {
		models[i] = Model{Modality: "stt", Provider: "deepgram", Model: fmt.Sprintf("%s-%d", s.model, i)}
	}
	return models
}

// reads is how many round trips of kind health has made, in every client of the process.
func reads(kind string) float64 { return testutil.ToFloat64(healthReads.WithLabelValues(kind)) }

func (s *LiveSuite) failed(model Model) Usage {
	usage := s.usage(100, 1000, false)
	usage.Model = model.Model
	return usage
}

func (s *LiveSuite) TestAColdSnapshotReadsAWalksModelsInOneRoundTripThenNoMore() {
	client := s.snapshotClient()
	models := s.models(21)
	for range 3 {
		s.Require().NoError(s.client.RecordRequest(s.ctx, s.failed(models[4])))
	}

	before := reads(readMiss)
	healths, err := client.Healths(s.ctx, models)
	s.Require().NoError(err)
	s.Equal(before+1, reads(readMiss), "every model missing from the snapshot is read in one pipelined round trip")
	s.Len(healths, 21)
	s.False(healths[4].Available)
	s.EqualValues(3, healths[4].Requests)
	s.True(healths[5].Available, "an unseen model is available")

	again, err := client.Healths(s.ctx, models)
	s.Require().NoError(err)
	s.Equal(before+1, reads(readMiss), "a warm snapshot answers without Redis")
	s.Equal(healths, again)
}

func (s *LiveSuite) TestAModelNamedTwiceInAWalkIsReadOnce() {
	client := s.snapshotClient()
	model := s.models(1)[0]

	healths, err := client.Healths(s.ctx, []Model{model, model})
	s.Require().NoError(err)
	s.Equal(healths[0], healths[1], "a model's instances in two regions share its health")
}

func (s *LiveSuite) TestARefreshReadsWhatOtherRoutersRecordedInOneRoundTrip() {
	client := s.snapshotClient()
	models := s.models(3)
	_, err := client.Healths(s.ctx, models)
	s.Require().NoError(err)

	for range 3 {
		s.Require().NoError(s.client.RecordRequest(s.ctx, s.failed(models[1])))
	}
	healths, err := client.Healths(s.ctx, models)
	s.Require().NoError(err)
	s.True(healths[1].Available, "another router's failures wait for the refresh")

	before := reads(readRefresh)
	s.Require().NoError(client.Refresh(s.ctx))
	s.Equal(before+1, reads(readRefresh))
	healths, err = client.Healths(s.ctx, models)
	s.Require().NoError(err)
	s.False(healths[1].Available)
}

func (s *LiveSuite) TestTrackedModelsAreReadByTheFirstRefresh() {
	client := s.snapshotClient()
	models := s.models(4)
	client.Track(models...)
	s.Require().NoError(client.Refresh(s.ctx))

	before := reads(readMiss)
	_, err := client.Healths(s.ctx, models)
	s.Require().NoError(err)
	s.Equal(before, reads(readMiss), "a router's models are in the snapshot before its first walk")
}

func (s *LiveSuite) TestAFatalFailureRecordedHereTakesTheModelOutAtOnce() {
	client := s.snapshotClient()
	other := s.snapshotClient()
	model := s.models(1)[0]
	for _, router := range []*Client{client, other} {
		healths, err := router.Healths(s.ctx, []Model{model})
		s.Require().NoError(err)
		s.Require().True(healths[0].Available)
	}

	fatal := s.failed(model)
	fatal.Fatal = true
	s.Require().NoError(client.RecordRequest(s.ctx, fatal))

	healths, err := client.Healths(s.ctx, []Model{model})
	s.Require().NoError(err)
	s.False(healths[0].Available, "the router that saw it stops routing there before any refresh")
	healths, err = other.Healths(s.ctx, []Model{model})
	s.Require().NoError(err)
	s.True(healths[0].Available, "another router learns it from its next refresh")
	s.Require().NoError(other.Refresh(s.ctx))
	healths, err = other.Healths(s.ctx, []Model{model})
	s.Require().NoError(err)
	s.False(healths[0].Available)
}

func (s *LiveSuite) TestAFailedRefreshKeepsTheLastSnapshotUntilTheWindowPasses() {
	client := s.snapshotClient()
	model := s.models(1)[0]
	for range 3 {
		s.Require().NoError(s.client.RecordRequest(s.ctx, s.failed(model)))
	}
	healths, err := client.Healths(s.ctx, []Model{model})
	s.Require().NoError(err)
	s.Require().False(healths[0].Available)

	client.shard.Close()
	before := testutil.ToFloat64(healthRefreshErrors)
	s.Error(client.Refresh(s.ctx))
	s.Equal(before+1, testutil.ToFloat64(healthRefreshErrors))
	healths, err = client.Healths(s.ctx, []Model{model})
	s.Require().NoError(err)
	s.False(healths[0].Available, "the last snapshot answers")

	started := time.Now()
	client.now = func() time.Time { return started.Add(client.options.Window + time.Second) }
	healths, err = client.Healths(s.ctx, []Model{model})
	s.Require().NoError(err)
	s.Equal(Health{Provider: model.Provider, Model: model.Model, Available: true}, healths[0],
		"past the window a model no read has reached is unmeasured, so a broken stats path cannot take routing down")
	s.GreaterOrEqual(client.snapshotAge(), client.options.Window)
}

func (s *LiveSuite) TestAFailedMissReadIsUnmeasuredAndSaysWhy() {
	client := s.snapshotClient()
	client.shard.Close()
	model := s.models(1)[0]

	healths, err := client.Healths(s.ctx, []Model{model})
	s.Error(err)
	s.Equal(Health{Provider: model.Provider, Model: model.Model, Available: true}, healths[0])
}

func (s *LiveSuite) TestTheBackgroundRefreshKeepsTheSnapshotCurrent() {
	client, err := New(Options{Address: os.Getenv(AddressEnvVar), Username: os.Getenv("ROUTER_REDIS_USERNAME"),
		Password: os.Getenv("ROUTER_REDIS_PASSWORD"), Refresh: 20 * time.Millisecond})
	s.Require().NoError(err)
	defer client.Close()
	model := s.models(1)[0]
	_, err = client.Healths(s.ctx, []Model{model})
	s.Require().NoError(err)

	for range 3 {
		s.Require().NoError(s.client.RecordRequest(s.ctx, s.failed(model)))
	}
	s.Eventually(func() bool {
		healths, err := client.Healths(s.ctx, []Model{model})
		return err == nil && !healths[0].Available
	}, 2*time.Second, 10*time.Millisecond)
	s.Less(client.snapshotAge(), time.Second)
}
