//go:build integration

package main

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/hibiken/asynq"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// clusterEnvVar names a node of a Redis Cluster with more than one shard, which CI does not
// run. Three `redis-server --cluster-enabled yes` joined by `redis-cli --cluster create` make
// one. Name a node that does not hold slot 3477, where asynq:queues
// lives (the second of three): a plain client of the node that does is never redirected, so
// nothing would fail for it.
const clusterEnvVar = "ROUTER_TEST_REDIS_CLUSTER_ADDR"

// QueueRedisSuite is the knowledge url queue on each kind of Redis the router may be given.
type QueueRedisSuite struct {
	suite.Suite
	ctx context.Context
}

func TestQueueRedisSuite(t *testing.T) {
	suite.Run(t, new(QueueRedisSuite))
}

func (s *QueueRedisSuite) SetupSuite() {
	s.ctx = context.Background()
}

func (s *QueueRedisSuite) TestAPlainRedisIsQueuedOnThroughAClientOfOneNode() {
	address := s.address("ROUTER_REDIS_ADDR")

	got := queueRedis(config.Redis{Addr: address, Username: "router", Password: "secret"}, s.connect(address))

	s.Equal(asynq.RedisClientOpt{Addr: address, Username: "router", Password: "secret"}, got)
}

func (s *QueueRedisSuite) TestAReadIsQueuedAndRunOnAPlainRedis() {
	address := s.address("ROUTER_REDIS_ADDR")

	s.queuesAndRuns(queueRedis(config.Redis{Addr: address}, s.connect(address)))
}

func (s *QueueRedisSuite) TestAClusterIsQueuedOnThroughAClusterClient() {
	address := s.address(clusterEnvVar)

	got := queueRedis(config.Redis{Addr: address, Username: "router", Password: "secret"}, s.connect(address))

	s.Equal(asynq.RedisClusterClientOpt{Addrs: []string{address}, Username: "router", Password: "secret"}, got)
}

// A client of one node is answered MOVED for the keys another shard holds, asynq:queues among
// them, so this fails for a cluster of several shards unless the queue knows it is one.
func (s *QueueRedisSuite) TestAReadIsQueuedAndRunOnACluster() {
	address := s.address(clusterEnvVar)

	s.queuesAndRuns(queueRedis(config.Redis{Addr: address}, s.connect(address)))
}

func (s *QueueRedisSuite) TestAPageIsQueuedThroughTheServiceOnAPlainRedis() {
	s.queuesThroughTheService(s.address("ROUTER_REDIS_ADDR"))
}

// The service is built the way the router builds it, so this fails when the router hands it
// anything but the queue the shared client's mode asks for.
func (s *QueueRedisSuite) TestAPageIsQueuedThroughTheServiceOnACluster() {
	s.queuesThroughTheService(s.address(clusterEnvVar))
}

// queuesThroughTheService adds a page to the url service buildKnowledgeURLs makes over the
// Redis at address, which queues a read of it.
func (s *QueueRedisSuite) queuesThroughTheService(address string) {
	s.T().Setenv("EXA_API_KEY", "exa-key")
	settings := config.Defaults()
	settings.Postgres.DSN = testenv.Database(os.Getenv(dsnEnvVar), "queueredis")
	st, err := openStore(s.ctx, settings)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(st.Close()) })

	pages, err := buildKnowledgeURLs(st, config.Redis{Addr: address}, s.connect(address), &metered.Knowledge{}, ingest.Limits{}, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.Require().NotNil(pages)
	s.T().Cleanup(func() { s.Require().NoError(pages.Close()) })

	_, err = pages.Add(s.ctx, "queueredis-"+uuid.NewString(), urls.Subscription{Namespace: "ns", URL: "https://example.com/pricing"})
	s.Require().NoError(err)
}

// queuesAndRuns does to a queue of its own on opt what the url service does to its queue:
// queues a read under the page's id, is refused a second one under the same id, looks the
// first up and runs it.
func (s *QueueRedisSuite) queuesAndRuns(opt asynq.RedisConnOpt) {
	queue := "queueredis-" + uuid.NewString()
	inspector := asynq.NewInspector(opt)
	defer func() {
		_ = inspector.DeleteQueue(queue, true)
		s.Require().NoError(inspector.Close())
	}()
	client := asynq.NewClient(opt)
	defer func() { s.Require().NoError(client.Close()) }()

	read := func() *asynq.Task {
		return asynq.NewTask("read", []byte("https://example.com/pricing"))
	}
	_, err := client.EnqueueContext(s.ctx, read(), asynq.Queue(queue), asynq.TaskID("page"))
	s.Require().NoError(err)
	_, err = client.EnqueueContext(s.ctx, read(), asynq.Queue(queue), asynq.TaskID("page"))
	s.Require().ErrorIs(err, asynq.ErrTaskIDConflict)
	info, err := inspector.GetTaskInfo(queue, "page")
	s.Require().NoError(err)
	s.Equal(asynq.TaskStatePending, info.State)

	ran := make(chan string, 1)
	mux := asynq.NewServeMux()
	mux.HandleFunc("read", func(_ context.Context, task *asynq.Task) error {
		ran <- string(task.Payload())
		return nil
	})
	server := asynq.NewServer(opt, asynq.Config{Queues: map[string]int{queue: 1}, LogLevel: asynq.FatalLevel})
	s.Require().NoError(server.Start(mux))
	defer server.Shutdown()

	select {
	case url := <-ran:
		s.Equal("https://example.com/pricing", url)
	case <-time.After(10 * time.Second):
		s.Fail("the queued read never ran")
	}
}

// address is the Redis the environment names in variable, and skips the test without one.
func (s *QueueRedisSuite) address(variable string) string {
	address := os.Getenv(variable)
	if address == "" {
		s.T().Skipf("%s not set", variable)
	}
	return address
}

// connect is a client of the Redis at address made the way the router makes its own, so it
// finds out the same way whether that Redis is a cluster.
func (s *QueueRedisSuite) connect(address string) rueidis.Client {
	client, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	return client.Redis()
}
