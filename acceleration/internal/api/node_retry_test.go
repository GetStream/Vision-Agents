//go:build integration

package api

import (
	"context"
	"io"
	"log/slog"
	"net"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// dialHeld is how long holdingProxy holds a connection: past rueidis' 5 s dial timeout, so the
// dial that waits on it fails.
const dialHeld = 6 * time.Second

// holdingProxy is a TCP forwarder in front of the suites' Redis that holds the nth
// connection accepted for dialHeld, then closes it, as Docker's port forwarding does on a
// loaded machine. Every other connection is forwarded. The first is redisready's PING and the
// second the client's one connection, so the third is the first subscription's dial.
func holdingProxy(t *testing.T, nth int64) string {
	target := os.Getenv("ROUTER_REDIS_ADDR")
	if target == "" {
		t.Skip("ROUTER_REDIS_ADDR must be set")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	t.Cleanup(func() {
		close(done)
		listener.Close()
	})
	var accepted atomic.Int64
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			held := accepted.Add(1) == nth
			go func() {
				defer conn.Close()
				if held {
					select {
					case <-time.After(dialHeld):
					case <-done:
					}
					return
				}
				upstream, err := net.Dial("tcp", target)
				if err != nil {
					return
				}
				defer upstream.Close()
				go io.Copy(upstream, conn)
				io.Copy(conn, upstream)
			}()
		}
	}()
	return listener.Addr().String()
}

// subscribers is how many connections are subscribed to channel, on any database.
func (s *RouterSuite) subscribers(channel string) int64 {
	redis := s.live.ShardRedis()
	counts, err := redis.Do(context.Background(), redis.B().PubsubNumsub().Channel(channel).Build()).AsIntMap()
	s.Require().NoError(err)
	return counts[channel]
}

// neverMoreSubscribers fails if channel has more than most connections subscribed at any time
// within a few resubscribeAfters: a subscription a failed attempt left comes back one
// resubscribeAfter after its dial failed.
func (s *RouterSuite) neverMoreSubscribers(channel string, most int64) {
	s.Never(func() bool { return s.subscribers(channel) > most }, 2500*time.Millisecond, 50*time.Millisecond,
		"%s is subscribed to by a node that is not there", channel)
}

// NodeRetrySuite is about the nodes the suites make again when a Redis dial stalls: the
// attempt that failed must leave no subscription behind.
type NodeRetrySuite struct {
	RouterSuite
}

func TestNodeRetrySuite(t *testing.T) {
	runSuite(t, new(NodeRetrySuite))
}

func (s *NodeRetrySuite) SetupSuite() {
	// A shard of its own, so the relay channels counted here are the second node's alone.
	s.otherShard = "retry"
	s.RouterSuite.SetupSuite()
}

func (s *NodeRetrySuite) TestAFailedAttemptAtANodeLeavesNoRelaySubscription() {
	address := s.shardRedisAddr
	s.shardRedisAddr = holdingProxy(s.T(), 3)
	s.T().Cleanup(func() { s.shardRedisAddr = address })
	prefix := s.relayPrefix + s.otherShard

	s.otherNode()

	for _, channel := range []string{prefix + ":session-events", prefix + ":session-commands"} {
		s.Require().EqualValues(1, s.subscribers(channel), "the node that came up holds %s", channel)
		s.neverMoreSubscribers(channel, 1)
	}
}

func (s *NodeRetrySuite) TestAFailedAttemptAtWorkersLeavesNoRequestSubscription() {
	address, work := s.redisAddr, s.workRedis
	s.redisAddr = holdingProxy(s.T(), 3)
	s.T().Cleanup(func() { s.redisAddr, s.workRedis = address, work })
	var err error
	s.workRedis, err = redisready.NewClient(rueidis.ClientOption{InitAddress: []string{s.redisAddr}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(s.workRedis.Close)
	logger := slog.New(slog.DiscardHandler)
	listener := httptest.NewUnstartedServer(nil)
	s.T().Cleanup(listener.Close)
	directory := s.nodeDirectory(listener, s.otherShard, logger)

	s.deploymentWorkers(dispatch.NewPool(), directory, s.workBus(logger))

	requests := s.relayPrefix + ":hosted:requests:" + listener.Listener.Addr().String()
	s.Require().EqualValues(1, s.subscribers(requests), "the workers that came up take work on %s", requests)
	s.neverMoreSubscribers(requests, 1)
}

func (s *ChatInboundQueueSuite) TestAFailedAttemptAtDeafWorkersLeavesNoRequestSubscription() {
	address, work := s.redisAddr, s.workRedis
	s.redisAddr = holdingProxy(s.T(), 3)
	s.T().Cleanup(func() { s.redisAddr, s.workRedis = address, work })
	var err error
	s.workRedis, err = redisready.NewClient(rueidis.ClientOption{InitAddress: []string{s.redisAddr}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(s.workRedis.Close)

	// deafWorkers cancels the attempt that came up, so no node is left taking work.
	s.deafWorkers(slog.New(slog.DiscardHandler))

	taken := func() []string {
		redis := s.live.ShardRedis()
		channels, err := redis.Do(context.Background(), redis.B().PubsubChannels().Pattern(s.relayPrefix+":hosted:requests:*").Build()).AsStrSlice()
		s.Require().NoError(err)
		return channels
	}
	s.Require().Eventually(func() bool { return len(taken()) == 0 }, settleFor, 50*time.Millisecond,
		"the workers deafWorkers cancelled still take work")
	s.Never(func() bool { return len(taken()) > 0 }, 2500*time.Millisecond, 50*time.Millisecond,
		"a failed attempt's workers take work on %v", taken())
}
