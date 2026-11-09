//go:build integration

package live

import (
	"context"
	"crypto/rand"
	"io"
	"net"
	"os"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/rueidis"
)

// measuredClient is a Client of the test's own, with no background refresh, so the only round
// trips its metrics hold are the test's.
func (s *LiveSuite) measuredClient(options Options) *Client {
	options.Username, options.Password = os.Getenv("ROUTER_REDIS_USERNAME"), os.Getenv("ROUTER_REDIS_PASSWORD")
	options.Shard.Username, options.Shard.Password = options.Username, options.Password
	options.Refresh = -1
	client, err := New(options)
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	return client
}

// redisSeries reads a client's metrics as /metrics would serve them, by metric and client: a
// histogram as how many round trips it holds.
func (s *LiveSuite) redisSeries(client *Client) map[string]float64 {
	registry := prometheus.NewPedanticRegistry()
	for _, collector := range client.Collectors() {
		s.Require().NoError(registry.Register(collector))
	}
	families, err := registry.Gather()
	s.Require().NoError(err)
	found := map[string]float64{}
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			name := family.GetName() + " " + metric.GetLabel()[0].GetValue()
			found[name] = metric.GetCounter().GetValue() + float64(metric.GetHistogram().GetSampleCount())
		}
	}
	return found
}

// proxy passes connections through to the suite's Redis until stop, which closes every one, as
// a Redis that went away does.
func (s *LiveSuite) proxy() (address string, stop func()) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	var mu sync.Mutex
	var open []net.Conn
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			upstream, err := net.Dial("tcp", os.Getenv(AddressEnvVar))
			if err != nil {
				_ = conn.Close()
				continue
			}
			mu.Lock()
			open = append(open, conn, upstream)
			mu.Unlock()
			go func() { _, _ = io.Copy(upstream, conn) }()
			go func() { _, _ = io.Copy(conn, upstream) }()
		}
	}()
	stop = sync.OnceFunc(func() {
		_ = listener.Close()
		mu.Lock()
		defer mu.Unlock()
		for _, conn := range open {
			_ = conn.Close()
		}
	})
	s.T().Cleanup(stop)
	return listener.Addr().String(), stop
}

func (s *LiveSuite) TestEachRoundTripIsTimedOnceWhateverItCarries() {
	client := s.measuredClient(Options{Address: os.Getenv(AddressEnvVar)})
	redis := client.Redis()
	key := "metrics-" + rand.Text()

	s.Require().NoError(redis.Do(s.ctx, redis.B().Set().Key(key).Value("1").Px(time.Minute).Build()).Error())
	for _, result := range redis.DoMulti(s.ctx,
		redis.B().Incr().Key(key).Build(), redis.B().Incr().Key(key).Build(), redis.B().Get().Key(key).Build()) {
		s.Require().NoError(result.Error())
	}

	s.Equal(map[string]float64{
		"router_redis_command_seconds shared":      2,
		"router_redis_command_errors_total shared": 0,
	}, s.redisSeries(client))
}

func (s *LiveSuite) TestAMissingKeyIsNoError() {
	client := s.measuredClient(Options{Address: os.Getenv(AddressEnvVar)})
	redis := client.Redis()
	missing := "metrics-" + rand.Text()

	s.Require().True(rueidis.IsRedisNil(redis.Do(s.ctx, redis.B().Get().Key(missing).Build()).Error()))
	get := redis.B().Get().Key(missing).Build()
	for _, result := range redis.DoMulti(s.ctx, get, redis.B().Get().Key(missing).Build()) {
		s.Require().True(rueidis.IsRedisNil(result.Error()))
	}

	s.Equal(2.0, s.redisSeries(client)["router_redis_command_seconds shared"])
	s.Zero(s.redisSeries(client)["router_redis_command_errors_total shared"])
}

func (s *LiveSuite) TestAnErrorReplyIsAnErrorWhereverItIsInAPipeline() {
	client := s.measuredClient(Options{Address: os.Getenv(AddressEnvVar)})
	redis := client.Redis()
	word := "metrics-" + rand.Text()
	s.Require().NoError(redis.Do(s.ctx, redis.B().Set().Key(word).Value("not a number").Px(time.Minute).Build()).Error())

	s.Require().Error(redis.Do(s.ctx, redis.B().Incr().Key(word).Build()).Error())
	s.Equal(1.0, s.redisSeries(client)["router_redis_command_errors_total shared"])

	// The failure is the second of the pipeline's answers, after a missing key.
	missing := redis.B().Get().Key("metrics-" + rand.Text()).Build()
	results := redis.DoMulti(s.ctx, missing, redis.B().Incr().Key(word).Build())
	s.Require().Error(results[1].Error())
	s.Equal(2.0, s.redisSeries(client)["router_redis_command_errors_total shared"])
}

func (s *LiveSuite) TestACommandItsCallerCancelledIsNoError() {
	client := s.measuredClient(Options{Address: os.Getenv(AddressEnvVar)})
	redis := client.Redis()
	cancelled, cancel := context.WithCancel(s.ctx)
	cancel()

	s.Require().ErrorIs(redis.Do(cancelled, redis.B().Get().Key("metrics-"+rand.Text()).Build()).Error(), context.Canceled)
	for _, result := range redis.DoMulti(cancelled, redis.B().Get().Key("metrics-"+rand.Text()).Build()) {
		s.Require().ErrorIs(result.Error(), context.Canceled)
	}

	s.Zero(s.redisSeries(client)["router_redis_command_errors_total shared"])
}

func (s *LiveSuite) TestACommandToARedisThatWentAwayIsAnError() {
	address, stop := s.proxy()
	client := s.measuredClient(Options{Address: address})
	redis := client.Redis()
	s.Require().NoError(redis.Do(s.ctx, redis.B().Ping().Build()).Error())
	s.Zero(s.redisSeries(client)["router_redis_command_errors_total shared"])

	stop()
	// Bounded, as a request's context is: a read-only command is retried on a lost connection.
	bounded, cancel := context.WithTimeout(s.ctx, 5*time.Second)
	defer cancel()
	s.Require().Error(redis.Do(bounded, redis.B().Ping().Build()).Error())
	s.Equal(1.0, s.redisSeries(client)["router_redis_command_errors_total shared"])
}

func (s *LiveSuite) TestAShardsOwnRedisIsMeasuredApartFromTheSharedOne() {
	address, _ := s.proxy()
	client := s.measuredClient(Options{Address: os.Getenv(AddressEnvVar), Shard: Shard{Address: address}})
	shard := client.ShardRedis()

	s.Require().NoError(shard.Do(s.ctx, shard.B().Ping().Build()).Error())

	s.Equal(map[string]float64{
		"router_redis_command_seconds shared":      0,
		"router_redis_command_errors_total shared": 0,
		"router_redis_command_seconds shard":       1,
		"router_redis_command_errors_total shard":  0,
	}, s.redisSeries(client))
}
