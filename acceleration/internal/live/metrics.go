package live

import (
	"context"
	"errors"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/rueidis"
)

// readPerPod is what every Redis metric's help ends with: what a monitor reads across router
// generations and pods.
const readPerPod = " Only the router generation serving /metrics is read, since a draining one stops serving it, " +
	"so a hand-over restarts the series and never adds two generations up: read a pod's own, and the max over pods."

// commandBuckets are client_golang's default buckets, which start at 5 ms, with three below
// them: Redis on the router's own network answers in well under 5 ms, so the default's first
// bucket alone would hold every healthy command.
var commandBuckets = append([]float64{0.0005, 0.001, 0.0025}, prometheus.DefBuckets...)

// redisMetrics are the round trips of one Client's connections, by client: "shared", or
// "shard" for this shard's own Redis when it has one.
type redisMetrics struct {
	seconds *prometheus.HistogramVec
	errors  *prometheus.CounterVec
}

func newRedisMetrics() redisMetrics {
	return redisMetrics{
		seconds: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "router",
			Name:      "redis_command_seconds",
			Help: "Round trips to Redis, one command or one pipeline each, from sent to answered, by client: shared, " +
				"or shard for this shard's own Redis. Subscriptions are not counted." + readPerPod,
			Buckets: commandBuckets,
		}, []string{"client"}),
		errors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "router",
			Name:      "redis_command_errors_total",
			Help: "Round trips to Redis that failed: a connection lost or refused, a timeout, or an error reply. A missing " +
				"key is not one, nor a command its caller cancelled. Read it as a rate: it starts at zero in each " +
				"generation." + readPerPod,
		}, []string{"client"}),
	}
}

// measure wraps a connection so each of its round trips is recorded under client.
func (m redisMetrics) measure(client rueidis.Client, name string) rueidis.Client {
	return &measured{Client: client, seconds: m.seconds.WithLabelValues(name), errors: m.errors.WithLabelValues(name)}
}

// measured records the round trips of Do and DoMulti, the calls this router makes on these
// connections. A subscription's Dedicated or Receive lasts as long as it is held, which is no
// latency, so those pass through.
type measured struct {
	rueidis.Client
	seconds prometheus.Observer
	errors  prometheus.Counter
}

func (m *measured) Do(ctx context.Context, cmd rueidis.Completed) rueidis.RedisResult {
	start := time.Now()
	result := m.Client.Do(ctx, cmd)
	m.record(start, result.Error())
	return result
}

func (m *measured) DoMulti(ctx context.Context, cmds ...rueidis.Completed) []rueidis.RedisResult {
	start := time.Now()
	results := m.Client.DoMulti(ctx, cmds...)
	var failed error
	for _, result := range results {
		if err := result.Error(); isFailure(err) {
			failed = err
			break
		}
	}
	m.record(start, failed)
	return results
}

func (m *measured) record(start time.Time, err error) {
	m.seconds.Observe(time.Since(start).Seconds())
	if isFailure(err) {
		m.errors.Inc()
	}
}

// isFailure reports whether err says Redis failed. A missing key is an answer, and a caller
// that cancelled, such as a request whose client hung up, says nothing of Redis.
func isFailure(err error) bool {
	return err != nil && !rueidis.IsRedisNil(err) && !errors.Is(err, context.Canceled)
}
