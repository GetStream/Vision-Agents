package store

import (
	"database/sql"

	"github.com/prometheus/client_golang/prometheus"
)

// readPerPod is what every pool metric's help ends with: why a hand-over does not look like
// exhaustion.
const readPerPod = " Only the router generation serving /metrics is read, as a draining one stops serving it, so a " +
	"hand-over never adds two generations up."

var poolLabels = []string{"pool", "shard"}

var (
	poolInUse = prometheus.NewDesc("router_db_pool_in_use_connections",
		"Connections of a Postgres pool running a query or held by a transaction, by pool (shared, shard or peer) "+
			"and shard. At max open the next query waits: alert on the max over pods of in use / max open, never "+
			"on a sum."+readPerPod, poolLabels, nil)
	poolIdle = prometheus.NewDesc("router_db_pool_idle_connections",
		"Connections of a Postgres pool open and waiting for a query."+readPerPod, poolLabels, nil)
	poolMaxOpen = prometheus.NewDesc("router_db_pool_max_open_connections",
		"The most connections a Postgres pool opens. A pod holds up to twice this during a hand-over, a pool per "+
			"generation, and only the new generation's is served."+readPerPod, poolLabels, nil)
	poolWaits = prometheus.NewDesc("router_db_pool_waits_total",
		"Queries that waited for a connection because every one of a Postgres pool was in use. Alert on its rate "+
			"per pod: it starts at zero in each generation."+readPerPod, poolLabels, nil)
	poolWaitSeconds = prometheus.NewDesc("router_db_pool_wait_seconds_total",
		"Time queries spent waiting for a connection of a Postgres pool. Alert on its rate per pod, the seconds "+
			"waited each second: it starts at zero in each generation."+readPerPod, poolLabels, nil)
)

// pool is one Postgres pool a store opened, as its metrics name it.
type pool struct {
	name  string
	shard string
	db    *sql.DB
}

// poolCollector reads the pools of one store each time /metrics is scraped.
type poolCollector struct {
	pools []pool
}

// Collector serves the saturation of every Postgres pool the store opened: the shared one,
// this shard's when it is a database of its own, and each other shard's.
func (s *Store) Collector() prometheus.Collector {
	pools := []pool{{name: "shared", db: s.db.DB}}
	if s.Sharded() {
		pools = append(pools, pool{name: "shard", shard: s.shardNames[0], db: s.shard.DB})
	}
	for i, peer := range s.peers {
		pools = append(pools, pool{name: "peer", shard: s.shardNames[i+1], db: peer.DB})
	}
	return poolCollector{pools: pools}
}

func (c poolCollector) Describe(descs chan<- *prometheus.Desc) {
	for _, desc := range []*prometheus.Desc{poolInUse, poolIdle, poolMaxOpen, poolWaits, poolWaitSeconds} {
		descs <- desc
	}
}

func (c poolCollector) Collect(metrics chan<- prometheus.Metric) {
	for _, p := range c.pools {
		emit := func(desc *prometheus.Desc, kind prometheus.ValueType, value float64) {
			metrics <- prometheus.MustNewConstMetric(desc, kind, value, p.name, p.shard)
		}
		stats := p.db.Stats()
		emit(poolInUse, prometheus.GaugeValue, float64(stats.InUse))
		emit(poolIdle, prometheus.GaugeValue, float64(stats.Idle))
		emit(poolMaxOpen, prometheus.GaugeValue, float64(stats.MaxOpenConnections))
		emit(poolWaits, prometheus.CounterValue, float64(stats.WaitCount))
		emit(poolWaitSeconds, prometheus.CounterValue, stats.WaitDuration.Seconds())
	}
}
