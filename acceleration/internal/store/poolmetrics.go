package store

import (
	"database/sql"

	"github.com/prometheus/client_golang/prometheus"
)

// readPerPod is what every pool metric's help ends with: what a monitor reads across router
// generations and pods.
const readPerPod = " Only the router generation serving /metrics is read, since a draining one stops serving it, " +
	"so a hand-over restarts the series and never adds two generations up: read a pod's own, and the max over pods."

var poolLabels = []string{"pool", "shard"}

var (
	poolInUse = prometheus.NewDesc("router_db_pool_in_use_connections",
		"Connections of a Postgres pool running a query or held by a transaction, by pool (shared, shard or peer) "+
			"and shard. At max open, the next query waits."+readPerPod, poolLabels, nil)
	poolIdle = prometheus.NewDesc("router_db_pool_idle_connections",
		"Connections of a Postgres pool open and waiting for a query."+readPerPod, poolLabels, nil)
	poolMaxOpen = prometheus.NewDesc("router_db_pool_max_open_connections",
		"The most connections a Postgres pool opens. Each generation has its own pool, so a pod holds up to "+
			"twice this during a hand-over, which these series do not show."+readPerPod, poolLabels, nil)
	poolWaits = prometheus.NewDesc("router_db_pool_waits_total",
		"Queries that waited for a connection because every one of a Postgres pool was in use. Read it as a rate: "+
			"it starts at zero in each generation."+readPerPod, poolLabels, nil)
	poolWaitSeconds = prometheus.NewDesc("router_db_pool_wait_seconds_total",
		"Time queries spent waiting for a connection of a Postgres pool. Read it as a rate: it starts at zero in "+
			"each generation."+readPerPod, poolLabels, nil)
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
		stats := p.db.Stats()
		metrics <- prometheus.MustNewConstMetric(poolInUse, prometheus.GaugeValue, float64(stats.InUse), p.name, p.shard)
		metrics <- prometheus.MustNewConstMetric(poolIdle, prometheus.GaugeValue, float64(stats.Idle), p.name, p.shard)
		metrics <- prometheus.MustNewConstMetric(poolMaxOpen, prometheus.GaugeValue, float64(stats.MaxOpenConnections), p.name, p.shard)
		metrics <- prometheus.MustNewConstMetric(poolWaits, prometheus.CounterValue, float64(stats.WaitCount), p.name, p.shard)
		metrics <- prometheus.MustNewConstMetric(poolWaitSeconds, prometheus.CounterValue, stats.WaitDuration.Seconds(), p.name, p.shard)
	}
}
