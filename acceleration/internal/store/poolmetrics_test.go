//go:build integration

package store

import (
	"context"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// poolSeries reads a store's pool metrics as /metrics would serve them, by metric, pool and
// shard ("router_db_pool_waits_total shared "), from a registry of the test's own.
func (s *StoreSuite) poolSeries(store *Store) map[string]float64 {
	registry := prometheus.NewPedanticRegistry()
	s.Require().NoError(registry.Register(store.Collector()))
	families, err := registry.Gather()
	s.Require().NoError(err)
	found := map[string]float64{}
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			value := metric.GetGauge().GetValue() + metric.GetCounter().GetValue()
			found[family.GetName()+" "+labels["pool"]+" "+labels["shard"]] = value
		}
	}
	return found
}

func (s *StoreSuite) TestAStoreOfOneDatabaseServesOnePoolOfItsDefaultSize() {
	one, err := Open(s.dsn)
	s.Require().NoError(err)
	defer one.Close()

	s.Equal(map[string]float64{
		"router_db_pool_in_use_connections shared ":   0,
		"router_db_pool_idle_connections shared ":     0,
		"router_db_pool_max_open_connections shared ": 40,
		"router_db_pool_waits_total shared ":          0,
		"router_db_pool_wait_seconds_total shared ":   0,
	}, s.poolSeries(one))
}

func (s *StoreSuite) TestAQueryWaitingForAPoolsLastConnectionIsCountedAsAWait() {
	one, err := Open(s.dsn)
	s.Require().NoError(err)
	defer one.Close()
	one.DB().DB.SetMaxOpenConns(1)
	held, err := one.DB().DB.Conn(s.ctx)
	s.Require().NoError(err)

	waited, cancel := context.WithTimeout(s.ctx, 200*time.Millisecond)
	defer cancel()
	_, err = one.DB().ExecContext(waited, "SELECT 1")
	s.Require().ErrorIs(err, context.DeadlineExceeded)

	series := s.poolSeries(one)
	s.Equal(1.0, series["router_db_pool_in_use_connections shared "])
	s.Equal(0.0, series["router_db_pool_idle_connections shared "])
	s.Equal(1.0, series["router_db_pool_max_open_connections shared "])
	s.Equal(1.0, series["router_db_pool_waits_total shared "])
	s.GreaterOrEqual(series["router_db_pool_wait_seconds_total shared "], 0.15)

	s.Require().NoError(held.Close())
	series = s.poolSeries(one)
	s.Equal(0.0, series["router_db_pool_in_use_connections shared "])
	s.Equal(1.0, series["router_db_pool_idle_connections shared "])
}

func (s *ShardedStoreSuite) TestAShardedStoreServesEachOfItsPoolsByShard() {
	held, err := s.store.DB().DB.Conn(s.ctx)
	s.Require().NoError(err)
	defer held.Close()

	series := s.poolSeries(s.store)
	s.Equal(1.0, series["router_db_pool_in_use_connections shared "])
	s.Equal(0.0, series["router_db_pool_in_use_connections shard here"])

	s.Equal(40.0, series["router_db_pool_max_open_connections shared "])
	s.Equal(40.0, series["router_db_pool_max_open_connections shard here"])
	s.Equal(8.0, series["router_db_pool_max_open_connections peer there"])
	// Five metrics for each of the three pools, and no fourth.
	s.Len(series, 15)
}
