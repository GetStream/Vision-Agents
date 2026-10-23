package main

import (
	"errors"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// serveMetrics serves Prometheus metrics at /metrics on an address of their own, until stop
// or handOver stops it. An empty address serves nothing: the server would listen on ":http".
//
// handOver is stop for a router that starts draining: it frees the address for the router
// taking over, which runs with the same one, and says so once. Either may be called more
// than once, and after the other.
func serveMetrics(addr string, logger *slog.Logger) (stop, handOver func()) {
	if addr == "" {
		return func() {}, func() {}
	}
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(prometheus.DefaultGatherer, promhttp.HandlerOpts{EnableOpenMetrics: true}))
	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: readHeaderTimeout}
	logger.Info("serving metrics", "address", addr, "setting", "metrics_addr")
	go func() {
		// A router taking over finds the address held until the one before it starts
		// draining and hands it over, or ends, so it keeps trying rather than going without.
		for {
			err := server.ListenAndServe()
			if errors.Is(err, http.ErrServerClosed) {
				return
			}
			logger.Warn("metrics are not serving, trying again", "error", err)
			time.Sleep(5 * time.Second)
		}
	}()
	var once sync.Once
	closeServer := func(handingOver bool) {
		once.Do(func() {
			if handingOver {
				logger.Info("stopped serving metrics for the router taking over", "address", addr, "setting", "metrics_addr")
			}
			if err := server.Close(); err != nil {
				logger.Warn("could not close the metrics listener", "error", err)
			}
		})
	}
	return func() { closeServer(false) }, func() { closeServer(true) }
}
