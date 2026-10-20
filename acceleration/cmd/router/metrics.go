package main

import (
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// serveMetrics serves Prometheus metrics at /metrics on an address of their own, until the
// returned function stops it.
func serveMetrics(addr string, logger *slog.Logger) (stop func()) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(prometheus.DefaultGatherer, promhttp.HandlerOpts{EnableOpenMetrics: true}))
	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: readHeaderTimeout}
	logger.Info("serving metrics", "address", addr, "setting", "metrics_addr")
	go func() {
		// A router taking over from one still draining its calls finds the address held until
		// that one ends, so it keeps trying rather than going without.
		for {
			err := server.ListenAndServe()
			if errors.Is(err, http.ErrServerClosed) {
				return
			}
			logger.Warn("metrics are not serving, trying again", "error", err)
			time.Sleep(5 * time.Second)
		}
	}()
	return func() {
		if err := server.Close(); err != nil {
			logger.Warn("could not close the metrics listener", "error", err)
		}
	}
}
