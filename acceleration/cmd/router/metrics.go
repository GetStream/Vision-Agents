package main

import (
	"errors"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// rateLimitedKeyVars are the key variables of the vendors whose rate limits the vendor metrics
// were checked against in their docs: OpenAI (and its Realtime API), Anthropic, ElevenLabs,
// Cerebras, xAI, Together, Fireworks, Deepgram, Google Gemini, Cartesia and OpenRouter. Their
// 429 counts are served at 0 from start-up, one under key "none" for a variable the deployment
// leaves unset, while their pools are built only by a first request or session. Any other
// vendor's are served from when its pool is built.
var rateLimitedKeyVars = []string{
	"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY", "CEREBRAS_API_KEY", "XAI_API_KEY",
	"TOGETHER_API_KEY", "FIREWORKS_API_KEY", "DEEPGRAM_API_KEY", "GOOGLE_API_KEY", "CARTESIA_API_KEY",
	"OPENROUTER_API_KEY",
}

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
	keypool.ServeRateLimited(rateLimitedKeyVars...)
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(prometheus.DefaultGatherer, promhttp.HandlerOpts{EnableOpenMetrics: true}))
	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: readHeaderTimeout, IdleTimeout: idleTimeout}
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

// serveSaturation adds the Postgres pools' and the Redis connections' metrics to what /metrics
// serves, for whichever of the two the router has. stop takes them out again, so a router started
// after this one in the same process, as tests start them, serves its own.
func serveSaturation(pgStore *store.Store, liveClient *live.Client, logger *slog.Logger) (stop func()) {
	var collectors []prometheus.Collector
	if pgStore != nil {
		collectors = append(collectors, pgStore.Collector())
	}
	if liveClient != nil {
		collectors = append(collectors, liveClient.Collectors()...)
	}
	var registered []prometheus.Collector
	for _, collector := range collectors {
		// One registry holds one series of a name and labels, so a second router running in
		// the same process, as an overlapping test runs, serves the first one's alone.
		if err := prometheus.Register(collector); err != nil {
			logger.Warn("could not serve saturation metrics", "error", err)
			continue
		}
		registered = append(registered, collector)
	}
	return func() {
		for _, collector := range registered {
			prometheus.Unregister(collector)
		}
	}
}
