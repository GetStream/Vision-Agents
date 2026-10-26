// Command usage serves the usage service: it records the usage events producers ship, once
// per usage id, keeps what every subject spent in every window, and mirrors those totals to
// the routers' Redis. It keeps organizations' accounts and the tiers' settings, which the
// operator manages through its management API. It runs from the router's image, with its
// own database.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/server"
)

// release is the version this binary was built from, set with -X main.release at link time
// by the workflow that publishes it.
var release string

const (
	// drainTimeout is how long requests in flight get to finish on SIGTERM.
	drainTimeout = 10 * time.Second
	// readHeaderTimeout bounds a client sending its headers.
	readHeaderTimeout = 10 * time.Second
	// readTimeout bounds a client sending a whole request, a 2 MiB batch included.
	readTimeout = 30 * time.Second
	// idleTimeout is how long a producer's kept-alive connection may sit unused.
	idleTimeout = 2 * time.Minute
	// sentryFlushTimeout bounds delivering buffered events on the way out.
	sentryFlushTimeout = 2 * time.Second
	// healthcheckTimeout bounds `usage healthcheck`.
	healthcheckTimeout = 3 * time.Second
	// pruneEvery is how often windows past their retention are deleted.
	pruneEvery = 24 * time.Hour
	// specPath is where `usage openapi` writes the service's spec, from acceleration/.
	specPath = "internal/usage/server/openapi.yaml"
)

// tierRetry is how often seeding the tiers is tried again after it failed at start.
var tierRetry = 30 * time.Second

// seedTimeout bounds one attempt at seeding the tiers.
const seedTimeout = 10 * time.Second

const usageText = `usage: usage [command]

  serve                    serve the API (the default)
  openapi                  write the API's OpenAPI document to ` + specPath + `, or to the path given
  healthcheck              exit 0 when the service on USAGE_ADDR is healthy, for a container's health check
  tiers check --file PATH  check a tier file, and exit non-zero on its first error
  tiers sync --dry-run     print the tiers the next start would add (USAGE_POSTGRES_DSN, USAGE_TIERS_FILE)

The service reads its settings from the environment:

  USAGE_ADDR           API listener (default :8070)
  USAGE_METRICS_ADDR   /metrics on a listener of its own (default: none)
  USAGE_POSTGRES_DSN   the service's own database (required)
  USAGE_INGEST_TOKEN   bearer token of /v1/events, /v1/windows and /v1/standings (required)
  USAGE_REDIS_URL      the routers' Redis, where window totals are mirrored (default: none)
  USAGE_MANAGE_KEY     X-Manage-Key of the management API, /v1/manage (default: none, the API answers 404)
  USAGE_TIERS_FILE     a tier file: its tiers the database lacks are added at start (default: none,
                       only the built-in default tier, which fails closed)
  SENTRY_DSN           where alerts go (default: logged only)
  SENTRY_ENVIRONMENT   the Sentry environment
`

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	slog.SetDefault(logger)
	command, args := "serve", os.Args[1:]
	if len(args) > 0 {
		command, args = args[0], args[1:]
	}
	switch command {
	case "help", "-h", "--help":
		fmt.Print(usageText)
		return
	case "openapi":
		if err := writeSpec(args); err != nil {
			logger.Error("usage stopped", "error", err)
			os.Exit(1)
		}
		return
	case "healthcheck":
		if err := healthcheck(os.Getenv("USAGE_ADDR")); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	case "tiers":
		if err := tiers(context.Background(), args, os.Environ, os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	case "serve":
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", command, usageText)
		os.Exit(2)
	}

	// The SDK reads SENTRY_DSN and SENTRY_ENVIRONMENT itself; none disables reporting.
	if err := sentry.Init(sentry.ClientOptions{Release: release, ServerName: "acceleration-usage", AttachStacktrace: true}); err != nil {
		logger.Error("sentry is disabled", "error", err)
	}
	settings, err := load(os.Environ)
	if err == nil {
		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
		err = serve(ctx, settings, logger)
		stop()
	}
	if err != nil {
		logger.Error("usage stopped", "error", err)
		sentry.CaptureException(err)
		sentry.Flush(sentryFlushTimeout)
		os.Exit(1)
	}
	sentry.Flush(sentryFlushTimeout)
}

// writeSpec renders the service's OpenAPI document from its operations.
func writeSpec(args []string) error {
	path := specPath
	if len(args) > 0 {
		path = args[0]
	}
	spec, err := server.Spec()
	if err != nil {
		return err
	}
	return os.WriteFile(path, spec, 0o644)
}

// healthcheck asks the service listening on address, :8070 when empty, for its health.
func healthcheck(address string) error {
	if address == "" {
		address = defaultAddr
	}
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("usage: healthcheck: %w", err)
	}
	if ip := net.ParseIP(host); host == "" || (ip != nil && ip.IsUnspecified()) {
		host = "127.0.0.1"
	}
	client := &http.Client{Timeout: healthcheckTimeout}
	response, err := client.Get("http://" + net.JoinHostPort(host, port) + "/healthz")
	if err != nil {
		return fmt.Errorf("usage: healthcheck: %w", err)
	}
	defer response.Body.Close() //nolint:errcheck // a read-only body
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("usage: healthcheck: the service answered %d", response.StatusCode)
	}
	return nil
}

// tiers runs `usage tiers check --file PATH` and `usage tiers sync --dry-run`.
func tiers(ctx context.Context, args []string, environ func() []string, out io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: tiers check --file PATH, or tiers sync --dry-run")
	}
	flags := flag.NewFlagSet("tiers "+args[0], flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	path := flags.String("file", "", "the tier file")
	dryRun := flags.Bool("dry-run", false, "write nothing")
	if err := flags.Parse(args[1:]); err != nil {
		return fmt.Errorf("usage: tiers %s: %w", args[0], err)
	}
	switch args[0] {
	case "check":
		if *path == "" {
			return errors.New("usage: tiers check needs --file PATH")
		}
		file, err := ledger.ReadTierFile(*path)
		if err != nil {
			return err
		}
		names := make([]string, 0, len(file.Tiers))
		for _, tier := range file.Tiers {
			names = append(names, tier.Name)
		}
		_, err = fmt.Fprintf(out, "%s: %d tiers: %s\n", *path, len(names), strings.Join(names, ", "))
		return err
	case "sync":
		if !*dryRun {
			return errors.New("usage: tiers sync takes --dry-run: the service adds the tiers itself as it starts")
		}
		loaded, err := read(environ)
		if err != nil {
			return err
		}
		if loaded.PostgresDSN == "" {
			return errors.New("usage: USAGE_POSTGRES_DSN is required: the database to compare with")
		}
		if *path == "" {
			*path = loaded.TiersFile
		}
		var file *ledger.TierFile
		if *path != "" {
			if file, err = ledger.ReadTierFile(*path); err != nil {
				return err
			}
		}
		opened, err := ledger.Open(ledger.Options{DSN: loaded.PostgresDSN})
		if err != nil {
			return err
		}
		defer opened.Close() //nolint:errcheck // the process is ending
		added, migrated, err := opened.PlanTierSync(ctx, file)
		if err != nil {
			return err
		}
		if !migrated {
			_, err = fmt.Fprintf(out, "the database has no tiers table yet: the next start migrates it, then adds: %s\n", strings.Join(added, ", "))
			return err
		}
		if len(added) == 0 {
			_, err = fmt.Fprintln(out, "the next start adds no tier")
			return err
		}
		_, err = fmt.Fprintf(out, "the next start adds: %s\n", strings.Join(added, ", "))
		return err
	}
	return fmt.Errorf("usage: unknown tiers command %q: check or sync", args[0])
}

// seedTiers adds the tier file's tiers the database lacks, and the built-in default tier
// when it has none. When the database fails it logs, and tries again every tierRetry in the
// background until it succeeds or ctx is done: the service serves meanwhile, a tier with no
// row answering the built-in default.
func seedTiers(ctx context.Context, opened *ledger.Ledger, file *ledger.TierFile, logger *slog.Logger, wg *sync.WaitGroup) {
	attempt := func() bool {
		// Bounded, so a database that hangs rather than fails does not hold the start.
		bounded, cancel := context.WithTimeout(ctx, seedTimeout)
		defer cancel()
		added, err := opened.SyncTiers(bounded, file, release)
		if err != nil {
			if ctx.Err() == nil {
				logger.Error("could not seed the tiers; trying again", "in", tierRetry, "error", err)
			}
			return false
		}
		if len(added) > 0 {
			logger.Info("seeded tiers", "tiers", added)
		}
		return true
	}
	if attempt() {
		return
	}
	wg.Go(func() {
		ticker := time.NewTicker(tierRetry)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if attempt() {
					return
				}
			}
		}
	})
}

// serve runs the service until ctx is done, then drains it.
func serve(ctx context.Context, settings settings, logger *slog.Logger) error {
	var tierFile *ledger.TierFile
	if settings.TiersFile != "" {
		var err error
		if tierFile, err = ledger.ReadTierFile(settings.TiersFile); err != nil {
			return err
		}
	}
	var mirror *ledger.Mirror
	if settings.RedisURL != "" {
		var err error
		if mirror, err = ledger.NewMirror(ledger.MirrorOptions{URL: settings.RedisURL, Logger: logger}); err != nil {
			return err
		}
		defer mirror.Close()
	}
	opened, err := ledger.Open(ledger.Options{DSN: settings.PostgresDSN, Mirror: mirror})
	if err != nil {
		return err
	}
	defer opened.Close() //nolint:errcheck // the process is ending
	if err := opened.Migrate(ctx); err != nil {
		return err
	}
	api, err := server.New(server.Options{Ledger: opened, IngestToken: settings.IngestToken, ManageKey: settings.ManageKey, Logger: logger})
	if err != nil {
		return err
	}

	background, stopBackground := context.WithCancel(context.WithoutCancel(ctx))
	var wg sync.WaitGroup
	defer func() {
		stopBackground()
		wg.Wait()
	}()
	seedTiers(background, opened, tierFile, logger, &wg)
	wg.Go(func() { mirror.Run(background) })
	host, _ := os.Hostname()
	wg.Go(func() {
		alerts.New(alerts.Options{Logger: logger, Source: "usage/" + host}, server.Conditions()...).Run(background)
	})
	wg.Go(func() { prune(background, opened, logger) })
	if settings.MetricsAddr != "" {
		metrics := &http.Server{Addr: settings.MetricsAddr, Handler: metricsHandler(), ReadHeaderTimeout: readHeaderTimeout}
		wg.Go(func() { listen(metrics, "metrics", logger) })
		defer metrics.Close() //nolint:errcheck // the process is ending
	}

	httpServer := &http.Server{
		Addr: settings.Addr, Handler: api.Handler(),
		ReadHeaderTimeout: readHeaderTimeout, ReadTimeout: readTimeout, IdleTimeout: idleTimeout,
	}
	failed := make(chan error, 1)
	go func() { failed <- httpServer.ListenAndServe() }()
	logger.Info("usage service serving", "address", settings.Addr, "metrics", settings.MetricsAddr, "mirror", mirror != nil,
		"management", settings.ManageKey != "", "tiers_file", settings.TiersFile)

	select {
	case err := <-failed:
		return fmt.Errorf("usage: serve: %w", err)
	case <-ctx.Done():
	}
	logger.Info("usage service draining", "for", drainTimeout)
	drain, cancel := context.WithTimeout(context.Background(), drainTimeout)
	defer cancel()
	if err := httpServer.Shutdown(drain); err != nil {
		return fmt.Errorf("usage: drain: %w", err)
	}
	return nil
}

// metricsHandler serves every registered metric.
func metricsHandler() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(prometheus.DefaultGatherer, promhttp.HandlerOpts{EnableOpenMetrics: true}))
	return mux
}

// listen serves server until it is closed.
func listen(server *http.Server, name string, logger *slog.Logger) {
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Error("a listener stopped", "listener", name, "address", server.Addr, "error", err)
	}
}

// prune deletes the windows past their retention at start and then daily, until ctx is done.
// Every replica does it; a second delete finds nothing.
func prune(ctx context.Context, opened *ledger.Ledger, logger *slog.Logger) {
	ticker := time.NewTicker(pruneEvery)
	defer ticker.Stop()
	for {
		deleted, err := opened.PruneWindows(ctx, time.Now().Add(-ledger.Retention))
		if err != nil && ctx.Err() == nil {
			logger.Error("could not prune usage windows", "error", err)
		} else if deleted > 0 {
			logger.Info("pruned usage windows", "deleted", deleted)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}
