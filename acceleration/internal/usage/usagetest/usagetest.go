// Package usagetest runs the real usage service for the integration suites of the service and
// of its producers: a database of the suite's own, a Redis database apart from the one routers
// use, and the service behind an httptest server.
package usagetest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/server"
)

// RedisDB is the Redis database the suites mirror to, so their keys never meet a local
// router's, which reads database 0: USAGE_TEST_REDIS_DB, or 9. Runs side by side each set
// their own.
var RedisDB = func() int {
	if db, err := strconv.Atoi(os.Getenv("USAGE_TEST_REDIS_DB")); err == nil {
		return db
	}
	return 9
}()

// Database is an empty, migrated database of the caller's own, named after name, so a suite
// may truncate it while another package's suite runs. It skips the test when there is no
// Postgres to run against.
func Database(t testing.TB, name string) string {
	t.Helper()
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ROUTER_POSTGRES_DSN not set")
	}
	own := testenv.Database(dsn, name)
	opened, err := ledger.Open(ledger.Options{DSN: own})
	require.NoError(t, err)
	defer opened.Close() //nolint:errcheck // closing a pool the test is done with
	ctx := context.Background()
	var database string
	require.NoError(t, opened.DB().QueryRowContext(ctx, "SELECT current_database()").Scan(&database))
	require.True(t, strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = opened.DB().ExecContext(ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	require.NoError(t, err)
	require.NoError(t, opened.Migrate(ctx))
	return own
}

// RedisURL is the suites' Redis database. It skips the test when there is no Redis.
func RedisURL(t testing.TB) string {
	t.Helper()
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	return "redis://" + address + "/" + strconv.Itoa(RedisDB)
}

// Redis is a client of the suites' Redis database, closed when the test ends.
func Redis(t testing.TB) rueidis.Client {
	t.Helper()
	options, err := rueidis.ParseURL(RedisURL(t))
	require.NoError(t, err)
	options.DisableCache = true
	client, err := rueidis.NewClient(options)
	require.NoError(t, err)
	t.Cleanup(client.Close)
	return client
}

// Organization is an organization id no other test or run uses, so its Redis keys are the
// test's alone.
func Organization() string {
	random := make([]byte, 6)
	_, _ = rand.Read(random)
	return "org-" + hex.EncodeToString(random)
}

// Forget deletes the mirror's keys of the organizations, and nothing else.
func Forget(t testing.TB, client rueidis.Client, organizations ...string) {
	t.Helper()
	ctx := context.Background()
	for _, organization := range organizations {
		var cursor uint64
		for {
			entry, err := client.Do(ctx, client.B().Scan().Cursor(cursor).Match(ledger.KeyPrefix+":{"+organization+"}:*").Count(100).Build()).AsScanEntry()
			require.NoError(t, err)
			if len(entry.Elements) > 0 {
				require.NoError(t, client.Do(ctx, client.B().Del().Key(entry.Elements...).Build()).Error())
			}
			cursor = entry.Cursor
			if cursor == 0 {
				break
			}
		}
	}
}

// Event is a valid billable event of organization, with change applied.
func Event(organization string, change ...func(*usage.Event)) usage.Event {
	event := usage.Event{
		UsageID:          uuid.Must(uuid.NewV7()),
		OccurredAt:       time.Now().UTC().Truncate(time.Microsecond),
		OrganizationID:   organization,
		OrganizationTier: usage.DefaultTier,
		AppID:            "app",
		APIKey:           "key",
		Meter:            "llm",
		Provider:         "openai",
		Model:            "gpt-test",
		Quantities:       map[string]int64{"input_tokens": 100, "output_tokens": 20},
		CostMicros:       1000,
		BaseCostMicros:   800,
		Billable:         true,
		Success:          true,
		Source:           "router/test/pod",
	}
	for _, apply := range change {
		apply(&event)
	}
	return event
}

// Service is the real usage service, serving on URL.
type Service struct {
	URL    string
	Token  string
	Ledger *ledger.Ledger
	// Mirror is nil unless the service was started with one.
	Mirror *ledger.Mirror
	// Client calls the service.
	Client *usage.Client
}

// Options configure Start.
type Options struct {
	// DSN is the service's database; empty is Database(t, "usage_service").
	DSN string
	// RedisURL turns the mirror on.
	RedisURL string
	// Mirror's retries, for a test that waits on one; zero takes the defaults.
	RetryEvery time.Duration
	Logger     *slog.Logger
	Now        func() time.Time
}

// Start runs the service for the test, and stops it when the test ends.
func Start(t testing.TB, options Options) *Service {
	t.Helper()
	if options.DSN == "" {
		options.DSN = Database(t, "usage_service")
	}
	service := &Service{Token: "ingest-" + uuid.NewString()}
	if options.RedisURL != "" {
		mirror, err := ledger.NewMirror(ledger.MirrorOptions{URL: options.RedisURL, Logger: options.Logger, RetryEvery: options.RetryEvery})
		require.NoError(t, err)
		service.Mirror = mirror
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		go func() {
			defer close(done)
			mirror.Run(ctx)
		}()
		t.Cleanup(func() {
			cancel()
			<-done
			mirror.Close()
		})
	}
	opened, err := ledger.Open(ledger.Options{DSN: options.DSN, Mirror: service.Mirror, Now: options.Now})
	require.NoError(t, err)
	service.Ledger = opened
	t.Cleanup(func() { _ = opened.Close() })

	api, err := server.New(server.Options{Ledger: opened, IngestToken: service.Token, Logger: options.Logger, Now: options.Now})
	require.NoError(t, err)
	httpServer := httptest.NewServer(api.Handler())
	t.Cleanup(httpServer.Close)
	service.URL = httpServer.URL
	service.Client, err = usage.NewClient(usage.ClientOptions{URL: service.URL, Token: service.Token})
	require.NoError(t, err)
	return service
}
