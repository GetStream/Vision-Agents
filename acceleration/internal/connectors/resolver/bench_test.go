//go:build integration

package resolver_test

import (
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// The load of the architecture doc's spike 4: 50 concurrent sessions, 20 calls each, no
// refresh due. Each session has a connection of its own.
const (
	sessions        = 50
	callsPerSession = 20
)

// BenchmarkResolveFastPath is spike 4's measurement: every resolve finds its credential
// cached, so each is the row read and the cache. It prints p50 and p95 and asserts no target.
func BenchmarkResolveFastPath(b *testing.B) {
	f, refs := spike(b)
	r := f.router(f.srv.Client())
	for _, ref := range refs {
		_, err := r.Resolve(f.ctx, ref, core.CredentialRequest{})
		require.NoError(b, err)
	}
	measure(b, refs, func() *resolver.Resolver { return r })
}

// BenchmarkResolveLockedPath is the same load with nothing cached: every resolve is a new
// resolver, so it takes the lock, opens the stored credentials and runs Retrieve. It is what
// the cache saves.
func BenchmarkResolveLockedPath(b *testing.B) {
	f, refs := spike(b)
	credentials, err := pgsealed.New(f.db, f.sealer)
	require.NoError(b, err)
	schemes := map[string]core.Scheme{oauth2code.Name: f.scheme(f.srv.Client())}
	measure(b, refs, func() *resolver.Resolver {
		r, err := resolver.New(resolver.Config{Store: f.db, Credentials: credentials, Schemes: schemes, Now: f.clock.Now})
		if err != nil {
			panic(err)
		}
		return r
	})
}

// spike is a fixture with one connected connection per session. The clock stands still, so
// no cached credential ages out during the run.
func spike(b *testing.B) (*fixture, []core.ConnectionRef) {
	dsn, db := database(b)
	b.Cleanup(func() { _ = db.Close() })
	f := newFixture(b, dsn, db)
	refs := make([]core.ConnectionRef, sessions)
	for i := range refs {
		refs[i] = f.connected()
	}
	return f, refs
}

// measure runs the load b.N times and reports the latency of one resolve at p50 and p95.
// resolverFor is asked before each resolve, outside the time measured.
func measure(b *testing.B, refs []core.ConnectionRef, resolverFor func() *resolver.Resolver) {
	var mu sync.Mutex
	samples := make([]time.Duration, 0, b.N*sessions*callsPerSession)
	b.ResetTimer()
	for range b.N {
		var wg sync.WaitGroup
		for _, ref := range refs {
			wg.Go(func() {
				own := make([]time.Duration, 0, callsPerSession)
				for range callsPerSession {
					r := resolverFor()
					start := time.Now()
					_, err := r.Resolve(b.Context(), ref, core.CredentialRequest{})
					own = append(own, time.Since(start))
					if err != nil {
						b.Error(err)
						return
					}
				}
				mu.Lock()
				samples = append(samples, own...)
				mu.Unlock()
			})
		}
		wg.Wait()
	}
	b.StopTimer()
	slices.Sort(samples)
	p50, p95 := percentile(samples, 50), percentile(samples, 95)
	b.ReportMetric(float64(p50.Microseconds()), "p50-us")
	b.ReportMetric(float64(p95.Microseconds()), "p95-us")
	b.Logf("%d resolves, %d sessions at once: p50 %v, p95 %v", len(samples), len(refs), p50, p95)
}

// percentile is the nearest-rank percentile p of sorted samples.
func percentile(sorted []time.Duration, p int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	rank := (p*len(sorted) + 99) / 100
	return sorted[max(rank, 1)-1]
}
