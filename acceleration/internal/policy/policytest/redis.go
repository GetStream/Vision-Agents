package policytest

import (
	"io"
	"net"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// Redis is the suites' Redis behind a TCP proxy a test can take down, as a Redis that went away:
// down, every connection through it is closed and new ones are refused, until it is restored.
type Redis struct {
	Client rueidis.Client
	down   atomic.Bool
	mu     sync.Mutex
	conns  []net.Conn
}

// SharedHolds is a Redis through a proxy of the test's own that the enforcer shares holds
// through, flushing every 10 ms, reading the others' every 50 ms and refusing billable work once
// a hold has waited a minute, so a flush a loaded machine holds up refuses nothing; a test of that
// refusal sets its own FlushMaxLag. It skips the test without ROUTER_REDIS_ADDR.
func SharedHolds(t testing.TB) (*Redis, func(*policy.Settings)) {
	t.Helper()
	target := os.Getenv("ROUTER_REDIS_ADDR")
	if target == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	r := &Redis{}
	t.Cleanup(func() {
		_ = listener.Close()
		r.Stop()
	})
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			if r.down.Load() {
				_ = conn.Close()
				continue
			}
			upstream, err := net.Dial("tcp", target)
			if err != nil {
				_ = conn.Close()
				continue
			}
			r.mu.Lock()
			r.conns = append(r.conns, conn, upstream)
			r.mu.Unlock()
			go func() { _, _ = io.Copy(upstream, conn); _ = upstream.Close() }()
			go func() { _, _ = io.Copy(conn, upstream); _ = conn.Close() }()
		}
	}()
	options, err := rueidis.ParseURL("redis://" + listener.Addr().String() + "/" + strconv.Itoa(usagetest.RedisDB))
	require.NoError(t, err)
	options.DisableCache = true
	r.Client, err = rueidis.NewClient(options)
	require.NoError(t, err)
	t.Cleanup(r.Client.Close)
	return r, func(settings *policy.Settings) {
		settings.Holds = r.Client
		settings.FlushEvery, settings.HoldRefresh, settings.FlushMaxLag = 10*time.Millisecond, 50*time.Millisecond, time.Minute
	}
}

// Stop takes Redis away: every connection through the proxy is closed, and every one after.
func (r *Redis) Stop() {
	r.down.Store(true)
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, conn := range r.conns {
		_ = conn.Close()
	}
	r.conns = nil
}

// Restore lets connections through again.
func (r *Redis) Restore() { r.down.Store(false) }
