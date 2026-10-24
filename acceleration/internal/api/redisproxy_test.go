//go:build integration

package api

import (
	"io"
	"net"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// redisProxy is the suites' Redis behind a TCP proxy a test can take down: down, every
// connection through it is closed and new ones are refused, as a Redis that went away.
type redisProxy struct {
	down  atomic.Bool
	mu    sync.Mutex
	conns []net.Conn
}

// newRedisProxy is a client of the suites' Redis through a proxy of the test's own.
func newRedisProxy(t *testing.T) (*redisProxy, rueidis.Client) {
	target := os.Getenv("ROUTER_REDIS_ADDR")
	if target == "" {
		t.Skip("ROUTER_REDIS_ADDR must be set")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	p := &redisProxy{}
	t.Cleanup(func() {
		_ = listener.Close()
		p.stop()
	})
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			if p.down.Load() {
				_ = conn.Close()
				continue
			}
			upstream, err := net.Dial("tcp", target)
			if err != nil {
				_ = conn.Close()
				continue
			}
			p.mu.Lock()
			p.conns = append(p.conns, conn, upstream)
			p.mu.Unlock()
			go func() { _, _ = io.Copy(upstream, conn); _ = upstream.Close() }()
			go func() { _, _ = io.Copy(conn, upstream); _ = conn.Close() }()
		}
	}()
	options, err := rueidis.ParseURL("redis://" + listener.Addr().String() + "/" + strconv.Itoa(usagetest.RedisDB))
	require.NoError(t, err)
	options.DisableCache = true
	client, err := rueidis.NewClient(options)
	require.NoError(t, err)
	t.Cleanup(client.Close)
	return p, client
}

// stop takes Redis away: every connection through the proxy is closed, and every one after,
// until down is cleared.
func (p *redisProxy) stop() {
	p.down.Store(true)
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, conn := range p.conns {
		_ = conn.Close()
	}
	p.conns = nil
}
