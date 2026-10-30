package main

import (
	"bufio"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

type IdleTimeoutSuite struct {
	suite.Suite
}

func TestIdleTimeoutSuite(t *testing.T) {
	suite.Run(t, new(IdleTimeoutSuite))
}

// TestEveryServerClosesIdleConnections: each http.Server the router builds sets IdleTimeout,
// so one that is added, or loses it, fails here. Left unset, an idle keep-alive connection
// is held until its client closes it. It reads the source, so it also sees a server a
// listener test does not start (the sandbox proxy); TestEveryListenerClosesAnIdleConnection
// watches the listeners close a connection.
func (s *IdleTimeoutSuite) TestEveryServerClosesIdleConnections() {
	files, err := filepath.Glob("*.go")
	s.Require().NoError(err)
	fset := token.NewFileSet()
	servers := map[string]int{}
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, name, nil, 0)
		s.Require().NoError(err)
		ast.Inspect(file, func(node ast.Node) bool {
			literal, ok := node.(*ast.CompositeLit)
			if !ok || !isHTTPServer(literal.Type) {
				return true
			}
			servers[name]++
			s.Contains([]string{"idleTimeout", "proxy.IdleTimeout()"}, idleTimeoutOf(fset, literal),
				"%s sets no IdleTimeout of the router's", fset.Position(literal.Pos()))
			return true
		})
	}

	// The search finds the servers it is meant to guard: the API and peers and pprof, the
	// metrics, and the sandbox proxy.
	s.Equal(map[string]int{"main.go": 3, "metrics.go": 1, "sandboxproxy.go": 1}, servers)
}

func isHTTPServer(expr ast.Expr) bool {
	selector, ok := expr.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := selector.X.(*ast.Ident)
	return ok && pkg.Name == "http" && selector.Sel.Name == "Server"
}

// idleTimeoutOf is the source of the IdleTimeout a server literal sets, empty when it sets none.
func idleTimeoutOf(fset *token.FileSet, literal *ast.CompositeLit) string {
	for _, element := range literal.Elts {
		pair, ok := element.(*ast.KeyValueExpr)
		if !ok {
			continue
		}
		if key, ok := pair.Key.(*ast.Ident); ok && key.Name == "IdleTimeout" {
			var source strings.Builder
			_ = format.Node(&source, fset, pair.Value)
			return source.String()
		}
	}
	return ""
}

// TestIdleTimeoutOutlastsEveryPoolingClient: a client that pools connections to the router
// must give up an idle one first, or it reuses a connection the router has just closed.
func (s *IdleTimeoutSuite) TestIdleTimeoutOutlastsEveryPoolingClient() {
	// The stream-accelerate gateway pools with http.DefaultTransport. Prometheus' scrape
	// client keeps a connection for the 5 min of prometheus/common v0.66.1
	// config/http_config.go:55.
	gateway := http.DefaultTransport.(*http.Transport).IdleConnTimeout
	prometheus := 5 * time.Minute

	s.Equal(90*time.Second, gateway, "the default transport's idle timeout moved: recheck idleTimeout")
	s.Greater(idleTimeout, gateway)
	s.Greater(idleTimeout, prometheus)
}

// TestEveryListenerClosesAnIdleConnection starts the router with its API, peer, pprof and
// metrics listeners and idleTimeout shortened, and sends one request on a keep-alive
// connection to each. The server must then close the connection. A server that loses its
// IdleTimeout, or has it overwritten after its literal, keeps the connection open and the
// read times out.
func (s *IdleTimeoutSuite) TestEveryListenerClosesAnIdleConnection() {
	previous := idleTimeout
	idleTimeout = 300 * time.Millisecond
	s.T().Cleanup(func() { idleTimeout = previous })

	settings := config.Defaults()
	settings.Auth.Mode = string(auth.NoAuth)
	settings.Limits.Mode = config.LimitsOff
	listeners := []struct {
		name    string
		setting *string
		path    string
	}{
		{"api", &settings.Addr, "/health"},
		{"peer", &settings.Node.Listen, "/health"},
		{"pprof", &settings.PprofAddr, "/debug/pprof/cmdline"},
		{"metrics", &settings.MetricsAddr, "/metrics"},
	}
	for i, listener := range listeners {
		// 20000-32767 is below the ephemeral ranges of Linux (32768-60999) and macOS
		// (49152-65535), so a client's own connection does not take the port. The
		// process id spreads parallel packages over the range.
		*listener.setting = s.freeAddress(20000 + (os.Getpid()%2000)*6 + i)
	}

	stopped := make(chan error, 1)
	go func() { stopped <- run(settings, slog.New(slog.DiscardHandler)) }()
	s.T().Cleanup(func() {
		select {
		case <-stopped:
			return
		default:
		}
		s.NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
		select {
		case err := <-stopped:
			s.NoError(err)
		// Generous for a starved CI runner: a stalled VM took 30 s to stop it once.
		case <-time.After(60 * time.Second):
			s.Fail("the router did not stop on SIGTERM")
		}
	})

	for _, listener := range listeners {
		s.Run(listener.name, func() {
			conn := s.dialWhenListening(*listener.setting)
			defer conn.Close() //nolint:errcheck // the test is ending
			reader := bufio.NewReader(conn)

			_, err := fmt.Fprintf(conn, "GET %s HTTP/1.1\r\nHost: x\r\n\r\n", listener.path)
			s.Require().NoError(err)
			response, err := http.ReadResponse(reader, nil)
			s.Require().NoError(err)
			_, err = io.Copy(io.Discard, response.Body)
			s.Require().NoError(err)
			s.Require().NoError(response.Body.Close())

			// A server without IdleTimeout never closes the connection, so 20 s still fails it, and is
			// generous for a starved CI runner (a stalled VM overran 5 s).
			s.Require().NoError(conn.SetReadDeadline(time.Now().Add(20 * time.Second)))
			_, err = reader.ReadByte()
			s.True(errors.Is(err, io.EOF), "the server held an idle connection open: %v", err)
		})
	}
}

// freeAddress is the first loopback address from port up that nothing listens on.
func (s *IdleTimeoutSuite) freeAddress(port int) string {
	for ; port < 32768; port++ {
		addr := fmt.Sprintf("127.0.0.1:%d", port)
		opened, err := net.Listen("tcp", addr)
		if err != nil {
			continue
		}
		s.Require().NoError(opened.Close())
		return addr
	}
	s.FailNow("no free port in 20000-32767")
	return ""
}

func (s *IdleTimeoutSuite) dialWhenListening(addr string) net.Conn {
	deadline := time.Now().Add(30 * time.Second)
	for {
		conn, err := net.DialTimeout("tcp", addr, time.Second)
		if err == nil {
			return conn
		}
		s.Require().True(time.Now().Before(deadline), "nothing listened on %s: %v", addr, err)
		time.Sleep(50 * time.Millisecond)
	}
}
