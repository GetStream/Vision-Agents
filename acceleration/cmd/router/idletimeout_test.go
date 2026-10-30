package main

import (
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type IdleTimeoutSuite struct {
	suite.Suite
}

func TestIdleTimeoutSuite(t *testing.T) {
	suite.Run(t, new(IdleTimeoutSuite))
}

// TestEveryServerClosesIdleConnections: each http.Server the router builds sets IdleTimeout,
// so one that is added, or loses it, fails here. Left unset, an idle keep-alive connection
// is held until its client closes it. The servers are built inside run and the serve
// helpers, whose timeout a test cannot shorten, so the source is read.
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
