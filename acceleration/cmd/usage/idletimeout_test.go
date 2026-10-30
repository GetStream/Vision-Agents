package main

import (
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
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

// TestEveryServerClosesIdleConnections: each http.Server the service builds sets IdleTimeout,
// so one that is added, or loses it, fails here. Left unset, an idle keep-alive connection is
// held until its client closes it. The servers are built inside run, so the source is read.
func (s *IdleTimeoutSuite) TestEveryServerClosesIdleConnections() {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "main.go", nil, 0)
	s.Require().NoError(err)
	idle := map[string]string{}
	ast.Inspect(file, func(node ast.Node) bool {
		literal, ok := node.(*ast.CompositeLit)
		if !ok || !isHTTPServer(literal.Type) {
			return true
		}
		var handler, timeout string
		for _, element := range literal.Elts {
			pair, ok := element.(*ast.KeyValueExpr)
			if !ok {
				continue
			}
			var source strings.Builder
			_ = format.Node(&source, fset, pair.Value)
			switch pair.Key.(*ast.Ident).Name {
			case "Handler":
				handler = source.String()
			case "IdleTimeout":
				timeout = source.String()
			}
		}
		idle[handler] = timeout
		return true
	})

	s.Equal(map[string]string{"metricsHandler()": "metricsIdleTimeout", "api.Handler()": "idleTimeout"}, idle)
}

func isHTTPServer(expr ast.Expr) bool {
	selector, ok := expr.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := selector.X.(*ast.Ident)
	return ok && pkg.Name == "http" && selector.Sel.Name == "Server"
}

// TestMetricsIdleTimeoutOutlastsPrometheus: Prometheus' scrape client keeps a connection 5 min
// (prometheus/common v0.66.1 config/http_config.go:55); a shorter timeout has the server close
// a connection Prometheus is about to reuse.
func (s *IdleTimeoutSuite) TestMetricsIdleTimeoutOutlastsPrometheus() {
	s.Greater(metricsIdleTimeout, 5*time.Minute)
}
