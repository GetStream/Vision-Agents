package standardwebhooks_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	"github.com/stretchr/testify/suite"
)

// GuardSuite reads verifier.go's source for what no request can show: whether the digests
// are compared in constant time. A timing difference is not something a test observes
// reliably, so the check is on the code, as core's guard tests are (core/guard_test.go) and
// hmacheader's is.
type GuardSuite struct {
	suite.Suite
	file *ast.File
}

func TestGuardSuite(t *testing.T) {
	suite.Run(t, new(GuardSuite))
}

func (s *GuardSuite) SetupSuite() {
	file, err := parser.ParseFile(token.NewFileSet(), "verifier.go", nil, parser.SkipObjectResolution)
	s.Require().NoError(err)
	s.file = file
}

// Standard Webhooks: «use a constant time comparison function to compare the calculated with
// the expected signature»
// (https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md).
// Any ==, != or bytes.Equal on a conversion or a byte comparison would be the direct
// comparison.
func (s *GuardSuite) TestTheDigestsAreComparedWithHMACEqualOnly() {
	compared, direct := 0, 0
	ast.Inspect(s.file, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.CallExpr:
			if selector, ok := n.Fun.(*ast.SelectorExpr); ok {
				if pkg, ok := selector.X.(*ast.Ident); ok {
					switch {
					case pkg.Name == "hmac" && selector.Sel.Name == "Equal":
						compared++
					case pkg.Name == "bytes" && (selector.Sel.Name == "Equal" || selector.Sel.Name == "Compare"):
						direct++
					}
				}
			}
		case *ast.BinaryExpr:
			if (n.Op == token.EQL || n.Op == token.NEQ) && (isConversion(n.X) || isConversion(n.Y)) {
				direct++
			}
		}
		return true
	})
	s.Equal(1, compared, "one hmac.Equal compares the digests")
	s.Zero(direct, "no digest is compared any other way")
}

// isConversion is a call such as string(b) or hex.EncodeToString(b), whatever a direct
// comparison of two digests would compare. len(b) is a length, not a digest.
func isConversion(expr ast.Expr) bool {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		return fun.Name == "string"
	case *ast.SelectorExpr:
		return true
	default:
		return false
	}
}
