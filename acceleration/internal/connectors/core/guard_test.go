package core

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
	"gopkg.in/yaml.v3"
)

// connectorsPath is the import path every connectors package sits under. core may import
// its own subpackages and nothing else here.
const connectorsPath = "github.com/GetStream/Vision-Agents/acceleration/internal/connectors/"

// deniedNames are provider names the core must never spell, even before a manifest for one
// is seeded. Where each comes from (paths on the connectors/planning branch, or on the
// prototype branch codex/connector-support at cf62af0d):
//
//   - salesforce, slack, calendly, shopify, google, microsoft, github, gong, linear,
//     twilio: the CI rule's deny list in acceleration/docs/connectors/architecture.md,
//     «CI rule that keeps provider code out of the core», item 2.
//   - calcom: a connector id in the prototype's acceleration/internal/mcp/connectors.yaml
//     (the other six ids there, slack, calendly, linear, github, gong and salesforce, are
//     already above).
//   - quickbooks, zoho, aws, linq: providers in the architecture doc's «Stress test: 12
//     awkward providers» (rows 2, 3, 10 and 12) not already listed.
//   - hubspot: the architecture doc's «Axes where providers differ», rows 1 and 15.
var deniedNames = []string{
	"salesforce", "slack", "calendly", "shopify", "google", "microsoft", "github", "gong",
	"linear", "twilio",
	"calcom",
	"quickbooks", "zoho", "aws", "linq",
	"hubspot",
}

// GuardSuite keeps the core free of adapters and provider names. It reads the package
// source rather than the compiled package, so it fails on a name before anything uses it.
type GuardSuite struct {
	suite.Suite
	fset  *token.FileSet
	files map[string]*ast.File
}

func TestGuardSuite(t *testing.T) {
	suite.Run(t, new(GuardSuite))
}

// SetupSuite parses every non-test Go file under core. Tests are left out because this
// file has to spell the names it refuses.
func (s *GuardSuite) SetupSuite() {
	s.fset = token.NewFileSet()
	s.files = map[string]*ast.File{}
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && d.Name() == "testdata" {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		file, err := parser.ParseFile(s.fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		s.files[path] = file
		return nil
	})
	s.Require().NoError(err)
	s.Require().NotEmpty(s.files)
}

// TestCoreImportsNoAdapter proves the arrow of dependency points inward: core imports no
// scheme, tool source, credential store, signal verifier or provider, nor any other connectors package.
// A direct import is enough to check, because an adapter imports core and so a transitive
// one would already be an import cycle the compiler refuses.
func (s *GuardSuite) TestCoreImportsNoAdapter() {
	for path, file := range s.files {
		for _, spec := range file.Imports {
			imported, err := strconv.Unquote(spec.Path.Value)
			s.Require().NoError(err)
			rest, under := strings.CutPrefix(imported, connectorsPath)
			if !under || rest == "core" || strings.HasPrefix(rest, "core/") {
				continue
			}
			s.Failf("core imports a connectors package outside core",
				"%s imports %s; adapters under schemes/, sources/, credentialstores/, verifiers/ and "+
					"providers/ import core, never the other way", path, imported)
		}
	}
}

// TestCoreNamesNoProvider proves no string literal in core is a provider name: neither a
// connector id from the seeded catalog nor one in the deny list. A literal like that is how
// a provider switch starts, and the switch is what makes a new provider a core change.
func (s *GuardSuite) TestCoreNamesNoProvider() {
	denied := s.denied()
	for _, file := range s.files {
		ast.Inspect(file, func(node ast.Node) bool {
			literal, ok := node.(*ast.BasicLit)
			if !ok || literal.Kind != token.STRING {
				return true
			}
			value, err := strconv.Unquote(literal.Value)
			s.Require().NoError(err)
			if denied[strings.ToLower(value)] {
				s.Failf("core names a provider",
					"%s: string literal %q is a provider name; put it in a manifest or an "+
						"adapter", s.fset.Position(literal.Pos()), value)
			}
			return true
		})
	}
}

// denied is the deny list plus every connector id seeded from providers/. Until the seeded
// catalog exists the glob finds nothing and the deny list stands alone.
func (s *GuardSuite) denied() map[string]bool {
	denied := map[string]bool{}
	for _, name := range deniedNames {
		denied[name] = true
	}
	manifests, err := filepath.Glob(filepath.Join("..", "providers", "*.yaml"))
	s.Require().NoError(err)
	for _, path := range manifests {
		raw, err := os.ReadFile(path)
		s.Require().NoError(err)
		var manifest struct {
			ID string `yaml:"id"`
		}
		s.Require().NoError(yaml.Unmarshal(raw, &manifest), path)
		if manifest.ID != "" {
			denied[strings.ToLower(manifest.ID)] = true
		}
	}
	return denied
}
