package streamapp

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// streamVariables are the variables naming a deployment-wide Stream app.
var streamVariables = map[string]bool{
	"STREAM_API_KEY": true, "STREAM_API_SECRET": true, "STREAM_USER_TOKEN": true, "STREAM_BASE_URL": true,
}

// readsStreamEnvironment are the packages allowed to read those variables: config, which
// owns every setting, and this package, which turns them into the deployment's identity.
var readsStreamEnvironment = []string{"internal/config/", "internal/streamapp/"}

// buildsClients are the files allowed to build a Stream client: this package's cache, the
// voice edge, which builds its own from the identity it is handed, the constructors the
// command line tools build theirs with from credentials they pass in, and the fakes.
var buildsClients = []string{
	"internal/streamapp/", "internal/agent/streamedge/", "internal/conversation/chattest/",
	"internal/chatlog/chatlog.go", "internal/phone/stream.go", "internal/chat/hooks.go",
}

// clientConstructors are what builds a Stream client, by import path and function.
var clientConstructors = map[string]map[string]bool{
	"github.com/GetStream/getstream-go/v5":     {"NewClient": true, "NewClientFromEnvVars": true},
	"github.com/GetStream/getstream-go-webrtc": {"NewRTCClient": true, "NewClient": true},
}

func TestOnlyConfigAndStreamappReadStreamCredentialsFromTheEnvironment(t *testing.T) {
	// The router acts in the Stream app each customer's work is pinned to. A package that
	// reads the environment, or builds a client of its own, acts in the deployment's app for
	// everyone, which is the bug this package exists to end.
	root, err := filepath.Abs(filepath.Join("..", ".."))
	require.NoError(t, err)

	var found []string
	for _, dir := range []string{"internal", filepath.Join("cmd", "router")} {
		require.NoError(t, filepath.WalkDir(filepath.Join(root, dir), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return err
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			found = append(found, offences(t, path, filepath.ToSlash(relative))...)
			return nil
		}))
	}

	require.Empty(t, found)
}

func offences(t *testing.T, path, relative string) []string {
	t.Helper()
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, path, nil, 0)
	require.NoError(t, err)

	imports := map[string]string{}
	for _, spec := range file.Imports {
		imported, _ := strconv.Unquote(spec.Path.Value)
		name := filepath.Base(imported)
		if strings.HasPrefix(name, "v") && len(name) <= 3 {
			name = filepath.Base(filepath.Dir(imported))
		}
		name = strings.TrimPrefix(name, "getstream-go-")
		if spec.Name != nil {
			name = spec.Name.Name
		}
		imports[name] = imported
	}
	constants := map[string]string{}
	for _, declared := range file.Decls {
		general, ok := declared.(*ast.GenDecl)
		if !ok || general.Tok != token.CONST {
			continue
		}
		for _, spec := range general.Specs {
			value := spec.(*ast.ValueSpec)
			for i, name := range value.Names {
				if i < len(value.Values) {
					if literal, ok := value.Values[i].(*ast.BasicLit); ok && literal.Kind == token.STRING {
						constants[name.Name], _ = strconv.Unquote(literal.Value)
					}
				}
			}
		}
	}

	var found []string
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		owner, ok := selector.X.(*ast.Ident)
		if !ok {
			return true
		}
		at := set.Position(call.Pos())
		where := relative + ":" + strconv.Itoa(at.Line)
		imported := imports[owner.Name]
		if imported == "os" && (selector.Sel.Name == "Getenv" || selector.Sel.Name == "LookupEnv") && len(call.Args) == 1 {
			if streamVariables[variableName(call.Args[0], constants)] && !allowed(relative, readsStreamEnvironment) {
				found = append(found, where+" reads a Stream credential from the environment")
			}
		}
		if clientConstructors[imported][selector.Sel.Name] && !allowed(relative, buildsClients) {
			found = append(found, where+" builds a Stream client of its own")
		}
		return true
	})
	return found
}

func variableName(argument ast.Expr, constants map[string]string) string {
	switch typed := argument.(type) {
	case *ast.BasicLit:
		name, _ := strconv.Unquote(typed.Value)
		return name
	case *ast.Ident:
		return constants[typed.Name]
	}
	return ""
}

func allowed(relative string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(relative, prefix) {
			return true
		}
	}
	return false
}
