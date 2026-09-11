package research

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T) (string, []Repository) {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	repos := []Repository{{ID: "react", URL: "https://github.com/GetStream/stream-chat-react", Product: "chat", SDK: "react", Revision: strings.Repeat("a", 40)}, {ID: "swiftui", URL: "https://github.com/GetStream/stream-chat-swiftui", Product: "chat", SDK: "ios-swiftui", Revision: strings.Repeat("b", 40)}}
	for _, r := range repos {
		if err = os.MkdirAll(filepath.Join(root, r.ID), 0755); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(root, r.ID, "client.ts"), []byte("// first line\nexport const maxMessages = 100;\n"), 0444); err != nil {
			t.Fatal(err)
		}
	}
	return root, repos
}
func TestVerifiedSource(t *testing.T) {
	root, repos := fixture(t)
	result := Result{Status: "answered", Answer: "The maximum is 100.", Citations: []Citation{{RepositoryID: "react", Path: "client.ts", Quote: "export const maxMessages = 100;", StartLine: 999}}}
	raw, _ := json.Marshal(result)
	verified, err := Verify(string(raw), root, repos[:1])
	if err != nil {
		t.Fatal(err)
	}
	if verified.Citations[0].StartLine != 2 || !strings.Contains(verified.Citations[0].URL, repos[0].Revision+"/client.ts#L2-L2") {
		t.Fatal(verified)
	}
	for _, c := range []Citation{{RepositoryID: "swiftui", Path: "client.ts", Quote: "export const maxMessages = 100;"}, {RepositoryID: "react", Path: "client.ts", Quote: "invented quote"}, {RepositoryID: "react", Path: "../swiftui/client.ts", Quote: "export const maxMessages = 100;"}} {
		result.Citations = []Citation{c}
		raw, _ = json.Marshal(result)
		if _, err = Verify(string(raw), root, repos[:1]); err == nil {
			t.Fatal("accepted invalid citation", c)
		}
	}
}
func TestSourceScopeAndIndex(t *testing.T) {
	root, repos := fixture(t)
	_ = os.Symlink(filepath.Join(root, "swiftui/client.ts"), filepath.Join(root, "react/link.ts"))
	for _, path := range []string{"link.ts", "../swiftui/client.ts", "/etc/passwd", ".env", "a/../../swiftui/client.ts"} {
		if _, err := ReadSource(root, "react", path); err == nil {
			t.Fatal("accepted", path)
		}
	}
	idx, err := BuildIndex(root, repos)
	if err != nil {
		t.Fatal(err)
	}
	ctx := idx.Context("What is the maxMessages const?", repos[:1])
	if !strings.Contains(ctx, "maxMessages") || strings.Contains(ctx, "swiftui/") {
		t.Fatal(ctx)
	}
}
func TestProfileScopes(t *testing.T) {
	_, repos := fixture(t)
	p := Profile{Name: "support", CustomerID: "acme", AgentID: "bot", Image: "build", Repositories: repos}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, q := range []Request{{Product: "video", SDK: "react", Question: "some question"}, {Product: "chat", SDK: "react", Question: "some question", RepositoryIDs: []string{"swiftui"}}, {Product: "chat", SDK: "react", Question: "some question", RepositoryIDs: []string{"../../etc"}}} {
		if _, err := p.Select(q); err == nil {
			t.Fatal(q)
		}
	}
	for _, u := range []string{"file:///etc", "https://github.com/GetStream/../other", "https://evil.test/GetStream/repo", "https://key@github.com/a/b"} {
		p.Repositories[0].URL = u
		if p.Validate() == nil {
			t.Fatal(u)
		}
	}
}

func TestSharedRepositoryScopesAndExplicitSelection(t *testing.T) {
	p := Profile{Name: "support", CustomerID: "owner", AgentID: "agent", Image: "build", Repositories: []Repository{
		{ID: "shared", URL: "https://github.com/GetStream/stream-node", Product: "video", SDK: "node", Scopes: []Scope{{Product: "feeds", SDK: "node"}}},
		{ID: "other", URL: "https://github.com/GetStream/other", Product: "feeds", SDK: "node"},
	}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	selected, err := p.Select(Request{Product: "feeds", SDK: "node", Question: "How do feeds work?", RepositoryIDs: []string{"shared"}})
	if err != nil || len(selected) != 1 || selected[0].ID != "shared" || selected[0].Product != "feeds" || len(selected[0].Scopes) != 0 {
		t.Fatalf("%+v, %v", selected, err)
	}
	if p.Repositories[0].Product != "video" {
		t.Fatal("selection mutated the profile")
	}
	for _, in := range []Request{
		{Product: "chat", SDK: "node", Question: "How does chat work?", RepositoryIDs: []string{"shared"}},
		{Product: "feeds", SDK: "react", Question: "How do feeds work?", RepositoryIDs: []string{"shared"}},
		{Product: "feeds", SDK: "node", Question: "How do feeds work?", RepositoryIDs: []string{"unknown"}},
	} {
		if _, err := p.Select(in); err == nil {
			t.Fatalf("accepted unsupported scope %+v", in)
		}
	}
	p.Repositories[0].Scopes = append(p.Repositories[0].Scopes, Scope{Product: "feeds", SDK: "node"})
	if p.Validate() == nil {
		t.Fatal("duplicate scope accepted")
	}
	p.Repositories[0].Scopes = nil
	p.Repositories[1].URL = p.Repositories[0].URL
	if p.Validate() == nil {
		t.Fatal("duplicate clone accepted")
	}
}

func TestAllSDKLanguagesCanBeReadIndexedAndCited(t *testing.T) {
	root, repos := fixture(t)
	for _, extension := range []string{"go", "py", "kt", "kts", "java", "dart", "cs", "c", "cpp", "h", "hpp", "m", "mm", "php", "rb", "proto", "swift", "ts", "tsx"} {
		t.Run(extension, func(t *testing.T) {
			path := "language-example." + extension
			quote := "SDKSourceExample represents a unique source symbol"
			if err := os.WriteFile(filepath.Join(root, "react", path), []byte(quote+"\n"), 0444); err != nil {
				t.Fatal(err)
			}
			idx, err := BuildIndex(root, repos[:1])
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, file := range idx.Files {
				if file.Path == path {
					found = true
				}
			}
			if !found {
				t.Fatal("language omitted from index")
			}
			raw, _ := json.Marshal(Result{Status: "answered", Answer: "A source symbol", Citations: []Citation{{RepositoryID: "react", Path: path, Quote: quote}}})
			if _, err := Verify(string(raw), root, repos[:1]); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, path := range []string{".env", "file.sh", "image.png", "../private/client.go"} {
		if _, err := ReadSource(root, "react", path); err == nil {
			t.Fatal("unsafe path accepted", path)
		}
	}
}

func TestIndexBudgetDoesNotStarveOtherRepositories(t *testing.T) {
	root, repos := fixture(t)
	// 100 scopes share a finite index budget. Only two distinct directories are
	// needed here to exercise per-repository allocation with oversized first files.
	many := make([]Repository, 100)
	for i := range many {
		many[i] = repos[0]
	}
	many[99] = repos[1]
	large := filepath.Join(root, "react", "a-large.go")
	if err := os.WriteFile(large, []byte(strings.Repeat("x", 1_100_000)), 0444); err != nil {
		t.Fatal(err)
	}
	idx, err := BuildIndex(root, many)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(idx.Context("What is maxMessages?", repos[1:]), "swiftui/") {
		t.Fatal("later SDK was starved")
	}
	if _, err := ReadSource(root, "react", "a-large.go"); err != nil {
		t.Fatal("unindexed source must remain readable", err)
	}
}
