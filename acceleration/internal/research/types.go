// Package research implements scoped, read-only source investigations over Cursor ACP.
package research

import (
	_ "embed"
	"errors"
	"net/url"
	"os"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

const DefaultModel = "gpt-5.6-luna-low"
const Root = "/opt/repositories"

var identifier = regexp.MustCompile(`^[a-z][a-z0-9-]{0,63}$`)
var githubPath = regexp.MustCompile(`^/[A-Za-z0-9_-]+/[A-Za-z0-9_-][A-Za-z0-9_.-]*$`)
var sha = regexp.MustCompile(`^[a-f0-9]{40}$`)

type Repository struct {
	ID       string  `json:"id" yaml:"id"`
	URL      string  `json:"url" yaml:"url"`
	Product  string  `json:"product" yaml:"product"`
	SDK      string  `json:"sdk" yaml:"sdk"`
	Scopes   []Scope `json:"scopes,omitempty" yaml:"scopes,omitempty"`
	Ref      string  `json:"ref,omitempty" yaml:"ref,omitempty"`
	Revision string  `json:"revision,omitempty" yaml:"-"`
}

// Scope adds a supported product/SDK to a repository shared by several SDKs.
// Product and SDK on Repository remain its primary, backward-compatible scope.
type Scope struct {
	Product string `json:"product" yaml:"product"`
	SDK     string `json:"sdk" yaml:"sdk"`
}

func (r Repository) Supports(product, sdk string) bool {
	if r.Product == product && r.SDK == sdk {
		return true
	}
	for _, scope := range r.Scopes {
		if scope.Product == product && scope.SDK == sdk {
			return true
		}
	}
	return false
}

type Profile struct {
	Name         string       `json:"name" yaml:"name"`
	CustomerID   string       `json:"customer_id" yaml:"customer_id"`
	AgentID      string       `json:"agent_id" yaml:"agent_id"`
	Image        string       `json:"image" yaml:"image"`
	Model        string       `json:"model" yaml:"model"`
	Repositories []Repository `json:"repositories" yaml:"repositories"`
}

func (p *Profile) Validate() error {
	if !identifier.MatchString(p.Name) || p.CustomerID == "" || p.AgentID == "" || p.Image == "" || len(p.Repositories) == 0 {
		return errors.New("research: profile needs name, owner, image, and repositories")
	}
	if p.Model == "" {
		p.Model = DefaultModel
	}
	seen := map[string]bool{}
	clones := map[string]bool{}
	for _, r := range p.Repositories {
		u, e := url.Parse(r.URL)
		if e != nil || u.Scheme != "https" || u.Host != "github.com" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || !githubPath.MatchString(u.Path) || !identifier.MatchString(r.ID) || seen[r.ID] || r.Product == "" || r.SDK == "" || strings.HasPrefix(r.Ref, "-") || strings.ContainsAny(r.Ref, "\x00\n\r") {
			return errors.New("research: invalid or duplicate repository")
		}
		seen[r.ID] = true
		clone := strings.TrimSuffix(r.URL, ".git") + "@" + r.Ref
		if clones[clone] {
			return errors.New("research: duplicate clone; use repository scopes")
		}
		clones[clone] = true
		scopes := map[Scope]bool{{Product: r.Product, SDK: r.SDK}: true}
		for _, scope := range r.Scopes {
			if !identifier.MatchString(scope.Product) || !identifier.MatchString(scope.SDK) || scopes[scope] {
				return errors.New("research: invalid or duplicate repository scope")
			}
			scopes[scope] = true
		}
	}
	return nil
}
func Load(path string) ([]Profile, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	var config struct {
		Profiles []Profile `yaml:"profiles"`
	}
	decoder := yaml.NewDecoder(strings.NewReader(string(b)))
	decoder.KnownFields(true)
	if e = decoder.Decode(&config); e != nil {
		return nil, e
	}
	seen := map[string]bool{}
	for i := range config.Profiles {
		p := &config.Profiles[i]
		if e = p.Validate(); e != nil {
			return nil, e
		}
		if seen[p.Name] {
			return nil, errors.New("research: duplicate profile")
		}
		seen[p.Name] = true
	}
	return config.Profiles, nil
}

type Request struct {
	Product       string   `json:"product"`
	SDK           string   `json:"sdk"`
	Question      string   `json:"question"`
	RepositoryIDs []string `json:"repository_ids,omitempty"`
}

func (p Profile) Select(in Request) ([]Repository, error) {
	if len(strings.TrimSpace(in.Question)) < 5 || len(in.Question) > 6000 {
		return nil, errors.New("invalid_question")
	}
	wanted := map[string]bool{}
	for _, id := range in.RepositoryIDs {
		wanted[id] = true
	}
	var selected []Repository
	for _, r := range p.Repositories {
		if r.Supports(in.Product, in.SDK) && (len(in.RepositoryIDs) == 0 || wanted[r.ID]) {
			// Cursor sees the requested scope, not a monorepo's unrelated primary scope.
			r.Product, r.SDK, r.Scopes = in.Product, in.SDK, nil
			selected = append(selected, r)
			delete(wanted, r.ID)
		}
	}
	if len(selected) == 0 || len(wanted) > 0 {
		return nil, errors.New("unsupported_scope")
	}
	return selected, nil
}

type Citation struct {
	RepositoryID string `json:"repository_id"`
	Path         string `json:"path"`
	StartLine    int    `json:"start_line"`
	EndLine      int    `json:"end_line"`
	Quote        string `json:"quote"`
	Revision     string `json:"revision,omitempty"`
	URL          string `json:"url,omitempty"`
}
type Result struct {
	Status      string     `json:"status"`
	Answer      string     `json:"answer,omitempty"`
	Citations   []Citation `json:"citations,omitempty"`
	Limitations []string   `json:"limitations,omitempty"`
	Code        string     `json:"code,omitempty"`
	ElapsedMS   int64      `json:"elapsed_ms"`
}
type Progress struct {
	Phase             string `json:"phase"`
	VerifiedCitations int    `json:"verified_citations,omitempty"`
	ElapsedMS         int64  `json:"elapsed_ms"`
}
type Frame struct {
	Progress *Progress `json:"progress,omitempty"`
	Result   *Result   `json:"result,omitempty"`
}

// PrepareScript removes executable repository policy and locks source before Cursor starts.
//
//go:embed prepare.sh
var PrepareScript string
