package plugins

import (
	"embed"
	"fmt"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// The logos are our own plain marks, not the vendors' artwork, so a deployment that has
// licensed the real thing replaces a file and changes nothing else.
//
//go:embed plugins.yaml logos
var catalogFS embed.FS

// Plugin is one hosted MCP server the dashboard may attach to an agent.
type Plugin struct {
	ID          string `yaml:"id"`
	Name        string `yaml:"name"`
	Category    string `yaml:"category"`
	Description string `yaml:"description"`
	URL         string `yaml:"url"`
	Auth        string `yaml:"auth"`
	// Logo is the file under logos/ this plugin is drawn with.
	Logo string `yaml:"logo"`
	// InstanceRequired means the URL is a template that needs a shop or org hostname.
	InstanceRequired bool   `yaml:"instance_required"`
	InstanceHint     string `yaml:"instance_hint"`
	// ClientRequired means the provider registers no OAuth client on the fly, so logging
	// in needs one registered in advance: the agent's own, or this deployment's.
	ClientRequired bool `yaml:"client_required"`
	// SetupURL is where the app creates that client, and SetupSteps what to do there, in
	// order, for a dashboard to show beside the form the client is pasted into.
	SetupURL   string      `yaml:"setup_url"`
	SetupSteps []SetupStep `yaml:"setup_steps"`
	// Scopes are asked for at consent. Empty asks for none and takes the server's default.
	Scopes []string `yaml:"scopes"`
	// ScopesSupported are what the server's protected-resource metadata says it accepts,
	// which an agent's own scopes must come from. Empty checks nothing.
	ScopesSupported []string `yaml:"scopes_supported"`
	// AuthorizeParams go on the authorize URL as well, for a provider that needs them.
	AuthorizeParams map[string]string `yaml:"authorize_params"`
	// AccessTTL is how long an access token lives when the token response gives no
	// expires_in, so it is renewed before the provider ends it. Zero leaves such a token
	// with no known expiry.
	AccessTTL time.Duration `yaml:"access_ttl"`
	// ReadonlyURL is the server's read-only endpoint, for a vendor that runs one, and
	// ReadonlyScopes what is asked for at consent there instead of Scopes.
	ReadonlyURL    string   `yaml:"readonly_url"`
	ReadonlyScopes []string `yaml:"readonly_scopes"`
	// Toolsets are the groups of tools the server can be limited to, by a toolsets query
	// parameter on its URL. Empty means it cannot be.
	Toolsets []string `yaml:"toolsets"`
	// Tools are the agent's own allowlist of the server's tools, as names or path.Match
	// patterns, set by Configured. Empty offers every tool.
	Tools []string `yaml:"-"`
	// ByURL is an MCP server an agent config names by its URL rather than from the catalog.
	// Its login registers a client of its own, never the deployment's, and asks for the
	// scopes its server advertises when given none.
	ByURL bool `yaml:"-"`
}

// SetupStep is one thing to do with the provider before its client can be pasted in.
type SetupStep struct {
	Title       string `yaml:"title"`
	Description string `yaml:"description"`
}

// Options are what an agent config changes about a catalog plugin.
type Options struct {
	// Readonly reaches the read-only endpoint.
	Readonly bool
	// Scopes are asked for at consent in place of the catalog's.
	Scopes []string
	// Toolsets limit the server to these groups of tools. Empty offers every tool.
	Toolsets []string
	// Tools offer only the server's tools matching these names or path.Match patterns.
	// Empty offers every tool.
	Tools []string
}

type catalogFile struct {
	Plugins []Plugin `yaml:"plugins"`
}

var catalog []Plugin

func init() {
	loaded, err := loadCatalog()
	if err != nil {
		panic(err)
	}
	catalog = loaded
}

func loadCatalog() ([]Plugin, error) {
	raw, err := catalogFS.ReadFile("plugins.yaml")
	if err != nil {
		return nil, fmt.Errorf("plugins: read catalog: %w", err)
	}
	var file catalogFile
	if err := yaml.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("plugins: parse catalog: %w", err)
	}
	if len(file.Plugins) == 0 {
		return nil, fmt.Errorf("plugins: catalog is empty")
	}
	seen := map[string]struct{}{}
	for _, plugin := range file.Plugins {
		if plugin.ID == "" || plugin.Name == "" || plugin.URL == "" {
			return nil, fmt.Errorf("plugins: every plugin needs an id, a name and a url")
		}
		if _, duplicate := seen[plugin.ID]; duplicate {
			return nil, fmt.Errorf("plugins: %s is declared twice", plugin.ID)
		}
		// Read now rather than when a card is drawn, so a misnamed file is a router that
		// will not start rather than a login nobody can see the plugin on.
		if _, err := catalogFS.ReadFile(logoFile(plugin.Logo)); err != nil {
			return nil, fmt.Errorf("plugins: %s has no logo: %w", plugin.ID, err)
		}
		seen[plugin.ID] = struct{}{}
	}
	return file.Plugins, nil
}

func logoFile(name string) string {
	return "logos/" + name
}

// LogoPath is where a plugin's logo is served, which is what an authorization attachment
// points its thumbnail at.
func LogoPath(id string) string {
	return "/v1/agents/plugins/" + id + "/logo"
}

// Logo is the SVG a catalog plugin is drawn with.
func Logo(id string) ([]byte, bool) {
	plugin, ok := Lookup(id)
	if !ok {
		return nil, false
	}
	raw, err := catalogFS.ReadFile(logoFile(plugin.Logo))
	if err != nil {
		return nil, false
	}
	return raw, true
}

// Catalog is the built-in set, in the order they are declared.
func Catalog() []Plugin {
	return append([]Plugin(nil), catalog...)
}

// Lookup finds a plugin by id.
func Lookup(id string) (Plugin, bool) {
	for _, plugin := range catalog {
		if plugin.ID == id {
			return plugin, true
		}
	}
	return Plugin{}, false
}

// Search filters the catalog by name, category or description. Empty query is the lot.
func Search(query string) []Plugin {
	wanted := strings.ToLower(strings.TrimSpace(query))
	if wanted == "" {
		return Catalog()
	}
	found := make([]Plugin, 0, len(catalog))
	for _, plugin := range catalog {
		haystack := strings.ToLower(plugin.ID + " " + plugin.Name + " " + plugin.Category + " " + plugin.Description)
		if strings.Contains(haystack, wanted) {
			found = append(found, plugin)
		}
	}
	return found
}

// Configured is the plugin as an agent config asks for it: at its read-only endpoint when
// readonly is set, asking for scopes at consent when any are given, and limited to the
// toolsets and tools named.
func (p Plugin) Configured(options Options) (Plugin, error) {
	supported := p.ScopesSupported
	if options.Readonly {
		if p.ReadonlyURL == "" {
			return Plugin{}, stack.Wrap(fmt.Errorf("plugins: %s has no read-only endpoint", p.Name))
		}
		p.URL = p.ReadonlyURL
		p.Scopes = p.ReadonlyScopes
		supported = p.ReadonlyScopes
	}
	if len(options.Scopes) > 0 {
		for _, scope := range options.Scopes {
			if len(supported) > 0 && !slices.Contains(supported, scope) {
				return Plugin{}, stack.Wrap(fmt.Errorf("plugins: %s does not accept the scope %q", p.Name, scope))
			}
		}
		p.Scopes = options.Scopes
	}
	if err := CheckToolPatterns(options.Tools); err != nil {
		return Plugin{}, err
	}
	p.Tools = options.Tools
	if len(options.Toolsets) > 0 {
		for _, toolset := range options.Toolsets {
			if !slices.Contains(p.Toolsets, toolset) {
				return Plugin{}, stack.Wrap(fmt.Errorf("plugins: %s has no toolset %q", p.Name, toolset))
			}
		}
		p.URL += "?toolsets=" + strings.Join(options.Toolsets, ",")
	}
	return p, nil
}

// Endpoint is the MCP URL this plugin is reached at. An instance is the shop or org
// hostname for the two that have no single global URL.
func (p Plugin) Endpoint(instance string) (string, error) {
	if !p.InstanceRequired {
		return p.URL, nil
	}
	host := strings.TrimSpace(instance)
	host = strings.TrimPrefix(host, "https://")
	host = strings.TrimPrefix(host, "http://")
	host = strings.TrimSuffix(host, "/")
	if host == "" {
		if p.InstanceHint != "" {
			return "", stack.Wrap(fmt.Errorf("plugins: %s needs an instance url: %s", p.Name, p.InstanceHint))
		}
		return "", stack.Wrap(fmt.Errorf("plugins: %s needs an instance url", p.Name))
	}
	return strings.ReplaceAll(p.URL, "{instance}", host), nil
}

func or(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
