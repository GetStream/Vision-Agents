package plugins

import (
	"embed"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
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
	// Scopes are asked for at consent. Empty asks for none and takes the server's default.
	Scopes []string `yaml:"scopes"`
	// AuthorizeParams go on the authorize URL as well, for a provider that needs them.
	AuthorizeParams map[string]string `yaml:"authorize_params"`
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
			return "", fmt.Errorf("plugins: %s needs an instance url: %s", p.Name, p.InstanceHint)
		}
		return "", fmt.Errorf("plugins: %s needs an instance url", p.Name)
	}
	return strings.ReplaceAll(p.URL, "{instance}", host), nil
}

func or(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
