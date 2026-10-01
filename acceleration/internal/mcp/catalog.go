package mcp

import (
	"embed"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

//go:embed connectors.yaml
var catalogFS embed.FS

// Connector is one hosted MCP server the dashboard may attach to an agent.
type Connector struct {
	ID                      string   `yaml:"id"`
	Name                    string   `yaml:"name"`
	Category                string   `yaml:"category"`
	Description             string   `yaml:"description"`
	URL                     string   `yaml:"url"`
	AuthMode                string   `yaml:"auth"`
	AuthHeader              string   `yaml:"auth_header"`
	OAuthMode               string   `yaml:"oauth_mode"`
	Issuer                  string   `yaml:"issuer"`
	Resource                string   `yaml:"resource"`
	AuthorizationEndpoint   string   `yaml:"authorization_endpoint"`
	TokenEndpoint           string   `yaml:"token_endpoint"`
	RefreshEndpoint         string   `yaml:"refresh_endpoint"`
	RegistrationEndpoint    string   `yaml:"registration_endpoint"`
	TokenEndpointAuthMethod string   `yaml:"token_endpoint_auth_method"`
	ClientEnv               string   `yaml:"client_env"`
	Scopes                  []string `yaml:"scopes"`
	// InstanceRequired means the URL is a template that needs a shop or org hostname.
	InstanceRequired bool   `yaml:"instance_required"`
	InstanceHint     string `yaml:"instance_hint"`
}

type catalogFile struct {
	Connectors []Connector `yaml:"connectors"`
}

var catalog []Connector

func init() {
	loaded, err := loadCatalog()
	if err != nil {
		panic(err)
	}
	catalog = loaded
}

func loadCatalog() ([]Connector, error) {
	raw, err := catalogFS.ReadFile("connectors.yaml")
	if err != nil {
		return nil, fmt.Errorf("mcp: read catalog: %w", err)
	}
	var file catalogFile
	if err := yaml.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("mcp: parse catalog: %w", err)
	}
	if len(file.Connectors) == 0 {
		return nil, fmt.Errorf("mcp: catalog is empty")
	}
	seen := map[string]struct{}{}
	for _, connector := range file.Connectors {
		if connector.ID == "" || connector.Name == "" || connector.URL == "" {
			return nil, fmt.Errorf("mcp: every connector needs an id, a name and a url")
		}
		if _, duplicate := seen[connector.ID]; duplicate {
			return nil, fmt.Errorf("mcp: %s is declared twice", connector.ID)
		}
		seen[connector.ID] = struct{}{}
	}
	return file.Connectors, nil
}

// Catalog is the built-in set, in the order they are declared.
func Catalog() []Connector {
	return append([]Connector(nil), catalog...)
}

// Lookup finds a connector by id.
func Lookup(id string) (Connector, bool) {
	for _, connector := range catalog {
		if connector.ID == id {
			return connector, true
		}
	}
	return Connector{}, false
}

// Search filters the catalog by name, category or description. Empty query is the lot.
func Search(query string) []Connector {
	wanted := strings.ToLower(strings.TrimSpace(query))
	if wanted == "" {
		return Catalog()
	}
	found := make([]Connector, 0, len(catalog))
	for _, connector := range catalog {
		haystack := strings.ToLower(connector.ID + " " + connector.Name + " " + connector.Category + " " + connector.Description)
		if strings.Contains(haystack, wanted) {
			found = append(found, connector)
		}
	}
	return found
}

// Endpoint is the MCP URL this connector is reached at. An instance is the shop or org
// hostname for the two that have no single global URL.
func (connector Connector) Endpoint(instance string) (string, error) {
	if connector.ID == "salesforce" {
		switch strings.ToLower(strings.TrimSpace(instance)) {
		case "", "production", "prod":
			return "https://api.salesforce.com/platform/mcp/v1/platform/sobject-all", nil
		case "sandbox", "test":
			return "https://api.salesforce.com/platform/mcp/v1/sandbox/platform/sobject-all", nil
		default:
			return "", fmt.Errorf("mcp: Salesforce instance must be production or sandbox")
		}
	}
	if !connector.InstanceRequired {
		return connector.URL, nil
	}
	host := strings.TrimSpace(instance)
	host = strings.TrimPrefix(host, "https://")
	host = strings.TrimPrefix(host, "http://")
	host = strings.TrimSuffix(host, "/")
	if host == "" {
		if connector.InstanceHint != "" {
			return "", fmt.Errorf("mcp: %s needs an instance url: %s", connector.Name, connector.InstanceHint)
		}
		return "", fmt.Errorf("mcp: %s needs an instance url", connector.Name)
	}
	return strings.ReplaceAll(connector.URL, "{instance}", host), nil
}

func or(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
