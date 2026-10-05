//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// MCPBrandingSuite covers a config's MCP servers describing themselves when it is saved.
// Every server is stood in for by one that answers initialize with a serverInfo, until it
// is told to stop answering.
type MCPBrandingSuite struct {
	RouterSuite
	down atomic.Bool
}

func TestMCPBrandingSuite(t *testing.T) {
	runSuite(t, new(MCPBrandingSuite))
}

func (s *MCPBrandingSuite) SetupSuite() {
	s.pluginMCP = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.down.Load() {
			http.Error(w, "down", http.StatusBadGateway)
			return
		}
		var body struct {
			ID int `json:"id"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": body.ID, "result": map[string]any{
			"protocolVersion": "2025-11-25",
			"serverInfo": map[string]any{
				"name":        "example-crm",
				"version":     "1.0.0",
				"title":       "Example CRM",
				"description": "Search your customers, contacts, and opportunities.",
				"icons":       []map[string]any{{"src": "https://example.com/icon.png", "mimeType": "image/png"}},
				"websiteUrl":  "https://example.com",
			},
		}})
	}))
	s.T().Cleanup(s.pluginMCP.Close)
	s.RouterSuite.SetupSuite()
}

func (s *MCPBrandingSuite) SetupTest() {
	s.down.Store(false)
	s.useApp(s.data.createApp())
}

func (s *MCPBrandingSuite) TestAServerSaysWhatItIsWhenTheConfigIsSaved() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "crm", "mcp_servers": []map[string]any{{"name": "crm", "url": "https://crm.example.com/mcp"}}}, &created))

	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	servers := value(read.McpServers)
	s.Require().Len(servers, 1)
	s.Equal(&McpServerBranding{
		Title:       optional("Example CRM"),
		Description: optional("Search your customers, contacts, and opportunities."),
		Version:     optional("1.0.0"),
		IconUrl:     optional("https://example.com/icon.png"),
		WebsiteUrl:  optional("https://example.com"),
	}, servers[0].Branding)
}

func (s *MCPBrandingSuite) TestAServerThatStopsAnsweringKeepsWhatItSaidBefore() {
	servers := []map[string]any{{"name": "crm", "url": "https://crm.example.com/mcp"}}
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "crm", "mcp_servers": servers}, &created))
	s.down.Store(true)

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"mcp_servers": servers}, &patched))

	s.Require().Len(value(patched.McpServers), 1)
	s.Require().NotNil(value(patched.McpServers)[0].Branding)
	s.Equal("Example CRM", value(value(patched.McpServers)[0].Branding.Title))
}

func (s *MCPBrandingSuite) TestAServerThatNeverAnsweredIsSavedWithoutBranding() {
	s.down.Store(true)

	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "crm", "mcp_servers": []map[string]any{{"name": "crm", "url": "https://crm.example.com/mcp"}}}, &created))

	s.Equal([]McpServer{{Name: "crm", Url: "https://crm.example.com/mcp"}}, value(created.McpServers))
}
