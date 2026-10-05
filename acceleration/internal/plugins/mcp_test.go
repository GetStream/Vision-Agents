package plugins

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type MCPSuite struct {
	suite.Suite
}

func TestMCPSuite(t *testing.T) {
	suite.Run(t, new(MCPSuite))
}

func (s *MCPSuite) TestOpenListsPrefixedToolsAndCallReturnsText() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.Equal("Bearer secret", r.Header.Get("Authorization"))
		var body rpcRequest
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		switch body.Method {
		case "initialize":
			writeRPC(w, body.ID, map[string]any{"protocolVersion": "2025-03-26"})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeRPC(w, body.ID, toolsListResult{Tools: []mcpTool{{
				Name:        "search",
				Description: "find a message",
				InputSchema: map[string]any{"type": "object"},
			}}})
		case "tools/call":
			params, _ := body.Params.(map[string]any)
			s.Equal("search", params["name"])
			writeRPC(w, body.ID, toolsCallResult{Content: []mcpContent{{
				Type: "text",
				Text: "channel #general said hello",
			}}})
		default:
			s.Fail("unexpected method " + body.Method)
		}
	}))
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		PluginID:    "slack",
		Endpoint:    server.URL,
		AccessToken: "secret",
	}}, server.Client())
	s.Empty(failures)
	s.Require().Len(tools, 1)
	s.Equal("slack__search", tools[0].Name)
	s.True(runtime.Owns("slack__search"))
	s.False(runtime.Owns("search"))

	result, err := runtime.Call(context.Background(), llm.ToolCall{
		Name:      "slack__search",
		Arguments: `{"query":"hello"}`,
	})
	s.Require().NoError(err)
	s.Equal("channel #general said hello", result)
	runtime.Close()
}

// A tool the agent left out of its allowlist is neither offered nor callable, so the model
// cannot reach it by guessing its name.
func (s *MCPSuite) TestOnlyTheToolsAnAgentAllowsAreOffered() {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body rpcRequest
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		switch body.Method {
		case "initialize":
			writeRPC(w, body.ID, map[string]any{"protocolVersion": "2025-03-26"})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeRPC(w, body.ID, toolsListResult{Tools: []mcpTool{
				{Name: "search_files"}, {Name: "read_file_content"}, {Name: "read_file_metadata"}, {Name: "create_file"},
			}})
		case "tools/call":
			called = true
			writeRPC(w, body.ID, toolsCallResult{})
		}
	}))
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		PluginID: "google_drive", Endpoint: server.URL, Tools: []string{"search_files", "read_*"},
	}}, server.Client())
	s.Empty(failures)

	var offered []string
	for _, tool := range tools {
		offered = append(offered, tool.Name)
	}
	s.Equal([]string{"google_drive__search_files", "google_drive__read_file_content", "google_drive__read_file_metadata"}, offered)
	_, err := runtime.Call(context.Background(), llm.ToolCall{Name: "google_drive__create_file"})
	s.Error(err)
	s.False(called, "a tool left out never reaches the server")
}

func (s *MCPSuite) TestAToolPatternThatCannotBeReadIsRefused() {
	s.ErrorContains(CheckToolPatterns([]string{"read_[*"}), "read_[*")
	s.NoError(CheckToolPatterns([]string{"read_*", "search_files"}))
}

func (s *MCPSuite) TestAServerOpenedWithoutALoginKeepsWhatItSaidAtInitialize() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.Empty(r.Header.Get("Authorization"))
		var body rpcRequest
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		switch body.Method {
		case "initialize":
			writeRPC(w, body.ID, map[string]any{
				"protocolVersion": "2025-03-26",
				"instructions":    "  Keep booking links whole.  ",
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeRPC(w, body.ID, toolsListResult{Tools: []mcpTool{{Name: "search_places"}}})
		}
	}))
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		PluginID: "tablejourney",
		Endpoint: server.URL,
	}}, server.Client())

	s.Empty(failures)
	s.Require().Len(tools, 1)
	s.Equal("tablejourney__search_places", tools[0].Name)
	s.Equal("Keep booking links whole.", runtime.Instructions("tablejourney"))
	s.Empty(runtime.Instructions("slack"))
}

func (s *MCPSuite) TestAServerDescribesItselfAtInitialize() {
	var methods []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body rpcRequest
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		methods = append(methods, body.Method)
		writeRPC(w, body.ID, map[string]any{
			"protocolVersion": "2025-11-25",
			"serverInfo": map[string]any{
				"name":        "example-crm",
				"version":     "1.0.0",
				"title":       " Example CRM ",
				"description": "Search your customers, contacts, and opportunities.",
				"icons": []map[string]any{
					{"src": "data:image/png;base64,AAAA", "mimeType": "image/png"},
					{"src": "https://example.com/icon.png", "mimeType": "image/png", "sizes": []string{"128x128"}},
				},
				"websiteUrl": "https://example.com",
			},
		})
	}))
	defer server.Close()

	branding, err := Describe(context.Background(), Connection{PluginID: "crm", Endpoint: server.URL}, server.Client())

	s.Require().NoError(err)
	s.Equal(Branding{
		Title:       "Example CRM",
		Description: "Search your customers, contacts, and opportunities.",
		Version:     "1.0.0",
		IconURL:     "https://example.com/icon.png",
		WebsiteURL:  "https://example.com",
	}, branding)
	s.Equal([]string{"initialize"}, methods)
}

func (s *MCPSuite) TestAServerThatOnlyNamesItselfIsTitledByItsName() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body rpcRequest
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		writeRPC(w, body.ID, map[string]any{
			"serverInfo": map[string]any{"name": "DeepWiki", "version": "2.14.3", "websiteUrl": "javascript:alert(1)"},
		})
	}))
	defer server.Close()

	branding, err := Describe(context.Background(), Connection{PluginID: "deepwiki", Endpoint: server.URL}, server.Client())

	s.Require().NoError(err)
	s.Equal(Branding{Title: "DeepWiki", Version: "2.14.3"}, branding)
}

func (s *MCPSuite) TestAServerOnAPrivateAddressIsNeverReached() {
	var reached atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		reached.Store(true)
	}))
	defer server.Close()
	conn := Connection{PluginID: "internal", Endpoint: server.URL}

	runtime, tools, failures := Open(context.Background(), []Connection{conn}, nil)
	_, described := Describe(context.Background(), conn, nil)

	s.Nil(runtime)
	s.Empty(tools)
	s.Len(failures, 1)
	s.Error(described)
	s.False(reached.Load())
}

func (s *MCPSuite) TestAServerThatWillNotStartIsSkipped() {
	runtime, tools, failures := Open(context.Background(), []Connection{{
		PluginID: "slack",
		Endpoint: "http://127.0.0.1:1",
	}}, nil)
	s.Nil(runtime)
	s.Empty(tools)
	s.Len(failures, 1)
}

func (s *MCPSuite) TestTheSessionTheServerGivesIsSentBack() {
	var sessions []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body rpcRequest
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		sessions = append(sessions, r.Header.Get("Mcp-Session-Id"))
		switch body.Method {
		case "initialize":
			w.Header().Set("Mcp-Session-Id", "session-1")
			writeRPC(w, body.ID, map[string]any{"protocolVersion": "2025-03-26"})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeRPC(w, body.ID, toolsListResult{})
		}
	}))
	defer server.Close()

	_, _, failures := Open(context.Background(), []Connection{{PluginID: "sentry", Endpoint: server.URL}}, server.Client())

	s.Empty(failures)
	s.Equal([]string{"", "session-1", "session-1"}, sessions)
}

func (s *MCPSuite) TestALoginTheServerRefusesIsToldApart() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"invalid_token"}`, http.StatusUnauthorized)
	}))
	defer server.Close()

	_, _, failures := Open(context.Background(), []Connection{{PluginID: "sentry", Endpoint: server.URL}}, server.Client())

	s.Require().Len(failures, 1)
	s.ErrorIs(failures[0], ErrUnauthorized)
}

func writeRPC(w http.ResponseWriter, id int, result any) {
	raw, _ := json.Marshal(result)
	_ = json.NewEncoder(w).Encode(rpcResponse{JSONRPC: "2.0", ID: id, Result: raw})
}
