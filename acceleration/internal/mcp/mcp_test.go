package mcp

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"unicode/utf8"

	"github.com/modelcontextprotocol/go-sdk/mcp"
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
	server := newMCPTestServer(s.T(), "Bearer secret", func(w http.ResponseWriter, request map[string]any) {
		method := request["method"].(string)
		params, _ := request["params"].(map[string]any)
		switch method {
		case "server/discover":
			writeMCPError(w, request["id"], -32601, "method not found")
		case "initialize":
			writeMCPResult(w, request["id"], map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "test", "version": "1"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeMCPResult(w, request["id"], map[string]any{"tools": []any{
				map[string]any{
					"name":        "search",
					"description": "find a message",
					"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
				},
			}})
		case "tools/call":
			s.Equal("search", params["name"])
			writeMCPResult(w, request["id"], map[string]any{"content": []any{
				map[string]any{"type": "text", "text": "channel #general said hello"},
			}})
		default:
			s.Fail("unexpected method " + method)
		}
	})
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID:    "slack",
		Endpoint:    server.URL,
		AccessToken: "secret",
	}}, server.Client())
	s.Empty(failures)
	s.Require().Len(tools, 1)
	s.Equal("slack__search", tools[0].Name)
	wantDigest, err := ToolSchemaDigest(&mcp.Tool{
		Name: "search", Description: "find a message",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
	})
	s.Require().NoError(err)
	s.Equal(wantDigest, tools[0].SchemaDigest)
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

func (s *MCPSuite) TestOpenReadsEveryToolPage() {
	var cursors []string
	server := newMCPTestServer(s.T(), "", func(w http.ResponseWriter, request map[string]any) {
		switch request["method"] {
		case "server/discover":
			writeMCPResult(w, request["id"], map[string]any{
				"resultType": "complete",
				"_meta": map[string]any{
					"io.modelcontextprotocol/serverInfo": map[string]any{"name": "test", "version": "1"},
				},
				"ttlMs": 0, "cacheScope": "public",
				"supportedVersions": []string{"2026-07-28", "2025-11-25"},
				"capabilities":      map[string]any{"tools": map[string]any{}},
			})
		case "tools/list":
			params, _ := request["params"].(map[string]any)
			cursor, _ := params["cursor"].(string)
			cursors = append(cursors, cursor)
			switch cursor {
			case "":
				writeMCPResult(w, request["id"], map[string]any{
					"tools": []any{map[string]any{
						"name": "first", "description": "first page", "inputSchema": map[string]any{"type": "object"},
					}}, "nextCursor": "page-2",
				})
			case "page-2":
				writeMCPResult(w, request["id"], map[string]any{
					"tools": []any{map[string]any{
						"name": "second", "description": "second page", "inputSchema": map[string]any{"type": "object"},
					}},
				})
			default:
				s.Fail("unexpected tools/list cursor " + cursor)
			}
		default:
			s.Fail("unexpected MCP method " + request["method"].(string))
		}
	})
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID: "linear", Endpoint: server.URL,
	}}, server.Client())
	s.Require().NotNil(runtime)
	defer runtime.Close()
	s.Empty(failures)
	s.Require().Len(tools, 2)
	s.ElementsMatch([]string{"linear__first", "linear__second"}, []string{tools[0].Name, tools[1].Name})
	s.Equal([]string{"", "page-2"}, cursors)
}

func (s *MCPSuite) TestToolArgumentsAreValidatedBeforeDispatch() {
	var calls atomic.Int32
	server := newMCPTestServer(s.T(), "", func(w http.ResponseWriter, request map[string]any) {
		switch request["method"] {
		case "server/discover":
			writeMCPError(w, request["id"], -32601, "method not found")
		case "initialize":
			writeMCPResult(w, request["id"], map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "test", "version": "1"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeMCPResult(w, request["id"], map[string]any{"tools": []any{
				map[string]any{
					"name":        "search",
					"description": "search messages",
					"inputSchema": map[string]any{
						"type":                 "object",
						"properties":           map[string]any{"query": map[string]any{"type": "string", "minLength": 3}},
						"required":             []string{"query"},
						"additionalProperties": false,
					},
				},
			}})
		case "tools/call":
			calls.Add(1)
			writeMCPResult(w, request["id"], map[string]any{"content": []any{
				map[string]any{"type": "text", "text": "searched"},
			}})
		default:
			s.Fail("unexpected method " + request["method"].(string))
		}
	})
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID: "slack", Endpoint: server.URL,
	}}, server.Client())
	s.Empty(failures)
	s.Require().Len(tools, 1)
	defer runtime.Close()

	for _, arguments := range []string{`{}`, `{"query":3}`, `{"query":"ok"}`, `{"query":"valid","other":true}`} {
		_, err := runtime.Call(context.Background(), llm.ToolCall{
			Name: "slack__search", Arguments: arguments,
		})
		s.ErrorContains(err, "arguments do not match the accepted tool schema")
	}
	s.Zero(calls.Load(), "invalid model arguments must not reach the connector")

	result, err := runtime.Call(context.Background(), llm.ToolCall{
		Name: "slack__search", Arguments: `{"query":"valid"}`,
	})
	s.Require().NoError(err)
	s.Equal("searched", result)
	s.Equal(int32(1), calls.Load())
}

func (s *MCPSuite) TestExternalToolSchemaReferencesAreNotFetched() {
	var externalRequests atomic.Int32
	externalSchema := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		externalRequests.Add(1)
		_, _ = io.WriteString(w, `{"type":"object"}`)
	}))
	defer externalSchema.Close()
	server := newMCPTestServer(s.T(), "", func(w http.ResponseWriter, request map[string]any) {
		switch request["method"] {
		case "server/discover":
			writeMCPError(w, request["id"], -32601, "method not found")
		case "initialize":
			writeMCPResult(w, request["id"], map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "test", "version": "1"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeMCPResult(w, request["id"], map[string]any{"tools": []any{
				map[string]any{
					"name":        "search",
					"description": "search messages",
					"inputSchema": map[string]any{"$ref": externalSchema.URL + "/schema.json"},
				},
			}})
		default:
			s.Fail("unexpected method " + request["method"].(string))
		}
	})
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID: "slack", Endpoint: server.URL,
	}}, server.Client())
	s.Empty(tools)
	s.Require().NotEmpty(failures)
	s.ErrorContains(failures[0], "invalid input schema")
	s.Zero(externalRequests.Load(), "tool schemas must not trigger external fetches")
	if runtime != nil {
		runtime.Close()
	}
}

func (s *MCPSuite) TestCollidingPrefixedNamesAreNotExposed() {
	newServer := func(toolName string) *httptest.Server {
		return newMCPTestServer(s.T(), "", func(w http.ResponseWriter, request map[string]any) {
			switch request["method"] {
			case "server/discover":
				writeMCPError(w, request["id"], -32601, "method not found")
			case "initialize":
				writeMCPResult(w, request["id"], map[string]any{
					"protocolVersion": "2025-11-25",
					"capabilities":    map[string]any{"tools": map[string]any{}},
					"serverInfo":      map[string]any{"name": "test", "version": "1"},
				})
			case "notifications/initialized":
				w.WriteHeader(http.StatusAccepted)
			case "tools/list":
				writeMCPResult(w, request["id"], map[string]any{"tools": []any{
					map[string]any{
						"name":        toolName,
						"description": "test tool",
						"inputSchema": map[string]any{"type": "object"},
					},
				}})
			default:
				s.Fail("unexpected method " + request["method"].(string))
			}
		})
	}
	first := newServer("bar__baz")
	second := newServer("baz")
	defer first.Close()
	defer second.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{
		{ConnectorID: "one", ToolPrefix: "foo", Endpoint: first.URL},
		{ConnectorID: "two", ToolPrefix: "foo__bar", Endpoint: second.URL},
	}, first.Client())
	s.Require().NotNil(runtime)
	defer runtime.Close()
	s.Empty(tools)
	s.False(runtime.Owns("foo__bar__baz"))
	s.Require().Len(failures, 2)
	s.Contains(failures[0].Error(), "one")
	s.Contains(failures[1].Error(), "two")
}

func (s *MCPSuite) TestToolResultsAreBoundedAndKeepStructuredAndErrorResults() {
	largeText := strings.Repeat("é", maxMCPToolResultBytes)
	server := newMCPTestServer(s.T(), "", func(w http.ResponseWriter, request map[string]any) {
		params, _ := request["params"].(map[string]any)
		switch request["method"] {
		case "server/discover":
			writeMCPError(w, request["id"], -32601, "method not found")
		case "initialize":
			writeMCPResult(w, request["id"], map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "test", "version": "1"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeMCPResult(w, request["id"], map[string]any{"tools": []any{
				map[string]any{"name": "large", "description": "large result", "inputSchema": map[string]any{"type": "object"}},
				map[string]any{"name": "structured", "description": "structured result", "inputSchema": map[string]any{"type": "object"}},
				map[string]any{"name": "image_error", "description": "image error", "inputSchema": map[string]any{"type": "object"}},
			}})
		case "tools/call":
			switch params["name"] {
			case "large":
				writeMCPResult(w, request["id"], map[string]any{"content": []any{
					map[string]any{"type": "text", "text": largeText},
				}})
			case "structured":
				writeMCPResult(w, request["id"], map[string]any{"structuredContent": map[string]any{"count": 3}})
			case "image_error":
				writeMCPResult(w, request["id"], map[string]any{
					"content": []any{map[string]any{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}},
					"isError": true,
				})
			}
		}
	})
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID: "gong", Endpoint: server.URL,
	}}, server.Client())
	s.Empty(failures)
	s.Require().Len(tools, 3)
	defer runtime.Close()

	large, err := runtime.Call(context.Background(), llm.ToolCall{Name: "gong__large", Arguments: "{}"})
	s.Require().NoError(err)
	s.True(utf8.ValidString(large))
	s.LessOrEqual(len(large), maxMCPToolResultBytes)
	s.True(strings.HasSuffix(large, truncatedToolResultNotice))

	structured, err := runtime.Call(context.Background(), llm.ToolCall{Name: "gong__structured", Arguments: "{}"})
	s.Require().NoError(err)
	s.Equal(`{"count":3}`, structured)

	_, err = runtime.Call(context.Background(), llm.ToolCall{Name: "gong__image_error", Arguments: "{}"})
	s.ErrorContains(err, "non-text content")
}

func (s *MCPSuite) TestMCPResponseBodyHasAStrictSizeLimit() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, strings.Repeat("x", maxMCPResponseBytes+1))
	}))
	defer server.Close()

	request := httptest.NewRequest(http.MethodGet, server.URL, nil)
	response, err := (authorizedTransport{base: server.Client().Transport}).RoundTrip(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	_, err = io.ReadAll(response.Body)
	s.ErrorAs(err, new(*http.MaxBytesError))
}

func (s *MCPSuite) TestExplicitEmptyToolGrantExposesNothing() {
	server := newMCPTestServer(s.T(), "", func(w http.ResponseWriter, request map[string]any) {
		switch request["method"] {
		case "server/discover":
			writeMCPError(w, request["id"], -32601, "method not found")
		case "initialize":
			writeMCPResult(w, request["id"], map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "test", "version": "1"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeMCPResult(w, request["id"], map[string]any{"tools": []any{
				map[string]any{"name": "search", "description": "search", "inputSchema": map[string]any{"type": "object"}},
			}})
		default:
			s.Fail("unexpected method " + request["method"].(string))
		}
	})
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID: "linear", Endpoint: server.URL, AllowedTools: []string{},
	}}, server.Client())
	s.Empty(failures)
	s.Empty(tools)
	s.False(runtime.Owns("linear__search"))
	runtime.Close()
}

func (s *MCPSuite) TestChangedToolSchemaIsNotExposed() {
	approved, err := ToolSchemaDigest(&mcp.Tool{
		Name: "search", Description: "find a message",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}}},
	})
	s.Require().NoError(err)
	server := newMCPTestServer(s.T(), "", func(w http.ResponseWriter, request map[string]any) {
		switch request["method"] {
		case "server/discover":
			writeMCPError(w, request["id"], -32601, "method not found")
		case "initialize":
			writeMCPResult(w, request["id"], map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "test", "version": "1"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeMCPResult(w, request["id"], map[string]any{"tools": []any{
				map[string]any{
					"name":        "search",
					"description": "search across messages",
					"inputSchema": map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}}},
				},
			}})
		default:
			s.Fail("unexpected method " + request["method"].(string))
		}
	})
	defer server.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID: "slack", Endpoint: server.URL,
		AllowedTools: []string{"search"}, ExpectedToolDigests: map[string]string{"search": approved},
	}}, server.Client())
	s.NotNil(runtime)
	s.Empty(tools)
	s.Require().Len(failures, 1)
	s.ErrorContains(failures[0], "slack: tool search schema changed")
	runtime.Close()
}

func (s *MCPSuite) TestAServerThatWillNotStartIsSkipped() {
	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID: "slack",
		Endpoint: "http://127.0.0.1:1",
	}}, nil)
	s.Nil(runtime)
	s.Empty(tools)
	s.Len(failures, 1)
}

func (s *MCPSuite) TestMCPRedirectCannotForwardTheBearerTokenToAnotherHost() {
	var receivedBearer atomic.Bool
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer secret" {
			receivedBearer.Store(true)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.Equal("Bearer secret", r.Header.Get("Authorization"))
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()

	runtime, tools, failures := Open(context.Background(), []Connection{{
		ConnectorID: "slack", Endpoint: source.URL, AccessToken: "secret",
	}}, source.Client())

	s.Nil(runtime)
	s.Empty(tools)
	s.NotEmpty(failures)
	s.False(receivedBearer.Load())
}

func (s *MCPSuite) TestRequestAuthorizerCanSupplyAProviderSpecificHeader() {
	var seen atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.Equal("secret-key", r.Header.Get("X-API-Key"))
		s.Empty(r.Header.Get("Authorization"))
		seen.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	transport := authorizedTransport{
		base: server.Client().Transport,
		authorize: func(_ context.Context, request *http.Request) error {
			request.Header.Set("X-API-Key", "secret-key")
			return nil
		},
	}
	for range 2 {
		request := httptest.NewRequest(http.MethodGet, server.URL, nil)
		response, err := transport.RoundTrip(request)
		s.Require().NoError(err)
		s.Require().NoError(response.Body.Close())
	}
	s.Equal(int32(2), seen.Load())
}

func newMCPTestServer(t *testing.T, authorization string, handle func(http.ResponseWriter, map[string]any)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != authorization {
			t.Errorf("authorization = %q, want %q", got, authorization)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode MCP request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		handle(w, request)
	}))
}

func writeMCPResult(w http.ResponseWriter, id any, result any) {
	_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
}

func writeMCPError(w http.ResponseWriter, id any, code int, message string) {
	_ = json.NewEncoder(w).Encode(map[string]any{
		"jsonrpc": "2.0", "id": id,
		"error": map[string]any{"code": code, "message": message},
	})
}
