package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// pingClientTimeout is the timeout of the connection's client in these tests, standing in
// for connectorHTTPTimeout (10 s in the router). A choice: short enough for a test to wait.
const pingClientTimeout = 300 * time.Millisecond

// boundedWithin is how long Discover or Close may take against the pinging server before the
// test says it hangs. A choice: several times every bound involved.
const boundedWithin = 5 * time.Second

// PingSuite runs the source against a server that asks the client something mid-request: a
// stateful MCP server of the official SDK that pings while it answers tools/list and
// tools/call, and never answers the POST that carries the client's reply. go-sdk v1.8.0 posts
// that reply with no deadline of its own (internal/jsonrpc2/conn.go), and closing the session
// waits for it, so only the client's timeout ends it. Both paths must stay bounded.
type PingSuite struct {
	suite.Suite
	server *httptest.Server
	source *Source
}

func TestPingSuite(t *testing.T) {
	suite.Run(t, new(PingSuite))
}

func (s *PingSuite) SetupTest() {
	server := mcp.NewServer(&mcp.Implementation{Name: "pinging", Version: "1"}, nil)
	server.AddTool(pingingTool, func(context.Context, *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "pinged"}}}, nil
	})
	// Ping on the request's own stream, and answer once the ping has gone out.
	server.AddReceivingMiddleware(func(next mcp.MethodHandler) mcp.MethodHandler {
		return func(ctx context.Context, method string, request mcp.Request) (mcp.Result, error) {
			if method == "tools/list" || method == "tools/call" {
				if session, ok := request.GetSession().(*mcp.ServerSession); ok {
					go func() { _ = session.Ping(ctx, nil) }()
					time.Sleep(100 * time.Millisecond)
				}
			}
			return next(ctx, method, request)
		}
	})
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	released := make(chan struct{})
	s.server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(raw))
		var message struct {
			Method string          `json:"method"`
			ID     json.RawMessage `json:"id"`
		}
		// A JSON-RPC response, the client's reply to the ping, is never answered.
		if json.Unmarshal(raw, &message) == nil && message.Method == "" && message.ID != nil {
			select {
			case <-released:
			case <-r.Context().Done():
			}
			return
		}
		handler.ServeHTTP(w, r)
	}))
	s.T().Cleanup(s.server.Close)
	s.T().Cleanup(func() { close(released) })
	s.source = New()
}

// TestDiscoverAgainstAServerThatPingsIsBoundedByTheClientsTimeout: validate's Discover sends
// with the connection's client as it is, so its timeout still ends the reply nobody answers.
func (s *PingSuite) TestDiscoverAgainstAServerThatPingsIsBoundedByTheClientsTimeout() {
	type discovered struct {
		specs []core.ToolSpec
		err   error
	}
	done := make(chan discovered, 1)

	go func() {
		specs, err := s.source.Discover(context.Background(), s.binding())
		done <- discovered{specs, err}
	}()

	select {
	case got := <-done:
		s.Require().NoError(got.err)
		s.Len(got.specs, 1)
	case <-time.After(boundedWithin):
		s.Fail("Discover waited on the reply nobody answers")
	}
}

// TestCloseAfterACallToAServerThatPingsIsBoundedByTheCallTimeout: an opened Toolset's client
// has callTimeout, which still ends the reply nobody answers.
func (s *PingSuite) TestCloseAfterACallToAServerThatPingsIsBoundedByTheCallTimeout() {
	s.source.callTimeout = pingClientTimeout
	digest, err := ToolSchemaDigest(pingingTool)
	s.Require().NoError(err)
	set, err := s.source.Open(context.Background(), s.binding(), []core.ToolGrant{{Name: pingingTool.Name, SchemaDigest: digest}})
	s.Require().NoError(err)
	result, err := set.Call(context.Background(), llm.ToolCall{Name: "crm__" + pingingTool.Name, Arguments: "{}"})
	s.Require().NoError(err)
	s.Equal("pinged", llm.TextOf(result.Parts))
	closed := make(chan struct{})

	go func() {
		set.Close()
		close(closed)
	}()

	select {
	case <-closed:
	case <-time.After(boundedWithin):
		s.Fail("Close waited on the reply nobody answers")
	}
}

func (s *PingSuite) TestAnOpenedToolsetsClientWaitsAtLeastTheCallTimeout() {
	short := &http.Client{Timeout: 10 * time.Second}

	s.Equal(35*time.Second, s.source.callClient(short).Timeout)
	s.Equal(10*time.Second, short.Timeout, "the connection's own client is left as it is")
}

// binding reaches the pinging server through core.Transports with the bearer scheme and a
// client whose timeout is pingClientTimeout.
func (s *PingSuite) binding() core.ResolvedBinding {
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: newResolver("token-1"),
		Timeout: pingClientTimeout, NewClient: loopback(s.server.Client())})
	s.Require().NoError(err)
	return core.ResolvedBinding{
		Binding:  core.Binding{Name: "crm"},
		Manifest: manifest(s.server.URL + "/mcp"),
		HTTP:     transports.Client(core.ConnectionRef{CustomerID: "app", ConnectionID: "crm-1"}, bearer.New()),
	}
}

// pingingTool is the one tool the pinging server offers.
var pingingTool = &mcp.Tool{Name: "pinging", Description: "Answers after a ping.",
	InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}
