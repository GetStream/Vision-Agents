package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// The client half of MCP Events' webhook delivery: asking a connection's MCP server to deliver
// an event to a callback the router serves. The draft is
// https://github.com/modelcontextprotocol/experimental-ext-triggers-events, docs/
// design-sketch-proposal.md at commit 6682596d (September 8, 2026), «Status: Draft proposal»;
// «Webhook-Based Delivery» is what is implemented here. The plugin system's client
// (internal/plugins/events.go) is the working code this follows, on the connection's client.

// eventsVersion is the MCP protocol version events are asked for in: the plugin system's
// plugins.EventsVersion (internal/plugins/events.go), and the modern version the fake provider
// speaks (fakeprovider.modernVersion, MCP 2026-07-28 «Versioning and Compatibility»).
const eventsVersion = "2026-07-28"

// metaProtocolVersion is the _meta key a 2026-07-28 request names its version with
// (fakeprovider.metaProtocolVersion).
const metaProtocolVersion = "io.modelcontextprotocol/protocolVersion"

// webhookMode is the delivery mode the draft's events/subscribe takes for a webhook
// («Subscribing: events/subscribe», delivery.mode).
const webhookMode = "webhook"

var _ core.EventSource = (*Source)(nil)

// Subscribe asks the connection's server to deliver one event to sub.URL, signed with
// sub.Secret, after checking with server/discover that it offers events at all. The server
// keys the subscription on the connection's principal, the URL, the event and its arguments,
// so asking again with the same four refreshes it («Subscription Identity»). cursor is null:
// delivery starts from now, as the plugin system asks.
func (s *Source) Subscribe(ctx context.Context, b core.ResolvedBinding, sub core.EventSubscription) (core.EventGrant, error) {
	client, err := s.eventsClient(ctx, b)
	if err != nil {
		return core.EventGrant{}, err
	}
	raw, err := client.call(ctx, "events/subscribe", map[string]any{
		"name":      sub.Name,
		"arguments": argumentsOf(sub.Arguments),
		"delivery":  map[string]string{"mode": webhookMode, "url": sub.URL, "secret": sub.Secret},
		"cursor":    nil,
	})
	if err != nil {
		return core.EventGrant{}, err
	}
	var granted struct {
		ID            string     `json:"id"`
		RefreshBefore *time.Time `json:"refreshBefore"`
	}
	if err := json.Unmarshal(raw, &granted); err != nil {
		return core.EventGrant{}, stack.Wrap(fmt.Errorf("mcp: %s: events/subscribe: %w", b.Manifest.ConnectorID, err))
	}
	return core.EventGrant{ID: granted.ID, RefreshBefore: granted.RefreshBefore}, nil
}

// Unsubscribe stops a subscription, named as it was made: event, arguments and URL
// («events/unsubscribe {name, arguments, delivery: {url}}»).
func (s *Source) Unsubscribe(ctx context.Context, b core.ResolvedBinding, sub core.EventSubscription) error {
	client, err := s.eventsClient(ctx, b)
	if err != nil {
		return err
	}
	_, err = client.call(ctx, "events/unsubscribe", map[string]any{
		"name":      sub.Name,
		"arguments": argumentsOf(sub.Arguments),
		"delivery":  map[string]string{"mode": webhookMode, "url": sub.URL},
	})
	return err
}

// eventsClient is a JSON-RPC client on the connection's MCP endpoint, through its own client
// with every response capped, once server/discover says the server offers events
// («Capability Declaration»: capabilities.events). Both requests are bounded by the startup
// timeout, as Discover is.
func (s *Source) eventsClient(ctx context.Context, b core.ResolvedBinding) (*rpcClient, error) {
	if b.HTTP == nil {
		return nil, stack.Wrap(errors.New("mcp: the binding has no client: build it with core.Transports"))
	}
	rule := sourceRule(b.Manifest)
	if rule == nil {
		return nil, stack.Wrap(fmt.Errorf("mcp: connector %q has no mcp source", b.Manifest.ConnectorID))
	}
	endpoint := b.Manifest.Endpoints[rule.Endpoint]
	if endpoint == "" {
		return nil, stack.Wrap(fmt.Errorf("mcp: connector %q has no %s endpoint resolved", b.Manifest.ConnectorID, rule.Endpoint))
	}
	// A copy whose transport caps each response, as connect's is: the credential, the 401
	// renewal and the egress checks all still run.
	http := *b.HTTP
	http.Transport = capped{base: b.HTTP.Transport}
	client := &rpcClient{connector: b.Manifest.ConnectorID, endpoint: endpoint, http: &http, timeout: s.startupTimeout}
	raw, err := client.call(ctx, "server/discover", map[string]any{})
	if err != nil {
		return nil, err
	}
	var discovered struct {
		Capabilities struct {
			Events json.RawMessage `json:"events"`
		} `json:"capabilities"`
	}
	if err := json.Unmarshal(raw, &discovered); err != nil {
		return nil, stack.Wrap(fmt.Errorf("mcp: %s: server/discover: %w", b.Manifest.ConnectorID, err))
	}
	if len(discovered.Capabilities.Events) == 0 || string(discovered.Capabilities.Events) == "null" {
		return nil, stack.Wrap(fmt.Errorf("%w: %s", core.ErrNoEvents, b.Manifest.ConnectorID))
	}
	return client, nil
}

// argumentsOf is the arguments as the draft sends them: an object, never null.
func argumentsOf(arguments map[string]any) map[string]any {
	if arguments == nil {
		return map[string]any{}
	}
	return arguments
}

// rpcClient sends one JSON-RPC request per POST in MCP 2026-07-28's stateless form: the version
// in the body's _meta and in MCP-Protocol-Version, the method in Mcp-Method, and no
// initialize (fakeprovider's «Server Validation»). The official SDK's ClientSession has no call
// for a method it does not know, so events are asked for by hand, as the plugin system's
// client (plugins.client) asks.
type rpcClient struct {
	connector string
	endpoint  string
	http      *http.Client
	timeout   time.Duration
	nextID    int
}

func (c *rpcClient) call(ctx context.Context, method string, params map[string]any) (json.RawMessage, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	c.nextID++
	params["_meta"] = map[string]string{metaProtocolVersion: eventsVersion}
	body, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": c.nextID, "method": method, "params": params})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, stack.Wrap(err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json, text/event-stream")
	request.Header.Set("MCP-Protocol-Version", eventsVersion)
	request.Header.Set("Mcp-Method", method)
	response, err := c.http.Do(request)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("mcp: %s: %s: %w", c.connector, method, err))
	}
	defer func() { _ = response.Body.Close() }()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("mcp: %s: %s: %w", c.connector, method, err))
	}
	if strings.Contains(response.Header.Get("Content-Type"), "text/event-stream") {
		raw = lastEventData(raw)
	}
	var answer struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	// A JSON-RPC error comes in a 200 or, from a 2026-07-28 server, with a 4xx; either way it
	// says what went wrong better than the status does.
	if json.Unmarshal(raw, &answer) == nil && answer.Error != nil {
		return nil, stack.Wrap(fmt.Errorf("mcp: %s: %s: %d %s", c.connector, method, answer.Error.Code, answer.Error.Message))
	}
	if response.StatusCode >= 300 {
		return nil, stack.Wrap(fmt.Errorf("mcp: %s: %s: the server answered %d", c.connector, method, response.StatusCode))
	}
	if answer.Result == nil {
		return nil, stack.Wrap(fmt.Errorf("mcp: %s: %s: no result", c.connector, method))
	}
	return answer.Result, nil
}

// lastEventData is the last data line of a server-sent event stream, which is the answer a
// server streaming one response ends with (the plugin system's sseData).
func lastEventData(raw []byte) []byte {
	var last []byte
	for _, line := range bytes.Split(raw, []byte("\n")) {
		if data, found := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data:")); found {
			last = bytes.TrimSpace(data)
		}
	}
	return last
}
