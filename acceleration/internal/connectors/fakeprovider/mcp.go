package fakeprovider

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

const (
	// The two MCP eras the endpoint speaks (MCP 2026-07-28 «Versioning and Compatibility»):
	// modern requests carry their version in _meta and need no handshake; legacy clients
	// open with initialize.
	modernVersion = "2026-07-28"
	legacyVersion = "2025-11-25"
	// metaProtocolVersion is the _meta key a modern request names its version with.
	metaProtocolVersion = "io.modelcontextprotocol/protocolVersion"
)

// ClaimsChallengeJSON is the claims request ClaimsChallenge asks for: the decoded example
// from Microsoft «Claims challenges, claims requests and client capabilities»
// (learn.microsoft.com/en-us/entra/identity-platform/claims-challenge). The 401 carries it
// base64-encoded; a client passes it back decoded as the claims parameter of authorize.
const ClaimsChallengeJSON = `{"access_token":{"acrs":{"essential":true,"value":"cp1"}}}`

// JSON-RPC 2.0 §5.1 error codes, and the MCP 2026-07-28 ones beside them.
const (
	codeParseError     = -32700
	codeMethodNotFound = -32601
	codeInvalidParams  = -32602
	// Streamable HTTP «Server Validation»: HeaderMismatch.
	codeHeaderMismatch = -32020
	// «Versioning and Compatibility»: UnsupportedProtocolVersionError.
	codeUnsupportedVersion = -32022
)

type rpcRequest struct {
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
	Params struct {
		Meta      map[string]any  `json:"_meta"`
		Cursor    string          `json:"cursor"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"params"`
}

// tools are what the endpoint offers, one per tools/list page so a client that ignores
// nextCursor misses one. Shapes follow MCP 2025-11-25 «Tools».
var tools = []map[string]any{
	{
		"name": "echo", "description": "Returns the text it is given.",
		"inputSchema": map[string]any{
			"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}},
			"required": []string{"text"},
		},
	},
	{
		"name": "fail", "description": "Always reports a tool execution error.",
		// «{ "type": "object", "additionalProperties": false } - Recommended» for no parameters.
		"inputSchema": map[string]any{"type": "object", "additionalProperties": false},
	},
}

// AnswerMCP has every MCP request answered with status and its reason phrase as plain text,
// whatever token it carries, until AnswerMCP(0). GitHub's MCP server answers a token it does
// not take so, with 400 Bad Request (volt E2E F60, 2026-10-09), where the fake answers 401.
func (s *Server) AnswerMCP(status int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.mcpStatus = status
}

// mcp is the MCP endpoint, a protected resource (RFC 6750) in front of a JSON-RPC server
// that answers each POST with one JSON object (Streamable HTTP).
func (s *Server) mcp(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.mcpStatus != 0 {
		http.Error(w, http.StatusText(s.mcpStatus), s.mcpStatus)
		return
	}
	metadata := `resource_metadata="` + s.URL + PathProtectedResource + `"`
	presented, found := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !found {
		// RFC 6750 §3.1: a request with no credentials gets no error code. RFC 9728 §5.1:
		// resource_metadata says where to start discovery.
		w.Header().Set("WWW-Authenticate", "Bearer "+metadata)
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	token := s.access[presented]
	if token == nil || token.grant.revoked || s.now().After(token.expires) {
		if s.is(BareChallenge) {
			w.Header().Set("WWW-Authenticate", "Bearer "+metadata)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		// RFC 6750 §3.1: «expired, revoked, malformed, or invalid for other reasons».
		w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token", `+metadata)
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	if s.is(RateLimited) {
		// RFC 9110 §10.2.3: delay-seconds.
		w.Header().Set("Retry-After", strconv.Itoa(int(RetryAfter.Seconds())))
		w.WriteHeader(http.StatusTooManyRequests)
		return
	}
	if s.is(ClaimsChallenge) && !sameJSON(token.grant.claims, ClaimsChallengeJSON) {
		// Microsoft's claims challenge: 401, realm, authorization_uri,
		// error="insufficient_claims" and claims, base64 of the claims request.
		w.Header().Set("WWW-Authenticate", `Bearer realm="", authorization_uri="`+s.URL+PathAuthorize+
			`", error="insufficient_claims", claims="`+base64.StdEncoding.EncodeToString([]byte(ClaimsChallengeJSON))+`"`)
		w.WriteHeader(http.StatusUnauthorized)
		return
	}

	var request rpcRequest
	body, err := io.ReadAll(r.Body)
	if err == nil {
		err = json.Unmarshal(body, &request)
	}
	if err != nil {
		writeRPCError(w, http.StatusBadRequest, nil, codeParseError, "parse error")
		return
	}
	if request.ID == nil {
		// A notification (JSON-RPC 2.0 §4.1); Streamable HTTP: «202 Accepted with no body».
		w.WriteHeader(http.StatusAccepted)
		return
	}
	version, modern := request.Params.Meta[metaProtocolVersion].(string)
	if modern {
		if version != modernVersion {
			writeRPCErrorData(w, http.StatusBadRequest, request.ID, codeUnsupportedVersion, "unsupported protocol version",
				map[string]any{"supported": []string{modernVersion, legacyVersion}, "requested": version})
			return
		}
		// Streamable HTTP «Server Validation»: MCP-Protocol-Version, Mcp-Method and, for
		// tools/call, Mcp-Name must be present and match the body.
		if r.Header.Get("MCP-Protocol-Version") != version || r.Header.Get("Mcp-Method") != request.Method ||
			(request.Method == "tools/call" && headerValue(r.Header.Get("Mcp-Name")) != request.Params.Name) {
			writeRPCError(w, http.StatusBadRequest, request.ID, codeHeaderMismatch, "header mismatch")
			return
		}
	} else if header := r.Header.Get("MCP-Protocol-Version"); header != "" && header != legacyVersion {
		// MCP 2025-11-25 transports: an unsupported MCP-Protocol-Version gets 400.
		http.Error(w, "unsupported MCP-Protocol-Version", http.StatusBadRequest)
		return
	}

	var result map[string]any
	switch {
	case modern && request.Method == "server/discover":
		// MCP 2026-07-28 «Discovery».
		capabilities := map[string]any{"tools": map[string]any{}}
		if s.is(MCPEvents) {
			// The draft's «Capability Declaration».
			capabilities["events"] = map[string]any{"listChanged": false}
		}
		result = map[string]any{
			"supportedVersions": []string{modernVersion, legacyVersion},
			"capabilities":      capabilities,
			"_meta":             map[string]any{"io.modelcontextprotocol/serverInfo": serverInfo()},
			"ttlMs":             0, "cacheScope": "public",
		}
	case !modern && request.Method == "initialize":
		// MCP 2025-11-25 «Lifecycle»: answer with the one legacy version spoken here.
		result = map[string]any{
			"protocolVersion": legacyVersion,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      serverInfo(),
		}
	case modern && s.is(MCPEvents) && (request.Method == "events/subscribe" || request.Method == "events/unsubscribe"):
		var envelope struct {
			Params json.RawMessage `json:"params"`
		}
		_ = json.Unmarshal(body, &envelope)
		var answered bool
		if request.Method == "events/subscribe" {
			result, answered = s.subscribeEvent(w, request.ID, token, envelope.Params)
		} else {
			result, answered = s.unsubscribeEvent(w, request.ID, token, envelope.Params)
		}
		if !answered {
			return
		}
	case request.Method == "ping":
		result = map[string]any{}
	case request.Method == "tools/list":
		page := 0
		switch request.Params.Cursor {
		case "":
		case "page-2":
			page = 1
		default:
			// MCP «Pagination»: «Invalid cursors SHOULD result in an error with code -32602».
			writeRPCError(w, http.StatusOK, request.ID, codeInvalidParams, "invalid cursor")
			return
		}
		listed := tools[page : page+1]
		if s.is(SlackUserToken) {
			listed = describedFor(listed, token.grant.account)
		}
		result = map[string]any{"tools": listed}
		if page == 0 {
			result["nextCursor"] = "page-2"
		}
		if modern {
			// MCP 2026-07-28 «Caching»: tools/list carries ttlMs and cacheScope. 0 keeps a
			// client from serving a cached list across a test's changes.
			result["ttlMs"], result["cacheScope"] = 0, "public"
		}
	case request.Method == "tools/call":
		if s.is(InsufficientScope) && !slices.Contains(token.scopes, RequiredScope) {
			// RFC 6750 §3.1 insufficient_scope with 403. MCP 2025-11-25 «Scope Challenge
			// Handling» recommends the granted scopes plus the missing one in scope.
			needed := strings.Join(append(slices.Clone(token.scopes), RequiredScope), " ")
			w.Header().Set("WWW-Authenticate", `Bearer error="insufficient_scope", scope="`+needed+`", `+metadata)
			w.WriteHeader(http.StatusForbidden)
			return
		}
		switch request.Params.Name {
		case "echo":
			var arguments struct {
				Text string `json:"text"`
			}
			_ = json.Unmarshal(request.Params.Arguments, &arguments)
			result = map[string]any{"content": []any{map[string]any{"type": "text", "text": arguments.Text}}, "isError": false}
		case "fail":
			// MCP «Tools», Error Handling: a tool execution error is a result with isError.
			result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "fail always fails"}}, "isError": true}
		default:
			// MCP «Tools»: an unknown tool is a protocol error, -32602.
			writeRPCError(w, http.StatusOK, request.ID, codeInvalidParams, "unknown tool: "+request.Params.Name)
			return
		}
	default:
		// MCP 2026-07-28 Streamable HTTP: an unknown method is 404 with -32601. A legacy
		// client gets the JSON-RPC error in a 200, as JSON-RPC over HTTP had it before.
		status := http.StatusOK
		if modern {
			status = http.StatusNotFound
		}
		writeRPCError(w, status, request.ID, codeMethodNotFound, "method not found")
		return
	}
	if modern {
		// MCP 2026-07-28 results say whether they are final.
		result["resultType"] = "complete"
	}
	writeJSON(w, http.StatusOK, map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": result})
}

// describedFor is listed with the user who consented named in each description, as Slack's
// MCP server names the signed-in user in slack_send_message, slack_search_users and
// slack_search_public («the current logged in user's user_id is U…», finding F8 of the AI-816
// end-to-end run on 2026-10-08), so the same tool has another schema digest for every user.
func describedFor(listed []map[string]any, account string) []map[string]any {
	described := make([]map[string]any, 0, len(listed))
	for _, tool := range listed {
		copied := maps.Clone(tool)
		copied["description"] = tool["description"].(string) + " The current logged in user's user_id is " + account + "."
		described = append(described, copied)
	}
	return described
}

func serverInfo() map[string]string {
	return map[string]string{"name": "fakeprovider", "version": "1"}
}

func writeRPCError(w http.ResponseWriter, status int, id json.RawMessage, code int, message string) {
	writeRPCErrorData(w, status, id, code, message, nil)
}

func writeRPCErrorData(w http.ResponseWriter, status int, id json.RawMessage, code int, message string, data any) {
	failure := map[string]any{"code": code, "message": message}
	if data != nil {
		failure["data"] = data
	}
	writeJSON(w, status, map[string]any{"jsonrpc": "2.0", "id": id, "error": failure})
}

// headerValue undoes the Base64 sentinel (=?base64?…?=) MCP 2026-07-28 Streamable HTTP
// «Value Encoding» uses for a name that is not header-safe.
func headerValue(raw string) string {
	inner, ok := strings.CutPrefix(raw, "=?base64?")
	if inner, ok2 := strings.CutSuffix(inner, "?="); ok && ok2 {
		if decoded, err := base64.StdEncoding.DecodeString(inner); err == nil {
			return string(decoded)
		}
	}
	return raw
}

// sameJSON compares two JSON documents by value, so key order and spacing do not matter.
func sameJSON(a, b string) bool {
	var x, y any
	if json.Unmarshal([]byte(a), &x) != nil || json.Unmarshal([]byte(b), &y) != nil {
		return false
	}
	return reflect.DeepEqual(x, y)
}
