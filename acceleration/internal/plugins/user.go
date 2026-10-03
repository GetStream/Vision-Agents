package plugins

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
)

// A plugin each end user connects with their own account is offered as two tools whose
// names do not depend on the account: the model sees them before anybody has logged in,
// and a session's tools are fixed when it opens. The first call without a login asks the
// user to connect, and the calls after it reach their account.
const (
	ListToolsSuffix = "list_tools"
	CallToolSuffix  = "call_tool"
)

// AuthorizationType is the Chat attachment type that asks an end user to connect a plugin.
const AuthorizationType = "plugin_authorization"

// AuthorizationRequired is the status a user plugin's tool answers with until the user logs in.
const AuthorizationRequired = "authorization_required"

const maxAuthorizeURL = 2048

// Authorization asks an end user to connect a plugin. It reaches Chat as an attachment of
// AuthorizationType, which a client renders as a button opening AuthorizeURL.
type Authorization struct {
	Type         string `json:"type"`
	PluginID     string `json:"plugin_id"`
	Title        string `json:"title"`
	AuthorizeURL string `json:"authorize_url"`
}

// UserTools are the tools an agent is offered for the plugins its end users connect.
// An id the catalog does not have is skipped.
func UserTools(ids []string) []harness.Tool {
	var tools []harness.Tool
	for _, id := range ids {
		plugin, ok := Lookup(id)
		if !ok {
			continue
		}
		tools = append(tools,
			harness.Tool{
				Name: Prefix(plugin.ID, ListToolsSuffix),
				Description: fmt.Sprintf("List what the user's own %s account can do: %s "+
					"Call this before %s. If the user has not connected %s yet, they are "+
					"shown a button to connect it; tell them to press it and ask again.",
					plugin.Name, plugin.Description, Prefix(plugin.ID, CallToolSuffix), plugin.Name),
				Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
			},
			harness.Tool{
				Name: Prefix(plugin.ID, CallToolSuffix),
				Description: fmt.Sprintf("Run one of the tools %s lists, on the user's own %s account.",
					Prefix(plugin.ID, ListToolsSuffix), plugin.Name),
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"tool":      map[string]any{"type": "string", "description": "The tool's name, as listed."},
						"arguments": map[string]any{"type": "object", "description": "The tool's arguments, as its input schema says."},
					},
					"required": []string{"tool"},
				},
			},
		)
	}
	return tools
}

// AuthorizationResult is what a user plugin's tool answers while the user has not logged
// in. The model reads the message; the conversation turns the attachment into a Chat one.
func AuthorizationResult(plugin Plugin, authorizeURL string) string {
	raw, _ := json.Marshal(struct {
		Status     string        `json:"status"`
		Message    string        `json:"message"`
		Attachment Authorization `json:"attachment"`
	}{
		Status: AuthorizationRequired,
		Message: fmt.Sprintf("The user has not connected %s. They have been shown a button to "+
			"connect it. Tell them to press it, then ask again once they have.", plugin.Name),
		Attachment: Authorization{
			Type:         AuthorizationType,
			PluginID:     plugin.ID,
			Title:        "Connect " + plugin.Name,
			AuthorizeURL: authorizeURL,
		},
	})
	return string(raw)
}

// RequestedAuthorization reads the authorization a tool result asks for, never model prose.
// Only a result that is exactly what AuthorizationResult writes, from one of the plugin's
// own tools, for a plugin in the catalog with an https authorize URL, asks for one.
func RequestedAuthorization(tool, result string) (Authorization, bool) {
	if result == "" || len(result) > 8<<10 {
		return Authorization{}, false
	}
	var payload struct {
		Status     string        `json:"status"`
		Message    string        `json:"message"`
		Attachment Authorization `json:"attachment"`
	}
	decoder := json.NewDecoder(strings.NewReader(result))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&payload) != nil || !errors.Is(decoder.Decode(&struct{}{}), io.EOF) ||
		payload.Status != AuthorizationRequired {
		return Authorization{}, false
	}
	found := payload.Attachment
	if !ValidAuthorization(found) {
		return Authorization{}, false
	}
	if owner, _, ok := Split(tool); !ok || owner != found.PluginID {
		return Authorization{}, false
	}
	return found, true
}

// ValidAuthorization reports whether an authorization names a catalog plugin, a short title
// and an https URL to open.
func ValidAuthorization(found Authorization) bool {
	if found.Type != AuthorizationType || found.Title == "" || !utf8.ValidString(found.Title) ||
		utf8.RuneCountInString(found.Title) > 200 || len(found.AuthorizeURL) > maxAuthorizeURL {
		return false
	}
	if _, ok := Lookup(found.PluginID); !ok {
		return false
	}
	parsed, err := url.Parse(found.AuthorizeURL)
	return err == nil && parsed.Scheme == "https" && parsed.Host != "" && parsed.User == nil
}

// ListedTools is what a user plugin's list_tools answers once the user is connected.
func ListedTools(pluginID string, tools []harness.Tool) string {
	type listed struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		InputSchema map[string]any `json:"input_schema,omitempty"`
	}
	out := make([]listed, 0, len(tools))
	for _, tool := range tools {
		_, name, ok := Split(tool.Name)
		if !ok {
			continue
		}
		out = append(out, listed{
			Name:        name,
			Description: strings.TrimSuffix(tool.Description, " (via "+pluginID+")"),
			InputSchema: tool.Parameters,
		})
	}
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(map[string]any{"tools": out})
	return strings.TrimSpace(buffer.String())
}
