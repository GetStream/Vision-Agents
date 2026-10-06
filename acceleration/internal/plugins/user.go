package plugins

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"slices"
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
//
// Everything after AuthorizeURL is a standard Chat attachment field, so a client that has
// no renderer for AuthorizationType still shows a card with the plugin's logo, what it is
// for and a link to the login rather than an empty bubble. They are the three Chat keeps
// as an attachment's own on a partial update, next to the type and the title; author_name
// is not one of them, and the title already names the plugin.
type Authorization struct {
	Type         string `json:"type"`
	PluginID     string `json:"plugin_id"`
	Title        string `json:"title"`
	AuthorizeURL string `json:"authorize_url"`
	Text         string `json:"text,omitempty"`
	ThumbURL     string `json:"thumb_url,omitempty"`
	TitleLink    string `json:"title_link,omitempty"`
	// Status is AuthorizationConnected once the user has finished this login.
	Status string `json:"status,omitempty"`
}

// AuthorizationConnected is the Status of an authorization the user has finished, which a
// client shows as connected rather than as a button to press again.
const AuthorizationConnected = "connected"

// UserTools are the tools an agent is offered for the plugins its end users connect.
func UserTools(offered []Plugin) []harness.Tool {
	var tools []harness.Tool
	for _, plugin := range offered {
		about := "."
		if plugin.Description != "" {
			about = ": " + plugin.Description
		}
		tools = append(tools,
			harness.Tool{
				Name: Prefix(plugin.ID, ListToolsSuffix),
				Description: fmt.Sprintf("List what the user's own %s account can do%s "+
					"Call this before %s. If the user has not connected %s yet, they are "+
					"shown a button to connect it; tell them to press it and ask again.",
					plugin.Name, about, Prefix(plugin.ID, CallToolSuffix), plugin.Name),
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
// logoURL is where this deployment serves the plugin's logo, which Auth.LogoURL gives.
func AuthorizationResult(plugin Plugin, authorizeURL, logoURL string) string {
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
			Text:         plugin.Description,
			ThumbURL:     logoURL,
			TitleLink:    authorizeURL,
		},
	})
	return string(raw)
}

// Unavailable is the status a user plugin's tool answers when the agent cannot log anybody
// into it, because the provider needs an OAuth client and the agent was given none.
const Unavailable = "unavailable"

// UnavailableResult is what a user plugin's tool answers when nobody can connect it here.
// It says nothing about how to set the plugin up: that is for whoever runs the agent, not
// for the person in the conversation.
func UnavailableResult(plugin Plugin) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status: Unavailable,
		Message: fmt.Sprintf("%s is not available to this agent yet, so the user cannot connect it. "+
			"Tell them it is not available here; do not offer to set it up.", plugin.Name),
	})
	return string(raw)
}

// RequestedAuthorization reads the authorization a tool result asks for, never model prose.
// Only a result that is exactly what AuthorizationResult writes, from one of the plugin's
// own tools, with an https authorize URL, asks for one. logins are the servers the session
// reaches that log in: a server that does not cannot ask, whatever its tools answer.
func RequestedAuthorization(tool, result string, logins []string) (Authorization, bool) {
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
	// Only the router's own callback says a login is finished, never a tool's result.
	found.Status = ""
	if !ValidAuthorization(found) {
		return Authorization{}, false
	}
	owner, _, ok := Split(tool)
	if !ok || owner != found.PluginID || !slices.Contains(logins, owner) {
		return Authorization{}, false
	}
	return found, true
}

// ValidAuthorization reports whether an authorization names a plugin, a short title and an
// https URL to open.
//
// Everything the card shows besides the title and that URL has to be what the catalog says
// for the plugin named, so a server answering through one plugin's own tool cannot describe
// itself as another's, and cannot put an image of its choosing in somebody's conversation.
// A server named by URL is in no catalog, so its card says only its name.
func ValidAuthorization(found Authorization) bool {
	if found.Type != AuthorizationType || found.Title == "" || !utf8.ValidString(found.Title) ||
		utf8.RuneCountInString(found.Title) > 200 || len(found.AuthorizeURL) > maxAuthorizeURL ||
		found.Status != "" && found.Status != AuthorizationConnected {
		return false
	}
	plugin, ok := Lookup(found.PluginID)
	if !ok {
		if found.PluginID == "" || found.Title != "Connect "+found.PluginID || found.Text != "" || found.ThumbURL != "" {
			return false
		}
		plugin = Plugin{ID: found.PluginID}
	}
	if found.Text != "" && found.Text != plugin.Description {
		return false
	}
	if found.TitleLink != "" && found.TitleLink != found.AuthorizeURL {
		return false
	}
	if found.ThumbURL != "" && !ownLogo(found.ThumbURL, plugin.ID) {
		return false
	}
	parsed, err := url.Parse(found.AuthorizeURL)
	return err == nil && parsed.Scheme == "https" && parsed.Host != "" && parsed.User == nil
}

// ownLogo reports whether a thumbnail is this deployment serving that plugin's own logo.
// The host is whatever public_url is, which this package is not told, so the path is what
// is checked. http is allowed because a router developed against localhost serves one.
func ownLogo(raw, pluginID string) bool {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" {
		return false
	}
	return (parsed.Scheme == "https" || parsed.Scheme == "http") && parsed.Path == LogoPath(pluginID)
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
